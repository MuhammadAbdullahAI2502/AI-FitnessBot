import os
import io
import math
import json
import sqlite3
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta

# --- Optional AI integrations (Groq via LangChain) ---
try:
    from langchain_groq import ChatGroq  # pip install langchain-groq
    from langchain.schema import HumanMessage, SystemMessage
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

DB_PATH = "coach.db"

# -----------------------------
# Utilities: DB
# -----------------------------

def init_db():
    cn = sqlite3.connect(DB_PATH)
    cur = cn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            gender TEXT,
            height_cm REAL,
            weight_kg REAL,
            age INTEGER,
            goal TEXT,
            diet_pref TEXT,
            activity_level TEXT,
            equipment TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            log_date TEXT,
            weight_kg REAL,
            calories_target INTEGER,
            notes TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    cn.commit()
    cn.close()


def upsert_user(gender, height_cm, weight_kg, age, goal, diet_pref, activity_level, equipment):
    cn = sqlite3.connect(DB_PATH)
    cur = cn.cursor()
    now = datetime.utcnow().isoformat()
    cur.execute(
        """
        INSERT INTO users (created_at, gender, height_cm, weight_kg, age, goal, diet_pref, activity_level, equipment)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (now, gender, height_cm, weight_kg, age, goal, ",".join(diet_pref), activity_level, equipment),
    )
    user_id = cur.lastrowid
    cn.commit()
    cn.close()
    return user_id


def log_progress(user_id: int, weight_kg: float, calories_target: int, notes: str = ""):
    cn = sqlite3.connect(DB_PATH)
    cur = cn.cursor()
    cur.execute(
        """
        INSERT INTO progress (user_id, log_date, weight_kg, calories_target, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, date.today().isoformat(), weight_kg, int(calories_target), notes),
    )
    cn.commit()
    cn.close()


def fetch_progress(user_id: int) -> pd.DataFrame:
    cn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT log_date, weight_kg, calories_target, notes FROM progress WHERE user_id=? ORDER BY log_date",
        cn,
        params=(user_id,),
    )
    cn.close()
    return df


# -----------------------------
# Calculations
# -----------------------------
ACTIVITY_FACTORS = {
    "Sedentary": 1.2,
    "Light": 1.375,
    "Moderate": 1.55,
    "High": 1.725,
}


def bmr_msj(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """Mifflin-St Jeor formula."""
    if gender.lower().startswith("m"):  # Male
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    elif gender.lower().startswith("f"):  # Female
        return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    else:
        # Default neutral (average of male/female offsets)
        return 10 * weight_kg + 6.25 * height_cm - 5 * age - 78


def tdee(bmr: float, activity_level: str) -> float:
    factor = ACTIVITY_FACTORS.get(activity_level, 1.2)
    return bmr * factor


def goal_calories(tdee_value: float, goal: str) -> int:
    goal = goal.lower()
    if "loss" in goal:
        return int(round(tdee_value * 0.80))  # ~20% deficit
    if "gain" in goal:
        return int(round(tdee_value * 1.15))  # ~15% surplus
    return int(round(tdee_value))


def macro_split(weight_kg: float, calories: int, goal: str, diet_pref: list[str]):
    """
    Protein: 1.6–2.2 g/kg (use 1.8 default, 2.0 for loss, 1.6 for gain)
    Fat: 0.8–1.0 g/kg (use 0.9 default)
    Carbs: remaining calories
    """
    goal_l = goal.lower()
    if "loss" in goal_l:
        protein_g_per_kg = 2.0
    elif "gain" in goal_l:
        protein_g_per_kg = 1.6
    else:
        protein_g_per_kg = 1.8

    fat_g_per_kg = 0.9

    protein_g = round(protein_g_per_kg * weight_kg)
    fat_g = round(fat_g_per_kg * weight_kg)

    # Calories from P/F
    p_cal = protein_g * 4
    f_cal = fat_g * 9
    remaining = max(calories - (p_cal + f_cal), 0)
    carbs_g = round(remaining / 4)

    # Simple vegetarian/vegan guardrails (can be expanded)
    if "Vegan" in diet_pref:
        # ensure protein isn't unrealistic; add note in UI
        pass

    return {
        "calories": calories,
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
    }


def make_meal_plan(macros: dict, diet_pref: list[str]):
    """Very simple templated meal plan; you can enhance with recipe DB or RAG."""
    veg = ("Vegetarian" in diet_pref) or ("Vegan" in diet_pref)

    def pick(protein_options, nonveg):
        if veg:
            return protein_options[0]
        return nonveg[0]

    meals = [
        {
            "meal": "Breakfast",
            "idea": "Oats with whey/plant protein, banana, nuts",
        },
        {
            "meal": "Snack",
            "idea": "Greek yogurt / soy yogurt with berries; or apple + peanut butter",
        },
        {
            "meal": "Lunch",
            "idea": "Grain bowl: brown rice, veggies, protein (" + pick(["tofu"], ["chicken breast"]) + ") + olive oil",
        },
        {
            "meal": "Snack 2",
            "idea": "Trail mix or hummus + cucumbers/carrots",
        },
        {
            "meal": "Dinner",
            "idea": "Stir-fry: mixed veggies + " + pick(["tempeh"], ["fish/lean beef"]) + " + quinoa",
        },
    ]
    return pd.DataFrame(meals)


def make_workout_plan(goal: str, equipment: str, activity_level: str):
    home = (equipment.lower().strip() in ["none", "bodyweight", "home"]) if equipment else True
    plan = []

    if home:
        split = [
            ("Day 1", "Full-body strength (push-ups, squats, lunges, planks, hip hinges) 30–40 min"),
            ("Day 2", "Low-intensity cardio (brisk walk/cycling) 30–45 min"),
            ("Day 3", "Mobility + core (yoga flows, planks) 25–35 min"),
            ("Day 4", "Full-body strength (progressions) 30–40 min"),
            ("Day 5", "Intervals: 10×1 min hard/1 min easy + warmup/cooldown"),
            ("Day 6", "Active recovery: long walk / light cycling 40–60 min"),
            ("Day 7", "Rest")
        ]
    else:
        split = [
            ("Day 1", "Upper body (bench/rows/overhead press/accessories) + 10 min cardio"),
            ("Day 2", "Lower body (squats/hinge/lunges/calf raises) + core"),
            ("Day 3", "LISS cardio 30–45 min + mobility"),
            ("Day 4", "Upper body (pull focus) + accessories + 10 min cardio"),
            ("Day 5", "Lower body (deadlift variant focus) + core"),
            ("Day 6", "Active recovery: incline walk 30–40 min"),
            ("Day 7", "Rest"),
        ]

    for day, desc in split:
        plan.append({"day": day, "workout": desc})
    return pd.DataFrame(plan)


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(page_title="AI Fitness Coach", page_icon="💪", layout="wide")
    
    init_db()
    
    st.title("💪 AI Fitness Coach")
    st.markdown("Get personalized meal and workout plans based on your goals!")
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("Your Details")
        
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        age = st.number_input("Age", min_value=15, max_value=100, value=25)
        
        goal = st.selectbox("Goal", ["Weight Loss", "Weight Gain", "Maintenance"])
        activity_level = st.selectbox("Activity Level", ["Sedentary", "Light", "Moderate", "High"])
        equipment = st.selectbox("Equipment", ["None/Bodyweight", "Home Gym", "Full Gym"])
        diet_pref = st.multiselect("Diet Preferences", ["Vegetarian", "Vegan", "Keto", "Paleo"])
        
        if st.button("Generate Plan", type="primary"):
            # Calculate BMR and TDEE
            bmr = bmr_msj(weight_kg, height_cm, age, gender)
            tdee_val = tdee(bmr, activity_level)
            calories = goal_calories(tdee_val, goal)
            macros = macro_split(weight_kg, calories, goal, diet_pref)
            
            # Store user data
            user_id = upsert_user(gender, height_cm, weight_kg, age, goal, diet_pref, activity_level, equipment)
            
            # Store in session state
            st.session_state.user_data = {
                'user_id': user_id,
                'bmr': bmr,
                'tdee': tdee_val,
                'calories': calories,
                'macros': macros,
                'goal': goal,
                'diet_pref': diet_pref,
                'equipment': equipment,
                'activity_level': activity_level
            }
    
    # Main content
    if 'user_data' in st.session_state:
        data = st.session_state.user_data
        
        # Display calculations
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("BMR", f"{data['bmr']:.0f} kcal")
        with col2:
            st.metric("TDEE", f"{data['tdee']:.0f} kcal")
        with col3:
            st.metric("Target Calories", f"{data['calories']} kcal")
        with col4:
            st.metric("Goal", data['goal'])
        
        # Macros
        st.subheader("📊 Macro Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Protein", f"{data['macros']['protein_g']}g")
        with col2:
            st.metric("Carbs", f"{data['macros']['carbs_g']}g")
        with col3:
            st.metric("Fat", f"{data['macros']['fat_g']}g")
        
        # Meal Plan
        st.subheader("🍽️ Meal Plan")
        meal_plan = make_meal_plan(data['macros'], data['diet_pref'])
        st.dataframe(meal_plan, use_container_width=True)
        
        # Workout Plan
        st.subheader("🏋️ Workout Plan")
        workout_plan = make_workout_plan(data['goal'], data['equipment'], data['activity_level'])
        st.dataframe(workout_plan, use_container_width=True)
        
        # Progress Tracking
        st.subheader("📈 Progress Tracking")
        col1, col2 = st.columns(2)
        with col1:
            new_weight = st.number_input("Current Weight (kg)", value=weight_kg, key="progress_weight")
            notes = st.text_area("Notes", placeholder="How are you feeling?")
            if st.button("Log Progress"):
                log_progress(data['user_id'], new_weight, data['calories'], notes)
                st.success("Progress logged!")
        
        with col2:
            progress_df = fetch_progress(data['user_id'])
            if not progress_df.empty:
                st.line_chart(progress_df.set_index('log_date')['weight_kg'])
            else:
                st.info("No progress data yet. Start logging!")
    
    else:
        st.info("👈 Enter your details in the sidebar to get started!")

if __name__ == "__main__":
    main()