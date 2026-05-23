import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Health AI System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Health Risk Prediction System")
st.write("Enter student health details to predict the risk level.")

# -----------------------------
# Load Model and Encoders
# -----------------------------
model = joblib.load("models/random_forest_model.pkl")
feature_encoders = joblib.load("models/feature_encoders.pkl")
target_encoder = joblib.load("models/target_encoder.pkl")

# -----------------------------
# User Input Section
# -----------------------------
st.header("Student Health Inputs")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 15, 40, 20)
    heart_rate = st.number_input("Heart Rate", 40, 150, 70)

    bp_sys = st.number_input("Blood Pressure (Systolic)", 80, 200, 120)
    bp_dia = st.number_input("Blood Pressure (Diastolic)", 50, 150, 80)

    stress_bio = st.slider("Stress Level (Biosensor)", 1, 10, 5)
    stress_self = st.slider("Stress Level (Self Report)", 1, 10, 5)

    study_hours = st.slider("Study Hours / Week", 0, 60, 20)
    project_hours = st.slider("Project Hours / Week", 0, 40, 10)

with col2:
    gender = st.selectbox("Gender", ["M", "F"])

    physical_activity = st.selectbox(
        "Physical Activity",
        ["Low", "Moderate", "High"]
    )

    sleep_quality = st.selectbox(
        "Sleep Quality",
        ["Poor", "Moderate", "Good"]
    )

    mood = st.selectbox(
        "Mood",
        ["Happy", "Neutral", "Stressed"]
    )

    family_members = st.slider("Family Members", 1, 15, 4)

# -----------------------------
# Encode categorical features
# -----------------------------
gender_encoded = feature_encoders["Gender"].transform([gender])[0]
activity_encoded = feature_encoders["Physical_Activity"].transform([physical_activity])[0]
sleep_encoded = feature_encoders["Sleep_Quality"].transform([sleep_quality])[0]
mood_encoded = feature_encoders["Mood"].transform([mood])[0]

# -----------------------------
# Create Input DataFrame
# -----------------------------

student_id = 1

columns = [
    "Student_ID",
    "Age",
    "Gender",
    "Heart_Rate",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Stress_Level_Biosensor",
    "Stress_Level_Self_Report",
    "Physical_Activity",
    "Sleep_Quality",
    "Mood",
    "Study_Hours",
    "Project_Hours",
    "Family_members"
]

data = [[
    student_id,
    age,
    gender_encoded,
    heart_rate,
    bp_sys,
    bp_dia,
    stress_bio,
    stress_self,
    activity_encoded,
    sleep_encoded,
    mood_encoded,
    study_hours,
    project_hours,
    family_members
]]

input_data = pd.DataFrame(data, columns=columns)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Health Risk"):

    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    risk_level = target_encoder.inverse_transform(prediction)[0]

    st.subheader("Prediction Result")

    if risk_level == "Low":
        st.success("🟢 Low Health Risk")

    elif risk_level == "Moderate":
        st.warning("🟡 Moderate Health Risk")

    else:
        st.error("🔴 High Health Risk")

    # Show probability
    st.subheader("Prediction Confidence")

    prob_df = pd.DataFrame({
        "Risk Level": target_encoder.classes_,
        "Probability": probabilities[0]
    })

    st.bar_chart(prob_df.set_index("Risk Level"))

# -----------------------------
# Feature Importance
# -----------------------------
st.header("Model Feature Importance")

feature_names = [
    "Student ID",
    "Age",
    "Gender",
    "Heart Rate",
    "BP Systolic",
    "BP Diastolic",
    "Stress (Biosensor)",
    "Stress (Self Report)",
    "Physical Activity",
    "Sleep Quality",
    "Mood",
    "Study Hours",
    "Project Hours",
    "Family Members"
]

importances = model.feature_importances_

min_len = min(len(feature_names), len(importances))

feature_names = feature_names[:min_len]
importances = importances[:min_len]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# 🔹 SMALLER FIGURE SIZE FOR PPT SCREENSHOT
fig, ax = plt.subplots(figsize=(6,4))   # <-- reduced size

ax.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    color="steelblue"
)

ax.set_xlabel("Importance Score", fontsize=9)
ax.set_title("Feature Importance (Random Forest)", fontsize=10)

ax.tick_params(axis='y', labelsize=8)

ax.invert_yaxis()

plt.tight_layout()

st.pyplot(fig)