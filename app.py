import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Health AI System",
    layout="wide"
)

st.title("🎓 Student Health Risk Prediction System")
st.write("AI-powered student health risk prediction")

# -----------------------------
# Load Model & Encoders
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/random_forest_model.pkl")
    feature_encoders = joblib.load("models/feature_encoders.pkl")
    target_encoder = joblib.load("models/target_encoder.pkl")
    return model, feature_encoders, target_encoder

model, feature_encoders, target_encoder = load_artifacts()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter Student Details")

age = st.sidebar.slider("Age", 15, 30, 20)
bmi = st.sidebar.slider("BMI", 15.0, 35.0, 22.0)
stress = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
heart_rate = st.sidebar.slider("Heart Rate", 60, 120, 75)

gender = st.sidebar.selectbox(
    "Gender",
    feature_encoders["Gender"].classes_
)

physical_activity = st.sidebar.selectbox(
    "Physical Activity",
    feature_encoders["Physical_Activity"].classes_
)

sleep_quality = st.sidebar.selectbox(
    "Sleep Quality",
    feature_encoders["Sleep_Quality"].classes_
)

mood = st.sidebar.selectbox(
    "Mood",
    feature_encoders["Mood"].classes_
)

# -----------------------------
# Prepare Input with Correct Feature Structure
# -----------------------------
expected_features = model.feature_names_in_

input_df = pd.DataFrame(columns=expected_features)

# Initialize all features with 0
for col in expected_features:
    input_df.loc[0, col] = 0

def assign_if_exists(column_name, value):
    if column_name in input_df.columns:
        input_df.loc[0, column_name] = value

# Numerical features
assign_if_exists("Age", age)
assign_if_exists("BMI", bmi)
assign_if_exists("Stress_Level", stress)
assign_if_exists("Heart_Rate", heart_rate)

# Encoded categorical features
assign_if_exists("Gender", feature_encoders["Gender"].transform([gender])[0])
assign_if_exists("Physical_Activity", feature_encoders["Physical_Activity"].transform([physical_activity])[0])
assign_if_exists("Sleep_Quality", feature_encoders["Sleep_Quality"].transform([sleep_quality])[0])
assign_if_exists("Mood", feature_encoders["Mood"].transform([mood])[0])

# -----------------------------
# Prediction Section
# -----------------------------
if st.button("🔍 Predict Health Risk"):

    prediction = model.predict(input_df)
    predicted_label = target_encoder.inverse_transform(prediction)[0]

    # Probability
    probability = model.predict_proba(input_df).max() * 100

    st.success(f"Predicted Health Risk Level: {predicted_label}")
    st.info(f"Model Confidence: {probability:.2f}%")

st.markdown("---")
st.caption("Built with Tuned Random Forest Model")