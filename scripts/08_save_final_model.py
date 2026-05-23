import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("===== SAVING FINAL RANDOM FOREST MODEL =====\n")

# Load cleaned dataset
file_path = "data/student_health_cleaned.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# ---------------------------------
# Encode Categorical Features
# ---------------------------------

categorical_columns = [
    "Gender",
    "Physical_Activity",
    "Sleep_Quality",
    "Mood"
]

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df["Health_Risk_Level"] = target_encoder.fit_transform(df["Health_Risk_Level"])

# ---------------------------------
# Split Data
# ---------------------------------

X = df.drop("Health_Risk_Level", axis=1)
y = df["Health_Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------
# Train Final Model
# ---------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("Final Model Training Completed.\n")

# ---------------------------------
# Save Model and Encoders
# ---------------------------------

joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(encoders, "models/feature_encoders.pkl")
joblib.dump(target_encoder, "models/target_encoder.pkl")

print("Model and encoders saved successfully in 'models' folder.")