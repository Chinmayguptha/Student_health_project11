import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("===== MODEL TRAINING: LOGISTIC REGRESSION (SCALED) =====\n")

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

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Encode target variable
target_encoder = LabelEncoder()
df["Health_Risk_Level"] = target_encoder.fit_transform(df["Health_Risk_Level"])

# ---------------------------------
# Split Features and Target
# ---------------------------------

X = df.drop("Health_Risk_Level", axis=1)
y = df["Health_Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------
# Scale Features
# ---------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature Scaling Completed.\n")

# ---------------------------------
# Train Logistic Regression Model
# ---------------------------------

model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

print("Model Training Completed.\n")

# ---------------------------------
# Predictions
# ---------------------------------

y_pred = model.predict(X_test_scaled)

# ---------------------------------
# Evaluation
# ---------------------------------

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nScaled Logistic Regression training completed successfully.")