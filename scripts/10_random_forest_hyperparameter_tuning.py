import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("===== RANDOM FOREST HYPERPARAMETER TUNING =====\n")

# Load dataset
file_path = "data/student_health_cleaned.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# Encode categorical features
categorical_columns = [
    "Gender",
    "Physical_Activity",
    "Sleep_Quality",
    "Mood"
]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Encode target
target_encoder = LabelEncoder()
df["Health_Risk_Level"] = target_encoder.fit_transform(df["Health_Risk_Level"])

# Split data
X = df.drop("Health_Risk_Level", axis=1)
y = df["Health_Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define parameter grid
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "class_weight": [None, "balanced"]
}

# Initialize base model
rf = RandomForestClassifier(random_state=42)

# Grid Search with 5-fold cross validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

print("Starting Grid Search...\n")
grid_search.fit(X_train, y_train)

print("\nBest Parameters Found:")
print(grid_search.best_params_)

# Best model
best_model = grid_search.best_estimator_

# Evaluate on test data
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy After Tuning:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nHyperparameter tuning completed successfully.")