import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("===== DATA PREPROCESSING FOR ML =====\n")

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

label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Categorical columns encoded successfully.\n")

# ---------------------------------
# Encode Target Variable
# ---------------------------------

target_encoder = LabelEncoder()
df["Health_Risk_Level"] = target_encoder.fit_transform(df["Health_Risk_Level"])

print("Target variable encoded successfully.\n")
print("Target Classes Mapping:")
for index, label in enumerate(target_encoder.classes_):
    print(f"{label} -> {index}")

# ---------------------------------
# Split Features and Target
# ---------------------------------

X = df.drop("Health_Risk_Level", axis=1)
y = df["Health_Risk_Level"]

print("\nFeature Matrix Shape:", X.shape)
print("Target Vector Shape:", y.shape)

# ---------------------------------
# Train-Test Split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain-Test Split Completed")
print("Training Set Size:", X_train.shape)
print("Testing Set Size:", X_test.shape)

print("\nPreprocessing completed successfully.")