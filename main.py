import pandas as pd

# Load dataset
file_path = "data/student_health_data.csv"
df = pd.read_csv(file_path, encoding="utf-8")

print("First 5 rows of dataset:\n")
print(df.head())

print("\nDataset Shape (Before Cleaning):")
print(df.shape)

print("\nColumn Names (Before Cleaning):")
print(df.columns)

# -----------------------------
# DATA CLEANING
# -----------------------------

# Remove Unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("\nDataset Shape (After Removing Unnamed Columns):")
print(df.shape)

print("\nColumn Names (After Cleaning):")
print(df.columns)

# Save cleaned dataset
df.to_csv("data/student_health_cleaned.csv", index=False, encoding="utf-8")

print("\nCleaned dataset saved as student_health_cleaned.csv")