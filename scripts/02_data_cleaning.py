import pandas as pd

print("===== DATA CLEANING =====\n")

# Load original dataset
file_path = "data/student_health_data.csv"
df = pd.read_csv(file_path, encoding="utf-8")

print("Shape Before Cleaning:")
print(df.shape)

# Remove columns that start with 'Unnamed'
df_cleaned = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("\nShape After Removing Unnamed Columns:")
print(df_cleaned.shape)

print("\nRemaining Columns:")
print(df_cleaned.columns)

# Save cleaned dataset
cleaned_path = "data/student_health_cleaned.csv"
df_cleaned.to_csv(cleaned_path, index=False, encoding="utf-8")

print("\nCleaned dataset saved successfully as student_health_cleaned.csv")