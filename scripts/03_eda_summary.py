import pandas as pd

print("===== EXPLORATORY DATA ANALYSIS (EDA) =====\n")

# Load cleaned dataset
file_path = "data/student_health_cleaned.csv"
df = pd.read_csv(file_path, encoding="utf-8")

print("Dataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values Per Column:")
print(df.isnull().sum())

print("\nStatistical Summary (Numerical Columns):")
print(df.describe())

print("\nUnique Values in Health_Risk_Level:")
print(df["Health_Risk_Level"].unique())