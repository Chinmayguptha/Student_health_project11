import pandas as pd

# Correct dataset path
DATA_PATH = "data/student_health_cleaned.csv"

df = pd.read_csv(DATA_PATH)

print("\nColumns in dataset:")
print(df.columns)

print("\nShape of dataset:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nUnique values in categorical columns:")

for col in df.select_dtypes(include="object"):
    print(f"\n{col}:")
    print(df[col].unique())