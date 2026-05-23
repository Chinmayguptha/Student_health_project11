import pandas as pd

# Load dataset
file_path = "data/student_health_data.csv"
df = pd.read_csv(file_path, encoding="utf-8")

print("===== DATA OVERVIEW =====\n")

print("First 5 Rows:\n")
print(df.head())

print("\nDataset Shape (Rows, Columns):")
print(df.shape)

print("\nColumn Names:")
print(df.columns)