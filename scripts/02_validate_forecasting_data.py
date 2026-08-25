import pandas as pd

# Load the forecasting dataset
df = pd.read_csv(
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College\Documents\THESIS\data\forecasting_data.csv",
    encoding="utf-8"
)

print("\n=== DATASET OVERVIEW ===")
print(df.head())

print("\n=== DATASET SHAPE ===")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# Check column names
print("\n=== COLUMN NAMES ===")
print(df.columns.tolist())

# Check Academic Years
print("\n=== ACADEMIC YEARS ===")
print(sorted(df["ACADEMIC_YEAR"].unique()))

# Check Semester values
print("\n=== SEMESTER VALUES ===")
print(sorted(df["SEMESTER"].unique()))

# Check Programs
print("\n=== PROGRAMS ===")
print(sorted(df["PROGRAM"].unique()))

# Check for missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# Check duplicate Year + Semester + Program combinations
duplicates = df.duplicated(
    subset=["ACADEMIC_YEAR", "SEMESTER", "PROGRAM"]
).sum()

print("\n=== DUPLICATE COMBINATIONS ===")
print(f"Duplicate rows: {duplicates}")

# Check total enrollment
print("\n=== TOTAL ENROLLMENT SUMMARY ===")
print(df["TOTAL_ENROLLMENT"].describe())

print("\nValidation completed successfully!")