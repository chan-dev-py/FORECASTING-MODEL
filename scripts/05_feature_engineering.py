import pandas as pd

# ==========================================
# 05 - FEATURE ENGINEERING
# ==========================================

# Load the prepared modeling dataset
df = pd.read_csv(
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College\Documents\THESIS\data\modeling_data.csv"
)

print("=== ORIGINAL MODELING DATA ===")
print(df.head())

# ------------------------------------------
# CREATE CHRONOLOGICAL TIME INDEX
# ------------------------------------------

# Sort the dataset by Academic Year, Semester, and Program
df = df.sort_values(
    by=["ACADEMIC_YEAR", "SEMESTER", "PROGRAM"]
).reset_index(drop=True)

# Get all unique Academic Year + Semester combinations
time_periods = (
    df[["ACADEMIC_YEAR", "SEMESTER"]]
    .drop_duplicates()
    .sort_values(["ACADEMIC_YEAR", "SEMESTER"])
    .reset_index(drop=True)
)

# Create a sequential time index
time_periods["TIME_INDEX"] = range(1, len(time_periods) + 1)

# Merge the TIME_INDEX back into the dataset
df = df.merge(
    time_periods,
    on=["ACADEMIC_YEAR", "SEMESTER"],
    how="left"
)

# ------------------------------------------
# CREATE PROGRAM-SPECIFIC TIME INDEX
# ------------------------------------------

# Count the observation sequence within each program
df["PROGRAM_TIME_INDEX"] = (
    df.groupby("PROGRAM")
    .cumcount()
    + 1
)

# ------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------

print("\n=== FEATURE ENGINEERED DATA ===")
print(df.head(20))

print("\n=== DATASET INFORMATION ===")
print(f"Total records: {len(df)}")
print(f"Total programs: {df['PROGRAM'].nunique()}")
print(f"Total time periods: {df['TIME_INDEX'].nunique()}")

print("\n=== TIME PERIODS ===")
print(time_periods)

# ------------------------------------------
# SAVE FEATURE ENGINEERED DATASET
# ------------------------------------------

output_path = (
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College"
    r"\Documents\THESIS\data\feature_engineered_data.csv"
)

df.to_csv(output_path, index=False)

print("\nFeature engineering completed successfully!")
print(f"Saved as: {output_path}")