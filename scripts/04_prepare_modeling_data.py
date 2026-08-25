import pandas as pd

# Load the forecasting dataset
df = pd.read_csv(
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College\Documents\THESIS\data\forecasting_data.csv"
)

# Count observations for each program
program_counts = df["PROGRAM"].value_counts()

# Minimum observations required for modeling
MIN_OBSERVATIONS = 6

# Get programs with enough historical observations
valid_programs = program_counts[
    program_counts >= MIN_OBSERVATIONS
].index

# Create modeling dataset
modeling_df = df[
    df["PROGRAM"].isin(valid_programs)
].copy()

# Sort the data properly
modeling_df = modeling_df.sort_values(
    by=["PROGRAM", "ACADEMIC_YEAR", "SEMESTER"]
)

# Save the modeling dataset
output_path = (
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College"
    r"\Documents\THESIS\data\modeling_data.csv"
)

modeling_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8"
)

# Display results
print("\n=== MODELING DATA PREPARATION ===")

print("\nMinimum observations required:", MIN_OBSERVATIONS)

print("\n=== RETAINED PROGRAMS ===")
for program in sorted(valid_programs):
    print(f"{program}: {program_counts[program]} observations")

# Programs with insufficient observations
excluded_programs = program_counts[
    program_counts < MIN_OBSERVATIONS
]

print("\n=== EXCLUDED FROM MODELING ===")
for program, count in excluded_programs.items():
    print(f"{program}: {count} observations")

print("\n=== DATASET SUMMARY ===")
print("Original records:", len(df))
print("Modeling records:", len(modeling_df))
print("Programs retained:", modeling_df["PROGRAM"].nunique())
print("Programs excluded:", len(excluded_programs))

print("\nDataset successfully saved!")
print("Saved as:", output_path)