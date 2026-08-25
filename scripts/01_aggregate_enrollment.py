import pandas as pd

# Load the cleaned dataset
df = pd.read_csv(
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College\Documents\THESIS\data\cleaned_student_records.csv",
    encoding="utf-8"
)

# Keep only Semester 1 and Semester 2
df = df[df["SEM"].isin([1, 2])]

# Remove Senior High School records
df = df[~df["COURSE"].isin(["GRADE 11", "GRADE 12"])]

# Aggregate unique students by Academic Year + Semester + Program
enrollment_df = (
    df.groupby(["ACADEMIC YEAR", "SEM", "COURSE"])["ID NUMBER"]
      .nunique()
      .reset_index(name="TOTAL_ENROLLMENT")
)

# Rename columns
enrollment_df = enrollment_df.rename(columns={
    "ACADEMIC YEAR": "ACADEMIC_YEAR",
    "SEM": "SEMESTER",
    "COURSE": "PROGRAM"
})

# Display results
print("FORECASTING DATA:")
print(enrollment_df)

print("\nTotal records:", len(enrollment_df))

# Save final forecasting dataset
enrollment_df.to_csv(
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College\Documents\THESIS\data\forecasting_data.csv",
    index=False
)

print("\nDataset successfully saved!")