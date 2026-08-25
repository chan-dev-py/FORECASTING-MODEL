import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# LOAD MODELING DATA
# ==============================

file_path = r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College\Documents\THESIS\data\modeling_data.csv"

df = pd.read_csv(file_path)

# Sort the data
df = df.sort_values(
    by=["ACADEMIC_YEAR", "SEMESTER", "PROGRAM"]
)

print("\n=== EXPLORATORY DATA ANALYSIS ===")

# ==============================
# DATASET OVERVIEW
# ==============================

print("\n=== DATASET INFORMATION ===")

print(f"Total records: {len(df)}")
print(f"Total programs: {df['PROGRAM'].nunique()}")

print("\nAcademic Years:")
print(sorted(df["ACADEMIC_YEAR"].unique()))

print("\nPrograms:")
print(df["PROGRAM"].unique())


# ==============================
# OVERALL ENROLLMENT BY PERIOD
# ==============================

overall = (
    df.groupby(["ACADEMIC_YEAR", "SEMESTER"])
    ["TOTAL_ENROLLMENT"]
    .sum()
    .reset_index()
)

overall["PERIOD"] = (
    overall["ACADEMIC_YEAR"].astype(str)
    + " S"
    + overall["SEMESTER"].astype(str)
)

print("\n=== OVERALL ENROLLMENT TREND ===")

print(overall[[
    "ACADEMIC_YEAR",
    "SEMESTER",
    "TOTAL_ENROLLMENT"
]])


# ==============================
# SEMESTER ANALYSIS
# ==============================

semester_summary = (
    df.groupby("SEMESTER")
    ["TOTAL_ENROLLMENT"]
    .sum()
)

print("\n=== ENROLLMENT BY SEMESTER ===")

print(semester_summary)


# ==============================
# PROGRAM ENROLLMENT SUMMARY
# ==============================

program_summary = (
    df.groupby("PROGRAM")
    ["TOTAL_ENROLLMENT"]
    .sum()
    .sort_values(ascending=False)
)

print("\n=== PROGRAM ENROLLMENT SUMMARY ===")

print(program_summary)


# ==============================
# OVERALL TREND VISUALIZATION
# ==============================

plt.figure(figsize=(12, 6))

plt.plot(
    overall["PERIOD"],
    overall["TOTAL_ENROLLMENT"],
    marker="o"
)

plt.title("Overall College Enrollment Trend")
plt.xlabel("Academic Period")
plt.ylabel("Total Enrollment")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College\Documents\THESIS\data\overall_enrollment_trend.png",
    dpi=300
)

plt.show()


# ==============================
# SEMESTER COMPARISON
# ==============================

plt.figure(figsize=(8, 5))

semester_summary.plot(
    kind="bar"
)

plt.title("Total Enrollment by Semester")
plt.xlabel("Semester")
plt.ylabel("Total Enrollment")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College\Documents\THESIS\data\semester_enrollment.png",
    dpi=300
)

plt.show()


# ==============================
# PROGRAM ENROLLMENT
# ==============================

plt.figure(figsize=(12, 7))

program_summary.plot(
    kind="bar"
)

plt.title("Total Enrollment by Program")
plt.xlabel("Program")
plt.ylabel("Total Enrollment")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College\Documents\THESIS\data\program_enrollment.png",
    dpi=300
)

plt.show()


print("\nExploratory Data Analysis completed successfully!")

print(
    "\nCharts saved in the data folder:"
)

print("- overall_enrollment_trend.png")
print("- semester_enrollment.png")
print("- program_enrollment.png")