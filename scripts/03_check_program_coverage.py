import pandas as pd

# Load the forecasting dataset
df = pd.read_csv(
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College\Documents\THESIS\data\forecasting_data.csv"
)

# Count the number of observations for each program
coverage = (
    df.groupby("PROGRAM")
      .agg(
          OBSERVATIONS=("TOTAL_ENROLLMENT", "count"),
          FIRST_YEAR=("ACADEMIC_YEAR", "min"),
          LAST_YEAR=("ACADEMIC_YEAR", "max")
      )
      .sort_values(
          by=["OBSERVATIONS", "PROGRAM"],
          ascending=[False, True]
      )
)

print("\n=== PROGRAM COVERAGE ===")
print(coverage.to_string())

# Show programs with limited observations
limited = coverage[coverage["OBSERVATIONS"] < 4]

print("\n=== PROGRAMS WITH LESS THAN 4 OBSERVATIONS ===")

if limited.empty:
    print("None")
else:
    print(limited.to_string())

print("\n=== SUMMARY ===")
print("Total programs:", len(coverage))
print("Programs with 4 or more observations:",
      len(coverage[coverage["OBSERVATIONS"] >= 4]))
print("Programs with less than 4 observations:",
      len(limited))