import pandas as pd

# ==========================================
# FUTURE FORECAST VALIDATION
# ==========================================

print("=== FUTURE FORECAST VALIDATION ===\n")

# Load historical modeling data
historical = pd.read_csv("../data/modeling_data.csv")

# Load future forecasts
forecast = pd.read_csv("../data/future_enrollment_forecast.csv")

print("Historical data loaded successfully!")
print(f"Historical records: {len(historical)}")

print("\nFuture forecast loaded successfully!")
print(f"Forecast records: {len(forecast)}")


# ==========================================
# CHECK NUMBER OF OBSERVATIONS PER PROGRAM
# ==========================================

print("\n=== OBSERVATIONS PER PROGRAM ===\n")

observations = (
    historical.groupby("PROGRAM")
    .size()
    .reset_index(name="OBSERVATIONS")
)

print(observations.to_string(index=False))


# ==========================================
# GET LATEST ACTUAL ENROLLMENT
# ==========================================

historical = historical.sort_values(
    by=["PROGRAM", "ACADEMIC_YEAR", "SEMESTER"]
)

latest_actual = (
    historical.groupby("PROGRAM")
    .tail(1)
    [["PROGRAM", "ACADEMIC_YEAR", "SEMESTER", "TOTAL_ENROLLMENT"]]
    .rename(
        columns={
            "ACADEMIC_YEAR": "LATEST_YEAR",
            "SEMESTER": "LATEST_SEMESTER",
            "TOTAL_ENROLLMENT": "LATEST_ENROLLMENT"
        }
    )
)

# ==========================================
# GET FIRST FUTURE FORECAST
# ==========================================

first_forecast = (
    forecast.sort_values(
        by=["PROGRAM", "ACADEMIC_YEAR", "SEMESTER"]
    )
    .groupby("PROGRAM")
    .head(1)
)

first_forecast = first_forecast[
    ["PROGRAM", "PREDICTED_ENROLLMENT"]
].rename(
    columns={
        "PREDICTED_ENROLLMENT": "NEXT_FORECAST"
    }
)

# ==========================================
# MERGE VALIDATION DATA
# ==========================================

validation = observations.merge(
    latest_actual,
    on="PROGRAM",
    how="left"
)

validation = validation.merge(
    first_forecast,
    on="PROGRAM",
    how="left"
)

# Calculate forecast difference
validation["DIFFERENCE"] = (
    validation["NEXT_FORECAST"]
    - validation["LATEST_ENROLLMENT"]
)

# Calculate percentage change
validation["PERCENT_CHANGE"] = (
    validation["DIFFERENCE"]
    / validation["LATEST_ENROLLMENT"]
    * 100
)

# ==========================================
# FLAG POTENTIAL ISSUES
# ==========================================

def check_forecast(row):

    if row["OBSERVATIONS"] < 6:
        return "LIMITED DATA"

    if row["NEXT_FORECAST"] <= 0:
        return "ZERO OR NEGATIVE FORECAST"

    if abs(row["PERCENT_CHANGE"]) > 50:
        return "LARGE CHANGE"

    return "OK"


validation["STATUS"] = validation.apply(
    check_forecast,
    axis=1
)

# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n=== FORECAST VALIDATION RESULTS ===\n")

print(
    validation[
        [
            "PROGRAM",
            "OBSERVATIONS",
            "LATEST_YEAR",
            "LATEST_SEMESTER",
            "LATEST_ENROLLMENT",
            "NEXT_FORECAST",
            "DIFFERENCE",
            "PERCENT_CHANGE",
            "STATUS"
        ]
    ].to_string(index=False)
)

# ==========================================
# SAVE VALIDATION RESULTS
# ==========================================

validation.to_csv(
    "../data/future_forecast_validation.csv",
    index=False
)

print("\nValidation successfully completed!")
print("Saved as: ../data/future_forecast_validation.csv")