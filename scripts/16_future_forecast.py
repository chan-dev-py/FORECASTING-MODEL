import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# ==========================================
# FUTURE ENROLLMENT FORECASTING USING ARIMA
# ==========================================

print("=== FUTURE ENROLLMENT FORECASTING ===\n")

# ------------------------------------------
# LOAD COMPLETE HISTORICAL DATA
# ------------------------------------------

data = pd.read_csv("../data/modeling_data.csv")

print("Historical data loaded successfully!")
print(f"Total records: {len(data)}\n")

# Make sure data is sorted chronologically
data = data.sort_values(
    by=["PROGRAM", "ACADEMIC_YEAR", "SEMESTER"]
)

# ------------------------------------------
# FUTURE PERIODS TO FORECAST
# ------------------------------------------

future_periods = [
    {"ACADEMIC_YEAR": 2026, "SEMESTER": 1},
    {"ACADEMIC_YEAR": 2026, "SEMESTER": 2}
]

all_forecasts = []

# ------------------------------------------
# FORECAST EACH PROGRAM
# ------------------------------------------

programs = data["PROGRAM"].unique()

for program in programs:

    print(f"Forecasting program: {program}")

    # Get enrollment history for one program
    program_data = data[
        data["PROGRAM"] == program
    ].copy()

    # Sort chronologically
    program_data = program_data.sort_values(
        by=["ACADEMIC_YEAR", "SEMESTER"]
    )

    enrollment_series = program_data[
        "TOTAL_ENROLLMENT"
    ].astype(float)

    # Skip programs with insufficient data
    if len(enrollment_series) < 4:
        print("Insufficient historical data. Skipping.\n")
        continue

    try:
        # Train ARIMA using ALL available historical data
        model = ARIMA(
            enrollment_series,
            order=(1, 1, 1)
        )

        fitted_model = model.fit()

        # Forecast 2 future semesters
        forecast = fitted_model.forecast(
            steps=2
        )

        # Save forecast results
        for i, future in enumerate(future_periods):

            predicted_value = forecast.iloc[i]

            # Prevent negative enrollment
            predicted_value = max(
                0,
                predicted_value
            )

            all_forecasts.append({
                "PROGRAM": program,
                "ACADEMIC_YEAR": future["ACADEMIC_YEAR"],
                "SEMESTER": future["SEMESTER"],
                "PREDICTED_ENROLLMENT": round(
                    predicted_value,
                    2
                )
            })

    except Exception as e:
        print(
            f"Error forecasting {program}: {e}"
        )

    print()

# ------------------------------------------
# CREATE FORECAST DATAFRAME
# ------------------------------------------

forecast_df = pd.DataFrame(all_forecasts)

# Sort results
forecast_df = forecast_df.sort_values(
    by=["ACADEMIC_YEAR", "SEMESTER", "PROGRAM"]
)

# ------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------

print("=== FUTURE ENROLLMENT FORECASTS ===\n")

print(
    forecast_df.to_string(
        index=False
    )
)

# ------------------------------------------
# SAVE RESULTS
# ------------------------------------------

forecast_df.to_csv(
    "../data/future_enrollment_forecast.csv",
    index=False
)

print(
    "\nFuture enrollment forecasts successfully saved!"
)

print(
    "Saved as: ../data/future_enrollment_forecast.csv"
)