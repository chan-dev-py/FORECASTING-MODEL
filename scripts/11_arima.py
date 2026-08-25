import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ==============================
# LOAD DATA
# ==============================

train_data = pd.read_csv("../data/train_data.csv")
test_data = pd.read_csv("../data/test_data.csv")


print("=== ARIMA FORECASTING ===")

print("\n=== TRAINING DATA ===")
print(f"Training records: {len(train_data)}")

print("\n=== TESTING DATA ===")
print(f"Testing records: {len(test_data)}")


# ==============================
# STORE PREDICTIONS
# ==============================

all_results = []


# Get all programs from testing data
programs = test_data["PROGRAM"].unique()


# ==============================
# FORECAST EACH PROGRAM
# ==============================

for program in programs:

    print(f"\nProcessing program: {program}")

    # Get training data for the program
    train_program = train_data[
        train_data["PROGRAM"] == program
    ].sort_values("PROGRAM_TIME_INDEX")

    # Get testing data for the program
    test_program = test_data[
        test_data["PROGRAM"] == program
    ].sort_values("PROGRAM_TIME_INDEX")

    # Training enrollment values
    y_train = train_program["TOTAL_ENROLLMENT"].values

    # Number of periods to forecast
    forecast_steps = len(test_program)

    try:

        # ARIMA MODEL
        model = ARIMA(
            y_train,
            order=(1, 1, 0)
        )

        model_fit = model.fit()

        # Generate forecasts
        predictions = model_fit.forecast(
            steps=forecast_steps
        )

    except Exception as e:

        print(
            f"ARIMA failed for {program}. "
            f"Using last value as fallback."
        )

        # Fallback prediction
        predictions = np.repeat(
            y_train[-1],
            forecast_steps
        )


    # ==============================
    # SAVE PROGRAM RESULTS
    # ==============================

    program_results = test_program[
        [
            "PROGRAM",
            "ACADEMIC_YEAR",
            "SEMESTER",
            "TOTAL_ENROLLMENT"
        ]
    ].copy()

    program_results[
        "PREDICTED_ENROLLMENT"
    ] = predictions

    all_results.append(program_results)


# ==============================
# COMBINE RESULTS
# ==============================

results = pd.concat(
    all_results,
    ignore_index=True
)


print("\n=== ARIMA PREDICTIONS ===")

print(
    results.to_string(
        index=False
    )
)


# ==============================
# PREPARE ACTUAL & PREDICTED
# ==============================

y_test = results[
    "TOTAL_ENROLLMENT"
]

predictions = results[
    "PREDICTED_ENROLLMENT"
]


# ==============================
# CALCULATE PERFORMANCE
# ==============================

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

mae = mean_absolute_error(
    y_test,
    predictions
)


# Calculate MAPE safely
non_zero_mask = y_test != 0

mape = (
    np.mean(
        np.abs(
            (
                y_test[non_zero_mask]
                - predictions[non_zero_mask]
            )
            /
            y_test[non_zero_mask]
        )
    )
    * 100
)


print("\n=== ARIMA PERFORMANCE ===")

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")


# ==============================
# SAVE RESULTS
# ==============================

results.to_csv(
    "../data/arima_predictions.csv",
    index=False
)


print("\nARIMA predictions successfully saved!")

print(
    "Saved as: "
    "../data/arima_predictions.csv"
)