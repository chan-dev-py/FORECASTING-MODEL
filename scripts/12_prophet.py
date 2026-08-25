import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# ==============================
# LOAD DATA
# ==============================

train = pd.read_csv("../data/train_data.csv")
test = pd.read_csv("../data/test_data.csv")

print("=== PROPHET FORECASTING ===")

print("\n=== TRAINING DATA ===")
print("Training records:", len(train))

print("\n=== TESTING DATA ===")
print("Testing records:", len(test))


# ==============================
# STORE PREDICTIONS
# ==============================

all_predictions = []


# ==============================
# FORECAST EACH PROGRAM
# ==============================

programs = test["PROGRAM"].unique()

for program in programs:

    print(f"\nProcessing program: {program}")

    # Get training data for the current program
    program_train = train[
        train["PROGRAM"] == program
    ].copy()

    # Get testing data for the current program
    program_test = test[
        test["PROGRAM"] == program
    ].copy()

    # Skip if there is not enough training data
    if len(program_train) < 4:
        print("Not enough training data. Skipping...")
        continue


    # ==============================
    # PREPARE DATA FOR PROPHET
    # ==============================

    program_train = program_train.sort_values(
        "PROGRAM_TIME_INDEX"
    )

    # Create artificial time-based dates
    start_date = pd.Timestamp("2018-01-01")

    program_train["ds"] = program_train[
        "PROGRAM_TIME_INDEX"
    ].apply(
        lambda x: start_date + pd.DateOffset(months=(x - 1) * 6)
    )

    program_train["y"] = program_train[
        "TOTAL_ENROLLMENT"
    ]


    # ==============================
    # TRAIN PROPHET MODEL
    # ==============================

    prophet_train = program_train[
        ["ds", "y"]
    ]

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False
    )

    model.fit(prophet_train)


    # ==============================
    # PREPARE FUTURE PERIODS
    # ==============================

    program_test = program_test.sort_values(
        "PROGRAM_TIME_INDEX"
    )

    program_test["ds"] = program_test[
        "PROGRAM_TIME_INDEX"
    ].apply(
        lambda x: start_date + pd.DateOffset(months=(x - 1) * 6)
    )


    # ==============================
    # MAKE PREDICTIONS
    # ==============================

    forecast = model.predict(
        program_test[["ds"]]
    )

    predicted_values = forecast["yhat"].values


    # Prevent negative enrollment predictions
    predicted_values = np.maximum(
        predicted_values,
        0
    )


    # ==============================
    # SAVE RESULTS
    # ==============================

    program_results = program_test[
        [
            "PROGRAM",
            "ACADEMIC_YEAR",
            "SEMESTER",
            "TOTAL_ENROLLMENT"
        ]
    ].copy()

    program_results[
        "PREDICTED_ENROLLMENT"
    ] = predicted_values

    all_predictions.append(program_results)


# ==============================
# COMBINE ALL PREDICTIONS
# ==============================

predictions = pd.concat(
    all_predictions,
    ignore_index=True
)


# ==============================
# CALCULATE PERFORMANCE
# ==============================

actual = predictions["TOTAL_ENROLLMENT"]
predicted = predictions["PREDICTED_ENROLLMENT"]

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted
    )
)

mae = mean_absolute_error(
    actual,
    predicted
)


# Calculate MAPE safely
non_zero_actual = actual != 0

mape = np.mean(
    np.abs(
        (
            actual[non_zero_actual]
            - predicted[non_zero_actual]
        )
        / actual[non_zero_actual]
    )
) * 100


# ==============================
# DISPLAY RESULTS
# ==============================

print("\n=== PROPHET PREDICTIONS ===")

print(
    predictions.to_string(
        index=False
    )
)

print("\n=== PROPHET PERFORMANCE ===")

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")


# ==============================
# SAVE PREDICTIONS
# ==============================

predictions.to_csv(
    "../data/prophet_predictions.csv",
    index=False
)

print(
    "\nProphet predictions successfully saved!"
)