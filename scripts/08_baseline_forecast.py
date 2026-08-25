import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# ==========================================
# LOAD TRAINING AND TESTING DATA
# ==========================================

train = pd.read_csv(
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College\Documents\THESIS\data\train_data.csv"
)

test = pd.read_csv(
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College\Documents\THESIS\data\test_data.csv"
)

print("=== BASELINE FORECAST ===")

# Store all predictions
results = []

# ==========================================
# NAIVE FORECAST PER PROGRAM
# ==========================================

for program in test["PROGRAM"].unique():

    # Get training data for the current program
    train_program = train[
        train["PROGRAM"] == program
    ].sort_values("PROGRAM_TIME_INDEX")

    # Get testing data for the current program
    test_program = test[
        test["PROGRAM"] == program
    ].sort_values("PROGRAM_TIME_INDEX")

    # Get the last enrollment value from training data
    last_enrollment = train_program[
        "TOTAL_ENROLLMENT"
    ].iloc[-1]

    # Predict the same value for all test observations
    predictions = [last_enrollment] * len(test_program)

    # Create result dataframe
    program_results = pd.DataFrame({
        "PROGRAM": program,
        "ACADEMIC_YEAR": test_program["ACADEMIC_YEAR"].values,
        "SEMESTER": test_program["SEMESTER"].values,
        "ACTUAL_ENROLLMENT": test_program[
            "TOTAL_ENROLLMENT"
        ].values,
        "PREDICTED_ENROLLMENT": predictions
    })

    results.append(program_results)


# ==========================================
# COMBINE ALL RESULTS
# ==========================================

results_df = pd.concat(results, ignore_index=True)

print("\n=== BASELINE PREDICTIONS ===")
print(results_df)


# ==========================================
# MODEL EVALUATION
# ==========================================

actual = results_df["ACTUAL_ENROLLMENT"]
predicted = results_df["PREDICTED_ENROLLMENT"]

rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

mae = mean_absolute_error(
    actual,
    predicted
)

# Avoid division by zero for MAPE
non_zero = actual != 0

mape = np.mean(
    np.abs(
        (actual[non_zero] - predicted[non_zero])
        / actual[non_zero]
    )
) * 100


print("\n=== BASELINE MODEL PERFORMANCE ===")

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")


# ==========================================
# SAVE RESULTS
# ==========================================

output_path = (
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College"
    r"\Documents\THESIS\data\baseline_predictions.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\nBaseline predictions successfully saved!")
print(f"Saved as: {output_path}")