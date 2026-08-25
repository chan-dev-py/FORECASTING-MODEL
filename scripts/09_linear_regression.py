import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ==========================================
# 1. LOAD TRAINING AND TESTING DATA
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_path = os.path.join(BASE_DIR, "data", "train_data.csv")
test_path = os.path.join(BASE_DIR, "data", "test_data.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("=== LINEAR REGRESSION FORECASTING ===")

print("\n=== TRAINING DATA ===")
print(f"Training records: {len(train_df)}")

print("\n=== TESTING DATA ===")
print(f"Testing records: {len(test_df)}")


# ==========================================
# 2. PREPARE FEATURES
# ==========================================

# Combine training and testing temporarily
# so both datasets have the same program columns

all_programs = pd.concat([
    train_df["PROGRAM"],
    test_df["PROGRAM"]
]).unique()

# Create program dummy variables

train_programs = pd.get_dummies(
    train_df["PROGRAM"]
).reindex(columns=all_programs, fill_value=0)

test_programs = pd.get_dummies(
    test_df["PROGRAM"]
).reindex(columns=all_programs, fill_value=0)


# Add time index

X_train = pd.concat(
    [
        train_df[["PROGRAM_TIME_INDEX"]].reset_index(drop=True),
        train_programs.reset_index(drop=True)
    ],
    axis=1
)

X_test = pd.concat(
    [
        test_df[["PROGRAM_TIME_INDEX"]].reset_index(drop=True),
        test_programs.reset_index(drop=True)
    ],
    axis=1
)


# Target variable

y_train = train_df["TOTAL_ENROLLMENT"]
y_test = test_df["TOTAL_ENROLLMENT"]


# ==========================================
# 3. TRAIN LINEAR REGRESSION MODEL
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")


# ==========================================
# 4. MAKE PREDICTIONS
# ==========================================

predictions = model.predict(X_test)

# Prevent negative enrollment predictions

predictions = np.maximum(predictions, 0)


# ==========================================
# 5. CREATE RESULTS TABLE
# ==========================================

results = test_df[
    [
        "PROGRAM",
        "ACADEMIC_YEAR",
        "SEMESTER",
        "TOTAL_ENROLLMENT"
    ]
].copy()

results["PREDICTED_ENROLLMENT"] = predictions

print("\n=== LINEAR REGRESSION PREDICTIONS ===")

print(
    results.to_string(index=False)
)


# ==========================================
# 6. EVALUATE MODEL
# ==========================================

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

mask = y_test != 0

mape = np.mean(
    np.abs(
        (
            y_test[mask]
            - predictions[mask]
        )
        / y_test[mask]
    )
) * 100


print("\n=== LINEAR REGRESSION PERFORMANCE ===")

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")


# ==========================================
# 7. SAVE PREDICTIONS
# ==========================================

output_path = os.path.join(
    BASE_DIR,
    "data",
    "linear_regression_predictions.csv"
)

results.to_csv(
    output_path,
    index=False
)

print("\nLinear Regression predictions successfully saved!")

print(
    f"Saved as: {output_path}"
)