import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ==============================
# LOAD TRAINING AND TESTING DATA
# ==============================

train_data = pd.read_csv("../data/train_data.csv")
test_data = pd.read_csv("../data/test_data.csv")


print("=== RANDOM FOREST FORECASTING ===")

print("\n=== TRAINING DATA ===")
print(f"Training records: {len(train_data)}")

print("\n=== TESTING DATA ===")
print(f"Testing records: {len(test_data)}")


# ==============================
# PREPARE FEATURES
# ==============================

# Convert PROGRAM into numerical values using one-hot encoding
X_train = pd.get_dummies(
    train_data[["PROGRAM", "TIME_INDEX", "SEMESTER"]],
    columns=["PROGRAM"]
)

X_test = pd.get_dummies(
    test_data[["PROGRAM", "TIME_INDEX", "SEMESTER"]],
    columns=["PROGRAM"]
)


# Make sure training and testing data have the same columns
X_train, X_test = X_train.align(
    X_test,
    join="left",
    axis=1,
    fill_value=0
)


# Target variable
y_train = train_data["TOTAL_ENROLLMENT"]
y_test = test_data["TOTAL_ENROLLMENT"]


# ==============================
# BUILD RANDOM FOREST MODEL
# ==============================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ==============================
# TRAIN THE MODEL
# ==============================

model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully!")


# ==============================
# MAKE PREDICTIONS
# ==============================

predictions = model.predict(X_test)


# ==============================
# CREATE RESULTS DATAFRAME
# ==============================

results = test_data[
    ["PROGRAM", "ACADEMIC_YEAR", "SEMESTER", "TOTAL_ENROLLMENT"]
].copy()

results["PREDICTED_ENROLLMENT"] = predictions


print("\n=== RANDOM FOREST PREDICTIONS ===")
print(results.to_string(index=False))


# ==============================
# MODEL PERFORMANCE
# ==============================

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

mae = mean_absolute_error(
    y_test,
    predictions
)


# Calculate MAPE safely
non_zero_mask = y_test != 0

mape = (
    abs(
        (y_test[non_zero_mask] - predictions[non_zero_mask])
        / y_test[non_zero_mask]
    ).mean()
    * 100
)


print("\n=== RANDOM FOREST PERFORMANCE ===")

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")


# ==============================
# SAVE PREDICTIONS
# ==============================

results.to_csv(
    "../data/random_forest_predictions.csv",
    index=False
)

print("\nRandom Forest predictions successfully saved!")

print(
    "Saved as: "
    "../data/random_forest_predictions.csv"
)