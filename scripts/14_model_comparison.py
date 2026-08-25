import pandas as pd

# ==========================================
# MODEL PERFORMANCE COMPARISON
# ==========================================

print("=== MODEL PERFORMANCE COMPARISON ===\n")

# Performance results from each model
results = {
    "MODEL": [
        "Linear Regression",
        "Random Forest",
        "ARIMA",
        "Prophet"
    ],

    "RMSE": [
        223.80,
        138.82,
        126.95,
        195.18
    ],

    "MAE": [
        187.67,
        86.17,
        66.01,
        109.07
    ],

    "MAPE": [
        2086.33,
        663.07,
        40.61,
        208.85
    ]
}

# Convert results into DataFrame
comparison = pd.DataFrame(results)

# Rank models
# Lower RMSE, MAE, and MAPE = better performance
comparison["RMSE_RANK"] = comparison["RMSE"].rank(method="min")
comparison["MAE_RANK"] = comparison["MAE"].rank(method="min")
comparison["MAPE_RANK"] = comparison["MAPE"].rank(method="min")

# Calculate average rank
comparison["AVERAGE_RANK"] = (
    comparison["RMSE_RANK"]
    + comparison["MAE_RANK"]
    + comparison["MAPE_RANK"]
) / 3

# Sort from best to worst
comparison = comparison.sort_values(
    by="AVERAGE_RANK",
    ascending=True
)

# Display results
print(comparison.to_string(index=False))

# Identify best model
best_model = comparison.iloc[0]

print("\n=== BEST PERFORMING MODEL ===")
print(f"Model: {best_model['MODEL']}")
print(f"RMSE: {best_model['RMSE']:.2f}")
print(f"MAE: {best_model['MAE']:.2f}")
print(f"MAPE: {best_model['MAPE']:.2f}%")
print(f"Average Rank: {best_model['AVERAGE_RANK']:.2f}")

# Save comparison results
comparison.to_csv(
    "../data/model_comparison.csv",
    index=False
)

print("\nModel comparison successfully saved!")