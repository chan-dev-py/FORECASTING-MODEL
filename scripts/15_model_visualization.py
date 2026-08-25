import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# MODEL PERFORMANCE VISUALIZATION
# ==========================================

print("=== MODEL PERFORMANCE VISUALIZATION ===")

# Load model comparison results
comparison = pd.read_csv("../data/model_comparison.csv")

print("\nLoaded model comparison data:")
print(comparison)

# ------------------------------------------
# Chart 1: RMSE Comparison
# ------------------------------------------
plt.figure(figsize=(10, 6))

plt.bar(comparison["MODEL"], comparison["RMSE"])

plt.title("Model Comparison Based on RMSE")
plt.xlabel("Forecasting Model")
plt.ylabel("RMSE")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "../data/model_rmse_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nRMSE chart successfully saved!")

# ------------------------------------------
# Chart 2: MAE Comparison
# ------------------------------------------
plt.figure(figsize=(10, 6))

plt.bar(comparison["MODEL"], comparison["MAE"])

plt.title("Model Comparison Based on MAE")
plt.xlabel("Forecasting Model")
plt.ylabel("MAE")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "../data/model_mae_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("MAE chart successfully saved!")

# ------------------------------------------
# Chart 3: MAPE Comparison
# ------------------------------------------
plt.figure(figsize=(10, 6))

plt.bar(comparison["MODEL"], comparison["MAPE"])

plt.title("Model Comparison Based on MAPE")
plt.xlabel("Forecasting Model")
plt.ylabel("MAPE (%)")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "../data/model_mape_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("MAPE chart successfully saved!")

# ------------------------------------------
# Chart 4: Overall Model Ranking
# ------------------------------------------
plt.figure(figsize=(10, 6))

plt.bar(
    comparison["MODEL"],
    comparison["AVERAGE_RANK"]
)

plt.title("Overall Model Ranking")
plt.xlabel("Forecasting Model")
plt.ylabel("Average Rank")

plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    "../data/model_ranking_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Ranking chart successfully saved!")

print("\n=== ALL VISUALIZATIONS SUCCESSFULLY CREATED! ===")