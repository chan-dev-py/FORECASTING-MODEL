import pandas as pd

# ==========================================
# 06 - PREPARE TRAIN AND TEST DATA
# ==========================================

# Load feature-engineered dataset
input_path = (
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College"
    r"\Documents\THESIS\data\feature_engineered_data.csv"
)

df = pd.read_csv(input_path)

print("=== ORIGINAL DATA ===")
print(df.head())

# ------------------------------------------
# SORT DATA CHRONOLOGICALLY
# ------------------------------------------

df = df.sort_values(
    by=["PROGRAM", "TIME_INDEX"]
).reset_index(drop=True)

# ------------------------------------------
# SPLIT EACH PROGRAM INTO TRAIN AND TEST
# ------------------------------------------

train_list = []
test_list = []

# Reserve the last 2 observations of each program
TEST_SIZE = 2

for program, group in df.groupby("PROGRAM"):

    # Sort each program chronologically
    group = group.sort_values("TIME_INDEX")

    # Split
    train = group.iloc[:-TEST_SIZE]
    test = group.iloc[-TEST_SIZE:]

    train_list.append(train)
    test_list.append(test)

# Combine all programs
train_df = pd.concat(train_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

# ------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------

print("\n=== TRAINING DATA ===")
print(train_df.head(20))

print("\n=== TEST DATA ===")
print(test_df.head(20))

print("\n=== DATASET SUMMARY ===")
print(f"Original records: {len(df)}")
print(f"Training records: {len(train_df)}")
print(f"Testing records: {len(test_df)}")
print(f"Programs: {df['PROGRAM'].nunique()}")

print("\n=== OBSERVATIONS PER PROGRAM ===")

summary = pd.DataFrame({
    "TOTAL": df.groupby("PROGRAM").size(),
    "TRAIN": train_df.groupby("PROGRAM").size(),
    "TEST": test_df.groupby("PROGRAM").size()
})

print(summary)

# ------------------------------------------
# SAVE TRAINING AND TEST DATA
# ------------------------------------------

train_path = (
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College"
    r"\Documents\THESIS\data\train_data.csv"
)

test_path = (
    r"C:\Users\User\OneDrive - Gov. Alfonso D. Tan College"
    r"\Documents\THESIS\data\test_data.csv"
)

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("\nTrain-test preparation completed successfully!")
print(f"Training data saved as: {train_path}")
print(f"Testing data saved as: {test_path}")