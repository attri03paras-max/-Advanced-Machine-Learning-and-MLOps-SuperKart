import pandas as pd
import os

# Load data
df = pd.read_csv("data/superkart_data.csv")

# Drop missing values
df.dropna(inplace=True)

# Separate target column (IMPORTANT)
target_column = "Item_Outlet_Sales"  # change if needed

y = df[target_column]
X = df.drop(columns=[target_column])

# Convert categorical to numeric
X = pd.get_dummies(X)

# Combine again
df_processed = pd.concat([X, y], axis=1)

# Save processed data
os.makedirs("data/processed", exist_ok=True)
df_processed.to_csv("data/processed/processed_data.csv", index=False)

print("✅ Preprocessing done")
