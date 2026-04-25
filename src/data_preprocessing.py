import pandas as pd
import os

df = pd.read_csv("data/superkart_data.csv")

df.dropna(inplace=True)

# Last column = target
y = df.iloc[:, -1]
X = df.iloc[:, :-1]

# Convert categorical to numeric
X = pd.get_dummies(X)

# Combine
df_processed = pd.concat([X, y], axis=1)

os.makedirs("data/processed", exist_ok=True)
df_processed.to_csv("data/processed/processed_data.csv", index=False)

print("✅ Preprocessing done")
