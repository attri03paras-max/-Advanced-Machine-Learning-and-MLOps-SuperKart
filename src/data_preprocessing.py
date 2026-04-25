import pandas as pd
import os

df = pd.read_csv("data/superkart_data.csv")

df.dropna(inplace=True)
df = pd.get_dummies(df)

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/processed_data.csv", index=False)

print("Preprocessing done")
