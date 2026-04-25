import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

df = pd.read_csv("data/processed/processed_data.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

print("Training done")
