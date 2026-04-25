import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/processed/processed_data.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = joblib.load("models/model.pkl")

preds = model.predict(X)

mse = mean_squared_error(y, preds)

print("MSE:", mse)
