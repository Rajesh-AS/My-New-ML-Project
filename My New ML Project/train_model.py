import os
import joblib
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/train.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Force datetime conversion (FIX)
df["dteday"] = pd.to_datetime(df["dteday"], errors="coerce")

# Remove invalid rows
df = df.dropna(subset=["dteday"])

df["weekday"] = df["dteday"].dt.weekday
df["month"] = df["dteday"].dt.month

X = df[
    [
        "season",
        "holiday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "weekday",
        "month",
    ]
]

y = df["cnt"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

selector = SelectKBest(score_func=f_regression, k="all")
X_selected = selector.fit_transform(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, "models/bike_demand_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(selector, "models/feature_selector.pkl")

print("✅ Model trained and saved successfully")
print("📁 Models folder:", os.listdir("models"))
