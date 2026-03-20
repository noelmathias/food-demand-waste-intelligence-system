import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import joblib

# =========================
# 1. LOAD RAW DATA
# =========================
df = pd.read_csv("data/waste/waste.csv")

# =========================
# 2. CLEAN COLUMN NAMES
# =========================
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("(", "")
df.columns = df.columns.str.replace(")", "")
df.columns = df.columns.str.replace("%", "percent")
df.columns = df.columns.str.replace("$", "")

# =========================
# 3. DROP NULL VALUES
# =========================
df = df.dropna()

print(df.columns)

df["waste_per_population"] = df["Total_Waste_Tons"] / (df['Population_Million'] + 1e-6)
df["economic_efficiency"] = df["Total_Waste_Tons"] / (df["Economic_Loss_Million_"] + 1e-6)

# =========================
# 4. ENCODE CATEGORICAL VARIABLES
# =========================
df = pd.get_dummies(df, columns=["Country", "Food_Category"], drop_first=True)

# =========================
# 5. DEFINE TARGET
# =========================
target = "Household_Waste_percent"

X = df.drop(columns=[target])
y = np.log1p(df[target])

# =========================
# 6. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 7. TRAIN MODEL
# =========================
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(X_train.columns.tolist(), "models/waste_model_features.pkl")
# =========================
# 8. PREDICTIONS
# =========================
preds = model.predict(X_test)

preds = np.expm1(preds)
y_test = np.expm1(y_test)
# =========================
# 9. EVALUATION
# =========================
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mape = mean_absolute_percentage_error(y_test, preds)

print("Waste Model Performance")
print("------------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("MAPE:", round(mape * 100, 2), "%")

# =========================
# 10. SAVE MODEL
# =========================
joblib.dump(model, "models/waste_model.pkl")

print("\nWaste model saved successfully!")
print(len(df))