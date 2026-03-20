import joblib
import pandas as pd
import numpy as np

# =========================
# 1. LOAD MODELS
# =========================
demand_model = joblib.load("models/demand_model.pkl")
waste_model = joblib.load("models/waste_model.pkl")
waste_features = joblib.load("models/waste_model_features.pkl")

# =========================
# 2. CREATE SAMPLE INPUT
# =========================
# (later this will come from UI)

demand_input = pd.DataFrame([{
    "store": 1,
    "item": 1,
    "day_of_week": 2,
    "month": 5,
    "year": 2017,
    "lag_1": 20,
    "lag_7": 18,
    "lag_30": 15,
    "rolling_mean_7": 19,
    "rolling_std_7": 2
}])

# =========================
# 3. PREDICT DEMAND
# =========================
predicted_demand = demand_model.predict(demand_input)[0]

# =========================
# 4. CREATE WASTE INPUT
# =========================
# IMPORTANT: must match waste model features

waste_input = pd.DataFrame([{
    "Total_Waste_Tons": 20000,
    "Economic_Loss_Million": 15000,
    "Avg_Waste_per_Capita_Kg": 120,
    "Population_Million": 1000,
    "Year": 2022,
    "Country":"India",
    "Food_Category":"Fruits & Vegetables"

}])

waste_input = pd.get_dummies(waste_input)

waste_features = joblib.load("models/waste_model_features.pkl")

waste_input = waste_input.reindex(columns=waste_features, fill_value=0)

# (If you used one-hot encoding, columns must match exactly — we’ll improve later)

# =========================
# 5. PREDICT WASTE
# =========================
waste_pred_log = waste_model.predict(waste_input)[0]

# reverse log transform
predicted_waste = np.expm1(waste_pred_log)

# =========================
# 6. DECISION LOGIC
# =========================
if predicted_waste > 50:
    recommended = predicted_demand * 0.7
    risk = "HIGH"
elif predicted_waste > 30:
    recommended = predicted_demand * 0.85
    risk = "MEDIUM"
else:
    recommended = predicted_demand
    risk = "LOW"

# =========================
# 7. OUTPUT
# =========================
print("\n===== DECISION ENGINE OUTPUT =====")
print("Predicted Demand:", round(predicted_demand, 2))
print("Predicted Waste %:", round(predicted_waste, 2))
print("Waste Risk Level:", risk)
print("Recommended Production:", round(recommended, 2))