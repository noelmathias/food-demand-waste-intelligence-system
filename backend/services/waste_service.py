import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = "../models/waste_model.pkl"
FEATURES_PATH = "../models/waste_model_features.pkl"
HISTORY_PATH = "../data/history.csv"

waste_model = joblib.load(MODEL_PATH)
waste_features = joblib.load(FEATURES_PATH)

def predict_waste(data: dict):

    # -----------------------------
    # SAFE HISTORY LOAD
    # -----------------------------
    if os.path.exists(HISTORY_PATH):
        try:
            history = pd.read_csv(HISTORY_PATH)
        except:
            history = pd.DataFrame(columns=["store", "item", "date", "demand"])
    else:
        history = pd.DataFrame(columns=["store", "item", "date", "demand"])

    # -----------------------------
    # SAFE FILTER (NO KEYERROR)
    # -----------------------------
    if "store" in data and "item" in data:
        item_history = history[
            (history["store"] == data["store"]) &
            (history["item"] == data["item"])
        ]
    else:
        item_history = pd.DataFrame()

    # -----------------------------
    # DERIVE FEATURES
    # -----------------------------
    if not item_history.empty:
        avg_waste = float(item_history["demand"].mean())
        total_waste = float(item_history["demand"].sum())
    else:
        # Cold start fallback
        avg_waste = 20.0
        total_waste = 100.0

    economic_loss = avg_waste * 10
    population = 50  # million

    # -----------------------------
    # SAFE INPUT KEYS
    # -----------------------------
    raw_input = {
        "Total_Waste_Tons": total_waste,
        "Economic_Loss_Million": economic_loss,
        "Avg_Waste_per_Capita_Kg": avg_waste,
        "Population_Million": population,
        "Year": data.get("waste_year", data.get("year", 2020)),
        "Country": data.get("country", "India"),
        "Food_Category": data.get("food_category", "General")
    }

    df = pd.DataFrame([raw_input])
    df = pd.get_dummies(df)
    df = df.reindex(columns=waste_features, fill_value=0)

    pred_log = waste_model.predict(df)[0]
    prediction = np.expm1(pred_log)

    return float(round(prediction, 2))