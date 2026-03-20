import pandas as pd
from datetime import datetime
import os
import joblib
from datetime import datetime



demand_model = joblib.load("../models/demand_model.pkl")
HISTORY_PATH = "../data/history.csv"

def predict_demand(data):

    # ✅ 🔥 STEP 1: CONVERT TYPES FIRST
    data["store"] = int(data["store"])
    data["item"] = int(data["item"])
    data["day"] = int(data["day"])
    data["month"] = int(data["month"])
    data["year"] = int(data["year"])

    # ✅ STEP 2: DATE CALCULATION
    date_obj = datetime(data["year"], data["month"], data["day"])
    day_of_week = date_obj.weekday()

    # 🔹 Load history
    if os.path.exists(HISTORY_PATH):
        try:
            history = pd.read_csv(HISTORY_PATH)
        except:
            history = pd.DataFrame(columns=["store", "item", "date", "demand"])
    else:
        history = pd.DataFrame(columns=["store", "item", "date", "demand"])

    # 🔹 Filter store + item
    hist = history[
        (history["store"] == data["store"]) &
        (history["item"] == data["item"])
    ].copy()

    # 🔹 Convert date
    if not hist.empty:
        hist["date"] = pd.to_datetime(hist["date"])
        hist = hist.sort_values("date")

    # 🔥 FEATURE ENGINEERING
    if len(hist) >= 7:
        lag_1 = hist.iloc[-1]["demand"]
        lag_7 = hist.iloc[-7]["demand"]
        lag_30 = hist.iloc[-30]["demand"] if len(hist) >= 30 else lag_7
        rolling_mean_7 = hist["demand"].tail(7).mean()
        rolling_std_7 = hist["demand"].tail(7).std()

    elif len(hist) > 0:
        lag_1 = hist.iloc[-1]["demand"]
        lag_7 = hist["demand"].mean()
        lag_30 = lag_7
        rolling_mean_7 = hist["demand"].mean()
        rolling_std_7 = hist["demand"].std() if len(hist) > 1 else 1

    else:
        lag_1 = 20
        lag_7 = 20
        lag_30 = 20
        rolling_mean_7 = 20
        rolling_std_7 = 5

    # ✅ STEP 3: CREATE DF (NOW CORRECT TYPES)
    df = pd.DataFrame([{
        "store": data["store"],
        "item": data["item"],
        "day_of_week": day_of_week,
        "month": data["month"],
        "year": data["year"],
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_30": lag_30,
        "rolling_mean_7": rolling_mean_7,
        "rolling_std_7": rolling_std_7
    }])

    # 🔹 Predict
    prediction = demand_model.predict(df)[0]

    # 🔹 Save prediction
    new_row = pd.DataFrame([{
        "store": data["store"],
        "item": data["item"],
        "date": datetime.now().strftime("%Y-%m-%d"),
        "demand": prediction
    }])

    history = pd.concat([history, new_row], ignore_index=True)
    history.to_csv(HISTORY_PATH, index=False)

    return float(round(prediction, 2))