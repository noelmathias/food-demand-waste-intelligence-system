from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas.input_schema import PredictionInput
from backend.schemas.waste_input import WasteInput

from backend.services.demand_service import predict_demand
from backend.services.waste_service import predict_waste
from backend.services.decision_service import make_decision

app = FastAPI(title="Food Demand & Waste Intelligence API")

# ✅ CORS (important for Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# HEALTH CHECK
# ---------------------------
@app.get("/")
def home():
    return {"message": "API is running"}

# ---------------------------
# DEMAND PREDICTION
# ---------------------------
@app.post("/predict-demand")
def predict_demand_api(data: dict):
    data = data
    demand = predict_demand(data)
    return {
        "demand": float(f"{float(demand):.2f}")

    }

# ---------------------------
# WASTE PREDICTION
# ---------------------------
@app.post("/predict-waste")
def predict_waste_api(data: dict):
    data = data
    waste = predict_waste(data)
    return {
        "waste":  float(f"{float(waste):.2f}")
    }

# ---------------------------
# FINAL DECISION (COMBINED)
# ---------------------------
@app.post("/predict-all")
def predict_all(data: dict):
    # If using Pydantic earlier → remove it for now

    demand = predict_demand(data)

    # If you have real waste model:
    try:
        waste = predict_waste(data)
    except:
        waste = demand * 0.5  # fallback

    decision = make_decision(demand, waste)

    return {
        "demand": round(demand, 2),
        "waste": round(waste, 2),
        "risk": decision["risk"],
        "recommended_production": round(decision["recommended_production"], 2)
    }