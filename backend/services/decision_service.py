def make_decision(demand: float, waste: float):
    """
    Generate production recommendation based on waste risk level
    """

    if waste > 50:
        recommended = demand * 0.7
        risk = "HIGH"
    elif waste > 30:
        recommended = demand * 0.85
        risk = "MEDIUM"
    else:
        recommended = demand
        risk = "LOW"

    return{
        "recommended_production": round(recommended, 2),
        "risk": risk     
    }
