from pydantic import BaseModel

class WasteInput(BaseModel):
    total_waste: float
    economic_loss: float
    avg_waste: float
    population: float
    waste_year: int
    country: str
    food_category: str