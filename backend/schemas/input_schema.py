from pydantic import BaseModel

class PredictionInput(BaseModel):
    # Demand Features
    store: int
    item: int
    day_of_week: int
    month: int
    year: int
    country : str
    food_category: str

   
    
