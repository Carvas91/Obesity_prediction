from pydantic import BaseModel

class ObesityData(BaseModel):
    Gender: str
    Age: float
    Height: float
    #Weight: float
    family_history_with_overweight: str
    FAVC: str  # Frequent consumption of high-caloric food (yes/no)
    FCVC: int  # Frequency of consumption of vegetables
    NCP: float  # Number of main meals per day
    CAEC: str  # Consumption of food between meals
    SMOKE: str  # Smokes or not (yes/no)
    CH2O: float  # Daily water consumption in liters
    SCC: str  # Monitors calories (yes/no)
    FAF: float  # Physical activity frequency (times per week)
    TUE: int  # Time spent on technology (hours per day)
    CALC: str  # Alcohol consumption frequency
    MTRANS: str  # Mode of transportation used

