import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI(
    title="Early Maternal Mortality Prediction API",
    version="1.0"
)

# Locate models folder
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

# Load model and preprocessor
ebm_model = joblib.load(MODEL_DIR / "ebm_model.pkl")
preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")


# -----------------------------
# Input Data Model
# -----------------------------
class PatientData(BaseModel):
    age: int
    residence: int
    education: int
    wealth: int
    children: int
    first_birth: int
    living_children: int


@app.get("/")
def home():
    return {
        "message": "Early Maternal Mortality Prediction API is running!",
        "model_loaded": True
    }

@app.post("/predict")
def predict(data: PatientData):

    # Convert incoming JSON into a DataFrame
    input_data = pd.DataFrame([{
        "v012": data.age,
        "v025": data.residence,
        "v106": data.education,
        "v190": data.wealth,
        "v201": data.children,
        "v212": data.first_birth,
        "v218": data.living_children
    }])

    # Apply preprocessing
    processed_data = preprocessor.transform(input_data)

    # Make prediction
    prediction = ebm_model.predict(processed_data)[0]
    probability = ebm_model.predict_proba(processed_data)[0][1]

    return {
        "prediction": "High Risk" if prediction == 1 else "Low Risk",
        "probability": round(float(probability), 4)
    }