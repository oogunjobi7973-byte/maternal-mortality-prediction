from pathlib import Path

import joblib
import pandas as pd
import shap

from fastapi import FastAPI
from pydantic import BaseModel

# ==========================================================
# FastAPI Application
# ==========================================================

app = FastAPI(
    title="Early Maternal Mortality Prediction API",
    version="2.0.0",
    description=(
        "REST API for predicting maternal mortality risk "
        "using an Explainable Boosting Machine (EBM) "
        "with SHAP explainability."
    )
)

# ==========================================================
# Load Model Files
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

ebm_model = joblib.load(MODEL_DIR / "ebm_model.pkl")
preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
shap_background = joblib.load(MODEL_DIR / "shap_background.pkl")

# ==========================================================
# SHAP Explainer
# ==========================================================

explainer = shap.Explainer(
    ebm_model.predict_proba,
    shap_background
)

# ==========================================================
# Feature Names
# ==========================================================

FEATURE_NAMES = [
    "Maternal Age",
    "Residence",
    "Education Level",
    "Wealth Index",
    "Children Ever Born",
    "Age at First Birth",
    "Births in Last 5 Years"
]

# ==========================================================
# Input Schema
# ==========================================================

class PatientData(BaseModel):
    age: int
    residence: int
    education: int
    wealth: int
    children: int
    first_birth: int
    living_children: int


# ==========================================================
# Recommendation Generator
# ==========================================================

def generate_recommendation(prediction: int) -> str:

    if prediction == 1:
        return (
            "Immediate clinical assessment is recommended. "
            "The patient should receive closer monitoring "
            "and appropriate maternal healthcare interventions."
        )

    return (
        "Continue routine antenatal care and attend all "
        "scheduled maternal health visits."
    )


# ==========================================================
# Home Endpoint
# ==========================================================

@app.get("/")
def home():

    return {
        "message": "Early Maternal Mortality Prediction API is running.",
        "model_loaded": True,
        "shap_enabled": True
    }
# ==========================================================
# Prediction Endpoint
# ==========================================================

@app.post("/predict")
def predict(data: PatientData):

    # ------------------------------------------------------
    # Create Input DataFrame
    # ------------------------------------------------------

    input_data = pd.DataFrame([{
        "v012": data.age,
        "v025": data.residence,
        "v106": data.education,
        "v190": data.wealth,
        "v201": data.children,
        "v212": data.first_birth,
        "v218": data.living_children
    }])

    # ------------------------------------------------------
    # Apply preprocessing
    # ------------------------------------------------------

    processed_data = preprocessor.transform(input_data)

    # ------------------------------------------------------
    # Generate prediction
    # ------------------------------------------------------

    prediction = int(
        ebm_model.predict(processed_data)[0]
    )

    probability = float(
        ebm_model.predict_proba(processed_data)[0][1]
    )

    prediction_label = (
        "High Risk"
        if prediction == 1
        else "Low Risk"
    )

    risk_probability = round(
        probability * 100,
        2
    )

    confidence = round(
        (1 - probability) * 100,
        2
    )

    # ------------------------------------------------------
    # SHAP Explanation
    # ------------------------------------------------------

    shap_values = explainer(processed_data)

    feature_importance = []

    for i, feature in enumerate(FEATURE_NAMES):

        contribution = float(
            shap_values.values[0][i][1]
        )

        feature_importance.append({
            "feature": feature,
            "impact": round(contribution, 4)
        })
    # ------------------------------------------------------
    # Separate SHAP contributions
    # ------------------------------------------------------

    risk_factors = sorted(
        [
            item
            for item in feature_importance
            if item["impact"] > 0
        ],
        key=lambda x: x["impact"],
        reverse=True
    )

    protective_factors = sorted(
        [
            item
            for item in feature_importance
            if item["impact"] < 0
        ],
        key=lambda x: x["impact"]
    )

    # ------------------------------------------------------
    # Keep only the strongest explanations
    # ------------------------------------------------------

    top_risk_factors = risk_factors[:3]
    top_protective_factors = protective_factors[:3]

    # ------------------------------------------------------
    # Build patient summary
    # ------------------------------------------------------

    patient_summary = {
        "age": data.age,
        "residence": data.residence,
        "education": data.education,
        "wealth": data.wealth,
        "children": data.children,
        "first_birth": data.first_birth,
        "living_children": data.living_children
    }
    # ------------------------------------------------------
    # Final API Response
    # ------------------------------------------------------

    return {
        "prediction": prediction_label,
        "risk_probability": risk_probability,
        "confidence": confidence,
        "recommendation": generate_recommendation(prediction),

        # SHAP Explainability
        "top_risk_factors": top_risk_factors,
        "top_protective_factors": top_protective_factors,

        # Full SHAP values (optional for future visualization)
        "feature_importance": feature_importance,

        # Patient Information
        "patient": patient_summary
    }