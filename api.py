from pathlib import Path
import os
import uuid

import joblib
import pandas as pd
import shap
import jwt

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from werkzeug.security import check_password_hash, generate_password_hash

from .database import Base, SessionLocal, engine
from .models import Patient, Prediction, SHAPExplanation, User

# ==========================================================
# Database
# ==========================================================

Base.metadata.create_all(bind=engine)

# ==========================================================
# FastAPI Application
# ==========================================================

app = FastAPI(
    title="Early Maternal Mortality Prediction API",
    version="3.0.0",
    description=(
        "REST API for predicting maternal mortality risk using an "
        "Explainable Boosting Machine (EBM) with SHAP explainability, "
        "PostgreSQL persistence, and basic authentication."
    ),
)

# ==========================================================
# Authentication Configuration
# ==========================================================

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET environment variable is not set.")

JWT_ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 12
security = HTTPBearer()


def create_access_token(user_id: int, role: str) -> str:
    import datetime as dt
    now = dt.datetime.utcnow()
    payload = {
        "sub": str(user_id),
        "role": role,
        "iat": now,
        "exp": now + dt.timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
        )
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
        )

    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account not found.",
        )
    return user

# ==========================================================
# Authentication Schemas
# ==========================================================


class RegisterData(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginData(BaseModel):
    email: EmailStr
    password: str

# ==========================================================
# Model Files
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

ebm_model = joblib.load(MODEL_DIR / "ebm_model.pkl")
preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
shap_background = joblib.load(MODEL_DIR / "shap_background.pkl")

explainer = shap.Explainer(
    ebm_model.predict_proba,
    shap_background,
)

FEATURE_NAMES = [
    "Maternal Age",
    "Residence",
    "Education Level",
    "Wealth Index",
    "Children Ever Born",
    "Age at First Birth",
    "Births in Last 5 Years",
]

# ==========================================================
# Input Schema
# ==========================================================


class PatientData(BaseModel):
    age: int = Field(ge=15, le=49)
    residence: int = Field(ge=1, le=2)
    education: int = Field(ge=0, le=3)
    wealth: int = Field(ge=1, le=5)
    children: int = Field(ge=0, le=20)
    first_birth: int = Field(ge=10, le=49)
    living_children: int = Field(ge=0, le=20)

# ==========================================================
# Recommendation Generator
# ==========================================================


def generate_recommendation(prediction: int) -> str:
    if prediction == 1:
        return (
            "Immediate clinical assessment is recommended. The patient should "
            "receive closer monitoring and appropriate maternal healthcare interventions."
        )
    return (
        "Continue routine antenatal care and attend all scheduled maternal health visits."
    )

# ==========================================================
# Endpoints
# ==========================================================


@app.get("/")
def home():
    return {
        "message": "Early Maternal Mortality Prediction API is running.",
        "model_loaded": True,
        "shap_enabled": True,
        "database": "PostgreSQL",
        "authentication": "JWT",
    }


@app.post("/auth/register")
def register(data: RegisterData, db: Session = Depends(get_db)):
    email = data.email.lower().strip()
    existing = db.query(User).filter(User.email == email).first()
    if existing:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    user = User(
        name=data.name.strip(),
        email=email,
        password_hash=generate_password_hash(data.password),
        role="healthcare_worker",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.user_id, user.role)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"user_id": user.user_id, "name": user.name, "email": user.email, "role": user.role},
    }


@app.post("/auth/login")
def login(data: LoginData, db: Session = Depends(get_db)):
    email = data.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()
    if not user or not check_password_hash(user.password_hash, data.password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = create_access_token(user.user_id, user.role)
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"user_id": user.user_id, "name": user.name, "email": user.email, "role": user.role},
    }


@app.get("/auth/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "user_id": current_user.user_id,
        "name": current_user.name,
        "email": current_user.email,
        "role": current_user.role,
    }


@app.post("/predict")
def predict(
    data: PatientData,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    input_data = pd.DataFrame([{
        "v012": data.age,
        "v025": data.residence,
        "v106": data.education,
        "v190": data.wealth,
        "v201": data.children,
        "v212": data.first_birth,
        "v218": data.living_children,
    }])

    processed_data = preprocessor.transform(input_data)

    prediction = int(ebm_model.predict(processed_data)[0])
    probability = float(ebm_model.predict_proba(processed_data)[0][1])

    prediction_label = "High Risk" if prediction == 1 else "Low Risk"
    risk_probability = round(probability * 100, 2)
    confidence = round((probability if prediction == 1 else 1 - probability) * 100, 2)

    shap_values = explainer(processed_data)
    feature_importance = []

    for i, feature in enumerate(FEATURE_NAMES):
        contribution = float(shap_values.values[0][i][1])
        feature_importance.append({
            "feature": feature,
            "impact": round(contribution, 4),
        })

    risk_factors = sorted(
        [item for item in feature_importance if item["impact"] > 0],
        key=lambda x: x["impact"],
        reverse=True,
    )
    protective_factors = sorted(
        [item for item in feature_importance if item["impact"] < 0],
        key=lambda x: x["impact"],
    )

    patient_code = f"PT-{uuid.uuid4().hex[:10].upper()}"
    patient = Patient(
        patient_code=patient_code,
        age=data.age,
        residence=data.residence,
        education=data.education,
        wealth=data.wealth,
        children=data.children,
        first_birth=data.first_birth,
        living_children=data.living_children,
        created_by=current_user.user_id,
    )
    db.add(patient)
    db.flush()

    prediction_record = Prediction(
        patient_id=patient.patient_id,
        risk_class=prediction_label,
        risk_probability=risk_probability,
        confidence=confidence,
        model_used="Explainable Boosting Machine (EBM)",
    )
    db.add(prediction_record)
    db.flush()

    for item in feature_importance:
        db.add(SHAPExplanation(
            prediction_id=prediction_record.prediction_id,
            feature_name=item["feature"],
            impact=item["impact"],
        ))

    db.commit()

    patient_summary = {
        "patient_code": patient_code,
        "age": data.age,
        "residence": data.residence,
        "education": data.education,
        "wealth": data.wealth,
        "children": data.children,
        "first_birth": data.first_birth,
        "living_children": data.living_children,
    }

    return {
        "prediction_id": prediction_record.prediction_id,
        "prediction": prediction_label,
        "risk_probability": risk_probability,
        "confidence": confidence,
        "recommendation": generate_recommendation(prediction),
        "top_risk_factors": risk_factors[:3],
        "top_protective_factors": protective_factors[:3],
        "feature_importance": feature_importance,
        "patient": patient_summary,
    }


@app.get("/predictions/history")
def prediction_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    records = (
        db.query(Prediction, Patient)
        .join(Patient, Prediction.patient_id == Patient.patient_id)
        .filter(Patient.created_by == current_user.user_id)
        .order_by(Prediction.created_at.desc())
        .limit(100)
        .all()
    )

    return [
        {
            "prediction_id": prediction.prediction_id,
            "patient_code": patient.patient_code,
            "risk_class": prediction.risk_class,
            "risk_probability": prediction.risk_probability,
            "model_used": prediction.model_used,
            "created_at": prediction.created_at.isoformat(),
        }
        for prediction, patient in records
    ]
