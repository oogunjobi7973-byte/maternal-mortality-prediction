from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=False)
    role = Column(String(50), nullable=False, default="healthcare_worker")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    patients = relationship("Patient", back_populates="creator")


class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column(Integer, primary_key=True, index=True)
    patient_code = Column(String(40), unique=True, nullable=False, index=True)
    age = Column(Integer, nullable=False)
    residence = Column(Integer, nullable=False)
    education = Column(Integer, nullable=False)
    wealth = Column(Integer, nullable=False)
    children = Column(Integer, nullable=False)
    first_birth = Column(Integer, nullable=False)
    living_children = Column(Integer, nullable=False)
    created_by = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    creator = relationship("User", back_populates="patients")
    predictions = relationship(
        "Prediction",
        back_populates="patient",
        cascade="all, delete-orphan",
    )


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(
        Integer,
        ForeignKey("patients.patient_id"),
        nullable=False,
        index=True,
    )
    risk_class = Column(String(30), nullable=False)
    risk_probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    model_used = Column(
        String(100),
        nullable=False,
        default="Explainable Boosting Machine",
    )
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    patient = relationship("Patient", back_populates="predictions")
    explanations = relationship(
        "SHAPExplanation",
        back_populates="prediction",
        cascade="all, delete-orphan",
    )


class SHAPExplanation(Base):
    __tablename__ = "shap_explanations"

    explanation_id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(
        Integer,
        ForeignKey("predictions.prediction_id"),
        nullable=False,
        index=True,
    )
    feature_name = Column(String(150), nullable=False)
    impact = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    prediction = relationship("Prediction", back_populates="explanations")