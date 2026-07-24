"""Pydantic schemas for API requests and responses."""

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """Response schema for risk prediction."""

    risk_probability: float
