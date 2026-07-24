"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic import Field


class PredictionRequest(BaseModel):
    """Request payload with model-ready feature fields."""

    features: dict[str, Any] = Field(
        ..., description="Feature dictionary matching training-time input columns."
    )


class PredictionResponse(BaseModel):
    """Response schema for risk prediction."""

    risk_probability: float
    risk_label: str
