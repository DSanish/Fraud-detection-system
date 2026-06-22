"""
Pydantic request/response schemas for the Fraud Detection API.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, validator


class TransactionRequest(BaseModel):
    """Input schema for /predict endpoint."""

    transaction_amount: float = Field(..., gt=0, description="Transaction amount in INR", example=50000.0)
    avg_transaction_amount: float = Field(default=1200.0, ge=0, description="User's average transaction amount", example=1200.0)
    daily_transaction_count: int = Field(default=1, ge=0, description="Number of transactions by user today", example=3)
    weekly_transaction_count: int = Field(default=5, ge=0, description="Number of transactions this week", example=5)
    monthly_transaction_count: int = Field(default=20, ge=0, description="Number of transactions this month", example=20)
    transaction_hour: int = Field(default=12, ge=0, le=23, description="Hour of the transaction (0–23)", example=2)
    transaction_type: Optional[str] = Field(default="UPI", description="Transaction type", example="UPI")
    merchant_category: Optional[str] = Field(default="Unknown", description="Merchant category", example="Electronics")
    unusual_time_flag: int = Field(default=0, ge=0, le=1, description="1 if transaction is at unusual hour", example=1)
    unusual_location_flag: int = Field(default=0, ge=0, le=1, description="1 if transaction is from unusual location", example=0)
    device_change_flag: int = Field(default=0, ge=0, le=1, description="1 if new device detected", example=1)
    user_id: Optional[int] = Field(default=None, description="User ID for logging", example=42)
    transaction_id: Optional[str] = Field(default=None, description="Transaction ID for logging")

    @validator("transaction_amount")
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError("Transaction amount must be positive")
        if v > 1_00_00_000:  # 1 Crore INR
            raise ValueError("Transaction amount exceeds maximum allowed limit")
        return v

    @validator("transaction_hour")
    def validate_hour(cls, v):
        if not 0 <= v <= 23:
            raise ValueError("Transaction hour must be between 0 and 23")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_amount": 50000.0,
                "avg_transaction_amount": 1200.0,
                "daily_transaction_count": 3,
                "weekly_transaction_count": 5,
                "monthly_transaction_count": 20,
                "transaction_hour": 2,
                "transaction_type": "UPI",
                "merchant_category": "Unknown",
                "unusual_time_flag": 1,
                "unusual_location_flag": 0,
                "device_change_flag": 1,
                "user_id": 42,
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for /predict endpoint."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of fraud (0–1)")
    prediction: str = Field(..., description="'Fraud' or 'Legitimate'")
    risk_level: Optional[str] = Field(None, description="Risk level: Low, Medium, or High")
    alert_triggered: bool = Field(..., description="Whether a fraud alert was created")
    alert_message: Optional[str] = Field(None, description="Alert details if triggered")
    processing_time_ms: float = Field(..., description="Prediction latency in milliseconds")
    timestamp: datetime = Field(..., description="Prediction timestamp (UTC)")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN_ABC123",
                "fraud_probability": 0.96,
                "prediction": "Fraud",
                "risk_level": "High",
                "alert_triggered": True,
                "alert_message": "[HIGH RISK ALERT] Suspicious transaction detected...",
                "processing_time_ms": 12.4,
                "timestamp": "2024-06-01T02:15:00Z",
            }
        }


class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""

    status: str
    model_loaded: bool
    database_connected: bool
    version: str
    timestamp: datetime


class MetricsResponse(BaseModel):
    """Response schema for /metrics endpoint."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    model_version: str
    last_trained: Optional[str]
    feature_count: int