"""
SQLAlchemy ORM Models for Fraud Detection System
Tables: users, transactions, alerts
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean,
    DateTime, ForeignKey, Text, Index,
)
from sqlalchemy.orm import relationship

from database.connection import Base


class User(Base):
    """Bank account user."""
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    account_number = Column(String(20), unique=True, nullable=False, index=True)
    email = Column(String(150), unique=True, nullable=True)
    phone = Column(String(15), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    transactions = relationship("Transaction", back_populates="user", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<User(user_id={self.user_id}, name={self.name}, account={self.account_number})>"


class Transaction(Base):
    """Banking/UPI transaction record with ML prediction."""
    __tablename__ = "transactions"

    transaction_id = Column(String(50), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    transaction_type = Column(String(30), nullable=True)
    merchant_category = Column(String(50), nullable=True)
    location = Column(String(100), nullable=True)
    device_id = Column(String(50), nullable=True)
    # timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    # ML prediction outputs
    prediction = Column(String(20), nullable=True)  # "Fraud" or "Legitimate"
    fraud_probability = Column(Float, nullable=True)
    risk_level = Column(String(20), nullable=True)  # "Low", "Medium", "High"

    # Feature snapshot
    avg_transaction_amount = Column(Float, nullable=True)
    daily_transaction_count = Column(Integer, nullable=True)
    unusual_time_flag = Column(Boolean, default=False)
    unusual_location_flag = Column(Boolean, default=False)
    device_change_flag = Column(Boolean, default=False)

    is_confirmed_fraud = Column(Boolean, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="transactions")
    alert = relationship("Alert", back_populates="transaction", uselist=False)

    __table_args__ = (
        Index("ix_transactions_timestamp_user", "timestamp", "user_id"),
        Index("ix_transactions_risk_level", "risk_level"),
    )

    def __repr__(self) -> str:
        return (
            f"<Transaction(id={self.transaction_id}, amount={self.amount}, "
            f"prediction={self.prediction}, prob={self.fraud_probability:.3f})>"
        )


class Alert(Base):
    """Fraud alert generated when fraud probability exceeds threshold."""
    __tablename__ = "alerts"

    alert_id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(50), ForeignKey("transactions.transaction_id"), nullable=False, unique=True)
    risk_level = Column(String(20), nullable=False)  # "Low", "Medium", "High"
    fraud_probability = Column(Float, nullable=False)
    alert_message = Column(Text, nullable=True)
    is_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    resolved_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    transaction = relationship("Transaction", back_populates="alert")

    # __table_args__ = (
    #     Index("ix_alerts_risk_level", "risk_level"),
    #     Index("ix_alerts_created_at", "created_at"),
    # )

    def __repr__(self) -> str:
        return (
            f"<Alert(id={self.alert_id}, transaction={self.transaction_id}, "
            f"risk={self.risk_level}, resolved={self.is_resolved})>"
        )