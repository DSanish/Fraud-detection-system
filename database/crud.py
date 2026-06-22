"""
CRUD operations for Fraud Detection System database.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from database.models import User, Transaction, Alert

logger = logging.getLogger(__name__)


# ─── USER OPERATIONS ────────────────────────────────────────────────────────

def create_user(db: Session, name: str, account_number: str, email: Optional[str] = None, phone: Optional[str] = None) -> User:
    user = User(name=name, account_number=account_number, email=email, phone=phone)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.user_id == user_id).first()


def get_user_by_account(db: Session, account_number: str) -> Optional[User]:
    return db.query(User).filter(User.account_number == account_number).first()


def get_all_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    return db.query(User).offset(skip).limit(limit).all()


# ─── TRANSACTION OPERATIONS ──────────────────────────────────────────────────

def create_transaction(db: Session, txn_data: Dict[str, Any]) -> Transaction:
    """Create and persist a new transaction with ML prediction results."""
    print("="*50)
    print(txn_data)
    print("="*50)

    txn = Transaction(**txn_data)

    db.add(txn)
    db.commit()
    db.refresh(txn)
    
    return txn


def get_transaction(db: Session, transaction_id: str) -> Optional[Transaction]:
    return db.query(Transaction).filter(Transaction.transaction_id == transaction_id).first()


def get_recent_transactions(db: Session, limit: int = 50) -> List[Transaction]:
    return db.query(Transaction).order_by(desc(Transaction.created_at)).limit(limit).all()


def get_user_transactions(db: Session, user_id: int, limit: int = 20) -> List[Transaction]:
    return (
        db.query(Transaction)
        .filter(Transaction.user_id == user_id)
        .order_by(desc(Transaction.created_at))
        .limit(limit)
        .all()
    )


def get_fraud_transactions(db: Session, limit: int = 100) -> List[Transaction]:
    return (
        db.query(Transaction)
        .filter(Transaction.prediction == "Fraud")
        .order_by(desc(Transaction.created_at))
        .limit(limit)
        .all()
    )


def get_transactions_by_risk(db: Session, risk_level: str, limit: int = 50) -> List[Transaction]:
    return (
        db.query(Transaction)
        .filter(Transaction.risk_level == risk_level)
        .order_by(desc(Transaction.created_at))
        .limit(limit)
        .all()
    )


def get_dashboard_stats(db: Session) -> Dict[str, Any]:
    """Return aggregated statistics for the dashboard overview."""
    total = db.query(func.count(Transaction.transaction_id)).scalar() or 0
    fraud_count = db.query(func.count(Transaction.transaction_id)).filter(Transaction.prediction == "Fraud").scalar() or 0
    total_volume = db.query(func.sum(Transaction.amount)).scalar() or 0.0
    avg_amount = db.query(func.avg(Transaction.amount)).scalar() or 0.0

    last_24h = datetime.utcnow() - timedelta(hours=24)
    recent_count = db.query(func.count(Transaction.transaction_id)).filter(Transaction.created_at >= last_24h).scalar() or 0
    recent_fraud = (
        db.query(func.count(Transaction.transaction_id))
        .filter(Transaction.created_at >= last_24h, Transaction.prediction == "Fraud")
        .scalar() or 0
    )

    return {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "fraud_rate": round((fraud_count / total * 100) if total > 0 else 0.0, 2),
        "total_volume": round(float(total_volume), 2),
        "avg_transaction_amount": round(float(avg_amount), 2),
        "last_24h_transactions": recent_count,
        "last_24h_fraud": recent_fraud,
    }


def get_hourly_fraud_stats(db: Session, days: int = 7) -> List[Dict]:
    """Return fraud counts grouped by hour of day for the last N days."""
    since = datetime.utcnow() - timedelta(days=days)
    results = (
        db.query(
            func.extract("hour", Transaction.created_at).label("hour"),
            func.count(Transaction.transaction_id).label("count"),
            func.sum(func.cast(Transaction.prediction == "Fraud", type_=func.Integer)).label("fraud_count"),
        )
        .filter(Transaction.created_at >= since)
        .group_by(func.extract("hour", Transaction.created_at))
        .order_by("hour")
        .all()
    )
    return [{"hour": int(r.hour), "count": r.count, "fraud_count": r.fraud_count or 0} for r in results]


# ─── ALERT OPERATIONS ────────────────────────────────────────────────────────

def create_alert(db: Session, transaction_id: str, risk_level: str, fraud_probability: float, message: str) -> Alert:
    existing = db.query(Alert).filter(Alert.transaction_id == transaction_id).first()
    if existing:
        return existing

    alert = Alert(
        transaction_id=transaction_id,
        risk_level=risk_level,
        fraud_probability=fraud_probability,
        alert_message=message,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    logger.info(f"Alert created: {alert.alert_id} | {risk_level} | txn={transaction_id}")
    return alert


def get_recent_alerts(db: Session, limit: int = 50, resolved: Optional[bool] = None) -> List[Alert]:
    query = db.query(Alert)
    if resolved is not None:
        query = query.filter(Alert.is_resolved == resolved)
    return query.order_by(desc(Alert.created_at)).limit(limit).all()


def resolve_alert(db: Session, alert_id: int, resolved_by: str) -> Optional[Alert]:
    alert = db.query(Alert).filter(Alert.alert_id == alert_id).first()
    if alert:
        alert.is_resolved = True
        alert.resolved_at = datetime.utcnow()
        alert.resolved_by = resolved_by
        db.commit()
        db.refresh(alert)
    return alert


def get_alert_counts_by_risk(db: Session) -> Dict[str, int]:
    results = (
        db.query(Alert.risk_level, func.count(Alert.alert_id).label("count"))
        .group_by(Alert.risk_level)
        .all()
    )
    return {r.risk_level: r.count for r in results}