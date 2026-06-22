"""
Fraud Alert Engine
Generates, classifies, and persists fraud alerts based on ML predictions.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Alert thresholds
ALERT_THRESHOLDS = {
    "High": 0.85,
    "Medium": 0.70,
    "Low": 0.50,
}


@dataclass
class AlertResult:
    """Result of alert engine evaluation."""
    should_alert: bool
    risk_level: Optional[str]
    fraud_probability: float
    message: str
    timestamp: datetime


class FraudAlertEngine:
    """
    Evaluates fraud predictions and generates structured alerts.
    Risk levels:
      High   → probability >= 0.85
      Medium → probability >= 0.70
      Low    → probability >= 0.50
    """

    def __init__(self, alert_threshold: float = 0.50):
        self.alert_threshold = alert_threshold

    def classify_risk(self, fraud_probability: float) -> Tuple[bool, Optional[str]]:
        """Determine if an alert should fire and its risk level."""
        if fraud_probability >= ALERT_THRESHOLDS["High"]:
            return True, "High"
        elif fraud_probability >= ALERT_THRESHOLDS["Medium"]:
            return True, "Medium"
        elif fraud_probability >= ALERT_THRESHOLDS["Low"]:
            return True, "Low"
        return False, None

    def build_alert_message(
        self,
        transaction_data: Dict[str, Any],
        fraud_probability: float,
        risk_level: str,
    ) -> str:
        """Build a human-readable alert message."""
        amount = transaction_data.get("transaction_amount", "N/A")
        user_id = transaction_data.get("user_id", "Unknown")
        avg_amount = transaction_data.get("avg_transaction_amount", "N/A")
        device_change = transaction_data.get("device_change_flag", 0)
        unusual_time = transaction_data.get("unusual_time_flag", 0)
        unusual_location = transaction_data.get("unusual_location_flag", 0)

        reasons = []
        if isinstance(amount, (int, float)) and isinstance(avg_amount, (int, float)):
            if avg_amount > 0 and amount > avg_amount * 5:
                reasons.append(f"transaction amount ₹{amount:,.0f} is {amount/avg_amount:.1f}x above user average ₹{avg_amount:,.0f}")
        if device_change:
            reasons.append("transaction from a new/unrecognized device")
        if unusual_time:
            reasons.append("transaction occurred at an unusual hour (12 AM–5 AM)")
        if unusual_location:
            reasons.append("transaction from an unusual or foreign location")

        reason_str = "; ".join(reasons) if reasons else "multiple risk factors detected"

        return (
            f"[{risk_level.upper()} RISK ALERT] Suspicious transaction detected for User {user_id}. "
            f"Fraud probability: {fraud_probability*100:.1f}%. "
            f"Reason(s): {reason_str}. "
            f"Immediate review recommended."
        )

    def evaluate(
        self,
        transaction_data: Dict[str, Any],
        fraud_probability: float,
        prediction: str,
    ) -> AlertResult:
        """
        Evaluate a transaction and return an AlertResult.

        Args:
            transaction_data: Raw transaction feature dict.
            fraud_probability: Model output probability (0–1).
            prediction: "Fraud" or "Legitimate".

        Returns:
            AlertResult with alert decision and metadata.
        """
        should_alert, risk_level = self.classify_risk(fraud_probability)

        if should_alert and risk_level:
            message = self.build_alert_message(transaction_data, fraud_probability, risk_level)
            logger.info(f"Alert triggered: {risk_level} | prob={fraud_probability:.3f} | {message[:80]}...")
        else:
            message = f"No alert. Fraud probability {fraud_probability*100:.1f}% below threshold."
            risk_level = None

        return AlertResult(
            should_alert=should_alert,
            risk_level=risk_level,
            fraud_probability=fraud_probability,
            message=message,
            timestamp=datetime.utcnow(),
        )

    def process_and_store(
        self,
        db,
        transaction_id: str,
        transaction_data: Dict[str, Any],
        fraud_probability: float,
        prediction: str,
    ) -> Optional[AlertResult]:
        """
        Evaluate a transaction and persist alert to database if triggered.

        Args:
            db: SQLAlchemy database session.
            transaction_id: Unique transaction ID.
            transaction_data: Feature dict.
            fraud_probability: ML model output.
            prediction: "Fraud" or "Legitimate".

        Returns:
            AlertResult if alert was created, else None.
        """
        from database.crud import create_alert

        result = self.evaluate(transaction_data, fraud_probability, prediction)

        if result.should_alert and result.risk_level:
            try:
                create_alert(
                    db=db,
                    transaction_id=transaction_id,
                    risk_level=result.risk_level,
                    fraud_probability=fraud_probability,
                    message=result.message,
                )
            except Exception as e:
                logger.error(f"Failed to store alert for transaction {transaction_id}: {e}")

        return result if result.should_alert else None


# Singleton instance for API reuse
alert_engine = FraudAlertEngine()