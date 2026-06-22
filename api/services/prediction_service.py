"""
Fraud Prediction Service
Loads model artifacts and provides real-time inference.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "models")


class FraudPredictionService:
    """Singleton service for loading and serving the fraud detection model."""

    _instance: Optional["FraudPredictionService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self, model_dir: str = MODEL_DIR) -> None:
        """Load all model artifacts from disk."""
        if self._loaded:
            return

        try:
            self.model = joblib.load(os.path.join(model_dir, "fraud_model.joblib"))
            self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
            self.label_encoders = joblib.load(os.path.join(model_dir, "label_encoders.joblib"))
            self.feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.joblib"))

            metrics_path = os.path.join(model_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    self.metrics = json.load(f)
            else:
                self.metrics = {}

            self._loaded = True
            logger.info(f"Model loaded. Features: {len(self.feature_columns)} | Dir: {model_dir}")

        except FileNotFoundError as e:
            logger.warning(f"Model artifacts not found ({e}). Using demo mode.")
            self._loaded = False
            self.model = None
            self.scaler = None
            self.label_encoders = {}
            self.feature_columns = []
            self.metrics = {}

    def _build_feature_vector(self, data: Dict[str, Any]) -> np.ndarray:
        """Convert raw transaction dict to scaled feature vector."""
        import pandas as pd

        df = pd.DataFrame([data])

        # Derived features
        df["amount_to_avg_ratio"] = df["transaction_amount"] / (df.get("avg_transaction_amount", pd.Series([1200])) + 1e-6)
        df["log_transaction_amount"] = np.log1p(df["transaction_amount"])
        df["log_avg_transaction_amount"] = np.log1p(df.get("avg_transaction_amount", pd.Series([1200])))
        df["composite_risk_score"] = (
            df.get("unusual_time_flag", pd.Series([0])) * 0.3
            + df.get("unusual_location_flag", pd.Series([0])) * 0.3
            + df.get("device_change_flag", pd.Series([0])) * 0.2
            + (df["amount_to_avg_ratio"] > 5).astype(int) * 0.2
        )
        df["high_velocity_flag"] = (df.get("daily_transaction_count", pd.Series([1])) > 10).astype(int)
        df["transaction_velocity"] = df.get("daily_transaction_count", pd.Series([1]))

        # Encode categoricals
        for col, le in self.label_encoders.items():
            encoded_col = f"{col}_encoded"
            df[encoded_col] = df[col].astype(str).apply(
                lambda x: int(le.transform([x])[0]) if x in le.classes_ else -1
            ) if col in df.columns else -1

        # Fill missing with 0
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        X = df[self.feature_columns].values
        return self.scaler.transform(X)

    def _rule_based_prediction(self, data: Dict[str, Any]) -> Tuple[float, str]:
        """Fallback rule-based prediction when model is unavailable."""
        amount = data.get("transaction_amount", 0)
        avg = data.get("avg_transaction_amount", 1200)
        device_change = data.get("device_change_flag", 0)
        unusual_time = data.get("unusual_time_flag", 0)
        unusual_location = data.get("unusual_location_flag", 0)
        velocity = data.get("daily_transaction_count", 1)

        score = 0.05
        if avg > 0:
            ratio = amount / avg
            if ratio > 20:
                score += 0.55
            elif ratio > 10:
                score += 0.35
            elif ratio > 5:
                score += 0.20
        if device_change:
            score += 0.15
        if unusual_time:
            score += 0.15
        if unusual_location:
            score += 0.20
        if velocity > 15:
            score += 0.20
        elif velocity > 10:
            score += 0.10

        score = min(score, 0.99)
        prediction = "Fraud" if score >= 0.5 else "Legitimate"
        return score, prediction

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run fraud prediction on a single transaction.

        Returns dict with fraud_probability, prediction, risk_level.
        """
        start = time.time()
        transaction_id = data.get("transaction_id") or f"TXN_{uuid.uuid4().hex[:12].upper()}"

        if self._loaded and self.model is not None:
            try:
                X = self._build_feature_vector(data)
                fraud_prob = float(self.model.predict_proba(X)[0, 1])
                prediction = "Fraud" if fraud_prob >= 0.5 else "Legitimate"
            except Exception as e:
                logger.error(f"Model inference failed: {e}. Falling back to rules.")
                fraud_prob, prediction = self._rule_based_prediction(data)
        else:
            fraud_prob, prediction = self._rule_based_prediction(data)

        # Risk classification
        if fraud_prob >= 0.85:
            risk_level = "High"
        elif fraud_prob >= 0.70:
            risk_level = "Medium"
        elif fraud_prob >= 0.50:
            risk_level = "Low"
        else:
            risk_level = None

        elapsed_ms = (time.time() - start) * 1000

        return {
            "transaction_id": transaction_id,
            "fraud_probability": round(fraud_prob, 4),
            "prediction": prediction,
            "risk_level": risk_level,
            "processing_time_ms": round(elapsed_ms, 2),
            "timestamp": datetime.utcnow(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Return stored model performance metrics."""
        if not self.metrics:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "roc_auc": 0.0,
                "model_version": "demo",
                "last_trained": None,
                "feature_count": len(self.feature_columns),
            }
        tm = self.metrics.get("test_metrics", {})
        return {
            "accuracy": tm.get("accuracy", 0.0),
            "precision": tm.get("precision", 0.0),
            "recall": tm.get("recall", 0.0),
            "f1_score": tm.get("f1_score", 0.0),
            "roc_auc": tm.get("roc_auc", 0.0),
            "model_version": "xgboost-v1",
            "last_trained": None,
            "feature_count": len(self.feature_columns),
        }

    @property
    def is_ready(self) -> bool:
        return self._loaded


# Module-level singleton
prediction_service = FraudPredictionService()