"""
Data Preprocessing Pipeline for Fraud Detection System
Handles data cleaning, feature engineering, and scaling.
"""

import logging
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

logger = logging.getLogger(__name__)

CATEGORICAL_FEATURES = ["transaction_type", "merchant_category"]
NUMERIC_FEATURES = [
    "transaction_amount",
    "avg_transaction_amount",
    "daily_transaction_count",
    "weekly_transaction_count",
    "monthly_transaction_count",
    "amount_deviation_from_average",
    "transaction_velocity",
    "transaction_hour",
]
BINARY_FEATURES = ["unusual_time_flag", "unusual_location_flag", "device_change_flag"]
TARGET = "fraud_label"

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + [f"{c}_encoded" for c in CATEGORICAL_FEATURES]


class FraudDataPreprocessor:
    """Complete preprocessing pipeline for fraud detection."""

    def __init__(self, model_dir: str = "models/"):
        self.model_dir = model_dir
        self.scaler = StandardScaler()
        self.label_encoders: dict = {}
        self.feature_columns: List[str] = []
        self.is_fitted = False

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and perform initial validation of the dataset."""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df):,} records with {df.shape[1]} columns")
        logger.info(f"Fraud rate: {df[TARGET].mean()*100:.2f}%")
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates and handle basic data quality issues."""
        logger.info("Cleaning data...")
        initial_size = len(df)

        # Drop duplicates
        df = df.drop_duplicates(subset=["transaction_id"], keep="first")
        logger.info(f"Removed {initial_size - len(df)} duplicate transactions")

        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col not in ["transaction_id", "device_id", "location", "transaction_time"]:
                df[col] = df[col].fillna(df[col].mode()[0])

        logger.info(f"After cleaning: {len(df):,} records")
        return df

    def remove_outliers(self, df: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
        """Remove extreme outliers using IQR method on non-fraud transactions."""
        logger.info("Detecting and handling outliers...")
        normal_df = df[df[TARGET] == 0].copy()

        for col in ["transaction_amount", "daily_transaction_count"]:
            Q1 = normal_df[col].quantile(0.25)
            Q3 = normal_df[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 3 * IQR
            # Cap outliers rather than remove (preserve fraud signals)
            df[col] = df[col].clip(upper=upper_bound * 10)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional derived features."""
        logger.info("Engineering features...")

        # Amount ratio vs user average
        df["amount_to_avg_ratio"] = df["transaction_amount"] / (df["avg_transaction_amount"] + 1e-6)

        # Log-transform skewed features
        df["log_transaction_amount"] = np.log1p(df["transaction_amount"])
        df["log_avg_transaction_amount"] = np.log1p(df["avg_transaction_amount"])

        # Time-based features
        if "transaction_time" in df.columns:
            df["transaction_time"] = pd.to_datetime(df["transaction_time"])
            df["transaction_day_of_week"] = df["transaction_time"].dt.dayofweek
            df["transaction_is_weekend"] = (df["transaction_day_of_week"] >= 5).astype(int)
            df["transaction_month"] = df["transaction_time"].dt.month

        # Risk score combination
        df["composite_risk_score"] = (
            df["unusual_time_flag"] * 0.3
            + df["unusual_location_flag"] * 0.3
            + df["device_change_flag"] * 0.2
            + (df["amount_to_avg_ratio"] > 5).astype(int) * 0.2
        )

        # High velocity flag
        df["high_velocity_flag"] = (df["daily_transaction_count"] > 10).astype(int)

        logger.info(f"Feature engineering complete. Columns: {df.shape[1]}")
        return df

    def encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label-encode categorical features."""
        for col in CATEGORICAL_FEATURES:
            encoded_col = f"{col}_encoded"
            if fit:
                le = LabelEncoder()
                df[encoded_col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                df[encoded_col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Return all feature columns available after engineering."""
        engineered_extra = [
            "amount_to_avg_ratio", "log_transaction_amount", "log_avg_transaction_amount",
            "composite_risk_score", "high_velocity_flag",
        ]
        time_features = [
            "transaction_day_of_week", "transaction_is_weekend", "transaction_month"
        ]
        cols = NUMERIC_FEATURES + BINARY_FEATURES
        cols += [f"{c}_encoded" for c in CATEGORICAL_FEATURES]
        cols += [c for c in engineered_extra if c in df.columns]
        cols += [c for c in time_features if c in df.columns]
        return [c for c in cols if c in df.columns]

    def scale_features(
        self, X: pd.DataFrame, fit: bool = True
    ) -> np.ndarray:
        """Apply StandardScaler to feature matrix."""
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def prepare_data(
        self,
        filepath: str,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Full pipeline: load → clean → engineer → encode → split → scale."""
        df = self.load_data(filepath)
        df = self.clean_data(df)
        df = self.remove_outliers(df)
        df = self.engineer_features(df)
        df = self.encode_categoricals(df, fit=True)

        self.feature_columns = self.get_feature_columns(df)
        logger.info(f"Using {len(self.feature_columns)} features: {self.feature_columns}")

        X = df[self.feature_columns]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        X_train_scaled = self.scale_features(X_train, fit=True)
        X_test_scaled = self.scale_features(X_test, fit=False)

        self.is_fitted = True
        self.save_artifacts()

        logger.info(f"Train size: {len(X_train):,} | Test size: {len(X_test):,}")
        logger.info(f"Train fraud rate: {y_train.mean()*100:.2f}% | Test fraud rate: {y_test.mean()*100:.2f}%")

        return X_train_scaled, X_test_scaled, y_train.values, y_test.values

    def preprocess_single(self, data: dict) -> np.ndarray:
        """Preprocess a single transaction for real-time prediction."""
        if not self.is_fitted:
            self.load_artifacts()

        df = pd.DataFrame([data])

        # Derive computed features
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

        # Encode categoricals
        for col in CATEGORICAL_FEATURES:
            encoded_col = f"{col}_encoded"
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                df[encoded_col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                df[encoded_col] = -1

        # Fill missing columns with 0
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0

        X = df[self.feature_columns]
        return self.scaler.transform(X)

    def save_artifacts(self) -> None:
        """Persist preprocessor artifacts."""
        import os
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.scaler, f"{self.model_dir}/scaler.joblib")
        joblib.dump(self.label_encoders, f"{self.model_dir}/label_encoders.joblib")
        joblib.dump(self.feature_columns, f"{self.model_dir}/feature_columns.joblib")
        logger.info(f"Preprocessor artifacts saved to {self.model_dir}")

    def load_artifacts(self) -> None:
        """Load persisted preprocessor artifacts."""
        self.scaler = joblib.load(f"{self.model_dir}/scaler.joblib")
        self.label_encoders = joblib.load(f"{self.model_dir}/label_encoders.joblib")
        self.feature_columns = joblib.load(f"{self.model_dir}/feature_columns.joblib")
        self.is_fitted = True
        logger.info("Preprocessor artifacts loaded.")