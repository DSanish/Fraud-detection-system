"""
XGBoost Fraud Detection Model Training
Handles class imbalance, hyperparameter tuning, and cross-validation.
"""

import json
import logging
import os
from typing import Dict, Any

import numpy as np
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

from preprocess import FraudDataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = "models"
DATA_PATH = "data/transactions.csv"

def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    """Compute scale_pos_weight for XGBoost class imbalance handling."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    ratio = n_neg / n_pos
    logger.info(f"Class ratio (neg/pos): {ratio:.2f}")
    return float(ratio)


def build_model(scale_pos_weight: float) -> XGBClassifier:
    """Build XGBoost model with tuned hyperparameters."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampling_strategy: float = 0.2,
) -> tuple:
    """Apply SMOTE to handle class imbalance in training data."""
    logger.info("Applying SMOTE for class balancing...")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE → samples: {len(X_resampled):,}, fraud rate: {y_resampled.mean()*100:.2f}%")
    return X_resampled, y_resampled


def cross_validate_model(
    model: XGBClassifier,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Run stratified k-fold cross-validation."""
    logger.info(f"Running {n_splits}-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    metrics = {}
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        metrics[metric] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }
        logger.info(f"  CV {metric}: {scores.mean():.4f} ± {scores.std():.4f}")

    return metrics


def evaluate_model(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate model on test set and return all metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "threshold": threshold,
        "n_test_samples": int(len(y_test)),
        "n_fraud_detected": int(y_pred.sum()),
        "n_actual_fraud": int(y_test.sum()),
    }

    logger.info("\n" + "=" * 50)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:30s}: {v:.4f}")
        else:
            logger.info(f"  {k:30s}: {v}")

    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion Matrix:\n{cm}")
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def get_feature_importances(model: XGBClassifier, feature_columns: list) -> Dict[str, float]:
    """Extract and return feature importance scores."""
    importances = model.feature_importances_
    importance_dict = dict(zip(feature_columns, importances.tolist()))
    sorted_importances = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    logger.info("\nTop 10 Feature Importances:")
    for feat, score in list(sorted_importances.items())[:10]:
        logger.info(f"  {feat:40s}: {score:.4f}")

    return sorted_importances


def train(
    data_path: str = DATA_PATH,
    model_dir: str = MODEL_DIR,
    use_smote: bool = True,
    run_cv: bool = True,
) -> Dict[str, Any]:
    """Full training pipeline."""
    os.makedirs(model_dir, exist_ok=True)
    logger.info("=" * 60)
    logger.info("FRAUD DETECTION MODEL TRAINING")
    logger.info("=" * 60)

    # 1. Preprocess
    preprocessor = FraudDataPreprocessor(model_dir=model_dir)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(data_path)

    # 2. Handle class imbalance
    if use_smote:
        X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train

    scale_pos_weight = compute_scale_pos_weight(y_train_balanced)

    # 3. Build model
    model = build_model(scale_pos_weight)

    # 4. Cross-validation (on original training data)
    cv_metrics = {}
    if run_cv:
        cv_metrics = cross_validate_model(model, X_train, y_train)

    # 5. Train on full balanced training set
    logger.info("\nTraining final model...")
    eval_set = [(X_test, y_test)]
    model.fit(
        X_train_balanced,
        y_train_balanced,
        eval_set=eval_set,
        verbose=50,
    )
    logger.info("Training complete.")

    # 6. Evaluate
    test_metrics = evaluate_model(model, X_test, y_test)

    # 7. Feature importances
    feature_importances = get_feature_importances(model, preprocessor.feature_columns)

    # 8. Save model
    model_path = os.path.join(model_dir, "fraud_model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"\nModel saved to {model_path}")

    # 9. Save metrics
    results = {
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
        "feature_importances": feature_importances,
        "feature_columns": preprocessor.feature_columns,
        "model_path": model_path,
    }
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")

    return results


if __name__ == "__main__":
    results = train()
    print("\nTraining completed successfully!")
    print(f"ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")
    print(f"F1 Score: {results['test_metrics']['f1_score']:.4f}")
    print(f"Precision: {results['test_metrics']['precision']:.4f}")
    print(f"Recall: {results['test_metrics']['recall']:.4f}")