"""
Model Evaluation and Reporting Utilities
Generates detailed evaluation reports and visualizations.
"""

import json
import logging
import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    roc_curve, precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and reporting."""

    def __init__(self, model_dir: str = "../models", output_dir: str = "../reports"):
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_model_artifacts(self) -> tuple:
        """Load model and preprocessor artifacts."""
        model = joblib.load(os.path.join(self.model_dir, "fraud_model.joblib"))
        scaler = joblib.load(os.path.join(self.model_dir, "scaler.joblib"))
        feature_columns = joblib.load(os.path.join(self.model_dir, "feature_columns.joblib"))
        return model, scaler, feature_columns

    def plot_roc_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, save: bool = True
    ) -> Dict[str, float]:
        """Plot and save ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_score:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if save:
            path = os.path.join(self.output_dir, "roc_curve.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"ROC curve saved to {path}")

        return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc_score}

    def plot_precision_recall_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, save: bool = True
    ) -> Dict[str, float]:
        """Plot and save Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        ap_score = average_precision_score(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="blue", lw=2, label=f"PR Curve (AP = {ap_score:.3f})")
        plt.axhline(y=y_true.mean(), color="red", linestyle="--", label=f"Baseline (fraud rate = {y_true.mean():.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save:
            path = os.path.join(self.output_dir, "pr_curve.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"PR curve saved to {path}")

        return {"precision": precision.tolist(), "recall": recall.tolist(), "average_precision": ap_score}

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, save: bool = True
    ) -> None:
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraud"])

        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Confusion Matrix")

        if save:
            path = os.path.join(self.output_dir, "confusion_matrix.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Confusion matrix saved to {path}")

    def plot_feature_importance(
        self, feature_importances: Dict[str, float], top_n: int = 15, save: bool = True
    ) -> None:
        """Plot top N feature importances."""
        top_features = dict(list(feature_importances.items())[:top_n])
        features = list(reversed(list(top_features.keys())))
        importances = list(reversed(list(top_features.values())))

        plt.figure(figsize=(10, 8))
        bars = plt.barh(features, importances, color="steelblue", edgecolor="white")
        plt.xlabel("Feature Importance Score")
        plt.title(f"Top {top_n} Feature Importances (XGBoost)")
        plt.grid(True, axis="x", alpha=0.3)

        for bar, val in zip(bars, importances):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", ha="left", fontsize=8)

        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "feature_importance.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Feature importance plot saved to {path}")

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """Find the optimal decision threshold maximizing F1 score."""
        thresholds = np.arange(0.1, 0.95, 0.01)
        best_threshold = 0.5
        best_f1 = 0.0

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        logger.info(f"Optimal threshold: {best_threshold:.2f} → F1: {best_f1:.4f}")
        return {"optimal_threshold": float(best_threshold), "best_f1": float(best_f1)}

    def generate_evaluation_report(
        self, metrics: Dict[str, Any], cv_metrics: Optional[Dict] = None
    ) -> str:
        """Generate a human-readable evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("FRAUD DETECTION MODEL - EVALUATION REPORT")
        report.append("=" * 60)

        tm = metrics.get("test_metrics", {})
        report.append("\n📊 Test Set Performance:")
        report.append(f"  Accuracy  : {tm.get('accuracy', 0):.4f}")
        report.append(f"  Precision : {tm.get('precision', 0):.4f}")
        report.append(f"  Recall    : {tm.get('recall', 0):.4f}")
        report.append(f"  F1 Score  : {tm.get('f1_score', 0):.4f}")
        report.append(f"  ROC-AUC   : {tm.get('roc_auc', 0):.4f}")

        if cv_metrics:
            report.append("\n🔄 Cross-Validation Results (5-Fold):")
            for metric, values in cv_metrics.items():
                report.append(f"  {metric:12s}: {values['mean']:.4f} ± {values['std']:.4f}")

        report_str = "\n".join(report)
        report_path = os.path.join(self.output_dir, "evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write(report_str)

        logger.info(f"Evaluation report saved to {report_path}")
        return report_str


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    metrics_path = "../models/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        report = evaluator.generate_evaluation_report(metrics, metrics.get("cv_metrics"))
        print(report)
        if "feature_importances" in metrics:
            evaluator.plot_feature_importance(metrics["feature_importances"])
    else:
        print("No metrics.json found. Run training first.")