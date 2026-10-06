import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.config import get_path

logger = get_logger("model_evaluation")

def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """
    Calculates comprehensive classification evaluation metrics for predictive maintenance.
    
    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probability scores for positive class (1).
        
    Returns:
        Dict containing accuracy, precision, recall, f1, roc_auc, pr_auc, confusion matrix counts.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    roc_auc = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else 0.0
    pr_auc = float(average_precision_score(y_true, y_prob)) if y_prob is not None else 0.0
    
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }
    
    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, save_path: Path = None) -> Path:
    """
    Renders and saves confusion matrix heatmap figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Normal (0)", "Failure (1)"],
                yticklabels=["Normal (0)", "Failure (1)"])
    plt.title(f"Confusion Matrix: {model_name}", fontweight="bold", fontsize=12)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    
    if save_path is None:
        figures_dir = get_path("reports/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_path = figures_dir / f"cm_{model_name.lower().replace(' ', '_')}.png"
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot: {save_path}")
    return save_path

def plot_model_comparison_curves(models_dict: dict, X_eval: np.ndarray, y_eval: np.ndarray, save_path: Path = None) -> Path:
    """
    Plots overlay ROC and Precision-Recall curves comparing all candidate models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_eval)[:, 1]
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_eval, y_prob)
            roc_auc = roc_auc_score(y_eval, y_prob)
            ax1.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
            
            # PR Curve
            precision, recall, _ = precision_recall_curve(y_eval, y_prob)
            pr_auc = average_precision_score(y_eval, y_prob)
            ax2.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.3f})")
            
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    ax1.set_title("ROC Curves Comparison", fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Precision-Recall Curves Comparison", fontweight="bold")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    if save_path is None:
        figures_dir = get_path("reports/figures")
        figures_dir.mkdir(parents=True, exist_ok=True)
        save_path = figures_dir / "model_comparison_curves.png"
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved model comparison curves: {save_path}")
    return save_path
