"""
Model Evaluation Module
Computes performance metrics, confusion matrices, ROC curves,
and cross-model comparison tables for Twitter sentiment analysis.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes standard evaluation metrics for classification.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro": round(rec_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "classification_report": report_dict,
        "labels": sorted(list(set(y_true)))
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    title: str = "Confusion Matrix",
    cmap: str = "Blues"
) -> plt.Figure:
    """
    Renders an aesthetically styled Seaborn confusion matrix figure.
    """
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        linewidths=1.0,
        linecolor="#f0f0f0",
        ax=ax
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Sentiment", fontsize=10, labelpad=8)
    ax.set_ylabel("True Sentiment", fontsize=10, labelpad=8)
    plt.tight_layout()
    return fig


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: List[str],
    model_name: str = "Model"
) -> plt.Figure:
    """
    Plots multi-class One-vs-Rest ROC curves with calculated AUC scores.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    if n_classes == 2:
        # Binary case
        fpr, tpr, _ = roc_curve(y_true_bin, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            color=colors[0],
            lw=2,
            label=f"{classes[1]} (AUC = {roc_auc:.3f})"
        )
    else:
        # Multiclass case
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            color = colors[i % len(colors)]
            ax.plot(
                fpr, tpr,
                color=color,
                lw=2,
                label=f"{classes[i]} (AUC = {roc_auc:.3f})"
            )

    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(f"ROC Curves - {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def generate_comparison_table(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Transforms a list of model metric dictionaries into a clean summary DataFrame.
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": m["model"],
            "Accuracy": m["accuracy"],
            "Precision (Macro)": m["precision_macro"],
            "Recall (Macro)": m["recall_macro"],
            "F1-Score (Macro)": m["f1_macro"],
            "F1-Score (Weighted)": m["f1_weighted"]
        })
    df = pd.DataFrame(rows)
    df.sort_values(by="F1-Score (Macro)", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
