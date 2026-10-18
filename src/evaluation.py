"""Shared evaluation helpers: held-out metrics plus confusion-matrix / ROC /
classification-report artifacts used by both train.py and tune.py."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_test_metrics(y_true, pred, proba) -> dict:
    return {
        "test_accuracy": float(accuracy_score(y_true, pred)),
        "test_precision": float(precision_score(y_true, pred, zero_division=0)),
        "test_recall": float(recall_score(y_true, pred, zero_division=0)),
        "test_f1": float(f1_score(y_true, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_true, proba)),
    }


def save_eval_artifacts(prefix, y_true, pred, proba, plot_dir, report_dir) -> list:
    """Write <prefix>_confusion_matrix.png, <prefix>_roc_curve.png and
    <prefix>_classification_report.txt. Returns the three paths."""
    plot_dir, report_dir = Path(plot_dir), Path(report_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, pred, ax=ax)
    ax.set_title(f"{prefix}: confusion matrix (test)")
    fig.tight_layout()
    cm_path = plot_dir / f"{prefix}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax)
    ax.set_title(f"{prefix}: ROC curve (test)")
    fig.tight_layout()
    roc_path = plot_dir / f"{prefix}_roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    report_path = report_dir / f"{prefix}_classification_report.txt"
    report_path.write_text(classification_report(y_true, pred, zero_division=0))
    return [cm_path, roc_path, report_path]
