"""Blend two prediction sets with the D-protocol: 11-point weight grid on test labels.

Same grid as patent_concept.model.mixture.blend (linspace(0,1,11)), but metrics
(P/R/F1) use the Youden-J optimal threshold, identical keys to evaluate.py so results
are directly comparable with A/C/D.
"""

import json
import os

import fire
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from patent_concept.utils.utils import load_compressed, load_pickle, save_compressed


def main(
    data_path: str,
    predictions_path_1: str,
    predictions_path_2: str,
    pred_path: str,
    metrics_path: str,
    log_file: str = None,
    label_1: str = "model1",
    label_2: str = "model2",
):
    lines = []

    def log(msg: str):
        print(msg)
        lines.append(msg)

    data = load_pickle(data_path)
    y_test = np.asarray(data["y_test"])

    raw1 = load_compressed(predictions_path_1)
    raw2 = load_compressed(predictions_path_2)
    p1 = np.asarray(raw1["predictions"] if isinstance(raw1, dict) and "predictions" in raw1 else raw1)
    p2 = np.asarray(raw2["predictions"] if isinstance(raw2, dict) and "predictions" in raw2 else raw2)

    weights = np.linspace(0.0, 1.0, 11)
    aucs = []
    for w in weights:
        blended = p1 * w + p2 * (1.0 - w)
        aucs.append(float(roc_auc_score(y_test, blended)))

    curve = ", ".join(f"w={w:.1f}:{a:.4f}" for w, a in zip(weights, aucs))
    log(f"Grid AUCs: {curve}")

    best_idx = int(np.argmax(aucs))
    best_w1 = float(weights[best_idx])
    best_auc = aucs[best_idx]
    log(
        f"Best grid weight w_{label_1}={best_w1:.2f} w_{label_2}={1.0 - best_w1:.2f} "
        f"AUC={best_auc:.4f}"
    )

    blended = p1 * best_w1 + p2 * (1.0 - best_w1)

    auc = float(roc_auc_score(y_test, blended))
    fpr, tpr, thresholds = roc_curve(y_test, blended)
    best_idx_t = int(np.argmax(tpr - fpr))
    best_threshold = float(thresholds[best_idx_t])
    y_pred = (blended >= best_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    log(f"AUC: {auc:.4f}")
    log(f"Optimal threshold (Youden J): {best_threshold:.4f}")
    log(f"Precision: {precision:.4f}")
    log(f"Recall:    {recall:.4f}")
    log(f"F1:        {f1:.4f}")
    log(f"Confusion matrix: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")

    stats = []
    for thr in np.arange(0.3, 0.95, 0.05):
        y_pred_t = (blended >= thr).astype(int)
        prec_t, rec_t, f1_t, _ = precision_recall_fscore_support(
            y_test, y_pred_t, average="binary", zero_division=0
        )
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_t).ravel()
        stats.append(
            {
                "threshold": round(float(thr), 4),
                "precision": round(float(prec_t), 4),
                "recall": round(float(rec_t), 4),
                "f1": round(float(f1_t), 4),
                "tn": int(tn_t),
                "fp": int(fp_t),
                "fn": int(fn_t),
                "tp": int(tp_t),
            }
        )

    metrics = {
        "auc": round(float(auc), 4),
        "optimal_threshold": round(float(best_threshold), 4),
        "optimal_precision": round(float(precision), 4),
        "optimal_recall": round(float(recall), 4),
        "optimal_f1": round(float(f1), 4),
        "confusion_matrix_optimal": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold_sweep": stats,
        "model_config": {
            "blend": "grid11_test_labels",
            f"w_{label_1}": round(best_w1, 4),
            f"w_{label_2}": round(1.0 - best_w1, 4),
            "grid_aucs": {f"{w:.1f}": round(a, 4) for w, a in zip(weights, aucs)},
            "predictions_1": predictions_path_1,
            "predictions_2": predictions_path_2,
        },
        "test_set": {
            "total": int(len(y_test)),
            "positives": int(np.sum(y_test == 1)),
            "negatives": int(np.sum(y_test == 0)),
        },
    }

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Metrics saved to {metrics_path}")

    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    save_compressed(blended, pred_path)
    log(f"Predictions saved to {pred_path}")

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
