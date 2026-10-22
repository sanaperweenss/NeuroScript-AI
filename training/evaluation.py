"""
NeuroScript AI - Model Evaluation and Comparative Metrics
Calculates comprehensive, mathematically sound metrics on held-out test splits:
- Accuracy, Macro/Weighted Precision, Recall, F1 Score
- Detailed per-class breakdown
- Confusion matrix array
- Measured average inference latency (milliseconds per sample)
"""

import time
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import tensorflow as tf
from data.labels import LabelMapper
from config.settings import ARCH_ANN, ARCH_CNN


def evaluate_model_on_test_set(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    task: str,
    architecture: str,
    batch_size: int = 128
) -> Dict[str, Any]:
    """
    Run evaluation on the held-out test set and calculate full statistical metrics.
    """
    mapper = LabelMapper(task)
    classes = mapper.get_classes()

    # Preprocess test images for architecture
    # x_test is uint8 (N, 28, 28) with values 0..255
    x_norm = x_test.astype(np.float32) / 255.0

    if architecture.upper() == ARCH_ANN:
        # Flatten to (N, 784)
        x_eval = x_norm.reshape(len(x_norm), -1)
    elif architecture.upper() == ARCH_CNN:
        # Add channel dimension (N, 28, 28, 1)
        x_eval = np.expand_dims(x_norm, axis=-1)
    else:
        x_eval = x_norm

    # Measure inference latency on a sample batch
    sample_size = min(200, len(x_eval))
    start_time = time.perf_counter()
    _ = model.predict(x_eval[:sample_size], batch_size=sample_size, verbose=0)
    elapsed = time.perf_counter() - start_time
    latency_ms_per_sample = round((elapsed / sample_size) * 1000.0, 3)

    # Full predictions
    probabilities = model.predict(x_eval, batch_size=batch_size, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))
    cm_list = cm.tolist()

    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(len(classes))), average=None, zero_division=0
    )

    per_class_dict = {}
    for i, class_name in enumerate(classes):
        per_class_dict[class_name] = {
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1_score": round(float(per_class_f1[i]), 4),
            "support": int(per_class_supp[i])
        }

    return {
        "task": task,
        "architecture": architecture.upper(),
        "total_test_samples": int(len(y_test)),
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(p_weighted), 4),
        "weighted_recall": round(float(r_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "latency_ms_per_sample": latency_ms_per_sample,
        "classes": classes,
        "confusion_matrix": cm_list,
        "per_class_metrics": per_class_dict
    }
