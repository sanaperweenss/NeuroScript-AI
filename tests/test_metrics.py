"""
Unit Tests - Model Evaluation Metrics
Tests evaluation computation, confusion matrix generation, and reporting logic.
"""

import numpy as np
import tensorflow as tf
from training.evaluation import evaluate_model_on_test_set
from models.ann import build_ann_model
from config.settings import TASK_DIGITS, ARCH_ANN


def test_evaluate_model_metrics():
    model = build_ann_model(num_classes=10, hidden_units_1=32, hidden_units_2=16)

    # 20 synthetic test samples
    x_test = np.random.randint(0, 256, (20, 28, 28), dtype=np.uint8)
    y_test = np.random.randint(0, 10, 20, dtype=np.uint8)

    metrics = evaluate_model_on_test_set(
        model=model,
        x_test=x_test,
        y_test=y_test,
        task=TASK_DIGITS,
        architecture=ARCH_ANN
    )

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "macro_precision" in metrics
    assert "macro_recall" in metrics
    assert "confusion_matrix" in metrics
    assert "latency_ms_per_sample" in metrics

    # Confusion matrix should be 10x10
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (10, 10)
    assert np.sum(cm) == 20
