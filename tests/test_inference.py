"""
Unit Tests - Inference Engine
Tests predictor processing, top-k rankings, and graceful empty input handling.
"""

import numpy as np
from PIL import Image
from inference.predictor import CharacterPredictor
from models.cnn import build_cnn_model
from config.settings import TASK_DIGITS, ARCH_CNN


def test_predictor_valid_inference():
    predictor = CharacterPredictor()
    model = build_cnn_model(num_classes=10, num_filters_1=8, num_filters_2=16)

    # Synthetic image with a stroke
    canvas = np.ones((280, 280), dtype=np.uint8) * 255
    canvas[100:200, 135:145] = 0

    res = predictor.predict(
        image_input=canvas,
        model=model,
        task=TASK_DIGITS,
        architecture=ARCH_CNN,
        top_k_count=3
    )

    assert res["success"] is True
    assert res["predicted_class"] in [str(i) for i in range(10)]
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_k"]) == 3
    assert res["preprocessed_image"].shape == (28, 28)
    assert res["inference_time_ms"] > 0


def test_predictor_empty_canvas_rejection():
    predictor = CharacterPredictor()
    model = build_cnn_model(num_classes=10)

    # Empty canvas
    blank = np.zeros((280, 280), dtype=np.uint8)
    res = predictor.predict(
        image_input=blank,
        model=model,
        task=TASK_DIGITS,
        architecture=ARCH_CNN
    )

    assert res["success"] is False
    assert res["error"] == "EMPTY_INPUT"
    assert "blank" in res["message"].lower()
