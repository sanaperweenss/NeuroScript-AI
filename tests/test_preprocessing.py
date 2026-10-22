"""
Unit Tests - Image Preprocessing Pipeline
Tests polarity normalization, bounding box extraction, LeCun centering,
empty canvas detection, and tensor output dimensions.
"""

import numpy as np
from PIL import Image
from data.preprocessing import ImagePreprocessor, get_default_preprocessor
from config.settings import ARCH_ANN, ARCH_CNN


def test_empty_image_detection():
    prep = get_default_preprocessor()
    # Pure black image
    black_canvas = np.zeros((280, 280), dtype=np.uint8)
    assert prep.is_image_empty(black_canvas)

    # Preprocessing an empty image should return None tensor and is_empty=True
    tensor, meta = prep.preprocess_image(black_canvas)
    assert tensor is None
    assert meta["is_empty"] is True


def test_polarity_inversion():
    prep = get_default_preprocessor()
    # White background (255) with dark stroke (0)
    white_bg = np.ones((100, 100), dtype=np.uint8) * 255
    white_bg[40:60, 45:55] = 0  # Dark vertical bar
    inverted = prep.normalize_polarity(white_bg)

    # Inverted should have dark border (0) and bright stroke (255)
    assert inverted[0, 0] == 0
    assert inverted[50, 50] == 255


def test_bounding_box_and_centering():
    prep = get_default_preprocessor()
    # Create dark canvas with a centered 10x10 bright square
    img = np.zeros((100, 100), dtype=np.uint8)
    img[45:55, 45:55] = 255

    cropped = prep.extract_bounding_box(img)
    assert cropped is not None
    assert cropped.shape[0] == 10
    assert cropped.shape[1] == 10

    # Center into 28x28
    centered = prep.center_and_pad(cropped)
    assert centered.shape == (28, 28)
    assert centered.dtype == np.uint8
    # Content should be non-zero
    assert np.max(centered) > 0


def test_cnn_output_tensor_shape():
    prep = get_default_preprocessor()
    # Mock drawn digit
    canvas = np.ones((280, 280), dtype=np.uint8) * 255
    canvas[80:200, 130:150] = 0  # Stroke

    tensor, meta = prep.preprocess_image(canvas, architecture=ARCH_CNN)
    assert meta["is_empty"] is False
    assert tensor is not None
    assert tensor.shape == (1, 28, 28, 1)
    assert tensor.dtype == np.float32
    assert 0.0 <= np.min(tensor) and np.max(tensor) <= 1.0


def test_ann_output_tensor_shape():
    prep = get_default_preprocessor()
    canvas = np.ones((280, 280), dtype=np.uint8) * 255
    canvas[80:200, 130:150] = 0

    tensor, meta = prep.preprocess_image(canvas, architecture=ARCH_ANN)
    assert meta["is_empty"] is False
    assert tensor is not None
    assert tensor.shape == (1, 784)
    assert tensor.dtype == np.float32
