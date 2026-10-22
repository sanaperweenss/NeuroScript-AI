"""
Unit Tests - ANN and CNN Model Architectures
Tests model building, parameter counts, summary metadata, and forward pass on synthetic tensors.
"""

import numpy as np
import tensorflow as tf
from models.ann import build_ann_model, get_ann_metadata
from models.cnn import build_cnn_model, get_cnn_metadata
from models.model_factory import ModelFactory
from config.settings import ARCH_ANN, ARCH_CNN, TASK_DIGITS, TASK_ALPHABET


def test_ann_model_construction():
    model = build_ann_model(num_classes=10, input_dim=784, hidden_units_1=128, hidden_units_2=64)
    assert model is not None
    assert model.name == "NeuroScript_ANN"

    # Forward pass on small synthetic batch of shape (2, 784)
    synthetic_batch = np.random.rand(2, 784).astype(np.float32)
    output = model(synthetic_batch)
    assert output.shape == (2, 10)

    # Softmax probabilities must sum to 1.0
    prob_sums = np.sum(output.numpy(), axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5)

    meta = get_ann_metadata(model)
    assert meta["architecture"] == "ANN"
    assert meta["trainable_parameters"] > 0


def test_cnn_model_construction():
    model = build_cnn_model(num_classes=26, input_shape=(28, 28, 1), num_filters_1=16, num_filters_2=32)
    assert model is not None
    assert model.name == "NeuroScript_CNN"

    # Forward pass on small synthetic batch of shape (2, 28, 28, 1)
    synthetic_batch = np.random.rand(2, 28, 28, 1).astype(np.float32)
    output = model(synthetic_batch)
    assert output.shape == (2, 26)

    prob_sums = np.sum(output.numpy(), axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5)

    meta = get_cnn_metadata(model)
    assert meta["architecture"] == "CNN"
    assert meta["trainable_parameters"] > 0


def test_model_factory():
    ann = ModelFactory.create_model(task=TASK_DIGITS, architecture=ARCH_ANN)
    assert ann.output_shape == (None, 10)

    cnn = ModelFactory.create_model(task=TASK_ALPHABET, architecture=ARCH_CNN)
    assert cnn.output_shape == (None, 26)
