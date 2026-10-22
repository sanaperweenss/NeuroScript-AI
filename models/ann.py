"""
NeuroScript AI - Artificial Neural Network (ANN) Architecture
Implements a clean, modular Multilayer Perceptron (MLP) for digit and alphabet classification.
Flattens 28x28 grayscale image tensors into 784-dimensional feature vectors,
passing through configurable dense hidden layers with ReLU activations, dropout regularization,
and a softmax classification layer.
"""

from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers


def build_ann_model(
    num_classes: int,
    input_dim: int = 784,
    hidden_units_1: int = 256,
    hidden_units_2: int = 128,
    use_two_hidden: bool = True,
    dropout_rate: float = 0.3,
    l2_reg: float = 0.0001,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Construct and compile feedforward Artificial Neural Network.
    Args:
        num_classes: Number of target classes (10 for digits, 26 for alphabet, 36 for combined)
        input_dim: Flattened input dimension (default 784 = 28x28)
        hidden_units_1: Neurons in first dense hidden layer
        hidden_units_2: Neurons in second dense hidden layer
        use_two_hidden: Whether to include second hidden layer
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 weight regularization penalty
        learning_rate: Initial Adam learning rate
    Returns:
        Compiled Keras Model
    """
    model = models.Sequential(name="NeuroScript_ANN")

    # Explicit input shape
    model.add(layers.Input(shape=(input_dim,), name="input_pixels"))

    # First Hidden Layer + Batch Normalization + ReLU + Dropout
    model.add(layers.Dense(
        hidden_units_1,
        kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
        name="dense_hidden_1"
    ))
    model.add(layers.BatchNormalization(name="bn_1"))
    model.add(layers.Activation("relu", name="relu_1"))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="dropout_1"))

    # Optional Second Hidden Layer
    if use_two_hidden and hidden_units_2 > 0:
        model.add(layers.Dense(
            hidden_units_2,
            kernel_regularizer=regularizers.l2(l2_reg) if l2_reg > 0 else None,
            name="dense_hidden_2"
        ))
        model.add(layers.BatchNormalization(name="bn_2"))
        model.add(layers.Activation("relu", name="relu_2"))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name="dropout_2"))

    # Classification Layer (Softmax)
    model.add(layers.Dense(num_classes, activation="softmax", name="output_probabilities"))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_ann_metadata(model: tf.keras.Model) -> Dict[str, Any]:
    """Extract structural summary and trainable parameters."""
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    layers_info = []
    for layer in model.layers:
        out_shape = getattr(layer, "output_shape", None)
        if out_shape is None and hasattr(layer, "output") and hasattr(layer.output, "shape"):
            out_shape = layer.output.shape
        layers_info.append({
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": str(out_shape if out_shape is not None else "N/A"),
            "params": layer.count_params()
        })

    return {
        "architecture": "ANN",
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "non_trainable_parameters": int(non_trainable_params),
        "layers": layers_info
    }
