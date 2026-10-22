"""
NeuroScript AI - Convolutional Neural Network (CNN) Architecture
Implements a state-of-the-art 2D Convolutional Neural Network for handwriting recognition.
Extracts localized spatial stroke features with translation invariance,
hierarchical pooling, dropout regularization, and a dense classification head.
Includes activation hook helpers for real-time feature-map visualization.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers


def build_cnn_model(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_filters_1: int = 32,
    num_filters_2: int = 64,
    dense_units: int = 128,
    dropout_rate: float = 0.35,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Construct and compile Convolutional Neural Network.
    Args:
        num_classes: Output class count (10, 26, or 36)
        input_shape: (28, 28, 1) single-channel image tensor
        num_filters_1: Filters in first convolution block
        num_filters_2: Filters in second convolution block
        dense_units: Units in classification dense hidden layer
        dropout_rate: Dropout fraction
        learning_rate: Initial Adam learning rate
    Returns:
        Compiled Keras Model
    """
    model = models.Sequential(name="NeuroScript_CNN")

    # Input Tensor
    model.add(layers.Input(shape=input_shape, name="input_image"))

    # Block 1: Feature Extraction (Low-level stroke edges & curves)
    model.add(layers.Conv2D(
        num_filters_1,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv2d_1"
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_1"))  # 28x28 -> 14x14

    # Block 2: Feature Extraction (Mid-level stroke junctions & loops)
    model.add(layers.Conv2D(
        num_filters_2,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="conv2d_2"
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="maxpool_2"))  # 14x14 -> 7x7

    # Classification Head
    model.add(layers.Flatten(name="flatten_features"))
    model.add(layers.Dense(dense_units, activation="relu", name="dense_classifier"))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate, name="dropout_classifier"))

    # Output Layer
    model.add(layers.Dense(num_classes, activation="softmax", name="output_probabilities"))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_cnn_metadata(model: tf.keras.Model) -> Dict[str, Any]:
    """Extract structural summary, trainable parameter count, and tensor dimensions."""
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
        "architecture": "CNN",
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "non_trainable_parameters": int(non_trainable_params),
        "layers": layers_info
    }


def extract_layer_activations(
    model: tf.keras.Model,
    input_tensor: np.ndarray,
    target_layer_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Feed an input image through the trained CNN and extract intermediate feature maps.
    Returns dictionary mapping layer name to activation array (H, W, Filters).
    """
    if target_layer_names is None:
        # Default interesting layers
        target_layer_names = ["conv2d_1", "relu_conv_1", "maxpool_1", "conv2d_2", "relu_conv_2", "maxpool_2"]

    # Filter layers that actually exist in the model
    valid_layers = [l for l in model.layers if l.name in target_layer_names]
    if not valid_layers:
        return {}

    activation_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[layer.output for layer in valid_layers]
    )

    activations = activation_model.predict(input_tensor, verbose=0)
    if not isinstance(activations, list):
        activations = [activations]

    result = {}
    for layer, act in zip(valid_layers, activations):
        result[layer.name] = act[0]  # First item in batch: shape (H, W, Channels)

    return result


def extract_conv_weights(model: tf.keras.Model, layer_name: str) -> Optional[np.ndarray]:
    """
    Extract convolutional kernel weights for visual inspection.
    Returns array of shape (Kernel_H, Kernel_W, In_Channels, Out_Channels).
    """
    for layer in model.layers:
        if layer.name == layer_name and isinstance(layer, layers.Conv2D):
            weights = layer.get_weights()
            if weights:
                return weights[0]  # Kernel weights
    return None
