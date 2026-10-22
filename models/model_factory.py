"""
NeuroScript AI - Model Factory
Instantiates and configures ANN and CNN models with consistent interfaces.
"""

from typing import Dict, Any, Optional
import tensorflow as tf
from config.settings import (
    ARCH_ANN, ARCH_CNN, TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED
)
from data.labels import LabelMapper
from models.ann import build_ann_model, get_ann_metadata
from models.cnn import build_cnn_model, get_cnn_metadata


class ModelFactory:
    """
    Constructs neural network instances based on task specifications and architecture requirements.
    """

    @staticmethod
    def create_model(
        task: str,
        architecture: str,
        learning_rate: float = 0.001,
        **kwargs
    ) -> tf.keras.Model:
        """
        Create and compile model.
        Args:
            task: 'digits', 'alphabet', or 'combined'
            architecture: 'ANN' or 'CNN'
            learning_rate: Optimizer learning rate
            **kwargs: Extra architectural parameters
        """
        mapper = LabelMapper(task)
        num_classes = mapper.num_classes
        arch = architecture.upper()

        if arch == ARCH_ANN:
            return build_ann_model(
                num_classes=num_classes,
                learning_rate=learning_rate,
                **kwargs
            )
        elif arch == ARCH_CNN:
            return build_cnn_model(
                num_classes=num_classes,
                learning_rate=learning_rate,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported architecture '{architecture}'. Choose '{ARCH_ANN}' or '{ARCH_CNN}'.")

    @staticmethod
    def get_metadata(model: tf.keras.Model, architecture: str) -> Dict[str, Any]:
        """Extract metadata for given architecture."""
        if architecture.upper() == ARCH_ANN:
            return get_ann_metadata(model)
        elif architecture.upper() == ARCH_CNN:
            return get_cnn_metadata(model)
        return {"architecture": architecture, "parameters": model.count_params()}
