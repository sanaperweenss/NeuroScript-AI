"""
NeuroScript AI - Model Training Engine
End-to-end training pipeline for ANN and CNN models:
1. Loads verified dataset partitions
2. Normalizes and reshapes tensors
3. Compiles architecture via ModelFactory
4. Tracks loss and accuracy convergence in real-time
5. Evaluates held-out test split with full confusion matrix and classification metrics
6. Persists artifacts to ModelRegistry and ExperimentStore
"""

import time
from typing import Dict, Any, Optional, Callable, Tuple
import numpy as np
import tensorflow as tf
from config.settings import (
    ARCH_ANN, ARCH_CNN, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE, DEFAULT_VALIDATION_SPLIT
)
from data.loaders import load_dataset_for_task
from data.labels import LabelMapper
from models.model_factory import ModelFactory
from models.model_registry import get_model_registry
from storage.experiment_store import get_experiment_store
from training.callbacks import StreamlitTrainingCallback
from training.evaluation import evaluate_model_on_test_set


class ModelTrainer:
    """
    Executes and monitors neural network training runs.
    """

    def __init__(self):
        self.model_registry = get_model_registry()
        self.experiment_store = get_experiment_store()

    def train(
        self,
        task: str,
        architecture: str,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        validation_split: float = DEFAULT_VALIDATION_SPLIT,
        model_name: Optional[str] = None,
        progress_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        max_train_samples: Optional[int] = None
    ) -> Tuple[tf.keras.Model, Dict[str, Any], str]:
        """
        Execute full training workflow.
        Returns:
            (trained_model, evaluation_metrics, experiment_id)
        """
        arch = architecture.upper()

        # Step 1: Load Dataset
        x_train, y_train, x_test, y_test = load_dataset_for_task(task)

        # Optional subsampling for CPU speed if requested
        if max_train_samples and max_train_samples < len(x_train):
            indices = np.random.choice(len(x_train), max_train_samples, replace=False)
            x_train = x_train[indices]
            y_train = y_train[indices]

        # Step 2: Normalize to [0.0, 1.0]
        x_train_norm = x_train.astype(np.float32) / 255.0

        # Step 3: Reshape for Architecture
        if arch == ARCH_ANN:
            x_train_tensor = x_train_norm.reshape(len(x_train_norm), -1)
        elif arch == ARCH_CNN:
            x_train_tensor = np.expand_dims(x_train_norm, axis=-1)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Step 4: Construct and Compile Model
        model = ModelFactory.create_model(
            task=task,
            architecture=arch,
            learning_rate=learning_rate
        )

        # Step 5: Callbacks
        st_callback = StreamlitTrainingCallback(
            total_epochs=epochs,
            epoch_callback=progress_callback,
            cancel_check=cancel_check
        )

        callbacks = [st_callback]
        if epochs >= 5:
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
                verbose=0
            )
            callbacks.append(early_stop)

        # Step 6: Train Model
        start_time = time.perf_counter()
        model.fit(
            x_train_tensor,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0,
            shuffle=True
        )
        duration_seconds = time.perf_counter() - start_time

        # Step 7: Evaluate on Held-out Test Set
        eval_metrics = evaluate_model_on_test_set(
            model=model,
            x_test=x_test,
            y_test=y_test,
            task=task,
            architecture=arch,
            batch_size=batch_size
        )

        hyperparams = {
            "task": task,
            "architecture": arch,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_split": validation_split,
            "train_samples": int(len(x_train)),
            "test_samples": int(len(x_test))
        }

        # Step 8: Save Model Checkpoint
        model_path, _ = self.model_registry.save_model(
            model=model,
            task=task,
            architecture=arch,
            model_name=model_name,
            hyperparameters=hyperparams,
            evaluation_metrics=eval_metrics
        )

        # Step 9: Save Experiment Record
        mapper = LabelMapper(task)
        exp_id = self.experiment_store.save_experiment(
            task=task,
            architecture=arch,
            model_name=model_name or model_path.stem,
            hyperparameters=hyperparams,
            training_history=st_callback.history_records,
            evaluation_metrics=eval_metrics,
            confusion_matrix=eval_metrics.get("confusion_matrix"),
            classes=mapper.get_classes(),
            duration_seconds=duration_seconds,
            model_path=str(model_path)
        )

        return model, eval_metrics, exp_id


_trainer = None


def get_model_trainer() -> ModelTrainer:
    global _trainer
    if _trainer is None:
        _trainer = ModelTrainer()
    return _trainer
