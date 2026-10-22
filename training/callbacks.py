"""
NeuroScript AI - Training Callbacks
Custom Keras callbacks for real-time training progress tracking, live metric streaming,
and safe cancellation support.
"""

from typing import Callable, Optional, Dict, Any, List
import tensorflow as tf


class StreamlitTrainingCallback(tf.keras.callbacks.Callback):
    """
    Keras callback that reports progress and logs back to Streamlit or custom hooks.
    """

    def __init__(
        self,
        total_epochs: int,
        epoch_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None
    ):
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_callback = epoch_callback
        self.cancel_check = cancel_check
        self.history_records: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None):
        logs = logs or {}

        # Check if cancellation requested
        if self.cancel_check and self.cancel_check():
            self.model.stop_training = True
            return

        loss = float(logs.get("loss", 0.0))
        acc = float(logs.get("accuracy", 0.0))
        val_loss = float(logs.get("val_loss", 0.0))
        val_acc = float(logs.get("val_accuracy", 0.0))

        self.history_records["loss"].append(loss)
        self.history_records["accuracy"].append(acc)
        self.history_records["val_loss"].append(val_loss)
        self.history_records["val_accuracy"].append(val_acc)

        if self.epoch_callback:
            self.epoch_callback(epoch + 1, {
                "epoch": epoch + 1,
                "total_epochs": self.total_epochs,
                "loss": loss,
                "accuracy": acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
