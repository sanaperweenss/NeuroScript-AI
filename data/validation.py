"""
NeuroScript AI - Input and Parameter Validation
Provides defensive guards for file uploads, hyperparameter bounds, and task selections.
"""

from typing import Tuple, Optional
from PIL import Image
from config.settings import TASK_CHOICES


def validate_uploaded_image(file_obj, max_size_mb: float = 10.0) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded image file type, size, and integrity.
    """
    if file_obj is None:
        return False, "No file provided."

    # Size check
    if hasattr(file_obj, "size"):
        size_mb = file_obj.size / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_size_mb} MB."

    # Image decoding check
    try:
        img = Image.open(file_obj)
        img.verify()
        return True, None
    except Exception as e:
        return False, f"Corrupted or invalid image file: {e}"


def validate_task_name(task: str) -> bool:
    """Validate task selection."""
    return task in TASK_CHOICES


def validate_hyperparameters(epochs: int, batch_size: int, learning_rate: float) -> Tuple[bool, Optional[str]]:
    """Verify hyperparameter bounds."""
    if not (1 <= epochs <= 100):
        return False, "Epochs must be between 1 and 100."
    if batch_size not in [16, 32, 64, 128, 256, 512]:
        return False, "Batch size must be a standard power of 2 (16 to 512)."
    if not (1e-5 <= learning_rate <= 1.0):
        return False, "Learning rate must be between 1e-5 and 1.0."
    return True, None
