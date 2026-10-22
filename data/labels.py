"""
NeuroScript AI - Label Mappings and Class Conventions
Handles class mappings for Digits (MNIST), Alphabet (EMNIST Letters), and Combined (36-class).
"""

from typing import List, Dict, Optional, Union
import string
from config.settings import TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED

# Task A: Digits 0-9
DIGIT_CLASSES: List[str] = [str(i) for i in range(10)]

# Task B: English Uppercase Alphabet A-Z (26 classes)
ALPHABET_CLASSES: List[str] = list(string.ascii_uppercase)

# Task C: Combined 36-class space (0-9 followed by A-Z)
COMBINED_CLASSES: List[str] = DIGIT_CLASSES + ALPHABET_CLASSES


class LabelMapper:
    """
    Translates between integer indices, categorical labels, and raw dataset representations.
    Ensures consistent label spaces across training, evaluation, and inference.
    """

    def __init__(self, task: str):
        if task not in [TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED]:
            raise ValueError(
                f"Unsupported task '{task}'. Must be one of {[TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED]}"
            )
        self.task = task
        self.classes = self._get_classes()
        self.num_classes = len(self.classes)
        self._label_to_index: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self._index_to_label: Dict[int, str] = {i: c for i, c in enumerate(self.classes)}

    def _get_classes(self) -> List[str]:
        if self.task == TASK_DIGITS:
            return DIGIT_CLASSES
        elif self.task == TASK_ALPHABET:
            return ALPHABET_CLASSES
        elif self.task == TASK_COMBINED:
            return COMBINED_CLASSES
        raise ValueError(f"Unknown task: {self.task}")

    def index_to_label(self, index: int) -> str:
        """Map integer class index (0-based) to string label."""
        if index not in self._index_to_label:
            raise KeyError(f"Index {index} out of range for task {self.task} (0 to {self.num_classes - 1})")
        return self._index_to_label[index]

    def label_to_index(self, label: str) -> int:
        """Map string label to 0-based integer class index."""
        cleaned = str(label).strip().upper()
        if cleaned not in self._label_to_index:
            raise KeyError(f"Label '{label}' not recognized for task {self.task}")
        return self._label_to_index[cleaned]

    def get_classes(self) -> List[str]:
        """Return list of all class labels in order."""
        return list(self.classes)

    @staticmethod
    def normalize_emnist_raw_label(raw_label: int, is_one_indexed: bool = True) -> int:
        """
        EMNIST Letters raw dataset files are officially 1-indexed (1=A ... 26=Z).
        Standard neural networks expect 0-indexed targets (0=A ... 25=Z).
        """
        if is_one_indexed:
            if 1 <= raw_label <= 26:
                return raw_label - 1
            raise ValueError(f"Invalid 1-indexed EMNIST label: {raw_label}. Expected range 1-26.")
        else:
            if 0 <= raw_label <= 25:
                return raw_label
            raise ValueError(f"Invalid 0-indexed EMNIST label: {raw_label}. Expected range 0-25.")

    @staticmethod
    def raw_emnist_to_char(raw_label: int) -> str:
        """Convert raw EMNIST label directly to character."""
        norm_idx = LabelMapper.normalize_emnist_raw_label(raw_label)
        return ALPHABET_CLASSES[norm_idx]

    def to_dict(self) -> Dict[str, Union[str, int, List[str]]]:
        """Export metadata dictionary for model checkpoints."""
        return {
            "task": self.task,
            "num_classes": self.num_classes,
            "classes": self.classes,
        }


def get_label_mapper(task: str) -> LabelMapper:
    """Factory helper to obtain a LabelMapper instance."""
    return LabelMapper(task)
