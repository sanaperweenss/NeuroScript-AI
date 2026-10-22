"""
NeuroScript AI - Dataset Management and Unified Loader
Provides caching, loading, validation, and sample inspection for:
- Task A: MNIST Digits (10 classes: 0-9)
- Task B: EMNIST Letters (26 classes: A-Z)
- Task C: Combined (36 classes: 0-9 and A-Z)
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from pathlib import Path
from config.settings import (
    PROCESSED_DATA_DIR, TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED
)
from data.labels import LabelMapper, DIGIT_CLASSES, ALPHABET_CLASSES, COMBINED_CLASSES
from data.emnist_loader import load_emnist_letters, is_emnist_cached, EMNIST_LETTERS_CACHE

MNIST_CACHE = PROCESSED_DATA_DIR / "mnist.npz"
COMBINED_CACHE = PROCESSED_DATA_DIR / "combined_36.npz"


def load_mnist_data(allow_download: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load standard MNIST digits.
    Uses local cache if available; otherwise loads via Keras / download.
    Returns:
        x_train: (60000, 28, 28), y_train: (60000,)
        x_test: (10000, 28, 28), y_test: (10000,)
    """
    if MNIST_CACHE.exists() and MNIST_CACHE.stat().st_size > 1000000:
        data = np.load(MNIST_CACHE)
        return data["x_train"], data["y_train"], data["x_test"], data["y_test"]

    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            MNIST_CACHE,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )
        return x_train, y_train, x_test, y_test
    except Exception:
        # Fallback to direct download from storage mirror
        import urllib.request
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, MNIST_CACHE)
        data = np.load(MNIST_CACHE)
        return data["x_train"], data["y_train"], data["x_test"], data["y_test"]


def load_dataset_for_task(
    task: str,
    allow_download: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unified loader for any supported task.
    Returns:
        (x_train, y_train, x_test, y_test)
        where images are 28x28 uint8 and labels are 0-indexed integers.
    """
    if task == TASK_DIGITS:
        return load_mnist_data(allow_download=allow_download)

    elif task == TASK_ALPHABET:
        return load_emnist_letters(allow_download=allow_download)

    elif task == TASK_COMBINED:
        if COMBINED_CACHE.exists() and COMBINED_CACHE.stat().st_size > 1000000:
            data = np.load(COMBINED_CACHE)
            return data["x_train"], data["y_train"], data["x_test"], data["y_test"]

        # Merge MNIST (0-9) and EMNIST Letters (A-Z shifted by 10)
        x_tr_d, y_tr_d, x_te_d, y_te_d = load_mnist_data(allow_download=allow_download)
        x_tr_a, y_tr_a, x_te_a, y_te_a = load_emnist_letters(allow_download=allow_download)

        # Shift alphabet labels by 10 (0..25 becomes 10..35)
        y_tr_a_shifted = y_tr_a + 10
        y_te_a_shifted = y_te_a + 10

        # Subsample to maintain balanced representation
        # MNIST has 60k train / 10k test (6k / class train, 1k / class test)
        # Take up to 2,000 per class for balanced combined training
        def balance_split(x, y, num_classes, max_per_class):
            balanced_x, balanced_y = [], []
            for c in range(num_classes):
                idx = np.where(y == c)[0]
                if len(idx) > max_per_class:
                    idx = np.random.choice(idx, max_per_class, replace=False)
                balanced_x.append(x[idx])
                balanced_y.append(y[idx])
            return np.concatenate(balanced_x, axis=0), np.concatenate(balanced_y, axis=0)

        x_train = np.concatenate([x_tr_d, x_tr_a], axis=0)
        y_train = np.concatenate([y_tr_d, y_tr_a_shifted], axis=0)
        x_test = np.concatenate([x_te_d, x_te_a], axis=0)
        y_test = np.concatenate([y_te_d, y_te_a_shifted], axis=0)

        # Shuffle
        p_tr = np.random.permutation(len(y_train))
        x_train, y_train = x_train[p_tr], y_train[p_tr]
        p_te = np.random.permutation(len(y_test))
        x_test, y_test = x_test[p_te], y_test[p_te]

        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            COMBINED_CACHE,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test
        )
        return x_train, y_train, x_test, y_test

    else:
        raise ValueError(f"Unknown task: {task}")


def get_dataset_stats(task: str) -> Dict[str, Any]:
    """Retrieve statistical summary of dataset without loading full arrays into memory if cached."""
    cache_map = {
        TASK_DIGITS: MNIST_CACHE,
        TASK_ALPHABET: EMNIST_LETTERS_CACHE,
        TASK_COMBINED: COMBINED_CACHE
    }
    target_cache = cache_map.get(task)
    mapper = LabelMapper(task)

    if target_cache and target_cache.exists():
        try:
            data = np.load(target_cache)
            y_train = data["y_train"]
            y_test = data["y_test"]
            class_counts = {mapper.index_to_label(i): int(np.sum(y_train == i)) for i in range(mapper.num_classes)}
            return {
                "task": task,
                "is_available": True,
                "train_samples": int(len(y_train)),
                "test_samples": int(len(y_test)),
                "total_samples": int(len(y_train) + len(y_test)),
                "num_classes": mapper.num_classes,
                "classes": mapper.get_classes(),
                "image_shape": (28, 28),
                "class_counts": class_counts
            }
        except Exception:
            pass

    return {
        "task": task,
        "is_available": False,
        "train_samples": 0,
        "test_samples": 0,
        "total_samples": 0,
        "num_classes": mapper.num_classes,
        "classes": mapper.get_classes(),
        "image_shape": (28, 28),
        "class_counts": {}
    }


def get_class_samples(task: str, samples_per_class: int = 1) -> Dict[str, List[np.ndarray]]:
    """Retrieve real image samples for UI Explorer display."""
    x_train, y_train, _, _ = load_dataset_for_task(task)
    mapper = LabelMapper(task)
    samples = {}

    for i, class_name in enumerate(mapper.get_classes()):
        indices = np.where(y_train == i)[0]
        if len(indices) > 0:
            chosen = indices[:samples_per_class]
            samples[class_name] = [x_train[idx] for idx in chosen]
        else:
            samples[class_name] = []

    return samples
