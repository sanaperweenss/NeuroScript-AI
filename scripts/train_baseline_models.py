"""
NeuroScript AI - Initial Baseline Model Trainer
Trains real baseline ANN and CNN models for Digits and Alphabet tasks,
evaluates them on held-out test sets, and registers checkpoints and experiments.
"""

import sys
import os
from pathlib import Path

# Ensure project root is in path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import TASK_DIGITS, TASK_ALPHABET, ARCH_ANN, ARCH_CNN
from training.trainer import get_model_trainer
from data.loaders import load_dataset_for_task

def main():
    print("=" * 60)
    print("NeuroScript AI - Training Initial Baseline Models")
    print("=" * 60)

    trainer = get_model_trainer()

    # 1. Train MNIST Digit Models
    print("\n[1/4] Training MNIST Digit ANN Model...")
    ann_dig_model, ann_dig_metrics, ann_dig_exp = trainer.train(
        task=TASK_DIGITS,
        architecture=ARCH_ANN,
        epochs=4,
        batch_size=128,
        learning_rate=0.001,
        max_train_samples=10000,
        model_name="ann_mnist_digits_baseline"
    )
    print(f"-> ANN Digits Test Accuracy: {ann_dig_metrics['accuracy']*100:.2f}% | Macro F1: {ann_dig_metrics['macro_f1']*100:.2f}%")

    print("\n[2/4] Training MNIST Digit CNN Model...")
    cnn_dig_model, cnn_dig_metrics, cnn_dig_exp = trainer.train(
        task=TASK_DIGITS,
        architecture=ARCH_CNN,
        epochs=4,
        batch_size=128,
        learning_rate=0.001,
        max_train_samples=10000,
        model_name="cnn_mnist_digits_baseline"
    )
    print(f"-> CNN Digits Test Accuracy: {cnn_dig_metrics['accuracy']*100:.2f}% | Macro F1: {cnn_dig_metrics['macro_f1']*100:.2f}%")

    # 2. Train Alphabet Models
    print("\n[3/4] Training EMNIST Alphabet ANN Model...")
    ann_alp_model, ann_alp_metrics, ann_alp_exp = trainer.train(
        task=TASK_ALPHABET,
        architecture=ARCH_ANN,
        epochs=4,
        batch_size=128,
        learning_rate=0.001,
        max_train_samples=5200,
        model_name="ann_emnist_alphabet_baseline"
    )
    print(f"-> ANN Alphabet Test Accuracy: {ann_alp_metrics['accuracy']*100:.2f}% | Macro F1: {ann_alp_metrics['macro_f1']*100:.2f}%")

    print("\n[4/4] Training EMNIST Alphabet CNN Model...")
    cnn_alp_model, cnn_alp_metrics, cnn_alp_exp = trainer.train(
        task=TASK_ALPHABET,
        architecture=ARCH_CNN,
        epochs=4,
        batch_size=128,
        learning_rate=0.001,
        max_train_samples=5200,
        model_name="cnn_emnist_alphabet_baseline"
    )
    print(f"-> CNN Alphabet Test Accuracy: {cnn_alp_metrics['accuracy']*100:.2f}% | Macro F1: {cnn_alp_metrics['macro_f1']*100:.2f}%")

    print("\n" + "=" * 60)
    print("All baseline models successfully trained and persisted!")
    print("=" * 60)

if __name__ == "__main__":
    main()
