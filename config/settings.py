"""
NeuroScript AI - System Configuration and Settings
Central configuration for paths, model parameters, preprocessing, and UI theme.
"""

from pathlib import Path
import os

# Base directory for NeuroScript AI project
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage Directories
DATA_DIR = BASE_DIR / "data_store"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

MODELS_DIR = BASE_DIR / "models_store"
ANN_MODELS_DIR = MODELS_DIR / "ann"
CNN_MODELS_DIR = MODELS_DIR / "cnn"

EXPERIMENTS_DIR = BASE_DIR / "experiments_store"
ANN_EXPERIMENTS_DIR = EXPERIMENTS_DIR / "ann"
CNN_EXPERIMENTS_DIR = EXPERIMENTS_DIR / "cnn"
COMPARISON_EXPERIMENTS_DIR = EXPERIMENTS_DIR / "comparisons"

HISTORY_DIR = BASE_DIR / "history_store"
HISTORY_DB_PATH = HISTORY_DIR / "predictions.db"

REPORTS_DIR = BASE_DIR / "reports_store"

# Ensure all directories exist
for directory in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, METADATA_DIR,
    MODELS_DIR, ANN_MODELS_DIR, CNN_MODELS_DIR,
    EXPERIMENTS_DIR, ANN_EXPERIMENTS_DIR, CNN_EXPERIMENTS_DIR, COMPARISON_EXPERIMENTS_DIR,
    HISTORY_DIR, REPORTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)

# Image dimensions for models
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 1
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
FLAT_INPUT_SHAPE = (IMAGE_HEIGHT * IMAGE_WIDTH,)

# Supported Tasks
TASK_DIGITS = "digits"
TASK_ALPHABET = "alphabet"
TASK_COMBINED = "combined"

TASK_CHOICES = [TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED]

TASK_DESCRIPTIONS = {
    TASK_DIGITS: "Digits (0-9) - 10 Classes - MNIST",
    TASK_ALPHABET: "Alphabet (A-Z) - 26 Classes - EMNIST Letters",
    TASK_COMBINED: "Combined (0-9 & A-Z) - 36 Classes - Merged Digits & Alphabet"
}

# Supported Architectures
ARCH_ANN = "ANN"
ARCH_CNN = "CNN"
ARCH_CHOICES = [ARCH_ANN, ARCH_CNN]

# Default Training Hyperparameters
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_VALIDATION_SPLIT = 0.15

# Color Palette (Dark AI Workspace)
THEME_COLORS = {
    "bg_navy": "#101426",
    "card_indigo": "#191E38",
    "sidebar_navy": "#0C0F1E",
    "primary_violet": "#7564F5",
    "accent_cyan": "#42D9E8",
    "text_white": "#F6F7FB",
    "text_muted": "#9AA5C0",
    "success_green": "#36B37E",
    "warning_amber": "#F5B942",
    "danger_red": "#E84855",
    "border_subtle": "#252B4E"
}

APP_TITLE = "NeuroScript AI"
APP_TAGLINE = "From handwritten strokes to intelligent recognition"
APP_VERSION = "1.0.0"
