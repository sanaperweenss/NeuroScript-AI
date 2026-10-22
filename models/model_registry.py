"""
NeuroScript AI - Model Registry and Checkpoint Manager
Scans, stores, versions, and loads trained ANN and CNN models with their metadata,
ensuring seamless retrieval for inference, comparison, and feature visualization.
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import tensorflow as tf
from config.settings import (
    MODELS_DIR, ANN_MODELS_DIR, CNN_MODELS_DIR, ARCH_ANN, ARCH_CNN
)
from data.labels import LabelMapper


class ModelRegistry:
    """
    Central repository for saved model artifacts (.keras) and companion metadata (.json).
    """

    def __init__(self):
        for d in [MODELS_DIR, ANN_MODELS_DIR, CNN_MODELS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def _get_arch_dir(self, architecture: str) -> Path:
        if architecture.upper() == ARCH_ANN:
            return ANN_MODELS_DIR
        elif architecture.upper() == ARCH_CNN:
            return CNN_MODELS_DIR
        return MODELS_DIR

    def save_model(
        self,
        model: tf.keras.Model,
        task: str,
        architecture: str,
        model_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        evaluation_metrics: Optional[Dict[str, Any]] = None
    ) -> Tuple[Path, Path]:
        """
        Save Keras model artifact and companion metadata JSON.
        Returns:
            (model_path, metadata_path)
        """
        arch = architecture.upper()
        arch_dir = self._get_arch_dir(arch)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not model_name:
            model_name = f"{task}_{arch.lower()}_{timestamp}"

        # Clean filename
        safe_name = "".join([c if c.isalnum() or c in "_-" else "_" for c in model_name])
        model_path = arch_dir / f"{safe_name}.keras"
        meta_path = arch_dir / f"{safe_name}_meta.json"

        # Save Keras 3 format
        model.save(str(model_path))

        mapper = LabelMapper(task)
        meta_content = {
            "model_name": safe_name,
            "task": task,
            "architecture": arch,
            "classes": mapper.get_classes(),
            "num_classes": mapper.num_classes,
            "created_at": datetime.now().isoformat(),
            "model_file": model_path.name,
            "hyperparameters": hyperparameters or {},
            "evaluation_metrics": evaluation_metrics or {},
            "total_parameters": int(model.count_params())
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_content, f, indent=2)

        return model_path, meta_path

    def load_model(self, model_name: str, architecture: str) -> Tuple[tf.keras.Model, Dict[str, Any]]:
        """
        Load a trained model and its metadata by name.
        """
        arch_dir = self._get_arch_dir(architecture)
        clean_name = model_name.replace(".keras", "").replace("_meta.json", "")

        model_path = arch_dir / f"{clean_name}.keras"
        meta_path = arch_dir / f"{clean_name}_meta.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load weights and model
        model = tf.keras.models.load_model(str(model_path))

        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        return model, metadata

    def list_models(
        self,
        task: Optional[str] = None,
        architecture: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all available models with metadata.
        """
        models_list = []
        target_dirs = []

        if architecture:
            target_dirs.append(self._get_arch_dir(architecture))
        else:
            target_dirs.extend([ANN_MODELS_DIR, CNN_MODELS_DIR])

        for d in target_dirs:
            if not d.exists():
                continue
            for meta_file in d.glob("*_meta.json"):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        if task and meta.get("task") != task:
                            continue
                        # Verify the binary model file exists
                        model_file = d / meta.get("model_file", f"{meta_file.stem.replace('_meta', '')}.keras")
                        if model_file.exists():
                            meta["file_size_mb"] = round(model_file.stat().st_size / (1024 * 1024), 2)
                            meta["path"] = str(model_file)
                            models_list.append(meta)
                except Exception:
                    continue

        models_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return models_list

    def get_latest_model(self, task: str, architecture: str) -> Optional[Tuple[tf.keras.Model, Dict[str, Any]]]:
        """
        Get the most recently trained model for a specific task and architecture.
        """
        models_list = self.list_models(task=task, architecture=architecture)
        if not models_list:
            return None
        latest_meta = models_list[0]
        return self.load_model(latest_meta["model_name"], architecture)

    def delete_model(self, model_name: str, architecture: str) -> bool:
        """Delete model artifact and metadata."""
        arch_dir = self._get_arch_dir(architecture)
        clean_name = model_name.replace(".keras", "").replace("_meta.json", "")

        model_path = arch_dir / f"{clean_name}.keras"
        meta_path = arch_dir / f"{clean_name}_meta.json"

        deleted = False
        if model_path.exists():
            model_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()
            deleted = True
        return deleted


_model_registry = None


def get_model_registry() -> ModelRegistry:
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry
