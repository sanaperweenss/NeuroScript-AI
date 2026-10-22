"""
NeuroScript AI - Experiment Store
Persists model training runs, hyperparameters, convergence curves,
evaluation metrics, and comparison artifacts as structured JSON.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from config.settings import (
    EXPERIMENTS_DIR, ANN_EXPERIMENTS_DIR, CNN_EXPERIMENTS_DIR,
    COMPARISON_EXPERIMENTS_DIR, ARCH_ANN, ARCH_CNN
)


class ExperimentStore:
    """
    Manages experiment metadata, training curves, and held-out test evaluations.
    """

    def __init__(self):
        for d in [EXPERIMENTS_DIR, ANN_EXPERIMENTS_DIR, CNN_EXPERIMENTS_DIR, COMPARISON_EXPERIMENTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def _get_target_dir(self, architecture: str) -> Path:
        if architecture.upper() == ARCH_ANN:
            return ANN_EXPERIMENTS_DIR
        elif architecture.upper() == ARCH_CNN:
            return CNN_EXPERIMENTS_DIR
        return EXPERIMENTS_DIR

    def save_experiment(
        self,
        task: str,
        architecture: str,
        model_name: str,
        hyperparameters: Dict[str, Any],
        training_history: Dict[str, List[float]],
        evaluation_metrics: Dict[str, Any],
        confusion_matrix: Optional[List[List[int]]] = None,
        classes: Optional[List[str]] = None,
        duration_seconds: float = 0.0,
        model_path: Optional[str] = None
    ) -> str:
        """
        Save experiment metadata and return unique experiment ID.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{task}_{architecture.lower()}_{timestamp}"
        target_dir = self._get_target_dir(architecture)
        file_path = target_dir / f"{exp_id}.json"

        data = {
            "experiment_id": exp_id,
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "architecture": architecture.upper(),
            "model_name": model_name,
            "model_path": str(model_path) if model_path else None,
            "duration_seconds": round(duration_seconds, 2),
            "hyperparameters": hyperparameters,
            "training_history": training_history,
            "evaluation_metrics": evaluation_metrics,
            "confusion_matrix": confusion_matrix,
            "classes": classes or []
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return exp_id

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Find and load an experiment by ID across experiment folders."""
        for target_dir in [ANN_EXPERIMENTS_DIR, CNN_EXPERIMENTS_DIR, COMPARISON_EXPERIMENTS_DIR, EXPERIMENTS_DIR]:
            candidate = target_dir / f"{exp_id}.json"
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    return None
        return None

    def list_experiments(
        self,
        task: Optional[str] = None,
        architecture: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all experiments matching filters, sorted newest first."""
        results = []
        target_dirs = []

        if architecture:
            target_dirs.append(self._get_target_dir(architecture))
        else:
            target_dirs.extend([ANN_EXPERIMENTS_DIR, CNN_EXPERIMENTS_DIR])

        for d in target_dirs:
            if not d.exists():
                continue
            for f in d.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                        if task and data.get("task") != task:
                            continue
                        # Summary entry
                        results.append({
                            "experiment_id": data.get("experiment_id", f.stem),
                            "timestamp": data.get("timestamp", ""),
                            "task": data.get("task", ""),
                            "architecture": data.get("architecture", ""),
                            "model_name": data.get("model_name", ""),
                            "test_accuracy": data.get("evaluation_metrics", {}).get("accuracy"),
                            "duration_seconds": data.get("duration_seconds", 0),
                            "epochs": len(data.get("training_history", {}).get("loss", []))
                        })
                except Exception:
                    continue

        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results

    def get_latest_experiment(self, task: str, architecture: str) -> Optional[Dict[str, Any]]:
        """Retrieve most recent experiment for a specific task and architecture."""
        exps = self.list_experiments(task=task, architecture=architecture)
        if not exps:
            return None
        return self.get_experiment(exps[0]["experiment_id"])

    def delete_experiment(self, exp_id: str) -> bool:
        """Remove an experiment file."""
        for target_dir in [ANN_EXPERIMENTS_DIR, CNN_EXPERIMENTS_DIR, COMPARISON_EXPERIMENTS_DIR, EXPERIMENTS_DIR]:
            candidate = target_dir / f"{exp_id}.json"
            if candidate.exists():
                candidate.unlink()
                return True
        return False


_experiment_store = None


def get_experiment_store() -> ExperimentStore:
    global _experiment_store
    if _experiment_store is None:
        _experiment_store = ExperimentStore()
    return _experiment_store
