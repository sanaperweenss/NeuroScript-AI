"""
Smoke Tests - UI Pages and Components Import & Execution
Ensures all page modules can be loaded, and their supporting loaders and functions execute cleanly.
"""

from config.settings import TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED
from data.loaders import get_dataset_stats
import data.loaders as loaders
from models.model_registry import get_model_registry
from storage.history import get_history_db
from storage.experiment_store import get_experiment_store


def test_dataset_stats_all_tasks():
    for task in [TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED]:
        stats = loaders.get_dataset_stats(task)
        assert isinstance(stats, dict)
        assert "is_available" in stats
        assert "task" in stats


def test_all_pages_importable():
    from ui.pages import (
        overview,
        recognize,
        model_comparison,
        training_center,
        dataset_explorer,
        model_visualizer,
        prediction_history,
        reports,
        settings as settings_page
    )
    assert callable(overview.render)
    assert callable(recognize.render)
    assert callable(model_comparison.render)
    assert callable(training_center.render)
    assert callable(dataset_explorer.render)
    assert callable(model_visualizer.render)
    assert callable(prediction_history.render)
    assert callable(reports.render)
    assert callable(settings_page.render)
