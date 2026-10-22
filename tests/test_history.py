"""
Unit Tests - SQLite Prediction History Storage
Tests database table initialization, record insertion, querying, and deletion.
"""

import tempfile
from pathlib import Path
from storage.history import PredictionHistoryDB


def test_history_database_lifecycle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_predictions.db"
        db = PredictionHistoryDB(db_path=db_path)

        assert db.count() == 0

        # Insert record
        pred_id = db.add_prediction(
            task="digits",
            model_name="cnn_mnist_v1",
            architecture="CNN",
            predicted_class="7",
            confidence=0.985,
            top_k=[{"class": "7", "probability": 0.985}, {"class": "1", "probability": 0.010}],
            source_type="canvas"
        )
        assert pred_id == 1
        assert db.count() == 1

        # Query records
        records = db.get_predictions(limit=10)
        assert len(records) == 1
        assert records[0]["predicted_class"] == "7"
        assert records[0]["confidence"] == 0.985
        assert len(records[0]["top_k"]) == 2

        # Filter by task
        digit_records = db.get_predictions(task_filter="digits")
        assert len(digit_records) == 1
        letter_records = db.get_predictions(task_filter="alphabet")
        assert len(letter_records) == 0

        # Export to dataframe
        df = db.to_dataframe()
        assert len(df) == 1
        assert "predicted_class" in df.columns

        # Delete single record
        deleted = db.delete_prediction(pred_id)
        assert deleted is True
        assert db.count() == 0
