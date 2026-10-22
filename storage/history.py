"""
NeuroScript AI - Prediction History Storage
Provides persistent SQLite storage for prediction audit logs, confidence scores,
input source types, and optional user-retained thumbnails.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from config.settings import HISTORY_DB_PATH, HISTORY_DIR


class PredictionHistoryDB:
    """
    SQLite wrapper for managing prediction history and audit tracking.
    """

    def __init__(self, db_path=HISTORY_DB_PATH):
        self.db_path = db_path
        self._ensure_dir()
        self._init_db()

    def _ensure_dir(self):
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    from contextlib import contextmanager

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    task TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    architecture TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    top_k_json TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    thumbnail_base64 TEXT
                )
            """)
            conn.commit()

    def add_prediction(
        self,
        task: str,
        model_name: str,
        architecture: str,
        predicted_class: str,
        confidence: float,
        top_k: List[Dict[str, Any]],
        source_type: str = "canvas",
        thumbnail_base64: Optional[str] = None
    ) -> int:
        """Record a single prediction event."""
        now_iso = datetime.now().isoformat()
        top_k_str = json.dumps(top_k)

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    timestamp, task, model_name, architecture,
                    predicted_class, confidence, top_k_json,
                    source_type, thumbnail_base64
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_iso, task, model_name, architecture,
                str(predicted_class), float(confidence), top_k_str,
                source_type, thumbnail_base64
            ))
            conn.commit()
            return cursor.lastrowid

    def get_predictions(
        self,
        limit: int = 100,
        task_filter: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve recent predictions with optional filtering."""
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []

        if task_filter and task_filter.lower() != "all":
            query += " AND task = ?"
            params.append(task_filter)

        if search_query:
            query += " AND (predicted_class LIKE ? OR model_name LIKE ? OR source_type LIKE ?)"
            wildcard = f"%{search_query}%"
            params.extend([wildcard, wildcard, wildcard])

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                try:
                    item["top_k"] = json.loads(item["top_k_json"])
                except Exception:
                    item["top_k"] = []
                results.append(item)
            return results

    def get_prediction_by_id(self, pred_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific prediction record."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,))
            row = cursor.fetchone()
            if not row:
                return None
            item = dict(row)
            try:
                item["top_k"] = json.loads(item["top_k_json"])
            except Exception:
                item["top_k"] = []
            return item

    def delete_prediction(self, pred_id: int) -> bool:
        """Delete an individual prediction record."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_history(self) -> int:
        """Clear all prediction history."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions")
            conn.commit()
            return cursor.rowcount

    def count(self) -> int:
        """Return total count of recorded predictions."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM predictions")
            return cursor.fetchone()[0]

    def to_dataframe(self, limit: int = 500) -> pd.DataFrame:
        """Export history to a pandas DataFrame."""
        records = self.get_predictions(limit=limit)
        if not records:
            return pd.DataFrame()
        # Drop raw JSON string and heavy base64 for tabular display
        df = pd.DataFrame(records)
        if "thumbnail_base64" in df.columns:
            df["has_thumbnail"] = df["thumbnail_base64"].apply(lambda x: bool(x))
            df = df.drop(columns=["thumbnail_base64"])
        if "top_k_json" in df.columns:
            df = df.drop(columns=["top_k_json"])
        return df


# Singleton instance
_history_db = None


def get_history_db() -> PredictionHistoryDB:
    global _history_db
    if _history_db is None:
        _history_db = PredictionHistoryDB()
    return _history_db
