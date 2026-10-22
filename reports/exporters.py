"""
NeuroScript AI - Report and Metric Exporters
Provides export capabilities for:
- Training history (CSV, JSON)
- Evaluation metrics and confusion matrices (CSV, JSON, PNG)
- Model comparison reports (CSV, JSON, PDF)
- Prediction audit history (CSV)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
from config.settings import REPORTS_DIR


class ReportExporter:
    """
    Handles report serialization, chart generation, and tabular downloads.
    """

    def __init__(self, export_dir: Path = REPORTS_DIR):
        self.export_dir = export_dir
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_training_history_csv(self, history: Dict[str, List[float]], filename_stem: str) -> Path:
        """Export epoch-by-epoch loss and accuracy metrics to CSV."""
        df = pd.DataFrame(history)
        df.index.name = "epoch"
        df.index += 1  # 1-indexed epochs
        out_path = self.export_dir / f"{filename_stem}_history.csv"
        df.to_csv(out_path)
        return out_path

    def export_metrics_json(self, metrics: Dict[str, Any], filename_stem: str) -> Path:
        """Export evaluation metrics to structured JSON."""
        out_path = self.export_dir / f"{filename_stem}_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        return out_path

    def export_confusion_matrix_csv(
        self,
        cm: List[List[int]],
        classes: List[str],
        filename_stem: str
    ) -> Path:
        """Export confusion matrix to CSV with row/column headers."""
        df = pd.DataFrame(cm, index=classes, columns=classes)
        out_path = self.export_dir / f"{filename_stem}_confusion_matrix.csv"
        df.to_csv(out_path)
        return out_path

    def export_confusion_matrix_plot(
        self,
        cm: List[List[int]],
        classes: List[str],
        title: str,
        filename_stem: str
    ) -> Path:
        """Render and save confusion matrix heatmap as PNG."""
        cm_arr = np.array(cm)
        num_classes = len(classes)

        fig_size = max(6, min(14, num_classes * 0.4))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=150)
        fig.patch.set_facecolor("#101426")
        ax.set_facecolor("#101426")

        cax = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
        fig.colorbar(cax, fraction=0.046, pad=0.04)

        ax.set_title(title, color="#F6F7FB", fontsize=12, pad=12)
        ax.set_xlabel("Predicted Label", color="#9AA5C0", fontsize=10, labelpad=8)
        ax.set_ylabel("True Label", color="#9AA5C0", fontsize=10, labelpad=8)

        tick_marks = np.arange(num_classes)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, color="#F6F7FB", fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, color="#F6F7FB", fontsize=8)

        # Draw cell values if matrix is not too dense
        if num_classes <= 15:
            thresh = cm_arr.max() / 2.0
            for i in range(cm_arr.shape[0]):
                for j in range(cm_arr.shape[1]):
                    ax.text(
                        j, i, format(cm_arr[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm_arr[i, j] > thresh else "#101426",
                        fontsize=7
                    )

        plt.tight_layout()
        out_path = self.export_dir / f"{filename_stem}_cm.png"
        plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        return out_path

    def generate_pdf_report(
        self,
        task: str,
        ann_exp: Optional[Dict[str, Any]],
        cnn_exp: Optional[Dict[str, Any]],
        filename_stem: str
    ) -> Optional[Path]:
        """
        Generate a multi-page PDF summary report using fpdf2.
        Falls back to Markdown report if fpdf2 fails or is unavailable.
        """
        try:
            from fpdf import FPDF

            class NeuroReport(FPDF):
                def header(self):
                    self.set_font("Helvetica", "B", 14)
                    self.cell(0, 10, "NeuroScript AI - Experiment Evaluation Report", 0, 1, "C")
                    self.ln(2)

                def footer(self):
                    self.set_y(-15)
                    self.set_font("Helvetica", "I", 8)
                    self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

            pdf = NeuroReport()
            pdf.add_page()
            pdf.set_font("Helvetica", "", 10)

            # Metadata
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, f"Task: {task.upper()} RECOGNITION", 0, 1)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 6, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
            pdf.ln(4)

            # ANN Section
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "1. Artificial Neural Network (ANN) Performance:", 0, 1)
            pdf.set_font("Helvetica", "", 9)
            if ann_exp:
                ann_m = ann_exp.get("evaluation_metrics", {})
                pdf.cell(0, 5, f"- Model Name: {ann_exp.get('model_name', 'N/A')}", 0, 1)
                pdf.cell(0, 5, f"- Test Accuracy: {ann_m.get('accuracy', 0.0) * 100:.2f}%", 0, 1)
                pdf.cell(0, 5, f"- Macro Precision: {ann_m.get('macro_precision', 0.0) * 100:.2f}%", 0, 1)
                pdf.cell(0, 5, f"- Macro Recall: {ann_m.get('macro_recall', 0.0) * 100:.2f}%", 0, 1)
                pdf.cell(0, 5, f"- Macro F1 Score: {ann_m.get('macro_f1', 0.0) * 100:.2f}%", 0, 1)
                pdf.cell(0, 5, f"- Training Duration: {ann_exp.get('duration_seconds', 0.0):.1f} s", 0, 1)
            else:
                pdf.cell(0, 5, "No completed ANN experiment recorded for this task.", 0, 1)
            pdf.ln(4)

            # CNN Section
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "2. Convolutional Neural Network (CNN) Performance:", 0, 1)
            pdf.set_font("Helvetica", "", 9)
            if cnn_exp:
                cnn_m = cnn_exp.get("evaluation_metrics", {})
                pdf.cell(0, 5, f"- Model Name: {cnn_exp.get('model_name', 'N/A')}", 0, 1)
                pdf.cell(0, 5, f"- Test Accuracy: {cnn_m.get('accuracy', 0.0) * 100:.2f}%", 0, 1)
                pdf.cell(0, 5, f"- Macro Precision: {cnn_m.get('macro_precision', 0.0) * 100:.2f}%", 0, 1)
                pdf.cell(0, 5, f"- Macro Recall: {cnn_m.get('macro_recall', 0.0) * 100:.2f}%", 0, 1)
                pdf.cell(0, 5, f"- Macro F1 Score: {cnn_m.get('macro_f1', 0.0) * 100:.2f}%", 0, 1)
                pdf.cell(0, 5, f"- Training Duration: {cnn_exp.get('duration_seconds', 0.0):.1f} s", 0, 1)
            else:
                pdf.cell(0, 5, "No completed CNN experiment recorded for this task.", 0, 1)
            pdf.ln(4)

            # Comparative Analysis
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "3. Synthesis and Comparative Analysis:", 0, 1)
            pdf.set_font("Helvetica", "", 9)
            if ann_exp and cnn_exp:
                ann_acc = ann_exp.get("evaluation_metrics", {}).get("accuracy", 0.0)
                cnn_acc = cnn_exp.get("evaluation_metrics", {}).get("accuracy", 0.0)
                diff = (cnn_acc - ann_acc) * 100
                if diff > 0:
                    analysis = (
                        f"The CNN outperformed the ANN by {diff:.2f}% test accuracy. "
                        "The spatial convolutions and pooling hierarchies effectively capture translation invariance "
                        "and local stroke patterns, which fully-connected flat layers struggle to model."
                    )
                else:
                    analysis = (
                        f"The ANN performed competitively (within {abs(diff):.2f}% of CNN). "
                        "This indicates that for clean, centered digit/letter inputs, dense linear combinations "
                        "can achieve respectable discriminative boundaries."
                    )
                pdf.multi_cell(0, 5, analysis)
            else:
                pdf.cell(0, 5, "Complete both ANN and CNN training runs to enable comparative synthesis.", 0, 1)

            out_pdf = self.export_dir / f"{filename_stem}_report.pdf"
            pdf.output(str(out_pdf))
            return out_pdf

        except Exception as e:
            # Fallback to Markdown
            out_md = self.export_dir / f"{filename_stem}_report.md"
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(f"# NeuroScript AI - Experiment Report\nTask: {task}\n")
            return out_md


_report_exporter = None


def get_report_exporter() -> ReportExporter:
    global _report_exporter
    if _report_exporter is None:
        _report_exporter = ReportExporter()
    return _report_exporter
