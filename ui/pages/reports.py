"""
NeuroScript AI - Reports & Export Center
Generates downloadable artifacts:
- Training convergence curves (CSV)
- Evaluation metrics and per-class reports (JSON)
- Confusion matrices (CSV and publication-ready PNG heatmap)
- Multi-page comprehensive PDF experiment evaluation report
"""

import streamlit as st
import pandas as pd
from storage.experiment_store import get_experiment_store
from reports.exporters import get_report_exporter
from ui.components import render_empty_state


def render():
    st.markdown("## 📑 Experiment Reports & Artifact Exporters")
    st.markdown("<p class='muted-text'>Generate and download structured CSV, JSON, PNG, and PDF documentation from genuine experiment runs.</p>", unsafe_allow_html=True)

    exp_store = get_experiment_store()
    exporter = get_report_exporter()

    all_experiments = exp_store.list_experiments()

    if not all_experiments:
        render_empty_state(
            title="No Experiment Runs Available for Reporting",
            message="Train an ANN or CNN model in the Training Center to generate metrics, convergence curves, and confusion matrix reports.",
            icon="📈"
        )
        return

    # Select Experiment
    exp_choices = {e["experiment_id"]: f"{e['model_name']} ({e['architecture']} - {e['task'].title()}) - Acc: {e.get('test_accuracy', 0)*100:.1f}%" for e in all_experiments}
    sel_id = st.selectbox("Select Experiment Run", options=list(exp_choices.keys()), format_func=lambda x: exp_choices[x])

    exp_data = exp_store.get_experiment(sel_id)
    if not exp_data:
        st.error("Failed to load experiment data.")
        return

    st.markdown("---")

    # Overview of Selected Experiment
    eval_m = exp_data.get("evaluation_metrics", {})
    hyp = exp_data.get("hyperparameters", {})
    hist = exp_data.get("training_history", {})
    cm = exp_data.get("confusion_matrix")
    classes = exp_data.get("classes", [])

    st.markdown(f"### 📋 Experiment: `{exp_data.get('model_name')}`")
    st.markdown(f"**Task:** {exp_data.get('task').title()} | **Architecture:** {exp_data.get('architecture')} | **Timestamp:** {exp_data.get('timestamp')}")

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("Test Accuracy", f"{eval_m.get('accuracy', 0)*100:.2f}%")
    with e2:
        st.metric("Macro F1", f"{eval_m.get('macro_f1', 0)*100:.2f}%")
    with e3:
        st.metric("Training Duration", f"{exp_data.get('duration_seconds', 0):.1f} s")
    with e4:
        st.metric("Epochs Completed", str(len(hist.get("loss", []))))

    st.markdown("---")

    # Downloadable Artifacts Grid
    st.markdown("### 📥 Download Artifacts")
    d1, d2 = st.columns(2)

    with d1:
        st.markdown("##### 1. Tabular Convergence Data")
        if hist.get("loss"):
            csv_path = exporter.export_training_history_csv(hist, sel_id)
            with open(csv_path, "rb") as fp:
                st.download_button(
                    label="📊 Download Training Curves (CSV)",
                    data=fp,
                    file_name=csv_path.name,
                    mime="text/csv",
                    use_container_width=True
                )

        st.markdown("##### 2. Evaluation Metrics")
        json_path = exporter.export_metrics_json(eval_m, sel_id)
        with open(json_path, "rb") as fp:
            st.download_button(
                label="⚙️ Download Metrics Summary (JSON)",
                data=fp,
                file_name=json_path.name,
                mime="application/json",
                use_container_width=True
            )

    with d2:
        st.markdown("##### 3. Confusion Matrix Artifacts")
        if cm and classes:
            cm_csv_path = exporter.export_confusion_matrix_csv(cm, classes, sel_id)
            with open(cm_csv_path, "rb") as fp:
                st.download_button(
                    label="📑 Download Confusion Matrix (CSV)",
                    data=fp,
                    file_name=cm_csv_path.name,
                    mime="text/csv",
                    use_container_width=True
                )

            cm_png_path = exporter.export_confusion_matrix_plot(
                cm, classes,
                title=f"{exp_data.get('model_name')} Confusion Matrix",
                filename_stem=sel_id
            )
            with open(cm_png_path, "rb") as fp:
                st.download_button(
                    label="🖼️ Download Confusion Matrix Plot (PNG)",
                    data=fp,
                    file_name=cm_png_path.name,
                    mime="image/png",
                    use_container_width=True
                )

    st.markdown("---")

    # PDF Report Generator
    st.markdown("### 📄 Formal Experiment Evaluation PDF")
    st.markdown("Generates a structured, printable evaluation document synthesizing architecture parameters, held-out metrics, and comparative findings.")

    if st.button("📑 Generate Formal PDF Report", type="primary", key="btn_gen_pdf"):
        with st.spinner("Compiling and generating PDF document..."):
            pdf_path = exporter.generate_pdf_report(
                task=exp_data.get("task"),
                ann_exp=exp_data if exp_data.get("architecture") == "ANN" else exp_store.get_latest_experiment(exp_data.get("task"), "ANN"),
                cnn_exp=exp_data if exp_data.get("architecture") == "CNN" else exp_store.get_latest_experiment(exp_data.get("task"), "CNN"),
                filename_stem=sel_id
            )
            if pdf_path and pdf_path.exists():
                st.session_state["latest_generated_pdf"] = str(pdf_path)
                st.success(f"Report compiled successfully: `{pdf_path.name}`")
            else:
                st.error("Failed to generate PDF report.")

    if "latest_generated_pdf" in st.session_state:
        from pathlib import Path
        cached_pdf = Path(st.session_state["latest_generated_pdf"])
        if cached_pdf.exists():
            with open(cached_pdf, "rb") as fp:
                st.download_button(
                    label=f"⬇️ Download Generated Report ({cached_pdf.name})",
                    data=fp.read(),
                    file_name=cached_pdf.name,
                    mime="application/pdf" if cached_pdf.suffix == ".pdf" else "text/markdown",
                    key="download_pdf_persisted"
                )

    st.markdown("---")

    # Full Project Word (.docx) Report
    st.markdown("### 📘 Comprehensive Project Architecture & Technical Report (Word .docx)")
    st.markdown(
        "A formal, publication-ready Microsoft Word document detailing mathematical foundations, "
        "LeCun data preprocessing algorithms, NIST orientation resolution, ANN vs. CNN architectures, "
        "empirical benchmark tables, confusion matrix analyses, software structure, and testing suites."
    )

    from pathlib import Path
    docx_file = Path("reports_store/NeuroScript_AI_Comprehensive_Project_Report.docx")
    if not docx_file.exists():
        docx_file = Path("NeuroScript_AI_Comprehensive_Project_Report.docx")

    if docx_file.exists():
        with open(docx_file, "rb") as fp:
            st.download_button(
                label="📥 Download Full Project Report (.docx)",
                data=fp.read(),
                file_name="NeuroScript_AI_Comprehensive_Project_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary",
                key="download_word_report_btn"
            )
    else:
        if st.button("🔨 Build Word (.docx) Report", type="primary"):
            from reports.generate_word_report import build_word_report
            build_word_report(docx_file)
            st.success("Word report generated! Click again to download.")
            st.rerun()
