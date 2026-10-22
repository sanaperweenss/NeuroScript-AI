"""
NeuroScript AI - Model Comparison Center
Enables direct, rigorous head-to-head comparison between ANN and CNN architectures
evaluated on identical held-out test splits.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from config.settings import (
    TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED,
    TASK_CHOICES, TASK_DESCRIPTIONS, ARCH_ANN, ARCH_CNN
)
from storage.experiment_store import get_experiment_store
from ui.components import render_confusion_matrix_interactive, render_empty_state


def render():
    st.markdown("## ⚖️ Model Comparison Center")
    st.markdown("<p class='muted-text'>Side-by-side empirical comparison between Feedforward ANN and Convolutional CNN models on identical test partitions.</p>", unsafe_allow_html=True)

    exp_store = get_experiment_store()

    # Task Selection
    task = st.selectbox(
        "Select Recognition Task to Compare",
        options=TASK_CHOICES,
        format_func=lambda x: TASK_DESCRIPTIONS[x],
        index=0
    )

    ann_experiments = exp_store.list_experiments(task=task, architecture=ARCH_ANN)
    cnn_experiments = exp_store.list_experiments(task=task, architecture=ARCH_CNN)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### 🧠 Artificial Neural Network (ANN)")
        if ann_experiments:
            ann_choices = {e["experiment_id"]: f"{e['model_name']} (Acc: {e.get('test_accuracy', 0)*100:.1f}%)" for e in ann_experiments}
            sel_ann_id = st.selectbox("Select ANN Experiment", options=list(ann_choices.keys()), format_func=lambda x: ann_choices[x], key="cmp_ann_sel")
            ann_data = exp_store.get_experiment(sel_ann_id)
        else:
            st.info("No ANN experiments available for this task.")
            ann_data = None

    with c2:
        st.markdown(f"#### 👁️ Convolutional Neural Network (CNN)")
        if cnn_experiments:
            cnn_choices = {e["experiment_id"]: f"{e['model_name']} (Acc: {e.get('test_accuracy', 0)*100:.1f}%)" for e in cnn_experiments}
            sel_cnn_id = st.selectbox("Select CNN Experiment", options=list(cnn_choices.keys()), format_func=lambda x: cnn_choices[x], key="cmp_cnn_sel")
            cnn_data = exp_store.get_experiment(sel_cnn_id)
        else:
            st.info("No CNN experiments available for this task.")
            cnn_data = None

    if not ann_data or not cnn_data:
        render_empty_state(
            title="Comparison Requires Both ANN and CNN Experiments",
            message="Train at least one ANN model and one CNN model for the selected task to unlock side-by-side metrics, confusion matrices, and convergence analysis.",
            icon="📊"
        )
        return

    st.markdown("---")

    # Metrics Summary Table
    ann_m = ann_data.get("evaluation_metrics", {})
    cnn_m = cnn_data.get("evaluation_metrics", {})

    ann_acc = ann_m.get("accuracy", 0.0) * 100.0
    cnn_acc = cnn_m.get("accuracy", 0.0) * 100.0

    ann_f1 = ann_m.get("macro_f1", 0.0) * 100.0
    cnn_f1 = cnn_m.get("macro_f1", 0.0) * 100.0

    ann_lat = ann_m.get("latency_ms_per_sample", 0.0)
    cnn_lat = cnn_m.get("latency_ms_per_sample", 0.0)

    ann_time = ann_data.get("duration_seconds", 0.0)
    cnn_time = cnn_data.get("duration_seconds", 0.0)

    summary_df = pd.DataFrame({
        "Metric": [
            "Test Accuracy (%)",
            "Macro Precision (%)",
            "Macro Recall (%)",
            "Macro F1 Score (%)",
            "Inference Latency (ms/char)",
            "Training Duration (s)",
            "Test Samples Evaluated"
        ],
        "ANN (Multilayer Perceptron)": [
            f"{ann_acc:.2f}%",
            f"{ann_m.get('macro_precision', 0.0)*100:.2f}%",
            f"{ann_m.get('macro_recall', 0.0)*100:.2f}%",
            f"{ann_f1:.2f}%",
            f"{ann_lat:.2f} ms",
            f"{ann_time:.1f} s",
            str(ann_m.get("total_test_samples", "N/A"))
        ],
        "CNN (Convolutional Network)": [
            f"{cnn_acc:.2f}%",
            f"{cnn_m.get('macro_precision', 0.0)*100:.2f}%",
            f"{cnn_m.get('macro_recall', 0.0)*100:.2f}%",
            f"{cnn_f1:.2f}%",
            f"{cnn_lat:.2f} ms",
            f"{cnn_time:.1f} s",
            str(cnn_m.get("total_test_samples", "N/A"))
        ],
        "Difference (CNN - ANN)": [
            f"{cnn_acc - ann_acc:+.2f}%",
            f"{(cnn_m.get('macro_precision', 0.0) - ann_m.get('macro_precision', 0.0))*100:+.2f}%",
            f"{(cnn_m.get('macro_recall', 0.0) - ann_m.get('macro_recall', 0.0))*100:+.2f}%",
            f"{cnn_f1 - ann_f1:+.2f}%",
            f"{cnn_lat - ann_lat:+.2f} ms",
            f"{cnn_time - ann_time:+.1f} s",
            "Identical Split"
        ]
    })

    st.markdown("### 📋 Empirical Metrics Matrix")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Bar Chart Metric Comparison
    st.markdown("### 📊 Classification Performance Comparison")
    metrics_names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    ann_vals = [ann_acc, ann_m.get("macro_precision", 0)*100, ann_m.get("macro_recall", 0)*100, ann_f1]
    cnn_vals = [cnn_acc, cnn_m.get("macro_precision", 0)*100, cnn_m.get("macro_recall", 0)*100, cnn_f1]

    fig_bar = go.Figure(data=[
        go.Bar(name="ANN (MLP)", x=metrics_names, y=ann_vals, marker_color="#7564F5"),
        go.Bar(name="CNN (Convolutional)", x=metrics_names, y=cnn_vals, marker_color="#42D9E8")
    ])
    fig_bar.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Score (%)", range=[min(min(ann_vals), min(cnn_vals)) * 0.9, 102], gridcolor="#191E38"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=320
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Convergence Curves Comparison
    ann_hist = ann_data.get("training_history", {})
    cnn_hist = cnn_data.get("training_history", {})

    if ann_hist.get("accuracy") and cnn_hist.get("accuracy"):
        st.markdown("### 📈 Convergence Trajectory")
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(y=ann_hist.get("loss", []), mode="lines", name="ANN Loss", line=dict(color="#7564F5", dash="dash")))
        fig_loss.add_trace(go.Scatter(y=ann_hist.get("val_loss", []), mode="lines", name="ANN Val Loss", line=dict(color="#7564F5")))
        fig_loss.add_trace(go.Scatter(y=cnn_hist.get("loss", []), mode="lines", name="CNN Loss", line=dict(color="#42D9E8", dash="dash")))
        fig_loss.add_trace(go.Scatter(y=cnn_hist.get("val_loss", []), mode="lines", name="CNN Val Loss", line=dict(color="#42D9E8")))

        fig_loss.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Epoch", gridcolor="#191E38"),
            yaxis=dict(title="Loss", gridcolor="#191E38"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # Side-by-side Confusion Matrices
    st.markdown("### 🧩 Confusion Matrix Analysis")
    cm_c1, cm_c2 = st.columns(2)
    classes = ann_m.get("classes", [])

    with cm_c1:
        if "confusion_matrix" in ann_m:
            render_confusion_matrix_interactive(ann_m["confusion_matrix"], classes, title=f"ANN Confusion Matrix ({task})")

    with cm_c2:
        if "confusion_matrix" in cnn_m:
            render_confusion_matrix_interactive(cnn_m["confusion_matrix"], classes, title=f"CNN Confusion Matrix ({task})")

    # Automated Scientific Synthesis
    st.markdown("### 📝 Research Synthesis & Conclusion")
    acc_diff = cnn_acc - ann_acc
    if acc_diff > 0.5:
        verdict = f"""
        **Empirical Finding:** The Convolutional Neural Network outperforms the Artificial Neural Network by **{acc_diff:.2f}%** test accuracy on the held-out split.
        - **Spatial Feature Representation:** The convolutional kernels successfully preserve 2D topology, capturing continuous strokes, loops, and intersections without distortion.
        - **Translation Invariance:** Pooling stages provide invariance against minor off-center offsets and stroke variations.
        - **Trade-off:** While CNN achieves superior accuracy, ANN exhibits {f'lower latency ({ann_lat:.2f} ms vs {cnn_lat:.2f} ms)' if ann_lat < cnn_lat else 'comparable latency'}.
        """
    elif acc_diff < -0.5:
        verdict = f"""
        **Empirical Finding:** The ANN achieved **{abs(acc_diff):.2f}%** higher accuracy on this particular experiment run.
        - This typically occurs when inputs are tightly centered and normalized, where dense linear combinations find clear separation boundaries.
        """
    else:
        verdict = f"""
        **Empirical Finding:** Both models performed within **{abs(acc_diff):.2f}%** of each other.
        - For well-centered, high-contrast character strokes, dense hidden layers provide competitive discriminative capacity relative to convolutional layers.
        """

    st.markdown(f"""
    <div class="neuro-card" style="border-left: 4px solid #42D9E8;">
        {verdict}
    </div>
    """, unsafe_allow_html=True)
