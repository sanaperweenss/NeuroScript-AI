"""
NeuroScript AI - Model Training Center
Interactive model training dashboard with real-time epoch telemetry,
live convergence curves, test evaluation metrics, and automatic checkpoint persistence.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from config.settings import (
    TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED,
    TASK_CHOICES, TASK_DESCRIPTIONS, ARCH_ANN, ARCH_CNN, ARCH_CHOICES,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_VALIDATION_SPLIT
)
from training.trainer import get_model_trainer
from ui.components import render_confusion_matrix_interactive, render_metric_card


def render():
    st.markdown("## 🏋️ Model Training Center")
    st.markdown("<p class='muted-text'>Configure hyperparameters and train genuine ANN or CNN models using real labeled datasets.</p>", unsafe_allow_html=True)

    trainer = get_model_trainer()

    # Training Configuration Form
    with st.form("training_config_form"):
        st.markdown("#### 1. Task & Architecture")
        c1, c2 = st.columns(2)
        with c1:
            task = st.selectbox(
                "Target Recognition Task",
                options=TASK_CHOICES,
                format_func=lambda x: TASK_DESCRIPTIONS[x]
            )
        with c2:
            architecture = st.selectbox(
                "Neural Network Architecture",
                options=ARCH_CHOICES,
                index=1
            )

        st.markdown("#### 2. Hyperparameters")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            epochs = st.slider("Epochs", min_value=1, max_value=25, value=5, step=1)
        with h2:
            batch_size = st.selectbox("Batch Size", options=[32, 64, 128, 256], index=1)
        with h3:
            learning_rate = st.selectbox("Learning Rate", options=[0.01, 0.005, 0.001, 0.0005, 0.0001], index=2)
        with h4:
            val_split = st.slider("Validation Split", min_value=0.10, max_value=0.25, value=0.15, step=0.05)

        st.markdown("#### 3. Execution & Optimization")
        o1, o2 = st.columns(2)
        with o1:
            max_samples = st.selectbox(
                "Training Samples (CPU Optimization)",
                options=[5000, 10000, 20000, 60000],
                index=1,
                help="Subsampling enables swift training on standard CPUs while retaining high accuracy."
            )
        with o2:
            custom_model_name = st.text_input(
                "Model Name (Optional)",
                placeholder=f"{task}_{architecture.lower()}_v1",
                help="Leave blank to auto-generate a timestamped name."
            )

        submit_training = st.form_submit_button("🚀 Start Model Training", type="primary", use_container_width=True)

    # Execution Container
    if submit_training:
        st.markdown("---")
        st.markdown("### 📊 Live Training Telemetry")

        progress_bar = st.progress(0.0)
        status_text = st.empty()
        metric_cols_placeholder = st.empty()
        chart_placeholder = st.empty()

        history_data = {
            "epoch": [],
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        def on_epoch_update(epoch, logs):
            progress_bar.progress(epoch / float(epochs))
            status_text.markdown(f"**Epoch {epoch} / {epochs}** — Loss: `{logs['loss']:.4f}` | Accuracy: `{logs['accuracy']*100:.2f}%` | Val Loss: `{logs['val_loss']:.4f}` | Val Accuracy: `{logs['val_accuracy']*100:.2f}%`")

            history_data["epoch"].append(epoch)
            history_data["loss"].append(logs["loss"])
            history_data["accuracy"].append(logs["accuracy"] * 100.0)
            history_data["val_loss"].append(logs["val_loss"])
            history_data["val_accuracy"].append(logs["val_accuracy"] * 100.0)

            # Live Plotly chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history_data["epoch"], y=history_data["accuracy"], mode="lines+markers", name="Train Accuracy %", line=dict(color="#42D9E8", width=2)))
            fig.add_trace(go.Scatter(x=history_data["epoch"], y=history_data["val_accuracy"], mode="lines+markers", name="Val Accuracy %", line=dict(color="#7564F5", width=2)))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                height=260,
                xaxis=dict(title="Epoch", gridcolor="#191E38"),
                yaxis=dict(title="Accuracy (%)", gridcolor="#191E38"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"live_chart_{epoch}")

        try:
            status_text.info("Initializing dataset tensors and compiling model graph...")
            trained_model, eval_metrics, exp_id = trainer.train(
                task=task,
                architecture=architecture,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_split=val_split,
                model_name=custom_model_name if custom_model_name.strip() else None,
                progress_callback=on_epoch_update,
                max_train_samples=max_samples
            )

            progress_bar.progress(1.0)
            status_text.success(f"🎉 Training complete! Checkpoint saved: `{exp_id}`")

            # Held-out Evaluation Results
            st.markdown("### 🏆 Held-out Test Set Evaluation")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric_card("Test Accuracy", f"{eval_metrics['accuracy'] * 100:.2f}%", "Zero data leakage", "#36B37E")
            with m2:
                render_metric_card("Macro F1", f"{eval_metrics['macro_f1'] * 100:.2f}%", "Balanced across classes", "#42D9E8")
            with m3:
                render_metric_card("Latency", f"{eval_metrics['latency_ms_per_sample']:.2f} ms", "Avg per character", "#7564F5")
            with m4:
                render_metric_card("Test Samples", str(eval_metrics["total_test_samples"]), f"{eval_metrics['architecture']} Model", "#F5B942")

            # Confusion Matrix
            if "confusion_matrix" in eval_metrics:
                st.markdown("#### Test Set Confusion Matrix")
                render_confusion_matrix_interactive(
                    cm=eval_metrics["confusion_matrix"],
                    classes=eval_metrics["classes"],
                    title=f"{architecture} Confusion Matrix ({task.title()})"
                )

        except Exception as e:
            st.error(f"Training failed: {e}")
