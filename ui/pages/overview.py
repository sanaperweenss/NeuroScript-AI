"""
NeuroScript AI - Overview Dashboard
Landing page showcasing real dataset and model states, visual workflow,
recent prediction audit metrics, and quick navigation actions.
"""

import streamlit as st
from config.settings import (
    APP_TITLE, APP_TAGLINE, TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED,
    ARCH_ANN, ARCH_CNN
)
from ui.components import render_metric_card, render_workflow_diagram, render_status_badge
from data.loaders import get_dataset_stats
from models.model_registry import get_model_registry
from storage.history import get_history_db
from storage.experiment_store import get_experiment_store


def render():
    st.markdown(f"# 🧠 {APP_TITLE}")
    st.markdown(f"<p class='muted-text' style='font-size: 1.1rem; margin-top: -0.5rem;'>{APP_TAGLINE}</p>", unsafe_allow_html=True)

    # Visual Workflow
    render_workflow_diagram()

    # System Status Metrics
    reg = get_model_registry()
    db = get_history_db()
    exp_store = get_experiment_store()

    mnist_stats = get_dataset_stats(TASK_DIGITS)
    emnist_stats = get_dataset_stats(TASK_ALPHABET)
    all_models = reg.list_models()
    total_preds = db.count()
    all_experiments = exp_store.list_experiments()

    best_ann = exp_store.get_latest_experiment(TASK_DIGITS, ARCH_ANN)
    best_cnn = exp_store.get_latest_experiment(TASK_DIGITS, ARCH_CNN)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card(
            label="Trained Models",
            value=str(len(all_models)),
            subtitle=f"{sum(1 for m in all_models if m.get('architecture') == ARCH_ANN)} ANN | {sum(1 for m in all_models if m.get('architecture') == ARCH_CNN)} CNN",
            accent_color="#7564F5"
        )

    with col2:
        render_metric_card(
            label="Audit History",
            value=str(total_preds),
            subtitle="Logged inferences in SQLite",
            accent_color="#42D9E8"
        )

    with col3:
        best_acc_str = "None"
        if all_experiments:
            accs = [e.get("test_accuracy") for e in all_experiments if e.get("test_accuracy") is not None]
            if accs:
                best_acc_str = f"{max(accs) * 100:.1f}%"
        render_metric_card(
            label="Peak Benchmark",
            value=best_acc_str,
            subtitle="Verified held-out test set",
            accent_color="#36B37E"
        )

    with col4:
        ds_count = sum([1 for s in [mnist_stats, emnist_stats] if s.get("is_available")])
        render_metric_card(
            label="Active Datasets",
            value=f"{ds_count} / 2",
            subtitle="MNIST & EMNIST Letters",
            accent_color="#F5B942"
        )

    # Core Features & Tasks Overview
    st.markdown("### 🎯 Supported Recognition Tasks")
    t1, t2, t3 = st.columns(3)

    with t1:
        st.markdown("""
        <div class="neuro-card neuro-card-accent">
            <h4 style="margin-top:0;">Task A: Digit Recognition</h4>
            <div class="muted-text">Standard MNIST benchmark spanning 10 numerical classes (0–9). High baseline accuracy with dense neural networks; state-of-the-art precision with convolutional filters.</div>
            <div style="margin-top: 1rem;">
                <span class="muted-text" style="font-size:0.8rem;">Classes: </span><code>0 - 9</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
        <div class="neuro-card neuro-card-cyan">
            <h4 style="margin-top:0;">Task B: Alphabet Recognition</h4>
            <div class="muted-text">EMNIST Letters dataset covering 26 English uppercase characters (A–Z). Verified NIST coordinate transposition and 0-indexed label normalization.</div>
            <div style="margin-top: 1rem;">
                <span class="muted-text" style="font-size:0.8rem;">Classes: </span><code>A - Z</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
        <div class="neuro-card" style="border-left: 4px solid #36B37E;">
            <h4 style="margin-top:0;">Task C: Combined 36-Class</h4>
            <div class="muted-text">Unified classification covering digits (0–9) and uppercase letters (A–Z). Class space merges MNIST and EMNIST with verified balanced class distributions.</div>
            <div style="margin-top: 1rem;">
                <span class="muted-text" style="font-size:0.8rem;">Classes: </span><code>0-9 & A-Z</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Architectural Contrast Section
    st.markdown("### 🔬 Why CNNs Excel for Handwriting")
    with st.expander("Explore the Deep Learning Theory: Dense MLP (ANN) vs Convolutional (CNN)", expanded=False):
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("""
            **Artificial Neural Network (ANN / MLP)**
            - **Input Vectorization:** Flattens the $28 \\times 28$ matrix into a $1 \\times 784$ vector, discarding 2D spatial adjacency.
            - **Parameter Density:** Every input pixel connects to every neuron in hidden layer 1 ($784 \\times 256 = 200,704$ weights), creating susceptibility to overfitting.
            - **Translation Sensitivity:** A stroke shifted 2 pixels to the left activates completely disjoint weights, requiring extensive training data to generalize across positions.
            """)
        with c_b:
            st.markdown("""
            **Convolutional Neural Network (CNN)**
            - **Local Receptive Fields:** $3 \\times 3$ sliding kernels inspect localized pixel neighborhoods to extract primary stroke edges, junctions, and curvature.
            - **Weight Sharing:** The same kernel slides across the entire image, drastically reducing parameter counts while enforcing shift and translation invariance.
            - **Hierarchical Pooling:** Max-pooling layers progressively downsample spatial resolution, enabling deeper layers to learn holistic topological character structures.
            """)

    # Recent Audit Log / Recent Predictions
    st.markdown("### 🕒 Recent Inferences")
    recent = db.get_predictions(limit=5)
    if recent:
        for r in recent:
            top_classes = ", ".join([f"{k.get('class')} ({k.get('percentage')})" for k in r.get("top_k", [])[:3]])
            st.markdown(f"""
            <div class="neuro-card" style="padding: 0.85rem 1.25rem; margin-bottom: 0.6rem; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: 700; font-size: 1.1rem; color: #42D9E8;">Class: {r.get('predicted_class')}</span>
                    <span class="muted-text" style="margin-left: 1rem; font-size: 0.85rem;">Confidence: {r.get('confidence') * 100:.1f}%</span>
                    <span class="muted-text" style="margin-left: 1rem; font-size: 0.85rem;">Task: {r.get('task').title()}</span>
                    <span class="muted-text" style="margin-left: 1rem; font-size: 0.85rem;">Model: {r.get('model_name')}</span>
                </div>
                <div class="muted-text" style="font-size: 0.8rem;">
                    Top alternatives: {top_classes}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No inference history recorded yet. Open the **Recognize** page to draw or upload characters.")
