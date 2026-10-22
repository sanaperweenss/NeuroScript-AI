"""
NeuroScript AI - Model Visualizer
Deep feature inspection and interpretability engine:
- Visualizes actual convolution filter kernels
- Extracts and renders real-time intermediate activation feature maps for CNN
- Displays ANN layer dimensions, weight distributions, and parameter flow
"""

import streamlit as st
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from config.settings import (
    TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED,
    TASK_CHOICES, TASK_DESCRIPTIONS, ARCH_ANN, ARCH_CNN
)
from models.model_registry import get_model_registry
from models.cnn import extract_layer_activations, extract_conv_weights
from data.loaders import load_dataset_for_task
from data.preprocessing import get_default_preprocessor
from ui.components import render_empty_state


def render():
    st.markdown("## 🔬 Neural Model Visualizer")
    st.markdown("<p class='muted-text'>Inspect internal representations: convolution kernels, intermediate feature maps, and dense weight distributions.</p>", unsafe_allow_html=True)

    reg = get_model_registry()

    col_t, col_a, col_m = st.columns([1.2, 1, 1.5])
    with col_t:
        task = st.selectbox("Task", options=TASK_CHOICES, format_func=lambda x: TASK_DESCRIPTIONS[x])
    with col_a:
        architecture = st.selectbox("Architecture", options=[ARCH_CNN, ARCH_ANN])

    models_list = reg.list_models(task=task, architecture=architecture)
    with col_m:
        if models_list:
            model_names = [m["model_name"] for m in models_list]
            selected_name = st.selectbox("Trained Model", options=model_names)
        else:
            selected_name = None
            st.selectbox("Trained Model", options=["No trained models"], disabled=True)

    if not selected_name:
        render_empty_state(
            title=f"No {architecture} Model Available to Visualize",
            message="Train a model in the Training Center first to extract and visualize its genuine internal weights and activations.",
            icon="🧠"
        )
        return

    # Load Model
    try:
        model, meta = reg.load_model(selected_name, architecture)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    st.markdown("---")

    # Select an input sample to pass through the network
    st.markdown("### 1. Select Input Sample Character")
    x_train, y_train, _, _ = load_dataset_for_task(task)
    classes = meta.get("classes", [])

    c_sel, c_prev = st.columns([2, 1])
    with c_sel:
        target_char_idx = st.slider("Browse Dataset Sample Index", min_value=0, max_value=min(500, len(x_train)-1), value=0)
        sample_img = x_train[target_char_idx]
        true_label = classes[y_train[target_char_idx]] if y_train[target_char_idx] < len(classes) else str(y_train[target_char_idx])

    with c_prev:
        st.image(Image.fromarray(sample_img).resize((100, 100), resample=Image.NEAREST), caption=f"Selected Input: '{true_label}'", width=100)

    # ---------------- CNN VISUALIZATION ----------------
    if architecture == ARCH_CNN:
        st.markdown("### 2. Intermediate Feature Maps (Layer Activations)")
        st.markdown("""
        Below are the genuine activation maps produced as the character stroke passes through convolutional and pooling layers.
        - **Block 1 (Conv2D & ReLU):** Responds to fine edge orientations, contrast gradients, and curvature.
        - **Block 1 (MaxPool):** Downsamples to 14x14, introducing spatial translation tolerance.
        - **Block 2 (Conv2D & ReLU):** Synthesizes stroke intersections, loops, and corners.
        - **Block 2 (MaxPool):** Downsamples to 7x7 high-level topological abstractions.
        """)

        # Prepare normalized tensor (1, 28, 28, 1)
        input_tensor = np.expand_dims(sample_img.astype(np.float32) / 255.0, axis=(0, -1))

        activations = extract_layer_activations(model, input_tensor)

        # Tabbed feature map viewer
        layer_tabs = st.tabs(["Conv2D Block 1 (32 Filters)", "MaxPool 1 (14×14)", "Conv2D Block 2 (64 Filters)", "MaxPool 2 (7×7)", "3×3 Convolution Kernels"])

        with layer_tabs[0]:
            if "conv2d_1" in activations:
                act = activations["conv2d_1"]  # Shape: (28, 28, 32)
                st.markdown("##### Filter Activations: Edge Detection & Stroke Orientation")
                cols = st.columns(8)
                for f_idx in range(min(16, act.shape[-1])):
                    with cols[f_idx % 8]:
                        f_map = act[:, :, f_idx]
                        # Normalize for display
                        f_norm = (f_map - f_map.min()) / (f_map.max() - f_map.min() + 1e-8) * 255
                        st.image(Image.fromarray(f_norm.astype(np.uint8)).resize((60, 60)), caption=f"F #{f_idx+1}", use_container_width=True)

        with layer_tabs[1]:
            if "maxpool_1" in activations:
                act = activations["maxpool_1"]  # (14, 14, 32)
                st.markdown("##### Downsampled Activations (14×14 Resolution)")
                cols = st.columns(8)
                for f_idx in range(min(16, act.shape[-1])):
                    with cols[f_idx % 8]:
                        f_map = act[:, :, f_idx]
                        f_norm = (f_map - f_map.min()) / (f_map.max() - f_map.min() + 1e-8) * 255
                        st.image(Image.fromarray(f_norm.astype(np.uint8)).resize((60, 60)), caption=f"Pool #{f_idx+1}", use_container_width=True)

        with layer_tabs[2]:
            if "conv2d_2" in activations:
                act = activations["conv2d_2"]  # (14, 14, 64)
                st.markdown("##### Higher-Level Feature Synthesis (Junctions & Loops)")
                cols = st.columns(8)
                for f_idx in range(min(16, act.shape[-1])):
                    with cols[f_idx % 8]:
                        f_map = act[:, :, f_idx]
                        f_norm = (f_map - f_map.min()) / (f_map.max() - f_map.min() + 1e-8) * 255
                        st.image(Image.fromarray(f_norm.astype(np.uint8)).resize((60, 60)), caption=f"F #{f_idx+1}", use_container_width=True)

        with layer_tabs[3]:
            if "maxpool_2" in activations:
                act = activations["maxpool_2"]  # (7, 7, 64)
                st.markdown("##### Compact Topological Feature Maps (7×7 Resolution)")
                cols = st.columns(8)
                for f_idx in range(min(16, act.shape[-1])):
                    with cols[f_idx % 8]:
                        f_map = act[:, :, f_idx]
                        f_norm = (f_map - f_map.min()) / (f_map.max() - f_map.min() + 1e-8) * 255
                        st.image(Image.fromarray(f_norm.astype(np.uint8)).resize((60, 60)), caption=f"Pool #{f_idx+1}", use_container_width=True)

        with layer_tabs[4]:
            st.markdown("##### Actual 3×3 Learned Convolution Kernels")
            weights = extract_conv_weights(model, "conv2d_1")
            if weights is not None:
                # Shape: (3, 3, 1, 32)
                cols = st.columns(8)
                for k_idx in range(min(16, weights.shape[-1])):
                    with cols[k_idx % 8]:
                        kernel = weights[:, :, 0, k_idx]
                        k_norm = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8) * 255
                        st.image(Image.fromarray(k_norm.astype(np.uint8)).resize((60, 60), resample=Image.NEAREST), caption=f"K #{k_idx+1}", use_container_width=True)

    # ---------------- ANN VISUALIZATION ----------------
    else:
        st.markdown("### 2. Artificial Neural Network Layer Topology")
        st.markdown("""
        An ANN vectorizes the 2D image into 784 individual pixel inputs, feeding into fully connected hidden layers:
        `Input(784) → Dense(256) → ReLU → Dense(128) → ReLU → Dense(Classes) → Softmax`
        """)

        # Layer parameters breakdown
        layers_data = []
        for l in model.layers:
            layers_data.append({
                "Layer Name": l.name,
                "Type": l.__class__.__name__,
                "Output Dimension": str(l.output_shape),
                "Parameters": l.count_params()
            })

        st.dataframe(pd.DataFrame(layers_data), use_container_width=True, hide_index=True)

        # Weight distribution histogram
        all_weights = []
        for l in model.layers:
            w = l.get_weights()
            if w:
                all_weights.append(w[0].flatten())

        if all_weights:
            flat_weights = np.concatenate(all_weights)
            st.markdown("##### Global Weight Distribution (ReLU & L2 Regularization)")
            fig = px.histogram(flat_weights, nbins=60, labels={"value": "Weight Magnitude"}, height=260)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                xaxis=dict(gridcolor="#191E38"),
                yaxis=dict(gridcolor="#191E38")
            )
            st.plotly_chart(fig, use_container_width=True)
