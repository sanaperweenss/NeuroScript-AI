"""
NeuroScript AI - Dataset Explorer
Interactive data inspection center displaying real dataset partitions,
class frequency distributions, authentic image sample grids, and preprocessing stages.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from config.settings import (
    TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED,
    TASK_CHOICES, TASK_DESCRIPTIONS
)
from data.loaders import get_dataset_stats, get_class_samples, load_dataset_for_task
from data.preprocessing import get_default_preprocessor
from ui.components import render_metric_card, render_empty_state


def render():
    st.markdown("## 🔍 Dataset Explorer")
    st.markdown("<p class='muted-text'>Inspect genuine labeled training and test splits, class distributions, and sample character strokes.</p>", unsafe_allow_html=True)

    task = st.selectbox(
        "Select Dataset",
        options=TASK_CHOICES,
        format_func=lambda x: TASK_DESCRIPTIONS[x],
        index=0
    )

    stats = get_dataset_stats(task)

    # Dataset Status
    if not stats.get("is_available"):
        render_empty_state(
            title=f"{task.title()} Dataset Not Initialized",
            message="This dataset has not yet been loaded or cached in local storage. Click below to load and cache verified samples.",
            icon="📦"
        )
        if st.button("📥 Load & Cache Dataset", type="primary"):
            with st.spinner("Downloading and processing dataset partitions..."):
                try:
                    load_dataset_for_task(task, allow_download=True)
                    st.success("✅ Dataset cached successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to initialize dataset: {e}")
        return

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric_card("Train Samples", f"{stats['train_samples']:,}", "Training partition", "#7564F5")
    with m2:
        render_metric_card("Test Samples", f"{stats['test_samples']:,}", "Held-out test partition", "#42D9E8")
    with m3:
        render_metric_card("Class Count", str(stats["num_classes"]), f"{task.title()} space", "#36B37E")
    with m4:
        render_metric_card("Tensor Shape", "28 × 28 × 1", "Grayscale ubyte", "#F5B942")

    st.markdown("---")

    # Tab 1: Class Distribution
    # Tab 2: Sample Image Gallery
    # Tab 3: Preprocessing Pipeline Verification
    tab_dist, tab_samples, tab_prep = st.tabs([
        "📊 Class Distribution",
        "🖼️ Authentic Sample Gallery",
        "🔬 Preprocessing Pipeline Verification"
    ])

    with tab_dist:
        st.markdown("#### Class Distribution (Train Split)")
        counts = stats.get("class_counts", {})
        if counts:
            dist_df = pd.DataFrame(list(counts.items()), columns=["Class", "Sample Count"])
            fig = px.bar(
                dist_df,
                x="Class",
                y="Sample Count",
                color="Sample Count",
                color_continuous_scale=["#7564F5", "#42D9E8"],
                height=320
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                xaxis=dict(gridcolor="#191E38"),
                yaxis=dict(gridcolor="#191E38")
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_samples:
        st.markdown("#### Real Labeled Image Samples")
        samples_per_class = st.slider("Samples to Display per Class", min_value=1, max_value=3, value=1)
        with st.spinner("Extracting sample images from dataset partition..."):
            samples_dict = get_class_samples(task, samples_per_class=samples_per_class)

        # Render in a responsive grid
        classes = stats["classes"]
        cols_per_row = 10 if len(classes) <= 10 else 13
        rows = [classes[i:i + cols_per_row] for i in range(0, len(classes), cols_per_row)]

        for row in rows:
            grid_cols = st.columns(len(row))
            for idx, c_name in enumerate(row):
                with grid_cols[idx]:
                    img_list = samples_dict.get(c_name, [])
                    if img_list:
                        # Enlarge for sharp display
                        img_arr = img_list[0]
                        disp_img = Image.fromarray(img_arr).resize((70, 70), resample=Image.NEAREST)
                        st.image(disp_img, caption=f"'{c_name}'", use_container_width=True)

    with tab_prep:
        st.markdown("#### LeCun Standardization Pipeline Visualizer")
        st.markdown("""
        Demonstrates the exact mathematical transformations applied to each input:
        1. **Grayscale conversion & Polarity inversion** (stroke=white, background=black)
        2. **Content bounding box extraction** (crops out empty borders)
        3. **Aspect-ratio scaling** (scales major dimension to 20px)
        4. **Center-of-mass canvas centering** (places inside 28x28 black canvas)
        """)

        if st.button("🎲 Inspect Random Sample Pipeline"):
            x_train, y_train, _, _ = load_dataset_for_task(task)
            rand_idx = np.random.randint(0, len(x_train))
            sample_img = x_train[rand_idx]
            sample_lbl = stats["classes"][y_train[rand_idx]]

            prep = get_default_preprocessor()
            _, meta = prep.preprocess_image(sample_img)

            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.image(Image.fromarray(sample_img).resize((140, 140), resample=Image.NEAREST), caption=f"Raw Input ('{sample_lbl}')")
            with p2:
                if meta.get("polarity_adjusted") is not None:
                    st.image(Image.fromarray(meta["polarity_adjusted"]).resize((140, 140), resample=Image.NEAREST), caption="Polarity Adjusted")
            with p3:
                cropped = prep.extract_bounding_box(meta["polarity_adjusted"])
                if cropped is not None:
                    st.image(Image.fromarray(cropped).resize((140, 140), resample=Image.NEAREST), caption="Extracted Bounding Box")
            with p4:
                if meta.get("normalized_28x28") is not None:
                    st.image(Image.fromarray(meta["normalized_28x28"]).resize((140, 140), resample=Image.NEAREST), caption="Final 28×28 Centered")
