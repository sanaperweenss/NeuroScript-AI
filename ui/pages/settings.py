"""
NeuroScript AI - System Settings, Diagnostics & About
Displays environment telemetry, GPU/CPU hardware acceleration status,
dataset licensing, cache controls, and technical limitations.
"""

import sys
import shutil
import streamlit as st
from config.settings import (
    APP_TITLE, APP_TAGLINE, APP_VERSION,
    DATA_DIR, MODELS_DIR, EXPERIMENTS_DIR, HISTORY_DIR, REPORTS_DIR
)
from storage.history import get_history_db
from models.model_registry import get_model_registry


def render():
    st.markdown("## ⚙️ Settings, Diagnostics & About")
    st.markdown(f"<p class='muted-text'>{APP_TITLE} v{APP_VERSION} — System environment and diagnostic telemetry.</p>", unsafe_allow_html=True)

    # Tab 1: System Telemetry
    # Tab 2: Storage & Cache Management
    # Tab 3: About & Attribution
    tab_sys, tab_cache, tab_about = st.tabs([
        "💻 System Diagnostics",
        "🗄️ Storage & Cache Controls",
        "ℹ️ About & Attribution"
    ])

    with tab_sys:
        st.markdown("#### Hardware & Environment Telemetry")

        # TensorFlow & Hardware Check
        tf_version = "Checking..."
        gpu_detected = False
        gpu_names = []

        try:
            import tensorflow as tf
            tf_version = tf.__version__
            gpus = tf.config.list_physical_devices("GPU")
            gpu_detected = len(gpus) > 0
            gpu_names = [g.name for g in gpus]
        except Exception as e:
            tf_version = f"Error: {e}"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Python Runtime", sys.version.split()[0])
        with col2:
            st.metric("TensorFlow Version", tf_version)
        with col3:
            st.metric("Compute Accelerator", f"{len(gpu_names)} GPU" if gpu_detected else "CPU Only (AVX2)")

        if gpu_detected:
            st.success(f"⚡ GPU Acceleration Enabled: {', '.join(gpu_names)}")
        else:
            st.info("ℹ️ Running in CPU-optimized execution mode. All ANN and CNN operations use multi-threaded AVX2 instructions.")

        st.markdown("##### Storage Directory Health")
        dirs = [
            ("Dataset Store", DATA_DIR),
            ("Models Store", MODELS_DIR),
            ("Experiments Store", EXPERIMENTS_DIR),
            ("Audit History", HISTORY_DIR),
            ("Reports Store", REPORTS_DIR)
        ]

        dir_data = []
        for name, p in dirs:
            exists = p.exists()
            file_count = len(list(p.glob("**/*"))) if exists else 0
            dir_data.append({
                "Store Component": name,
                "Absolute Path": str(p),
                "Status": "✅ Ready" if exists else "❌ Missing",
                "Items": file_count
            })

        import pandas as pd
        st.dataframe(pd.DataFrame(dir_data), use_container_width=True, hide_index=True)

    with tab_cache:
        st.markdown("#### Cache Maintenance")
        st.markdown("Manage local database records and temporary artifacts.")

        c_db, c_mod = st.columns(2)

        with c_db:
            st.markdown("##### Prediction History Database")
            db = get_history_db()
            st.markdown(f"Total recorded predictions: **{db.count()}**")
            if st.button("Clear Prediction Audit History", type="secondary"):
                cleared = db.clear_history()
                st.success(f"Cleared {cleared} predictions from database.")
                st.rerun()

        with c_mod:
            st.markdown("##### Model Cache Reset")
            reg = get_model_registry()
            all_m = reg.list_models()
            st.markdown(f"Trained models stored: **{len(all_m)}**")
            with st.expander("Delete Specific Model"):
                if all_m:
                    m_to_delete = st.selectbox("Select Model to Delete", options=[m["model_name"] for m in all_m])
                    arch_to_delete = next(m["architecture"] for m in all_m if m["model_name"] == m_to_delete)
                    if st.button(f"Permanently Delete {m_to_delete}"):
                        reg.delete_model(m_to_delete, arch_to_delete)
                        st.success(f"Model {m_to_delete} deleted.")
                        st.rerun()
                else:
                    st.write("No models to delete.")

    with tab_about:
        st.markdown(f"### {APP_TITLE}")
        st.markdown(f"**{APP_TAGLINE}**")
        st.markdown("""
        **NeuroScript AI** is a deep learning handwriting recognition system implementing Artificial Neural Networks (ANN)
        and Convolutional Neural Networks (CNN) for handwritten digits (0–9) and English alphabet letters (A–Z).

        #### 🛡️ Local Privacy & On-Device Processing
        - All neural network inferences, drawing strokes, and uploaded images are processed **100% locally** on your machine.
        - No telemetry, external API calls, or cloud servers are required.
        - Image uploads are discarded from memory immediately after inference unless you explicitly check the opt-in box to retain local thumbnails.

        #### 📚 Dataset Attribution & Licensing
        - **MNIST Dataset:** Created by Yann LeCun, Corinna Cortes, and Christopher Burges. Available under Creative Commons Attribution-Share Alike 3.0 license.
        - **EMNIST (Extended MNIST) Letters:** Created by Gregory Cohen, Saeed Afshar, Jonathan Tapson, and André van Schaik (The MARCS Institute, Western Sydney University). Published as open research data.

        #### ⚠️ Technical Limitations & Scope
        - **Single Character Input:** The classifiers expect a single centered digit or letter. They do not perform full-page continuous OCR or multi-character word segmentation.
        - **Case Conventions:** The EMNIST Letters task trains on 26 distinct character classes corresponding to English letters A–Z.
        """)
