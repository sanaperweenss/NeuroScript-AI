"""
NeuroScript AI - Main Application Entrypoint
Handwritten Alphabet and Digit Recognition Using ANN and CNN
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

import streamlit as st
from config.settings import APP_TITLE, APP_TAGLINE, APP_VERSION
from ui.theme import inject_custom_theme, render_brand_header

# Initialize Streamlit application layout
st.set_page_config(
    page_title=f"{APP_TITLE} - Handwritten Digit & Alphabet AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium dark AI workspace theme
inject_custom_theme()

# Import page modules
from ui.pages import (
    overview,
    recognize,
    model_comparison,
    training_center,
    dataset_explorer,
    model_visualizer,
    prediction_history,
    reports,
    settings as settings_page
)

# Navigation Pages
PAGES = {
    "Overview": ("📊", overview.render),
    "Recognize": ("✍️", recognize.render),
    "Model Comparison": ("⚖️", model_comparison.render),
    "Training Center": ("🏋️", training_center.render),
    "Dataset Explorer": ("🔍", dataset_explorer.render),
    "Model Visualizer": ("🔬", model_visualizer.render),
    "Prediction History": ("📜", prediction_history.render),
    "Reports": ("📑", reports.render),
    "Settings & About": ("⚙️", settings_page.render)
}

# Sidebar Brand & Navigation
render_brand_header(sidebar=True)

# Default navigation state
if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "Overview"

# Navigation radio list
selected_page = st.sidebar.radio(
    "Navigation",
    options=list(PAGES.keys()),
    format_func=lambda x: f"{PAGES[x][0]}  {x}",
    index=list(PAGES.keys()).index(st.session_state["nav_selection"]),
    key="nav_radio"
)
st.session_state["nav_selection"] = selected_page

# Sidebar Footer & System Status
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="padding: 0 0.5rem; font-size: 0.75rem; color: #9AA5C0;">
    <div><b>Version:</b> {APP_VERSION}</div>
    <div style="margin-top: 4px;"><b>Engine:</b> TensorFlow / Keras 3</div>
    <div style="margin-top: 4px;"><b>Privacy:</b> 100% On-Device Inference</div>
</div>
""", unsafe_allow_html=True)

# Render selected page
try:
    page_renderer = PAGES[selected_page][1]
    page_renderer()
except Exception as e:
    st.error(f"Error rendering {selected_page}: {e}")
    st.exception(e)
