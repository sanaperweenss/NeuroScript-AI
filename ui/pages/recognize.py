"""
NeuroScript AI - Interactive Recognition Workspace
Core interactive prediction environment providing:
1. Drawing Canvas with real-time stroke capture and adjustable brush size
2. Image Upload with validation and pre-scaling
3. Standardized LeCun MNIST/EMNIST preprocessing preview
4. ANN vs CNN inference with top-k confidence distribution
5. Audit persistence to SQLite prediction history
"""

import io
import base64
import streamlit as st
import numpy as np
from PIL import Image

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False

from config.settings import (
    TASK_DIGITS, TASK_ALPHABET, TASK_COMBINED,
    TASK_CHOICES, TASK_DESCRIPTIONS, ARCH_ANN, ARCH_CNN, ARCH_CHOICES
)
from models.model_registry import get_model_registry
from inference.predictor import get_predictor
from storage.history import get_history_db
from ui.components import render_top_k_chart, render_empty_state


def render():
    st.markdown("## ✍️ Character Recognition Workspace")
    st.markdown("<p class='muted-text'>Draw a character or upload an image to test ANN and CNN recognition models.</p>", unsafe_allow_html=True)

    reg = get_model_registry()
    predictor = get_predictor()
    history_db = get_history_db()

    # Model Configuration Toolbar
    col_t, col_a, col_m = st.columns([1.2, 1, 1.5])

    with col_t:
        selected_task = st.selectbox(
            "Recognition Task",
            options=TASK_CHOICES,
            format_func=lambda x: TASK_DESCRIPTIONS[x],
            index=0
        )

    with col_a:
        selected_arch = st.selectbox(
            "Model Architecture",
            options=ARCH_CHOICES,
            index=1  # Default to CNN
        )

    # Fetch available models for this task & architecture
    available_models = reg.list_models(task=selected_task, architecture=selected_arch)

    with col_m:
        if available_models:
            model_options = [m["model_name"] for m in available_models]
            selected_model_name = st.selectbox("Select Trained Model", options=model_options)
        else:
            selected_model_name = None
            st.selectbox("Select Trained Model", options=["No trained model available"], disabled=True)

    if not available_models:
        render_empty_state(
            title=f"No {selected_arch} Model Found for {selected_task.title()}",
            message=f"You need to train an {selected_arch} model first. Head over to the Training Center to train a model in seconds with verified datasets.",
            icon="⚠️"
        )
        if st.button("🚀 Go to Training Center", key="goto_train_cta"):
            st.session_state["nav_selection"] = "Training Center"
            st.rerun()
        return

    # Load active model
    try:
        active_model, active_metadata = reg.load_model(selected_model_name, selected_arch)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    st.markdown("---")

    # Input Method Tabs
    tab_canvas, tab_upload = st.tabs(["🖌️ Interactive Drawing Canvas", "📁 Image Upload"])

    # Shared container for inference results
    input_image_to_process = None
    input_source_type = "canvas"

    # ---------------- TAB 1: DRAWING CANVAS ----------------
    with tab_canvas:
        c1, c2 = st.columns([1.2, 1])

        with c1:
            st.markdown("#### Draw Character Here")
            brush_size = st.slider("Brush Thickness", min_value=8, max_value=32, value=18, step=2)

            if CANVAS_AVAILABLE:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 0)",
                    stroke_width=brush_size,
                    stroke_color="#000000",
                    background_color="#FFFFFF",
                    update_streamlit=True,
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="drawing_canvas",
                    return_image_data=True
                )

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    predict_canvas_clicked = st.button("⚡ Predict Drawing", type="primary", use_container_width=True)

                if predict_canvas_clicked:
                    if canvas_result is not None:
                        try:
                            img_data = canvas_result.image_data
                            if img_data is not None:
                                input_image_to_process = img_data
                                input_source_type = "canvas"
                            else:
                                st.warning("⚠️ No stroke detected. Please draw a digit or letter on the canvas.")
                        except Exception as e:
                            st.warning(f"Unable to read canvas stroke: {e}")
                    else:
                        st.warning("⚠️ Canvas is not ready. Please draw a stroke.")
            else:
                st.warning("`streamlit-drawable-canvas` is initializing or not available. Please use Image Upload tab.")

        with c2:
            st.markdown("#### 💡 Tips for Accurate Recognition")
            st.markdown("""
            - Draw a **single character** centered on the white canvas.
            - Keep strokes continuous and natural, like writing on paper.
            - For digits, draw numbers from **0 to 9**.
            - For letters, draw uppercase characters from **A to Z**.
            - The preprocessor automatically centers your stroke and normalizes polarity to match training data.
            """)

    # ---------------- TAB 2: IMAGE UPLOAD ----------------
    with tab_upload:
        u_col1, u_col2 = st.columns([1.2, 1])

        with u_col1:
            st.markdown("#### Upload Character Image")
            uploaded_file = st.file_uploader(
                "Choose a PNG, JPG, or JPEG file",
                type=["png", "jpg", "jpeg"],
                key="image_uploader"
            )

            if uploaded_file is not None:
                try:
                    pil_img = Image.open(uploaded_file)
                    st.image(pil_img, caption="Uploaded Image Preview", width=220)
                    if st.button("⚡ Predict Uploaded Image", type="primary", key="btn_pred_upload"):
                        input_image_to_process = pil_img
                        input_source_type = "upload"
                except Exception as e:
                    st.error(f"Invalid image file: {e}")

        with u_col2:
            st.markdown("#### 📌 Upload Requirements")
            st.markdown("""
            - Supported formats: **PNG, JPG, JPEG**.
            - Optimal resolution: 50x50 to 800x800 pixels.
            - The model expects a **single isolated character**.
            - Note: Full-page document OCR or multi-character words are not supported by this single-character model.
            """)

    # ---------------- EXECUTE INFERENCE ----------------
    if input_image_to_process is not None:
        with st.spinner("Processing image and executing neural inference..."):
            pred_result = predictor.predict(
                image_input=input_image_to_process,
                model=active_model,
                task=selected_task,
                architecture=selected_arch,
                top_k_count=5
            )
            st.session_state["active_prediction"] = {
                "result": pred_result,
                "source_type": input_source_type,
                "model_name": selected_model_name,
                "architecture": selected_arch,
                "task": selected_task,
                "saved": False
            }

    # ---------------- DISPLAY INFERENCE & RESULTS ----------------
    active_pred = st.session_state.get("active_prediction")
    if active_pred is not None:
        result = active_pred["result"]

        st.markdown("---")
        header_col, action_col = st.columns([3, 1])
        with header_col:
            st.markdown("### 🎯 Recognition Analysis")
        with action_col:
            if st.button("🧹 Clear Result", use_container_width=True):
                st.session_state.pop("active_prediction", None)
                st.rerun()

        if not result.get("success"):
            st.warning(f"⚠️ {result.get('message', 'Recognition failed')}")
        else:
            pred_class = result["predicted_class"]
            confidence = result["confidence"]
            top_k = result["top_k"]
            norm_28x28 = result["preprocessed_image"]
            latency_ms = result["inference_time_ms"]

            # Layout results
            res_col1, res_col2, res_col3 = st.columns([1, 1.2, 1.8])

            with res_col1:
                st.markdown("##### Preprocessed (28×28)")
                if norm_28x28 is not None:
                    disp_img = Image.fromarray(norm_28x28).resize((180, 180), resample=Image.NEAREST)
                    st.image(disp_img, caption="What the Model Sees", width=180)
                st.markdown(f"<div class='muted-text'>Latency: <code>{latency_ms:.1f} ms</code></div>", unsafe_allow_html=True)

            with res_col2:
                st.markdown("##### Prediction Result")
                st.markdown(f"""
                <div class="neuro-card" style="text-align: center; padding: 1.5rem 1rem;">
                    <div class="metric-label">Predicted Class</div>
                    <div style="font-size: 3.5rem; font-weight: 800; color: #42D9E8; line-height: 1.1;">{pred_class}</div>
                    <div style="margin-top: 0.5rem; font-weight: 600; color: #36B37E; font-size: 1.1rem;">
                        {confidence * 100:.1f}% Confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res_col3:
                st.markdown("##### Top-5 Class Probabilities")
                render_top_k_chart(top_k)

            # Audit History Logging
            st.markdown("##### 💾 Save to Prediction Audit History")
            h_col1, h_col2 = st.columns([2, 1])
            with h_col1:
                store_thumbnail = st.checkbox("Retain input thumbnail image locally", value=False, key="chk_store_thumb")

            with h_col2:
                if active_pred.get("saved"):
                    st.success("✅ Saved to audit log!")
                else:
                    if st.button("📥 Save Record", key="btn_save_history"):
                        thumb_b64 = None
                        if store_thumbnail and norm_28x28 is not None:
                            thumb_img = Image.fromarray(norm_28x28)
                            buf = io.BytesIO()
                            thumb_img.save(buf, format="PNG")
                            thumb_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                        history_db.add_prediction(
                            task=active_pred["task"],
                            model_name=active_pred["model_name"],
                            architecture=active_pred["architecture"],
                            predicted_class=pred_class,
                            confidence=confidence,
                            top_k=top_k,
                            source_type=active_pred["source_type"],
                            thumbnail_base64=thumb_b64
                        )
                        active_pred["saved"] = True
                        st.session_state["active_prediction"] = active_pred
                        st.success("✅ Prediction saved to history audit database!")
                        st.rerun()
