"""
NeuroScript AI - Prediction History and Audit Log
Audit and record inspection center for stored inference predictions,
confidence scores, source inputs, and CSV export capabilities.
"""

import io
import base64
import streamlit as st
import pandas as pd
from PIL import Image
from storage.history import get_history_db
from config.settings import TASK_CHOICES, TASK_DESCRIPTIONS
from ui.components import render_empty_state


def render():
    st.markdown("## 📜 Prediction History & Audit Log")
    st.markdown("<p class='muted-text'>Track, inspect, filter, and export persistent records of historical model inferences.</p>", unsafe_allow_html=True)

    db = get_history_db()
    total_count = db.count()

    if total_count == 0:
        render_empty_state(
            title="No Prediction Records Yet",
            message="Perform predictions in the Recognize workspace using either the drawing canvas or image upload to build an audit history.",
            icon="📝"
        )
        return

    # Filter Toolbar
    f_col1, f_col2, f_col3 = st.columns([1, 1.5, 1])

    with f_col1:
        task_filter = st.selectbox("Filter by Task", options=["All"] + TASK_CHOICES, format_func=lambda x: "All Tasks" if x == "All" else TASK_DESCRIPTIONS[x])

    with f_col2:
        search_query = st.text_input("Search (Class, Model, Source)", placeholder="e.g. '7' or 'CNN' or 'canvas'")

    with f_col3:
        record_limit = st.selectbox("Display Limit", options=[25, 50, 100, 250], index=1)

    # Fetch Records
    records = db.get_predictions(
        limit=record_limit,
        task_filter=None if task_filter == "All" else task_filter,
        search_query=search_query.strip() if search_query else None
    )

    st.markdown(f"**Showing {len(records)} of {total_count} total logged predictions**")

    # Export & Clear Toolbar
    exp_c1, exp_c2 = st.columns([2, 1])
    with exp_c1:
        # Convert to CSV for download
        if records:
            df = db.to_dataframe(limit=record_limit)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export History to CSV",
                data=csv_bytes,
                file_name="neuroscript_prediction_history.csv",
                mime="text/csv"
            )

    with exp_c2:
        with st.expander("⚠️ Clear History"):
            confirm_clear = st.checkbox("Confirm database wipe", key="chk_wipe_hist")
            if st.button("Permanently Clear History", type="secondary", disabled=not confirm_clear):
                deleted = db.clear_history()
                st.success(f"Cleared {deleted} records.")
                st.rerun()

    st.markdown("---")

    # Records Display
    for r in records:
        with st.container():
            c_meta, c_pred, c_thumb, c_act = st.columns([2, 1.2, 1, 0.8])

            with c_meta:
                st.markdown(f"**Model:** `{r.get('model_name')}` ({r.get('architecture')})")
                st.markdown(f"<span class='muted-text'>Time: {r.get('timestamp')[:19].replace('T', ' ')} | Task: <b>{r.get('task').title()}</b> | Source: <b>{r.get('source_type')}</b></span>", unsafe_allow_html=True)

            with c_pred:
                st.markdown(f"""
                <div>
                    <span style="font-size: 1.8rem; font-weight: 800; color: #42D9E8;">{r.get('predicted_class')}</span>
                    <span style="color: #36B37E; font-weight: 600; margin-left: 0.5rem;">{r.get('confidence')*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                top_items = r.get("top_k", [])[:3]
                if top_items:
                    top_text = ", ".join([f"{k['class']} ({k['percentage']})" for k in top_items])
                    st.markdown(f"<span class='muted-text' style='font-size: 0.78rem;'>{top_text}</span>", unsafe_allow_html=True)

            with c_thumb:
                b64 = r.get("thumbnail_base64")
                if b64:
                    try:
                        img_bytes = base64.b64decode(b64)
                        thumb_img = Image.open(io.BytesIO(img_bytes))
                        st.image(thumb_img, width=48)
                    except Exception:
                        st.markdown("<span class='muted-text'>[Preview error]</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='muted-text'>[No image saved]</span>", unsafe_allow_html=True)

            with c_act:
                if st.button("🗑️ Delete", key=f"del_pred_{r.get('id')}"):
                    db.delete_prediction(r.get("id"))
                    st.rerun()

            st.markdown("<div style='border-bottom: 1px solid #191E38; margin: 0.5rem 0;'></div>", unsafe_allow_html=True)
