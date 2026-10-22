"""
NeuroScript AI - Reusable UI Components
Provides styled metric cards, interactive Plotly probability distribution charts,
interactive confusion matrix heatmaps, workflow diagrams, and elegant empty states.
"""

from typing import List, Dict, Any, Optional
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def render_metric_card(
    label: str,
    value: str,
    subtitle: Optional[str] = None,
    accent_color: str = "#7564F5"
):
    """Render a card displaying a key performance indicator."""
    card_html = f"""
    <div class="neuro-card" style="border-left: 4px solid {accent_color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {f'<div class="muted-text" style="margin-top: 0.35rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_status_badge(text: str, is_ready: bool = True) -> str:
    """Return inline HTML pill badge."""
    bg = "rgba(54, 179, 126, 0.15)" if is_ready else "rgba(245, 185, 66, 0.15)"
    fg = "#36B37E" if is_ready else "#F5B942"
    dot = "●"
    return f"""<span style="background:{bg}; color:{fg}; padding:3px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; display:inline-flex; align-items:center; gap:5px;"><span>{dot}</span> {text}</span>"""


def render_empty_state(
    title: str,
    message: str,
    icon: str = "ℹ️"
):
    """Render a clean placeholder when data or models have not yet been initialized."""
    html = f"""
    <div class="neuro-card" style="text-align: center; padding: 2.5rem 1.5rem; border: 1px dashed #252B4E;">
        <div style="font-size: 2.2rem; margin-bottom: 0.75rem;">{icon}</div>
        <div style="font-weight: 700; font-size: 1.15rem; color: #F6F7FB; margin-bottom: 0.4rem;">{title}</div>
        <div class="muted-text" style="max-width: 480px; margin: 0 auto;">{message}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_top_k_chart(top_k: List[Dict[str, Any]]):
    """Render interactive Plotly horizontal bar chart of top-k class probabilities."""
    if not top_k:
        return

    # Invert order for horizontal bar chart (highest on top)
    classes = [item["class"] for item in reversed(top_k)]
    probabilities = [item["probability"] * 100.0 for item in reversed(top_k)]
    text_labels = [f"{p:.1f}%" for p in probabilities]

    # Gradient color logic: top prediction gets cyan accent, others electric violet
    colors = ["#7564F5"] * (len(classes) - 1) + ["#42D9E8"]

    fig = go.Figure(go.Bar(
        x=probabilities,
        y=classes,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(255, 255, 255, 0.1)", width=1)
        ),
        text=text_labels,
        textposition="outside",
        textfont=dict(color="#F6F7FB", size=11, family="JetBrains Mono")
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=40, t=10, b=20),
        xaxis=dict(
            range=[0, max(probabilities) * 1.25 if max(probabilities) > 0 else 100],
            showgrid=True,
            gridcolor="#191E38",
            zeroline=False,
            showticklabels=True,
            tickfont=dict(color="#9AA5C0", size=10)
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#F6F7FB", size=12, family="Plus Jakarta Sans")
        ),
        height=max(180, len(classes) * 38),
        font=dict(family="Plus Jakarta Sans")
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_confusion_matrix_interactive(
    cm: List[List[int]],
    classes: List[str],
    title: str = "Confusion Matrix"
):
    """Render Plotly interactive confusion matrix heatmap."""
    cm_arr = np.array(cm)

    fig = go.Figure(data=go.Heatmap(
        z=cm_arr,
        x=classes,
        y=classes,
        colorscale=[
            [0.0, "#101426"],
            [0.2, "#191E38"],
            [0.5, "#4B3FC9"],
            [0.8, "#7564F5"],
            [1.0, "#42D9E8"]
        ],
        colorbar=dict(
            title=dict(text="Samples", font=dict(color="#9AA5C0", size=10)),
            tickfont=dict(color="#9AA5C0")
        ),
        hoverongaps=False,
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#F6F7FB", size=13)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=dict(text="Predicted Label", font=dict(color="#9AA5C0", size=11)),
            tickfont=dict(color="#F6F7FB", size=9),
            tickangle=45
        ),
        yaxis=dict(
            title=dict(text="True Label", font=dict(color="#9AA5C0", size=11)),
            tickfont=dict(color="#F6F7FB", size=9),
            autorange="reversed"
        ),
        margin=dict(l=40, r=20, t=40, b=40),
        height=450,
        font=dict(family="Plus Jakarta Sans")
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


def render_workflow_diagram():
    """Render the handwriting recognition pipeline stages."""
    html = """
    <div style="display: flex; flex-wrap: wrap; gap: 10px; align-items: center; justify-content: space-between; margin: 1.5rem 0;">
        <div class="neuro-card" style="flex: 1; min-width: 140px; text-align: center; padding: 1rem 0.5rem; margin-bottom: 0;">
            <div style="font-size: 1.4rem;">✍️</div>
            <div style="font-weight: 700; font-size: 0.85rem; color: #F6F7FB; margin-top: 4px;">1. Stroke Capture</div>
            <div class="muted-text" style="font-size: 0.72rem;">Canvas / Upload</div>
        </div>
        <div style="color: #7564F5; font-size: 1.2rem; font-weight: bold;">→</div>
        <div class="neuro-card" style="flex: 1; min-width: 140px; text-align: center; padding: 1rem 0.5rem; margin-bottom: 0;">
            <div style="font-size: 1.4rem;">⚙️</div>
            <div style="font-weight: 700; font-size: 0.85rem; color: #F6F7FB; margin-top: 4px;">2. Preprocessing</div>
            <div class="muted-text" style="font-size: 0.72rem;">Center, Invert, 28x28</div>
        </div>
        <div style="color: #7564F5; font-size: 1.2rem; font-weight: bold;">→</div>
        <div class="neuro-card" style="flex: 1; min-width: 140px; text-align: center; padding: 1rem 0.5rem; margin-bottom: 0;">
            <div style="font-size: 1.4rem;">🧠</div>
            <div style="font-weight: 700; font-size: 0.85rem; color: #F6F7FB; margin-top: 4px;">3. Neural Network</div>
            <div class="muted-text" style="font-size: 0.72rem;">ANN vs CNN Inference</div>
        </div>
        <div style="color: #7564F5; font-size: 1.2rem; font-weight: bold;">→</div>
        <div class="neuro-card" style="flex: 1; min-width: 140px; text-align: center; padding: 1rem 0.5rem; margin-bottom: 0;">
            <div style="font-size: 1.4rem;">🎯</div>
            <div style="font-weight: 700; font-size: 0.85rem; color: #42D9E8; margin-top: 4px;">4. Prediction</div>
            <div class="muted-text" style="font-size: 0.72rem;">Confidence & Top-K</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
