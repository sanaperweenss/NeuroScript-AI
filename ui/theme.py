"""
NeuroScript AI - UI Styling and Design System
Provides custom CSS injection and styling constants for the premium dark AI workspace.
Colors:
- Midnight Navy: #101426
- Deep Indigo: #191E38
- Electric Violet: #7564F5
- Cyan: #42D9E8
- Off-white: #F6F7FB
- Muted Slate: #9AA5C0
- Success Green: #36B37E
- Warning Amber: #F5B942
"""

import streamlit as st


def inject_custom_theme():
    """Inject custom CSS to establish the premium NeuroScript dark AI workspace."""
    custom_css = """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-navy: #101426;
        --card-indigo: #191E38;
        --sidebar-navy: #0C0F1E;
        --electric-violet: #7564F5;
        --cyan-accent: #42D9E8;
        --text-offwhite: #F6F7FB;
        --text-muted: #9AA5C0;
        --border-subtle: #252B4E;
        --success-green: #36B37E;
        --warning-amber: #F5B942;
    }

    /* Main background */
    .stApp {
        background-color: var(--bg-navy);
        color: var(--text-offwhite);
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-navy) !important;
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] div.block-container {
        padding-top: 2rem;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-offwhite) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    p, span, label, li {
        color: var(--text-offwhite);
    }
    .muted-text {
        color: var(--text-muted) !important;
        font-size: 0.9rem;
    }

    /* Custom Cards */
    .neuro-card {
        background: var(--card-indigo);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .neuro-card:hover {
        border-color: rgba(117, 100, 245, 0.4);
    }

    .neuro-card-accent {
        border-left: 4px solid var(--electric-violet);
    }
    .neuro-card-cyan {
        border-left: 4px solid var(--cyan-accent);
    }

    /* Metric Badges */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-offwhite);
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    /* Brand Logo styling */
    .brand-container {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0 1.5rem 0;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 1.5rem;
    }
    .brand-icon {
        width: 38px;
        height: 38px;
        background: linear-gradient(135deg, #7564F5 0%, #42D9E8 100%);
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.2rem;
        color: #FFFFFF;
        box-shadow: 0 0 15px rgba(117, 100, 245, 0.45);
    }
    .brand-title {
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #FFFFFF 30%, #42D9E8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .brand-tagline {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin: 0;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7564F5 0%, #5E4BF0 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 10px rgba(117, 100, 245, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(117, 100, 245, 0.5) !important;
    }

    /* Secondary / outline buttons */
    .stButton > button[kind="secondary"] {
        background: #191E38 !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-offwhite) !important;
        box-shadow: none !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: var(--cyan-accent) !important;
        color: var(--cyan-accent) !important;
    }

    /* Form controls & selectboxes */
    .stSelectbox > div > div, .stTextInput > div > div, .stNumberInput > div > div {
        background-color: #191E38 !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        color: var(--text-offwhite) !important;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7564F5 0%, #42D9E8 100%) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        color: var(--text-muted);
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--card-indigo) !important;
        color: var(--cyan-accent) !important;
        border-bottom: 2px solid var(--cyan-accent) !important;
    }

    /* Code blocks */
    code {
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--cyan-accent) !important;
        background-color: rgba(25, 30, 56, 0.8) !important;
        border-radius: 4px;
        padding: 0.15rem 0.35rem;
    }

    /* Hide Streamlit default branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_brand_header(sidebar: bool = True):
    """Render typographic branding logo."""
    html = """
    <div class="brand-container">
        <div class="brand-icon">Ψ</div>
        <div>
            <div class="brand-title">NeuroScript AI</div>
            <div class="brand-tagline">Stroke to Intelligence</div>
        </div>
    </div>
    """
    if sidebar:
        st.sidebar.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)
