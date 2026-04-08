"""
Portfolio Optimization Engine — Streamlit Dashboard (8 tabs, Bloomberg dark mode).

Runs the full pipeline on load, then renders interactive tabs.
"""

import sys
from pathlib import Path

# Ensure project root is on path for imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st

# ── Page config — MUST be the first Streamlit call ──
st.set_page_config(
    page_title="Portfolio Optimization Engine",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject design system ──
from style_inject import (
    inject_styles,
    styled_header,
    styled_section_label,
    styled_divider,
    TOKENS,
)

inject_styles()

# ── Sidebar branding ──
st.sidebar.markdown(
    f"""
    <div style="margin-bottom: 1.5rem;">
        <span style="
            font-family: {TOKENS['font_display']};
            font-weight: 700;
            font-size: 1.3rem;
            color: {TOKENS['accent_primary']};
        ">◆ Portfolio Engine</span>
        <br>
        <span style="
            font-family: {TOKENS['font_body']};
            font-size: 0.8rem;
            color: {TOKENS['text_muted']};
        ">MV · RP · HRP · Black-Litterman</span>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Running pipeline — fetching data & optimising...")
def load_results():
    """Run the full pipeline and cache results."""
    from main import run_pipeline
    return run_pipeline()


# ── Load data ──
results = load_results()

# ── Tabs ──
tab_names = [
    "OVERVIEW",
    "EFFICIENT FRONTIER",
    "WEIGHTS",
    "RISK DECOMPOSITION",
    "BACKTEST",
    "COVARIANCE LAB",
    "NAIVE VS ROBUST",
    "BLACK-LITTERMAN",
]

tabs = st.tabs(tab_names)

import tab_overview, tab_frontier, tab_weights, tab_risk
import tab_backtest, tab_covariance, tab_naive_vs_robust, tab_bl_views

with tabs[0]:
    tab_overview.render(results)

with tabs[1]:
    tab_frontier.render(results)

with tabs[2]:
    tab_weights.render(results)

with tabs[3]:
    tab_risk.render(results)

with tabs[4]:
    tab_backtest.render(results)

with tabs[5]:
    tab_covariance.render(results)

with tabs[6]:
    tab_naive_vs_robust.render(results)

with tabs[7]:
    tab_bl_views.render(results)

# ── Footer ──
st.markdown(
    f"""
    <div style="text-align: center; margin-top: 3rem; padding: 1rem;">
        <span style="
            font-family: {TOKENS['font_body']};
            font-size: 0.75rem;
            color: {TOKENS['text_muted']};
        ">Portfolio Optimization Engine · MV / RP / HRP / Black-Litterman ·
        14 ETFs · 10yr backtest · Bloomberg Dark Mode</span>
    </div>
    """,
    unsafe_allow_html=True,
)
