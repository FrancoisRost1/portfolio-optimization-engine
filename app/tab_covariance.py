"""
Tab 6: Covariance Lab, correlation heatmap, eigenvalue spectrum with
Marchenko-Pastur bounds, condition number comparison, shrinkage intensity.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from style_inject import (
    styled_section_label,
    styled_kpi,
    styled_divider,
    styled_card,
    apply_plotly_theme,
    TOKENS,
)
from src.covariance import (
    estimate_covariance,
    covariance_to_correlation,
    sample_covariance,
)
from src.returns import log_returns


def render(results: dict):
    """Render the Covariance Lab tab."""
    config = results["config"]
    prices = results["prices"]
    log_ret = log_returns(prices)

    # Compute all three methods
    cov_sample, _ = estimate_covariance(log_ret, "sample", config)
    cov_lw, meta_lw = estimate_covariance(log_ret, "ledoit_wolf", config)
    cov_rmt, meta_rmt = estimate_covariance(log_ret, "rmt", config)

    # ── KPIs ──
    styled_section_label("Covariance Matrix Properties")

    cond_sample = np.linalg.cond(cov_sample.values)
    cond_lw = np.linalg.cond(cov_lw.values)
    cond_rmt = np.linalg.cond(cov_rmt.values)
    shrinkage = meta_lw.get("shrinkage_intensity", 0)
    n_cleaned = meta_rmt.get("n_cleaned", 0)
    n_total = meta_rmt.get("n_total", len(cov_rmt))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        styled_kpi("LW Shrinkage", f"{shrinkage:.4f}", "intensity alpha",
                    TOKENS["accent_primary"])
    with c2:
        styled_kpi("RMT Cleaned", f"{n_cleaned}/{n_total}", "eigenvalues",
                    TOKENS["accent_warning"])
    with c3:
        styled_kpi("Cond # (Sample)", f"{cond_sample:.0f}", "higher = noisier",
                    TOKENS["accent_danger"])
    with c4:
        styled_kpi("Cond # (LW)", f"{cond_lw:.0f}", "lower = stabler",
                    TOKENS["accent_success"])

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    # ── Correlation heatmap ──
    styled_section_label("Correlation Heatmap")

    cov_choice = st.radio(
        "Covariance method",
        ["Ledoit-Wolf", "Sample", "RMT"],
        horizontal=True,
        key="cov_lab_heatmap",
    )
    cov_map = {"Sample": cov_sample, "Ledoit-Wolf": cov_lw, "RMT": cov_rmt}
    corr = covariance_to_correlation(cov_map[cov_choice])

    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=[
            [0, TOKENS["accent_danger"]],
            [0.5, TOKENS["bg_elevated"]],
            [1, TOKENS["accent_success"]],
        ],
        zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Corr"),
    ))
    fig_corr.update_layout(title="Correlation Matrix: " + cov_choice, height=450, yaxis_autorange="reversed")
    apply_plotly_theme(fig_corr)
    st.plotly_chart(fig_corr, use_container_width=True)

    styled_divider()

    # ── Eigenvalue spectrum ──
    styled_section_label("Eigenvalue Spectrum with Marchenko-Pastur Bounds")

    eig_sample = np.linalg.eigvalsh(cov_sample.values)[::-1]
    eig_lw = np.linalg.eigvalsh(cov_lw.values)[::-1]
    eig_rmt = np.linalg.eigvalsh(cov_rmt.values)[::-1]

    lambda_max = meta_rmt.get("lambda_max", 0)
    lambda_min = meta_rmt.get("lambda_min", 0)

    fig_eig = go.Figure()
    x_axis = list(range(1, len(eig_sample) + 1))

    fig_eig.add_trace(go.Bar(
        x=x_axis, y=eig_sample, name="Sample",
        marker_color=TOKENS["accent_danger"], opacity=0.6,
    ))
    fig_eig.add_trace(go.Bar(
        x=x_axis, y=eig_lw, name="Ledoit-Wolf",
        marker_color=TOKENS["accent_primary"], opacity=0.6,
    ))
    fig_eig.add_trace(go.Bar(
        x=x_axis, y=eig_rmt, name="RMT Cleaned",
        marker_color=TOKENS["accent_success"], opacity=0.6,
    ))

    # MP bounds
    fig_eig.add_hline(y=lambda_max, line_dash="dash",
                      line_color=TOKENS["accent_warning"],
                      annotation_text=f"MP upper = {lambda_max:.2e}",
                      annotation_font_color=TOKENS["accent_warning"])
    fig_eig.add_hline(y=lambda_min, line_dash="dot",
                      line_color=TOKENS["text_muted"],
                      annotation_text=f"MP lower = {lambda_min:.2e}",
                      annotation_font_color=TOKENS["text_muted"])

    fig_eig.update_layout(
        title="Eigenvalue Spectrum with Marchenko-Pastur Bounds",
        barmode="group",
        xaxis_title="Eigenvalue Rank",
        yaxis_title="Eigenvalue (log scale)",
        yaxis_type="log",
        height=400,
        legend=dict(orientation="h", y=1.12),
    )
    apply_plotly_theme(fig_eig)
    st.plotly_chart(fig_eig, use_container_width=True)

    styled_divider()

    # ── Condition number comparison ──
    styled_section_label("Condition Number Comparison")

    fig_cond = go.Figure(go.Bar(
        x=["Sample", "Ledoit-Wolf", "RMT"],
        y=[cond_sample, cond_lw, cond_rmt],
        marker_color=[TOKENS["accent_danger"], TOKENS["accent_primary"],
                      TOKENS["accent_success"]],
        text=[f"{cond_sample:.0f}", f"{cond_lw:.0f}", f"{cond_rmt:.0f}"],
        textposition="outside",
    ))
    fig_cond.update_layout(
        title="Condition Number by Method",
        yaxis_title="Condition Number",
        yaxis_type="log",
        height=320,
    )
    apply_plotly_theme(fig_cond)
    st.plotly_chart(fig_cond, use_container_width=True)

    styled_card(
        "High condition number (>300) = unstable matrix inversion = noisy weights. "
        "Shrinkage reduces this by pulling eigenvalues toward a structured target, "
        "making the covariance matrix better conditioned and the optimizer more stable.",
        accent_color=TOKENS["accent_warning"],
    )
