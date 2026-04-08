"""
Tab 2: Efficient Frontier — frontier curve, portfolio overlays,
covariance method toggle.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from style_inject import (
    styled_section_label,
    styled_divider,
    apply_plotly_theme,
    TOKENS,
)
from src.covariance import estimate_covariance
from src.expected_returns import estimate_expected_returns
from src.returns import log_returns, arithmetic_returns
from src.efficient_frontier import generate_efficient_frontier
from src.risk_decomposition import portfolio_volatility


def render(results: dict):
    """Render the Efficient Frontier tab."""
    config = results["config"]
    prices = results["prices"]
    weights = results["weights"]

    # ── Sidebar: covariance method toggle ──
    cov_method = st.sidebar.selectbox(
        "Covariance Method (Frontier)",
        ["ledoit_wolf", "sample", "rmt"],
        index=0,
        key="frontier_cov",
    )

    # Recompute frontier for selected cov method
    log_ret = log_returns(prices)
    arith_ret = arithmetic_returns(prices)
    cov, cov_meta = estimate_covariance(log_ret, cov_method, config)
    mu = estimate_expected_returns(arith_ret, "shrinkage", config)

    styled_section_label("Efficient Frontier")

    frontier = generate_efficient_frontier(mu, cov, config, n_points=40)

    fig = go.Figure()

    # Frontier curve
    if len(frontier) > 0:
        fig.add_trace(go.Scatter(
            x=frontier["volatility"],
            y=frontier["target_return"],
            mode="lines",
            name="Efficient Frontier",
            line=dict(color=TOKENS["accent_primary"], width=2.5),
            hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        ))

    # Overlay optimised portfolios
    method_colors = {
        "Mean-Variance": TOKENS["accent_primary"],
        "Risk Parity": TOKENS["accent_success"],
        "HRP": TOKENS["accent_warning"],
        "Black-Litterman": TOKENS["accent_secondary"],
    }

    for name, w in weights.items():
        port_vol = portfolio_volatility(w, cov)
        port_ret = float(w.values @ mu.reindex(w.index).values)
        fig.add_trace(go.Scatter(
            x=[port_vol],
            y=[port_ret],
            mode="markers+text",
            name=name,
            marker=dict(size=14, color=method_colors.get(name, TOKENS["text_muted"]),
                        line=dict(width=2, color=TOKENS["bg_base"])),
            text=[name],
            textposition="top center",
            textfont=dict(size=10, color=TOKENS["text_secondary"]),
            hovertemplate=f"{name}<br>Vol: %{{x:.2%}}<br>Return: %{{y:.2%}}<extra></extra>",
        ))

    # Overlay benchmarks (equal weight as a reference)
    n = len(cov.index)
    ew = pd.Series(1.0 / n, index=cov.index)
    ew_vol = portfolio_volatility(ew, cov)
    ew_ret = float(ew.values @ mu.reindex(ew.index).values)
    fig.add_trace(go.Scatter(
        x=[ew_vol], y=[ew_ret],
        mode="markers+text",
        name="Equal Weight",
        marker=dict(size=10, symbol="diamond", color=TOKENS["text_muted"],
                    line=dict(width=1.5, color=TOKENS["text_secondary"])),
        text=["EW"],
        textposition="bottom center",
        textfont=dict(size=9, color=TOKENS["text_muted"]),
    ))

    fig.update_layout(
        title="Efficient Frontier — Risk vs Return",
        xaxis_title="Annualised Volatility",
        yaxis_title="Annualised Return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ── Cov method info ──
    styled_divider()
    cov_label = {"sample": "Sample", "ledoit_wolf": "Ledoit-Wolf", "rmt": "RMT"}[cov_method]
    info_text = f"Frontier computed with <b>{cov_label}</b> covariance."
    if "shrinkage_intensity" in cov_meta:
        info_text += f" Shrinkage intensity: {cov_meta['shrinkage_intensity']:.4f}"
    if "n_cleaned" in cov_meta:
        info_text += f" Eigenvalues cleaned: {cov_meta['n_cleaned']}/{cov_meta['n_total']}"
    st.markdown(
        "<p style='color:" + TOKENS["text_muted"] + ";font-size:0.85rem'>" + info_text + "</p>",
        unsafe_allow_html=True,
    )
