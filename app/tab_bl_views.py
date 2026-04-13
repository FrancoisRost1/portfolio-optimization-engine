"""
Tab 8: Black-Litterman Views, equilibrium returns, interactive view editor,
prior vs posterior comparison, confidence slider.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from style_inject import (
    styled_section_label,
    styled_divider,
    styled_card,
    apply_plotly_theme,
    TOKENS,
)
from src.expected_returns import bl_implied_returns
from src.optimizer_bl import (
    build_views,
    compute_view_uncertainty,
    bl_posterior,
    optimize_black_litterman,
)
from src.covariance import covariance_to_correlation


def render(results: dict):
    """Render the Black-Litterman Views tab."""
    config = results["config"]
    cov = results["covariance"]
    market_weights = results["market_weights"]
    bl_info = results["bl_info"]
    tickers = cov.index.tolist()

    bl_config = config.get("black_litterman", {})
    tau = bl_config.get("tau", 0.05)
    risk_aversion = bl_config.get("risk_aversion", 2.5)

    # ── Sidebar: Interactive view editor ──
    st.sidebar.markdown("---")
    color = TOKENS["accent_secondary"]
    st.sidebar.markdown(f"**<span style='color:{color}'>BLACK-LITTERMAN VIEWS</span>**",
                        unsafe_allow_html=True)

    # Get current views from session state or config defaults
    if "bl_views" not in st.session_state:
        st.session_state.bl_views = dict(bl_config.get("default_views", {}))
    if "bl_confidence" not in st.session_state:
        st.session_state.bl_confidence = dict(bl_config.get("view_confidence", {}))

    # Add new view
    new_ticker = st.sidebar.selectbox("Add view for", ["(none)"] + tickers, key="bl_new_ticker")
    if new_ticker != "(none)":
        new_return = st.sidebar.number_input(
            f"Expected return for {new_ticker}", value=0.05,
            min_value=-0.50, max_value=0.50, step=0.01, key="bl_new_ret",
        )
        new_conf = st.sidebar.slider(
            f"Confidence for {new_ticker}", 0.1, 1.0, 0.5, 0.05, key="bl_new_conf",
        )
        if st.sidebar.button("Add View", key="bl_add"):
            st.session_state.bl_views[new_ticker] = new_return
            st.session_state.bl_confidence[new_ticker] = new_conf
            st.rerun()

    # Display and remove existing views
    if st.session_state.bl_views:
        st.sidebar.markdown("**Current views:**")
        to_remove = None
        for ticker, ret in st.session_state.bl_views.items():
            conf = st.session_state.bl_confidence.get(ticker, 0.5)
            col1, col2 = st.sidebar.columns([3, 1])
            col1.markdown(f"`{ticker}`: {ret:.1%} (conf: {conf:.0%})")
            if col2.button("X", key=f"bl_rm_{ticker}"):
                to_remove = ticker
        if to_remove:
            del st.session_state.bl_views[to_remove]
            st.session_state.bl_confidence.pop(to_remove, None)
            st.rerun()

    # Recompute BL with current views
    views = st.session_state.bl_views
    view_conf = st.session_state.bl_confidence

    prior = bl_implied_returns(cov, market_weights, risk_aversion=risk_aversion)
    P, Q, confidences = build_views(tickers, views, view_conf, bl_config.get("default_confidence", 0.5))
    cov_annual = cov * 252
    omega = compute_view_uncertainty(P, cov_annual.values, tau, confidences) if P.shape[0] > 0 else np.empty((0, 0))
    posterior, posterior_cov = bl_posterior(prior, cov_annual, P, Q, omega, tau)

    # ── Equilibrium returns bar chart ──
    styled_section_label("Implied Equilibrium Returns (prior)")

    fig_pi = go.Figure(go.Bar(
        x=prior.index, y=prior.values,
        marker_color=[TOKENS["accent_success"] if v > 0 else TOKENS["accent_danger"]
                      for v in prior.values],
        text=[f"{v:.2%}" for v in prior.values],
        textposition="outside",
        textfont=dict(size=9),
    ))
    fig_pi.update_layout(
        title="Implied Equilibrium Returns",
        yaxis_tickformat=".1%", yaxis_title="Implied Return",
        height=350,
    )
    apply_plotly_theme(fig_pi)
    st.plotly_chart(fig_pi, use_container_width=True)

    styled_divider()

    # ── Prior vs posterior returns ──
    styled_section_label("Prior vs Posterior Expected Returns")

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=tickers, y=prior.reindex(tickers).values,
        name="Prior (equilibrium)",
        marker_color=TOKENS["accent_info"],
    ))
    fig_comp.add_trace(go.Bar(
        x=tickers, y=posterior.reindex(tickers).values,
        name="Posterior (with views)",
        marker_color=TOKENS["accent_secondary"],
    ))

    fig_comp.update_layout(
        title="Prior vs Posterior Expected Returns",
        barmode="group",
        yaxis_tickformat=".1%", yaxis_title="Expected Return",
        height=380,
        legend=dict(orientation="h", y=1.12),
    )
    apply_plotly_theme(fig_comp)
    st.plotly_chart(fig_comp, use_container_width=True)

    styled_card(
        "Posterior returns tilt toward assets with active views while anchoring "
        "to equilibrium for unviewed assets.",
        accent_color=TOKENS["accent_secondary"],
    )

    styled_divider()

    # ── Prior vs posterior weights ──
    styled_section_label("Prior vs Posterior Weights")

    prior_weights = market_weights.reindex(tickers)
    bl_weights_new, _ = optimize_black_litterman(
        cov, market_weights, config, views=views, view_confidence=view_conf,
    )

    fig_w = go.Figure()
    fig_w.add_trace(go.Bar(
        x=tickers, y=prior_weights.values,
        name="Market Cap (prior)",
        marker_color=TOKENS["accent_info"],
    ))
    fig_w.add_trace(go.Bar(
        x=tickers, y=bl_weights_new.reindex(tickers).fillna(0).values,
        name="BL Optimal (posterior)",
        marker_color=TOKENS["accent_secondary"],
    ))
    fig_w.update_layout(
        title="Prior vs Posterior Weights",
        barmode="group",
        yaxis_tickformat=".0%", yaxis_title="Weight",
        height=380,
        legend=dict(orientation="h", y=1.12),
    )
    apply_plotly_theme(fig_w)
    st.plotly_chart(fig_w, use_container_width=True)

    styled_divider()

    # ── Posterior covariance heatmap ──
    styled_section_label("Posterior Covariance Effect")

    corr_prior = covariance_to_correlation(cov_annual)
    corr_posterior = covariance_to_correlation(posterior_cov)
    diff = corr_posterior.values - corr_prior.values

    fig_diff = go.Figure(go.Heatmap(
        z=diff,
        x=tickers, y=tickers,
        colorscale=[
            [0, TOKENS["accent_danger"]],
            [0.5, TOKENS["bg_elevated"]],
            [1, TOKENS["accent_success"]],
        ],
        zmid=0,
        text=np.round(diff, 3),
        texttemplate="%{text}",
        textfont=dict(size=8),
        colorbar=dict(title="Diff"),
        hovertemplate="%{x} vs %{y}: %{z:.4f}<extra></extra>",
    ))
    fig_diff.update_layout(
        height=400,
        yaxis_autorange="reversed",
        title=dict(text="Posterior - Prior Correlation Change"),
    )
    apply_plotly_theme(fig_diff)
    st.plotly_chart(fig_diff, use_container_width=True)

    # ── Views summary card ──
    if views:
        view_lines = [f"<b>{t}</b>: {r:.1%} (conf {view_conf.get(t, 0.5):.0%})"
                      for t, r in views.items()]
        styled_card(
            "<b>Active Views:</b><br>" + "<br>".join(view_lines),
            accent_color=TOKENS["accent_secondary"],
        )
    else:
        styled_card(
            "No active views: posterior equals prior (pure equilibrium returns).",
            accent_color=TOKENS["text_muted"],
        )
