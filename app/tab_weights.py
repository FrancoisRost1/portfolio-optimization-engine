"""
Tab 3: Weight Comparison — side-by-side bars, asset class pies,
rolling weight heatmap from backtest.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from style_inject import (
    styled_section_label,
    styled_divider,
    apply_plotly_theme,
    TOKENS,
)


def render(results: dict):
    """Render the Weight Comparison tab."""
    weights = results["weights"]
    config = results["config"]
    asset_classes = config.get("universe", {}).get("asset_classes", {})
    backtest_results = results["backtest_results"]

    # ── Side-by-side weight bars ──
    styled_section_label("Weight Comparison Across Methods")

    method_colors = [
        TOKENS["accent_primary"],
        TOKENS["accent_success"],
        TOKENS["accent_warning"],
        TOKENS["accent_secondary"],
    ]

    fig = go.Figure()
    for i, (name, w) in enumerate(weights.items()):
        fig.add_trace(go.Bar(
            x=w.index,
            y=w.values,
            name=name,
            marker_color=method_colors[i % len(method_colors)],
            text=[f"{v:.1%}" for v in w.values],
            textposition="outside",
            textfont=dict(size=8),
        ))

    fig.update_layout(
        title="Weight Comparison Across Methods",
        barmode="group",
        yaxis_tickformat=".0%",
        yaxis_title="Weight",
        xaxis_title="Asset",
        height=400,
        legend=dict(orientation="h", y=1.12),
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    styled_divider()

    # ── Asset class allocation pies ──
    styled_section_label("Asset Class Allocation")

    cols = st.columns(len(weights))
    class_colors_map = {
        "equities": TOKENS["accent_primary"],
        "bonds": TOKENS["accent_info"],
        "commodities": TOKENS["accent_warning"],
        "reits": TOKENS["accent_success"],
        "fx": TOKENS["accent_secondary"],
    }

    for col, (name, w) in zip(cols, weights.items()):
        with col:
            class_weights = {}
            for ticker, weight in w.items():
                cls = asset_classes.get(ticker, "other")
                class_weights[cls] = class_weights.get(cls, 0) + weight

            labels = list(class_weights.keys())
            values = list(class_weights.values())
            colors = [class_colors_map.get(c, TOKENS["text_muted"]) for c in labels]

            fig_pie = go.Figure(go.Pie(
                labels=[c.title() for c in labels],
                values=values,
                hole=0.65,
                marker=dict(colors=colors, line=dict(color=TOKENS["bg_base"], width=2)),
                textinfo="percent",
                textfont=dict(size=10, color=TOKENS["text_primary"]),
                hovertemplate="%{label}: %{value:.1%}<extra></extra>",
            ))
            fig_pie.update_layout(
                height=250,
                showlegend=False,
                title=dict(text=name, font=dict(size=12, color=TOKENS["text_secondary"]),
                           x=0.5, xanchor="center"),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            apply_plotly_theme(fig_pie)
            st.plotly_chart(fig_pie, use_container_width=True)

    styled_divider()

    # ── Rolling weight heatmap from backtest ──
    styled_section_label("Weight Stability Over Time")

    method_select = st.selectbox(
        "Select method", list(backtest_results.keys()),
        format_func=lambda x: {"mv": "Mean-Variance", "rp": "Risk Parity",
                                "hrp": "HRP", "bl": "Black-Litterman"}.get(x, x),
        key="weight_heatmap_method",
    )

    wh = backtest_results[method_select]["weights_history"]
    if wh:
        dates = [d for d, _ in wh]
        tickers = wh[0][1].index.tolist()
        matrix = np.array([w.values for _, w in wh])

        fig_heat = go.Figure(go.Heatmap(
            z=matrix.T,
            x=dates,
            y=tickers,
            colorscale=[
                [0, TOKENS["bg_base"]],
                [0.5, TOKENS["accent_primary"]],
                [1, TOKENS["accent_warning"]],
            ],
            hovertemplate="Date: %{x}<br>Asset: %{y}<br>Weight: %{z:.2%}<extra></extra>",
            colorbar=dict(title="Weight", tickformat=".0%"),
        ))
        fig_heat.update_layout(title="Rolling Weight Heatmap", height=350, xaxis_title="Rebalance Date", yaxis_title="Asset")
        apply_plotly_theme(fig_heat)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No weight history available for this method.")
