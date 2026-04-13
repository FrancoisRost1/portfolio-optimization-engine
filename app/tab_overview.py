"""
Tab 1: Overview, summary metrics table, current weights bar chart,
key takeaway callout.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from style_inject import (
    styled_header,
    styled_kpi,
    styled_divider,
    styled_section_label,
    styled_card,
    apply_plotly_theme,
    TOKENS,
)


def render(results: dict):
    """Render the Overview tab."""
    styled_header(
        "Portfolio Optimization Engine",
        "Comparing MV, Risk Parity, HRP & Black-Litterman across covariance methods",
    )

    weights = results["weights"]
    metrics_table = results["metrics_table"]
    herfindahl = results["herfindahl"]

    # ── KPI row ──
    if metrics_table:
        best_sharpe_name = max(
            metrics_table, key=lambda k: metrics_table[k].get("Sharpe Ratio", float("-inf"))
        )
        best_sharpe = metrics_table[best_sharpe_name].get("Sharpe Ratio", 0)
        best_dd_name = max(
            metrics_table, key=lambda k: metrics_table[k].get("Max Drawdown", float("-inf"))
        )
        best_dd = metrics_table[best_dd_name].get("Max Drawdown", 0)
        min_herf_name = min(herfindahl, key=herfindahl.get) if herfindahl else "N/A"
        min_herf = herfindahl.get(min_herf_name, 0)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            styled_kpi("Best Sharpe", f"{best_sharpe:.2f}", best_sharpe_name,
                        TOKENS["accent_success"])
        with c2:
            styled_kpi("Shallowest DD", f"{best_dd:.1%}", best_dd_name,
                        TOKENS["accent_success"])
        with c3:
            styled_kpi("Most Diversified", f"{min_herf:.4f}", f"HHI: {min_herf_name}",
                        TOKENS["accent_info"])
        with c4:
            styled_kpi("Assets", str(len(results["prices"].columns)), "in universe",
                        TOKENS["accent_primary"])

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    # ── Metrics summary table ──
    styled_section_label("Performance Summary")
    if metrics_table:
        df = pd.DataFrame(metrics_table).T
        fmt_cols = {
            "CAGR": "{:.2%}", "Annualized Vol": "{:.2%}",
            "Sharpe Ratio": "{:.2f}", "Sortino Ratio": "{:.2f}",
            "Max Drawdown": "{:.2%}", "Calmar Ratio": "{:.2f}",
            "CVaR 95%": "{:.4f}",
        }
        for col, fmt in fmt_cols.items():
            if col in df.columns:
                df[col] = df[col].map(lambda x: fmt.format(x) if pd.notna(x) else "N/A")
        # Drop TE/IR for cleaner overview
        for drop_col in ["Tracking Error", "Information Ratio"]:
            if drop_col in df.columns:
                df = df.drop(columns=drop_col)
        st.dataframe(df, width="stretch", hide_index=False)

    styled_divider()

    # ── Current weights bar chart ──
    styled_section_label("Current Optimal Weights")
    config = results["config"]
    asset_classes = config.get("universe", {}).get("asset_classes", {})

    # Colour by asset class
    class_colors = {
        "equities": TOKENS["accent_primary"],
        "bonds": TOKENS["accent_info"],
        "commodities": TOKENS["accent_warning"],
        "reits": TOKENS["accent_success"],
        "fx": TOKENS["accent_secondary"],
    }

    fig = go.Figure()
    for method_name, w in weights.items():
        colors = [class_colors.get(asset_classes.get(t, ""), TOKENS["text_muted"]) for t in w.index]
        fig.add_trace(go.Bar(
            x=w.index,
            y=w.values,
            name=method_name,
            marker_color=colors,
            text=[f"{v:.1%}" for v in w.values],
            textposition="outside",
            textfont=dict(size=9),
        ))

    fig.update_layout(
        title="Current Optimal Weights by Method",
        barmode="group",
        yaxis_tickformat=".0%",
        yaxis_title="Weight",
        xaxis_title="Asset",
        height=400,
        legend=dict(orientation="h", y=1.12),
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, width="stretch")

    # ── Key takeaway ──
    if metrics_table:
        styled_card(
            f"<b>Key Takeaway:</b> {best_sharpe_name} achieves the highest backtest "
            f"Sharpe ({best_sharpe:.2f}), while {min_herf_name} offers the best "
            f"diversification (HHI = {min_herf:.4f}). Risk Parity delivers the most "
            f"balanced risk contributions across assets.",
            accent_color=TOKENS["accent_primary"],
        )

    styled_divider()

    # ── Strategy verdicts ──
    styled_section_label("Strategy Verdicts")

    verdicts = [
        ("Mean-Variance", TOKENS["accent_primary"],
         "Highest turnover, fragile to estimation error: use with shrinkage only"),
        ("HRP", TOKENS["accent_warning"],
         "Most stable diversification, best risk-adjusted at equal volatility"),
        ("Risk Parity", TOKENS["accent_success"],
         "Lowest turnover, robust baseline: the default choice for most allocators"),
        ("Black-Litterman", TOKENS["accent_secondary"],
         "View-dependent: not a standalone alpha engine, but the right tool for expressing conviction"),
    ]

    for name, color, verdict in verdicts:
        styled_card(
            f"<b>{name}:</b> {verdict}",
            accent_color=color,
        )
