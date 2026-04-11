"""
Tab 4: Risk Decomposition — risk contribution bars, marginal risk table,
pct risk vs pct weight, Herfindahl comparison.
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


def render(results: dict):
    """Render the Risk Decomposition tab."""
    risk_decomp = results["risk_decomp"]
    herfindahl = results["herfindahl"]
    weights = results["weights"]

    # ── Herfindahl KPIs ──
    styled_section_label("Portfolio Concentration (Herfindahl Index)")
    cols = st.columns(len(herfindahl))
    method_colors = [
        TOKENS["accent_primary"], TOKENS["accent_success"],
        TOKENS["accent_warning"], TOKENS["accent_secondary"],
    ]
    for col, ((name, h), color) in zip(cols, zip(herfindahl.items(), method_colors)):
        with col:
            eff_n = 1.0 / h if h > 0 else float("nan")
            styled_kpi(name, f"{h:.4f}", f"Eff. N = {eff_n:.1f}", color)

    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    # ── Risk contribution bar charts ──
    styled_section_label("Risk Contribution by Asset")

    method_select = st.selectbox(
        "Select method", list(risk_decomp.keys()), key="risk_method_select"
    )

    rc = risk_decomp[method_select]
    tickers = rc.index.tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=tickers,
        y=rc["pct_risk_contribution"],
        name="% Risk Contribution",
        marker_color=TOKENS["accent_danger"],
        text=[f"{v:.1%}" for v in rc["pct_risk_contribution"]],
        textposition="outside",
        textfont=dict(size=9),
    ))
    fig.add_trace(go.Bar(
        x=tickers,
        y=rc["weight"],
        name="% Weight",
        marker_color=TOKENS["accent_info"],
        text=[f"{v:.1%}" for v in rc["weight"]],
        textposition="outside",
        textfont=dict(size=9),
    ))

    fig.update_layout(
        title="Risk Contribution vs Weight",
        barmode="group",
        yaxis_tickformat=".0%",
        yaxis_title="Percentage",
        height=400,
        legend=dict(orientation="h", y=1.12),
    )
    apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    styled_divider()

    # ── Marginal risk contribution table ──
    styled_section_label("Marginal Risk Contributions")
    display_df = rc[["weight", "marginal_risk", "risk_contribution", "pct_risk_contribution"]].copy()
    display_df.columns = ["Weight", "Marginal RC", "Absolute RC", "% RC"]
    for col in ["Weight", "% RC"]:
        display_df[col] = display_df[col].map("{:.2%}".format)
    for col in ["Marginal RC", "Absolute RC"]:
        display_df[col] = display_df[col].map("{:.6f}".format)
    st.dataframe(display_df, use_container_width=True, hide_index=False)

    styled_divider()

    # ── Cross-method risk contribution comparison ──
    styled_section_label("Risk Contribution Comparison: All Methods")

    fig2 = go.Figure()
    colors = [TOKENS["accent_primary"], TOKENS["accent_success"],
              TOKENS["accent_warning"], TOKENS["accent_secondary"]]

    for i, (name, rc_df) in enumerate(risk_decomp.items()):
        fig2.add_trace(go.Bar(
            x=rc_df.index,
            y=rc_df["pct_risk_contribution"],
            name=name,
            marker_color=colors[i % len(colors)],
        ))

    target_line = 1.0 / len(tickers)
    fig2.add_hline(
        y=target_line, line_dash="dash",
        line_color=TOKENS["text_muted"],
        annotation_text=f"Equal RC = {target_line:.1%}",
        annotation_font_color=TOKENS["text_muted"],
    )

    fig2.update_layout(
        title="Risk Contribution: All Methods",
        barmode="group",
        yaxis_tickformat=".0%",
        yaxis_title="% Risk Contribution",
        height=400,
        legend=dict(orientation="h", y=1.12),
    )
    apply_plotly_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    styled_card(
        "Negative risk contribution indicates a diversification/hedging benefit: "
        "the asset reduces total portfolio risk rather than adding to it.",
        accent_color=TOKENS["accent_info"],
    )
