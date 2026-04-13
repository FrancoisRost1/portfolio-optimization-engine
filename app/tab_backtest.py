"""
Tab 5: Backtest, cumulative returns, drawdowns, rolling Sharpe,
metrics table (raw + vol-normalised), turnover analysis, gross vs net.
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
from src.metrics import (
    drawdown_series,
    rolling_sharpe,
    compute_all_metrics,
    vol_normalize,
    compute_gross_net_stats,
)


METHOD_LABELS = {"mv": "Mean-Variance", "rp": "Risk Parity",
                 "hrp": "HRP", "bl": "Black-Litterman"}

METHOD_COLORS = {
    "mv": TOKENS["accent_primary"],
    "rp": TOKENS["accent_success"],
    "hrp": TOKENS["accent_warning"],
    "bl": TOKENS["accent_secondary"],
}

BENCH_COLORS = {
    "SPY B&H": TOKENS["text_muted"],
    "60/40": TOKENS["accent_info"],
    "Equal Weight": TOKENS["accent_danger"],
    "Static RP": TOKENS["accent_warning"],
}


def _format_metrics_df(metrics_table: dict) -> pd.DataFrame:
    """Format a metrics dict into a display DataFrame."""
    df = pd.DataFrame(metrics_table).T
    fmt_map = {
        "CAGR": "{:.2%}", "Annualized Vol": "{:.2%}",
        "Sharpe Ratio": "{:.2f}", "Sortino Ratio": "{:.2f}",
        "Max Drawdown": "{:.2%}", "Calmar Ratio": "{:.2f}",
        "CVaR 95%": "{:.4f}", "Tracking Error": "{:.2%}",
        "Information Ratio": "{:.2f}",
    }
    for col, fmt in fmt_map.items():
        if col in df.columns:
            df[col] = df[col].map(lambda x: fmt.format(x) if pd.notna(x) else "N/A")
    return df


def render(results: dict):
    """Render the Backtest tab."""
    backtest_results = results["backtest_results"]
    benchmark_returns = results["benchmark_returns"]
    metrics_table = results["metrics_table"]
    config = results["config"]

    # ── Vol-normalisation toggle ──
    vol_norm = st.sidebar.checkbox("Normalize to 10% vol", value=False, key="vol_norm_toggle")
    target_vol = 0.10

    # Build return series dict (possibly vol-normalised)
    all_returns = {}
    for method, bt in backtest_results.items():
        ret = bt["returns"]
        if len(ret) > 0:
            all_returns[METHOD_LABELS.get(method, method)] = ret
    for name, ret in benchmark_returns.items():
        if len(ret) > 0:
            all_returns[name] = ret

    if vol_norm:
        display_returns = {k: vol_normalize(v, target_vol) for k, v in all_returns.items()}
        vol_label = " (Vol-Normalised 10%)"
    else:
        display_returns = all_returns
        vol_label = ""

    # ── Cumulative returns ──
    styled_section_label("Cumulative Returns" + vol_label)

    fig_cum = go.Figure()
    for name, ret in display_returns.items():
        cum = (1 + ret).cumprod()
        is_bench = name in BENCH_COLORS
        color = BENCH_COLORS.get(name) or _method_color(name)
        fig_cum.add_trace(go.Scatter(
            x=cum.index, y=cum.values, mode="lines", name=name,
            line=dict(color=color, width=1.5 if is_bench else 2,
                      dash="dot" if is_bench else "solid"),
        ))

    fig_cum.update_layout(
        title="Cumulative Returns" + vol_label,
        yaxis_title="Growth of $1", height=420,
        legend=dict(orientation="h", y=-0.15), hovermode="x unified",
    )
    apply_plotly_theme(fig_cum)
    st.plotly_chart(fig_cum, width="stretch")

    styled_divider()

    # ── Drawdown chart ──
    styled_section_label("Drawdowns" + vol_label)

    fig_dd = go.Figure()
    for method, bt in backtest_results.items():
        ret = bt["returns"]
        if len(ret) == 0:
            continue
        if vol_norm:
            ret = vol_normalize(ret, target_vol)
        dd = drawdown_series(ret)
        color = METHOD_COLORS.get(method, TOKENS["text_muted"])
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values, mode="lines",
            name=METHOD_LABELS.get(method, method),
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor="rgba({},0.08)".format(_hex_to_rgb(color)),
        ))

    fig_dd.update_layout(
        title="Drawdowns" + vol_label,
        yaxis_title="Drawdown", yaxis_tickformat=".0%", height=320,
        legend=dict(orientation="h", y=-0.2), hovermode="x unified",
    )
    apply_plotly_theme(fig_dd)
    st.plotly_chart(fig_dd, width="stretch")

    styled_divider()

    # ── Rolling Sharpe ──
    styled_section_label("Rolling Sharpe (12-month)" + vol_label)
    window = config.get("metrics", {}).get("rolling_sharpe_window", 252)
    rf = config.get("metrics", {}).get("risk_free_rate", 0.04)

    fig_rs = go.Figure()
    for method, bt in backtest_results.items():
        ret = bt["returns"]
        if len(ret) == 0:
            continue
        if vol_norm:
            ret = vol_normalize(ret, target_vol)
        rs = rolling_sharpe(ret, window=window, rf=rf)
        fig_rs.add_trace(go.Scatter(
            x=rs.index, y=rs.values, mode="lines",
            name=METHOD_LABELS.get(method, method),
            line=dict(color=METHOD_COLORS.get(method, TOKENS["text_muted"]), width=1.5),
        ))

    fig_rs.add_hline(y=0, line_dash="dash", line_color=TOKENS["text_muted"])
    fig_rs.update_layout(
        title="Rolling 12-Month Sharpe" + vol_label,
        yaxis_title="Rolling Sharpe", height=320,
        legend=dict(orientation="h", y=-0.2), hovermode="x unified",
    )
    apply_plotly_theme(fig_rs)
    st.plotly_chart(fig_rs, width="stretch")

    styled_divider()

    # ── Metrics table (raw + vol-normalised side by side) ──
    styled_section_label("Backtest Metrics Summary")

    if vol_norm:
        # Compute vol-normalised metrics
        spy_ret = benchmark_returns.get("SPY B&H")
        spy_vn = vol_normalize(spy_ret, target_vol) if spy_ret is not None and len(spy_ret) > 0 else None

        vn_metrics = {}
        for name, ret in display_returns.items():
            if len(ret) > 0:
                vn_metrics[name] = compute_all_metrics(ret, config, spy_vn)

        col_raw, col_vn = st.columns(2)
        with col_raw:
            st.markdown("**Raw (unscaled)**")
            if metrics_table:
                st.dataframe(_format_metrics_df(metrics_table), width="stretch")
        with col_vn:
            st.markdown("**Vol-Normalised (10%)**")
            if vn_metrics:
                st.dataframe(_format_metrics_df(vn_metrics), width="stretch")
    else:
        if metrics_table:
            st.dataframe(_format_metrics_df(metrics_table), width="stretch")

    styled_divider()

    # ── Gross vs Net Comparison (Upgrade 3) ──
    styled_section_label("Gross vs Net Return Analysis")

    tc_bps = config.get("backtest", {}).get("transaction_cost_bps", 10)
    gn_rows = {}
    for method, bt in backtest_results.items():
        label = METHOD_LABELS.get(method, method)
        stats = compute_gross_net_stats(
            bt["returns"], bt["turnover"], tc_bps, config
        )
        gn_rows[label] = stats

    if gn_rows:
        gn_df = pd.DataFrame(gn_rows).T
        gn_display = pd.DataFrame({
            "Avg Annual Turnover": gn_df["avg_annual_turnover"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"),
            "Total TC Drag": gn_df["total_tc_drag"].map(
                lambda x: f"{x:.2%}" if pd.notna(x) else "N/A"),
            "Gross Sharpe": gn_df["gross_sharpe"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"),
            "Net Sharpe": gn_df["net_sharpe"].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"),
            "Sharpe Cost": gn_df.apply(
                lambda r: f"{r['gross_sharpe'] - r['net_sharpe']:.3f}"
                if pd.notna(r['gross_sharpe']) and pd.notna(r['net_sharpe']) else "N/A",
                axis=1),
        })
        st.dataframe(gn_display, width="stretch")

        # Bar chart: gross vs net Sharpe
        fig_gn = go.Figure()
        names = list(gn_rows.keys())
        gross_vals = [gn_rows[n]["gross_sharpe"] for n in names]
        net_vals = [gn_rows[n]["net_sharpe"] for n in names]

        fig_gn.add_trace(go.Bar(
            x=names, y=gross_vals, name="Gross Sharpe",
            marker_color=TOKENS["accent_info"], opacity=0.7,
        ))
        fig_gn.add_trace(go.Bar(
            x=names, y=net_vals, name="Net Sharpe",
            marker_color=TOKENS["accent_primary"],
        ))
        fig_gn.update_layout(
            title="Gross vs Net Sharpe Ratio",
            barmode="group", yaxis_title="Sharpe Ratio", height=320,
            legend=dict(orientation="h", y=1.12),
        )
        apply_plotly_theme(fig_gn)
        st.plotly_chart(fig_gn, width="stretch")

        styled_card(
            "<b>Turnover cost insight:</b> HRP and Risk Parity have lower turnover "
            "than Mean-Variance, so the gap between gross and net Sharpe is smaller. "
            "Stability pays off when transaction costs are included.",
            accent_color=TOKENS["accent_success"],
        )

    styled_divider()

    # ── Turnover time series + histogram ──
    styled_section_label("Turnover Time Series")

    c1, c2 = st.columns(2)

    with c1:
        fig_to = go.Figure()
        for method, bt in backtest_results.items():
            if bt["turnover"]:
                dates = [d for d, _ in bt["turnover"]]
                vals = [v for _, v in bt["turnover"]]
                fig_to.add_trace(go.Bar(
                    x=dates, y=vals,
                    name=METHOD_LABELS.get(method, method),
                    marker_color=METHOD_COLORS.get(method, TOKENS["text_muted"]),
                    opacity=0.8,
                ))
        fig_to.update_layout(
            title="Turnover per Rebalance",
            barmode="group", yaxis_title="Turnover",
            height=300, legend=dict(orientation="h", y=1.15),
        )
        apply_plotly_theme(fig_to)
        st.plotly_chart(fig_to, width="stretch")

    with c2:
        fig_hist = go.Figure()
        for method, bt in backtest_results.items():
            if bt["turnover"]:
                vals = [v for _, v in bt["turnover"]]
                fig_hist.add_trace(go.Histogram(
                    x=vals, name=METHOD_LABELS.get(method, method),
                    marker_color=METHOD_COLORS.get(method, TOKENS["text_muted"]),
                    opacity=0.7, nbinsx=20,
                ))
        fig_hist.update_layout(
            title="Turnover Distribution",
            barmode="overlay", xaxis_title="Turnover",
            yaxis_title="Frequency", height=300,
            legend=dict(orientation="h", y=1.15),
        )
        apply_plotly_theme(fig_hist)
        st.plotly_chart(fig_hist, width="stretch")


def _method_color(name: str) -> str:
    """Get color for a method by label name."""
    label_to_key = {"Mean-Variance": "mv", "Risk Parity": "rp",
                    "HRP": "hrp", "Black-Litterman": "bl"}
    return METHOD_COLORS.get(label_to_key.get(name, ""), TOKENS["text_muted"])


def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' to 'R,G,B' string for rgba()."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r},{g},{b}"
