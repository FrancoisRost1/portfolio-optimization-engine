# CLAUDE.md — Portfolio Optimization Engine (Project 8)

> Local source of truth. Read this fully before writing any code.
> Master CLAUDE.md is at `~/Documents/CODE/CLAUDE.md` — carries only a summary.

---

## What this project does

Advanced portfolio construction engine that compares four optimization methods
(Mean-Variance, Risk Parity, HRP, Black-Litterman) across three covariance
estimation techniques (Sample, Ledoit-Wolf shrinkage, RMT cleaning). Includes
a rolling backtest, full risk decomposition, and an 8-tab Bloomberg dark mode
Streamlit dashboard.

**Core thesis:** Naive mean-variance with sample covariance is garbage for real
portfolios. This project proves it by showing how shrinkage, robust methods,
and Bayesian priors produce stabler, better-diversified allocations.

---

## Asset universe

14 ETFs across 5 asset classes:

| Asset class  | Tickers                 | Default max class exposure |
|-------------|--------------------------|---------------------------|
| Equities    | SPY, EFA, EEM, IWM       | 60%                       |
| Bonds       | TLT, IEF, LQD, HYG, EMB | 60%                       |
| Commodities | GLD, SLV, DBC            | 30%                       |
| REITs       | VNQ                      | 20%                       |
| FX          | UUP                      | 15%                       |

All tickers, class mappings, and caps live in `config.yaml`.

---

## Data source

- **Provider:** yfinance (adjusted close prices)
- **Lookback:** configurable in `config.yaml` (default: 10 years)
- **Returns:** daily log returns for covariance estimation; daily arithmetic returns for performance
- **Frequency:** daily prices, resampled as needed for rebalance
- **Signal timing:** signal at close(t), trade at close(t+1) — strict no-lookahead (consistent with Projects 5/6/7)

---

## Covariance estimation methods

### 1. Sample covariance (comparison baseline only)

```
Σ_sample = (1 / (T-1)) × Xᵀ X
```

where X = demeaned daily returns matrix (T × N).

**Known problems:** noisy eigenvalues, unstable inverse, poor out-of-sample.
Used ONLY as a comparison baseline in the Naive vs Robust tab — never as default.

### 2. Ledoit-Wolf shrinkage (DEFAULT)

```
Σ_LW = α × F + (1 - α) × Σ_sample
```

- F = structured target (constant correlation model)
- α = optimal shrinkage intensity (Ledoit-Wolf 2004 analytical formula)
- Use `sklearn.covariance.LedoitWolf` or implement from paper
- **This is the default covariance for all optimizers unless user overrides.**

### 3. RMT / Marchenko-Pastur cleaning (advanced option)

```
λ_max = σ² × (1 + √(N/T))²
λ_min = σ² × (1 - √(N/T))²
```

- Eigendecompose Σ_sample
- Replace eigenvalues below λ_max with their average (noise removal)
- Reconstruct: Σ_clean = Q × Λ_clean × Qᵀ
- Preserve trace (total variance) by rescaling
- σ² = median eigenvalue (robust noise estimate)

---

## Optimization methods

### 1. Mean-Variance (Markowitz)

**CRITICAL CONSTRAINT:** Never use raw historical mean returns as expected returns.
Two permitted return estimators:

**Option A — Shrinkage returns (default for MV):**
```
μ_shrink = δ × μ_grand + (1 - δ) × μ_historical
μ_grand  = cross-sectional mean of all asset mean returns
δ        = shrinkage intensity (default 0.5, from config)
```
This pulls extreme return estimates toward the cross-sectional mean.

**Option B — Black-Litterman implied returns:**
```
π = λ × Σ × w_mkt
```
(see BL section below). MV can consume BL posterior returns as input.

**Optimizer:**
```
max  w'μ - (γ/2) × w'Σw
s.t. Σ w_i = 1
     0 ≤ w_i ≤ w_max_i                 (per-asset bounds, default 0%–30%)
     Σ w_i (class=k) ≤ cap_k           (per-class caps)
     Σ |w_new - w_old| ≤ max_turnover   (optional, configurable, off by default)
```
- γ = risk aversion coefficient (default 2.5, from config)
- Solver: `scipy.optimize.minimize` with SLSQP method
- Generate efficient frontier by sweeping target returns

### 2. Risk Parity (Equal Risk Contribution)

```
w* = argmin Σᵢ Σⱼ (w_i × (Σw)_i - w_j × (Σw)_j)²
s.t. Σ w_i = 1, w_i ≥ 0
```

- Each asset contributes equally to total portfolio variance
- Risk contribution of asset i: RC_i = w_i × (Σw)_i / (w'Σw)
- Target: RC_i = 1/N for all i
- Solver: `scipy.optimize.minimize` with SLSQP
- Per-asset bounds and per-class caps applied as secondary constraints

### 3. Hierarchical Risk Parity (HRP)

Implementation follows López de Prado (2016):

```
Step 1 — Cluster:   hierarchical clustering on correlation distance
                    d(i,j) = √(0.5 × (1 - ρ_ij))
                    linkage method: single (default, configurable)
Step 2 — Quasi-diag: reorder covariance matrix by dendrogram leaf order
Step 3 — Bisection:  recursive bisection allocating inverse-variance weights
                    within each cluster branch
```

- No optimizer needed — purely algorithmic (no matrix inversion)
- Naturally produces diversified, stable allocations
- Per-asset bounds enforced as post-allocation clipping + renormalization
- Per-class caps enforced as post-allocation scaling

### 4. Black-Litterman

**Step 1 — Implied equilibrium returns:**
```
π = λ × Σ × w_mkt
```
- λ = risk aversion (default 2.5, from config — same γ as MV)
- Σ = covariance matrix (Ledoit-Wolf default)
- w_mkt = market-cap weights from yfinance (fetched at runtime)
- **Non-negotiable:** must use real market-cap weights, not equal-weight

**Step 2 — Incorporate investor views:**
```
Views expressed as: Q = P × μ + ε, where ε ~ N(0, Ω)
P = pick matrix (K × N), K views on N assets
Q = view vector (K × 1), expected returns per view
Ω = uncertainty of views (diagonal, default: τ × P × Σ × Pᵀ)
τ = scaling factor (default 0.05, from config)
```

**Step 3 — Posterior returns:**
```
μ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ × [(τΣ)⁻¹π + PᵀΩ⁻¹Q]
Σ_BL = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ + Σ
```

**Step 4 — Optimize:** Feed μ_BL into MV optimizer (same constraints apply).

**Views input:**
- Default views in `config.yaml` (dict of ticker: expected_return)
- Dashboard sidebar allows interactive override (add/remove/edit views)
- Confidence per view adjustable (scales diagonal of Ω)

---

## Portfolio constraints (all configurable in config.yaml)

| Constraint               | Default           | Notes                               |
|--------------------------|-------------------|-------------------------------------|
| Long-only                | Yes               | w_i ≥ 0 always                      |
| Per-asset floor           | 0%                | Minimum weight per asset             |
| Per-asset cap             | 30%               | Maximum weight per asset             |
| Per-class cap (equities)  | 60%               | Sum of equity weights                |
| Per-class cap (bonds)     | 60%               | Sum of bond weights                  |
| Per-class cap (commodities)| 30%              | Sum of commodity weights             |
| Per-class cap (REITs)     | 20%               | Sum of REIT weights                  |
| Per-class cap (FX)        | 15%               | Sum of FX weights                    |
| Max turnover              | 100% (off)        | Optional; enforced when < 100%       |

Per-asset and per-class constraints enforced simultaneously in the optimizer.
For HRP (non-optimizer), applied as post-allocation clipping + renormalization.

---

## Post-optimization overlays

### Volatility targeting (optional, off by default)

```
σ_portfolio = √(w'Σw) × √252      (annualized)
scale_factor = σ_target / σ_portfolio
w_final = w_optimized × scale_factor
```

- σ_target = target annualized volatility (default 10%, from config)
- If scale_factor > 1, cap at 1.0 (no leverage in long-only context)
- Toggle on/off in config.yaml
- Applied AFTER optimization, not baked into objective function

---

## Rolling backtest

### Estimation & rebalance

- **Estimation window:** rolling (default 252 trading days, configurable) or expanding
- **Rebalance frequency:** configurable (default monthly — last trading day of month)
- **Window type:** `rolling` or `expanding` (from config)

### Signal timing (CRITICAL)

```
At close of day t:
  1. Observe prices through t
  2. Estimate covariance on [t - window, t]
  3. Run optimizer → produce target weights for t+1
At close of day t+1:
  4. Execute trades at close(t+1) prices
  5. Measure returns from t+1 forward
```

No lookahead. No same-day trading. Consistent with Projects 5, 6, 7.

### Transaction costs

```
turnover_t = Σ |w_new_i - w_old_i|
cost_t     = turnover_t × tc_bps / 10000
```
- tc_bps = transaction cost in basis points (default 10, from config)
- Applied to returns on rebalance days only
- Turnover tracked per rebalance for time-series analysis

### Benchmarks

| Benchmark         | Construction                                     |
|-------------------|--------------------------------------------------|
| SPY buy-and-hold  | 100% SPY, no rebalance                           |
| 60/40             | 60% SPY + 40% AGG, monthly rebalance             |
| Equal-weight (1/N)| 1/14 per ETF, same rebalance frequency            |
| Static risk-parity| Risk parity weights computed once on full sample, held static |

**Note:** AGG is used in the 60/40 benchmark only — not in the optimization universe.

---

## Performance metrics

Computed on daily returns, annualized where applicable:

| Metric              | Formula / Notes                                         |
|---------------------|---------------------------------------------------------|
| CAGR                | (ending_value / starting_value)^(252/T) - 1             |
| Annualized Vol      | std(daily_returns) × √252                               |
| Sharpe Ratio        | (CAGR - rf) / Vol; rf from config (default 0.04)        |
| Sortino Ratio       | (CAGR - rf) / downside_vol                              |
| Max Drawdown        | max peak-to-trough decline                              |
| Calmar Ratio        | CAGR / |Max Drawdown|                                   |
| CVaR 95%            | mean of returns below 5th percentile (daily)            |
| Tracking Error      | std(portfolio_returns - benchmark_returns) × √252       |
| Information Ratio   | (CAGR_port - CAGR_bench) / Tracking Error               |
| Herfindahl Index    | Σ w_i² (portfolio concentration; lower = more diversified)|
| Turnover            | Σ |w_new - w_old| per rebalance                         |

---

## Dashboard (8 tabs — Bloomberg dark mode)

### Tab 1: Overview
- Summary metrics table for all 4 optimizers + 4 benchmarks
- Current optimal weights (bar chart, stacked by asset class)
- Key takeaway: which method wins on Sharpe, which on stability

### Tab 2: Efficient Frontier
- Efficient frontier curve (scatter: Vol vs Return)
- Overlay dots for MV, RP, HRP, BL portfolios + benchmarks
- Interactive: hover for portfolio details
- Toggle covariance method (sample / LW / RMT) to see frontier shift

### Tab 3: Weight Comparison
- Side-by-side weight bar charts for all 4 methods
- Asset class allocation pie charts
- Weight stability over time (for backtest): rolling weight heatmap

### Tab 4: Risk Decomposition
- Risk contribution bar chart per asset for each method
- Marginal risk contribution table
- Percentage risk contribution vs percentage weight (diversification check)
- Herfindahl index comparison across methods

### Tab 5: Backtest
- Cumulative return lines (all methods + benchmarks)
- Drawdown chart
- Rolling Sharpe (12-month window)
- Metrics summary table
- Turnover time series (bar chart per rebalance date)
- Turnover distribution histogram

### Tab 6: Covariance Lab
- Correlation heatmap (toggleable: sample / LW / RMT)
- Eigenvalue spectrum bar chart with Marchenko-Pastur bounds overlay
- Condition number comparison (sample vs LW vs RMT)
- Shrinkage intensity display (LW α value)
- Number of eigenvalues cleaned (RMT)

### Tab 7: Naive vs Robust
- Head-to-head comparison: MV+Sample vs MV+LW vs HRP
- Backtest curves overlaid
- Weight stability comparison (rolling weight std per asset)
- Out-of-sample Sharpe degradation from in-sample
- Key message: sample covariance → concentrated, unstable portfolios

### Tab 8: Black-Litterman Views
- Current equilibrium returns (π) bar chart
- Interactive view editor: add/remove/edit views in sidebar
- Prior vs posterior return comparison (grouped bar chart)
- Prior vs posterior weight comparison
- View confidence slider (adjusts Ω diagonal)
- Posterior covariance effect visualization

---

## File structure

```
portfolio-optimization-engine/
├── CLAUDE.md                         ← this file (project spec)
├── config.yaml                       ← all parameters
├── main.py                           ← orchestrator only (no logic)
├── requirements.txt
├── src/
│   ├── data_loader.py                ← yfinance fetch + clean + cache
│   ├── returns.py                    ← return computation (log + arithmetic)
│   ├── covariance.py                 ← 3 methods: sample, LW, RMT
│   ├── expected_returns.py           ← shrinkage returns + BL implied returns
│   ├── optimizer_mv.py               ← mean-variance with constraints
│   ├── optimizer_rp.py               ← risk parity (equal risk contribution)
│   ├── optimizer_hrp.py              ← hierarchical risk parity
│   ├── optimizer_bl.py               ← Black-Litterman (views → posterior → MV)
│   ├── constraints.py                ← constraint builder (asset + class bounds + turnover)
│   ├── backtester.py                 ← rolling backtest engine
│   ├── benchmarks.py                 ← benchmark portfolio construction
│   ├── metrics.py                    ← all performance + risk metrics
│   ├── risk_decomposition.py         ← risk contribution, marginal risk, Herfindahl
│   ├── vol_target.py                 ← optional post-optimization vol targeting overlay
│   └── efficient_frontier.py         ← frontier generation (sweep target returns)
├── utils/
│   ├── config_loader.py              ← load config.yaml, pass as dict
│   └── helpers.py                    ← shared utilities (date handling, etc.)
├── app/
│   ├── app.py                        ← Streamlit main app (8 tabs)
│   ├── tab_overview.py
│   ├── tab_frontier.py
│   ├── tab_weights.py
│   ├── tab_risk.py
│   ├── tab_backtest.py
│   ├── tab_covariance.py
│   ├── tab_naive_vs_robust.py
│   ├── tab_bl_views.py
│   └── style_inject.py              ← Bloomberg dark mode (from DESIGN.md)
├── tests/
│   ├── test_covariance.py
│   ├── test_expected_returns.py
│   ├── test_optimizer_mv.py
│   ├── test_optimizer_rp.py
│   ├── test_optimizer_hrp.py
│   ├── test_optimizer_bl.py
│   ├── test_constraints.py
│   ├── test_backtester.py
│   ├── test_metrics.py
│   ├── test_risk_decomposition.py
│   ├── test_vol_target.py
│   └── test_integration.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── cache/
├── outputs/
├── docs/
│   └── analysis.md
└── .streamlit/
    └── config.toml
```

---

## Simplifying assumptions (document these in code)

1. **No leverage:** long-only, weights sum to 1, vol target capped at scale ≤ 1.0
2. **No shorting:** w_i ≥ 0 always
3. **Market-cap proxy:** yfinance `info["marketCap"]` for BL equilibrium weights; if unavailable for an ETF, fall back to AUM or configurable manual override in config.yaml
4. **Transaction costs:** flat bps model, no bid-ask spread modeling
5. **No intraday:** daily close prices only
6. **Rebalance execution:** assume trades execute at close(t+1) — no slippage model
7. **Risk-free rate:** constant from config (default 4%), not fetched dynamically
8. **HRP constraints:** post-allocation clipping is an approximation — true constrained HRP would require modifying the bisection algorithm
9. **BL view uncertainty:** default Ω = τ × P × Σ × Pᵀ (proportional to prior uncertainty); user can override confidence per view
10. **Constrained HRP is approximate:** `clip_to_constraints` iteratively clips and renormalises HRP weights to satisfy per-asset and per-class caps. This is a practical approximation — a theoretically pure approach would require modifying the recursive bisection algorithm itself to be constraint-aware, which López de Prado (2016) does not address.
11. **BL market-cap weights are current-day proxies:** `fetch_market_caps()` retrieves today's yfinance `marketCap`/`totalAssets` for BL equilibrium weights. These are NOT point-in-time historical values. In a rolling backtest, BL uses the same current-day proxy at every rebalance — this is a known simplification. A production system would require a historical market-cap database. For point-in-time analysis and dashboard use, this is acceptable.

---

## Edge cases to handle

- **Missing data:** drop tickers with < 80% coverage in lookback window; warn user
- **Singular covariance:** if sample cov is singular, force Ledoit-Wolf or RMT
- **Negative eigenvalues:** after RMT cleaning, floor at small positive ε (1e-10)
- **Market-cap fetch failure:** fall back to equal-weight for BL equilibrium; log warning
- **Division by zero:** portfolio vol = 0 → Sharpe = np.nan, vol target scale = 1.0
- **No valid views:** if BL has no views, posterior = prior (pure implied returns)
- **Infeasible constraints:** if per-asset + per-class caps are contradictory, warn and relax class caps
- **Turnover constraint infeasible:** if max_turnover too tight on first rebalance (no prior weights), skip turnover constraint for first period

---

## Dependencies

```
pandas
numpy
scipy
scikit-learn          # LedoitWolf covariance
yfinance
pyyaml
streamlit
plotly
pytest
```

---

## Cross-project connections

| Direction | Project | What flows |
|-----------|---------|------------|
| OUT → | factor-backtest-engine (P3) | Covariance cleaning module (future upgrade) |
| IN ← | factor-backtest-engine (P3) | Factor computation pipeline (future, not v1) |
| REUSE | volatility-regime-engine (P5) | Vol targeting pattern |
| REUSE | tsmom-engine (P6) | Signal timing pattern (t/t+1) |
| REUSE | strategy-robustness-lab (P7) | Backtest engine pattern |

---

## Current state

- **Status:** 🔲 NOT STARTED — spec locked, ready for Phase 2 (scaffold)
- **Next step:** Create folder structure, drop in CLAUDE.md + config.yaml, scaffold all source modules

---

## Streamlit Cloud pass (2026-04-13)

- Migrated `use_container_width` → `width="stretch"/"content"` across the app layer ahead of the Streamlit deprecation-to-error window.
- Rewrote the footer at `app/app.py:109` from a multi-line indented triple-quoted f-string to single-line concatenated f-strings. Leading whitespace in `st.markdown` input is parsed as a Markdown code block even when `unsafe_allow_html=True`, which was leaking `</div>` as visible text in the Streamlit Cloud render.

---

*Project CLAUDE.md — Created: 2025-04-08*
*Last updated: 2026-04-13*
