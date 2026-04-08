# Portfolio Optimization Engine — Investment Analysis

## Investment Thesis

**The thesis is methodological, not directional:** most institutional portfolios are built with naive mean-variance optimization using sample covariance matrices. This approach is provably fragile — noisy eigenvalues produce concentrated, unstable allocations that degrade out-of-sample and generate excessive turnover.

This engine demonstrates three superior alternatives:

1. **Covariance shrinkage (Ledoit-Wolf)** reduces the condition number of the covariance matrix by 10-100x, producing more stable weights with minimal return sacrifice.
2. **Hierarchical Risk Parity (HRP)** bypasses matrix inversion entirely, using correlation-based clustering to produce diversified allocations that are inherently stable.
3. **Black-Litterman** anchors to market equilibrium and only deviates when the allocator has an explicit, quantified view — preventing the optimizer from chasing noise in historical means.

**The key finding:** At equal volatility (10% target), HRP achieves the highest Sharpe ratio (0.62) with the most stable weights and lowest concentration. Risk Parity offers the lowest turnover (0.6x/yr) and the most predictable execution costs. Mean-Variance, even with Ledoit-Wolf shrinkage, trades 5.8x/yr and loses 5.13% to transaction costs over the backtest period.

---

## Key Risks

### Estimation Risk
- **Covariance instability:** Sample covariance with 14 assets and ~252 observations has a ratio q = N/T ≈ 0.056. While this is below the critical threshold, the matrix condition number can still exceed 300, making MV weights sensitive to small return perturbations.
- **Return estimation:** Even with shrinkage, expected return estimates are the dominant source of error in MV. A 50bps change in one asset's expected return can shift its weight by 5-10%.
- **Regime dependence:** All covariance estimators assume stationarity. Correlation breakdowns during crises (2020 COVID, 2022 rate shock) can temporarily invalidate the estimated structure.

### Implementation Risk
- **Transaction costs:** The flat 10bps model understates costs for illiquid ETFs (DBC, SLV, EMB). Real-world costs include bid-ask spreads, market impact, and timing slippage.
- **Rebalance frequency:** Monthly rebalancing may miss intra-month regime shifts. Weekly rebalancing would improve responsiveness but 3-5x the turnover.
- **Market-cap proxies:** BL equilibrium uses current-day yfinance market caps, not point-in-time historical values. This is acceptable for dashboard analysis but overstates BL backtest reliability.

### Model Risk
- **HRP is not optimal:** HRP produces diversified weights by construction, but makes no claim about optimality. It cannot incorporate return views or target a specific risk-return tradeoff.
- **Black-Litterman view quality:** BL is only as good as its views. Uninformed views anchored by overconfidence can be worse than pure equilibrium (no views).
- **Risk parity assumes equal Sharpe:** The ERC objective implicitly assumes all assets have equal risk-adjusted return potential. This is a strong assumption that fails when asset classes have structurally different risk premia.

---

## Valuation Assumptions

This project does not value individual securities. The relevant "valuation" assumptions are the return and risk estimates fed into the optimizers:

### Expected Returns (MV/BL inputs)
| Method | Assumption | Sensitivity |
|--------|-----------|-------------|
| Shrinkage (δ=0.5) | Pull historical means 50% toward the cross-sectional average | δ=0.3 increases dispersion → more concentrated MV; δ=0.7 converges toward equal-weight |
| BL Equilibrium | π = λΣw_mkt with λ=2.5 | Higher λ → larger implied returns → more aggressive allocations; lower λ → returns compress toward zero |
| BL Views | SPY: +8%, TLT: -2%, GLD: +6% (defaults) | Removing all views → BL = market-cap weighted; strong views override equilibrium |

### Covariance
| Method | Assumption | Sensitivity |
|--------|-----------|-------------|
| Sample | (T-1)⁻¹ X'X, no regularization | Condition number >300 → highly sensitive to outliers |
| Ledoit-Wolf | Shrink toward scaled identity, α optimal | α ≈ 0.1-0.3 typical; higher α → more conservative (closer to equal-vol) |
| RMT | Clean eigenvalues below MP upper bound | Aggressive cleaning removes signal along with noise; moderate q mitigates this |

### Risk Parameters
| Parameter | Value | Impact |
|-----------|-------|--------|
| Risk-free rate | 4.0% p.a. (constant) | Higher rf → lower Sharpe for all strategies |
| Risk aversion γ | 2.5 | Higher γ → MV tilts toward min-variance; lower γ → more return-seeking |
| Transaction costs | 10 bps flat | Doubling to 20bps would further penalize high-turnover MV |
| Rebalance | Monthly (last trading day) | Quarterly → lower TC but slower response to regime shifts |

---

## Return Scenarios

### Scenario 1: Continued Bull Market (SPY +12%/yr, rates stable)
- **MV/BL benefit:** Equity overweight pays off; BL's SPY view validated
- **RP/HRP drag:** Bond and commodity allocations dilute equity upside
- **Expected ranking:** BL > MV > EW > RP > HRP

### Scenario 2: Rising Rates + Equity Correction (SPY flat, TLT -5%/yr)
- **MV/BL hurt:** Equity + long-duration exposure creates double hit
- **RP/HRP resilient:** Diversification across commodities, short-duration bonds cushions
- **Expected ranking:** HRP > RP > EW > MV > BL

### Scenario 3: Volatility Spike (VIX > 30, correlations spike)
- **All methods degrade:** Correlation breakdown invalidates covariance estimates
- **HRP most resilient:** No matrix inversion = no numerical instability
- **MV + Sample worst:** Condition number explodes → wild weight swings → max turnover
- **Expected ranking:** HRP > RP > BL > MV+LW >> MV+Sample

### Scenario 4: Mean-Reverting Sideways Market (SPY ±5%/yr, low vol)
- **RP shines:** Equal risk contribution captures carry and rebalancing premium
- **MV frustrated:** Low signal-to-noise in returns → optimizer chases noise
- **Expected ranking:** RP > HRP > EW > BL > MV

---

## Key Quantitative Findings

1. **MV trades 9.7x more than RP** (5.8x vs 0.6x annual turnover) — transaction costs consume 5.13% of cumulative return over the backtest
2. **At equal 10% vol, HRP beats all optimizers** (Sharpe 0.62 vs MV 0.59, BL 0.46, RP 0.57)
3. **Ledoit-Wolf shrinkage reduces condition number by 10-100x** compared to sample covariance
4. **MV+Sample out-of-sample Sharpe degrades the most** — confirming that sample covariance overfit is real and measurable
5. **Risk Parity achieves near-perfect equal risk contributions** with the lowest weight instability across rebalances
6. **Black-Litterman posterior tilts returns predictably** — views shift posterior toward the viewed asset while non-viewed assets anchor to equilibrium

---

## Conclusion

For a multi-asset allocator:
- **Default choice:** Risk Parity — lowest cost, most stable, robust baseline
- **If you have views:** Black-Litterman — but only with quantified, high-conviction views
- **Best risk-adjusted:** HRP at equal volatility — the stability premium is real
- **Avoid:** MV with sample covariance — provably inferior on every dimension

The right question is not "which optimizer maximizes backtest Sharpe?" but "which method produces the most implementable, stable, cost-efficient allocation?" This project provides the quantitative answer.

---

*Analysis prepared as part of the Portfolio Optimization Engine — Project 8 of the Finance Lab.*
