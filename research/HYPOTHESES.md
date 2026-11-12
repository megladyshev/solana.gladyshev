# Pre-registered rules for the single HOLDOUT run (2025-07-01 → now)

Written and committed before `HOLDOUT_UNLOCKED` exists. Nothing below changes after the run.

All three are one mechanism: **a market-wide liquidation flush overshoots, and SOL bounces within about an hour.**
An idiosyncratic SOL drop (BTC calm) does not bounce: it is information, not forced selling. Out of about 115 variants across 30 ideas, nothing else survived DISCOVERY plus VALIDATION.

| # | Scheme | Exact rule | Chosen on VALIDATION |
|---|---|---|---|
| H1 | Flush, 1-hour bars | At an hourly close: SOL log return < −2.5·σ and BTC log return < −1.5·σ (σ = std of each coin's hourly returns over the previous 168 h) → long SOL perp for 1 h | variant C, α +0.25 |
| H2 | Flush, 15-minute bars | At a 15-minute close: SOL < −3·σ and BTC < −2.5·σ (σ over the previous 384 bars) → long SOL perp for 60 min | variant C, α +0.50 |
| H3 | Flush, market-neutral pair | Rule as H1 but SOL < −3·σ; position = long SOL / short BTC (equal notional) for 1 h | variant A, α +0.11 |

Costs: 0.06% per unit of turnover per leg (0.12% round trip; ×2 for the pair), plus the funding the position pays or receives.

Pass (from PROTOCOL.md): HOLDOUT Sharpe > 0. The report gives alpha with its 90% day-block bootstrap interval, the trade count, and whether each year-half is positive. k = 3 schemes, highly correlated with each other: one passing alone is weak evidence; all three passing together is evidence for the mechanism, not for three independent edges.
