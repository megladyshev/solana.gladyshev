# Research protocol (fixed before any test)

Goal: find 2–3 trading schemes on SOL that make money after costs, on data the search never saw.

## Periods
| Period | Dates | Use |
|---|---|---|
| DISCOVERY | 2021-01-01 → 2024-06-30 | free search, any number of tries, every try is logged in `LOG.md` |
| VALIDATION | 2024-07-01 → 2025-06-30 | filter: a scheme that fails here is dropped, and is never "fixed" until it passes |
| HOLDOUT | 2025-07-01 → now | sealed by code (`lab.period` raises). Opened **once**, after the rules are committed in `HYPOTHESES.md` |

Earlier work in this repo (the direction and volatility models) looked at 2025–2026 data through other features. None of the schemes below were tested on it.

## Accounting
- Position decided at hour t with data up to t, held over (t, t+1]. Archived data is used with its real publication lag (30 h for Binance archives, end of day + 1 h for daily on-chain data).
- Costs: 0.06% per unit of turnover (Binance perp taker 0.05% + slippage), plus the perp funding the position pays or receives.
- Compared with buy-and-hold over the same hours.

## What counts as "works"
A scheme passes when:
1. on DISCOVERY and VALIDATION its Sharpe is above buy-and-hold **or** it earns with clearly lower risk (Sharpe > 0.5 at under half the exposure), and
2. on HOLDOUT: Sharpe > 0, and the day-block bootstrap 90% interval of the excess over buy-and-hold (or over zero for market-neutral schemes) is reported honestly.

At most 4 schemes go to HOLDOUT. With k = 4, a single lucky pass is expected about 1 time in 5, so the report names how many passed out of how many tried.

## Ban list
- No parameter is tuned on VALIDATION beyond choosing between at most 3 variants fixed on DISCOVERY.
- No rule is changed after HOLDOUT is opened. A scheme that fails on HOLDOUT is reported as failed.
