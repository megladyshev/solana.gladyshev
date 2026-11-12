# Research log

Every test, in order. DISCOVERY unless marked.

## 1. explore1.py — structure of SOL returns (DISCOVERY)
- Hour of day: no stable effect (the two halves of the period disagree on sign). **Dropped.**
- Day of week: Sat +83 bp (t 2.2), Sun +74 bp (t 1.9); weekend hourly vol is lower (121 vs 147 bp). → candidate **weekend drift**.
- Autocorrelation: none at the hour or day scale; **weekly lag-1 +0.175** (about t 2.4). → candidate **weekly momentum**.
- After hours below −3σ: +61 bp the next hour (t 2.9), +90 bp after 4 h (t 2.5), n 230. After +3σ: nothing at 1–4 h. → candidate **liquidation-flush bounce**.
- Funding: median +1 bp per 8 h, mean +0.4% APR. A plain carry trade earns nothing on SOL. **Dropped.**
- SOL/BTC: deviation above the 7-day mean → next 3 days +230 bp (t 3.1), but the bottom quintile is also positive (+79). Needs demeaning. → **check**.
- Solana DEX volume growth: top quintile → next 7 days +176 bp, bottom quintile → −151 bp (average −39), Spearman only 0.04. → candidate **on-chain activity momentum**.

## 2. explore2.py — wave 1 as schemes, with costs (DISCOVERY)
Buy-and-hold has Sharpe 1.64 (SOL ×100, drawdown −96%), so Sharpe alone flatters any long-biased scheme. From here on the yardstick is alpha over beta × market.
- Vol-managed holding (capped at 1×) lags plain holding. **Dropped.**
- DEX-volume momentum: Sharpe 0.2–0.5, negative in 2021–2022. **Dropped.**
- SOL/BTC relative momentum: the edge is mostly SOL beating BTC overall; 2024 negative. **Dropped.**

## 3. explore3.py — alpha (DISCOVERY)
- Weekend Fri 23 → Sun 23: α +0.45/yr, t 1.7, **positive in all 4 years** (0.69, 0.15, 0.35, 0.25). → keep.
- Weekly momentum 4w: alpha concentrated in 2021, negative in 2023–2024. **Dropped.**
- Flush z−3 hold 1 h: α +0.27, t 2.6.
  - Split: **market-wide flush** (BTC also below −1.5σ) α +0.31, t 2.7, positive in all 4 years. **SOL-only flush** α −0.04. → keep the market-wide version. Reading: forced liquidations across the market overshoot; an idiosyncratic SOL drop is news.
- BTC→SOL catch-up: negative, mostly costs from about 1,700 trades a year. **Dropped.**
- Funding regime (crowd): t 0.3. **Dropped.**
- Sessions, breakouts in calm regimes, calm-fade / storm-follow: negative alpha after costs. **Dropped.**
- 48 h after the monthly options expiry: α +0.16, t 1.2, positive every year. → weak; keep as a reserve.

## 4. explore4.py — robustness (DISCOVERY)
- Market-wide flush: every neighbour on the 3×3 grid (SOL −2.5/−3/−3.5σ × BTC −1/−1.5/−2σ) has positive alpha. Hold 1–4 h works; 8 h does not. Costs ×2: α +0.25 (t 2.2); ×3: +0.19 (t 1.7). **Entry one hour late: α −0.04, the edge is gone** → it lives in the first hour after the crash candle closes. Needs an entry-timing check on minute data.
- Weekend: all three windows positive (α 0.31–0.45, t 1.2–1.7); Saturday or Sunday alone is weak. Marginal.

## 5. explore5.py — entry timing and funding flows (DISCOVERY)
- Market-wide flush on 1-minute bars (183 events, hold 60 min, net of 0.12% round trip): entry at the close +68 bp (t 3.1); +1 min +51 (t 2.7); +5 min +56 (t 3.0); +15 min +82 (t 3.4). The bounce lasts about an hour; a 15-minute delay still works, a 60-minute one does not.
- Funding timing: after a payment whose last known rate was > +0.02%, the next hour gains +17.9 bp (t 2.4, n 584). For rates 0..0.02% and below 0: nothing. → candidate **post-funding re-entry** (longs who closed to dodge the payment come back).

## 6. explore6.py — (DISCOVERY)
- Post-funding re-entry as a scheme: unstable (2023 negative at every threshold; 2022 had no such payments). **Dropped.**
- Squeeze continuation after a market-wide jump: t ≤ 0.7. **Dropped.**
- BTC < −3σ → long SOL for 4 h: α +0.46, t 2.4, positive in all years. **Same family as the market-wide flush** (the trigger hours mostly coincide); counted as one idea, not a new one.

## 7. explore7.py — other families (DISCOVERY)
- **Turn of the month**: last 3 days α +0.31, t 2.1, positive in all 4 years (0.49, 0.37, 0.23, 0.05). First 3 days α +0.41, t 2.2, but negative in 2022 and 2024. Mid-month (10–20) α −0.31. Known anomaly in equities (month-end flows and rebalancing); for crypto add the last-Friday options expiry. → candidate.
- Solana stablecoin growth, network-fee momentum: t ≤ 0.2. **Dropped.**
- SOL/ETH: fading 3-day extremes loses (t −1.9); following them α +0.44, t 1.7, but only 2 of 4 years positive. **Dropped.**
- Daily market-wide flush (1–3 days): t ≤ 0.1. The flush edge lives at the hourly scale only.

## 8. explore8.py — turn-of-month neighbours (DISCOVERY)
Last 2/3/4/5 days: α 0.20/0.31/0.29/0.20. Last 3 + first 1/2/3 days: α 0.54 (t 3.0)/0.58 (3.0)/0.73 (3.4). Last 4 + first 1: 0.52 (2.5). Last 3 + first 1 is positive in every year (0.97, 0.47, 0.34, 0.26).

## Going to VALIDATION (fixed before running it, 3 variants each)
| Scheme | A | B | C |
|---|---|---|---|
| Market-wide flush | SOL<−3σ & BTC<−1.5σ, hold 1 h | same, hold 4 h | SOL<−2.5σ & BTC<−1.5σ, hold 1 h |
| Weekend | Fri 23 → Sun 23 | Fri 22 → Sun 22 | Sat 00 → Mon 00 |
| Turn of month | last 3 + first 1 days | last 3 days | last 4 + first 1 days |
Pass: alpha > 0 on VALIDATION. The variant with the best VALIDATION alpha goes to HOLDOUT.

Tried on DISCOVERY so far: about 60 variants across 17 ideas.

## VALIDATION, round 1 (2024-07 → 2025-06), run once
Buy-and-hold: Sharpe 0.47, CAGR +1.7%, drawdown −66%.
| Scheme | A | B | C | Verdict |
|---|---|---|---|---|
| Market-wide flush | α +0.20 (t 1.4), SR 1.55 | −0.01 | **+0.25 (t 1.4)**, SR 1.42 | **passes** (A and C) |
| Weekend | −0.00 | −0.11 | −0.01 | **fails**, dropped |
| Turn of month | **−0.49 (t −2.3)** | −0.30 | −0.47 | **fails**, dropped: the DISCOVERY effect was not real |

## 9. explore9.py — liquidity mechanics (DISCOVERY)
Stop-hunt sweep and reclaim, failed breakout, volume-conditioned 4 h reversal or momentum, absorption: none has positive alpha with a t above 1.1 that holds across years. **All dropped.**

## 10. explore10.py — ecosystem, funding spread, storms (DISCOVERY)
- RAY → SOL lead: negative (t −3.3 at 1 h), mostly costs; no lead. **Dropped.**
- SOL−BTC funding spread pair: t ≤ 0.4. **Dropped.**
- Hold SOL except in high-vol storms: α −0.03. **Dropped.**
- Market-wide flush as a market-neutral pair (long SOL / short BTC): α +0.22, t 2.5. Confirms the flush mechanism; same scheme, not a new one.

## 11. explore11.py — (DISCOVERY)
OI deleveraging (t ≤ 1.2), residual reversal or momentum at 24/72 h (one-year effects), daily ±3σ overreaction (t 0.2): **all dropped.**
Tally: about 95 variants across 25 ideas; one robust family (market-wide flush).

## 12. explore12.py — trend filter (DISCOVERY)
Hold above the 50/100/200-day average: total α +0.33/−0.23/+0.38 but **negative in most single years**. The total comes from timing between years, not a repeatable edge. **Dropped.**

## 13. explore14.py — perp discount in a crash (DISCOVERY)
A perp discount to spot (premium z < −2/−3) with a price drop: α +0.05/+0.10; outside flush hours it adds nothing (t ≤ 1.5); flush hours without a discount carry the edge (α +0.21, t 2.2). **Dropped as a separate idea**; it does not sharpen the flush either.

## 14. explore15.py — illiquidity premium (DISCOVERY)
Long when weekly Amihud illiquidity is in the top 30%/20% of the last year: α −0.61 (t −2.4) / −0.34. The opposite of the equity result. **Dropped.**

## 15. explore13.py — market-wide flush on 15-minute bars (DISCOVERY)
SOL<−3σ & BTC<−1.5σ (σ of the last 4 days), hold 60 min: α +0.64, t 3.5, SR 2.16, 1,234 trades, every year ≥ 0 (1.51, 0.66, 0.01, 0.37). Hold 30/120 min and stricter thresholds: all positive (t 1.5–2.7). **SOL-only 15-minute drops: α 0.00** → the same mechanism, at another time scale.

## Going to VALIDATION, round 2 (new implementations of the flush; fixed before running)
| Scheme | A | B | C |
|---|---|---|---|
| Flush, 15-minute bars | SOL<−3σ & BTC<−1.5σ, hold 60 min | same, hold 120 min | SOL<−3σ & BTC<−2.5σ, hold 60 min |
| Flush, market-neutral pair (long SOL / short BTC), 1 h | SOL<−3σ & BTC<−1.5σ, hold 1 h | same, hold 4 h | SOL<−2.5σ & BTC<−1.5σ, hold 1 h |

Tally before HOLDOUT: about 115 variants across 30 ideas. Survivors are all one mechanism.

## VALIDATION, round 2, run once
| Scheme | A | B | C | Verdict |
|---|---|---|---|---|
| Flush 15 m | +0.37 (t 1.3) | +0.28 (t 1.0) | **+0.50 (t 1.9)**, SR 1.97 | passes → C |
| Flush pair 1 h | **+0.11 (t 1.2)** | −0.01 | +0.08 | passes → A |
Rules for HOLDOUT are in HYPOTHESES.md, committed before the unlock.

