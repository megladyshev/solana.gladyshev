"""Exploration 3 (DISCOVERY only): alpha over exposure-matched holding, and a second wave of ideas.

Alpha = what a scheme earns beyond beta × market. On a market that rose ~100× this is the only fair yardstick
for long-biased timing schemes."""
import numpy as np
import pandas as pd

import lab
import schemes as S

d = lab.load()
c, fund = d["sol"]["close"], d["funding"]
sol = d["sol"]
b = d["btc"]["close"].reindex(c.index)
r = np.log(c).diff()
rb = np.log(b).diff()
vol168 = r.rolling(168).std()
X = lab.period(c, "DISCOVERY")
BENCH = lab.pnl(pd.Series(1.0, index=X.index), X, None, 0)       # gross market return per hour


def run(name, pos, cost=lab.COST):
    p = lab.period(pos, "DISCOVERY")
    ret = lab.pnl(p, X, fund, cost)
    s, a = lab.stats(ret, p), lab.alpha(ret, BENCH)
    ys = {}
    for y in (2021, 2022, 2023, 2024):
        m = ret.index.year == y
        if m.sum() > 200:
            ys[y] = lab.alpha(ret[m], BENCH[BENCH.index.year == y]).get("alpha")
    print(f"{name:34} α {a['alpha']:+.2f}/yr {a['alpha90']} t{a['t']:+.1f} β{a['beta']:.2f} | SR {s['sharpe']:+.2f} exp {s['exposure']:.0%} tr {s['trades']:5} | α by year {ys}")


print("-- reference: wave 1 on alpha")
run("weekend Fri23→Sun23", S.weekend(d))
run("weekly mom 4w long-only", S.weekly_mom(d, 4))
run("weekly mom 4w long/short", S.weekly_mom(d, 4, short=True))
run("flush z-3 hold1h", S.flush(d, -3, 1))
run("flush z-2.5 hold4h", S.flush(d, -2.5, 4))

print("-- 1. flush, split by volume spike and by whether BTC crashed too")
z = r / vol168.shift(1)
qv = np.log(sol["qvol"])
vz = (qv - qv.rolling(168).mean().shift(1)) / qv.rolling(168).std().shift(1)
zb = rb / rb.rolling(168).std().shift(1)
for nm, cond in (("with volume spike (vz>3)", vz > 3), ("no volume spike", vz <= 3),
                 ("SOL-only (BTC z>-1.5)", zb > -1.5), ("market-wide (BTC z<-1.5)", zb <= -1.5)):
    run(f"flush z-3 h1 {nm}", S._hold_after((z < -3) & cond, 1))

print("-- 3. BTC→SOL catch-up: last-hour BTC move SOL has not followed yet")
beta = (r.rolling(720).cov(rb) / rb.rolling(720).var()).shift(1)
gap = (beta * rb - r) / vol168.shift(1)          # >0: SOL lagged a BTC rise
for th in (1.0, 2.0):
    run(f"catch-up long gap>{th}σ h1", S._hold_after(gap > th, 1))
    run(f"catch-up l/s |gap|>{th}σ h1", S._hold_after(gap > th, 1) - S._hold_after(gap < -th, 1))

print("-- 4. funding sign regime (crowd): long only while last 3 payments averaged below 0")
f = fund["rate"].groupby(fund.index.floor("h")).sum().rolling(3).mean()
fneg = (f < 0).astype(float).reindex(c.index, method="ffill").fillna(0)
run("long while funding<0", fneg)
run("long while funding>=0", 1 - fneg)

print("-- 6. sessions")
h = c.index.hour
for nm, m in (("Asia 00–08", (h >= 0) & (h < 8)), ("Europe 08–13", (h >= 8) & (h < 13)), ("US 13–21", (h >= 13) & (h < 21)), ("late 21–24", h >= 21)):
    run(f"session {nm}", pd.Series(m.astype(float), index=c.index))

print("-- 7. monthly options expiry (last Friday 08:00 UTC): 48h before vs 48h after")
lastfri = pd.Series(c.index, index=c.index).apply(lambda t: t.dayofweek == 4 and t.hour == 8 and (t + pd.Timedelta(days=7)).month != t.month)
exp_t = c.index[lastfri.values]
pre = pd.Series(0.0, index=c.index); post = pd.Series(0.0, index=c.index)
for t0 in exp_t:
    pre[(c.index >= t0 - pd.Timedelta(hours=48)) & (c.index < t0)] = 1
    post[(c.index >= t0) & (c.index < t0 + pd.Timedelta(hours=48))] = 1
run("48h before monthly expiry", pre)
run("48h after monthly expiry", post)

print("-- 11. breakout after compression: 48h high broken while 24h vol < 168h vol")
hi48 = sol["high"].rolling(48).max().shift(1)
lo48 = sol["low"].rolling(48).min().shift(1)
calm = r.rolling(24).std() < vol168
for hold in (12, 24):
    run(f"breakout up+calm hold{hold}", S._hold_after((c > hi48) & calm, hold))
    run(f"breakout l/s+calm hold{hold}", S._hold_after((c > hi48) & calm, hold) - S._hold_after((c < lo48) & calm, hold))

print("-- 12. 4h reversal in calm regime vs momentum in stormy regime")
r4 = np.log(c).diff(4) / (vol168 * 2)
lowv = vol168 < vol168.rolling(2160).median()
run("calm: fade 4h moves >1.5σ, hold 4h", S._hold_after(lowv & (r4 < -1.5), 4) - S._hold_after(lowv & (r4 > 1.5), 4))
run("storm: follow 4h moves >1.5σ, hold 4h", S._hold_after(~lowv & (r4 > 1.5), 4) - S._hold_after(~lowv & (r4 < -1.5), 4))
