"""Exploration 1 (DISCOVERY only): how SOL returns are structured, before any scheme."""
import numpy as np
import pandas as pd

import lab

d = lab.load()
c = lab.period(d["sol"]["close"], "DISCOVERY")
r = np.log(c).diff().dropna()
half = r.index[len(r) // 2]
t = lambda x: x.mean() / x.std() * np.sqrt(len(x))

print("== A. hour of day (UTC): mean bp/h, t-stat, first half vs second half")
g = r.groupby(r.index.hour)
for h, x in g:
    a, b = x[x.index < half], x[x.index >= half]
    print(f"  {h:02d}h {x.mean()*1e4:+6.2f}bp t{t(x):+5.1f} | {a.mean()*1e4:+6.2f} {b.mean()*1e4:+6.2f}")

print("== B. day of week: mean bp/day")
dr = np.log(c.resample("D").last()).diff().dropna()
for k, x in dr.groupby(dr.index.dayofweek):
    print(f"  {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][k]} {x.mean()*1e4:+7.1f}bp t{t(x):+5.1f} n{len(x)}")

print("== C. autocorrelation")
print("  hourly lags:", {k: round(r.autocorr(k), 3) for k in (1, 2, 3, 6, 12, 24)})
print("  daily lags :", {k: round(dr.autocorr(k), 3) for k in (1, 2, 3, 5, 7)})
w = np.log(c.resample("W").last()).diff().dropna()
print("  weekly lags:", {k: round(w.autocorr(k), 3) for k in (1, 2, 4)})

print("== D. after extreme hours (|r| > 3σ of last week): forward log return, bp")
sd = r.rolling(168).std().shift(1)
z = r / sd
for name, m in (("drop<-3σ", z < -3), ("jump>+3σ", z > 3)):
    idx = r.index[m.fillna(False)]
    row = []
    for k in (1, 4, 12, 24, 72):
        fwd = (np.log(c).shift(-k) - np.log(c)).reindex(idx).dropna()
        row.append(f"{k}h {fwd.mean()*1e4:+6.0f} (t{t(fwd):+.1f})")
    print(f"  {name:9} n{len(idx):4} " + " · ".join(row))

print("== E. weekend vs weekday hourly vol and mean")
wk = r.index.dayofweek >= 5
print(f"  weekday mean {r[~wk].mean()*1e4:+.2f}bp vol {r[~wk].std()*1e4:.0f}bp | weekend mean {r[wk].mean()*1e4:+.2f}bp vol {r[wk].std()*1e4:.0f}bp")

print("== F. funding (per 8h)")
f = lab.period(d["funding"]["rate"], "DISCOVERY")
print("  quantiles:", {q: f"{f.quantile(q)*1e4:+.2f}bp" for q in (0.05, 0.25, 0.5, 0.75, 0.95)}, f"APR mean {f.mean()*3*365:+.1%}")
print("  share of payments > 0:", round((f > 0).mean(), 3))

print("== G. SOL/BTC: daily autocorrelation of the relative return, and deviation from 7d mean → next 3d")
b = lab.period(d["btc"]["close"], "DISCOVERY")
rel = np.log(c / b)
rd = rel.resample("D").last()
print("  rel daily autocorr:", {k: round(rd.diff().autocorr(k), 3) for k in (1, 2, 3, 7)})
dev = (rd - rd.rolling(7).mean())
fwd = rd.shift(-3) - rd
for qn, m in (("bottom 20%", dev < dev.quantile(.2)), ("top 20%", dev > dev.quantile(.8))):
    x = fwd[m].dropna()
    print(f"  dev {qn}: next 3d relative {x.mean()*1e4:+.0f}bp t{t(x):+.1f} n{len(x)}")

print("== H. Solana DEX volume growth (7d vs prior 28d) → next 7d SOL return")
dx = d["dexs"]
dv = np.log(dx.rolling(7).mean() / dx.rolling(28).mean().shift(7))
dc = np.log(c.resample("D").last())
nxt = (dc.shift(-8) - dc.shift(-1))            # day D value is known after D ends: trade from D+1
j = pd.concat([dv, nxt], axis=1, keys=["g", "n"]).dropna()
j = j[(j.index >= pd.Timestamp("2021-01-01", tz="UTC")) & (j.index < pd.Timestamp("2024-07-01", tz="UTC"))]
print("  spearman:", round(j.corr("spearman").iloc[0, 1], 3), "n", len(j))
for qn, m in (("growth top 20%", j.g > j.g.quantile(.8)), ("growth bottom 20%", j.g < j.g.quantile(.2))):
    print(f"  {qn}: next 7d {j.n[m].mean()*1e4:+.0f}bp vs all {j.n.mean()*1e4:+.0f}bp")
