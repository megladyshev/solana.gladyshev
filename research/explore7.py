"""Exploration 7 (DISCOVERY only): other families: calendar, Solana stablecoins and fees, SOL vs ETH, daily flush."""
import numpy as np
import pandas as pd

import lab
import schemes as S

d = lab.load()
c, fund = d["sol"]["close"], d["funding"]
X = lab.period(c, "DISCOVERY")
BENCH = lab.pnl(pd.Series(1.0, index=X.index), X, None, 0)
LAG = pd.Timedelta(hours=25)


def run(name, pos, close=c, f=fund, cost=lab.COST, bench=None):
    x = lab.period(close, "DISCOVERY")
    p = lab.period(pos, "DISCOVERY")
    ret = lab.pnl(p, x, f, cost)
    bn = BENCH if bench is None else bench
    a, s = lab.alpha(ret, bn), lab.stats(ret, p)
    ys = [lab.alpha(ret[ret.index.year == y], bn[bn.index.year == y]).get("alpha") for y in (2021, 2022, 2023, 2024)]
    print(f"{name:42} α {a['alpha']:+.2f} t{a['t']:+.1f} β{a['beta']:.2f} SR {s['sharpe']:+.2f} tr {s['trades']:4} | years {ys}")


idx = c.index
print("-- calendar")
run("first 3 days of month", pd.Series((idx.day <= 3).astype(float), index=idx))
run("last 3 days of month", pd.Series((idx.day >= idx.days_in_month - 2).astype(float), index=idx))
run("days 10–20 (mid-month)", pd.Series(((idx.day >= 10) & (idx.day <= 20)).astype(float), index=idx))


def daily_signal(s):
    s.index = s.index + LAG
    return s.astype(float).reindex(idx, method="ffill").fillna(0)


print("-- Solana stablecoin supply: weekly growth above its 90-day median → long")
st = d["stable"]["stable"]
g = np.log(st).diff(7)
run("stable growth > rolling median", daily_signal(g > g.rolling(90).median()))
run("stable growth > 0", daily_signal(g > 0))
print("-- Solana network fees: 7d avg above 28d avg → long")
fe = d["fees"]
run("fees 7d > 28d", daily_signal(fe.rolling(7).mean() > fe.rolling(28).mean()))

print("-- SOL vs ETH: 3-day relative move beyond ±2σ → fade it for 3 days (pair, double cost)")
e = d["eth"]["close"].reindex(idx)
pr = (1 + (c.pct_change() - e.pct_change()).fillna(0)).cumprod()
rel3 = np.log(c / e).diff(72)
zr = rel3 / rel3.rolling(2160).std()
pb = lab.pnl(pd.Series(1.0, index=lab.period(pr, "DISCOVERY").index), lab.period(pr, "DISCOVERY"), None, 0)
run("SOL/ETH fade 3d ±2σ hold72", S._hold_after(zr < -2, 72) - S._hold_after(zr > 2, 72), pr, None, 2 * lab.COST, pb)
run("SOL/ETH follow 3d ±2σ hold72", S._hold_after(zr > 2, 72) - S._hold_after(zr < -2, 72), pr, None, 2 * lab.COST, pb)

print("-- daily market-wide flush: day close below −2σ for SOL and −1.5σ for BTC → long next 1–3 days")
dc = c[idx.hour == 0]
db = d["btc"]["close"].reindex(dc.index)
zd = np.log(dc).diff() / np.log(dc).diff().rolling(60).std().shift(1)
zdb = np.log(db).diff() / np.log(db).diff().rolling(60).std().shift(1)
sig = ((zd < -2) & (zdb < -1.5)).reindex(idx).fillna(False)
for h in (24, 72):
    run(f"daily market flush hold{h}h", S._hold_after(sig, h))
