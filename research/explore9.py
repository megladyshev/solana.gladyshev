"""Exploration 9 (DISCOVERY only): liquidity mechanics beyond the flush:
 stop-hunt sweep and reclaim, failed breakout, low/high-volume reversal (Campbell–Grossman–Wang), absorption."""
import numpy as np
import pandas as pd

import lab
import schemes as S

d = lab.load()
sol = d["sol"]
c, fund = sol["close"], d["funding"]
X = lab.period(c, "DISCOVERY")
BENCH = lab.pnl(pd.Series(1.0, index=X.index), X, None, 0)


def run(name, pos):
    p = lab.period(pos, "DISCOVERY")
    ret = lab.pnl(p, X, fund)
    a, s = lab.alpha(ret, BENCH), lab.stats(ret, p)
    ys = [lab.alpha(ret[ret.index.year == y], BENCH[BENCH.index.year == y]).get("alpha") for y in (2021, 2022, 2023, 2024)]
    print(f"{name:44} α {a['alpha']:+.2f} t{a['t']:+.1f} β{a['beta']:.2f} SR {s['sharpe']:+.2f} tr {s['trades']:4} | years {ys}")


hi, lo = sol["high"], sol["low"]
r = np.log(c).diff()
sd = r.rolling(168).std().shift(1)
lo24, hi24 = lo.rolling(24).min().shift(1), hi.rolling(24).max().shift(1)
lwick = np.log(np.minimum(sol["close"], c.shift(1)) / lo) / sd      # lower wick in σ (open ≈ previous close)
uwick = np.log(hi / np.maximum(sol["close"], c.shift(1))) / sd
print("-- stop-hunt sweep: low breaks the 24h low, close back above it, long lower wick → long")
for w in (1.0, 2.0):
    for h in (4, 12):
        run(f"sweep-reclaim low, wick>{w}σ hold{h}", S._hold_after((lo < lo24) & (c > lo24) & (lwick > w), h))
print("-- failed breakout: high breaks the 24h high, close back below it, long upper wick → short")
for w in (1.0, 2.0):
    run(f"failed breakout, wick>{w}σ short hold4", -S._hold_after((hi > hi24) & (c < hi24) & (uwick > w), 4))
print("-- volume-conditioned reversal on 4h moves (|r4| > 2σ)")
qv = np.log(sol["qvol"].rolling(4).sum())
vz = (qv - qv.rolling(720).mean().shift(4)) / qv.rolling(720).std().shift(4)
r4 = np.log(c).diff(4) / (sd * 2)
for nm, m in (("low volume (vz<0)", vz < 0), ("high volume (vz>1.5)", vz > 1.5)):
    run(f"fade 4h move, {nm}", S._hold_after(m & (r4 < -2), 4) - S._hold_after(m & (r4 > 2), 4))
    run(f"follow 4h move, {nm}", S._hold_after(m & (r4 > 2), 4) - S._hold_after(m & (r4 < -2), 4))
print("-- absorption: hour volume z > 3 while |r| < 0.5σ → follow the next hour's direction? (long only test)")
vh = np.log(sol["qvol"])
vhz = (vh - vh.rolling(168).mean().shift(1)) / vh.rolling(168).std().shift(1)
run("absorption → long 4h", S._hold_after((vhz > 3) & ((r / sd).abs() < 0.5), 4))
