"""Diagnosis after the HOLDOUT run (rules unchanged): gross bounce per H1 event by period, before costs."""
import numpy as np
import pandas as pd

import lab
import schemes as S

d = lab.load()
c = d["sol"]["close"]
b = d["btc"]["close"].reindex(c.index)
r, rb = np.log(c).diff(), np.log(b).diff()
z, zb = r / r.rolling(168).std().shift(1), rb / rb.rolling(168).std().shift(1)
ev = c.index[((z < -2.5) & (zb < -1.5)).fillna(False).values]
fwd = (np.log(c).shift(-1) - np.log(c)).reindex(ev)
for name, (a, bnd) in lab.PERIODS.items():
    x = fwd[(fwd.index >= pd.Timestamp(a, tz="UTC")) & (fwd.index < pd.Timestamp(bnd, tz="UTC"))].dropna()
    print(f"{name:10} events {len(x):4} gross next hour {x.mean()*1e4:+6.1f} bp (t{x.mean()/x.std()*np.sqrt(len(x)):+.1f}), win {np.mean(x>0):.0%} | cost 12 bp")
big = fwd[fwd.index >= pd.Timestamp("2025-07-01", tz="UTC")].sort_index()
print("largest HOLDOUT events (SOL hour z, next-hour return):")
zz = z.reindex(big.index)
for t in zz.nsmallest(6).index:
    print(f"  {t:%Y-%m-%d %H:%M}  z {zz[t]:+.1f}  next hour {big[t]*1e4:+.0f} bp")
