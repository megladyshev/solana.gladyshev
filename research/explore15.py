"""Exploration 15 (DISCOVERY only): illiquidity premium in time (Amihud): when SOL is unusually illiquid
(big moves per dollar traded), are the next days' returns higher?"""
import numpy as np
import pandas as pd

import lab

d = lab.load()
sol = d["sol"]
c, fund = sol["close"], d["funding"]
X = lab.period(c, "DISCOVERY")
BENCH = lab.pnl(pd.Series(1.0, index=X.index), X, None, 0)
r = np.log(c).diff().abs()
amihud = (r / sol["qvol"]).rolling(24 * 7).mean()           # weekly average |return| per dollar
az = np.log(amihud) - np.log(amihud).rolling(24 * 90).mean()
for q in (0.7, 0.8):
    thr = az.rolling(24 * 365, min_periods=24 * 90).quantile(q)
    pos = (az > thr).astype(float).where(c.index.hour == 0).ffill().fillna(0)     # decided once a day
    p = lab.period(pos, "DISCOVERY")
    ret = lab.pnl(p, X, fund)
    a, s = lab.alpha(ret, BENCH), lab.stats(ret, p)
    ys = [lab.alpha(ret[ret.index.year == y], BENCH[BENCH.index.year == y]).get("alpha") for y in (2021, 2022, 2023, 2024)]
    print(f"illiquid above {q:.0%} quantile → long: α {a['alpha']:+.2f} t{a['t']:+.1f} β{a['beta']:.2f} SR {s['sharpe']:+.2f} exp {s['exposure']:.0%} tr {s['trades']} | years {ys}")
