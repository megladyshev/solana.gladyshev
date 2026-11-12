"""Exploration 12 (DISCOVERY only): long-term trend filter on SOL (hold only above an n-day average)."""
import numpy as np
import pandas as pd

import lab

d = lab.load()
c, fund = d["sol"]["close"], d["funding"]
X = lab.period(c, "DISCOVERY")
BENCH = lab.pnl(pd.Series(1.0, index=X.index), X, None, 0)
for n in (50, 100, 200):
    pos = (c > c.rolling(24 * n).mean()).astype(float)
    p = lab.period(pos, "DISCOVERY")
    ret = lab.pnl(p, X, fund)
    a, s = lab.alpha(ret, BENCH), lab.stats(ret, p)
    ys = [lab.alpha(ret[ret.index.year == y], BENCH[BENCH.index.year == y]).get("alpha") for y in (2021, 2022, 2023, 2024)]
    print(f"above {n}d average: α {a['alpha']:+.2f} t{a['t']:+.1f} β{a['beta']:.2f} SR {s['sharpe']:+.2f} DD {s['maxdd']:.0%} exp {s['exposure']:.0%} tr {s['trades']} | years {ys}")
