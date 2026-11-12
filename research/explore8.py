"""Exploration 8 (DISCOVERY only): turn-of-month window neighbours."""
import pandas as pd

import lab

d = lab.load()
c, fund = d["sol"]["close"], d["funding"]
X = lab.period(c, "DISCOVERY")
BENCH = lab.pnl(pd.Series(1.0, index=X.index), X, None, 0)
idx = c.index


def run(name, pos):
    p = lab.period(pos, "DISCOVERY")
    ret = lab.pnl(p, X, fund)
    a, s = lab.alpha(ret, BENCH), lab.stats(ret, p)
    ys = [lab.alpha(ret[ret.index.year == y], BENCH[BENCH.index.year == y]).get("alpha") for y in (2021, 2022, 2023, 2024)]
    print(f"{name:34} α {a['alpha']:+.2f} t{a['t']:+.1f} β{a['beta']:.2f} SR {s['sharpe']:+.2f} exp {s['exposure']:.0%} | years {ys}")


left = idx.days_in_month - idx.day          # 0 on the last day
for k in (2, 3, 4, 5):
    run(f"last {k} days", pd.Series((left < k).astype(float), index=idx))
for k, j in ((3, 1), (3, 2), (3, 3), (4, 1)):
    run(f"last {k} + first {j} days", pd.Series(((left < k) | (idx.day <= j)).astype(float), index=idx))
