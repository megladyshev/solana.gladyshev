"""VALIDATION run (2024-07 → 2025-06) of the variants fixed in LOG.md."""
import pandas as pd

import lab
import schemes as S

d = lab.load()
c, fund = d["sol"]["close"], d["funding"]
VARIANTS = {
    "flush": {"A": S.market_flush(d, -3, -1.5, 1), "B": S.market_flush(d, -3, -1.5, 4), "C": S.market_flush(d, -2.5, -1.5, 1)},
    "weekend": {"A": S.weekend(d, ("Fri", 23), ("Sun", 23)), "B": S.weekend(d, ("Fri", 22), ("Sun", 22)),
                "C": pd.Series((c.index.dayofweek >= 5).astype(float), index=c.index)},
    "turn_of_month": {"A": S.turn_of_month(d, 3, 1), "B": S.turn_of_month(d, 3, 0), "C": S.turn_of_month(d, 4, 1)},
}
X = lab.period(c, "VALIDATION")
BENCH = lab.pnl(pd.Series(1.0, index=X.index), X, None, 0)
print(f"buy & hold VALIDATION: {lab.stats(lab.pnl(pd.Series(1.0, index=X.index), X, fund))}")
for name, vs in VARIANTS.items():
    for v, pos in vs.items():
        p = lab.period(pos, "VALIDATION")
        ret = lab.pnl(p, X, fund)
        a, s = lab.alpha(ret, BENCH), lab.stats(ret, p)
        print(f"{name:14} {v}: α {a['alpha']:+.2f} {a['alpha90']} t{a['t']:+.1f} β{a['beta']:.2f} | SR {s['sharpe']:+.2f} CAGR {s['cagr']:+.1%} DD {s['maxdd']:.0%} exp {s['exposure']:.0%} tr {s['trades']}")
