"""Exploration 4 (DISCOVERY only): robustness of the survivors: neighbours of each parameter, harsher costs."""
import numpy as np
import pandas as pd

import lab
import schemes as S

d = lab.load()
c, fund = d["sol"]["close"], d["funding"]
X = lab.period(c, "DISCOVERY")
BENCH = lab.pnl(pd.Series(1.0, index=X.index), X, None, 0)


def run(name, pos, cost=lab.COST):
    p = lab.period(pos, "DISCOVERY")
    ret = lab.pnl(p, X, fund, cost)
    a, s = lab.alpha(ret, BENCH), lab.stats(ret, p)
    ys = [lab.alpha(ret[ret.index.year == y], BENCH[BENCH.index.year == y]).get("alpha") for y in (2021, 2022, 2023, 2024)]
    print(f"{name:40} α {a['alpha']:+.2f} t{a['t']:+.1f} β{a['beta']:.2f} SR {s['sharpe']:+.2f} tr {s['trades']:4} | years {ys}")


r = np.log(c).diff()
rb = np.log(d["btc"]["close"].reindex(c.index)).diff()
z = r / r.rolling(168).std().shift(1)
zb = rb / rb.rolling(168).std().shift(1)
print("-- market-wide flush: neighbours")
for zs in (-2.5, -3.0, -3.5):
    for zbt in (-1.0, -1.5, -2.0):
        run(f"flush SOL<{zs} BTC<{zbt} h1", S._hold_after((z < zs) & (zb < zbt), 1))
for h in (1, 2, 4, 8):
    run(f"flush SOL<-3 BTC<-1.5 hold{h}", S._hold_after((z < -3) & (zb < -1.5), h))
print("-- market-wide flush: costs ×2, ×3 (slippage in a crash)")
base = S._hold_after((z < -3) & (zb < -1.5), 1)
for k in (2, 3):
    run(f"flush base, cost ×{k}", base, lab.COST * k)
print("-- market-wide flush: entry one hour late (a slow trader)")
run("flush base, entry +1h", base.shift(1).fillna(0))

print("-- weekend: neighbours and costs")
for st, en in ((("Fri", 22), ("Sun", 22)), (("Fri", 23), ("Sun", 23)), (("Sat", 0), ("Mon", 0))):
    p = S.weekend(d, st, en) if en[0] != "Mon" else pd.Series((c.index.dayofweek >= 5).astype(float), index=c.index)
    run(f"weekend {st}→{en}", p)
run("weekend Fri23→Sun23 cost ×2", S.weekend(d), lab.COST * 2)
run("weekend only Saturday", pd.Series((c.index.dayofweek == 5).astype(float), index=c.index))
run("weekend only Sunday", pd.Series((c.index.dayofweek == 6).astype(float), index=c.index))
