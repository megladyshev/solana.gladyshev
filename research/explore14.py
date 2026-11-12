"""Exploration 14 (DISCOVERY only): perp discount to spot in a crash hour as a footprint of forced selling.
Research uses the archived premium at its true hour (live it is public in real time on Binance)."""
import numpy as np
import pandas as pd

import lab
import schemes as S

d = lab.load()
c, fund = d["sol"]["close"], d["funding"]
X = lab.period(c, "DISCOVERY")
BENCH = lab.pnl(pd.Series(1.0, index=X.index), X, None, 0)


def run(name, pos):
    p = lab.period(pos, "DISCOVERY")
    ret = lab.pnl(p, X, fund)
    a, s = lab.alpha(ret, BENCH), lab.stats(ret, p)
    ys = [lab.alpha(ret[ret.index.year == y], BENCH[BENCH.index.year == y]).get("alpha") for y in (2021, 2022, 2023, 2024)]
    print(f"{name:46} α {a['alpha']:+.2f} t{a['t']:+.1f} β{a['beta']:.2f} SR {s['sharpe']:+.2f} tr {s['trades']:4} | years {ys}")


prem = d["premium"]["premium"].reindex(c.index)
pz = (prem - prem.rolling(720).mean().shift(1)) / prem.rolling(720).std().shift(1)
r = np.log(c).diff()
z = r / r.rolling(168).std().shift(1)
mf = S.market_flush(d, -3, -1.5, 1) > 0
for k in (2, 3):
    run(f"perp discount pz<-{k} & price<-2σ, h1", S._hold_after((pz < -k) & (z < -2), 1))
    run(f"perp discount pz<-{k} & price<-2σ, not flush, h1", S._hold_after((pz < -k) & (z < -2) & ~mf, 1))
run("flush AND perp discount pz<-2, h1", S._hold_after(mf & (pz < -2), 1))
run("flush WITHOUT perp discount (pz>=-2), h1", S._hold_after(mf & (pz >= -2), 1))
