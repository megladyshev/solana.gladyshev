"""Exploration 6 (DISCOVERY only): post-funding re-entry as a scheme; squeeze continuation; BTC flush → SOL."""
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
    print(f"{name:42} α {a['alpha']:+.2f} t{a['t']:+.1f} β{a['beta']:.2f} SR {s['sharpe']:+.2f} tr {s['trades']:4} | years {ys}")


f = fund["rate"].groupby(fund.index.floor("h")).sum()
print("-- post-funding re-entry: long for h hours after a payment whose previous rate > th")
for th in (0.0001, 0.0002, 0.0004):
    for h in (1, 2, 4):
        sig = pd.Series(False, index=c.index)
        pays = f.index[(f.shift(1) > th).values]
        sig[sig.index.isin(pays)] = True
        run(f"post-funding prev>{th*100:.2f}% hold{h}h", S._hold_after(sig, h))
print("-- same, but the payment's own rate (known just before it: the premium average is public)")
for th in (0.0002,):
    sig = pd.Series(c.index.isin(f.index[(f > th).values]), index=c.index)
    run(f"post-funding own>{th*100:.2f}% hold1h", S._hold_after(sig, 1))

r = np.log(c).diff()
rb = np.log(d["btc"]["close"].reindex(c.index)).diff()
z = r / r.rolling(168).std().shift(1)
zb = rb / rb.rolling(168).std().shift(1)
print("-- squeeze continuation: market-wide jump (SOL>+3σ, BTC>+1.5σ) → long")
for h in (4, 12, 24):
    run(f"market jump hold{h}", S._hold_after((z > 3) & (zb > 1.5), h))
print("-- BTC flush (<−3σ) → long SOL")
for h in (1, 4):
    run(f"BTC<-3σ → SOL hold{h}", S._hold_after(zb < -3, h))
