"""VALIDATION round 2 (2024-07 → 2025-06): the 15-minute and the market-neutral pair versions of the flush."""
import numpy as np
import pandas as pd

import lab
import schemes as S

d = lab.load()
fund = d["funding"]
s15, b15 = lab.klines15("SOLUSDT"), lab.klines15("BTCUSDT")


def flush15(zs, zb_, hold):
    c = s15["close"]; b = b15["close"].reindex(c.index)
    r, rb = np.log(c).diff(), np.log(b).diff()
    W = 384
    return S._hold_after((r / r.rolling(W).std().shift(1) < zs) & (rb / rb.rolling(W).std().shift(1) < zb_), hold)


def show(name, pos, close, f, cost, bench):
    p = lab.period(pos, "VALIDATION")
    x = lab.period(close, "VALIDATION")
    ret = lab.pnl(p, x, f, cost)
    a, s = lab.alpha(ret, bench), lab.stats(ret, p)
    print(f"{name:30} α {a['alpha']:+.2f} {a['alpha90']} t{a['t']:+.1f} β{a['beta']:.2f} | SR {s['sharpe']:+.2f} CAGR {s['cagr']:+.1%} DD {s['maxdd']:.0%} tr {s['trades']}")


x15 = lab.period(s15["close"], "VALIDATION")
b15n = lab.pnl(pd.Series(1.0, index=x15.index), x15, None, 0)
for v, args in (("A", (-3, -1.5, 4)), ("B", (-3, -1.5, 8)), ("C", (-3, -2.5, 4))):
    show(f"flush 15m {v}", flush15(*args), s15["close"], fund, lab.COST, b15n)

pair = S.pair_close(d)
fb = lab.funding("BTCUSDT")
fs_ = fund["rate"].groupby(fund.index.floor("h")).sum()
fb_ = fb["rate"].groupby(fb.index.floor("h")).sum()
pf = pd.DataFrame({"rate": (fs_ - fb_).dropna()})
xp = lab.period(pair, "VALIDATION")
pbn = lab.pnl(pd.Series(1.0, index=xp.index), xp, None, 0)
for v, args in (("A", (-3, -1.5, 1)), ("B", (-3, -1.5, 4)), ("C", (-2.5, -1.5, 1))):
    show(f"flush pair 1h {v}", S.market_flush(d, *args), pair, pf, 2 * lab.COST, pbn)
