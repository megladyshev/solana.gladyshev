"""The single HOLDOUT run of the rules in HYPOTHESES.md (commit fcab2e6). Do not edit after running."""
import numpy as np
import pandas as pd

import lab
import schemes as S

d = lab.load()
fund = d["funding"]
c = d["sol"]["close"]
s15, b15 = lab.klines15("SOLUSDT"), lab.klines15("BTCUSDT")
pair = S.pair_close(d)
fb = lab.funding("BTCUSDT")
fs_ = fund["rate"].groupby(fund.index.floor("h")).sum()
fb_ = fb["rate"].groupby(fb.index.floor("h")).sum()
pf = pd.DataFrame({"rate": (fs_ - fb_).dropna()})


def flush15(zs, zb_, hold):
    x = s15["close"]; b = b15["close"].reindex(x.index)
    r, rb = np.log(x).diff(), np.log(b).diff()
    return S._hold_after((r / r.rolling(384).std().shift(1) < zs) & (rb / rb.rolling(384).std().shift(1) < zb_), hold)


def show(name, pos, close, f, cost):
    x = lab.period(close, "HOLDOUT")
    p = lab.period(pos, "HOLDOUT")
    bench = lab.pnl(pd.Series(1.0, index=x.index), x, None, 0)
    ret = lab.pnl(p, x, f, cost)
    a, s = lab.alpha(ret, bench), lab.stats(ret, p)
    halves = {}
    for h0, h1 in (("2025-07", "2026-01"), ("2026-01", "2026-07"), ("2026-07", "2027-01")):
        m = (ret.index >= pd.Timestamp(h0, tz="UTC")) & (ret.index < pd.Timestamp(h1, tz="UTC"))
        halves[h0] = round(float(ret[m].sum()), 3)
    print(f"{name:26} SR {s['sharpe']:+.2f} {s['sharpe90']} | α {a['alpha']:+.2f} {a['alpha90']} t{a['t']:+.1f} β{a['beta']:.2f} | "
          f"CAGR {s['cagr']:+.1%} DD {s['maxdd']:.0%} trades {s['trades']} | sum of returns per half-year {halves}")
    return ret


x = lab.period(c, "HOLDOUT")
print("buy & hold HOLDOUT:", lab.stats(lab.pnl(pd.Series(1.0, index=x.index), x, fund)))
show("H1 flush 1h", S.market_flush(d, -2.5, -1.5, 1), c, fund, lab.COST)
show("H2 flush 15m", flush15(-3, -2.5, 4), s15["close"], fund, lab.COST)
show("H3 flush pair 1h", S.market_flush(d, -3, -1.5, 1), pair, pf, 2 * lab.COST)
