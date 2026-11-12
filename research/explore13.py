"""Exploration 13 (DISCOVERY only): the market-wide flush on 15-minute bars: an independent check of the
mechanism at another time scale, with ~4× more chances to see it."""
import numpy as np
import pandas as pd

import lab
import schemes as S

d = lab.load()
fund = d["funding"]
s15, b15 = lab.klines15("SOLUSDT"), lab.klines15("BTCUSDT")
c = s15["close"]
b = b15["close"].reindex(c.index)
X = lab.period(c, "DISCOVERY")
BENCH = lab.pnl(pd.Series(1.0, index=X.index), X, None, 0)
r, rb = np.log(c).diff(), np.log(b).diff()
W = 4 * 24 * 4                                   # σ over the last 4 days of 15-minute bars


def run(name, pos):
    p = lab.period(pos, "DISCOVERY")
    ret = lab.pnl(p, X, fund)
    a, s = lab.alpha(ret, BENCH), lab.stats(ret, p)
    ys = [lab.alpha(ret[ret.index.year == y], BENCH[BENCH.index.year == y]).get("alpha") for y in (2021, 2022, 2023, 2024)]
    print(f"{name:44} α {a['alpha']:+.2f} t{a['t']:+.1f} β{a['beta']:.2f} SR {s['sharpe']:+.2f} tr {s['trades']:5} | years {ys}")


z = r / r.rolling(W).std().shift(1)
zb = rb / rb.rolling(W).std().shift(1)
for zs, zbt in ((-3, -1.5), (-4, -2), (-3, -2.5)):
    for hold in (2, 4, 8):
        run(f"15m flush SOL<{zs} BTC<{zbt} hold {hold*15}min", S._hold_after((z < zs) & (zb < zbt), hold))
run("15m SOL-only drop (BTC>-1) hold 60min", S._hold_after((z < -3) & (zb > -1), 4))
