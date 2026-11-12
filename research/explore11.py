"""Exploration 11 (DISCOVERY only): OI deleveraging events (research timing; live would need real-time OI),
residual reversal at days, daily catch-up to BTC, daily overreaction."""
import numpy as np
import pandas as pd

import lab
import schemes as S

d = lab.load()
c, fund = d["sol"]["close"], d["funding"]
idx = c.index
X = lab.period(c, "DISCOVERY")
BENCH = lab.pnl(pd.Series(1.0, index=X.index), X, None, 0)


def run(name, pos):
    p = lab.period(pos, "DISCOVERY")
    ret = lab.pnl(p, X, fund)
    a, s = lab.alpha(ret, BENCH), lab.stats(ret, p)
    ys = [lab.alpha(ret[ret.index.year == y], BENCH[BENCH.index.year == y]).get("alpha") for y in (2021, 2022, 2023, 2024)]
    print(f"{name:46} α {a['alpha']:+.2f} t{a['t']:+.1f} β{a['beta']:.2f} SR {s['sharpe']:+.2f} tr {s['trades']:4} | years {ys}")


m = d["metrics"]["oi_usd"]                       # hourly, labelled at the hour close (5-min data resampled)
oi = np.log(m).reindex(idx)
doi = oi.diff()
doi_z = doi / doi.rolling(720).std().shift(1)
r = np.log(c).diff()
z = r / r.rolling(168).std().shift(1)
print("-- OI deleveraging: OI falls > k·σ in the hour while price falls > 2σ → long (timing as of the hour close)")
for k in (2.0, 3.0):
    for h in (1, 4):
        run(f"deleverage OI<-{k}σ & price<-2σ hold{h}", S._hold_after((doi_z < -k) & (z < -2), h))
print("-- deleveraging, but no market-wide flush in that hour (is it new information?)")
mf = S.market_flush(d, -3, -1.5, 1) > 0
run("deleverage OI<-2σ & price<-2σ, not flush, h1", S._hold_after((doi_z < -2) & (z < -2) & ~mf, 1))

b = d["btc"]["close"].reindex(idx)
rb = np.log(b).diff()
beta = (r.rolling(720).cov(rb) / rb.rolling(720).var()).shift(1)
print("-- residual (SOL − β·BTC) over 24/72h beyond ±2σ → fade for the same time")
for k in (24, 72):
    res = np.log(c).diff(k) - beta * np.log(b).diff(k)
    zr = res / res.rolling(2160).std()
    run(f"residual fade {k}h", S._hold_after(zr < -2, k) - S._hold_after(zr > 2, k))
    run(f"residual follow {k}h", S._hold_after(zr > 2, k) - S._hold_after(zr < -2, k))
print("-- daily overreaction: SOL day > +3σ → short next day; < −3σ → long next day (decided at 00:00 UTC)")
dz = (np.log(c).diff(24) / (r.rolling(720).std() * np.sqrt(24))).where(idx.hour == 0)
run("fade daily ±3σ hold24", S._hold_after(dz < -3, 24) - S._hold_after(dz > 3, 24))
run("follow daily ±3σ hold24", S._hold_after(dz > 3, 24) - S._hold_after(dz < -3, 24))
