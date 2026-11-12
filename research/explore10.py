"""Exploration 10 (DISCOVERY only): Solana ecosystem lead (RAY → SOL), SOL−BTC funding spread pair,
storm avoidance, market-neutral flush."""
import numpy as np
import pandas as pd

import lab
import schemes as S
from lab import data

d = lab.load()
c, fund = d["sol"]["close"], d["funding"]
idx = c.index
X = lab.period(c, "DISCOVERY")
BENCH = lab.pnl(pd.Series(1.0, index=X.index), X, None, 0)


def run(name, pos, close=c, f=fund, cost=lab.COST, bench=None):
    x = lab.period(close, "DISCOVERY")
    p = lab.period(pos, "DISCOVERY")
    ret = lab.pnl(p, x, f, cost)
    bn = BENCH if bench is None else bench
    a, s = lab.alpha(ret, bn), lab.stats(ret, p)
    ys = [lab.alpha(ret[ret.index.year == y], bn[bn.index.year == y]).get("alpha") for y in (2021, 2022, 2023, 2024)]
    print(f"{name:44} α {a['alpha']:+.2f} t{a['t']:+.1f} β{a['beta']:.2f} SR {s['sharpe']:+.2f} tr {s['trades']:4} | years {ys}")


print("-- RAY → SOL: residual RAY move (vs SOL) over k hours → long/short SOL for k hours")
ray = data.klines("RAYUSDT")["close"].reindex(idx)
for k in (1, 4, 24):
    res = np.log(ray).diff(k) - np.log(c).diff(k)
    zr = res / res.rolling(720).std()
    run(f"RAY leads, k={k}h, |z|>1.5", S._hold_after(zr > 1.5, k) - S._hold_after(zr < -1.5, k))
    run(f"RAY leads, k={k}h, long z>1.5", S._hold_after(zr > 1.5, k))

print("-- funding spread pair: SOL funding − BTC funding (3-payment mean); pair = long SOL / short BTC")
fb = lab.funding("BTCUSDT")
fs_ = fund["rate"].groupby(fund.index.floor("h")).sum()
fb_ = fb["rate"].groupby(fb.index.floor("h")).sum()
spread = (fs_ - fb_).rolling(3).mean().reindex(idx, method="ffill")
pair = S.pair_close(d)
pb = lab.pnl(pd.Series(1.0, index=lab.period(pair, "DISCOVERY").index), lab.period(pair, "DISCOVERY"), None, 0)
# pair funding: long SOL pays SOL funding, short BTC receives BTC funding → net cost = fs − fb per payment
pf = pd.DataFrame({"rate": (fs_ - fb_).dropna()})
for th in (0.0001, 0.0003):
    pos = -(spread > th).astype(float) + (spread < -th).astype(float)
    run(f"fade funding spread |s|>{th*100:.2f}%", pos, pair, pf, 2 * lab.COST, pb)

print("-- storm avoidance: hold SOL except when last-24h vol is in the top 20% of the last 90 days")
r = np.log(c).diff()
v24 = r.rolling(24).std()
hot = v24 > v24.rolling(2160).quantile(0.8)
run("hold except storms", (~hot).astype(float))

print("-- market-neutral flush: pair long SOL / short BTC after a market-wide flush")
mf = S.market_flush(d, -3, -1.5, 1)
run("flush as pair (hold 1h)", mf, pair, pf, 2 * lab.COST, pb)
