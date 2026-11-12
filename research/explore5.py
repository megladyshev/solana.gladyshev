"""Exploration 5 (DISCOVERY only):
 a) market-wide flush on 1-minute data: how much bounce is left if entry is 1, 5, 15 minutes after the candle close
 b) funding-timing flows: price in the hour before/after a funding payment, by the sign of the last known rate"""
import json
import time

import numpy as np
import pandas as pd

from lab import data
import lab
import schemes as S

d = lab.load()
c, fund = d["sol"]["close"], d["funding"]
r = np.log(c).diff()
rb = np.log(d["btc"]["close"].reindex(c.index)).diff()
z = r / r.rolling(168).std().shift(1)
zb = rb / rb.rolling(168).std().shift(1)
ev = lab.period(((z < -3) & (zb < -1.5)).astype(int), "DISCOVERY")
ev = ev[ev == 1].index

print(f"== a) {len(ev)} market-wide flush hours; 1-minute bars for the 75 minutes after each close")
cache = data.D / "flush_1m.json"
got = json.loads(cache.read_text()) if cache.exists() else {}
for t in ev:
    k = str(int(t.timestamp() * 1000))
    if k in got:
        continue
    j = json.loads(data.get(f"https://data-api.binance.vision/api/v3/klines?symbol=SOLUSDT&interval=1m&startTime={k}&limit=75"))
    got[k] = [[x[0], float(x[1]), float(x[4])] for x in j]
    time.sleep(0.05)
cache.write_text(json.dumps(got))
rows = []
for t in ev:
    m = got[str(int(t.timestamp() * 1000))]
    if len(m) < 61:
        continue
    o = m[0][1]                                    # open of the first minute after the close = the close price
    px = {lag: (m[lag - 1][2] if lag else o) for lag in (0, 1, 5, 15)}
    exit_ = {lag: m[min(lag + 59, len(m) - 1)][2] for lag in (0, 1, 5, 15)}   # hold 60 minutes from entry
    rows.append({lag: np.log(exit_[lag] / px[lag]) for lag in (0, 1, 5, 15)})
R = pd.DataFrame(rows)
for lag in (0, 1, 5, 15):
    x = R[lag] - 2 * lab.COST
    print(f"  entry +{lag:2d} min, hold 60 min: mean net {x.mean()*1e4:+6.1f} bp, t{x.mean()/x.std()*np.sqrt(len(x)):+.1f}, win {np.mean(x>0):.0%}, n {len(x)}")

print("== b) funding timing (payments at 00/08/16 UTC), DISCOVERY")
f = fund["rate"].groupby(fund.index.floor("h")).sum()
fp = f.shift(1)                                    # the last known rate before this payment
pay = lab.period(f, "DISCOVERY").index
lr = np.log(c)
g = lambda t: lr.get(t, np.nan)
for nm, m in (("last rate > +0.02%", fp > 0.0002), ("last rate in 0..0.02%", (fp >= 0) & (fp <= 0.0002)), ("last rate < 0", fp < 0)):
    ts = [t for t in pay if m.get(t, False)]
    before = pd.Series([g(t) - g(t - pd.Timedelta(hours=1)) for t in ts]).dropna()
    after = pd.Series([g(t + pd.Timedelta(hours=1)) - g(t) for t in ts]).dropna()
    tt = lambda x: x.mean() / x.std() * np.sqrt(len(x))
    print(f"  {nm:22} n{len(ts):4} | hour before {before.mean()*1e4:+5.1f}bp t{tt(before):+.1f} | hour after {after.mean()*1e4:+5.1f}bp t{tt(after):+.1f}")
