"""Exploration 2 (DISCOVERY only): candidate schemes with costs, variants and year-by-year stability."""
import pandas as pd

import lab
import schemes as S

d = lab.load()
c, fund = d["sol"]["close"], d["funding"]


def run(name, pos, close=c, f=fund, cost=lab.COST):
    x = lab.period(close, "DISCOVERY")
    p = lab.period(pos, "DISCOVERY")
    s = lab.stats(lab.pnl(p, x, f, cost), p)
    ys = {}
    for y in (2021, 2022, 2023, 2024):
        m = x.index.year == y
        if m.sum() > 200:
            ys[y] = lab.stats(lab.pnl(p[m], x[m], f, cost)).get("sharpe")
    print(f"{name:30} SR {s['sharpe']:+.2f} {s['sharpe90']} CAGR {s['cagr']:+.1%} DD {s['maxdd']:.0%} exp {s['exposure']:.0%} trades {s['trades']:5} | by year {ys}")


hold = pd.Series(1.0, index=c.index)
run("buy & hold", hold)
print("-- flush bounce")
for z in (-2.5, -3.0, -3.5):
    for h in (1, 4, 12):
        run(f"flush z{z} hold{h}h", S.flush(d, z, h))
print("-- weekend")
run("weekend Fri23→Sun23", S.weekend(d))
run("weekend Sat00→Mon00", S.weekend(d, ("Sat", 0), ("Mon", 0) if False else ("Sun", 24 - 1)))
run("weekend Fri20→Sun20", S.weekend(d, ("Fri", 20), ("Sun", 20)))
print("-- weekly momentum")
for lb in (1, 2, 4):
    run(f"weekly mom {lb}w long-only", S.weekly_mom(d, lb))
    run(f"weekly mom {lb}w long/short", S.weekly_mom(d, lb, short=True))
print("-- on-chain DEX momentum")
for fa, sl in ((7, 28), (3, 14), (14, 56)):
    run(f"dex mom {fa}/{sl}", S.dex_mom(d, fa, sl))
print("-- vol-managed hold")
for tg in (0.4, 0.6, 0.8):
    run(f"vol managed target {tg:.0%}", S.vol_managed(d, tg))
print("-- SOL/BTC relative momentum (pair, two legs → double cost, no funding)")
pc = S.pair_close(d)
for lb in (24, 168, 672):
    run(f"rel mom {lb}h", S.rel_mom(d, lb), pc, None, 2 * lab.COST)
run("pair hold (long SOL/short BTC)", hold, pc, None, 2 * lab.COST)
