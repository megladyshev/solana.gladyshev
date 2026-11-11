"""Leak check: on a pure random walk nothing is predictable, so the walk-forward must score ~50%.
Any look-ahead in features.py or backtest.py shows up here as a hit rate well above 50%.

    python test_backtest.py
"""
import numpy as np
import pandas as pd

import backtest as BT
import features as FT


def fake(n, seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    q = rng.uniform(1e6, 2e6, n)
    return pd.DataFrame({"close": c, "high": c * 1.004, "low": c * 0.996, "qvol": q, "taker_q": q * rng.uniform(0.4, 0.6, n)}, index=idx)


def test_random_walk_is_unpredictable():
    n = 24 * 365 * 3
    F, c, vol = FT.build({"sol": fake(n, 1), "btc": fake(n, 2), "eth": fake(n, 3)})
    y, move = FT.targets(c, vol)
    for h in (1, 4):
        r = BT.score(BT.walk_forward(F, y[h], move, FT.cols(FT.SCHEMES), h), h)
        print(f"{h}h random walk: hit {r['hit']:.3f} (n {r['n']}), all {r['hit_all']:.3f}, BSS {r['brier_skill']:+.4f}")
        assert abs(r["hit"] - 0.5) < 4 * 0.5 / np.sqrt(r["n"]), r       # within 4 standard errors
        assert abs(r["hit_all"] - 0.5) < 0.02, r
        assert r["brier_skill"] < 0.005, r                               # no skill to find


if __name__ == "__main__":
    test_random_walk_is_unpredictable()
    print("ok: no look-ahead")
