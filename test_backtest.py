"""Leak check: on a random walk with constant volatility the size of the next move is unpredictable,
so the walk-forward must not beat "typical range". Any look-ahead in features.py or backtest.py
shows up here as a clearly positive R².

    python test_backtest.py
"""
import numpy as np
import pandas as pd

import backtest as BT
import features as FT


def fake(n, seed):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="1h", tz="UTC")
    r = rng.normal(0, 0.01, n)
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[100, c[:-1]]
    wick = np.abs(rng.normal(0, 0.004, (2, n)))
    q = rng.uniform(1e6, 2e6, n)
    return pd.DataFrame({"close": c, "high": np.maximum(o, c) * (1 + wick[0]), "low": np.minimum(o, c) * (1 - wick[1]),
                         "qvol": q, "taker_q": q * rng.uniform(0.4, 0.6, n)}, index=idx)


def run(leak=False):
    n = 24 * 365 * 3
    F, sol = FT.build({"sol": fake(n, 1), "btc": fake(n, 2), "eth": fake(n, 3)})
    T = FT.targets(sol)
    if leak:
        F["hl1"] = F["hl1"].shift(-1)          # deliberate one-hour look-ahead
    return BT.score(BT.walk_forward(F, T[4], FT.cols(FT.SCHEMES), 4))


def test_random_walk_is_unpredictable():
    r = run()
    print("random walk 4h:", BT.fmt(r))
    assert r["r2_clim"] < 0.01, r
    assert abs(r["cover"] - 0.8) < 0.05, r


def test_leak_is_caught():
    r = run(leak=True)
    print("with a planted leak:", BT.fmt(r))
    assert r["r2_clim"] > 0.03, r


if __name__ == "__main__":
    test_random_walk_is_unpredictable()
    test_leak_is_caught()
    print("ok: no look-ahead, and a planted one is caught")
