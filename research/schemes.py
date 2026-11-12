"""Candidate schemes as position series. Each takes the data dict and parameters, returns pos[t] in [-1, 1]
decided with data up to hour t only."""
import numpy as np
import pandas as pd

LAG_DAILY = pd.Timedelta(hours=25)     # a daily on-chain value for day D is used from D+1 01:00 UTC


def _hold_after(signal, hours):
    """1 for `hours` hours after each signal hour (signal at t → long over t..t+hours−1 decisions)."""
    s = signal.astype(float)
    return s[::-1].rolling(1).max()[::-1].rolling(hours, min_periods=1).max().fillna(0)


def flush(d, z=-3.0, hold=4, win=168):
    """Liquidation-flush bounce: an hour falling below z·σ of the last week → long for `hold` hours."""
    c = d["sol"]["close"]
    r = np.log(c).diff()
    sd = r.rolling(win).std().shift(1)
    return _hold_after(r / sd < z, hold)


def weekend(d, start=("Fri", 23), end=("Sun", 23)):
    """Long from Friday 23:00 to Sunday 23:00 UTC (decision hours), flat otherwise."""
    idx = d["sol"].index
    days = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    how = idx.dayofweek * 24 + idx.hour
    a, b = days[start[0]] * 24 + start[1], days[end[0]] * 24 + end[1]
    return pd.Series(((how >= a) & (how < b)).astype(float), index=idx)


def weekly_mom(d, lookback_w=1, short=False):
    """Weekly momentum: at Monday 00:00 UTC take the sign of the last `lookback_w` weeks' return, hold a week."""
    c = d["sol"]["close"]
    wk = c[c.index.dayofweek.isin([0]) & (c.index.hour == 0)]
    sig = np.sign(np.log(wk).diff(lookback_w))
    if not short:
        sig = sig.clip(lower=0)
    return sig.reindex(c.index).ffill().fillna(0)


def dex_mom(d, fast=7, slow=28):
    """On-chain activity momentum: long while Solana DEX volume (fast avg) is above its slow avg, else flat."""
    dx = d["dexs"]
    g = dx.rolling(fast).mean() > dx.rolling(slow).mean()
    g.index = g.index + LAG_DAILY
    c = d["sol"]["close"]
    return g.astype(float).reindex(c.index, method="ffill").fillna(0)


def vol_managed(d, target=0.6, win=168, base=None):
    """Hold SOL (or a base scheme) scaled by target vol / recent vol, capped at 1 (no leverage)."""
    c = d["sol"]["close"]
    vol = np.log(c).diff().rolling(win).std() * np.sqrt(8760)
    w = (target / vol).clip(upper=1).fillna(0)
    return w * (1 if base is None else base)


def rel_mom(d, lookback=168):
    """SOL vs BTC: long SOL/short BTC when SOL beat BTC over the lookback, short the pair otherwise.
    Returned as a position on the synthetic pair (see pair_close)."""
    c, b = d["sol"]["close"], d["btc"]["close"].reindex(d["sol"].index)
    return np.sign(np.log(c / b).diff(lookback)).fillna(0)


def pair_close(d):
    """Synthetic price of a long SOL / short BTC pair, rebalanced hourly (equal notional)."""
    c, b = d["sol"]["close"], d["btc"]["close"].reindex(d["sol"].index)
    r = c.pct_change() - b.pct_change()
    return (1 + r.fillna(0)).cumprod()


def market_flush(d, z_sol=-3.0, z_btc=-1.5, hold=1, win=168):
    """Market-wide liquidation flush: SOL hour below z_sol·σ AND BTC hour below z_btc·σ (σ of the last week)
    → long SOL for `hold` hours from that close."""
    c = d["sol"]["close"]
    b = d["btc"]["close"].reindex(c.index)
    r, rb = np.log(c).diff(), np.log(b).diff()
    z = r / r.rolling(win).std().shift(1)
    zb = rb / rb.rolling(win).std().shift(1)
    return _hold_after((z < z_sol) & (zb < z_btc), hold)


def turn_of_month(d, last=3, first=1):
    """Long during the last `last` days of a month and the first `first` days of the next (UTC), flat otherwise."""
    idx = d["sol"].index
    left = idx.days_in_month - idx.day
    return pd.Series(((left < last) | (idx.day <= first)).astype(float), index=idx)
