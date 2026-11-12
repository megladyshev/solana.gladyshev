"""Research lab: honest backtests of trading schemes on SOL.

Three periods, fixed before any test (see PROTOCOL.md):
    DISCOVERY   2021-01-01 → 2024-06-30   free exploration, any number of tries
    VALIDATION  2024-07-01 → 2025-06-30   only to filter survivors and pick between few variants
    HOLDOUT     2025-07-01 → now          sealed: touched once, for pre-registered rules only

A scheme is a position series pos[t] in [-1, 1] decided at hour t with data up to t, held over (t, t+1].
PnL counts taker fees + slippage on every change of position and the perp funding a position pays/receives.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import data  # noqa: E402

PERIODS = {"DISCOVERY": ("2021-01-01", "2024-07-01"), "VALIDATION": ("2024-07-01", "2025-07-01"), "HOLDOUT": ("2025-07-01", "2100-01-01")}
COST = 0.0006          # per unit of turnover: 0.05% taker on Binance perps + 0.01% slippage
HOLDOUT_UNLOCK = ROOT / "research" / "HOLDOUT_UNLOCKED"   # created only for the single final run


def funding(symbol="SOLUSDT"):
    """Perp funding history (rate per payment, at its timestamp)."""
    f = data.D / f"{symbol}_funding.csv"
    df = pd.read_csv(f, index_col=0, parse_dates=[0]) if f.exists() else None
    t = int(df.index[-1].timestamp() * 1000) + 1 if df is not None else int(pd.Timestamp(data.START, tz="UTC").timestamp() * 1000)
    rows = []
    while True:
        k = json.loads(data.get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&startTime={t}&limit=1000"))
        if not k:
            break
        rows += k
        t = k[-1]["fundingTime"] + 1
        if len(k) < 1000:
            break
    if rows:
        new = pd.DataFrame({"rate": [float(r["fundingRate"]) for r in rows]}, index=pd.to_datetime([r["fundingTime"] for r in rows], unit="ms", utc=True))
        df = new if df is None else pd.concat([df, new])
        df = df[~df.index.duplicated()].sort_index()
        df.to_csv(f)
    return df


def llama(kind):
    """Daily Solana DEX volume or fees from DefiLlama (value for day D is complete only after D ends)."""
    j = json.loads(data.get(f"https://api.llama.fi/overview/{kind}/solana?excludeTotalDataChart=false&excludeTotalDataChartBreakdown=true"))
    s = pd.Series({pd.Timestamp(int(ts), unit="s", tz="UTC"): float(v) for ts, v in j["totalDataChart"]}, name=kind)
    return s.sort_index()


def load():
    d = data.load()
    d["funding"] = funding()
    for k in ("dexs", "fees"):
        try:
            d[k] = llama(k)
        except Exception as e:  # noqa: BLE001  (optional sources)
            print("skip", k, e)
    return d


def period(x, name):
    if name == "HOLDOUT" and not HOLDOUT_UNLOCK.exists():
        raise RuntimeError("HOLDOUT is sealed until the rules are pre-registered (see PROTOCOL.md)")
    a, b = PERIODS[name]
    return x[(x.index >= pd.Timestamp(a, tz="UTC")) & (x.index < pd.Timestamp(b, tz="UTC"))]


def pnl(pos, close, fund=None, cost=COST):
    """Hourly net returns of holding pos[t] over (t, t+1]."""
    pos = pos.reindex(close.index).fillna(0).clip(-1, 1)
    r = close.pct_change().shift(-1)                       # return of the hour after t
    gross = pos * r
    fees = cost * pos.diff().abs().fillna(pos.abs())
    out = gross - fees
    if fund is not None:                                   # long pays, short receives the funding at its timestamp
        f = fund["rate"].groupby(fund.index.floor("h")).sum()     # payment times carry a few ms of jitter
        fr = f.reindex(close.index.shift(1, freq="h")).fillna(0).values
        out = out - pos * fr
    return out.shift(1).dropna()                           # booked at t+1


def stats(ret, pos=None):
    """Annualised numbers plus a day-block bootstrap 90% interval for the Sharpe ratio."""
    ret = ret.dropna()
    if len(ret) < 48:
        return {}
    ann = 8760
    sr = ret.mean() / ret.std() * np.sqrt(ann) if ret.std() > 0 else 0.0
    eq = (1 + ret).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    days = ret.groupby(ret.index.floor("D"))
    dm, dn = days.sum().values, days.size().values
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(500):
        i = rng.integers(0, len(dm), len(dm))
        bs.append(dm[i].sum() / dn[i].sum())
    lo, hi = np.percentile(np.array(bs) / ret.std() * np.sqrt(ann), [5, 95])
    out = {"sharpe": round(float(sr), 2), "sharpe90": [round(float(lo), 2), round(float(hi), 2)],
           "cagr": round(float(eq.iloc[-1] ** (ann / len(ret)) - 1), 3), "maxdd": round(dd, 3), "hours": int(len(ret))}
    if pos is not None:
        p = pos.reindex(ret.index).fillna(0)
        out["exposure"] = round(float((p != 0).mean()), 3)
        out["trades"] = int((p.diff().abs() > 0).sum())
    return out


def compare(name, pos, close, fund, periods=("DISCOVERY", "VALIDATION")):
    """Scheme vs buy-and-hold, per period. Returns a dict and prints one line per period."""
    res = {}
    for p in periods:
        c = period(close, p)
        s = stats(pnl(period(pos, p), c, fund), period(pos, p))
        b = stats(pnl(pd.Series(1.0, index=c.index), c, fund))
        res[p] = {"scheme": s, "hold": b}
        if s:
            print(f"{name:34} {p:10} SR {s['sharpe']:+.2f} {s['sharpe90']} CAGR {s['cagr']:+.1%} DD {s['maxdd']:.0%} "
                  f"exp {s.get('exposure', 0):.0%} trades {s.get('trades', 0)} | hold SR {b['sharpe']:+.2f} CAGR {b['cagr']:+.1%}")
    return res
