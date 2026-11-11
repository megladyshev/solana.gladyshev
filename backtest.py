"""Walk-forward backtest of the volatility forecast: does it beat the naive guesses?

Each month is forecast by a model trained only on windows whose outcome was known before the month
began (embargo = horizon). One forecast per horizon window, so n is honest. Model choice is made on
2023–2024 (SELECT); the numbers that count are on 2025→now (HOLDOUT), which the choice never saw.

What is scored, on the log of the coming range:
  r2_clim   1 − SSE/SSE of "typical range of the last 30 days"   (> 0 = better than typical)
  r2_last   same against "same as the last window"
  cover     how often the actual range fell inside the 10–90% band  (should be ≈ 80%)
  storm BSS skill of P(range above the 90-day 80th percentile) against the constant 20%
  lift      how much bigger the actual range is in the top fifth of forecasts than in the bottom fifth

    python backtest.py        full report, writes model.json
"""
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as FT

warnings.filterwarnings("ignore")
SPLIT = pd.Timestamp("2025-01-01", tz="UTC")
CANDS = {"HAR (V)": ("ridge", ["V"]), "V+T": ("ridge", ["V", "T"]), "V+T+E": ("ridge", ["V", "T", "E"]),
         "all": ("ridge", FT.SCHEMES), "HGB all": ("hgb", FT.SCHEMES)}


def make(kind):
    if kind == "hgb":
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, min_samples_leaf=200)
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def on_grid(idx, h):
    return (idx.hour % h == 0) if h < 24 else (idx.hour == 0)


def fit(F, t, cols, kind, train_mask):
    """Fit on rows of train_mask; return model and residual stats (σ and 10/90% quantiles)."""
    X, y = F.loc[train_mask, cols].values, t.loc[train_mask, "y"].values
    m = make(kind).fit(X, y)
    res = y - m.predict(X)
    return m, float(res.std()), float(np.quantile(res, 0.1)), float(np.quantile(res, 0.9))


def walk_forward(F, t, cols, h, kind="ridge"):
    cols = [k for k in cols if k in F]
    ok = F[cols].notna().all(axis=1) & t[["clim", "last", "thr"]].notna().all(axis=1)
    first = F[ok].index[0] + pd.Timedelta(days=365)
    months = pd.date_range(first.normalize().replace(day=1) + pd.offsets.MonthBegin(1), F.index[-1], freq="MS", tz="UTC")
    grid, out = on_grid(F.index, h), []
    for a, b in zip(months, list(months[1:]) + [F.index[-1] + pd.Timedelta(hours=1)]):
        tr = ok & t["y"].notna() & (F.index < a - pd.Timedelta(hours=h))
        te = ok & grid & t["y"].notna() & (F.index >= a) & (F.index < b)
        if tr.sum() < 2000 or te.sum() == 0:
            continue
        m, s, q10, q90 = fit(F, t, cols, kind, tr)
        p = m.predict(F.loc[te, cols].values)
        o = t.loc[te, ["y", "clim", "last", "thr"]].copy()
        o["pred"], o["lo"], o["hi"] = p, p + q10, p + q90
        o["p_storm"] = 1 - norm.cdf((o["thr"] - p) / s)
        out.append(o)
    return pd.concat(out) if out else pd.DataFrame()


def score(o):
    if not len(o):
        return {}
    sse = lambda p: float(((o["y"] - p) ** 2).sum())
    storm = (o["y"] > o["thr"]).astype(float)
    q = pd.qcut(o["pred"], 5, labels=False, duplicates="drop")
    return {"n": int(len(o)), "r2_clim": 1 - sse(o["pred"]) / sse(o["clim"]), "r2_last": 1 - sse(o["pred"]) / sse(o["last"]),
            "cover": float(((o["y"] >= o["lo"]) & (o["y"] <= o["hi"])).mean()),
            "storm_rate": float(storm.mean()), "storm_bss": float(1 - ((o["p_storm"] - storm) ** 2).mean() / ((0.2 - storm) ** 2).mean()),
            "lift": float(np.exp(o["y"][q == q.max()].mean() - o["y"][q == 0].mean())),
            "mae_pct": float((np.exp(o["y"]) - np.exp(o["pred"])).abs().mean() * 100)}


def fmt(r):
    if not r:
        return "—"
    return (f"R² vs typical {r['r2_clim']:+.3f} · vs last {r['r2_last']:+.3f} · band 80% → {r['cover']:.0%} · "
            f"storm BSS {r['storm_bss']:+.3f} ({r['storm_rate']:.0%} storms) · top/bottom fifth ×{r['lift']:.2f} · n {r['n']}")


def run():
    import data
    d = data.load()
    F, sol = FT.build(d)
    T = FT.targets(sol)
    best = {}
    for h in FT.HORIZONS:
        print(f"\n=== horizon {h}h ===")
        res = {}
        for name, (kind, sch) in CANDS.items():
            o = walk_forward(F, T[h], FT.cols(sch), h, kind)
            res[name] = (kind, sch, score(o[o.index < SPLIT]), score(o[o.index >= SPLIT]))
            print(f"{name:8} SELECT  {fmt(res[name][2])}")
            print(f"{'':8} HOLDOUT {fmt(res[name][3])}")
        name = max(res, key=lambda k: res[k][2].get("r2_clim", -9))          # chosen on SELECT only
        kind, sch, sel, hold = res[name]
        best[f"{h}h"] = {"h": h, "model": kind, "schemes": sch, "chosen": name, "enabled": sel.get("r2_clim", 0) > 0.02, "select": sel, "backtest": hold}
        print(f"→ chosen on SELECT: {name}")
    main = "4h" if best["4h"]["enabled"] else max(best, key=lambda k: best[k]["select"].get("r2_clim", -9))
    json.dump({"main": main, "horizons": best, "split": str(SPLIT.date()), "target": "range"}, open("model.json", "w"), indent=1, default=float)
    print(f"\nmodel.json written · main horizon {main}")


if __name__ == "__main__":
    run()
