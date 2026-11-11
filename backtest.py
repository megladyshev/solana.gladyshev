"""Walk-forward backtest: does any scheme beat the baselines out of sample?

Each month is predicted by a model trained only on data whose outcome was already known before the
month began (embargo = horizon). Calls are scored on a non-overlapping grid (one call per horizon
window), so n is honest. Model and scheme choice is made on 2023–2024 only (SELECT); the numbers that
count are the ones on 2025→now (HOLDOUT), which the choice never saw.

    python backtest.py            full report, writes model.json
    python backtest.py --quick    LR only, no ablation
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as FT

warnings.filterwarnings("ignore")
SPLIT = pd.Timestamp("2025-01-01", tz="UTC")
COVER = 0.30          # aim: speak on ~30% of windows, stay silent on the rest
FEE = 0.002           # round trip, only for the PnL column


def make(kind):
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, min_samples_leaf=200)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=1000))


def on_grid(idx, h):
    return (idx.hour % h == 0) if h < 24 else (idx.hour == 0)


def walk_forward(F, y, move, cols, h, kind="lr", start=None):
    X = F[[k for k in cols if k in F]]
    ok = X.notna().all(axis=1)
    first = X[ok].index[0] + pd.Timedelta(days=365)          # at least a year to learn from
    months = pd.date_range(max(first, start or first).normalize().replace(day=1) + pd.offsets.MonthBegin(1), F.index[-1], freq="MS", tz="UTC")
    grid, out = on_grid(F.index, h), []
    for a, b in zip(months, list(months[1:]) + [F.index[-1] + pd.Timedelta(hours=1)]):
        tr = ok & y.notna() & (F.index < a - pd.Timedelta(hours=h))
        te = ok & grid & (F.index >= a) & (F.index < b)
        if tr.sum() < 2000 or te.sum() == 0:
            continue
        m = make(kind).fit(X[tr].values, y[tr].values)
        out.append(pd.DataFrame({"p": m.predict_proba(X[te].values)[:, 1], "y": y[te], "move": move[h][te]}, index=F.index[te]))
    return pd.concat(out) if out else pd.DataFrame(columns=["p", "y", "move"])


def margins(p, cover=COVER, warm=60):
    """Abstain threshold for each call from PAST calls only: the |p−0.5| that the top `cover` share reached."""
    a = (p - 0.5).abs()
    return a.expanding(warm).quantile(1 - cover).shift(1).fillna(np.inf)


def score(o, h):
    """Metrics on scored windows (flat outcomes are left out of hit rates, kept in coverage)."""
    if not len(o):
        return {}
    o = o.copy()
    o["conf"] = (o["p"] - 0.5).abs() >= margins(o["p"])
    s = o[o["y"].notna()]
    c = s[s["conf"]]
    hit = lambda d: float(((d["p"] > 0.5) == (d["y"] == 1)).mean()) if len(d) else None
    n = len(c)
    z = (hit(c) - 0.5) * np.sqrt(n) / 0.5 if n else None
    sign = np.where(o["p"] > 0.5, 1, -1)
    pnl = (sign * o["move"] - FEE)[o["conf"] & o["move"].notna()]
    return {"n": int(n), "hit": hit(c), "z": z, "coverage": float(o["conf"].mean()), "hit_all": hit(s),
            "brier_skill": float(1 - ((s["p"] - s["y"]) ** 2).mean() / 0.25), "up_rate": float(s["y"].mean()),
            "pnl": float(pnl.sum()), "pnl_per_call": float(pnl.mean()) if len(pnl) else None}


def baselines(F, y, move, h, idx):
    g = idx[on_grid(idx, h)]
    yy = y.reindex(g)
    s = yy.notna()
    mom_h = (F[f"r{h}"] if f"r{h}" in F else F["r24"]).reindex(g) > 0
    mom_w = F["r168"].reindex(g) > 0
    f = lambda pred: float((pred[s] == (yy[s] == 1)).mean())
    return {"always up": f(pd.Series(True, index=g)), f"same as last {h}h": f(mom_h), "week trend": f(mom_w)}


def fmt(r):
    if not r:
        return "—"
    v = lambda x, f: "—" if x is None else f.format(x)
    return (f"hit {v(r['hit'], '{:.1%}')} (z {v(r['z'], '{:+.1f}')}, n {r['n']}) · cover {r['coverage']:.0%} · "
            f"all {v(r['hit_all'], '{:.1%}')} · BSS {r['brier_skill']:+.3f} · PnL {r['pnl']:+.2f}")


def run(quick=False):
    import data
    d = data.load()
    F, c, vol = FT.build(d)
    y, move = FT.targets(c, vol)
    allc = FT.cols(FT.SCHEMES)
    best = {}
    for h in FT.HORIZONS:
        cands = {"LR all": ("lr", FT.SCHEMES)}
        if not quick:
            cands["HGB all"] = ("hgb", FT.SCHEMES)
            for s in FT.SCHEMES:
                cands[f"LR only {s}"] = ("lr", [s])
                cands[f"LR all−{s}"] = ("lr", [x for x in FT.SCHEMES if x != s])
        print(f"\n=== horizon {h}h ===")
        res = {}
        for name, (kind, sch) in cands.items():
            o = walk_forward(F, y[h], move, FT.cols(sch), h, kind)
            sel, hold = o[o.index < SPLIT], o[o.index >= SPLIT]
            res[name] = (kind, sch, score(sel, h), score(hold, h), float((o["p"] - 0.5).abs().quantile(1 - COVER)) if len(o) else 0.05)
            print(f"{name:12} SELECT  {fmt(res[name][2])}")
            print(f"{'':12} HOLDOUT {fmt(res[name][3])}")
        idx = F.index[F.index >= SPLIT]
        print("baselines HOLDOUT:", {k: f"{v:.1%}" for k, v in baselines(F, y[h], move, h, idx).items()})
        name = max(res, key=lambda k: res[k][2].get("brier_skill", -9))  # chosen on SELECT only
        kind, sch, sel, hold, margin = res[name]
        enabled = bool(sel.get("z") and sel["z"] >= 2 and sel["brier_skill"] > 0)  # judged on SELECT only
        best[f"{h}h"] = {"h": h, "model": kind, "schemes": sch, "chosen": name, "margin": margin, "enabled": enabled, "select": sel, "backtest": hold}
        print(f"→ chosen on SELECT: {name}")
    main = max(best, key=lambda k: best[k]["select"].get("brier_skill", -9))
    json.dump({"main": main, "horizons": best, "split": str(SPLIT.date())}, open("model.json", "w"), indent=1, default=float)
    print(f"\nmodel.json written · main horizon {main} (chosen on SELECT Brier skill)")


if __name__ == "__main__":
    run(quick="--quick" in sys.argv)
