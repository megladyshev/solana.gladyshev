"""Hourly run: refit on all known history, forecast how big the coming move is for each horizon,
grade old forecasts, write the app's JSON, and warn in Telegram when a storm is coming.

    python predict.py             (needs TG_TOKEN, TG_CHAT, APP_URL in the environment to notify)
    python predict.py --dry-run   print only, send nothing

Forecasts are made on a fixed grid (every h hours, UTC) so each one is graded once, after h hours.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import backtest as BT
import data
import features as FT

OUT = Path(__file__).parent / "docs"
STORM_P, CALM_RATIO = 0.4, 0.8   # storm: P(top-20% range) ≥ 40%; calm: forecast under 0.8× the typical range


def load_json(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def reasons(F, t, cols, at, k=5):
    """Push of each feature on the forecast range, from a linear model: × change of the range."""
    ok = F[cols].notna().all(axis=1) & t["y"].notna()
    m = BT.make("ridge").fit(F.loc[ok, cols].values, t.loc[ok, "y"].values)
    sc, lr = m.steps[0][1], m.steps[1][1]
    push = lr.coef_ * (F.loc[at, cols].values - sc.mean_) / sc.scale_
    agg = {}                                       # time-of-day columns are one factor
    for i, c in enumerate(cols):
        name = FT.NAMES[c]
        agg[name] = agg.get(name, 0.0) + push[i]
    top = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:k]
    return [{"name": n, "scheme": s, "push": round(float(np.exp(v) - 1), 4)} for (s, n), v in top]


def live_stats(hist, key, now):
    a = now - pd.Timedelta(days=30)
    g = [x for x in hist if x["h"] == key and x.get("result") and pd.Timestamp(x["t"]) >= a]
    st = [x for x in g if x["label"] == "storm"]
    return {"cover30": sum(x["result"] == "inside" for x in g) / len(g) if g else None, "n30": len(g),
            "storm_ok30": sum(x["storm_act"] for x in st) / len(st) if st else None, "storm_n30": len(st)}


def telegram(text):
    tok, chat, url = os.environ.get("TG_TOKEN"), os.environ.get("TG_CHAT"), os.environ.get("APP_URL")
    if not (tok and chat):
        return print("TG_TOKEN/TG_CHAT not set: no message")
    body = {"chat_id": chat, "text": text, "parse_mode": "HTML"}
    if url:
        body["reply_markup"] = json.dumps({"inline_keyboard": [[{"text": "Открыть прогноз", "web_app": {"url": url}}]]})
    urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", urllib.parse.urlencode(body).encode(), timeout=20)


def main(dry=False):
    cfg = json.loads((Path(__file__).parent / "model.json").read_text())
    d = data.load()
    F, sol = FT.build(d)
    T = FT.targets(sol)
    c = sol["close"]
    now = c.index[-1]
    prev = load_json(OUT / "data.json", {})
    hist = load_json(OUT / "history.json", [])

    for x in hist:                                   # grade forecasts whose window has closed
        if x.get("result") is None and pd.Timestamp(x["checkAt"]) <= now:
            w = sol[(sol.index > pd.Timestamp(x["t"])) & (sol.index <= pd.Timestamp(x["checkAt"]))]
            if len(w):
                r = float((w["high"].max() - w["low"].min()) / x["price"])
                x["range_act"] = round(r, 5)
                x["result"] = "inside" if x["lo"] <= r <= x["hi"] else "above" if r > x["hi"] else "below"
                x["storm_act"] = bool(r > x["thr"])

    out = {"updated": now.isoformat(), "price": float(c.iloc[-1]), "change24h": float(c.iloc[-1] / c.iloc[-25] - 1),
           "spark": [round(float(v), 4) for v in c.iloc[-168:]], "main": cfg["main"], "event": None, "horizons": {}}
    storm_now = None
    for key, hc in cfg["horizons"].items():
        h, t = hc["h"], None
        tg = T[h]
        at = now.floor(f"{h}h") if h < 24 else now.normalize()
        cols = [k for k in FT.cols(hc["schemes"]) if k in F]
        ok = F[cols].notna().all(axis=1) & tg["y"].notna()
        m, s, q10, q90 = BT.fit(F, tg, cols, hc["model"], ok)
        pred = float(m.predict(F.loc[[at], cols].values)[0])
        clim, thr = float(tg.loc[at, "clim"]), float(tg.loc[at, "thr"])
        p_storm = float(1 - norm.cdf((thr - pred) / s))
        ratio = float(np.exp(pred - clim))
        label = "off" if not hc.get("enabled", True) else "storm" if p_storm >= STORM_P else "calm" if ratio < CALM_RATIO else "normal"
        rng, lo, hi = np.exp(pred), np.exp(pred + q10), np.exp(pred + q90)
        check = at + pd.Timedelta(hours=h)
        if not any(x["h"] == key and x["t"] == at.isoformat() for x in hist):
            hist.append({"t": at.isoformat(), "h": key, "label": label, "p_storm": round(p_storm, 4), "range": round(float(rng), 5),
                         "lo": round(float(lo), 5), "hi": round(float(hi), 5), "thr": round(float(np.exp(thr)), 5), "price": float(c[at]),
                         "checkAt": check.isoformat(), "result": None, "range_act": None, "storm_act": None})
            if key == cfg["main"] and label == "storm" and ((prev.get("horizons") or {}).get(key) or {}).get("label") != "storm":
                storm_now = (rng, lo, hi, p_storm)
        b = hc["backtest"]
        out["horizons"][key] = {"label": label, "enabled": label != "off", "p_storm": round(p_storm, 4), "range": round(float(rng), 5),
                                "lo": round(float(lo), 5), "hi": round(float(hi), 5), "typical": round(float(np.exp(clim)), 5), "ratio": round(ratio, 3),
                                "storm_range": round(float(np.exp(thr)), 5), "price": float(c[at]), "checkAt": check.isoformat(),
                                "reasons": reasons(F, tg, cols, at),
                                "backtest": {k: b.get(k) for k in ("r2_clim", "cover", "storm_bss", "lift", "mae_pct", "n")},
                                "live": live_stats(hist, key, now)}
        print(f"{key:>4}: {label:6} range {rng:.2%} (80%: {lo:.2%}–{hi:.2%}) typical {np.exp(clim):.2%} ×{ratio:.2f} · P(storm) {p_storm:.0%}")

    OUT.mkdir(exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (OUT / "history.json").write_text(json.dumps(hist[-3000:], ensure_ascii=False), encoding="utf-8")
    if storm_now and not dry:
        rng, lo, hi, p = storm_now
        telegram(f"<b>SOL: впереди сильное движение</b> · {cfg['main']}\n"
                 f"Ожидаемый размах ≈{rng:.1%} (обычно {out['horizons'][cfg['main']]['typical']:.1%}), "
                 f"80%: {lo:.1%}–{hi:.1%} · вероятность бури {p:.0%}\n"
                 f"Цена ${out['price']:.2f}. Направление неизвестно — только размер.\n<i>Не финансовый совет.</i>")


if __name__ == "__main__":
    main(dry="--dry-run" in sys.argv)
