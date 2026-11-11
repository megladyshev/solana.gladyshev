"""Hourly run: refit on all known history, make the call for each horizon, grade old calls, write the app's
JSON, and tell Telegram about a new call on the main horizon.

    python predict.py             (needs TG_TOKEN, TG_CHAT, APP_URL in the environment to notify)
    python predict.py --dry-run   print only, send nothing

Calls are made on a fixed grid (every h hours, UTC) so each one is scored once, after h hours.
"""
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as BT
import data
import features as FT

OUT = Path(__file__).parent / "docs"
EVENTS = Path(__file__).parent / "events.txt"   # lines like "2026-12-10T19:00Z FOMC"; ±2 h around one = no call
WORDS = {"UP": "ВВЕРХ", "DOWN": "ВНИЗ"}


def load_json(p, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def event_near(t):
    if not EVENTS.exists():
        return None
    for line in EVENTS.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            when, _, what = line.strip().partition(" ")
            if abs(pd.Timestamp(when).tz_convert("UTC") - t) <= pd.Timedelta(hours=2):
                return what or "событие"
    return None


def reasons(F, y, cols, t, k=5):
    """Pushes of each feature on the call, from a linear model: coefficient × standardized value."""
    X = F[cols]
    ok = X.notna().all(axis=1) & y.notna()
    m = BT.make("lr").fit(X[ok].values, y[ok].values)
    sc, lr = m.steps[0][1], m.steps[1][1]
    push = lr.coef_[0] * (X.loc[t].values - sc.mean_) / sc.scale_ * 0.25      # logit → ≈ probability near 0.5
    top = np.argsort(-np.abs(push))[:k]
    return [{"name": FT.NAMES[cols[i]][1], "scheme": FT.NAMES[cols[i]][0], "push": round(float(push[i]), 4)} for i in top]


def live_stats(hist, h, since, now):
    out = {}
    for days in (30, 90):
        a = max(pd.Timestamp(since), now - pd.Timedelta(days=days))
        calls = [x for x in hist if x["h"] == h and pd.Timestamp(x["t"]) >= a]
        graded = [x for x in calls if x["result"] in ("hit", "miss")]
        windows = max(1, int((now - a) / pd.Timedelta(hours=int(h[:-1]))))
        out[f"hit{days}"] = sum(x["result"] == "hit" for x in graded) / len(graded) if graded else None
        out[f"cov{days}"] = min(1.0, len(calls) / windows) if now - a >= pd.Timedelta(days=1) else None
        out[f"n{days}"] = len(graded)
    return out


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
    F, c, vol = FT.build(d)
    y, move = FT.targets(c, vol)
    now = c.index[-1]
    prev = load_json(OUT / "data.json", {})
    hist = load_json(OUT / "history.json", [])

    for x in hist:                                   # grade calls whose window has closed
        if x["result"] is None and pd.Timestamp(x["checkAt"]) <= now:
            after = float(c.asof(pd.Timestamp(x["checkAt"])))
            mv = np.log(after / x["price"])
            x["priceAfter"] = after
            x["result"] = "flat" if abs(mv) < x["dz"] else "hit" if (mv > 0) == (x["call"] == "UP") else "miss"

    out = {"updated": now.isoformat(), "price": float(c.iloc[-1]), "change24h": float(c.iloc[-1] / c.iloc[-25] - 1),
           "spark": [round(float(v), 4) for v in c.iloc[-168:]], "main": cfg["main"], "event": None, "horizons": {}}
    new_main = None
    for key, hc in cfg["horizons"].items():
        h = hc["h"]
        t = now.floor(f"{h}h") if h < 24 else now.normalize()     # latest grid time; its row is in F
        cols = [k for k in FT.cols(hc["schemes"]) if k in F]
        ok = F[cols].notna().all(axis=1) & y[h].notna()
        p = float(BT.make(hc["model"]).fit(F.loc[ok, cols].values, y[h][ok].values).predict_proba(F.loc[[t], cols].values)[0, 1])
        ev = event_near(t)
        off = not hc.get("enabled", True)          # backtest found no reliable signal on this horizon
        call = "NONE" if off or ev or abs(p - 0.5) < hc["margin"] else ("UP" if p > 0.5 else "DOWN")
        if ev and key == cfg["main"]:
            out["event"] = f"{ev}: сигнал не даём ±2 ч"
        check = t + pd.Timedelta(hours=h)
        if call != "NONE" and not any(x["h"] == key and x["t"] == t.isoformat() for x in hist):
            hist.append({"t": t.isoformat(), "h": key, "call": call, "p_up": round(p, 4), "price": float(c[t]), "checkAt": check.isoformat(),
                         "dz": float(FT.DEAD * vol[t] * np.sqrt(h)), "result": None, "priceAfter": None})
            if key == cfg["main"]:
                new_main = (call, p, t)
        since = ((prev.get("horizons") or {}).get(key) or {}).get("since") or t.isoformat()
        out["horizons"][key] = {"call": call, "p_up": round(p, 4), "threshold": round(0.5 + hc["margin"], 4), "checkAt": check.isoformat(),
                                "reasons": reasons(F, y[h], cols, t), "since": since, "enabled": not off,
                                "backtest": {k: hc["backtest"].get(k) for k in ("hit", "coverage", "n")},
                                "live": live_stats(hist, key, since, now)}
        print(f"{key:>4}: {call:4} p_up {p:.3f} (threshold ±{hc['margin']:.3f}) at {t:%Y-%m-%d %H:%M}Z → check {check:%m-%d %H:%M}Z")

    OUT.mkdir(exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    (OUT / "history.json").write_text(json.dumps(hist[-2000:], ensure_ascii=False), encoding="utf-8")
    if new_main and not dry:
        call, p, t = new_main
        b = cfg["horizons"][cfg["main"]]["backtest"]
        telegram(f"<b>SOL {WORDS[call]}</b> на {cfg['main']} · вероятность роста {p:.0%}\n"
                 f"Цена ${out['price']:.2f} · проверка через {cfg['main']}\n"
                 + (f"<i>Бэктест: {b['hit']:.0%} попаданий на уверенных сигналах. </i>" if b.get("hit") else "")
                 + "<i>Не финансовый совет.</i>")


if __name__ == "__main__":
    main(dry="--dry-run" in sys.argv)
