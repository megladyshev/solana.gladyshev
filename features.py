"""Schemes A–G → feature columns. Every column at hour t uses only what was public at t.

Index = candle close time (UTC). Returns are scaled by recent volatility so the numbers mean the same
thing in 2021 and in 2026. Archived data (derivatives, stablecoins, sentiment) is read LAG hours late,
the same in history and live.
"""
import numpy as np
import pandas as pd

LAG = pd.Timedelta(hours=30)   # daily archives appear within a day; 30 h is safely after that
HORIZONS = [1, 4, 12, 24]
DEAD = 0.25                    # |move| under 0.25 σ of the horizon is "flat": not scored

# feature → (scheme, short name shown in the app)
NAMES = {
    "r1": ("A", "Последний час"), "r4": ("A", "Последние 4 ч"), "r24": ("A", "Тренд за сутки"), "r72": ("A", "Тренд за 3 дня"),
    "r168": ("A", "Тренд за неделю"), "r336": ("A", "Тренд за 2 недели"), "r672": ("A", "Тренд за 4 недели"), "ma20d": ("A", "Отрыв от 20-дневной средней"),
    "rv_ratio": ("D", "Волатильность: сжатие/расширение"), "rv_z": ("D", "Уровень волатильности"), "range7": ("D", "Место в 7-дневном диапазоне"),
    "btc4": ("E", "BTC за 4 ч"), "btc24": ("E", "BTC за сутки"), "btc168": ("E", "BTC за неделю"), "solbtc168": ("E", "SOL/BTC за неделю"), "eth24": ("E", "ETH за сутки"),
    "taker24": ("B", "Спот: покупатели vs продавцы"), "oi24": ("B", "Открытый интерес за сутки"), "oi168": ("B", "Открытый интерес за неделю"),
    "ls_top": ("B", "Топ-трейдеры long/short"), "ls_acc": ("B", "Толпа long/short"), "taker_ls": ("B", "Фьючерсы: агрессивные покупки"),
    "prem8": ("C", "Премия/funding за 8 ч"), "prem72": ("C", "Премия/funding за 3 дня"),
    "stable7": ("F", "Стейблкоины на Solana за неделю"), "stable30": ("F", "Стейблкоины на Solana за месяц"),
    "fng": ("G", "Индекс страха и жадности"), "fng7": ("G", "Изменение страха/жадности"),
}
SCHEMES = sorted({s for s, _ in NAMES.values()})


def _lr(c, k):
    return np.log(c / c.shift(k))


def _z(s, w=720):
    return (s - s.rolling(w, min_periods=w // 4).mean()) / s.rolling(w, min_periods=w // 4).std()


def _lagged(df, idx):
    """Values of an archived/daily frame as known at each hour of idx (LAG late)."""
    df = df.sort_index()
    df.index = df.index.as_unit("us")
    out = pd.merge_asof(pd.DataFrame(index=(idx - LAG).as_unit("us")), df, left_index=True, right_index=True, direction="backward")
    out.index = idx
    return out


def build(d):
    sol = d["sol"]
    c = sol["close"]
    idx = c.index
    r = _lr(c, 1)
    vol = r.rolling(168, min_periods=48).std()           # σ of 1 h returns over a week
    f = pd.DataFrame(index=idx)
    for k in (1, 4, 24, 72, 168, 336, 672):
        f[f"r{k}"] = _lr(c, k) / (vol * np.sqrt(k))
    f["ma20d"] = np.log(c / c.rolling(480).mean()) / (vol * np.sqrt(480))
    f["rv_ratio"] = np.log(r.rolling(24).std() / vol)
    f["rv_z"] = _z(np.log(vol), 2160)
    hi, lo = sol["high"].rolling(168).max(), sol["low"].rolling(168).min()
    f["range7"] = (c - lo) / (hi - lo) - 0.5
    f["taker24"] = sol["taker_q"].rolling(24).sum() / sol["qvol"].rolling(24).sum() - 0.5

    for name, k in (("btc", 4), ("btc", 24), ("btc", 168), ("eth", 24)):
        x = d[name]["close"].reindex(idx)
        xv = _lr(x, 1).rolling(168, min_periods=48).std()
        f[f"{name}{k}"] = _lr(x, k) / (xv * np.sqrt(k))
    f["solbtc168"] = _lr(c / d["btc"]["close"].reindex(idx), 168) / (vol * np.sqrt(168))

    if d.get("metrics") is not None:
        m = _lagged(d["metrics"], idx)
        f["oi24"], f["oi168"] = _z(np.log(m["oi_usd"]).diff(24)), _z(np.log(m["oi_usd"]).diff(168))
        f["ls_top"], f["ls_acc"] = _z(np.log(m["ls_top"])), _z(np.log(m["ls_acc"]))
        f["taker_ls"] = _z(np.log(m["taker_ls"]).rolling(24).mean())
    if d.get("premium") is not None:
        p = _lagged(d["premium"], idx)["premium"]
        f["prem8"], f["prem72"] = _z(p.rolling(8).mean()), _z(p.rolling(72).mean())
    if d.get("stable") is not None:
        s = np.log(_lagged(d["stable"], idx)["stable"])
        f["stable7"], f["stable30"] = _z(s.diff(168)), _z(s.diff(720))
    if d.get("fng") is not None:
        g = _lagged(d["fng"], idx)["fng"]
        f["fng"], f["fng7"] = (g - 50) / 25, g.diff(168) / 25

    f = f.replace([np.inf, -np.inf], np.nan)
    return f, c, vol


def targets(c, vol):
    """y[h] = 1 up / 0 down / NaN flat or unknown, and the log move itself."""
    y, move = {}, {}
    for h in HORIZONS:
        m = np.log(c.shift(-h) / c)
        dz = DEAD * vol * np.sqrt(h)
        y[h] = pd.Series(np.where(m > dz, 1.0, np.where(m < -dz, 0.0, np.nan)), index=c.index).where(m.notna())
        move[h] = m
    return y, move


def cols(schemes):
    return [k for k, (s, _) in NAMES.items() if s in schemes]
