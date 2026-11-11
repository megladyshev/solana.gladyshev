"""Volatility features and target. Every column at hour t uses only what was public at t.

Target: the size of the coming move, log((max high − min low) over the next H hours / close now).
The range is what a trader actually lives through: it decides whether a stop is hit.
Index = candle close time (UTC). Archived data (derivatives, stablecoins, sentiment) is read LAG
hours late, the same in history and live.
"""
import numpy as np
import pandas as pd

LAG = pd.Timedelta(hours=30)   # daily archives appear within a day; 30 h is safely after that
HORIZONS = [1, 4, 12, 24]
CLIM = 720                     # "typical" range = mean of the last 30 days
STORM_Q, STORM_WIN = 0.8, 2160 # a storm = a range above the 80th percentile of the last 90 days

# feature → (scheme, name shown in the app)
NAMES = {
    "hl1": ("V", "Размах последнего часа"), "rv4": ("V", "Волатильность за 4 ч"), "rv24": ("V", "Волатильность за сутки"),
    "rv168": ("V", "Волатильность за неделю"), "rv720": ("V", "Волатильность за месяц"), "rng24": ("V", "Диапазон за сутки"),
    "vol24": ("V", "Объём торгов к норме"),
    "hs1": ("T", "Время суток"), "hc1": ("T", "Время суток"), "hs2": ("T", "Время суток"), "hc2": ("T", "Время суток"), "wkend": ("T", "Выходные"),
    "r24": ("A", "Движение за сутки"), "r168": ("A", "Движение за неделю"), "ar24": ("A", "Сила движения за сутки"), "dd7": ("A", "Просадка от недельного максимума"),
    "brv24": ("E", "Волатильность BTC"), "brv4": ("E", "Волатильность BTC за 4 ч"), "rel": ("E", "SOL нервнее BTC"),
    "oi24": ("B", "Открытый интерес за сутки"), "aoi24": ("B", "Резкость изменения OI"), "ls_acc": ("B", "Толпа long/short"), "taker_ls": ("B", "Фьючерсы: агрессивные покупки"),
    "prem8": ("C", "Премия/funding"), "aprem8": ("C", "Перекос премии/funding"),
    "stable7": ("F", "Стейблкоины на Solana"),
    "fng": ("G", "Страх и жадность"), "afng": ("G", "Экстремум страха/жадности"),
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


def _lrv(r, w):
    return np.log(np.sqrt((r ** 2).rolling(w, min_periods=max(2, w // 2)).mean()) + 1e-6)


def build(d):
    sol = d["sol"]
    c, hi, lo = sol["close"], sol["high"], sol["low"]
    idx = c.index
    r = _lr(c, 1)
    s1 = r.rolling(168, min_periods=48).std()
    f = pd.DataFrame(index=idx)
    f["hl1"] = np.log((hi - lo) / c + 1e-6)
    for w in (4, 24, 168, 720):
        f[f"rv{w}"] = _lrv(r, w)
    f["rng24"] = np.log((hi.rolling(24).max() - lo.rolling(24).min()) / c)
    qv = np.log(sol["qvol"].rolling(24).sum())
    f["vol24"] = qv - qv.rolling(720, min_periods=180).mean()
    hr = idx.hour.values
    for k in (1, 2):
        f[f"hs{k}"], f[f"hc{k}"] = np.sin(2 * np.pi * k * hr / 24), np.cos(2 * np.pi * k * hr / 24)
    f["wkend"] = (idx.dayofweek >= 5).astype(float)
    f["r24"] = _lr(c, 24) / (s1 * np.sqrt(24))
    f["r168"] = _lr(c, 168) / (s1 * np.sqrt(168))
    f["ar24"] = f["r24"].abs()
    f["dd7"] = np.log(c / hi.rolling(168).max())

    b = d["btc"].reindex(idx)
    br = _lr(b["close"], 1)
    f["brv24"], f["brv4"] = _lrv(br, 24), _lrv(br, 4)
    f["rel"] = f["rv24"] - f["brv24"]

    if d.get("metrics") is not None:
        m = _lagged(d["metrics"], idx)
        f["oi24"] = _z(np.log(m["oi_usd"]).diff(24))
        f["aoi24"] = f["oi24"].abs()
        f["ls_acc"] = _z(np.log(m["ls_acc"]))
        f["taker_ls"] = _z(np.log(m["taker_ls"]).rolling(24).mean())
    if d.get("premium") is not None:
        p = _lagged(d["premium"], idx)["premium"]
        f["prem8"] = _z(p.rolling(8).mean())
        f["aprem8"] = f["prem8"].abs()
    if d.get("stable") is not None:
        f["stable7"] = _z(np.log(_lagged(d["stable"], idx)["stable"]).diff(168))
    if d.get("fng") is not None:
        g = _lagged(d["fng"], idx)["fng"]
        f["fng"], f["afng"] = (g - 50) / 25, (g - 50).abs() / 25

    f = f.replace([np.inf, -np.inf], np.nan)
    return f, sol


def targets(sol):
    """For each horizon: y = log of the coming range, plus what was known about ranges at t
    (climatology = typical range, storm threshold), all without looking past t."""
    c, hi, lo = sol["close"], sol["high"], sol["low"]
    out = {}
    for h in HORIZONS:
        fh = pd.concat([hi.shift(-k) for k in range(1, h + 1)], axis=1).max(axis=1, skipna=False)
        fl = pd.concat([lo.shift(-k) for k in range(1, h + 1)], axis=1).min(axis=1, skipna=False)
        y = np.log((fh - fl) / c).replace([np.inf, -np.inf], np.nan)   # a frozen market (zero range) is not a forecastable window
        done = y.shift(h)                                   # ranges whose window has already closed at t
        out[h] = pd.DataFrame({"y": y, "clim": done.rolling(CLIM, min_periods=200).mean(),
                               "last": done, "thr": done.rolling(STORM_WIN, min_periods=500).quantile(STORM_Q)})
    return out


def cols(schemes):
    return [k for k, (s, _) in NAMES.items() if s in schemes]
