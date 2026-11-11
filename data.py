"""Free data for the SOL predictor, cached in data/.

Spot klines come from data-api.binance.vision (live). Derivatives come from the data.binance.vision
daily archives, which are published about a day late. That lag is applied the same way in training and
live (see features.LAG), so the model never sees in history what it could not see live.
"""
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

D = Path(__file__).parent / "data"
D.mkdir(exist_ok=True)
START = "2021-01-01"
UA = {"User-Agent": "sol-predictor/0.1"}


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if i == tries - 1:
                raise
        except OSError:
            if i == tries - 1:
                raise
        time.sleep(2 * (i + 1))


def _cached(name):
    f = D / name
    return f, (pd.read_csv(f, index_col=0, parse_dates=[0]) if f.exists() else None)


def klines(symbol):
    """Hourly spot candles, indexed by CLOSE time (UTC): the row at t is known at t."""
    f, df = _cached(f"{symbol}_1h.csv")
    t = int(df.index[-1].timestamp() * 1000) - 3600_000 + 1 if df is not None else int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    rows = []
    while True:
        k = json.loads(get(f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&startTime={t}&limit=1000"))
        if not k:
            break
        rows += k
        t = k[-1][0] + 1
        if len(k) < 1000:
            break
    now = time.time() * 1000
    rows = [r for r in rows if r[6] < now]  # closed candles only
    if rows:
        new = pd.DataFrame({"close": [float(r[4]) for r in rows], "high": [float(r[2]) for r in rows], "low": [float(r[3]) for r in rows],
                            "qvol": [float(r[7]) for r in rows], "taker_q": [float(r[10]) for r in rows]},
                           index=pd.to_datetime([r[0] + 3600_000 for r in rows], unit="ms", utc=True))
        df = new if df is None else pd.concat([df, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(f)
    return df


def _day_csv(url):
    b = get(url)
    if b is None:
        return None
    z = zipfile.ZipFile(io.BytesIO(b))
    raw = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None, dtype=str)
    return raw[~raw[0].str.contains("time", na=True)]  # some files have a header row, some do not


def _archive(name, url_of, parse):
    """Daily archive files → one cached frame; only missing days are downloaded."""
    f, df = _cached(name)
    first = (df.index[-1].normalize() + pd.Timedelta(days=1)) if df is not None else pd.Timestamp(START, tz="UTC")
    days = pd.date_range(first, pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1), freq="D")
    with ThreadPoolExecutor(16) as ex:
        parts = [parse(p) for p in ex.map(lambda d: _day_csv(url_of(d.strftime("%Y-%m-%d"))), days) if p is not None and len(p)]
    if parts:
        new = pd.concat(parts)
        df = new if df is None else pd.concat([df, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(f)
    return df


def metrics(symbol="SOLUSDT"):
    """Futures open interest and long/short ratios, hourly."""
    def parse(r):
        x = pd.DataFrame({"oi_usd": r[3].astype(float), "ls_top": r[5].astype(float), "ls_acc": r[6].astype(float), "taker_ls": r[7].astype(float)})
        x.index = pd.to_datetime(r[0], utc=True)
        return x.resample("1h", label="right", closed="right").last().dropna()
    return _archive(f"{symbol}_metrics.csv", lambda d: f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{d}.zip", parse)


def premium(symbol="SOLUSDT"):
    """Perp premium index (what funding is paid on), hourly, indexed by close time."""
    def parse(r):
        return pd.DataFrame({"premium": r[4].astype(float).values}, index=pd.to_datetime(r[0].astype("int64") + 3600_000, unit="ms", utc=True))
    return _archive(f"{symbol}_premium.csv", lambda d: f"https://data.binance.vision/data/futures/um/daily/premiumIndexKlines/{symbol}/1h/{symbol}-1h-{d}.zip", parse)


def stablecoins():
    """USD stablecoins circulating on Solana, daily (DefiLlama)."""
    j = json.loads(get("https://stablecoins.llama.fi/stablecoincharts/Solana"))
    s = pd.Series({pd.Timestamp(int(x["date"]), unit="s", tz="UTC"): (x.get("totalCirculatingUSD") or {}).get("peggedUSD") for x in j}, name="stable").dropna()
    return s.to_frame()


def fear_greed():
    j = json.loads(get("https://api.alternative.me/fng/?limit=0"))["data"]
    return pd.Series({pd.Timestamp(int(x["timestamp"]), unit="s", tz="UTC"): float(x["value"]) for x in j}, name="fng").sort_index().to_frame()


def load():
    return {"sol": klines("SOLUSDT"), "btc": klines("BTCUSDT"), "eth": klines("ETHUSDT"),
            "metrics": metrics(), "premium": premium(), "stable": stablecoins(), "fng": fear_greed()}


if __name__ == "__main__":
    for k, v in load().items():
        print(f"{k:8} {len(v):7} rows  {v.index[0]} -> {v.index[-1]}")
