"""Liquidation collector: forced orders from Binance, Bybit and OKX in real time, plus a minute snapshot of
Binance open interest, mark/index price and funding. This data is not in any free history, so it has to be
recorded from now on.

    python liq/collect.py          runs until stopped (Ctrl+C); reconnects by itself

Writes to data/liq/ (one file per kind per UTC day):
    liq_YYYY-MM-DD.csv     ts_ms, exchange, coin, symbol, side (the position that was liquidated), price, qty, usd
    oi_YYYY-MM-DD.csv      ts_ms, coin, open_interest (coins), mark, index, funding
    up_YYYY-MM-DD.csv      ts_ms, binance, bybit, okx  (1 = connected that minute), so gaps are known later

Known limits:
  * Binance pushes at most one liquidation per symbol per second (the largest snapshot), so its counts are a floor.
  * All three exchanges are recorded for every USDT perp; Bybit is subscribed symbol by symbol at connect.
  * OKX replays its most recent liquidations on every (re)connect: drop duplicates when reading (liq/summary.py does).
"""
import asyncio
import csv
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import websockets

OUT = Path(__file__).resolve().parent.parent / "data" / "liq"
OI_COINS = ["SOL", "BTC", "ETH"]
HEAD = {"liq": ["ts_ms", "exchange", "coin", "symbol", "side", "price", "qty", "usd"],
        "oi": ["ts_ms", "coin", "open_interest", "mark", "index", "funding"],
        "up": ["ts_ms", "binance", "bybit", "okx"]}
up = {"binance": 0, "bybit": 0, "okx": 0}
_files = {}


def write(kind, row):
    day = datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    key = (kind, day)
    if key not in _files:
        for k in [k for k in _files if k[0] == kind]:          # a new day: close yesterday's file
            _files.pop(k).close()
        f = OUT / f"{kind}_{day}.csv"
        new = not f.exists()
        _files[key] = f.open("a", newline="", encoding="utf-8")
        if new:
            csv.writer(_files[key]).writerow(HEAD[kind])
    csv.writer(_files[key]).writerow(row)
    _files[key].flush()


def coin_of(symbol):
    for q in ("USDT", "USDC", "-USDT-SWAP", "-USDC-SWAP", "-USD-SWAP"):
        if symbol.endswith(q):
            return symbol[: -len(q)].rstrip("-")
    return symbol


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "liq-collector/0.1"}), timeout=15) as r:
        return json.loads(r.read())


async def run_forever(name, session):
    wait = 1
    while True:
        try:
            await session()
            wait = 1
        except Exception as e:  # noqa: BLE001  (any network failure: reconnect)
            up[name] = 0
            print(f"{datetime.now():%H:%M:%S} {name}: {type(e).__name__} {str(e)[:120]} → reconnect in {wait}s", flush=True)
            await asyncio.sleep(wait)
            wait = min(60, wait * 2)


async def binance():
    async with websockets.connect("wss://fstream.binance.com/market/ws/!forceOrder@arr", ping_interval=20) as ws:
        up["binance"] = 1
        async for m in ws:
            o = json.loads(m)["o"]
            if not o["s"].endswith(("USDT", "USDC")):
                continue
            qty, px = float(o["z"]), float(o["ap"])
            write("liq", [o["T"], "binance", coin_of(o["s"]), o["s"], "long" if o["S"] == "SELL" else "short", px, qty, round(qty * px, 2)])


async def bybit():
    t = await asyncio.to_thread(get, "https://api.bybit.com/v5/market/tickers?category=linear")
    syms = [x["symbol"] for x in t["result"]["list"] if x["symbol"].endswith("USDT")]
    async with websockets.connect("wss://stream.bybit.com/v5/public/linear", ping_interval=None) as ws:
        for i in range(0, len(syms), 10):                         # Bybit takes up to 10 topics per request
            await ws.send(json.dumps({"op": "subscribe", "args": [f"allLiquidation.{s}" for s in syms[i:i + 10]]}))
        up["bybit"] = 1

        async def ping():
            while True:
                await asyncio.sleep(20)
                await ws.send(json.dumps({"op": "ping"}))
        task = asyncio.create_task(ping())
        try:
            async for m in ws:
                j = json.loads(m)
                for x in j.get("data", []) if j.get("topic", "").startswith("allLiquidation") else []:
                    qty, px = float(x["v"]), float(x["p"])
                    # Bybit: "Buy" means a long position was liquidated
                    write("liq", [x["T"], "bybit", coin_of(x["s"]), x["s"], "long" if x["S"] == "Buy" else "short", px, qty, round(qty * px, 2)])
        finally:
            task.cancel()


async def okx():
    inst = {i["instId"]: i for i in (await asyncio.to_thread(get, "https://www.okx.com/api/v5/public/instruments?instType=SWAP"))["data"]}
    async with websockets.connect("wss://ws.okx.com:443/ws/v5/public", ping_interval=None) as ws:
        await ws.send(json.dumps({"op": "subscribe", "args": [{"channel": "liquidation-orders", "instType": "SWAP"}]}))
        up["okx"] = 1

        async def ping():
            while True:
                await asyncio.sleep(25)
                await ws.send("ping")
        task = asyncio.create_task(ping())
        try:
            async for m in ws:
                if m == "pong":
                    continue
                j = json.loads(m)
                for x in j.get("data", []):
                    i = inst.get(x["instId"])
                    if not i:
                        continue
                    for dd in x.get("details", []):
                        px, sz, ctv = float(dd["bkPx"]), float(dd["sz"]), float(i["ctVal"])
                        if i.get("ctType") == "inverse":          # contract value in USD
                            usd, qty = sz * ctv, sz * ctv / px if px else 0
                        else:                                     # contract value in coins
                            qty = sz * ctv
                            usd = qty * px
                        side = "short" if dd["side"] == "buy" else "long"
                        write("liq", [int(dd["ts"]), "okx", coin_of(x["instId"]), x["instId"], side, px, round(qty, 6), round(usd, 2)])
        finally:
            task.cancel()


async def minute():
    """Every minute: Binance open interest, mark/index price, funding; and which feeds are connected."""
    while True:
        await asyncio.sleep(60 - time.time() % 60)
        ts = int(time.time() * 1000)
        write("up", [ts, up["binance"], up["bybit"], up["okx"]])
        for c in OI_COINS:
            try:
                oi = await asyncio.to_thread(get, f"https://fapi.binance.com/fapi/v1/openInterest?symbol={c}USDT")
                pi = await asyncio.to_thread(get, f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={c}USDT")
                write("oi", [ts, c, oi["openInterest"], pi["markPrice"], pi["indexPrice"], pi["lastFundingRate"]])
            except Exception as e:  # noqa: BLE001
                print(f"{datetime.now():%H:%M:%S} oi {c}: {e}", flush=True)


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"collecting into {OUT} · Ctrl+C to stop", flush=True)
    await asyncio.gather(run_forever("binance", binance), run_forever("bybit", bybit), run_forever("okx", okx), minute())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
