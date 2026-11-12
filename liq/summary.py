"""What the collector has recorded so far: coverage, totals, and the biggest liquidation bursts.

    python liq/summary.py
"""
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parent.parent / "data" / "liq"
load = lambda kind: pd.concat([pd.read_csv(f) for f in sorted(D.glob(f"{kind}_*.csv"))], ignore_index=True) if list(D.glob(f"{kind}_*.csv")) else pd.DataFrame()

liq, up = load("liq"), load("up")
if liq.empty:
    raise SystemExit("nothing recorded yet: run python liq/collect.py")
liq = liq.drop_duplicates(["exchange", "symbol", "ts_ms", "side", "qty"])   # OKX replays recent ones on (re)connect
liq["t"] = pd.to_datetime(liq.ts_ms, unit="ms", utc=True)
up["t"] = pd.to_datetime(up.ts_ms, unit="ms", utc=True)

span = (up.t.max() - up.t.min()) if len(up) else pd.Timedelta(0)
print(f"recorded {liq.t.min():%Y-%m-%d %H:%M} → {liq.t.max():%Y-%m-%d %H:%M} UTC · {len(up)} minutes logged over {span}")
for ex in ("binance", "bybit", "okx"):
    if ex in up:
        print(f"  {ex:8} connected {up[ex].mean():.1%} of logged minutes")

print("\nliquidated, USD (all coins):")
print(liq.pivot_table(index="exchange", columns="side", values="usd", aggfunc="sum").round(0).to_string())
print("\nSOL liquidated, USD:")
sol = liq[liq.coin == "SOL"]
print(sol.pivot_table(index="exchange", columns="side", values="usd", aggfunc="sum").round(0).to_string() if len(sol) else "  none yet")

print("\nbiggest 1-minute bursts, all exchanges (USD):")
m = liq.assign(minute=liq.t.dt.floor("min")).pivot_table(index="minute", columns="side", values="usd", aggfunc="sum").fillna(0)
m["total"] = m.sum(axis=1)
print(m.nlargest(5, "total").round(0).to_string())
