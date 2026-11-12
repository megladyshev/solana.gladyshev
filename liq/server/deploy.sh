#!/usr/bin/env bash
# Runs on YOUR computer (Git Bash). Copies the collector to the server and installs it; safe to re-run
# after changing liq/collect.py.
#   bash liq/server/deploy.sh ubuntu@<server-ip> <path-to-private-key>
set -euo pipefail
HOST=${1:?usage: deploy.sh ubuntu@IP KEY}
KEY=${2:?usage: deploy.sh ubuntu@IP KEY}
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$HOST")
cd "$(dirname "$0")/../.."

echo "== can the server reach the exchanges? (403/451 = blocked region)"
"${SSH[@]}" 'for u in https://fapi.binance.com/fapi/v1/ping https://api.bybit.com/v5/market/time https://www.okx.com/api/v5/public/time; do echo "$(curl -s -o /dev/null -w %{http_code} --max-time 10 $u) $u"; done'

"${SSH[@]}" 'mkdir -p ~/trading/liq/server'
scp -i "$KEY" liq/collect.py "$HOST:trading/liq/collect.py"
scp -i "$KEY" liq/server/setup.sh "$HOST:trading/liq/server/setup.sh"
"${SSH[@]}" 'bash ~/trading/liq/server/setup.sh'
