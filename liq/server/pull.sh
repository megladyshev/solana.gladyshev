#!/usr/bin/env bash
# Runs on YOUR computer: copies everything the server has recorded into data/liq/ (re-run any time).
#   bash liq/server/pull.sh ubuntu@<server-ip> <path-to-private-key>
set -euo pipefail
HOST=${1:?usage: pull.sh ubuntu@IP KEY}
KEY=${2:?usage: pull.sh ubuntu@IP KEY}
cd "$(dirname "$0")/../.."
mkdir -p data/liq
scp -i "$KEY" -p "$HOST:trading/data/liq/*" data/liq/
# a day pulled before it was gzipped and again after: keep only the gzipped copy
for f in data/liq/*.csv.gz; do [ -e "${f%.gz}" ] && rm "${f%.gz}"; done; true
ls data/liq | tail -5
