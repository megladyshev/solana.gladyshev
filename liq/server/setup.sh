#!/usr/bin/env bash
# Runs ON the cloud server (Ubuntu), from ~/trading. Installs the collector as a systemd service that
# restarts itself on failure and starts on boot, plus a nightly job that gzips finished days.
set -euo pipefail
cd ~/trading

sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv >/dev/null
python3 -m venv .venv
.venv/bin/pip install -q "websockets==17.0.1"
mkdir -p data/liq

sudo tee /etc/systemd/system/liq-collector.service >/dev/null <<EOF
[Unit]
Description=Liquidation collector (Binance, Bybit, OKX)
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$HOME/trading
ExecStart=$HOME/trading/.venv/bin/python liq/collect.py
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 00:30 UTC: gzip every daily file that is not today's (the collector has moved on to the new day by then)
( crontab -l 2>/dev/null | grep -v liq-gzip ; \
  echo "30 0 * * * find $HOME/trading/data/liq -name '*_20*.csv' ! -name \"*\$(date -u +\\%F)*\" -exec gzip -f {} + # liq-gzip" ) | crontab -

sudo timedatectl set-ntp true
sudo systemctl daemon-reload
sudo systemctl enable --now liq-collector
sleep 5
systemctl --no-pager --lines=5 status liq-collector
