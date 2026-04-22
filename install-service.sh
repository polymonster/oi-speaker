#!/usr/bin/env bash
set -e

SERVICE="oi-speaker@${USER}"

sudo cp setup/oi-speaker@.service /etc/systemd/system/oi-speaker@.service
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE"
sudo systemctl restart "$SERVICE"

echo "Service $SERVICE installed and started."
echo "  Status:  sudo systemctl status $SERVICE"
echo "  Logs:    journalctl -u $SERVICE -f"
