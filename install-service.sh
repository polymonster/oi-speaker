#!/usr/bin/env bash
set -e

# Any args passed here are appended to the speaker command line, eg:
#   bash install-service.sh --worker-ip 192.168.1.247:8000

SERVICE="oi-speaker@${USER}"
ENV_DIR="$HOME/.config/oi-speaker"
ENV_FILE="$ENV_DIR/env"
DROPIN_DIR="/etc/systemd/system/${SERVICE}.service.d"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UID_NUM="$(id -u)"

# python used by the service — honour an active venv, else the first python3.11 on PATH
PYTHON="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
PYTHON="${PYTHON:-$(command -v python3.11 || command -v python3)}"
if [ ! -x "$PYTHON" ]; then
    echo "no python found — activate your venv or install python3.11" >&2
    exit 1
fi

mkdir -p "$ENV_DIR"
{
    echo "OI_SPEAKER_PYTHON=$PYTHON"
    echo "OI_SPEAKER_ARGS=$*"
} > "$ENV_FILE"

sudo cp setup/oi-speaker@.service /etc/systemd/system/oi-speaker@.service

# %h/%U resolve against the service manager (root), not User=, so the per-user absolute
# paths have to be written out here.
sudo mkdir -p "$DROPIN_DIR"
sudo tee "$DROPIN_DIR/paths.conf" > /dev/null <<EOF
[Unit]
After=user@$UID_NUM.service
Wants=user@$UID_NUM.service

[Service]
WorkingDirectory=$REPO_DIR
EnvironmentFile=-$ENV_FILE
Environment=XDG_RUNTIME_DIR=/run/user/$UID_NUM
Environment=PULSE_SERVER=unix:/run/user/$UID_NUM/pulse/native
EOF

# /run/user/$UID_NUM only exists while the user has a session — linger keeps it alive from
# boot so the service can reach pulse without anyone logging in.
sudo loginctl enable-linger "$USER"

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE"
sudo systemctl restart "$SERVICE"

echo "Service $SERVICE installed and started."
echo "  Python:  $PYTHON"
echo "  Args:    $*"
echo "  Workdir: $REPO_DIR"
echo "  Env:     $ENV_FILE"
echo "  Dropin:  $DROPIN_DIR/paths.conf"
echo "  Linger:  enabled for $USER"
echo "  Status:  sudo systemctl status $SERVICE"
echo "  Logs:    journalctl -u $SERVICE -f"
