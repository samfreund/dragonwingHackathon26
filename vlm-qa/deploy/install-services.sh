#!/usr/bin/env bash
# Run vlm-qa as a service: GenieX on the NPU, the WebSocket front end on the
# network, both under systemd so they come back after a crash or a reboot.
#
#   ssh ubuntu@iq9 'bash -s' < deploy/install-services.sh
#
# Idempotent. Re-running updates the units and restarts them; it never
# regenerates the auth token, so clients that already have it keep working.
set -euo pipefail

APP_DIR="${APP_DIR:-$HOME/vlm-qa}"
RUN_USER="${RUN_USER:-$(id -un)}"
ENV_DIR=/etc/vlmqa
ENV_FILE="$ENV_DIR/vlmqa.env"

# 0.0.0.0 so the phone can reach it. The token is what keeps it from being
# open season on the NPU for anything else on the wifi.
WS_HOST="${VLMQA_WS_HOST:-0.0.0.0}"
WS_PORT="${VLMQA_WS_PORT:-8765}"

log() { printf '\n=== %s\n' "$*"; }

GENIEX="$(command -v geniex || echo "$HOME/.local/bin/geniex")"
[ -x "$GENIEX" ] || { echo "geniex not found; run setup-iq9.sh first" >&2; exit 1; }
[ -x "$APP_DIR/.venv/bin/python" ] || { echo "no venv in $APP_DIR; run setup-iq9.sh first" >&2; exit 1; }

log "Writing $ENV_FILE"
sudo mkdir -p "$ENV_DIR"
if sudo test -f "$ENV_FILE"; then
  echo "Keeping the existing token in $ENV_FILE"
  # Host/port may still have moved since last time.
  sudo sed -i "s|^VLMQA_WS_HOST=.*|VLMQA_WS_HOST=$WS_HOST|; s|^VLMQA_WS_PORT=.*|VLMQA_WS_PORT=$WS_PORT|" "$ENV_FILE"
else
  TOKEN="$(openssl rand -hex 24 2>/dev/null || head -c 24 /dev/urandom | base64 | tr -d '/+=')"
  sudo tee "$ENV_FILE" >/dev/null <<EOF
# Read by vlmqa-ws.service. Root-owned and 0600: it holds the auth token.
# Clear VLMQA_WS_TOKEN to turn authentication off (only sane on loopback).
VLMQA_WS_HOST=$WS_HOST
VLMQA_WS_PORT=$WS_PORT
VLMQA_WS_TOKEN=$TOKEN
EOF
fi
sudo chown root:root "$ENV_FILE"
sudo chmod 600 "$ENV_FILE"

log "Installing systemd units"
# GenieX holds the model on the Hexagon NPU. Loading the 4.1 GiB bundle takes
# a while, so give it a generous start timeout and let it settle before the
# socket server advertises itself.
sudo tee /etc/systemd/system/geniex.service >/dev/null <<EOF
[Unit]
Description=GenieX server (Qwen3-VL on the Hexagon NPU)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$HOME
Environment=HOME=$HOME
Environment=PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$GENIEX serve
Restart=always
RestartSec=5
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/vlmqa-ws.service >/dev/null <<EOF
[Unit]
Description=vlm-qa WebSocket server (photo/video Q&A)
After=network-online.target geniex.service
Wants=network-online.target geniex.service

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
Environment=PYTHONUNBUFFERED=1
ExecStart=$APP_DIR/.venv/bin/python -m vlmqa serve
Restart=always
RestartSec=3
# Uploads land in a private /tmp that is discarded with the service.
PrivateTmp=yes
NoNewPrivileges=yes

[Install]
WantedBy=multi-user.target
EOF

log "Starting"
sudo systemctl daemon-reload
sudo systemctl enable --now geniex.service vlmqa-ws.service
sudo systemctl restart vlmqa-ws.service      # pick up any change to the unit

sleep 3
systemctl is-active geniex.service vlmqa-ws.service >/dev/null || {
  echo "A unit failed to start:" >&2
  systemctl --no-pager --lines=20 status geniex.service vlmqa-ws.service >&2 || true
  exit 1
}

log "Listening on"
TOKEN="$(sudo sed -n 's/^VLMQA_WS_TOKEN=//p' "$ENV_FILE")"
for addr in $(ip -4 -o addr show scope global | awk '{sub(/\/.*/,"",$4); print $4}'); do
  echo "  ws://$addr:$WS_PORT"
done
echo
echo "  token: ${TOKEN:-(none - authentication is off)}"
echo
cat <<EOF
Logs:     journalctl -u vlmqa-ws -f        (and -u geniex)
Restart:  sudo systemctl restart vlmqa-ws
Token:    sudo cat $ENV_FILE
EOF
