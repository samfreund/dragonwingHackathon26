#!/usr/bin/env bash
# Provision a Dragonwing IQ-9075 (QCS9075) to run Qwen3-VL-4B on the Hexagon NPU.
#
# Reproduces, from a stock Ubuntu 24.04 arm64 image, the exact stack this app
# was verified against. Safe to re-run: every step is idempotent.
#
#   ssh ubuntu@iq9 'bash -s' < setup-iq9.sh
set -euo pipefail

MODEL="${MODEL:-ai-hub-models/Qwen3-VL-4B-Instruct}"
PRECISION="${PRECISION:-w4a16}"
APP_DIR="${APP_DIR:-$HOME/vlm-qa}"

log() { printf '\n=== %s\n' "$*"; }

log "Checking this is a supported chipset"
soc=$(cat /sys/devices/soc0/machine 2>/dev/null || echo unknown)
echo "SoC: $soc"
case "$soc" in
  QCS9075|QCS8275) ;;
  *) echo "WARNING: GenieX validates QCS9075 / QCS8275. Continuing anyway." >&2 ;;
esac

log "Installing base packages"
sudo apt-get update -qq
# ffmpeg does all frame extraction and tiling, so no OpenCV/PyAV wheels needed.
sudo apt-get install -y \
  libatomic1 libglib2.0-0 ocl-icd-libopencl1 \
  ffmpeg python3-venv python3-pip

log "Installing Qualcomm NPU driver packages"
# From the ubuntu-qcom-iot PPA, pre-configured on the IQ-9075 EVK image.
# qcom-adreno1 pulls in qcom-libdmabufheap and qcom-property-vault;
# libqnn1 brings the QAIRT runtime the qairt backend loads.
sudo apt-get install -y qcom-adreno1 qcom-fastrpc1 libqnn1

log "Installing the GenieX CLI"
if ! command -v geniex >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/geniex" ]; then
  curl -fsSL https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/install.sh | sh
else
  echo "geniex already installed, skipping"
fi
export PATH="$HOME/.local/bin:$PATH"
grep -qs '.local/bin' "$HOME/.bashrc" || \
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"

# Every geniex call gets </dev/null. When this script is piped in via
# `bash -s < setup-iq9.sh`, stdin IS the script -- a subcommand that prompts
# (geniex pull asks for a precision when one isn't pinned) would otherwise eat
# the remaining lines of this file.
geniex --version </dev/null
echo "Detected chipset: $(geniex config get chipset </dev/null)"

log "Pulling $MODEL:$PRECISION (~4.1 GiB, one time)"
cached=$(geniex list </dev/null 2>/dev/null || true)
if printf '%s' "$cached" | grep -qi 'Qwen3-VL-4B'; then
  echo "Model already cached, skipping"
else
  geniex pull "$MODEL:$PRECISION" --model-type vlm </dev/null
fi
geniex list </dev/null

log "Setting up the Python app"
cd "$APP_DIR"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip -q install -r requirements.txt

log "Starting the GenieX server if it isn't already running"
if curl -sf -m 5 http://127.0.0.1:18181/v1/models >/dev/null 2>&1; then
  echo "Server already up"
else
  nohup geniex serve > "$HOME/serve.log" 2>&1 </dev/null &
  for _ in $(seq 1 30); do
    sleep 2
    curl -sf -m 5 http://127.0.0.1:18181/v1/models >/dev/null 2>&1 && break
  done
fi

log "Done"
.venv/bin/python -m vlmqa status
cat <<EOF

Try it:
  cd $APP_DIR
  .venv/bin/python -m vlmqa ask -m <image-or-video> -q "your question"
EOF
