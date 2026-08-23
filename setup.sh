#!/usr/bin/env bash
# Person detector setup for Raspberry Pi Zero 2W (Raspberry Pi OS Lite, trixie).
# Installs packages, a venv with LiteRT, the detection model and the systemd
# service.  Run from the project folder:  sudo bash setup.sh
# System tuning tips (SD longevity, governor, overclock) live in README.md.

set -euo pipefail   # strict mode: stop on errors, unset vars, failed pipes

[ "$(id -u)" -eq 0 ] || { echo "Run with sudo"; exit 1; }

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_NAME="${SUDO_USER:-pi}"
MODEL_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"

echo "[1/4] Packages"
apt-get update
apt-get install -y --no-install-recommends python3-picamera2
apt-get install -y --no-install-recommends \
  python3-opencv python3-venv python3-gpiozero wget unzip

echo "[2/4] Virtualenv + LiteRT"
sudo -u "$USER_NAME" python3 -m venv --system-site-packages "$DIR/venv"
sudo -u "$USER_NAME" "$DIR/venv/bin/pip" install --upgrade pip ai-edge-litert

echo "[3/4] Model"
if [ ! -f "$DIR/detect.tflite" ]; then
  wget -q -O /tmp/model.zip "$MODEL_URL"
  unzip -o /tmp/model.zip -d "$DIR" detect.tflite labelmap.txt
  rm -f /tmp/model.zip
fi

echo "[4/4] Systemd service"
cat > /etc/systemd/system/person-detector.service <<EOF
[Unit]
Description=Person detector (Camera Module 3 + buzzer)
After=local-fs.target

[Service]
User=$USER_NAME
SupplementaryGroups=video gpio
WorkingDirectory=$DIR
ExecStart=$DIR/venv/bin/python3 $DIR/detector_litert.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable person-detector.service
chown -R "$USER_NAME":"$USER_NAME" "$DIR"

echo "Done. Reboot, then: journalctl -u person-detector -f"
