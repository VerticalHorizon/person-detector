# Person Tracker

Person detection on a Raspberry Pi Zero 2W with Camera Module 3.
When a person enters the frame the buzzer beeps and the best crop of the
episode is saved next to the scripts as `person_YYYYmmdd_HHMMSS.jpg`.
One episode = one appearance; the crops form a gallery for future re-ID.

## Hardware

- Raspberry Pi Zero 2W, Raspberry Pi OS Lite 64-bit (trixie)
- Camera Module 3 on the ribbon cable
- Active 3.3 V buzzer: "+" to GPIO17 (physical pin 11), "-" to GND (pin 9).
  Passive or high-power buzzers need an NPN transistor stage.

## Install

    cd ~/person_tracker
    sudo bash setup.sh
    sudo reboot
    journalctl -u person-detector -f

## Layout

- `common.py` - settings, camera, episode loop, crop saving
- `detector_litert.py` - LiteRT backend (quantized SSD, COCO), the default
- `detector_caffe.py` - OpenCV DNN backend (Caffe SSD, VOC), zero pip deps
- `setup.sh` - packages, venv, model download, systemd service

Both backends expose an identical `detect()` and share `common.run()`,
so their behaviour can be compared 1:1 by simply running either file.
To switch the service to the Caffe backend, change `ExecStart` in
`/etc/systemd/system/person-detector.service`, then:
`sudo systemctl daemon-reload && sudo systemctl restart person-detector`

## Tuning

All knobs are constants at the top of `common.py`: confidence threshold,
camera size, rotation, episode timeout, stats interval. The frame is
rotated 90 degrees counterclockwise right after capture (1280x720 becomes
720x1280); all boxes and crops use the rotated frame. The stats line
prints the best raw person score per interval - if it hovers around
0.3-0.45, lower `CONF_THRES` or add light.

## Recommended system tweaks (optional, not applied by setup.sh)

SD longevity: trixie already ships zram swap (`rpi-swap`) and a tmpfs
`/tmp`. Two extras worth doing:

    # keep the systemd journal in RAM
    sudo mkdir -p /etc/systemd/journald.conf.d
    printf '[Journal]\nStorage=volatile\nRuntimeMaxUse=32M\n' | \
      sudo tee /etc/systemd/journald.conf.d/volatile.conf
    sudo systemctl restart systemd-journald

    # drop unused services (keep avahi if you rely on .local hostnames)
    sudo systemctl disable --now triggerhappy ModemManager

CPU governor - hold the max clock for steady inference:

    sudo tee /etc/systemd/system/cpu-performance.service <<'EOF'
    [Unit]
    Description=CPU governor performance
    [Service]
    Type=oneshot
    ExecStart=/bin/sh -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
    RemainAfterExit=yes
    [Install]
    WantedBy=multi-user.target
    EOF
    sudo systemctl daemon-reload && sudo systemctl enable --now cpu-performance.service

Bluetooth off (saves power, frees the UART): add `dtoverlay=disable-bt`
to `/boot/firmware/config.txt`.

Overclocking - only after 24 h of stable stock operation on a solid
5 V / 2.5 A supply, one step at a time, checking `vcgencmd get_throttled`
stays `0x0`. Start with `arm_freq=1100`, `over_voltage=2`.
Field experience on this build: 1200 MHz / over_voltage=6 caused crashes
and WiFi loss on a weak supply.

## Troubleshooting

- Camera: `rpicam-hello --list-cameras`
- Power: `vcgencmd get_throttled` (`0x0` = OK)
- Logs: `journalctl -u person-detector -f`

## Roadmap

Feed the saved crops to a light embedding model (e.g. OSNet-x0.25) and
match by cosine similarity to tell people apart and count appearances.
Best done offline or on a stronger machine; the Zero 2W is busy detecting.
