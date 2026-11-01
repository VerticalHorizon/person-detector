# Person Detector

Person detection on a Raspberry Pi Zero 2W with Camera Module 3.
When a person enters the frame the buzzer beeps and the best crop of the
episode is saved to `captures/` as `person_YYYYmmdd_HHMMSS.jpg`.
One episode = one appearance; the crops form a gallery for future re-ID.

## Hardware

- Raspberry Pi Zero 2W, Raspberry Pi OS Lite 64-bit (trixie)
- Camera Module 3 on the ribbon cable
- Active 3.3 V buzzer: "+" to GPIO17 (physical pin 11), "-" to GND (pin 9).
  Passive or high-power buzzers need an NPN transistor stage.

## Install

    cd ~/person_detector
    sudo bash setup.sh
    sudo reboot
    journalctl -u person-detector -f

After installing, apply the system tweaks from `OPTIMIZATIONS.md`
(CPU governor, config.txt, volatile journal, watchdog) — they are
deliberately not applied by setup.sh.

## Layout

- `common.py` - settings, camera, episode loop, crop saving
- `detector_litert.py` - LiteRT backend (quantized SSD, COCO), the default
- `detector_caffe.py` - OpenCV DNN backend (Caffe SSD, VOC), zero pip deps
- `setup.sh` - packages, venv, model download, systemd service
- `OPTIMIZATIONS.md` - what was optimized and recommended OS/firmware tweaks
- `captures/` - saved crops (created on first run)

Both backends expose an identical `detect()` and share `common.run()`,
so their behaviour can be compared 1:1 by simply running either file.
To switch the service to the Caffe backend, change `ExecStart` in
`/etc/systemd/system/person-detector.service`, then:
`sudo systemctl daemon-reload && sudo systemctl restart person-detector`

## How the loop stays fast on a Zero 2W

Detection runs on the camera ISP's hardware-scaled 320x320 `lores`
stream; the full 1280x720 frame is only read out of the camera when a
person is actually detected, to take the crop. Rotation is applied to
the small detection frame and to final crops only — never to the full
frame. Details and benchmark reasoning: `OPTIMIZATIONS.md`.

## Tuning

All knobs are constants at the top of `common.py`: confidence threshold,
camera/lores size, rotation, episode timeout, stats interval. Detection
happens in rotated (upright) orientation; boxes are mapped back to
camera coordinates and saved crops are rotated upright. The stats line
prints the best raw person score per interval - if it hovers around
0.3-0.45, lower `CONF_THRES` or add light.

## Troubleshooting

- Camera: `rpicam-hello --list-cameras`
- Power: `vcgencmd get_throttled` (`0x0` = OK)
- Logs: `journalctl -u person-detector -f`

## Roadmap

Feed the saved crops to a light embedding model (e.g. OSNet-x0.25) and
match by cosine similarity to tell people apart and count appearances.
Best done offline or on a stronger machine; the Zero 2W is busy detecting.
