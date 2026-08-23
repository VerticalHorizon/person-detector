# Optimization notes for the Pi Zero 2 W

Two parts: what was already changed **in the code**, and recommended
**external changes** (OS, firmware, systemd) that setup.sh deliberately
does not apply for you.

## Code changes (already applied)

The Zero 2 W has 4× Cortex-A53 @ 1 GHz and 512 MB RAM shared with the
GPU/ISP; memory bandwidth is the scarcest resource. The main loop was
restructured around that:

- **Detection runs on the camera's `lores` stream.** The ISP scales
  every frame to 320×320 YUV420 in hardware, for free. Previously each
  iteration copied the full 1280×720 frame out of the camera, rotated
  it (≈2.7 MB read + write) and resized it to 300×300 in software —
  all of that is gone.
- **The full-resolution frame is only touched when a person is in
  view.** `capture_request()` is held during inference; the main-stream
  buffer is materialized (`make_array("main")`) only when a detection
  needs a crop. In the idle state (the vast majority of frames) the
  loop moves ~15 KB per frame instead of ~5.5 MB.
- **Rotation is applied only to small images**: the 320×320 detection
  frame and the final saved crops. Boxes are mapped back to unrotated
  camera coordinates analytically (`unrotate_box`).
- **Sharpness metric uses `CV_16S` instead of `CV_64F`** for the
  Laplacian — same ranking, no double-precision pass over the crop.
- Saved crops now go to `captures/` instead of cluttering the code dir.
- Legacy `app.py` (old `cv2.VideoCapture` standalone) removed; the
  `common.py` + backend structure supersedes it.
- `setup.sh` installs with `--no-install-recommends` and drops unused
  packages (`git`, `python3-smbus`) — less SD space, faster install.

Expected effect: ~10–20 % higher FPS (inference still dominates), much
lower memory traffic and steadier frame pacing. Inference itself is
already near-optimal: quantized MobileNet SSD via LiteRT/XNNPACK on
4 threads.

## Recommended external changes

Roughly in order of value. None are applied by `setup.sh`.

### 1. Hold the CPU at max clock (biggest easy win)

The default `ondemand` governor ramps up per-frame and hurts steady
inference latency:

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

### 2. `/boot/firmware/config.txt` for headless operation

    # libcamera allocates from CMA, not gpu_mem - keep the GPU slice minimal
    gpu_mem=16

    # frees the UART, saves a little power
    dtoverlay=disable-bt

    # headless: no HDMI, no audio
    dtparam=audio=off

Reboot after editing. Verify camera still works: `rpicam-hello --list-cameras`.

### 3. Keep the journal in RAM (SD longevity)

Trixie already ships zram swap and a tmpfs `/tmp`. Add:

    sudo mkdir -p /etc/systemd/journald.conf.d
    printf '[Journal]\nStorage=volatile\nRuntimeMaxUse=32M\n' | \
      sudo tee /etc/systemd/journald.conf.d/volatile.conf
    sudo systemctl restart systemd-journald

Note: logs are lost on reboot — fine for this appliance, the crops in
`captures/` are the real record.

### 4. Drop unused services

    # keep avahi if you rely on the .local hostname
    sudo systemctl disable --now triggerhappy ModemManager

### 5. WiFi power save off

Prevents multi-second SSH/scp stalls (the detector itself doesn't care):

    printf '[connection]\nwifi.powersave = 2\n' | \
      sudo tee /etc/NetworkManager/conf.d/wifi-powersave.conf
    sudo systemctl restart NetworkManager

Costs ~0.2 W. Skip on battery builds.

### 6. Hardware watchdog

An unattended appliance should recover from hard hangs (e.g. brownout
lockups seen when overclocking):

    # /etc/systemd/system.conf.d/watchdog.conf
    sudo mkdir -p /etc/systemd/system.conf.d
    printf '[Manager]\nRuntimeWatchdogSec=15\nRebootWatchdogSec=2min\n' | \
      sudo tee /etc/systemd/system.conf.d/watchdog.conf
    sudo systemctl daemon-reexec

The systemd `Restart=always` in the service already handles ordinary
crashes; the watchdog covers kernel-level hangs.

### 7. Memory guard on the service

512 MB total; if the process ever balloons, better a clean restart than
a system-wide OOM crawl. Add to `[Service]` in
`/etc/systemd/system/person-detector.service`:

    MemoryMax=300M

then `sudo systemctl daemon-reload && sudo systemctl restart person-detector`.

### 8. Overclocking (optional, last)

Only after 24 h of stable stock operation on a solid 5 V / 2.5 A supply,
one step at a time, checking `vcgencmd get_throttled` stays `0x0`.
Start with `arm_freq=1100`, `over_voltage=2`.
Field experience on this build: 1200 MHz / `over_voltage=6` caused
crashes and WiFi loss on a weak supply.

### 9. Things considered and *not* recommended

- **Read-only rootfs / overlayfs**: maximal SD protection, but the
  detector must write `captures/` — you'd need a separate writable
  partition. Only worth it if SD corruption actually bites.
- **Bigger/newer detection models** (EfficientDet-Lite0, YOLO-nano):
  2–5× slower on the A53 for modest accuracy gain. The quantized
  MobileNet SSD v1 is the sweet spot for this CPU; spend accuracy
  effort on the re-ID stage (roadmap) on a stronger machine instead.
- **Higher `CAMERA_SIZE`**: detection quality is set by the 320×320
  lores stream, not the main stream; raising the main size only makes
  crops larger and slows the person-in-frame path. 1280×720 is a good
  balance. Conversely, dropping to 640×480 speeds up episodes slightly
  at the cost of crop detail for future re-ID — not worth it.
