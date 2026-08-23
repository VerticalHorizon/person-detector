"""Shared core for the person detector: settings, camera, episode loop.

A detector backend supplies a ``detect(small) -> (persons, best_raw)``
callable and passes it to ``run()``.  ``small`` is a low-resolution BGR
frame (already rotated upright), ``persons`` is a list of
``(score, (xmin, ymin, xmax, ymax))`` normalized boxes in that rotated
space, ``best_raw`` is the best raw person score seen in the frame
(diagnostics only).

Pi Zero 2 W notes: detection runs on the camera ISP's hardware-scaled
``lores`` stream, so the full-resolution frame is never copied, resized
or rotated in software unless a person is actually in view.  Only small
images and final crops are rotated.
"""

import os
import signal
import sys
import time
from datetime import datetime
from math import log1p

import cv2
from gpiozero import Buzzer
from picamera2 import Picamera2

# --------------- settings ---------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR     = os.path.join(SCRIPT_DIR, "captures")

BUZZER_GPIO  = 17                 # BCM numbering (physical pin 11)
CONF_THRES   = 0.5                # detection confidence threshold
CAMERA_SIZE  = (1280, 720)        # main stream (crop source), before rotation
LORES_SIZE   = (320, 320)         # ISP-scaled detection stream (YUV420)
ROTATION     = cv2.ROTATE_90_COUNTERCLOCKWISE   # None to disable
ABSENCE_SEC  = 3.0                # seconds without a person = episode end
MIN_BOX_SIDE = 40                 # ignore boxes smaller than this (px)
JPEG_QUALITY = 90
STATS_EVERY  = 20                 # log FPS/temp/score every N frames
# ----------------------------------------

# main-frame shape (h, w) as delivered by the camera, before rotation
MAIN_SHAPE = (CAMERA_SIZE[1], CAMERA_SIZE[0])


def log(msg):
    print(msg, flush=True)        # journald adds timestamps


def unrotate_box(xmin, ymin, xmax, ymax):
    """Normalized box in rotated space -> normalized box in camera space."""
    if ROTATION == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return 1.0 - ymax, xmin, 1.0 - ymin, xmax
    if ROTATION == cv2.ROTATE_90_CLOCKWISE:
        return ymin, 1.0 - xmax, ymax, 1.0 - xmin
    if ROTATION == cv2.ROTATE_180:
        return 1.0 - xmax, 1.0 - ymax, 1.0 - xmin, 1.0 - ymin
    return xmin, ymin, xmax, ymax


def clip_box(shape, xmin, ymin, xmax, ymax):
    """Normalized coords -> pixel box, or None if the box is too small."""
    h, w = shape[:2]
    x1, y1 = max(0, int(xmin * w)), max(0, int(ymin * h))
    x2, y2 = min(w, int(xmax * w)), min(h, int(ymax * h))
    if x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE:
        return None
    return x1, y1, x2, y2


def crop_quality(frame, box, score):
    """Rank crops: confidence x (1 + area) x log(sharpness)."""
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_16S).var()
    area = (x2 - x1) * (y2 - y1) / (frame.shape[0] * frame.shape[1])
    return score * (1.0 + area) * log1p(sharpness), crop


def cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except OSError:
        return -1.0


def start_camera():
    """Picamera2 note: format "RGB888" yields BGR arrays - OpenCV order.

    The lores stream is scaled by the ISP for free; on the Zero 2 W it
    must be YUV420.
    """
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={"size": CAMERA_SIZE, "format": "RGB888"},
        lores={"size": LORES_SIZE, "format": "YUV420"},
        buffer_count=3))
    picam2.start()
    time.sleep(1.5)               # let auto-exposure settle
    return picam2


def run(detect, engine):
    """Main loop: episodes, buzzer signal, best-crop saving."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    buzzer = Buzzer(BUZZER_GPIO)
    picam2 = start_camera()

    episode, last_seen, started = False, 0.0, None
    best_q, best_crop = 0.0, None
    episodes, interval_raw = 0, 0.0
    frames, t_stat = 0, time.monotonic()

    def save_best():
        """Write the best crop of the current episode, if any."""
        nonlocal best_q, best_crop
        if best_crop is None:
            return
        name = f"person_{started:%Y%m%d_%H%M%S}.jpg"
        path = os.path.join(SAVE_DIR, name)
        if cv2.imwrite(path, best_crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            log(f"Saved {name}")
        else:
            log(f"ERROR: failed to write {path}")
        best_q, best_crop = 0.0, None

    def shutdown(*_):
        log("Stopping...")
        save_best()               # do not lose an unfinished episode
        picam2.stop()
        buzzer.off()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    buzzer.beep(0.08, 0.08, n=2, background=False)   # startup self-test
    log(f"Engine: {engine}, threshold {CONF_THRES}")

    while True:
        # Hold the request through inference so the main frame stays
        # available (and in sync with lores) if a crop is needed.
        request = picam2.capture_request()
        try:
            small = cv2.cvtColor(request.make_array("lores"),
                                 cv2.COLOR_YUV420p2BGR)
            if ROTATION is not None:
                small = cv2.rotate(small, ROTATION)
            raw_persons, raw = detect(small)

            persons = []
            for score, nbox in raw_persons:
                box = clip_box(MAIN_SHAPE, *unrotate_box(*nbox))
                if box:
                    persons.append((score, box))
            frame = request.make_array("main") if persons else None
        finally:
            request.release()

        interval_raw = max(interval_raw, raw)
        now = time.monotonic()

        if persons:
            last_seen = now
            if not episode:
                episode, started = True, datetime.now()
                best_q, best_crop = 0.0, None
                buzzer.beep(0.15, 0.1, n=3, background=True)
                log(f">> Person in frame (count {len(persons)}, "
                    f"score {persons[0][0]:.2f})")
            for score, box in persons:
                q, crop = crop_quality(frame, box, score)
                if q > best_q:
                    best_q = q
                    best_crop = (cv2.rotate(crop, ROTATION)
                                 if ROTATION is not None else crop.copy())
        elif episode and now - last_seen > ABSENCE_SEC:
            episode = False
            episodes += 1
            log(f"<< Episode #{episodes} ended")
            save_best()

        frames += 1
        if frames % STATS_EVERY == 0:
            dt = time.monotonic() - t_stat
            log(f"FPS ~{STATS_EVERY / dt:.2f}, t={cpu_temp():.1f}C, "
                f"max raw person score: {interval_raw:.2f}")
            t_stat, interval_raw = time.monotonic(), 0.0
