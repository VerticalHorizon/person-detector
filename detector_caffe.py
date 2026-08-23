"""Person detector, OpenCV DNN backend (MobileNet SSD, Caffe, VOC).

No extra dependencies: uses ``python3-opencv`` from apt.  Model files:
``*.prototxt`` and ``*.caffemodel`` next to this script.
Person = class 15, model expects BGR input.
"""

import glob
import os

import cv2

import common


def _find(pattern):
    files = sorted(glob.glob(os.path.join(common.SCRIPT_DIR, pattern)))
    if not files:
        raise FileNotFoundError(pattern)
    return files[0]


net = cv2.dnn.readNetFromCaffe(_find("*.prototxt"), _find("*.caffemodel"))


def detect(small):
    """Small BGR frame -> ([(score, norm box)], best raw person score)."""
    net.setInput(cv2.dnn.blobFromImage(small, 0.007843, (300, 300), 127.5))
    detections = net.forward()[0, 0]   # rows: id, class, score, x1, y1, x2, y2

    persons, best_raw = [], 0.0
    for _, cls, score, xmin, ymin, xmax, ymax in detections:
        if int(cls) != 15:                    # 15 = person in VOC
            continue
        best_raw = max(best_raw, float(score))
        if score < common.CONF_THRES:
            continue
        persons.append((float(score), (float(xmin), float(ymin),
                                       float(xmax), float(ymax))))
    return persons, best_raw


if __name__ == "__main__":
    common.run(detect, "OpenCV-DNN Caffe")
