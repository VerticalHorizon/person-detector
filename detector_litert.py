"""Person detector, LiteRT backend (quantized MobileNet SSD, COCO).

LiteRT (pip package ``ai-edge-litert``) is Google's successor to the
frozen tflite-runtime.  Model file: ``detect.tflite`` next to this
script.  Person = class 0, model expects RGB input.
"""

import os

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

import common

MODEL = os.path.join(common.SCRIPT_DIR, "detect.tflite")

interpreter = Interpreter(model_path=MODEL, num_threads=4)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()
in_h, in_w = int(inp["shape"][1]), int(inp["shape"][2])


def detect(frame):
    """BGR frame -> ([(score, box)], best raw person score)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    interpreter.set_tensor(inp["index"],
                           np.expand_dims(cv2.resize(rgb, (in_w, in_h)), 0))
    interpreter.invoke()
    boxes   = interpreter.get_tensor(out[0]["index"])[0]
    classes = interpreter.get_tensor(out[1]["index"])[0]
    scores  = interpreter.get_tensor(out[2]["index"])[0]

    persons, best_raw = [], 0.0
    for (ymin, xmin, ymax, xmax), cls, score in zip(boxes, classes, scores):
        if int(cls) != 0:                     # 0 = person in COCO
            continue
        best_raw = max(best_raw, float(score))
        if score < common.CONF_THRES:
            continue
        box = common.clip_box(frame.shape, xmin, ymin, xmax, ymax)
        if box:
            persons.append((float(score), box))
    return persons, best_raw


if __name__ == "__main__":
    common.run(detect, "LiteRT")
