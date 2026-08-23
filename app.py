import os
import time
import urllib.request
import cv2
import numpy as np
from gpiozero import Buzzer

# --- CONFIGURATION ---
BUZZER_PIN = 17  # GPIO pin connected to the active buzzer
CONFIDENCE_THRESHOLD = 0.5  # Detection threshold (50%)
MIN_BLUR_SCORE = 50.0  # Minimum variance of Laplacian for sharp image
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Model file paths
PROTOTXT_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_Deploy.prototxt")
MODEL_PATH = os.path.join(MODEL_DIR, "MobileNetSSD_Deploy.caffemodel")

# URLs to download MobileNet-SSD weights automatically if missing
PROTOTXT_URL = "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt"
MODEL_URL = "https://www.dropbox.com/s/32529ijyefid3ls/MobileNetSSD_deploy.caffemodel?dl=1"


def download_file(url, target_path):
    """Download missing model files."""
    if not os.path.exists(target_path):
        print(f"Downloading {os.path.basename(target_path)}...")
        urllib.request.urlretrieve(url, target_path)


def calculate_sharpness(image):
    """Calculate image sharpness using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    # 1. Download model files if necessary
    download_file(PROTOTXT_URL, PROTOTXT_PATH)
    download_file(MODEL_URL, MODEL_PATH)

    # 2. Initialize Hardware & Model
    buzzer = Buzzer(BUZZER_PIN)
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

    # 3. Initialize Camera (0 = default CSI or USB camera)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    # Class ID for 'person' in MobileNet-SSD Caffe model is 15
    PERSON_CLASS_ID = 15

    person_in_frame = False
    best_crop = None
    best_score = -1.0

    print("System ready. Monitoring for persons...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]

            # Prepare input blob for the neural network
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                0.007843,
                (300, 300),
                127.5
            )
            net.setInput(blob)
            detections = net.forward()

            current_person_detected = False
            highest_conf_crop = None
            highest_conf_score = 0.0

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                class_id = int(detections[0, 0, i, 1])

                if class_id == PERSON_CLASS_ID and confidence > CONFIDENCE_THRESHOLD:
                    current_person_detected = True

                    # Extract bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Ensure coordinates stay within frame dimensions
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w, endX), min(h, endY)

                    crop = frame[startY:endY, startX:endX]

                    if crop.size > 0 and confidence > highest_conf_score:
                        highest_conf_score = confidence
                        highest_conf_crop = crop

            if current_person_detected:
                buzzer.on()

                # Evaluate sharpness and quality score for best frame selection
                if highest_conf_crop is not None:
                    sharpness = calculate_sharpness(highest_conf_crop)
                    combined_score = highest_conf_score * 100 + sharpness

                    if combined_score > best_score:
                        best_score = combined_score
                        best_crop = highest_conf_crop.copy()

                person_in_frame = True

            else:
                buzzer.off()

                # Person has left the frame -> save the best recorded crop
                if person_in_frame and best_crop is not None:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(MODEL_DIR, f"person_{timestamp}.jpg")
                    cv2.imwrite(filename, best_crop)
                    print(f"Saved best person crop: {filename}")

                    # Reset tracking variables
                    person_in_frame = False
                    best_crop = None
                    best_score = -1.0

            time.sleep(0.03)  # Frame rate limiting (~30 FPS max check)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        buzzer.off()
        cap.release()


if __name__ == "__main__":
    main()
