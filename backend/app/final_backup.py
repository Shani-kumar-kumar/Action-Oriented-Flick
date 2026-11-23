import os
import sqlite3
import subprocess
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

from paths import DATABASE_PATH, MODEL_PATH, ensure_directories

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MediaPipe Setup - Lite model optimized for low-end CPU, supports 1 or 2 hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Support 1 or 2 hands dynamically
    model_complexity=0,  # 0 is Lite model - fast on low-end CPU
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Load the trained model
ensure_directories()
if MODEL_PATH.exists():
    model = load_model(str(MODEL_PATH))
    print("SUCCESS: Model loaded.")
else:
    print(f"ERROR: Model '{MODEL_PATH}' not found.")
    exit()

image_x, image_y = 50, 50
x, y, w, h = 60, 40, 520, 400
is_voice_on = True


def get_hand_mask(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb.flags.writeable = False
    results = hands.process(img_rgb)
    img_rgb.flags.writeable = True

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    if not results.multi_hand_landmarks:
        return mask, 0

    num_hands = len(results.multi_hand_landmarks)
    for hand_landmarks in results.multi_hand_landmarks:
        points = np.array(
            [[int(lm.x * img.shape[1]), int(lm.y * img.shape[0])] for lm in hand_landmarks.landmark],
            dtype=np.int32,
        )
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection[0], connection[1]
            p1 = (
                int(hand_landmarks.landmark[start_idx].x * img.shape[1]),
                int(hand_landmarks.landmark[start_idx].y * img.shape[0]),
            )
            p2 = (
                int(hand_landmarks.landmark[end_idx].x * img.shape[1]),
                int(hand_landmarks.landmark[end_idx].y * img.shape[0]),
            )
            cv2.line(mask, p1, p2, 255, 15)

    return mask, num_hands


def keras_process_image(img):
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32) / 255.0
    return np.reshape(img, (1, image_x, image_y, 1))


def keras_predict(model_instance, image):
    processed = keras_process_image(image)
    # Direct tensor call for fast inference on low-end CPU
    pred_probab = model_instance(processed, training=False).numpy()[0]
    pred_class = int(np.argmax(pred_probab))
    return float(np.max(pred_probab)), pred_class


def get_pred_text_from_db(pred_class):
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.execute("SELECT g_name FROM gesture WHERE g_id=?", (int(pred_class),))
            row = cursor.fetchone()
            if row:
                return row[0]
    except Exception:
        pass
    return "Unknown"


def predict_from_points(points, thresh):
    x1, y1, w1, h1 = cv2.boundingRect(points)
    save_img = thresh[y1 : y1 + h1, x1 : x1 + w1]
    if save_img.size == 0:
        return 0.0, 0

    if w1 > h1:
        pad = int((w1 - h1) / 2)
        save_img = cv2.copyMakeBorder(save_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    elif h1 > w1:
        pad = int((h1 - w1) / 2)
        save_img = cv2.copyMakeBorder(save_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)

    return keras_predict(model, save_img)


def say_text(text):
    if not is_voice_on or not text.strip():
        return

    print(f"[SPEECH] -> {text}")
    try:
        safe_text = text.replace("'", "").replace('"', "")
        cmd = (
            f"Add-Type -AssemblyName System.Speech; "
            f"$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speak.Speak('{safe_text}')"
        )
        subprocess.Popen(["powershell", "-Command", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"SPEECH ERROR: {e}")


def get_img_contour_thresh(img):
    if img.shape[0] != 480 or img.shape[1] != 640:
        img = cv2.resize(img, (640, 480))
    img = cv2.flip(img, 1)

    mask, num_hands = get_hand_mask(img)
    thresh = mask[y : y + h, x : x + w]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    return img, contours, thresh, num_hands


def text_mode(cam):
    text = ""
    word = ""
    count_same_frame = 0
    last_added_text = ""
    current_conf = 0.0

    say_text("System ready")

    while True:
        ret, img = cam.read()
        if not ret:
            break

        img, contours, thresh, num_hands = get_img_contour_thresh(img)
        old_text = text

        valid_contours = [c for c in contours if cv2.contourArea(c) > 500]

        if len(valid_contours) > 0 and num_hands > 0:
            all_points = np.vstack(valid_contours)
            best_prob, best_class = predict_from_points(all_points, thresh)

            if num_hands > 1 and best_prob * 100 < 70:
                top_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:2]
                for c in top_contours:
                    prob, cls_id = predict_from_points(c, thresh)
                    if prob > best_prob:
                        best_prob, best_class = prob, cls_id

            current_conf = best_prob * 100
            if current_conf > 68:
                cand_text = get_pred_text_from_db(best_class)
                text = cand_text if cand_text != "Unknown" else ""
            else:
                text = ""

            if text == "":
                last_added_text = ""
                count_same_frame = 0
            else:
                if old_text == text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0

                if count_same_frame > 15:
                    if len(text) > 1:
                        if text != last_added_text:
                            if len(word) > 0 and word[-1] != " ":
                                word += " "
                            word += text + " "
                            say_text(text)
                            last_added_text = text
                    else:
                        if text != last_added_text:
                            word += text
                            last_added_text = text

                    count_same_frame = 0
        else:
            text = ""
            last_added_text = ""
            count_same_frame = 0
            current_conf = 0.0

        blackboard = np.zeros((480, 600, 3), dtype=np.uint8)
        cv2.putText(blackboard, "MediaPipe Mode", (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 0, 0))
        cv2.putText(blackboard, f"Hands: {num_hands}", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 255, 0) if num_hands > 0 else (0, 0, 255))
        pred_label = f"{text} ({current_conf:.1f}%)" if text else "Waiting..."
        cv2.putText(blackboard, "Predicted: " + pred_label, (50, 150), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (255, 255, 0))
        cv2.putText(blackboard, word, (50, 250), cv2.FONT_HERSHEY_TRIPLEX, 1.1, (255, 255, 255))

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"Hands: {num_hands}", (x + 10, y + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)

        res = np.hstack((img, blackboard))
        cv2.imshow("Recognizing gesture", res)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            word = ""
            last_added_text = ""
        elif key == ord('s'):
            say_text(word)
        elif key == ord(' '):
            word += " "

    return 0


def recognize():
    cam = None
    for index in [0, 1, 2]:
        print(f"DEBUG: Trying camera index {index}...")
        temp_cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if temp_cam.isOpened():
            temp_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            temp_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            temp_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = temp_cam.read()
            if ret:
                cam = temp_cam
                print(f"SUCCESS: Using Camera Index {index}")
                break
        temp_cam.release()

    if cam is None:
        print("CRITICAL ERROR: Could not open any camera.")
        return

    text_mode(cam)
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize()
