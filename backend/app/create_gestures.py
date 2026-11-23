import os
import random
import sqlite3
import cv2
import numpy as np
import mediapipe as mp

from paths import DATABASE_PATH, GESTURES_DIR, ensure_directories

# Parameters
image_x, image_y = 50, 50


def init_create_folder_database():
    ensure_directories()
    if not DATABASE_PATH.exists():
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute("CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )")
        conn.commit()
        conn.close()


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)


def store_in_db(g_id, g_name):
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        conn.execute("INSERT INTO gesture (g_id, g_name) VALUES (?, ?)", (g_id, g_name))
    except sqlite3.IntegrityError:
        choice = input(f"Gesture ID {g_id} already exists. Update name to '{g_name}'? (y/n): ")
        if choice.lower() == "y":
            conn.execute("UPDATE gesture SET g_name = ? WHERE g_id = ?", (g_name, g_id))
    conn.commit()
    conn.close()


def get_hand_mask(img, hands_module):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # CPU optimization: writeable=False avoids memory copies in MediaPipe
    img_rgb.flags.writeable = False
    results = hands_module.process(img_rgb)
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

        for connection in mp.solutions.hands.HAND_CONNECTIONS:
            p1 = (
                int(hand_landmarks.landmark[connection[0]].x * img.shape[1]),
                int(hand_landmarks.landmark[connection[0]].y * img.shape[0]),
            )
            p2 = (
                int(hand_landmarks.landmark[connection[1]].x * img.shape[1]),
                int(hand_landmarks.landmark[connection[1]].y * img.shape[0]),
            )
            cv2.line(mask, p1, p2, 255, 15)

    return mask, num_hands


def store_images(g_id):
    # MediaPipe Hands with model_complexity=0 (Lite) for maximum CPU efficiency
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Supports 1 or 2 hands dynamically
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("DEBUG: Searching for camera...")
    cam = None
    for index in [0, 1, 2]:
        temp_cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if temp_cam.isOpened():
            temp_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            temp_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            temp_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, frame = temp_cam.read()
            if ret:
                cam = temp_cam
                print(f"SUCCESS: Using Camera {index}")
                break
        temp_cam.release()

    if cam is None:
        print("CRITICAL ERROR: Could not open any camera.")
        return

    total_pics = 300
    gesture_folder = GESTURES_DIR / str(g_id)
    create_folder(gesture_folder)
    pic_no = 0
    flag_capture = False

    print("\n--- OPERATING INSTRUCTIONS ---")
    print("1. A window will open showing your webcam.")
    print("2. Position 1 hand OR 2 hands inside the green box.")
    print("3. Press 'c' to START/PAUSE capture.")
    print("4. Press 'q' to QUIT early.")
    print("------------------------------\n")

    x, y, w, h = 60, 40, 520, 400  # Widened ROI to comfortably fit two hands or one hand

    while True:
        ret, img = cam.read()
        if not ret:
            break
        if img.shape[0] != 480 or img.shape[1] != 640:
            img = cv2.resize(img, (640, 480))
        img = cv2.flip(img, 1)

        mask, num_hands = get_hand_mask(img, hands_detector)
        thresh = mask[y : y + h, x : x + w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

        valid_contours = [c for c in contours if cv2.contourArea(c) > 500]

        if flag_capture and len(valid_contours) > 0 and num_hands > 0:
            # Combines 1 hand or 2 hands into one unified bounding box
            all_points = np.vstack(valid_contours)
            x1, y1, w1, h1 = cv2.boundingRect(all_points)
            save_img = thresh[y1 : y1 + h1, x1 : x1 + w1]

            if save_img.size > 0:
                if w1 > h1:
                    pad = int((w1 - h1) / 2)
                    save_img = cv2.copyMakeBorder(save_img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
                elif h1 > w1:
                    pad = int((h1 - w1) / 2)
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)

                save_img = cv2.resize(save_img, (image_x, image_y))

                # Slight random flip augmentation
                if random.randint(0, 10) % 2 == 0:
                    save_img = cv2.flip(save_img, 1)

                pic_no += 1
                cv2.imwrite(str(gesture_folder / f"{pic_no}.jpg"), save_img)
                if pic_no % 50 == 0:
                    print(f"PROGRESS: {pic_no}/{total_pics} images saved.")

        # Annotate view
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"Captured: {pic_no}/{total_pics}", (30, 450), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (127, 127, 255), 2)
        hand_text = f"Hands: {num_hands}"
        cv2.putText(img, hand_text, (30, 410), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 255, 0) if num_hands > 0 else (0, 0, 255), 2)

        if flag_capture:
            cv2.putText(img, "CAPTURING...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Press 'c' to start capture", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (255, 255, 0), 2)

        cv2.imshow("Webcam Feed", img)
        cv2.imshow("Hand Mask (White on Black)", mask)

        kp = cv2.waitKey(1) & 0xFF
        if kp == ord('c'):
            flag_capture = not flag_capture
            print("Capture Status: " + ("ON" if flag_capture else "OFF"))
        if kp == ord('q') or pic_no >= total_pics:
            break

    cam.release()
    cv2.destroyAllWindows()
    if pic_no >= total_pics:
        print(f"\nSUCCESS: Gesture {g_id} captured completely ({pic_no} images)!")


if __name__ == "__main__":
    init_create_folder_database()
    print("=== SIGN LANGUAGE DATA CAPTURE (1 OR 2 HANDS) ===")
    try:
        g_id_in = int(input("Step 1: Enter Gesture ID (e.g. 1): "))
        g_name_in = input("Step 2: Enter Gesture Name (e.g. A): ")
        store_in_db(g_id_in, g_name_in)
        store_images(g_id_in)
    except Exception as e:
        print(f"Error: {e}")
