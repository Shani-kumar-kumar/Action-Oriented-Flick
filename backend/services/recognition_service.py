import logging
import sqlite3
import threading
from collections import deque
from datetime import datetime, timezone

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

from backend.utils.paths import DATABASE_PATH, MODEL_PATH, ensure_directories

LOGGER = logging.getLogger(__name__)


class RecognitionError(RuntimeError):
    pass


class RecognitionService:
    def __init__(self):
        self.image_x = 50
        self.image_y = 50

        # Widened and centered ROI so both hands fit comfortably
        self.roi_x = 60
        self.roi_y = 40
        self.roi_w = 520
        self.roi_h = 400
        self.confidence_threshold = 68.0

        self._lock = threading.RLock()
        self._model = None
        self._model_mtime = None
        self._camera = None
        self._camera_index = None
        self._running = False
        self._source = None
        self._history = deque(maxlen=50)
        self._smooth_buffer = deque(maxlen=4)
        self._total_detected = 0
        self._total_confidence = 0.0
        self._session_started_at = None
        self._last_prediction = None

        # MediaPipe Hands initialized with model_complexity=0 (Lite) for low-end CPU optimization
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support 1 or 2 hands simultaneously
            model_complexity=0,  # 0 is Lite model - 2x faster, minimum CPU usage
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def health(self):
        model_exists = MODEL_PATH.exists()
        database_exists = DATABASE_PATH.exists()
        return {
            "status": "ready" if model_exists and database_exists else "degraded",
            "model_loaded": self._model is not None,
            "model_path": str(MODEL_PATH),
            "model_exists": model_exists,
            "database_exists": database_exists,
            "running": self._running,
            "source": self._source,
            "camera_index": self._camera_index,
        }

    def start(self, source="browser"):
        with self._lock:
            self._load_model()
            if self._running:
                return self._session_payload("already_running")

            source = self._normalize_source(source)
            if source == "server":
                self._camera = self._open_camera()
            else:
                self._release_camera()

            self._running = True
            self._source = source
            self._session_started_at = datetime.now(timezone.utc)
            self._smooth_buffer.clear()
            LOGGER.info("Recognition started in %s mode", source)
            return self._session_payload("started")

    def stop(self):
        with self._lock:
            self._release_camera()
            self._running = False
            self._source = None
            self._smooth_buffer.clear()
            LOGGER.info("Recognition stopped")
            return self._session_payload("stopped")

    def predict(self, frame=None):
        with self._lock:
            self._load_model()
            if frame is None:
                if not self._running or self._camera is None or self._source != "server":
                    raise RecognitionError("A browser image frame is required for prediction.")
                ok, frame = self._camera.read()
                if not ok or frame is None:
                    raise RecognitionError("Could not read a frame from the camera.")

            prediction = self._predict_frame(frame)
            self._last_prediction = prediction

            if prediction["detected"]:
                self._history.appendleft(prediction)
                self._total_detected += 1
                self._total_confidence += prediction["confidence"]

            return prediction

    def stats(self):
        with self._lock:
            average = round(self._total_confidence / self._total_detected, 2) if self._total_detected else 0
            return {
                "running": self._running,
                "source": self._source,
                "total_gestures": self._total_detected,
                "average_confidence": average,
                "session_started_at": self._session_started_at.isoformat() if self._session_started_at else None,
                "history_count": len(self._history),
                "last_prediction": self._last_prediction,
            }

    def history(self):
        with self._lock:
            return list(self._history)

    def clear(self):
        with self._lock:
            self._history.clear()
            self._smooth_buffer.clear()
            self._total_detected = 0
            self._total_confidence = 0.0
            self._last_prediction = None
            self._session_started_at = datetime.now(timezone.utc) if self._running else None
            return self.stats()

    def _load_model(self):
        ensure_directories()
        if not MODEL_PATH.exists():
            raise RecognitionError(f"Model file not found: {MODEL_PATH}")

        current_mtime = MODEL_PATH.stat().st_mtime
        if self._model is not None and self._model_mtime == current_mtime:
            return

        LOGGER.info("Loading Keras model from %s (timestamp: %s)", MODEL_PATH, current_mtime)
        self._model = load_model(str(MODEL_PATH))
        self._model_mtime = current_mtime

    def _open_camera(self):
        for index in (0, 1, 2):
            LOGGER.info("Trying camera index %s", index)
            camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Low latency on low-end CPU
                ok, _ = camera.read()
                if ok:
                    self._camera_index = index
                    return camera
            camera.release()

        self._camera_index = None
        raise RecognitionError("Could not open a webcam on camera indices 0, 1, or 2.")

    def _release_camera(self):
        if self._camera is not None:
            self._camera.release()
        self._camera = None
        self._camera_index = None

    def _predict_frame(self, frame):
        display_frame, contours, threshold, num_hands = self._get_img_contour_thresh(frame)
        if not contours or num_hands == 0:
            self._smooth_buffer.clear()
            return self._empty_prediction("No hand detected", num_hands=0)

        # Filter valid contours (reject small background noise)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 450]
        if not valid_contours:
            self._smooth_buffer.clear()
            return self._empty_prediction("Hand contour too small", num_hands=num_hands)

        # Combined prediction (all detected hands in one bounding box - works for 1 hand or 2 hands)
        all_points = np.vstack(valid_contours)
        comb_conf, comb_class_id = self._predict_from_points(all_points, threshold)

        final_conf = comb_conf
        final_class_id = comb_class_id

        # If 2 hands are detected and combined confidence is low, also test individual large contours
        if num_hands > 1 and comb_conf < self.confidence_threshold:
            large_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)[:2]
            for c in large_contours:
                conf, cid = self._predict_from_points(c, threshold)
                if conf > final_conf:
                    final_conf = conf
                    final_class_id = cid

        label = self._label_for_class(final_class_id)
        raw_detected = final_conf >= self.confidence_threshold and label != "Unknown"

        # Temporal smoothing buffer for smooth, flicker-free output on low-end CPUs
        if raw_detected:
            self._smooth_buffer.append((label, final_class_id, final_conf))
        else:
            if self._smooth_buffer:
                self._smooth_buffer.popleft()

        # Majority vote / smoothed label
        if self._smooth_buffer:
            labels = [item[0] for item in self._smooth_buffer]
            smoothed_label = max(set(labels), key=labels.count)
            # Find item corresponding to smoothed label
            smoothed_item = next(item for item in reversed(self._smooth_buffer) if item[0] == smoothed_label)
            display_label = smoothed_label
            display_conf = smoothed_item[2]
            display_class_id = smoothed_item[1]
            detected = True
        else:
            display_label = label if raw_detected else "Waiting"
            display_conf = final_conf
            display_class_id = final_class_id
            detected = raw_detected

        text = display_label if detected else ""
        hand_text = f"{num_hands} hand{'s' if num_hands > 1 else ''}"

        return {
            "detected": detected,
            "gesture": display_label if detected else "Waiting",
            "text": text,
            "confidence": round(display_conf, 2),
            "class_id": int(display_class_id) if display_class_id is not None else None,
            "num_hands": num_hands,
            "message": f"Detected {hand_text}" if detected else f"Waiting ({hand_text})",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "frame_width": int(display_frame.shape[1]),
            "frame_height": int(display_frame.shape[0]),
        }

    def _predict_from_points(self, points, threshold):
        x, y, width, height = cv2.boundingRect(points)
        image = threshold[y : y + height, x : x + width]
        if image.size == 0:
            return 0.0, 0

        # Pad to perfect square to maintain aspect ratio for 1-hand or 2-hand gestures
        if width > height:
            pad = int((width - height) / 2)
            image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
        elif height > width:
            pad = int((height - width) / 2)
            image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)

        processed = self._process_image(image)
        # Low-end CPU optimization: direct tensor call avoids heavy Keras generator overhead (23ms vs 230ms)
        probabilities = self._model(processed, training=False).numpy()[0]
        class_id = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities) * 100)
        return confidence, class_id

    def _empty_prediction(self, message, num_hands=0):
        return {
            "detected": False,
            "gesture": "Waiting",
            "text": "",
            "confidence": 0,
            "class_id": None,
            "num_hands": num_hands,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _get_img_contour_thresh(self, frame):
        # Resize frame to standard size for consistent processing and speed
        if frame.shape[0] != 480 or frame.shape[1] != 640:
            frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)

        mask, num_hands = self._get_hand_mask(frame)
        threshold = mask[self.roi_y : self.roi_y + self.roi_h, self.roi_x : self.roi_x + self.roi_w]
        contours = cv2.findContours(threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        return frame, contours, threshold, num_hands

    def _get_hand_mask(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # CPU optimization: writeable=False avoids internal buffer copying in MediaPipe
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if not results.multi_hand_landmarks:
            return mask, 0

        num_hands = len(results.multi_hand_landmarks)

        # Draw ALL detected hands (both hand 1 and hand 2) onto the mask
        for landmarks in results.multi_hand_landmarks:
            points = np.array(
                [[int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])] for lm in landmarks.landmark],
                dtype=np.int32,
            )
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

            for connection in self._mp_hands.HAND_CONNECTIONS:
                p1 = (
                    int(landmarks.landmark[connection[0]].x * frame.shape[1]),
                    int(landmarks.landmark[connection[0]].y * frame.shape[0]),
                )
                p2 = (
                    int(landmarks.landmark[connection[1]].x * frame.shape[1]),
                    int(landmarks.landmark[connection[1]].y * frame.shape[0]),
                )
                cv2.line(mask, p1, p2, 255, 15)

        return mask, num_hands

    def _process_image(self, image):
        image = cv2.resize(image, (self.image_x, self.image_y))
        image = np.array(image, dtype=np.float32) / 255.0
        return np.reshape(image, (1, self.image_x, self.image_y, 1))

    def _label_for_class(self, class_id):
        try:
            with sqlite3.connect(DATABASE_PATH) as conn:
                row = conn.execute("SELECT g_name FROM gesture WHERE g_id=?", (int(class_id),)).fetchone()
        except sqlite3.Error:
            LOGGER.exception("Failed to read gesture label from database")
            return "Unknown"

        return row[0] if row else "Unknown"

    def _session_payload(self, status):
        return {
            "status": status,
            "running": self._running,
            "camera_index": self._camera_index,
            "source": self._source,
            "session_started_at": self._session_started_at.isoformat() if self._session_started_at else None,
        }

    def _normalize_source(self, source):
        normalized = str(source or "browser").strip().lower()
        if normalized not in {"browser", "server"}:
            raise RecognitionError("Recognition source must be 'browser' or 'server'.")
        return normalized


recognition_service = RecognitionService()
