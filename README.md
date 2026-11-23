# Action Oriented Flick

A full-stack Action Oriented Flick gesture recognition system that captures webcam input in the browser, streams frames to a Flask API, processes single or dual hand gestures with OpenCV and MediaPipe, classifies gestures with a TensorFlow/Keras CNN model, and returns recognized text, confidence scores, speech output, session history, and analytics.

This project is designed as a practical assistive-technology prototype for converting sign language gestures into readable text and speech through a browser-based interface.

## Project Overview

Sign language communication can be difficult to interpret for people who do not know the gesture vocabulary. Action Oriented Flick addresses that gap by providing a real-time web application that detects hand gestures from a webcam and translates them into text and speech.

The application uses a static responsive frontend for webcam capture and user interaction, a Flask backend for API orchestration, MediaPipe for hand landmark detection, OpenCV for image preprocessing, TensorFlow/Keras for CNN inference, SQLite for gesture label mapping, and browser/backend speech synthesis for audio output. It also includes a recognition history table and statistics dashboard for session-level analytics.

## Features

- Real-time webcam recognition from the browser
- Browser-to-Flask frame streaming with base64 JPEG frames
- Single and dual hand detection (supports 1 or 2 hands simultaneously via MediaPipe Hands)
- TensorFlow/Keras CNN gesture predictions with confidence scoring
- Widened Region of Interest (ROI) and prediction smoothing buffer for stable detection
- MediaPipe Lite hand landmark detection (`model_complexity=0`) optimized for low-end CPUs
- OpenCV frame decoding, masking, convex hull extraction, and preprocessing
- Interactive gesture data management CLI (`manage_data.py`, `list_gestures.py`, `delete_gestures.py`)
- Automated one-step capture and training pipeline (`train_pipeline.py` & `train_gesture.bat`)
- One-click server launcher (`run.bat`)
- Text-to-speech conversion
- Recognition history and statistics dashboard
- Responsive frontend with custom favicon and modern UI
- REST API backend with automated unit and integration tests

## Architecture

```text
Browser Webcam
↓
Frontend (HTML/CSS/JS)
↓
Flask API
↓
OpenCV + MediaPipe
↓
TensorFlow Model
↓
SQLite Database
↓
Prediction Response
```

The recommended runtime path is a single Flask server. `backend/app.py` serves the static frontend from `frontend/` and exposes all JSON endpoints under `/api`.

## Tech Stack

### Frontend

- HTML
- CSS
- JavaScript

### Backend

- Python
- Flask
- Flask-Cors

### Machine Learning

- TensorFlow/Keras
- OpenCV
- MediaPipe
- NumPy

### Database

- SQLite

### Testing

- Python `unittest`
- Node.js built-in test runner

## Project Structure

```text
Action-Oriented-Flick/
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── wsgi.py
│   ├── requirements.txt
│   ├── routes/
│   │   └── api.py
│   ├── services/
│   │   ├── recognition_service.py
│   │   └── tts_service.py
│   ├── utils/
│   │   ├── paths.py
│   │   └── responses.py
│   ├── models/
│   │   └── cnn_model_keras2.h5
│   ├── data/
│   │   ├── database/
│   │   │   └── gesture_db.db
│   │   ├── datasets/
│   │   │   ├── train_images
│   │   │   ├── train_labels
│   │   │   ├── test_images
│   │   │   ├── test_labels
│   │   │   ├── val_images
│   │   │   └── val_labels
│   │   └── gestures/
│   ├── artifacts/
│   │   └── hist
│   └── app/
│       ├── cnn_model_train.py
│       ├── create_gestures.py
│       ├── delete_gestures.py
│       ├── display_gestures.py
│       ├── final.py
│       ├── final_backup.py
│       ├── list_gestures.py
│       ├── load_images.py
│       ├── manage_data.py
│       ├── paths.py
│       ├── Rotate_images.py
│       ├── set_hand_histogram.py
│       └── train_pipeline.py
├── frontend/
│   ├── index.html
│   ├── favicon.ico
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   ├── api.js
│   │   └── app.js
│   ├── assets/
│   │   ├── images/
│   │   │   └── hero-ai-sign-language.png
│   │   └── icons/
│   └── pages/
├── tests/
│   ├── __init__.py
│   ├── test_backend_api.py
│   └── frontend_static.test.mjs
├── manage_data.bat
├── train_gesture.bat
├── run.bat
├── package.json
├── .gitignore
└── README.md
```

## Installation

### Prerequisites

- Windows 10 or newer
- Python 3.11 recommended
  - The project virtual environment uses Python 3.11.9.
  - `backend/requirements.txt` pins TensorFlow/Keras 2.15.0, so Python 3.11 is the safest version.
- Git
- A modern browser with webcam support, such as Chrome or Edge
- A working webcam
- Node.js only if you want to run frontend tests

### Backend Setup

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

If PowerShell blocks script activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

*(Note: `backend\venv` is also supported by launcher scripts).*

### Required Runtime Files

The web application expects these files to exist:

```text
backend/models/cnn_model_keras2.h5
backend/data/database/gesture_db.db
```

Verify them with:

```powershell
Test-Path backend\models\cnn_model_keras2.h5
Test-Path backend\data\database\gesture_db.db
```

Both commands should return `True`.

The dataset files in `backend/data/datasets/` and gesture images in `backend/data/gestures/` are used by the data preparation and training scripts. They are not required for normal web-app inference if the trained model and SQLite database already exist.

### Environment Variables

Local development works with the defaults in `backend/config.py`. Optional variables:

```powershell
$env:APP_ENV="development"
$env:FLASK_ENV="development"
$env:CORS_ORIGINS="*"
$env:MAX_CONTENT_LENGTH="4194304"
$env:RECOGNITION_FRAME_REQUIRED="true"
$env:LOG_LEVEL="INFO"
```

## Running the Application

### Option 1: One-Click Launcher (Fastest)

Run `run.bat` from the root folder (or double-click it in File Explorer):

```powershell
.\run.bat
```

### Option 2: Terminal (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
python -m backend.app
```

Then open your browser at:

```text
http://127.0.0.1:5000
```

The Flask server starts at `127.0.0.1:5000`, serves `frontend/index.html`, and exposes the REST API under `/api`.

## Usage Guide

1. Open `http://127.0.0.1:5000`.
2. Click `Start Recognition`.
3. Allow webcam permission when the browser prompts you.
4. Keep one or both hands visible in the webcam preview within the detection area.
5. Watch the current gesture, confidence score, detected hand count, and recognized text update in the recognition panel.
6. Use `Text-to-Speech` to speak the generated sentence.
7. Review recent predictions in the recognition history table.
8. Monitor session totals and average confidence in the statistics dashboard.
9. Click `Stop Recognition` to stop the session and release the browser camera stream.
10. Click `Clear Text` to reset local text, backend history, and session statistics.

## Gesture Training and Dataset Management

Action Oriented Flick provides built-in tools to capture, augment, inspect, and train new single-hand or two-hand gestures.

### 1. One-Click Capture & Train Pipeline

Run the automated pipeline to capture gesture frames, augment data (flipping), package datasets, and train the CNN model in one step:

```powershell
.\train_gesture.bat
```
*Or via terminal:*
```powershell
.\.venv\Scripts\python.exe backend\app\train_pipeline.py
```

### 2. Interactive Data Manager

Manage datasets, review existing classes, or clean up obsolete gesture folders via an interactive console menu:

```powershell
.\manage_data.bat
```
*Or via terminal:*
```powershell
.\.venv\Scripts\python.exe backend\app\manage_data.py
```

### 3. View All Gesture IDs and Image Counts

```powershell
.\.venv\Scripts\python.exe backend\app\list_gestures.py
```

### 4. Delete Specific or All Gestures

Delete a single gesture by ID (e.g. ID 500):
```powershell
.\.venv\Scripts\python.exe backend\app\delete_gestures.py --id 500
```

Clean slate (delete all gesture folders and reset database):
```powershell
.\.venv\Scripts\python.exe backend\app\delete_gestures.py --all
```

### 5. Standalone Desktop Recognition Mode

Run real-time OpenCV GUI recognition directly on the desktop with visual hand annotations and speech synthesis:

```powershell
.\.venv\Scripts\python.exe backend\app\final.py
```

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Returns model, database, camera, and session status. |
| `POST` | `/api/start-recognition` | Loads the model if needed and starts recognition. Browser mode is the default. |
| `POST` | `/api/stop-recognition` | Stops recognition and releases any server-side camera session. |
| `POST` | `/api/predict` | Accepts a base64 image frame and returns gesture prediction, confidence, and `num_hands`. |
| `POST` | `/api/text-to-speech` | Queues text for backend speech synthesis. |
| `GET` | `/api/stats` | Returns session totals, average confidence, and last prediction. |
| `GET` | `/api/history` | Returns recent confident predictions. |
| `POST` | `/api/clear` | Clears recognition history and statistics. |

Example browser prediction response:

```json
{
  "detected": true,
  "gesture": "Peace",
  "text": "Peace",
  "confidence": 94.8,
  "class_id": 1,
  "num_hands": 1,
  "message": "Prediction successful",
  "timestamp": "2026-11-23T15:30:00Z"
}
```

The backend also supports explicit server-camera mode with `"source": "server"`, but browser mode is the recommended web-app path.

## Machine Learning Pipeline

1. The browser captures webcam video with `navigator.mediaDevices.getUserMedia`.
2. `frontend/js/app.js` draws the video frame to an off-DOM canvas.
3. The canvas is encoded as a JPEG data URL and sent to `POST /api/predict`.
4. `backend/routes/api.py` decodes the base64 payload and converts it into an OpenCV image.
5. `backend/services/recognition_service.py` flips and resizes the frame to 640x480.
6. MediaPipe Hands Lite model (`model_complexity=0`) detects up to 2 hands simultaneously (`max_num_hands=2`) with CPU optimizations.
7. A binary hand mask is generated from convex hull contours and landmark skeletal connections across all detected hands.
8. A widened, centered Region of Interest (`520x400`) is extracted from the mask so single and dual hand gestures fit comfortably.
9. OpenCV finds the largest contour and normalizes it into a `50x50` padded grayscale image.
10. The Keras model in `backend/models/cnn_model_keras2.h5` predicts gesture class probabilities (with hot-reloading when the model file updates).
11. A temporal prediction buffer stabilizes live output across consecutive frames to eliminate flicker.
12. The predicted class ID maps to its gesture label from SQLite (`backend/data/database/gesture_db.db`).
13. The API returns the gesture label, text, confidence score, detected hand count (`num_hands`), and timestamp.

## Testing

Run backend API tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run frontend contract tests:

```powershell
npm test
```

Run Python compile checks:

```powershell
.\.venv\Scripts\python.exe -m compileall -q backend tests
```

Current test coverage includes all Flask API endpoints and static frontend integration contracts. Full browser webcam verification still requires an interactive browser with camera permission.

## Screenshots

### Landing Page

Add screenshot here:

```text
screenshots/landing-page.png
```

### Recognition Page

Add screenshot here:

```text
screenshots/recognition-page.png
```

### Dashboard

Add screenshot here:

```text
screenshots/dashboard.png
```

### Statistics

Add screenshot here:

```text
screenshots/statistics.png
```

## Current Status

The application is integrated as a Flask-served web app with browser webcam capture, backend frame decoding, TensorFlow/MediaPipe/OpenCV recognition logic, SQLite label lookup, text-to-speech, session history, and dashboard statistics.

Known remaining work:

- Verify browser webcam prediction end-to-end in an interactive browser.
- Add full browser/DOM frontend tests.
- Add CI for backend tests, frontend tests, and compile checks.
- Write a full production deployment runbook.
- Refactor or archive duplicate legacy scripts under `backend/app/`.
- Decide a long-term artifact strategy for large model and dataset files.

## Future Improvements

- User authentication
- Cloud deployment
- Persistent analytics storage
- Multi-user support
- Improved model accuracy
- Production CI/CD workflow
- Full browser automation tests
- Model artifact storage with Git LFS or external artifact hosting

## Resume Highlights

- Built **Action Oriented Flick**, a full-stack real-time gesture recognition web application with Flask, TensorFlow/Keras, OpenCV, MediaPipe, and SQLite.
- Implemented browser-based webcam capture and base64 frame streaming from JavaScript to a Flask REST API.
- Extended the architecture to detect and classify both single-hand and dual-hand gestures simultaneously with MediaPipe Hands Lite.
- Integrated a CNN inference pipeline with confidence scoring, temporal prediction smoothing, and hot model reloading.
- Created interactive and automated CLI tools (`manage_data.py`, `train_pipeline.py`, `delete_gestures.py`) and one-click batch runners (`run.bat`, `train_gesture.bat`, `manage_data.bat`).
- Designed a responsive frontend with live recognition controls, prediction cards, text output, history, and analytics dashboard.
- Added text-to-speech support for converting recognized gesture text into speech.
- Implemented REST endpoints for recognition lifecycle, prediction, text-to-speech, statistics, history, and clearing session data.
- Added automated backend API tests and frontend integration contract tests.

## Troubleshooting

### Backend Does Not Start

Check that the virtual environment is active and dependencies are installed:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m backend.app
```

### TensorFlow Or MediaPipe Install Fails

Use Python 3.11 on Windows. The dependency versions in `backend/requirements.txt` are pinned around TensorFlow/Keras 2.15.0.

### Model File Not Found

Verify:

```powershell
Test-Path backend\models\cnn_model_keras2.h5
```

### Database Not Found

Verify:

```powershell
Test-Path backend\data\database\gesture_db.db
```

### API Unavailable In Browser

Confirm Flask is running and open:

```text
http://127.0.0.1:5000/api/health
```

### Webcam Does Not Start

- Use `http://127.0.0.1:5000`.
- Allow browser camera permission.
- Close other applications using the webcam.
- Try Chrome or Edge.

## Author & Project Info

- **Project Name:** Action Oriented Flick
- **Author:** Shani kumar
- **Context:** AI/ML 3rd-Year Project

## License

This project is provided under the MIT License. You may use, modify, and distribute the code with attribution. If this repository is published publicly, add a dedicated `LICENSE` file containing the full MIT License text.
