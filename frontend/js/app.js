import { recognitionApi } from "./api.js";

// Optimized interval & resolution for low-end CPU: smooth real-time response without CPU throttling
const PREDICTION_INTERVAL_MS = 380;
const MAX_HISTORY_ITEMS = 10;
const FRAME_CAPTURE_WIDTH = 480;

const state = {
  running: false,
  loading: false,
  isPredicting: false, // Concurrency guard: prevents queuing requests on low-end CPUs
  timerId: null,
  sessionTimerId: null,
  dashboardTimerId: null,
  sessionStart: null,
  history: [],
  sentence: [],
  totalDetected: 0,
  totalConfidence: 0,
  cameraAvailable: false,
  apiMode: recognitionApi.mode,
  voiceEnabled: true,      // On/off toggle for automatic voice sound
  lastSpokenGesture: "",   // Debounce: tracks the last gesture that was spoken and added
  noHandCount: 0,          // Tracks consecutive cycles with no detected hand
  selectedVoice: null,
  selectedVoiceURI: localStorage.getItem("preferred_voice") || "",
  voiceRate: parseFloat(localStorage.getItem("preferred_voice_rate") || "1.0"),
};

const elements = {
  navToggle: document.querySelector(".nav-toggle"),
  navLinks: document.querySelector(".nav-links"),
  videoFrame: document.getElementById("videoFrame"),
  webcamVideo: document.getElementById("webcamVideo"),
  cameraPlaceholder: document.getElementById("cameraPlaceholder"),
  liveBadge: document.getElementById("liveBadge"),
  handsBadge: document.getElementById("handsBadge"),
  gestureHandsCount: document.getElementById("gestureHandsCount"),
  connectionStatus: document.getElementById("connectionStatus"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  clearBtn: document.getElementById("clearBtn"),
  speakBtn: document.getElementById("speakBtn"),
  voiceToggleBtn: document.getElementById("voiceToggleBtn"),
  voiceToggleText: document.getElementById("voiceToggleText"),
  voiceToggleIcon: document.getElementById("voiceToggleIcon"),
  speechVoiceToggleBtn: document.getElementById("speechVoiceToggleBtn"),
  speechVoiceToggleText: document.getElementById("speechVoiceToggleText"),
  speechVoiceToggleIcon: document.getElementById("speechVoiceToggleIcon"),
  voiceSelect: document.getElementById("voiceSelect"),
  voiceSpeedRange: document.getElementById("voiceSpeedRange"),
  voiceSpeedVal: document.getElementById("voiceSpeedVal"),
  gestureLabel: document.getElementById("gestureLabel"),
  confidenceScore: document.getElementById("confidenceScore"),
  confidenceBar: document.getElementById("confidenceBar"),
  recognizedText: document.getElementById("recognizedText"),
  sentenceBox: document.getElementById("sentenceBox"),
  audioStatus: document.getElementById("audioStatus"),
  totalGestures: document.getElementById("totalGestures"),
  accuracyStat: document.getElementById("accuracyStat"),
  sessionDuration: document.getElementById("sessionDuration"),
  backendMode: document.getElementById("backendMode"),
  historyTable: document.getElementById("historyTable"),
  appAlert: document.getElementById("appAlert"),
};

const frameCanvas = document.createElement("canvas");
const frameContext = frameCanvas.getContext("2d");

function setText(element, text) {
  if (element) {
    element.textContent = text;
  }
}

function setAlert(message, type = "info") {
  if (!elements.appAlert) return;

  elements.appAlert.textContent = message;
  elements.appAlert.className = `app-alert ${type}`;
  elements.appAlert.hidden = !message;
}

function setLoading(isLoading, label = "Starting...") {
  state.loading = isLoading;
  elements.startBtn.disabled = isLoading || state.running;
  elements.stopBtn.disabled = isLoading || !state.running;
  elements.clearBtn.disabled = isLoading;
  elements.startBtn.setAttribute("aria-busy", String(isLoading));

  if (isLoading) {
    elements.startBtn.dataset.originalText ||= elements.startBtn.textContent;
    elements.startBtn.textContent = label;
  } else if (elements.startBtn.dataset.originalText) {
    elements.startBtn.textContent = elements.startBtn.dataset.originalText;
  }
}

function setStatusDot(container, text, modifier = "") {
  container.replaceChildren();

  const dot = document.createElement("span");
  dot.className = modifier ? `status-dot ${modifier}` : "status-dot";
  dot.setAttribute("aria-hidden", "true");

  container.append(dot, document.createTextNode(text));
}

function normalizePrediction(prediction) {
  const gesture = String(prediction?.gesture || "Waiting");
  const text = String(prediction?.text || "");
  const confidence = Math.max(0, Math.min(100, Number(prediction?.confidence || 0)));
  const numHands = Number(prediction?.num_hands || 0);

  return {
    gesture,
    text,
    confidence,
    numHands,
    detected: Boolean(prediction?.detected ?? text),
    source: prediction?.source || state.apiMode,
    warning: prediction?.warning || "",
    message: prediction?.message || "",
  };
}

function initNavigation() {
  elements.navToggle.addEventListener("click", () => {
    const isOpen = elements.navToggle.getAttribute("aria-expanded") === "true";
    elements.navToggle.setAttribute("aria-expanded", String(!isOpen));
    elements.navLinks.classList.toggle("is-open", !isOpen);
    document.body.classList.toggle("nav-open", !isOpen);
  });

  elements.navLinks.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      elements.navToggle.setAttribute("aria-expanded", "false");
      elements.navLinks.classList.remove("is-open");
      document.body.classList.remove("nav-open");
    });
  });
}

function initRevealAnimations() {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    document.querySelectorAll(".reveal").forEach((item) => item.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.16 }
  );

  document.querySelectorAll(".reveal").forEach((item) => observer.observe(item));
}

async function startCameraPreview() {
  if (!navigator.mediaDevices?.getUserMedia) {
    state.cameraAvailable = false;
    throw new Error("This browser does not support webcam access.");
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    elements.webcamVideo.srcObject = stream;
    await elements.webcamVideo.play();
    elements.webcamVideo.hidden = false;
    elements.cameraPlaceholder.hidden = true;
    state.cameraAvailable = true;
    return true;
  } catch (error) {
    state.cameraAvailable = false;
    elements.cameraPlaceholder.hidden = false;
    let detail = "Enable camera permission and try again.";
    if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
      detail = "No camera hardware was detected on your computer. Please plug in or enable a webcam.";
    } else if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
      detail = "Camera permission was denied in your browser. Click the camera/lock icon in the address bar to allow camera access.";
    } else if (error.name === "NotReadableError" || error.name === "TrackStartError") {
      detail = "The camera is currently in use by another program (e.g. Zoom, Teams, Camera app). Please close it and try again.";
    }
    throw new Error(`Camera access was denied or unavailable. ${detail}`);
  }
}

function stopCameraPreview() {
  const stream = elements.webcamVideo.srcObject;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }

  elements.webcamVideo.srcObject = null;
  elements.webcamVideo.hidden = true;
  elements.cameraPlaceholder.hidden = false;
}

function updateHandsIndicators(numHands) {
  const count = Number(numHands || 0);
  const text = count === 1 ? "1 Hand" : count >= 2 ? "2 Hands" : "0 Hands";
  const modifier = count === 1 ? "one-hand" : count >= 2 ? "two-hands" : "no-hand";

  if (elements.handsBadge) {
    elements.handsBadge.hidden = !state.running;
    elements.handsBadge.textContent = text;
    elements.handsBadge.className = `hands-badge ${modifier}`;
  }
  if (elements.gestureHandsCount) {
    elements.gestureHandsCount.textContent = text;
    elements.gestureHandsCount.className = `hands-badge ${modifier}`;
  }
}

function setRecognitionStatus(isRunning) {
  state.running = isRunning;
  state.isPredicting = false;
  elements.startBtn.disabled = state.loading || isRunning;
  elements.stopBtn.disabled = state.loading || !isRunning;
  elements.liveBadge.textContent = isRunning ? "Live" : "Idle";
  elements.liveBadge.classList.toggle("active", isRunning);
  elements.videoFrame.classList.toggle("is-running", isRunning);

  if (elements.handsBadge) {
    elements.handsBadge.hidden = !isRunning;
    if (!isRunning) {
      elements.handsBadge.textContent = "0 Hands";
      elements.handsBadge.className = "hands-badge no-hand";
    }
  }
  if (elements.gestureHandsCount && !isRunning) {
    elements.gestureHandsCount.textContent = "0 Hands";
    elements.gestureHandsCount.className = "hands-badge no-hand";
  }

  const label = isRunning
    ? state.cameraAvailable
      ? "Recognition active"
      : "Camera unavailable"
    : "Recognition idle";
  setText(elements.connectionStatus, label);
}

async function startRecognition() {
  if (state.running || state.loading) return;

  setAlert("", "info");
  setLoading(true, "Starting...");

  try {
    await startCameraPreview();
    const session = await recognitionApi.startSession();

    state.apiMode = session.source || "backend";

    setRecognitionStatus(true);
    state.sessionStart = Date.now();
    updateSessionDuration();
    state.sessionTimerId = window.setInterval(updateSessionDuration, 1000);
    state.dashboardTimerId = window.setInterval(refreshDashboard, 3000);
    state.timerId = window.setInterval(runPredictionCycle, PREDICTION_INTERVAL_MS);
    await refreshDashboard();
    await runPredictionCycle();
  } catch (error) {
    console.error(error);
    setAlert(error.message || "Recognition could not start. Check camera permissions or backend availability.", "error");
    stopCameraPreview();
    setRecognitionStatus(false);
  } finally {
    setLoading(false);
  }
}

async function stopRecognition() {
  if (!state.running || state.loading) return;

  setLoading(true, "Stopping...");

  try {
    await recognitionApi.stopSession();
  } catch (error) {
    console.error(error);
    setAlert("The backend did not confirm stop, but the local session was closed.", "warning");
  } finally {
    window.clearInterval(state.timerId);
    window.clearInterval(state.sessionTimerId);
    window.clearInterval(state.dashboardTimerId);
    state.timerId = null;
    state.sessionTimerId = null;
    state.dashboardTimerId = null;
    stopCameraPreview();
    setRecognitionStatus(false);
    await refreshDashboard().catch((error) => console.error(error));
    setLoading(false);
  }
}

function captureFrame() {
  const video = elements.webcamVideo;
  if (!state.cameraAvailable || !video.srcObject || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    throw new Error("Camera frame is not ready yet.");
  }

  const sourceWidth = video.videoWidth || FRAME_CAPTURE_WIDTH;
  const sourceHeight = video.videoHeight || Math.round(FRAME_CAPTURE_WIDTH * 0.75);
  const scale = FRAME_CAPTURE_WIDTH / sourceWidth;
  frameCanvas.width = FRAME_CAPTURE_WIDTH;
  frameCanvas.height = Math.max(1, Math.round(sourceHeight * scale));
  frameContext.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
  // Optimized capture width (480) reduces CPU encoding time & bandwidth by ~50%
  return frameCanvas.toDataURL("image/jpeg", 0.82);
}

async function runPredictionCycle() {
  // Low-end CPU optimization: never pile up overlapping prediction requests
  if (!state.running || state.isPredicting) return;
  state.isPredicting = true;

  try {
    const frame = captureFrame();
    const prediction = normalizePrediction(await recognitionApi.getPrediction(frame));
    appendPrediction(prediction);
    updateHandsIndicators(prediction.numHands);
    await refreshDashboard();

    if (prediction.warning) {
      setAlert(prediction.warning, "warning");
    }
  } catch (error) {
    console.error(error);
    setAlert(error.message || "Prediction failed. Waiting for the next recognition frame.", "error");
  } finally {
    state.isPredicting = false;
  }
}

function appendPrediction(prediction) {
  state.apiMode = prediction.source === "backend" ? "backend" : state.apiMode;

  // Step 1: Immediately show the recognized text & metrics corresponding to current gesture of hand
  setText(elements.gestureLabel, prediction.gesture);
  setText(elements.confidenceScore, `${prediction.confidence}%`);
  elements.confidenceBar.style.width = `${prediction.confidence}%`;
  updateHandsIndicators(prediction.numHands);

  if (prediction.detected && prediction.text) {
    setText(elements.recognizedText, prediction.text);
    state.noHandCount = 0;

    // Step 2: One-time execution: only one time gesture, one time speech of one meaning
    if (prediction.text !== state.lastSpokenGesture) {
      state.lastSpokenGesture = prediction.text;

      const event = {
        ...prediction,
        time: new Date(),
      };

      state.history.unshift(event);
      state.history = state.history.slice(0, MAX_HISTORY_ITEMS);
      state.totalDetected += 1;
      state.totalConfidence += prediction.confidence;
      state.sentence.push(prediction.text);

      renderSentence();
      renderStats();
      renderHistory();

      // Step 3: Automatic speech starts only once per detected gesture
      if (state.voiceEnabled) {
        speakGesture(prediction.text);
      }
    }
  } else {
    setText(elements.recognizedText, prediction.text || prediction.message || "Waiting for a confident gesture.");
    state.noHandCount += 1;

    // When hand is removed or no gesture detected for 2 consecutive cycles,
    // reset lastSpokenGesture so the user can perform the same gesture again
    if (state.noHandCount >= 2) {
      state.lastSpokenGesture = "";
    }
  }
}

function renderSentence() {
  if (!state.sentence.length) {
    setText(elements.sentenceBox, "Your generated sentence will appear here.");
    elements.sentenceBox.classList.add("is-empty");
    elements.speakBtn.disabled = true;
    return;
  }

  setText(elements.sentenceBox, state.sentence.join(" ").replace(/\s+/g, " ").trim());
  elements.sentenceBox.classList.remove("is-empty");
  elements.speakBtn.disabled = false;
}

function renderStats() {
  setText(elements.totalGestures, String(state.totalDetected));
  const averageConfidence = state.totalDetected ? Math.round(state.totalConfidence / state.totalDetected) : 0;
  setText(elements.accuracyStat, `${averageConfidence}%`);
  setText(elements.backendMode, "Backend");
}

function createCell(text) {
  const cell = document.createElement("td");
  cell.textContent = text;
  return cell;
}

function renderHistory() {
  elements.historyTable.replaceChildren();

  if (!state.history.length) {
    const row = document.createElement("tr");
    row.className = "empty-row";
    const cell = createCell("No recognition events yet.");
    cell.colSpan = 4;
    row.append(cell);
    elements.historyTable.append(row);
    return;
  }

  state.history.forEach((item) => {
    const row = document.createElement("tr");
    row.append(
      createCell(item.time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })),
      createCell(item.gesture),
      createCell(item.text),
      createCell(`${item.confidence}%`)
    );
    elements.historyTable.append(row);
  });
}

function parseHistoryItem(item) {
  return {
    gesture: item.gesture || "Waiting",
    text: item.text || "",
    confidence: Number(item.confidence || 0),
    time: item.timestamp ? new Date(item.timestamp) : new Date(),
  };
}

function applyStats(stats) {
  if (!stats) return;
  state.totalDetected = Number(stats.total_gestures || 0);
  state.totalConfidence = state.totalDetected * Number(stats.average_confidence || 0);
  if (stats.session_started_at && !state.sessionStart) {
    state.sessionStart = Date.parse(stats.session_started_at);
  }
  renderStats();
}

function applyHistory(history) {
  if (!Array.isArray(history)) return;
  state.history = history.slice(0, MAX_HISTORY_ITEMS).map(parseHistoryItem);
  const words = [];
  state.history
    .slice()
    .reverse()
    .forEach((item) => {
      if (item.text && words[words.length - 1] !== item.text) {
        words.push(item.text);
      }
    });
  state.sentence = words;
  renderSentence();
  renderHistory();
}

async function refreshDashboard() {
  const [statsResult, historyResult] = await Promise.all([
    recognitionApi.stats(),
    recognitionApi.history(),
  ]);
  applyStats(statsResult.stats);
  applyHistory(historyResult.history);
}

function updateSessionDuration() {
  if (!state.sessionStart) return;

  const elapsed = Math.floor((Date.now() - state.sessionStart) / 1000);
  const minutes = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const seconds = String(elapsed % 60).padStart(2, "0");
  setText(elements.sessionDuration, `${minutes}:${seconds}`);
}

function updateVoiceToggleUI() {
  const isEnabled = state.voiceEnabled;
  const icon = isEnabled ? "🔊" : "🔇";
  const label = isEnabled ? "Voice: ON" : "Voice: OFF";
  const stateClass = isEnabled ? "voice-on" : "voice-off";
  const removeClass = isEnabled ? "voice-off" : "voice-on";

  [elements.voiceToggleBtn, elements.speechVoiceToggleBtn].forEach((btn) => {
    if (!btn) return;
    btn.setAttribute("aria-pressed", String(isEnabled));
    btn.classList.add(stateClass);
    btn.classList.remove(removeClass);
    btn.title = isEnabled ? "Click to mute voice sound (Voice OFF)" : "Click to unmute voice sound (Voice ON)";
  });

  if (elements.voiceToggleIcon) elements.voiceToggleIcon.textContent = icon;
  if (elements.voiceToggleText) elements.voiceToggleText.textContent = label;
  if (elements.speechVoiceToggleIcon) elements.speechVoiceToggleIcon.textContent = icon;
  if (elements.speechVoiceToggleText) elements.speechVoiceToggleText.textContent = label;

  if (!isEnabled) {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    setStatusDot(elements.audioStatus, "Voice Muted (Off)", "muted");
  } else {
    setStatusDot(elements.audioStatus, "Audio idle", "muted");
  }
}

function toggleVoice() {
  state.voiceEnabled = !state.voiceEnabled;
  updateVoiceToggleUI();
}

function guessGender(name) {
  const lower = name.toLowerCase();
  if (
    lower.includes("female") ||
    lower.includes("woman") ||
    lower.includes("girl") ||
    lower.includes("heera") ||
    lower.includes("zira") ||
    lower.includes("hazel") ||
    lower.includes("susan") ||
    lower.includes("jenny") ||
    lower.includes("neerja") ||
    lower.includes("swara") ||
    lower.includes("aditi") ||
    lower.includes("veena") ||
    lower.includes("kalpana") ||
    lower.includes("priya") ||
    lower.includes("ananya") ||
    lower.includes("pooja")
  ) {
    return "Female";
  }
  if (
    lower.includes("male") ||
    lower.includes("man") ||
    lower.includes("boy") ||
    lower.includes("ravi") ||
    lower.includes("david") ||
    lower.includes("george") ||
    lower.includes("mark") ||
    lower.includes("guy") ||
    lower.includes("madhur") ||
    lower.includes("prabhat") ||
    lower.includes("amit") ||
    lower.includes("raj")
  ) {
    return "Male";
  }
  return "";
}

function getRegionGroup(voice) {
  const lang = (voice.lang || "").toLowerCase();
  const name = voice.name.toLowerCase();

  if (
    lang.includes("-in") ||
    lang === "hi" ||
    lang.startsWith("hi-") ||
    lang.startsWith("ta-") ||
    lang.startsWith("te-") ||
    lang.startsWith("bn-") ||
    lang.startsWith("mr-") ||
    lang.startsWith("gu-") ||
    name.includes("india") ||
    name.includes("hindi")
  ) {
    return "india";
  }
  if (lang.includes("-us") || name.includes("united states") || name.includes("us english")) {
    return "us";
  }
  if (lang.includes("-gb") || lang.includes("-uk") || name.includes("united kingdom") || name.includes("uk english")) {
    return "uk";
  }
  if (lang.includes("-au") || name.includes("australia")) {
    return "au";
  }
  if (lang.includes("-ca") || name.includes("canada")) {
    return "ca";
  }
  return "other";
}

function populateVoiceList() {
  if (!("speechSynthesis" in window) || !elements.voiceSelect) return;

  const voices = window.speechSynthesis.getVoices();
  if (!voices.length) return;

  const groups = {
    india: { label: "🇮🇳 Indian Voices (Male & Female)", voices: [] },
    us: { label: "🇺🇸 United States (US)", voices: [] },
    uk: { label: "🇬🇧 United Kingdom (UK)", voices: [] },
    au: { label: "🇦🇺 Australia", voices: [] },
    ca: { label: "🇨🇦 Canada", voices: [] },
    other: { label: "🌐 Other Country Voices", voices: [] },
  };

  voices.forEach((voice) => {
    const region = getRegionGroup(voice);
    groups[region].voices.push(voice);
  });

  elements.voiceSelect.replaceChildren();

  const defaultOption = document.createElement("option");
  defaultOption.value = "";
  defaultOption.textContent = "— Default System Voice —";
  elements.voiceSelect.appendChild(defaultOption);

  let matchFound = false;

  Object.values(groups).forEach((grp) => {
    if (!grp.voices.length) return;

    const optgroup = document.createElement("optgroup");
    optgroup.label = grp.label;

    grp.voices.forEach((voice) => {
      const option = document.createElement("option");
      const uri = voice.voiceURI || voice.name;
      option.value = uri;

      const gender = guessGender(voice.name);
      const genderLabel = gender ? ` [${gender}]` : "";
      option.textContent = `${voice.name} (${voice.lang})${genderLabel}`;

      if (state.selectedVoiceURI && uri === state.selectedVoiceURI) {
        option.selected = true;
        state.selectedVoice = voice;
        matchFound = true;
      }

      optgroup.appendChild(option);
    });

    elements.voiceSelect.appendChild(optgroup);
  });

  if (!matchFound && !state.selectedVoiceURI) {
    if (groups.india.voices.length > 0) {
      const preferred = groups.india.voices[0];
      state.selectedVoice = preferred;
      state.selectedVoiceURI = preferred.voiceURI || preferred.name;
      elements.voiceSelect.value = state.selectedVoiceURI;
    }
  } else if (matchFound && state.selectedVoiceURI) {
    elements.voiceSelect.value = state.selectedVoiceURI;
  }
}

function speakGesture(text) {
  if (!state.voiceEnabled || !text) return;

  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    if (state.selectedVoice) {
      utterance.voice = state.selectedVoice;
    }
    utterance.rate = state.voiceRate || 1.0;
    utterance.pitch = 1.0;

    utterance.onstart = () => {
      setStatusDot(elements.audioStatus, `Speaking: "${text}"`);
    };
    utterance.onend = () => {
      setStatusDot(elements.audioStatus, state.voiceEnabled ? "Audio idle" : "Voice Muted (Off)", "muted");
    };
    utterance.onerror = (err) => {
      if (err.error !== "canceled" && err.error !== "interrupted") {
        console.warn("Speech error:", err);
      }
      setStatusDot(elements.audioStatus, state.voiceEnabled ? "Audio idle" : "Voice Muted (Off)", "muted");
    };

    window.speechSynthesis.speak(utterance);
  } else {
    setStatusDot(elements.audioStatus, "Speech API unavailable", "error");
  }
}

async function clearText() {
  state.history = [];
  state.sentence = [];
  state.totalDetected = 0;
  state.totalConfidence = 0;
  state.lastSpokenGesture = "";
  state.noHandCount = 0;

  setText(elements.gestureLabel, "Waiting");
  setText(elements.confidenceScore, "0%");
  elements.confidenceBar.style.width = "0";
  setText(elements.recognizedText, "No gestures detected yet.");
  setText(elements.sessionDuration, "00:00");
  updateHandsIndicators(0);
  setAlert("", "info");
  state.sessionStart = state.running ? Date.now() : null;

  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  setStatusDot(elements.audioStatus, state.voiceEnabled ? "Audio idle" : "Voice Muted (Off)", "muted");

  try {
    await recognitionApi.clear();
    await refreshDashboard();
  } catch (error) {
    console.error(error);
    setAlert("Local text was cleared, but backend history could not be reset.", "warning");
  }

  renderSentence();
  renderStats();
  renderHistory();
}

async function speakGeneratedSentence() {
  const text = state.sentence.join(" ").trim();
  if (!text) {
    setStatusDot(elements.audioStatus, "No sentence available", "muted");
    return;
  }

  elements.speakBtn.disabled = true;

  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    if (state.selectedVoice) {
      utterance.voice = state.selectedVoice;
    }
    utterance.rate = state.voiceRate || 1.0;
    utterance.pitch = 1.0;

    utterance.onstart = () => setStatusDot(elements.audioStatus, "Speaking");
    utterance.onend = () => {
      setStatusDot(elements.audioStatus, state.voiceEnabled ? "Audio idle" : "Voice Muted (Off)", "muted");
      elements.speakBtn.disabled = false;
    };
    utterance.onerror = (err) => {
      if (err.error !== "canceled" && err.error !== "interrupted") {
        console.warn("Speech error:", err);
      }
      setStatusDot(elements.audioStatus, state.voiceEnabled ? "Audio idle" : "Voice Muted (Off)", "muted");
      elements.speakBtn.disabled = false;
    };

    window.speechSynthesis.speak(utterance);
  } else {
    try {
      setStatusDot(elements.audioStatus, "Preparing speech");
      const speechResult = await recognitionApi.speak(text);
      setStatusDot(elements.audioStatus, speechResult.status === "queued" ? "Speech queued" : "Speech sent");
    } catch (error) {
      console.error(error);
      setStatusDot(elements.audioStatus, "Speech failed", "error");
    } finally {
      elements.speakBtn.disabled = false;
    }
  }
}

function bindEvents() {
  elements.startBtn.addEventListener("click", startRecognition);
  elements.stopBtn.addEventListener("click", stopRecognition);
  elements.clearBtn.addEventListener("click", clearText);
  elements.speakBtn.addEventListener("click", speakGeneratedSentence);
  if (elements.voiceToggleBtn) {
    elements.voiceToggleBtn.addEventListener("click", toggleVoice);
  }
  if (elements.speechVoiceToggleBtn) {
    elements.speechVoiceToggleBtn.addEventListener("click", toggleVoice);
  }

  if (elements.voiceSelect) {
    elements.voiceSelect.addEventListener("change", (e) => {
      const uri = e.target.value;
      state.selectedVoiceURI = uri;
      localStorage.setItem("preferred_voice", uri);
      if ("speechSynthesis" in window) {
        const voices = window.speechSynthesis.getVoices();
        state.selectedVoice = voices.find((v) => (v.voiceURI || v.name) === uri) || null;
      }
      if (state.voiceEnabled) {
        speakGesture("Voice changed");
      }
    });
  }

  if (elements.voiceSpeedRange) {
    elements.voiceSpeedRange.value = String(state.voiceRate);
    if (elements.voiceSpeedVal) {
      elements.voiceSpeedVal.textContent = `${state.voiceRate.toFixed(1)}x`;
    }
    elements.voiceSpeedRange.addEventListener("input", (e) => {
      const rate = parseFloat(e.target.value);
      state.voiceRate = rate;
      localStorage.setItem("preferred_voice_rate", String(rate));
      if (elements.voiceSpeedVal) {
        elements.voiceSpeedVal.textContent = `${rate.toFixed(1)}x`;
      }
    });
  }
}

async function initBackendStatus() {
  const health = await recognitionApi.health();
  state.apiMode = health.source || recognitionApi.mode;
  setText(elements.backendMode, "Backend");
  setText(elements.connectionStatus, health.service?.status === "ready" ? "Backend connected" : "Backend degraded");
  await refreshDashboard();
}

async function initApp() {
  initNavigation();
  initRevealAnimations();
  bindEvents();
  updateVoiceToggleUI();

  if ("speechSynthesis" in window) {
    populateVoiceList();
    window.speechSynthesis.onvoiceschanged = populateVoiceList;
  }

  renderSentence();
  renderStats();
  renderHistory();

  try {
    await initBackendStatus();
  } catch (error) {
    console.error(error);
    setText(elements.connectionStatus, "Backend unavailable");
    setAlert(error.message || "Backend API is unavailable.", "error");
  }
}

document.addEventListener("DOMContentLoaded", initApp);
