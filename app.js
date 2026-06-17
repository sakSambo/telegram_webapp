import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";
import { createLocalPredictor } from "./local_model.js";
import { frameDelta as landmarkFrameDelta, normalizeFrame as normalizeLandmarksFrame } from "./landmark_features.js";

const HAND_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
];

const API_BASE_KEY = "aslTelegramApiBase";
const MAX_FRAMES = 30;
const MIN_FRAMES = 8;
const PREDICT_INTERVAL_MS = 260;
const NO_HAND_RESET_FRAMES = 4;

const els = {
    video: document.getElementById("camera-video"),
    canvas: document.getElementById("overlay-canvas"),
    connection: document.getElementById("connection-status"),
    signal: document.getElementById("signal-status"),
    prediction: document.getElementById("current-prediction"),
    confidence: document.getElementById("confidence-pill"),
    output: document.getElementById("output-text"),
    wordSuggestions: document.getElementById("word-suggestions"),
    sentenceSuggestions: document.getElementById("sentence-suggestions"),
    settingsPanel: document.getElementById("settings-panel"),
    guideSheet: document.getElementById("guide-sheet"),
    apiBaseInput: document.getElementById("api-base-input"),
    skeletonToggle: document.getElementById("toggle-skeleton"),
    autoAppendToggle: document.getElementById("toggle-auto-append"),
};

const state = {
    apiBase: loadApiBase(),
    handLandmarker: null,
    frames: [],
    normalizedFrames: [],
    noHandFrames: 0,
    lastPredictAt: 0,
    pendingPrediction: false,
    localPredictor: null,
    candidate: "",
    candidateCount: 0,
    lastCommitted: "",
    cooldownUntil: 0,
    latestSentenceAssist: null,
};

const ctx = els.canvas.getContext("2d");

bootstrapTelegram();
wireControls();
startApp();

function bootstrapTelegram() {
    const telegram = window.Telegram && window.Telegram.WebApp;
    if (!telegram) return;
    telegram.ready();
    telegram.expand();
    document.body.style.background = telegram.themeParams.bg_color || "";
}

function wireControls() {
    els.apiBaseInput.value = state.apiBase;

    document.getElementById("btn-settings").addEventListener("click", () => {
        els.settingsPanel.classList.remove("hidden");
    });
    document.getElementById("btn-close-settings").addEventListener("click", () => {
        els.settingsPanel.classList.add("hidden");
    });
    document.getElementById("btn-save-api").addEventListener("click", () => {
        state.apiBase = els.apiBaseInput.value.trim().replace(/\/$/, "");
        localStorage.setItem(API_BASE_KEY, state.apiBase);
        setStatus("Saved backend URL.");
    });
    document.getElementById("btn-test-api").addEventListener("click", testApi);

    document.getElementById("btn-guide").addEventListener("click", () => {
        els.guideSheet.classList.remove("hidden");
    });
    document.getElementById("btn-close-guide").addEventListener("click", () => {
        els.guideSheet.classList.add("hidden");
    });

    document.getElementById("btn-space").addEventListener("click", () => {
        appendText(" ");
    });
    document.getElementById("btn-backspace").addEventListener("click", () => {
        els.output.value = els.output.value.slice(0, -1);
        scheduleWordAssist();
        state.latestSentenceAssist = null;
    });
    document.getElementById("btn-clear").addEventListener("click", () => {
        els.output.value = "";
        state.lastCommitted = "";
        clearWordSuggestions("Word suggestions appear here.");
        clearSentenceSuggestions("Check a phrase when it is ready.");
    });
    document.getElementById("btn-sentence").addEventListener("click", fetchSentenceAssist);
    document.getElementById("btn-apply-safe").addEventListener("click", applySafeSentenceFixes);
    els.output.addEventListener("input", () => {
        scheduleWordAssist();
        state.latestSentenceAssist = null;
    });
}

async function startApp() {
    try {
        await initCamera();
        await initHandLandmarker();
        await initLocalPredictor();
        setStatus(state.localPredictor ? `Local model ready (${state.localPredictor.provider}).` : "Camera ready. Set backend URL if prediction is unavailable.");
        requestAnimationFrame(processFrame);
    } catch (error) {
        setStatus(error.message || "Unable to start camera.");
    }
}

async function initLocalPredictor() {
    try {
        setStatus("Loading local ASL model...");
        state.localPredictor = await createLocalPredictor();
    } catch (error) {
        state.localPredictor = null;
        console.warn("Local ONNX model unavailable", error);
        setStatus("Local model unavailable. Backend fallback available.");
    }
}

async function initCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: {
            facingMode: "user",
            width: { ideal: 720 },
            height: { ideal: 960 },
        },
        audio: false,
    });
    els.video.srcObject = stream;
    await els.video.play();
}

async function initHandLandmarker() {
    setStatus("Loading hand tracker...");
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
    );
    state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
    });
}

async function processFrame(now) {
    resizeCanvasToVideo();
    const results = state.handLandmarker.detectForVideo(els.video, now);
    ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);

    if (!results.landmarks || results.landmarks.length === 0) {
        observeNoHand();
        requestAnimationFrame(processFrame);
        return;
    }

    const landmarks = results.landmarks[0];
    observeHand(landmarks);
    if (els.skeletonToggle.checked) {
        drawHand(landmarks);
    }

    if (now - state.lastPredictAt >= PREDICT_INTERVAL_MS && !state.pendingPrediction) {
        state.lastPredictAt = now;
        predictCurrentSequence();
    }

    requestAnimationFrame(processFrame);
}

function observeNoHand() {
    state.noHandFrames += 1;
    if (state.noHandFrames >= NO_HAND_RESET_FRAMES) {
        state.frames = [];
        state.normalizedFrames = [];
        state.candidate = "";
        state.candidateCount = 0;
        state.lastCommitted = "";
        els.prediction.textContent = "Resting...";
        els.confidence.textContent = "--";
        els.signal.textContent = "No hand detected.";
    }
}

function observeHand(landmarks) {
    state.noHandFrames = 0;
    const frame = flattenMirroredLandmarks(landmarks);
    state.frames.push(frame);
    state.normalizedFrames.push(normalizeLandmarksFrame(frame));
    if (state.frames.length > MAX_FRAMES) state.frames.shift();
    if (state.normalizedFrames.length > MAX_FRAMES) state.normalizedFrames.shift();
    els.signal.textContent = `${state.frames.length} frames ready`;
}

async function predictCurrentSequence() {
    if (state.frames.length < MIN_FRAMES) return;
    state.pendingPrediction = true;
    try {
        if (state.localPredictor) {
            const result = await state.localPredictor.predict(state.normalizedFrames, modeHint());
            handlePrediction(result);
            return;
        }
        if (!state.apiBase) {
            setStatus("Set backend URL or reload after local model is available.");
            return;
        }
        const response = await fetch(apiUrl("/api/predict_sequence"), {
            method: "POST",
            headers: telegramHeaders(),
            body: JSON.stringify({
                frames: state.frames,
                mode_hint: modeHint(),
            }),
        });
        const result = await response.json();
        handlePrediction(result);
    } catch (error) {
        setStatus("Prediction backend unavailable.");
    } finally {
        state.pendingPrediction = false;
    }
}

function handlePrediction(result) {
    if (!result || result.status !== "ok") return;
    const display = result.raw_label || result.display || "Reading...";
    els.prediction.textContent = display;
    els.confidence.textContent = result.confidence ? `${Math.round(result.confidence * 100)}%` : "--";
        const provider = result.provider ? ` | ${result.provider}` : "";
        els.signal.textContent = `${result.frame_count || state.frames.length} frames | ${result.mode_hint || "auto"}${provider}`;

    if (!result.accepted || !result.label || !els.autoAppendToggle.checked) return;

    if (result.label === state.candidate) {
        state.candidateCount += 1;
    } else {
        state.candidate = result.label;
        state.candidateCount = 1;
    }

    const commitCount = result.label === "X" ? 4 : 2;
    if (state.candidateCount < commitCount || Date.now() < state.cooldownUntil) return;
    if (result.label === state.lastCommitted) return;

    appendText(result.label);
    state.lastCommitted = result.label;
    state.cooldownUntil = Date.now() + 950;
}

function appendText(value) {
    els.output.value += value;
    els.output.focus();
    scheduleWordAssist();
    state.latestSentenceAssist = null;
}

function flattenMirroredLandmarks(landmarks) {
    const frame = [];
    for (const landmark of landmarks) {
        frame.push(1 - landmark.x, landmark.y, landmark.z || 0);
    }
    return frame;
}

function modeHint() {
    if (state.normalizedFrames.length < 2) return "auto";
    const recent = state.normalizedFrames.slice(-12);
    const deltas = [];
    for (let index = 1; index < recent.length; index += 1) {
        deltas.push(landmarkFrameDelta(recent[index - 1], recent[index]));
    }
    const mean = deltas.reduce((sum, value) => sum + value, 0) / Math.max(deltas.length, 1);
    const max = Math.max(...deltas, 0);
    if (mean >= 0.050 || max >= 0.140) return "motion";
    return "static";
}

function drawHand(landmarks) {
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#38bdf8";
    ctx.fillStyle = "#22c55e";

    for (const [start, end] of HAND_CONNECTIONS) {
        const first = landmarks[start];
        const second = landmarks[end];
        ctx.beginPath();
        ctx.moveTo(first.x * els.canvas.width, first.y * els.canvas.height);
        ctx.lineTo(second.x * els.canvas.width, second.y * els.canvas.height);
        ctx.stroke();
    }

    for (const landmark of landmarks) {
        ctx.beginPath();
        ctx.arc(landmark.x * els.canvas.width, landmark.y * els.canvas.height, 5, 0, Math.PI * 2);
        ctx.fill();
    }
}

let wordAssistTimer = null;
function scheduleWordAssist() {
    window.clearTimeout(wordAssistTimer);
    wordAssistTimer = window.setTimeout(fetchWordAssist, 260);
}

async function fetchWordAssist() {
    if (!els.output.value.trim()) {
        clearWordSuggestions("Word suggestions appear here.");
        return;
    }
    try {
        const response = await fetch(apiUrl("/api/text_assist"), {
            method: "POST",
            headers: telegramHeaders(),
            body: JSON.stringify({ text: els.output.value }),
        });
        const data = await response.json();
        renderWordSuggestions(data);
    } catch (error) {
        clearWordSuggestions("Word suggestions unavailable.");
    }
}

function renderWordSuggestions(data) {
    els.wordSuggestions.innerHTML = "";
    if (!data.suggestions || data.suggestions.length === 0) {
        clearWordSuggestions(data.current_word ? "No correction needed." : "Word suggestions appear here.");
        return;
    }
    for (const suggestion of data.suggestions) {
        const chip = document.createElement("button");
        chip.className = "chip";
        chip.type = "button";
        chip.textContent = suggestion.text;
        chip.addEventListener("click", () => applyWordSuggestion(data, suggestion));
        els.wordSuggestions.appendChild(chip);
    }
}

function clearWordSuggestions(text) {
    els.wordSuggestions.innerHTML = "";
    const hint = document.createElement("span");
    hint.className = "hint";
    hint.textContent = text;
    els.wordSuggestions.appendChild(hint);
}

function applyWordSuggestion(data, suggestion) {
    const current = els.output.value.slice(data.start, data.end);
    if (current.toLowerCase() !== (data.current_word || "").toLowerCase()) {
        scheduleWordAssist();
        return;
    }
    els.output.value = `${els.output.value.slice(0, data.start)}${suggestion.text}${els.output.value.slice(data.end)}`;
    scheduleWordAssist();
    state.latestSentenceAssist = null;
}

async function fetchSentenceAssist() {
    if (!els.output.value.trim()) {
        clearSentenceSuggestions("No text to check.");
        return;
    }
    clearSentenceSuggestions("Checking sentence...");
    try {
        const response = await fetch(apiUrl("/api/sentence_assist"), {
            method: "POST",
            headers: telegramHeaders(),
            body: JSON.stringify({ text: els.output.value }),
        });
        const data = await response.json();
        state.latestSentenceAssist = data;
        renderSentenceSuggestions(data);
    } catch (error) {
        clearSentenceSuggestions("Sentence assist unavailable.");
    }
}

function renderSentenceSuggestions(data) {
    els.sentenceSuggestions.innerHTML = "";
    if (!data.suggestions || data.suggestions.length === 0) {
        clearSentenceSuggestions("Sentence looks acceptable.");
        return;
    }

    for (const suggestion of data.suggestions) {
        const row = document.createElement("div");
        row.className = "sentence-item";
        const text = document.createElement("div");
        const title = document.createElement("strong");
        title.textContent = suggestion.title || "Suggestion";
        const detail = document.createElement("span");
        detail.textContent = suggestion.replacement
            ? `${suggestion.original} -> ${suggestion.replacement}`
            : suggestion.message || "";
        const button = document.createElement("button");
        button.type = "button";
        button.className = suggestion.safe ? "" : "soft-btn";
        button.textContent = "Apply";
        button.addEventListener("click", () => applySentenceSuggestion(suggestion));
        text.appendChild(title);
        text.appendChild(detail);
        row.appendChild(text);
        row.appendChild(button);
        els.sentenceSuggestions.appendChild(row);
    }
}

function clearSentenceSuggestions(text) {
    els.sentenceSuggestions.innerHTML = "";
    const hint = document.createElement("span");
    hint.className = "hint";
    hint.textContent = text;
    els.sentenceSuggestions.appendChild(hint);
}

async function applySafeSentenceFixes() {
    if (!state.latestSentenceAssist) {
        await fetchSentenceAssist();
    }
    if (!state.latestSentenceAssist || !state.latestSentenceAssist.has_safe_fixes) {
        clearSentenceSuggestions("No safe fixes to apply.");
        return;
    }
    els.output.value = state.latestSentenceAssist.safe_text;
    scheduleWordAssist();
    fetchSentenceAssist();
}

function applySentenceSuggestion(suggestion) {
    if (suggestion.corrected_text) {
        els.output.value = suggestion.corrected_text;
    } else if (Number.isInteger(suggestion.start) && Number.isInteger(suggestion.end)) {
        const current = els.output.value.slice(suggestion.start, suggestion.end);
        if (suggestion.original && current !== suggestion.original) {
            fetchSentenceAssist();
            return;
        }
        els.output.value = `${els.output.value.slice(0, suggestion.start)}${suggestion.replacement || ""}${els.output.value.slice(suggestion.end)}`;
    }
    scheduleWordAssist();
    fetchSentenceAssist();
}

async function testApi() {
    try {
        const response = await fetch(apiUrl("/api/health"), { headers: telegramHeaders("GET") });
        const data = await response.json();
        setStatus(data.status === "ok" ? "Backend connected." : "Backend returned an error.");
    } catch (error) {
        setStatus("Could not reach backend.");
    }
}

function resizeCanvasToVideo() {
    const width = els.video.videoWidth || els.video.clientWidth;
    const height = els.video.videoHeight || els.video.clientHeight;
    if (els.canvas.width !== width || els.canvas.height !== height) {
        els.canvas.width = width;
        els.canvas.height = height;
    }
}

function loadApiBase() {
    return (localStorage.getItem(API_BASE_KEY) || window.ASL_API_BASE || "").replace(/\/$/, "");
}

function apiUrl(path) {
    return `${state.apiBase}${path}`;
}

function telegramHeaders(method = "POST") {
    const headers = method === "GET" ? {} : { "Content-Type": "application/json" };
    const initData = window.Telegram && window.Telegram.WebApp && window.Telegram.WebApp.initData;
    if (initData) headers["X-Telegram-Init-Data"] = initData;
    return headers;
}

function setStatus(text) {
    els.connection.textContent = text;
}
