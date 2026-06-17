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
const TEXT_PANEL_COLLAPSED_KEY = "aslTextPanelCollapsed";
const MAX_FRAMES = 30;
const MIN_FRAMES = 8;
const PREDICT_INTERVAL_MS = 260;
const NO_HAND_RESET_FRAMES = 4;
const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const WEAK_SET = ["U", "V", "R", "T", "A", "N", "M", "S", "K"];
const MOTION_SET = ["J", "Z"];

const els = {
    video: document.getElementById("camera-video"),
    canvas: document.getElementById("overlay-canvas"),
    connection: document.getElementById("connection-status"),
    signal: document.getElementById("signal-status"),
    prediction: document.getElementById("current-prediction"),
    confidence: document.getElementById("confidence-pill"),
    cameraStartPanel: document.getElementById("camera-start-panel"),
    startCameraButton: document.getElementById("btn-start-camera"),
    textPanel: document.getElementById("text-panel"),
    textPanelToggle: document.getElementById("btn-text-panel-toggle"),
    compactOutput: document.getElementById("compact-output-text"),
    textPanelToggleAction: document.getElementById("text-panel-toggle-action"),
    translatorTab: document.getElementById("translator-tab"),
    gameTab: document.getElementById("game-tab"),
    translatorView: document.getElementById("translator-view"),
    gameView: document.getElementById("game-view"),
    output: document.getElementById("output-text"),
    wordSuggestions: document.getElementById("word-suggestions"),
    sentenceSuggestions: document.getElementById("sentence-suggestions"),
    gameMode: document.getElementById("game-mode"),
    roundCount: document.getElementById("round-count"),
    targetLetter: document.getElementById("target-letter"),
    roundValue: document.getElementById("round-value"),
    timerValue: document.getElementById("timer-value"),
    modeValue: document.getElementById("mode-value"),
    lastTryValue: document.getElementById("last-try-value"),
    scoreValue: document.getElementById("score-value"),
    correctValue: document.getElementById("correct-value"),
    accuracyValue: document.getElementById("accuracy-value"),
    streakValue: document.getElementById("streak-value"),
    attemptsValue: document.getElementById("attempts-value"),
    bestTimeValue: document.getElementById("best-time-value"),
    targetClosenessValue: document.getElementById("target-closeness-value"),
    targetClosenessMeter: document.getElementById("target-closeness-meter"),
    gameTopPredictions: document.getElementById("game-top-predictions"),
    gameResult: document.getElementById("game-result"),
    historyList: document.getElementById("history-list"),
    settingsPanel: document.getElementById("settings-panel"),
    guideSheet: document.getElementById("guide-sheet"),
    apiBaseInput: document.getElementById("api-base-input"),
    skeletonToggle: document.getElementById("toggle-skeleton"),
    autoAppendToggle: document.getElementById("toggle-auto-append"),
};

const state = {
    appMode: "translator",
    appStarted: false,
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
    game: {
        active: false,
        targets: [],
        index: 0,
        score: 0,
        correct: 0,
        attempts: 0,
        streak: 0,
        bestTime: null,
        history: [],
        roundLocked: false,
        roundStartedAt: 0,
        gameStartedAt: 0,
        lastHandledLetter: "",
        lastHandledAt: 0,
    },
};

const ctx = els.canvas.getContext("2d");

bootstrapTelegram();
wireControls();
startWhenAppropriate();

function bootstrapTelegram() {
    const telegram = window.Telegram && window.Telegram.WebApp;
    if (!telegram) return;
    telegram.ready();
    telegram.expand();
    document.body.style.background = telegram.themeParams.bg_color || "";
}

function wireControls() {
    els.apiBaseInput.value = state.apiBase;
    setTextPanelCollapsed(shouldStartTextPanelCollapsed(), false);
    updateCompactOutput();

    els.textPanelToggle.addEventListener("click", () => {
        setTextPanelCollapsed(!isTextPanelCollapsed(), true);
    });
    els.startCameraButton.addEventListener("click", startFromUserGesture);
    els.translatorTab.addEventListener("click", () => setAppMode("translator"));
    els.gameTab.addEventListener("click", () => setAppMode("game"));

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
        updateCompactOutput();
        scheduleWordAssist();
        state.latestSentenceAssist = null;
    });
    document.getElementById("btn-clear").addEventListener("click", () => {
        els.output.value = "";
        state.lastCommitted = "";
        updateCompactOutput();
        clearWordSuggestions("Word suggestions appear here.");
        clearSentenceSuggestions("Check a phrase when it is ready.");
    });
    document.getElementById("btn-sentence").addEventListener("click", fetchSentenceAssist);
    document.getElementById("btn-apply-safe").addEventListener("click", applySafeSentenceFixes);
    els.output.addEventListener("input", () => {
        updateCompactOutput();
        scheduleWordAssist();
        state.latestSentenceAssist = null;
    });

    document.getElementById("btn-start-game").addEventListener("click", startGame);
    document.getElementById("btn-end-game").addEventListener("click", () => finishGame("manual"));
    document.getElementById("btn-retry").addEventListener("click", retryRound);
    document.getElementById("btn-skip").addEventListener("click", skipRound);
    els.gameMode.addEventListener("change", updateGameUi);
    els.roundCount.addEventListener("change", updateGameUi);
    window.setInterval(updateGameTimer, 100);
    updateGameUi();
}

async function startApp() {
    if (state.appStarted) return true;
    state.appStarted = true;
    try {
        await initCamera();
        await initHandLandmarker();
        await initLocalPredictor();
        setStatus(state.localPredictor ? `Local model ready (${state.localPredictor.provider}).` : "Camera ready. Set backend URL if prediction is unavailable.");
        els.cameraStartPanel.classList.add("hidden");
        requestAnimationFrame(processFrame);
        return true;
    } catch (error) {
        state.appStarted = false;
        setStatus(error.message || "Unable to start camera.");
        els.cameraStartPanel.classList.remove("hidden");
        els.startCameraButton.disabled = false;
        els.startCameraButton.textContent = "Retry Camera";
        return false;
    }
}

function startWhenAppropriate() {
    if (isTelegramMiniApp()) {
        setStatus("Tap Start Camera to begin in Telegram.");
        els.cameraStartPanel.classList.remove("hidden");
        return;
    }
    els.cameraStartPanel.classList.add("hidden");
    startApp();
}

async function startFromUserGesture() {
    els.startCameraButton.disabled = true;
    els.startCameraButton.textContent = "Starting...";
    const started = await startApp();
    if (!started) return;
    els.startCameraButton.disabled = false;
    els.startCameraButton.textContent = "Start Camera";
}

function isTelegramMiniApp() {
    const telegram = window.Telegram && window.Telegram.WebApp;
    return Boolean(telegram && telegram.initData);
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
    const isPhoneWidth = window.matchMedia("(max-width: 639px)").matches;
    const viewportAspect = window.innerWidth && window.innerHeight
        ? window.innerWidth / window.innerHeight
        : 9 / 16;
    const phoneAspect = Math.min(Math.max(viewportAspect, 0.46), 0.75);
    const videoConstraints = {
        facingMode: "user",
        width: { ideal: isPhoneWidth ? 720 : 1280 },
        height: { ideal: isPhoneWidth ? 1280 : 720 },
    };

    if (isPhoneWidth) {
        videoConstraints.aspectRatio = { ideal: phoneAspect };
    }

    const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
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

    if (!result.accepted || !result.label) return;

    if (result.label === state.candidate) {
        state.candidateCount += 1;
    } else {
        state.candidate = result.label;
        state.candidateCount = 1;
    }

    const commitCount = result.label === "X" ? 4 : 2;
    if (state.candidateCount < commitCount || Date.now() < state.cooldownUntil) return;

    if (state.appMode === "game") {
        handleGameLetter(result.label, result);
    } else if (els.autoAppendToggle.checked && result.label !== state.lastCommitted) {
        appendText(result.label);
        state.lastCommitted = result.label;
    }

    state.cooldownUntil = Date.now() + 950;
}

function appendText(value) {
    els.output.value += value;
    updateCompactOutput();
    if (!isTextPanelCollapsed()) {
        els.output.focus();
    }
    scheduleWordAssist();
    state.latestSentenceAssist = null;
}

function shouldStartTextPanelCollapsed() {
    const stored = localStorage.getItem(TEXT_PANEL_COLLAPSED_KEY);
    if (stored !== null) return stored === "true";
    return isTextPanelCollapsible();
}

function isTextPanelCollapsed() {
    return isTextPanelCollapsible() && els.textPanel.classList.contains("is-collapsed");
}

function isTextPanelCollapsible() {
    return window.matchMedia("(max-width: 839px)").matches;
}

function setTextPanelCollapsed(collapsed, persist) {
    els.textPanel.classList.toggle("is-collapsed", collapsed);
    els.textPanelToggle.setAttribute("aria-expanded", String(!collapsed));
    els.textPanelToggleAction.textContent = collapsed ? "Open" : "Hide";
    if (collapsed && els.textPanel.contains(document.activeElement)) {
        document.activeElement.blur();
    }
    if (persist) {
        localStorage.setItem(TEXT_PANEL_COLLAPSED_KEY, String(collapsed));
    }
}

function updateCompactOutput() {
    if (state.appMode === "game") {
        const game = state.game;
        const target = currentTarget();
        els.compactOutput.textContent = game.active
            ? `Target ${target} | Score ${game.score}`
            : "Sign Sprint ready.";
        return;
    }
    const value = els.output.value.replace(/\s+/g, " ").trim();
    els.compactOutput.textContent = value || "Stable letters appear here.";
}

function setAppMode(mode) {
    state.appMode = mode;
    const gameModeActive = mode === "game";
    if (!gameModeActive && state.game.active) {
        finishGame("mode_switch");
    }
    els.translatorTab.classList.toggle("active", !gameModeActive);
    els.gameTab.classList.toggle("active", gameModeActive);
    els.translatorView.classList.toggle("active", !gameModeActive);
    els.gameView.classList.toggle("active", gameModeActive);
    resetRecognitionBuffer();
    if (gameModeActive) {
        setTextPanelCollapsed(false, true);
    }
    updateCompactOutput();
    updateGameUi();
}

function resetRecognitionBuffer() {
    state.frames = [];
    state.normalizedFrames = [];
    state.candidate = "";
    state.candidateCount = 0;
    state.lastCommitted = "";
    state.cooldownUntil = 0;
}

function shuffle(array) {
    const copy = [...array];
    for (let index = copy.length - 1; index > 0; index -= 1) {
        const targetIndex = Math.floor(Math.random() * (index + 1));
        [copy[index], copy[targetIndex]] = [copy[targetIndex], copy[index]];
    }
    return copy;
}

function buildTargets(mode, count) {
    if (mode === "alphabet") return LETTERS.slice(0, count);
    const source = mode === "weak" ? WEAK_SET : mode === "motion" ? MOTION_SET : LETTERS;
    const targets = [];
    while (targets.length < count) {
        for (const letter of shuffle(source)) {
            if (targets.length >= count) break;
            if (targets[targets.length - 1] === letter && source.length > 1) continue;
            targets.push(letter);
        }
    }
    return targets;
}

function currentTarget() {
    return state.game.targets[state.game.index] || "";
}

function updateGameUi() {
    const game = state.game;
    const target = currentTarget();
    els.targetLetter.textContent = game.active ? target : "--";
    els.roundValue.textContent = `${game.active ? Math.min(game.index + 1, game.targets.length) : 0}/${game.targets.length}`;
    els.scoreValue.textContent = game.score;
    els.correctValue.textContent = game.correct;
    els.attemptsValue.textContent = game.attempts;
    els.streakValue.textContent = game.streak;
    els.accuracyValue.textContent = game.attempts ? `${Math.round((game.correct / game.attempts) * 100)}%` : "--";
    els.bestTimeValue.textContent = game.bestTime === null ? "--" : `${game.bestTime.toFixed(1)}s`;
    els.modeValue.textContent = game.active ? els.gameMode.options[els.gameMode.selectedIndex].text : "Idle";
    updateCompactOutput();
}

function updateGameTimer() {
    if (!state.game.active || !state.game.roundStartedAt) {
        els.timerValue.textContent = "0.0s";
        return;
    }
    els.timerValue.textContent = `${((performance.now() - state.game.roundStartedAt) / 1000).toFixed(1)}s`;
}

function setGameResult(text, status = "") {
    els.gameResult.textContent = text;
    els.gameResult.classList.toggle("correct", status === "correct");
    els.gameResult.classList.toggle("wrong", status === "wrong");
}

function clamp01(value) {
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(1, value));
}

function formatPercent(value) {
    return `${Math.round(clamp01(value) * 100)}%`;
}

function topPredictionsFromResult(result) {
    return Array.isArray(result.top_predictions) ? result.top_predictions : [];
}

function getTargetConfidence(target, predicted, result) {
    const topMatch = topPredictionsFromResult(result).find((item) => item.label === target);
    if (topMatch && typeof topMatch.confidence === "number") return clamp01(topMatch.confidence);
    if (target === predicted) return clamp01(result.confidence || 0);
    return 0;
}

function getTargetRank(target, result) {
    const topIndex = topPredictionsFromResult(result).findIndex((item) => item.label === target);
    return topIndex >= 0 ? topIndex + 1 : null;
}

function buildGuessFeedback(target, predicted, result) {
    const predictedConfidence = clamp01(result.confidence || 0);
    const targetConfidence = getTargetConfidence(target, predicted, result);
    return {
        targetConfidence,
        confidenceGap: Math.max(0, predictedConfidence - targetConfidence),
        targetRank: getTargetRank(target, result),
        topPredictions: topPredictionsFromResult(result),
    };
}

function resetGuessFeedback() {
    els.targetClosenessValue.textContent = "--";
    els.targetClosenessMeter.style.width = "0%";
    els.targetClosenessMeter.classList.remove("close", "on-target");
    els.gameTopPredictions.textContent = "Top guesses appear after each attempt.";
}

function updateGuessFeedback(target, feedback, correct) {
    const closeness = feedback.targetConfidence;
    const rankText = feedback.targetRank ? `rank #${feedback.targetRank}` : "not ranked";
    els.targetClosenessValue.textContent = `${formatPercent(closeness)} (${rankText})`;
    els.targetClosenessMeter.style.width = `${Math.round(closeness * 100)}%`;
    els.targetClosenessMeter.classList.toggle("on-target", correct || closeness >= 0.8);
    els.targetClosenessMeter.classList.toggle("close", !correct && closeness >= 0.35 && closeness < 0.8);

    els.gameTopPredictions.innerHTML = "";
    const renderedLabels = new Set();
    for (const item of feedback.topPredictions.slice(0, 3)) {
        renderedLabels.add(item.label);
        const chip = document.createElement("span");
        chip.className = `prediction-chip ${item.label === target ? "target" : ""}`;
        chip.textContent = `${item.label} ${formatPercent(item.confidence || 0)}`;
        els.gameTopPredictions.appendChild(chip);
    }
    if (!renderedLabels.has(target)) {
        const chip = document.createElement("span");
        chip.className = "prediction-chip target";
        chip.textContent = `${target} ${formatPercent(feedback.targetConfidence)}`;
        els.gameTopPredictions.appendChild(chip);
    }
}

function renderHistory() {
    els.historyList.innerHTML = "";
    for (const item of state.game.history.slice(-8).reverse()) {
        const row = document.createElement("div");
        row.className = `history-item ${item.correct ? "correct" : "wrong"}`;
        const target = document.createElement("strong");
        const guess = document.createElement("span");
        const time = document.createElement("span");
        target.textContent = item.target;
        guess.textContent = `${item.predicted} | ${formatPercent(item.closeness || 0)}`;
        time.textContent = item.time ? `${item.time.toFixed(1)}s` : "--";
        row.append(target, guess, time);
        els.historyList.appendChild(row);
    }
}

function startGame() {
    const game = state.game;
    game.active = true;
    game.targets = buildTargets(els.gameMode.value, Number(els.roundCount.value));
    game.index = 0;
    game.score = 0;
    game.correct = 0;
    game.attempts = 0;
    game.streak = 0;
    game.bestTime = null;
    game.history = [];
    game.roundLocked = false;
    game.lastHandledLetter = "";
    game.lastHandledAt = 0;
    game.gameStartedAt = performance.now();
    game.roundStartedAt = performance.now();
    els.lastTryValue.textContent = "--";
    resetRecognitionBuffer();
    resetGuessFeedback();
    setGameResult("Sign the target letter.");
    renderHistory();
    updateGameUi();
}

function finishGame(reason = "complete") {
    if (!state.game.active) return;
    state.game.active = false;
    state.game.roundLocked = false;
    resetRecognitionBuffer();
    resetGuessFeedback();
    setGameResult(`Finished: ${state.game.correct}/${state.game.attempts} correct, score ${state.game.score}.`);
    updateGameUi();
    if (reason === "complete") {
        els.lastTryValue.textContent = "Done";
    }
}

function advanceRound() {
    if (!state.game.active) return;
    state.game.index += 1;
    if (state.game.index >= state.game.targets.length) {
        finishGame("complete");
        return;
    }
    state.game.roundLocked = false;
    state.game.roundStartedAt = performance.now();
    els.lastTryValue.textContent = "--";
    resetRecognitionBuffer();
    resetGuessFeedback();
    setGameResult("Next letter.");
    updateGameUi();
}

function handleGameLetter(predicted, result) {
    const game = state.game;
    if (!game.active || game.roundLocked || !predicted) return;

    const now = performance.now();
    if (game.lastHandledLetter === predicted && now - game.lastHandledAt < 900) return;
    game.lastHandledLetter = predicted;
    game.lastHandledAt = now;

    const target = currentTarget();
    const elapsed = (now - game.roundStartedAt) / 1000;
    const correct = predicted === target;
    const feedback = buildGuessFeedback(target, predicted, result);
    game.attempts += 1;
    els.lastTryValue.textContent = `${predicted} (${Math.round((result.confidence || 0) * 100)}%)`;
    updateGuessFeedback(target, feedback, correct);

    if (correct) {
        game.correct += 1;
        game.streak += 1;
        const speedBonus = Math.max(0, Math.round(60 - elapsed * 12));
        const streakBonus = Math.min(80, Math.max(0, (game.streak - 1) * 10));
        game.score += 100 + speedBonus + streakBonus;
        game.bestTime = game.bestTime === null ? elapsed : Math.min(game.bestTime, elapsed);
        game.roundLocked = true;
        setGameResult(`Correct: ${target} in ${elapsed.toFixed(1)}s.`, "correct");
        window.setTimeout(advanceRound, 850);
    } else {
        game.streak = 0;
        game.score = Math.max(0, game.score - 15);
        setGameResult(`Try again: expected ${target}, saw ${predicted}.`, "wrong");
    }

    game.history.push({
        target,
        predicted,
        correct,
        time: elapsed,
        confidence: result.confidence || 0,
        closeness: feedback.targetConfidence,
    });
    renderHistory();
    updateGameUi();
}

function retryRound() {
    if (!state.game.active) return;
    state.game.roundStartedAt = performance.now();
    state.game.roundLocked = false;
    els.lastTryValue.textContent = "--";
    resetRecognitionBuffer();
    resetGuessFeedback();
    setGameResult("Retry current letter.");
    updateGameUi();
}

function skipRound() {
    const game = state.game;
    if (!game.active) return;
    game.streak = 0;
    game.attempts += 1;
    game.history.push({
        target: currentTarget(),
        predicted: "Skipped",
        correct: false,
        time: 0,
        confidence: 0,
        closeness: 0,
    });
    renderHistory();
    advanceRound();
}

function flattenMirroredLandmarks(landmarks) {
    const frame = [];
    for (const landmark of landmarks) {
        frame.push(1 - landmark.x, landmark.y, landmark.z || 0);
    }
    return frame;
}

function modeHint() {
    if (state.appMode === "game" && state.game.active) {
        const target = currentTarget();
        if (els.gameMode.value === "motion" || MOTION_SET.includes(target)) return "motion";
        return "static";
    }
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
    updateCompactOutput();
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
    updateCompactOutput();
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
    updateCompactOutput();
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
