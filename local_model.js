import {
    RICH_FEATURE_SIZE,
    averageFrameDelta,
    extractRichSequenceFeatures,
    handPresenceRatio,
} from "./landmark_features.js";

const MODEL_PATH = "./models/asl_lstm_improved_features.onnx";
const LABELS_PATH = "./labels/asl_labels_improved_features.json";
const NO_PREDICTION_LABEL = "NOTHING";
const MOTION_LABELS = new Set(["J", "Z"]);
const DEFAULT_MOBILE_THRESHOLD = 0.82;
const DEFAULT_MARGIN_THRESHOLD = 0.04;

export async function createLocalPredictor() {
    const ort = window.ort;
    if (!ort) {
        throw new Error("ONNX Runtime Web was not loaded.");
    }

    if (ort.env && ort.env.wasm) {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
        // Single-threaded WASM avoids cross-origin isolation requirements in Telegram WebView.
        ort.env.wasm.numThreads = 1;
    }

    const labels = await fetch(LABELS_PATH).then((response) => {
        if (!response.ok) throw new Error("Could not load model labels.");
        return response.json();
    });

    const attempts = [
        { provider: "webgpu", options: { executionProviders: ["webgpu"] } },
        { provider: "wasm", options: { executionProviders: ["wasm"] } },
    ];

    let lastError = null;
    for (const attempt of attempts) {
        try {
            const session = await ort.InferenceSession.create(MODEL_PATH, attempt.options);
            return new LocalPredictor(ort, session, labels, attempt.provider);
        } catch (error) {
            lastError = error;
            console.warn(`ONNX ${attempt.provider} initialization failed`, error);
        }
    }

    throw lastError || new Error("Could not initialize local ASL model.");
}

class LocalPredictor {
    constructor(ort, session, labels, provider) {
        this.ort = ort;
        this.session = session;
        this.labels = labels.map((label) => String(label).toUpperCase());
        this.provider = provider;
        this.labelSet = new Set(this.labels);
    }

    async predict(sequence, modeHint = "auto") {
        const normalizedSequence = sequence.slice();
        const metrics = sequenceMetrics(normalizedSequence);
        if (normalizedSequence.length < 8) {
            return emptyResponse(normalizedSequence, metrics, "Need more frames.");
        }

        const features = extractRichSequenceFeatures(normalizedSequence);
        const flatFeatures = new Float32Array(features.length * RICH_FEATURE_SIZE);
        for (let row = 0; row < features.length; row += 1) {
            flatFeatures.set(features[row], row * RICH_FEATURE_SIZE);
        }

        const input = new this.ort.Tensor("float32", flatFeatures, [1, features.length, RICH_FEATURE_SIZE]);
        const output = await this.session.run({ features: input });
        const logits = Array.from(output.logits.data);
        const probabilities = softmax(logits);
        const topPredictions = topPredictionsFromProbabilities(probabilities, this.labels);
        const allowedLabels = allowedLabelsFor(this.labelSet, modeHint, metrics);
        const selected = bestAllowed(probabilities, this.labels, allowedLabels);

        if (!selected) {
            return emptyResponse(normalizedSequence, metrics, "No allowed label passed filtering.", topPredictions);
        }

        const accepted =
            selected.label !== NO_PREDICTION_LABEL &&
            selected.confidence >= DEFAULT_MOBILE_THRESHOLD &&
            selected.margin >= DEFAULT_MARGIN_THRESHOLD;

        return {
            status: "ok",
            accepted,
            label: accepted ? selected.label : "",
            raw_label: selected.label,
            display: accepted ? selected.label : "Reading...",
            confidence: round4(selected.confidence),
            margin: round4(selected.margin),
            threshold: DEFAULT_MOBILE_THRESHOLD,
            margin_threshold: DEFAULT_MARGIN_THRESHOLD,
            mode_hint: modeHint,
            provider: this.provider,
            allowed_labels: Array.from(allowedLabels).sort(),
            top_predictions: topPredictions,
            frame_count: normalizedSequence.length,
            ...metrics,
        };
    }
}

function allowedLabelsFor(labelSet, modeHint, metrics) {
    const mode = String(modeHint || "auto").toLowerCase();
    const staticLabels = difference(labelSet, new Set([...MOTION_LABELS, NO_PREDICTION_LABEL]));

    if (mode === "static") return staticLabels;
    if (mode === "motion") return new Set(MOTION_LABELS);
    if (metrics.hand_presence < 0.35) return new Set([NO_PREDICTION_LABEL]);
    if (metrics.motion_mean >= 0.050 || metrics.motion_max >= 0.140) {
        return difference(labelSet, new Set([NO_PREDICTION_LABEL]));
    }
    return staticLabels;
}

function sequenceMetrics(sequence) {
    let motionMax = 0;
    for (let index = 1; index < sequence.length; index += 1) {
        motionMax = Math.max(motionMax, frameDelta(sequence[index - 1], sequence[index]));
    }
    return {
        hand_presence: round4(handPresenceRatio(sequence)),
        motion_mean: round4(averageFrameDelta(sequence)),
        motion_max: round4(motionMax),
    };
}

function frameDelta(left, right) {
    let total = 0;
    for (let index = 0; index < 21; index += 1) {
        const offset = index * 3;
        total += Math.hypot(
            right[offset] - left[offset],
            right[offset + 1] - left[offset + 1],
            right[offset + 2] - left[offset + 2]
        );
    }
    return total / 21;
}

function bestAllowed(probabilities, labels, allowedLabels) {
    const ranked = probabilities
        .map((confidence, index) => ({ confidence, index, label: labels[index] }))
        .sort((left, right) => right.confidence - left.confidence);

    for (const candidate of ranked) {
        if (!allowedLabels.has(candidate.label)) continue;
        const second = ranked.length > 1 ? ranked[1].confidence : 0;
        return {
            label: candidate.label,
            confidence: candidate.confidence,
            margin: candidate.confidence - second,
        };
    }
    return null;
}

function topPredictionsFromProbabilities(probabilities, labels, limit = 5) {
    return probabilities
        .map((confidence, index) => ({ label: labels[index], confidence }))
        .sort((left, right) => right.confidence - left.confidence)
        .slice(0, limit)
        .map((prediction) => ({
            label: prediction.label,
            confidence: round4(prediction.confidence),
        }));
}

function softmax(logits) {
    const maxValue = Math.max(...logits);
    const exps = logits.map((value) => Math.exp(value - maxValue));
    const total = exps.reduce((sum, value) => sum + value, 0);
    return exps.map((value) => value / total);
}

function emptyResponse(sequence, metrics, message, topPredictions = []) {
    return {
        status: "ok",
        accepted: false,
        label: "",
        raw_label: "",
        display: sequence.length ? "Reading..." : "Resting...",
        confidence: 0,
        margin: 0,
        message,
        top_predictions: topPredictions,
        frame_count: sequence.length,
        ...metrics,
    };
}

function difference(left, right) {
    return new Set(Array.from(left).filter((item) => !right.has(item)));
}

function round4(value) {
    return Math.round(value * 10000) / 10000;
}
