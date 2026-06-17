export const INPUT_SIZE = 63;
export const RICH_FEATURE_SIZE = 335;

const WRIST = 0;
const THUMB = [1, 2, 3, 4];
const INDEX = [5, 6, 7, 8];
const MIDDLE = [9, 10, 11, 12];
const RING = [13, 14, 15, 16];
const PINKY = [17, 18, 19, 20];
const FINGERS = [THUMB, INDEX, MIDDLE, RING, PINKY];
const FINGERTIPS = [4, 8, 12, 16, 20];
const MCPS = [1, 5, 9, 13, 17];

const HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
];

const JOINT_TRIPLES = [
    [1, 2, 3], [2, 3, 4],
    [0, 5, 6], [5, 6, 7], [6, 7, 8],
    [0, 9, 10], [9, 10, 11], [10, 11, 12],
    [0, 13, 14], [13, 14, 15], [14, 15, 16],
    [0, 17, 18], [17, 18, 19], [18, 19, 20],
];

const BASE_SHAPE_FEATURE_SIZE =
    INPUT_SIZE +
    HAND_EDGES.length * 3 +
    HAND_EDGES.length +
    JOINT_TRIPLES.length +
    5 + 5 + 5 +
    4 + 4 +
    5 +
    3;

export function normalizeFrame(frame) {
    if (!Array.isArray(frame) && !(frame instanceof Float32Array)) {
        return zeros(INPUT_SIZE);
    }
    if (frame.length !== INPUT_SIZE || !isPresentFrame(frame)) {
        return zeros(INPUT_SIZE);
    }

    const coords = reshapeFrame(frame);
    const wrist = coords[WRIST];
    let scale = 0;
    const centered = coords.map(([x, y, z]) => {
        const point = [x - wrist[0], y - wrist[1], z - wrist[2]];
        scale = Math.max(scale, vectorNorm(point));
        return point;
    });
    const divisor = scale > 1e-6 ? scale : 1;
    return centered.flatMap(([x, y, z]) => [x / divisor, y / divisor, z / divisor]);
}

export function extractRichSequenceFeatures(sequence) {
    const normalized = sequence.map((frame) => normalizeFrame(frame));
    const shapeFeatures = normalized.map((frame) => extractFrameShapeFeatures(frame));

    const velocities = normalized.map(() => zeros(INPUT_SIZE));
    const accelerations = normalized.map(() => zeros(INPUT_SIZE));

    for (let index = 1; index < normalized.length; index += 1) {
        if (isPresentFrame(normalized[index]) && isPresentFrame(normalized[index - 1])) {
            velocities[index] = subtractFrames(normalized[index], normalized[index - 1]);
        }
        if (
            index >= 2 &&
            isPresentFrame(normalized[index]) &&
            isPresentFrame(normalized[index - 1]) &&
            isPresentFrame(normalized[index - 2])
        ) {
            accelerations[index] = subtractFrames(velocities[index], velocities[index - 1]);
        }
    }

    const rows = [];
    for (let index = 0; index < normalized.length; index += 1) {
        const landmarkSpeeds = [];
        for (let landmark = 0; landmark < 21; landmark += 1) {
            const offset = landmark * 3;
            landmarkSpeeds.push(vectorNorm([
                velocities[index][offset],
                velocities[index][offset + 1],
                velocities[index][offset + 2],
            ]));
        }
        rows.push([
            ...shapeFeatures[index],
            ...velocities[index],
            ...landmarkSpeeds,
            ...accelerations[index],
        ]);
    }
    return rows;
}

function extractFrameShapeFeatures(frame) {
    if (frame.length !== INPUT_SIZE || !isPresentFrame(frame)) {
        return zeros(BASE_SHAPE_FEATURE_SIZE);
    }

    const coords = reshapeFrame(frame);
    const boneVectors = HAND_EDGES.map(([parent, child]) => subtractPoint(coords[child], coords[parent]));
    const boneLengths = boneVectors.map(vectorNorm);
    const jointAngles = JOINT_TRIPLES.map(([first, center, last]) => angleAt(coords, first, center, last));

    const palmCenter = meanPoints([coords[0], coords[5], coords[9], coords[13], coords[17]]);
    const fingertipToWrist = FINGERTIPS.map((tip) => vectorNorm(subtractPoint(coords[tip], coords[WRIST])));
    const fingertipToPalm = FINGERTIPS.map((tip) => vectorNorm(subtractPoint(coords[tip], palmCenter)));
    const mcpToTip = FINGERTIPS.map((tip, index) => vectorNorm(subtractPoint(coords[tip], coords[MCPS[index]])));
    const thumbToFingers = [8, 12, 16, 20].map((tip) => vectorNorm(subtractPoint(coords[tip], coords[4])));
    const adjacentTips = [[4, 8], [8, 12], [12, 16], [16, 20]]
        .map(([left, right]) => vectorNorm(subtractPoint(coords[right], coords[left])));
    const fingerCurls = FINGERS.map((finger, index) => {
        const tip = FINGERTIPS[index];
        return vectorNorm(subtractPoint(coords[tip], coords[WRIST])) / fingerPathLength(coords, finger);
    });
    const palmNormal = safeUnit(crossProduct(
        subtractPoint(coords[5], coords[0]),
        subtractPoint(coords[17], coords[0])
    ));

    return [
        ...coords.flat(),
        ...boneVectors.flat(),
        ...boneLengths,
        ...jointAngles,
        ...fingertipToWrist,
        ...fingertipToPalm,
        ...mcpToTip,
        ...thumbToFingers,
        ...adjacentTips,
        ...fingerCurls,
        ...palmNormal,
    ];
}

export function frameDelta(left, right) {
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

export function handPresenceRatio(sequence) {
    if (!sequence.length) return 0;
    const present = sequence.filter(isPresentFrame).length;
    return present / sequence.length;
}

export function averageFrameDelta(sequence) {
    if (sequence.length < 2) return 0;
    let total = 0;
    for (let index = 1; index < sequence.length; index += 1) {
        total += frameDelta(sequence[index - 1], sequence[index]);
    }
    return total / (sequence.length - 1);
}

function isPresentFrame(frame) {
    return frame.some((value) => Math.abs(value) > 1e-6);
}

function reshapeFrame(frame) {
    const coords = [];
    for (let index = 0; index < 21; index += 1) {
        coords.push([frame[index * 3], frame[index * 3 + 1], frame[index * 3 + 2]]);
    }
    return coords;
}

function subtractFrames(left, right) {
    const result = new Array(left.length);
    for (let index = 0; index < left.length; index += 1) {
        result[index] = left[index] - right[index];
    }
    return result;
}

function subtractPoint(left, right) {
    return [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
}

function vectorNorm(values) {
    return Math.hypot(...values);
}

function dotProduct(left, right) {
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function crossProduct(left, right) {
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ];
}

function safeUnit(vector) {
    const norm = vectorNorm(vector);
    if (norm < 1e-6) return [0, 0, 0];
    return vector.map((value) => value / norm);
}

function angleAt(coords, first, center, last) {
    const left = subtractPoint(coords[first], coords[center]);
    const right = subtractPoint(coords[last], coords[center]);
    const denom = vectorNorm(left) * vectorNorm(right);
    if (denom < 1e-6) return 0;
    const cosine = clamp(dotProduct(left, right) / denom, -1, 1);
    return Math.acos(cosine) / Math.PI;
}

function fingerPathLength(coords, finger) {
    const joints = [WRIST, ...finger];
    let length = 0;
    for (let index = 1; index < joints.length; index += 1) {
        length += vectorNorm(subtractPoint(coords[joints[index]], coords[joints[index - 1]]));
    }
    return Math.max(length, 1e-6);
}

function meanPoints(points) {
    const total = [0, 0, 0];
    for (const point of points) {
        total[0] += point[0];
        total[1] += point[1];
        total[2] += point[2];
    }
    return total.map((value) => value / points.length);
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function zeros(length) {
    return Array.from({ length }, () => 0);
}
