/**
 * Image-conditioned hand refinement for COCO-WholeBody observations.
 * Model: OpenCV Zoo's Apache-2.0 MediaPipe hand landmark FP32 ONNX.
 * The model's presence score is NOT a per-joint visibility measurement.
 */
export const HAND_MODEL = Object.freeze({
  id: 'opencv/mediapipe-hand-fp32',
  url: 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/handpose_estimation_mediapipe/handpose_estimation_mediapipe_2023feb.onnx',
  sha256: 'db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c',
  bytes: 4099621,
  license: 'Apache-2.0',
  inputShape: [1, 224, 224, 3],
});

const SIZE = 224;
const PALM_IDS = [0, 5, 9, 13, 17, 1, 2];
const HANDS = Object.freeze({
  left: { offset: 91, wrist: 9, elbow: 7 },
  right: { offset: 112, wrist: 10, elbow: 8 },
});
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const finitePoint = (point) => point && Number.isFinite(point.u) && Number.isFinite(point.v);
const pointScore = (point) => Number.isFinite(point?.score) ? point.score : 0;
const pointXY = (point) => [point.u, point.v];
const norm = (value) => Math.hypot(value[0], value[1]);

function abortIfRequested(signal) {
  if (signal?.aborted) throw signal.reason || new DOMException('Aborted', 'AbortError');
}

/**
 * Construct the same rotated-palm crop convention as OpenCV MPHandPose:
 * palm bounds shifted by -0.4 palm-heights, enlarged by 3, then square padded.
 * The affine transform preserves off-screen padding instead of clipping its
 * coordinate system; no image mirroring or handedness-based ID reassignment.
 */
export function estimateHandRoi(points, side, imageWidth, imageHeight, options = {}) {
  const group = HANDS[side];
  if (!group) throw new Error(`Unknown hand side: ${side}`);
  if (!(imageWidth > 0 && imageHeight > 0)) return null;
  const minScore = options.minPointScore ?? 0.15;
  const valid = (point) => finitePoint(point) && pointScore(point) >= minScore
    && point.u >= 0 && point.v >= 0 && point.u < imageWidth && point.v < imageHeight;
  const hand = Array.from({ length: 21 }, (_, index) => points[group.offset + index]);
  const bodyWrist = points[group.wrist];
  const wrist = valid(hand[0]) ? hand[0] : valid(bodyWrist) ? bodyWrist : null;
  const palm = PALM_IDS.map((index) => hand[index]).filter(valid);
  const middle = valid(hand[9]) ? hand[9] : null;
  let center;
  let axisX;
  let axisY;
  let span;
  let source;

  if (wrist && middle && palm.length >= 4) {
    const direction = [wrist.u - middle.u, wrist.v - middle.v];
    const length = norm(direction);
    if (length < 2) return null;
    axisY = direction.map((component) => component / length);
    axisX = [axisY[1], -axisY[0]];
    const projected = palm.map((point) => [dot(pointXY(point), axisX), dot(pointXY(point), axisY)]);
    const minX = Math.min(...projected.map((point) => point[0]));
    const maxX = Math.max(...projected.map((point) => point[0]));
    const minY = Math.min(...projected.map((point) => point[1]));
    const maxY = Math.max(...projected.map((point) => point[1]));
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2 - 0.4 * (maxY - minY);
    center = [axisX[0] * centerX + axisY[0] * centerY, axisX[1] * centerX + axisY[1] * centerY];
    span = Math.max(maxX - minX, maxY - minY) * 3;
    source = 'wholebody-palm';
  } else if (valid(bodyWrist) && valid(points[group.elbow]) && options.allowForearmFallback === true) {
    // Recovery proposal only: the hand model must subsequently confirm presence.
    const elbow = points[group.elbow];
    const direction = [bodyWrist.u - elbow.u, bodyWrist.v - elbow.v];
    const length = norm(direction);
    if (length < 12) return null;
    axisY = direction.map((component) => -component / length);
    axisX = [axisY[1], -axisY[0]];
    span = length * 0.95;
    center = [bodyWrist.u + direction[0] * 0.25, bodyWrist.v + direction[1] * 0.25];
    source = 'forearm-proposal';
  } else {
    return null;
  }

  if (!Number.isFinite(span) || span < (options.minRoiPixels ?? 12) || span > Math.hypot(imageWidth, imageHeight) * 0.8) return null;
  const radius = span * Math.SQRT1_2;
  if (center[0] + radius <= 0 || center[1] + radius <= 0 || center[0] - radius >= imageWidth || center[1] - radius >= imageHeight) return null;
  return { side, center, axisX, axisY, size: span, source, imageWidth, imageHeight };
}

export function handCropToImage(roi, x, y) {
  const localX = (x / SIZE - 0.5) * roi.size;
  const localY = (y / SIZE - 0.5) * roi.size;
  return [
    roi.center[0] + roi.axisX[0] * localX + roi.axisY[0] * localY,
    roi.center[1] + roi.axisX[1] * localX + roi.axisY[1] * localY,
  ];
}

function readTensor(outputs, name, length) {
  const tensor = outputs[name];
  if (!tensor || tensor.data?.length !== length) throw new Error(`Hand model output ${name} must contain ${length} values.`);
  return tensor.data;
}

/** Decode the published model's explicit output contract, without guessing axes. */
export function decodeHandOutputs(outputs, roi, outputNames = {}) {
  const screen = readTensor(outputs, outputNames.screen || 'Identity', 63);
  const presence = Number(readTensor(outputs, outputNames.presence || 'Identity_1', 1)[0]);
  const handedness = Number(readTensor(outputs, outputNames.handedness || 'Identity_2', 1)[0]);
  const world = readTensor(outputs, outputNames.world || 'Identity_3', 63);
  if (![...screen, ...world, presence, handedness].every(Number.isFinite)) throw new Error('Hand model returned non-finite coordinates.');
  const landmarks = Array.from({ length: 21 }, (_, index) => {
    const pixel = handCropToImage(roi, screen[index * 3], screen[index * 3 + 1]);
    const wx = world[index * 3] - world[0];
    const wy = world[index * 3 + 1] - world[1];
    const wz = world[index * 3 + 2] - world[2];
    return {
      u: pixel[0],
      v: pixel[1],
      localDepthPixels: (screen[index * 3 + 2] - screen[2]) * roi.size / SIZE,
      // Retain raw hand-model axes for diagnostics; do not mix this learned
      // local scale/depth into the whole-body camera or world calibration.
      handLocalModelXYZm: [wx, wy, wz],
    };
  });
  return { landmarks, presence: clamp(presence, 0, 1), rawHandedness: clamp(handedness, 0, 1), roi };
}

function makeCanvas() {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(SIZE, SIZE);
  if (typeof document !== 'undefined') {
    const canvas = document.createElement('canvas');
    canvas.width = SIZE;
    canvas.height = SIZE;
    return canvas;
  }
  throw new Error('Hand refinement needs a canvasFactory in this environment.');
}

function fillInput(canvas, context, image, roi, data) {
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.fillStyle = '#000';
  context.fillRect(0, 0, SIZE, SIZE);
  const scale = SIZE / roi.size;
  context.setTransform(
    roi.axisX[0] * scale,
    roi.axisY[0] * scale,
    roi.axisX[1] * scale,
    roi.axisY[1] * scale,
    SIZE / 2 - dot(roi.axisX, roi.center) * scale,
    SIZE / 2 - dot(roi.axisY, roi.center) * scale,
  );
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';
  context.drawImage(image, 0, 0, roi.imageWidth, roi.imageHeight);
  context.setTransform(1, 0, 0, 1, 0, 0);
  const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let pixel = 0, target = 0; pixel < rgba.length; pixel += 4) {
    data[target++] = rgba[pixel] / 255;
    data[target++] = rgba[pixel + 1] / 255;
    data[target++] = rgba[pixel + 2] / 255;
  }
}

async function fetchModelBytes({ modelBytes, modelUrl, expectedSha256, signal, onProgress }) {
  let bytes;
  if (modelBytes) {
    bytes = modelBytes instanceof Uint8Array ? modelBytes : new Uint8Array(modelBytes);
  } else {
    const response = await fetch(modelUrl, { signal });
    if (!response.ok) throw new Error(`Hand model download failed: HTTP ${response.status}`);
    const total = Number(response.headers.get('content-length')) || HAND_MODEL.bytes;
    if (response.body && onProgress) {
      const reader = response.body.getReader();
      const chunks = [];
      let received = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        onProgress({ received, total });
      }
      bytes = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    } else {
      bytes = new Uint8Array(await response.arrayBuffer());
    }
  }
  abortIfRequested(signal);
  if (expectedSha256) {
    if (!globalThis.crypto?.subtle) throw new Error('Model verification requires a secure context.');
    const hash = await crypto.subtle.digest('SHA-256', bytes);
    const actual = Array.from(new Uint8Array(hash), (value) => value.toString(16).padStart(2, '0')).join('');
    if (actual !== expectedSha256.toLowerCase()) throw new Error('Hand model SHA-256 does not match the selected model.');
  }
  return bytes;
}

/**
 * Supply an existing ORT session, or let this function create the published
 * model's session. No implicit WASM fallback: the caller selects the backend.
 * refine() returns a new points array and never mutates its input observations.
 */
export async function createHandRefiner(options) {
  const { ort, signal } = options || {};
  if (!ort?.Tensor || !ort?.InferenceSession) throw new Error('Pass the same ONNX Runtime instance used by the body estimator.');
  const ownsSession = !options.session;
  const modelUrl = options.modelUrl || HAND_MODEL.url;
  const expectedSha256 = options.expectedSha256 ?? (modelUrl === HAND_MODEL.url ? HAND_MODEL.sha256 : null);
  const session = options.session || await ort.InferenceSession.create(
    await fetchModelBytes({ ...options, modelUrl, expectedSha256 }),
    { executionProviders: [options.backend || 'webgpu'], graphOptimizationLevel: 'all' },
  );
  const inputName = options.inputName || session.inputNames[0];
  if (!inputName || session.inputNames.length !== 1) throw new Error('Expected a single-input hand landmark model.');
  const canvas = (options.canvasFactory || makeCanvas)();
  canvas.width = SIZE;
  canvas.height = SIZE;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  if (!context) throw new Error('Unable to create a 2D hand crop canvas.');
  const input = new Float32Array(SIZE * SIZE * 3);
  let processing = false;
  let disposed = false;

  return {
    model: HAND_MODEL,
    async refine(image, points, frameOptions = {}) {
      if (disposed) throw new Error('Hand refiner has been disposed.');
      if (processing) throw new Error('Hand refinement is already processing a frame.');
      if (!Array.isArray(points) || points.length < 133) throw new Error('Expected 133 COCO-WholeBody observations.');
      const frameSignal = frameOptions.signal || signal;
      abortIfRequested(frameSignal);
      const started = performance.now();
      const imageWidth = frameOptions.imageWidth || image.videoWidth || image.naturalWidth || image.width;
      const imageHeight = frameOptions.imageHeight || image.videoHeight || image.naturalHeight || image.height;
      const refined = points.map((point) => point ? { ...point } : point);
      const hands = [];
      processing = true;
      try {
        for (const side of frameOptions.sides || ['left', 'right']) {
          abortIfRequested(frameSignal);
          const roi = estimateHandRoi(points, side, imageWidth, imageHeight, { ...options, ...frameOptions });
          if (!roi) { hands.push({ side, accepted: false, reason: 'no-reliable-roi' }); continue; }
          fillInput(canvas, context, image, roi, input);
          const tensor = new ort.Tensor('float32', input, [1, SIZE, SIZE, 3]);
          let outputs;
          let result;
          try {
            outputs = await session.run({ [inputName]: tensor });
            result = decodeHandOutputs(outputs, roi, options.outputNames);
          } finally {
            tensor.dispose?.();
            for (const output of Object.values(outputs || {})) output.dispose?.();
          }
          abortIfRequested(frameSignal);
          const group = HANDS[side];
          const anchor = finitePoint(points[group.wrist]) && pointScore(points[group.wrist]) >= (options.minPointScore ?? 0.15)
            ? points[group.wrist] : points[group.offset];
          const wristError = finitePoint(anchor) ? Math.hypot(result.landmarks[0].u - anchor.u, result.landmarks[0].v - anchor.v) : 0;
          const presenceThreshold = frameOptions.minPresence ?? options.minPresence ?? 0.65;
          let reason = null;
          if (result.presence < presenceThreshold) reason = 'low-hand-presence';
          else if (wristError > Math.max(18, roi.size * 0.3)) reason = 'wrist-disagreement';
          else if (result.landmarks.some((point) => Math.abs(point.u - roi.center[0]) > roi.size || Math.abs(point.v - roi.center[1]) > roi.size)) reason = 'invalid-hand-extent';
          else if (result.landmarks.some((point) => point.u < 0 || point.v < 0 || point.u >= imageWidth || point.v >= imageHeight)) reason = 'outside-image';
          if (reason) { hands.push({ side, accepted: false, reason, presence: result.presence, wristError, roi }); continue; }

          for (let index = 0; index < 21; index++) {
            const original = refined[group.offset + index] || {};
            const predicted = result.landmarks[index];
            const point = {
              ...original,
              ...predicted,
              id: original.id ?? group.offset + index,
              // Presence answers whether this crop contains a hand, not which
              // finger is visible. Do not increase per-joint confidence with it.
              score: Math.min(clamp(pointScore(original), 0, 1), result.presence),
              source: 'hand-roi',
              visibility: 'unknown',
              observable: original.observable === false ? false : null,
              handPresence: result.presence,
            };
            refined[group.offset + index] = point;
          }
          hands.push({ side, accepted: true, presence: result.presence, rawHandedness: result.rawHandedness, wristError, roi });
        }
        return { points: refined, hands, ms: performance.now() - started };
      } finally {
        processing = false;
      }
    },
    async dispose() {
      if (processing) throw new Error('Wait for the current hand frame before disposing its session.');
      if (disposed) return;
      disposed = true;
      if (ownsSession) await session.release();
    },
  };
}
