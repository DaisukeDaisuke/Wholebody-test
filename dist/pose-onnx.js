import { DEFAULT_MODEL_BASE, loadManifest, resolveModel, fetchModel, sourceSize, normalizeRoi, makeCanvas, HAND_NAMES, bodyCoverage, suppressInconsistentFace, suppressAmbiguousBilateralOverlap } from './pose-models.js';

const ORT_VERSION = '1.27.0';
const ORT_ROOT = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist`;
const INPUT_W = 192;
const INPUT_H = 256;
const MEAN = [123.675, 116.28, 103.53];
const STD = [58.395, 57.12, 57.375];
const BODY_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot_index', 'left_small_toe', 'left_heel', 'right_foot_index', 'right_small_toe', 'right_heel'];
const KEYPOINT_NAMES = [...BODY_NAMES, ...Array.from({ length: 68 }, (_, i) => `face_${i}`), ...HAND_NAMES.map(name => `left_${name}`), ...HAND_NAMES.map(name => `right_${name}`)];
let ortPromise;

function aspectRoi(rectangle, padding = 1.25) {
  const centerX = rectangle.x + rectangle.width / 2;
  const centerY = rectangle.y + rectangle.height / 2;
  let width = rectangle.width;
  let height = rectangle.height;
  const ratio = INPUT_W / INPUT_H;
  if (width > height * ratio) height = width / ratio;
  else width = height * ratio;
  width *= padding;
  height *= padding;
  return { x: centerX - width / 2, y: centerY - height / 2, width, height };
}

export class PoseOnnxRuntime {
  constructor() { this.states = new Map(); this.busy = false; this.lifecycle = 0; }

  async initialize({ variant = 'dwpose-m', backend = 'GPU', onProgress, manifest, modelBaseUrl = DEFAULT_MODEL_BASE, confidence = 0.35 } = {}) {
    const lifecycle = ++this.lifecycle;
    this.variant = variant;
    this.confidence = confidence;
    this.manifest = await loadManifest({ manifest, modelBaseUrl, onProgress });
    const key = variant === 'rtmpose-s' || variant === 'rtmpose_s_wholebody' ? 'rtmpose_s_wholebody' : 'dwpose_m_wholebody';
    const entry = resolveModel(this.manifest, key, `${key}.onnx`, modelBaseUrl);
    ortPromise ||= import(`${ORT_ROOT}/ort.webgpu.min.mjs`);
    const [ort, bytes] = await Promise.all([ortPromise, fetchModel(entry, onProgress)]);
    if (lifecycle !== this.lifecycle) throw new Error('ONNX モデルの初期化は取り消されました。');
    this.ort = ort;
    ort.env.wasm.wasmPaths = `${ORT_ROOT}/`;
    ort.env.wasm.numThreads = 1;
    const cpu = backend === 'CPU' || backend === 'WASM';
    if (!cpu && !navigator.gpu) throw new Error('WebGPU が利用できません。対応した Chrome / Edge か、明示的な CPU 診断モードを使用してください。');
    this.backend = cpu ? 'ONNX Runtime WASM' : 'ONNX Runtime WebGPU';
    onProgress?.({ phase: 'initialize', file: entry.file, ratio: null });
    const session = await ort.InferenceSession.create(bytes, { executionProviders: [cpu ? 'wasm' : 'webgpu'], graphOptimizationLevel: 'all', logSeverityLevel: 3 });
    if (lifecycle !== this.lifecycle) { await session.release(); throw new Error('ONNX モデルの初期化は取り消されました。'); }
    this.session = session;
    if (this.session.inputNames.length !== 1) throw new Error(`ONNX の画像入力は 1 個である必要があります: ${this.session.inputNames.join(', ')}`);
    this.modelId = `${this.manifest.repository || 'RabbitDaisuke/Wholebody-Video-Tracking'}@${this.manifest.revision || 'unresolved'}/${entry.file}`;
    this.canvas = makeCanvas(INPUT_W, INPUT_H);
    this.drawing = this.canvas.getContext('2d', { alpha: false, willReadFrequently: true });
    this.inputData = new Float32Array(3 * INPUT_W * INPUT_H);
    onProgress?.({ phase: 'ready', ratio: 1, modelId: this.modelId, backend: this.backend });
    return { modelId: this.modelId, backend: this.backend, relative3D: false };
  }

  state(sourceId) {
    if (!this.states.has(sourceId)) this.states.set(sourceId, { roi: null, missed: 0, lastTime: null, frames: 0 });
    return this.states.get(sourceId);
  }

  async infer(source, { sourceId = 'video-1', mediaTimeMs = 0, frameId = 0, roi = null } = {}) {
    if (!this.session) throw new Error('ONNX モデルが未初期化です。');
    const imageSize = sourceSize(source);
    if (this.busy) return { sourceId, mediaTimeMs, frameId, imageSize, keypoints: [], hands: [], detected: false, skipped: true, skipReason: 'inference-busy', modelId: this.modelId, backend: this.backend, inferenceMs: 0, warnings: [] };
    this.busy = true;
    this.inferenceDone = new Promise(resolve => { this.finishInference = resolve; });
    const started = performance.now();
    let input;
    let outputs;
    try {
      const state = this.state(sourceId);
      if (state.lastTime != null && (mediaTimeMs < state.lastTime - 1 || mediaTimeMs - state.lastTime > 1000)) { state.roi = null; state.missed = 0; }
      state.lastTime = mediaTimeMs;
      const manual = normalizeRoi(roi, imageSize);
      const rectangle = manual || (state.missed < 3 ? state.roi : null) || { x: 0, y: 0, width: imageSize.width, height: imageSize.height };
      const crop = aspectRoi(rectangle, manual?.padding ?? 1.25);
      this.drawing.setTransform(1, 0, 0, 1, 0, 0);
      this.drawing.fillStyle = '#000';
      this.drawing.fillRect(0, 0, INPUT_W, INPUT_H);
      this.drawing.setTransform(INPUT_W / crop.width, 0, 0, INPUT_H / crop.height, -crop.x * INPUT_W / crop.width, -crop.y * INPUT_H / crop.height);
      this.drawing.drawImage(source, 0, 0);
      this.drawing.setTransform(1, 0, 0, 1, 0, 0);
      const rgba = this.drawing.getImageData(0, 0, INPUT_W, INPUT_H).data;
      const plane = INPUT_W * INPUT_H;
      for (let index = 0; index < plane; index++) {
        this.inputData[index] = (rgba[index * 4] - MEAN[0]) / STD[0];
        this.inputData[index + plane] = (rgba[index * 4 + 1] - MEAN[1]) / STD[1];
        this.inputData[index + plane * 2] = (rgba[index * 4 + 2] - MEAN[2]) / STD[2];
      }
      input = new this.ort.Tensor('float32', this.inputData, [1, 3, INPUT_H, INPUT_W]);
      const computeStarted = performance.now();
      outputs = await this.session.run({ [this.session.inputNames[0]]: input });
      const computeMs = performance.now() - computeStarted;
      const tensors = Object.values(outputs);
      const xOutput = tensors.find(value => value.dims.length === 3 && value.dims[1] === 133 && value.dims[2] === INPUT_W * 2);
      const yOutput = tensors.find(value => value.dims.length === 3 && value.dims[1] === 133 && value.dims[2] === INPUT_H * 2);
      if (!xOutput || !yOutput) throw new Error(`期待した SimCC 133点出力がありません: ${tensors.map(value => `[${value.dims}]`).join(' ')}`);
      const xData = xOutput.data;
      const yData = yOutput.data;
      const keypoints = [];
      for (let joint = 0; joint < 133; joint++) {
        let maxX = -Infinity, maxY = -Infinity, argX = 0, argY = 0;
        const xOffset = joint * INPUT_W * 2;
        const yOffset = joint * INPUT_H * 2;
        for (let x = 0; x < INPUT_W * 2; x++) if (xData[xOffset + x] > maxX) { maxX = xData[xOffset + x]; argX = x; }
        for (let y = 0; y < INPUT_H * 2; y++) if (yData[yOffset + y] > maxY) { maxY = yData[yOffset + y]; argY = y; }
        if (!Number.isFinite(maxX) || !Number.isFinite(maxY)) throw new Error(`ONNX の関節 ${joint} が非有限値です。`);
        const u = crop.x + (argX / 2) / INPUT_W * crop.width;
        const v = crop.y + (argY / 2) / INPUT_H * crop.height;
        const score = Math.max(0, Math.min(1, Math.min(maxX, maxY)));
        keypoints.push({ id: joint, name: KEYPOINT_NAMES[joint], u, v, score, score2D: score, score2DType: 'simcc-peak', score3D: null, modelId2D: this.modelId, modelId3D: null, observable: score >= this.confidence && u >= 0 && u < imageSize.width && v >= 0 && v < imageSize.height, relativeXYZm: null, relativeZm: null, source: 'observed-2d', observationTimeMs: mediaTimeMs });
      }
      const headConsistency = suppressInconsistentFace(keypoints, imageSize);
      const bilateralConsistency = suppressAmbiguousBilateralOverlap(keypoints, imageSize);
      const body = keypoints.slice(5, 23).filter(point => point.observable);
      const detected = body.length >= 4;
      state.missed = detected ? 0 : state.missed + 1;
      state.frames++;
      // Bounding boxes follow visible body/feet, never the 68 face outputs.
      // A partial body does not justify shrinking a tracked full-body crop.
      if (!manual && body.length >= 8 && keypoints[11].observable && keypoints[12].observable) {
        const x1 = Math.min(...body.map(point => point.u));
        const y1 = Math.min(...keypoints.slice(0, 23).filter(point => point.observable).map(point => point.v));
        const x2 = Math.max(...body.map(point => point.u));
        const y2 = Math.max(...body.map(point => point.v));
        if (x2 - x1 > 20 && y2 - y1 > 40) {
          const next = { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
          if (state.roi) for (const key of ['x', 'y', 'width', 'height']) next[key] = 0.5 * next[key] + 0.5 * state.roi[key];
          state.roi = next;
        }
      }
      const hands = ['left', 'right'].map((side, index) => {
        const landmarks = keypoints.slice(index ? 112 : 91, index ? 133 : 112);
        return { side, landmarks, score: landmarks.reduce((sum, point) => sum + point.score, 0) / 21, mediaTimeMs, ageMs: 0, observed: true };
      });
      const coverage = bodyCoverage(keypoints, { hands: true });
      return {
        sourceId, mediaTimeMs, frameId, imageSize, keypoints, hands, detected,
        visibleCore: body.length, modelId: this.modelId, modelVariant: this.variant,
        inferenceMs: performance.now() - started, computeMs, backend: this.backend, roi: crop,
        relative3D: false, relative3DCount: 0, coverage, headConsistency, bilateralConsistency,
        coordinateSystem: 'original-image-pixels-no-monocular-depth',
        warnings: ['このモデルは 2D です。Z は校正済み 2 視点の三角測量で求めます。', ...(!manual && !detected ? ['人体の ROI を手動指定してください。未検出の全身を補完しません。'] : [])],
      };
    } finally {
      try { input?.dispose(); } catch { /* Device loss must still release the busy flag. */ }
      if (outputs) for (const output of Object.values(outputs)) {
        try { output.dispose(); } catch { /* Preserve the original inference error. */ }
      }
      this.busy = false;
      this.finishInference?.();
      this.finishInference = null;
    }
  }

  reset(sourceId) { if (sourceId) this.states.delete(sourceId); else this.states.clear(); }
  async dispose() { this.lifecycle++; if (this.busy) await this.inferenceDone; if (this.session) await this.session.release(); this.session = null; this.states.clear(); }
}

export { KEYPOINT_NAMES };
