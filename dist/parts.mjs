// Public model downloads only. Pixel data and inferred landmarks remain local.
const clamp = (value, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, value));
const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const finitePoint = point => point && Number.isFinite(point.x) && Number.isFinite(point.y);
const pointScore = point => point?.score ?? point?.confidence ?? 0;

function sourceSize(source) {
  return [source.videoWidth || source.naturalWidth || source.width, source.videoHeight || source.naturalHeight || source.height];
}

function canvas2d() {
  const canvas = typeof OffscreenCanvas === 'function' ? new OffscreenCanvas(128, 128) : document.createElement('canvas');
  return {canvas, context: canvas.getContext('2d', {willReadFrequently: true})};
}

function cropTensor(surface, source, roi, spec, padding) {
  const [x, y, width, height] = roi;
  const [sourceWidth, sourceHeight] = sourceSize(source);
  if (![x, y, width, height, sourceWidth, sourceHeight].every(Number.isFinite) || Math.min(width, height, sourceWidth, sourceHeight) <= 0) {
    throw new Error('顔・足の範囲が無効です。動画上で範囲を選んでください。');
  }
  const side = Math.ceil(Math.max(width, height) * padding);
  const left = Math.floor(x + width / 2 - Math.max(width, height) * padding / 2);
  const top = Math.floor(y + height / 2 - Math.max(width, height) * padding / 2);
  const [targetWidth, targetHeight] = spec.inputWH;
  const {canvas, context} = surface;
  if (canvas.width !== targetWidth) canvas.width = targetWidth;
  if (canvas.height !== targetHeight) canvas.height = targetHeight;
  context.fillStyle = '#000';
  context.fillRect(0, 0, targetWidth, targetHeight);
  context.drawImage(source, 0, 0, sourceWidth, sourceHeight,
    -left * targetWidth / side, -top * targetHeight / side,
    sourceWidth * targetWidth / side, sourceHeight * targetHeight / side);
  const rgba = context.getImageData(0, 0, targetWidth, targetHeight).data;
  const length = targetWidth * targetHeight;
  const data = new Float32Array(length * 3);
  const mean = spec.mean || [0, 0, 0];
  const std = spec.std || [1, 1, 1];
  const scale = spec.scale ?? 1 / 255;
  for (let i = 0; i < length; i++) {
    for (let channel = 0; channel < 3; channel++) data[channel * length + i] = (rgba[i * 4 + channel] * scale - mean[channel]) / std[channel];
  }
  return {data, shape: [1, 3, targetHeight, targetWidth], bounds: [left, top, side, side]};
}

async function verifyBytes(buffer, expected) {
  if (!expected) return;
  const bytes = await crypto.subtle.digest('SHA-256', buffer);
  const actual = [...new Uint8Array(bytes)].map(value => value.toString(16).padStart(2, '0')).join('');
  if (actual !== expected.toLowerCase()) throw new Error('部位モデルの検証に失敗しました。取得したファイルが公開記録と一致しません。');
}

async function modelBytes(spec) {
  const url = new URL(spec.url, globalThis.location?.href || 'https://localhost/').href;
  let cache;
  try { if (globalThis.caches) cache = await caches.open('vrc-compact-parts-v1'); } catch { /* Cache unavailable in private modes. */ }
  const stored = cache ? await cache.match(url) : null;
  if (stored) {
    const bytes = await stored.arrayBuffer();
    try { await verifyBytes(bytes, spec.sha256); return bytes; }
    catch { await cache.delete(url); }
  }
  const sameOrigin = new URL(url).origin === globalThis.location?.origin;
  const response = await fetch(url, {method: 'GET', mode: 'cors', credentials: sameOrigin ? 'same-origin' : 'omit'});
  if (!response.ok) throw new Error(`部位モデルを取得できませんでした (${response.status})。`);
  const buffer = await response.arrayBuffer();
  await verifyBytes(buffer, spec.sha256);
  if (cache) {
    try { await cache.put(url, new Response(buffer, {headers: {'Content-Type': 'application/octet-stream'}})); } catch { /* Inference does not require storage. */ }
  }
  return buffer;
}

export function faceMetrics68(points, calibration = {}) {
  const ear = start => (distance(points[start + 1], points[start + 5]) + distance(points[start + 2], points[start + 4])) /
    Math.max(2 * distance(points[start], points[start + 3]), 1e-6);
  const center = start => ({x: points.slice(start, start + 6).reduce((n, p) => n + p.x, 0) / 6,
    y: points.slice(start, start + 6).reduce((n, p) => n + p.y, 0) / 6});
  const eyeLeftEAR = ear(42), eyeRightEAR = ear(36);
  const mouthOpenRatio = (distance(points[61], points[67]) + distance(points[62], points[66]) + distance(points[63], points[65])) /
    Math.max(3 * distance(points[60], points[64]), 1e-6);
  const cheekWidthNormalized = distance(points[3], points[13]) / Math.max(distance(center(36), center(42)), 1e-6);
  const leftOpen = calibration.eyeLeftOpen ?? 0.28, rightOpen = calibration.eyeRightOpen ?? 0.28;
  return {
    eyeLeftEAR, eyeRightEAR,
    blinkLeft: clamp((leftOpen - eyeLeftEAR) / Math.max(leftOpen - 0.1, 0.05)),
    blinkRight: clamp((rightOpen - eyeRightEAR) / Math.max(rightOpen - 0.1, 0.05)),
    mouthOpenRatio, mouthOpen: clamp((mouthOpenRatio - 0.035) / 0.55),
    cheekWidthNormalized,
    cheekWidthChange: Number.isFinite(calibration.cheekWidthNormalized) && calibration.cheekWidthNormalized > 0
      ? cheekWidthNormalized / calibration.cheekWidthNormalized - 1 : null,
    cheekPuff: null, nose: {...points[30]}, stateSource: 'landmark-geometry',
  };
}

export class PartModels {
  constructor(ort, manifest, options = {}) {
    if (!ort || !Array.isArray(manifest?.models)) throw new Error('部位モデルの定義が読み込まれていません。');
    this.ort = ort;
    this.manifest = manifest;
    this.executionProviders = options.executionProviders || ['wasm'];
    this.sessions = new Map();
    this.loading = new Map();
    this.surfaces = {face: canvas2d(), foot: canvas2d()};
  }

  spec(kind) {
    const model = this.manifest.models.find(item => item.kind === (kind === 'face' ? 'faceLandmarks' : 'footView'));
    if (!model) throw new Error(`${kind === 'face' ? '顔' : '足'}モデルはこの公開版に含まれていません。`);
    return model;
  }

  async load(kind) {
    if (kind !== 'face' && kind !== 'foot') throw new Error('未知の部位モデルです。');
    if (this.sessions.has(kind)) return this.sessions.get(kind);
    if (this.loading.has(kind)) return this.loading.get(kind);
    const pending = (async () => {
      const spec = this.spec(kind);
      const bytes = await modelBytes(spec);
      const session = await this.ort.InferenceSession.create(bytes, {executionProviders: this.executionProviders,
        graphOptimizationLevel: 'all', ...(spec.freeDimensionOverrides ? {freeDimensionOverrides: spec.freeDimensionOverrides} : {})});
      this.sessions.set(kind, session);
      return session;
    })();
    this.loading.set(kind, pending);
    try { return await pending; } finally { this.loading.delete(kind); }
  }

  async runFace(source, roiXYWH, {roiConfidence = 1, calibration = {}} = {}) {
    const session = await this.load('face'), spec = this.spec('face');
    const begin = performance.now();
    const input = cropTensor(this.surfaces.face, source, roiXYWH, spec, 1.1);
    const tensor = new this.ort.Tensor('float32', input.data, input.shape);
    const preprocessMs = performance.now() - begin;
    const start = performance.now();
    const outputs = await session.run({[session.inputNames[0]]: tensor});
    const inferenceMs = performance.now() - start;
    const output = outputs[session.outputNames[0]];
    const values = output.data;
    if (values.length !== 136) throw new Error('顔モデルの出力が68点ではありません。');
    let plausible = 0;
    const [left, top, side] = input.bounds;
    const points = Array.from({length: 68}, (_, i) => {
      const x = Number(values[i * 2]), y = Number(values[i * 2 + 1]);
      if (Number.isFinite(x) && Number.isFinite(y) && x >= -0.15 && x <= 1.15 && y >= -0.15 && y <= 1.15) plausible++;
      return {x: left + x * side, y: top + y * side};
    });
    for (const value of Object.values(outputs)) value.dispose?.();
    tensor.dispose?.();
    const valid = roiConfidence >= 0.3 && Math.min(roiXYWH[2], roiXYWH[3]) >= 24 && plausible >= 64 &&
      distance(points[36], points[39]) > side * 0.035 && distance(points[42], points[45]) > side * 0.035;
    const features = points.every(finitePoint) ? faceMetrics68(points, calibration) : {};
    return {valid, points, ...features, status: valid ? 'estimated' : 'unknown', roiConfidence,
      inferenceMs, preprocessMs, modelId: spec.id,
      limitation: '顔ROI内のランドマーク由来の推定。頬の膨らみは未学習。'};
  }

  async runFoot(source, roiXYWH, {roiConfidence = 1} = {}) {
    const session = await this.load('foot'), spec = this.spec('foot');
    const begin = performance.now();
    const input = cropTensor(this.surfaces.foot, source, roiXYWH, spec, 1.15);
    const tensor = new this.ort.Tensor('float32', input.data, input.shape);
    const preprocessMs = performance.now() - begin;
    const start = performance.now();
    const outputs = await session.run({[session.inputNames[0]]: tensor});
    const inferenceMs = performance.now() - start;
    const probabilities = Array.from(outputs[session.outputNames[0]].data, Number);
    for (const value of Object.values(outputs)) value.dispose?.();
    tensor.dispose?.();
    if (probabilities.length !== 3 || !probabilities.every(Number.isFinite)) throw new Error('足モデルの出力が不正です。');
    const index = probabilities.indexOf(Math.max(...probabilities));
    const confidence = probabilities[index], threshold = spec.confidenceThreshold ?? 0.7;
    const valid = confidence >= threshold && roiConfidence >= 0.25 && Math.min(roiXYWH[2], roiXYWH[3]) >= 16;
    return {valid, status: valid ? 'observed' : 'unknown', view: valid ? spec.classOrder[index] : null,
      label: valid ? spec.classLabels[index] : '不明', probabilities, confidence, threshold,
      roiConfidence, inferenceMs, preprocessMs, modelId: spec.id, knownFootRequired: true};
  }

  async dispose() {
    await Promise.all([...this.sessions.values()].map(session => session.release?.()));
    this.sessions.clear();
  }
}

function roiFromPoints(points, indices, padding, minimum) {
  const selected = indices.map(index => points[index]).filter(point => finitePoint(point) && pointScore(point) >= minimum);
  if (selected.length < 3) return null;
  const xs = selected.map(p => p.x), ys = selected.map(p => p.y);
  const x = Math.min(...xs), y = Math.min(...ys), w = Math.max(...xs) - x, h = Math.max(...ys) - y;
  if (Math.max(w, h) < 6) return null;
  const side = Math.max(w, h) * padding;
  return {roi: [x + w / 2 - side / 2, y + h / 2 - side / 2, side, side],
    confidence: Math.min(...selected.map(pointScore)), source: 'compact59-landmarks'};
}

export function footRoiFrom59(points, side) {
  if (!Array.isArray(points) || points.length !== 59) return null;
  return roiFromPoints(points, side === 'left' ? [11, 13, 14, 15] : [12, 16, 17, 18], 1.3, 0.18);
}

export function faceRoiFrom59(points) {
  if (!Array.isArray(points) || points.length !== 59) return null;
  // Mouth and cheeks alone cover too little forehead: enlarge the spread of
  // nose, eye, mouth and cheek points, then shift upward by 10% of its side.
  const region = roiFromPoints(points, [0, ...Array.from({length: 10}, (_, i) => 49 + i)], 1.6, 0.22);
  if (region) region.roi[1] -= region.roi[3] * 0.1;
  return region;
}

export async function loadPartsManifest(url = './models/parts/parts-manifest.json') {
  const response = await fetch(url, {credentials: 'same-origin'});
  if (!response.ok) throw new Error('部位モデルの結果を読み込めません。');
  const manifest = await response.json();
  if (manifest.schemaVersion !== 1 || !Array.isArray(manifest.models)) throw new Error('部位モデルの定義が不正です。');
  return manifest;
}
