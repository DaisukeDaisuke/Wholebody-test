import { DEFAULT_MODEL_BASE, loadManifest, resolveModel, fetchModel, sourceSize, normalizeRoi, makeCanvas, HAND_NAMES, finiteXYZ, bodyCoverage, suppressInconsistentFace, suppressAmbiguousBilateralOverlap, fuseSameFrameModels } from './pose-models.js';
import { PoseOnnxRuntime } from './pose-onnx.js';

const VISION_VERSION = '0.10.21';
const VISION_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VISION_VERSION}`;
const POSE_NAMES = ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'];
const CORE_IDS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
let visionModulePromise;
let visionFilesPromise;

async function visionLibrary() {
  visionModulePromise ||= import(`${VISION_ROOT}/vision_bundle.mjs`);
  const library = await visionModulePromise;
  visionFilesPromise ||= library.FilesetResolver.forVisionTasks(`${VISION_ROOT}/wasm`);
  return { library, files: await visionFilesPromise };
}

function clamp01(value) { return Math.max(0, Math.min(1, Number(value) || 0)); }
function isInside(u, v, size) { return u >= 0 && u < size.width && v >= 0 && v < size.height; }

/**
 * Shared observation contract:
 * u/v are ORIGINAL unmirrored image pixels.
 * relativeXYZm is GHUM camera-coordinate, hip-centred 3D: x right, y down,
 * z farther from the camera. It is a learned monocular estimate, NOT root
 * distance or a triangulated measurement. Missing 3D is null, never z = 0.
 */
export class PoseRuntime {
  constructor() {
    this.contexts = new Map();
    this.generation = 0;
    this.busy = false;
    this.ready = false;
    this.onnx = null;
    this.rescueOnnx = null;
    this.rescueReady = false;
    this.rescueInitialization = null;
    this.rescueError = null;
    this.sharedPose = null;
    this.sharedPoseInitialization = null;
  }

  async initialize({ variant = 'hybrid', backend = 'GPU', hands = true, onProgress, modelBaseUrl = DEFAULT_MODEL_BASE, manifest, confidence = 0.35, handFps = 15, rescueIntervalMs = null, rescueFps = 5, rescueModel = 'dwpose-m', poseMode = 'IMAGE' } = {}) {
    await this.dispose();
    this.variant = variant;
    this.poseMode = poseMode === 'VIDEO' ? 'VIDEO' : 'IMAGE';
    this.backendRequested = backend;
    this.handsEnabled = Boolean(hands);
    this.confidence = confidence;
    this.handIntervalMs = 1000 / Math.max(1, Math.min(30, handFps));
    const rescueRate = Math.max(5, Math.min(12, Number(rescueFps) || 5));
    this.rescueIntervalMs = rescueIntervalMs == null ? 1000 / rescueRate : Math.max(1000 / 12, Number(rescueIntervalMs) || 200);
    this.rescueVariant = rescueModel === 'rtmpose-s' || rescueModel === 'rtmpose_s_wholebody' ? 'rtmpose-s' : 'dwpose-m';
    this.rescueLabel = this.rescueVariant === 'rtmpose-s' ? 'RTMPose-s' : 'DWPose-m';
    this.onProgress = onProgress;
    this.modelBaseUrl = modelBaseUrl;
    this.manifest = await loadManifest({ manifest, modelBaseUrl, onProgress });
    this.modelId = `${this.manifest.repository || 'RabbitDaisuke/Wholebody-Video-Tracking'}@${this.manifest.revision || 'unresolved'}/${variant}`;
    if (['dwpose-m', 'dwpose_m_wholebody', 'rtmpose-s', 'rtmpose_s_wholebody'].includes(variant)) {
      this.onnx = new PoseOnnxRuntime();
      await this.onnx.initialize({ variant, backend, onProgress, manifest: this.manifest, modelBaseUrl, confidence });
      this.backend = this.onnx.backend;
      this.ready = true;
      return { backend: this.backend, modelId: this.onnx.modelId, variant, relative3D: false };
    }
    if (!['hybrid', 'full', 'lite'].includes(variant)) throw new Error(`未対応のモデルです: ${variant}`);
    const poseVariant = variant === 'lite' ? 'lite' : 'full';
    const poseEntry = resolveModel(this.manifest, poseVariant === 'lite' ? 'pose_lite' : 'pose_full', `pose_landmarker_${poseVariant}.task`, modelBaseUrl);
    this.modelId = `${this.manifest.repository || 'RabbitDaisuke/Wholebody-Video-Tracking'}@${this.manifest.revision || 'unresolved'}/${poseEntry.file}`;
    const handEntry = resolveModel(this.manifest, 'hands', 'hand_landmarker.task', modelBaseUrl);
    const [vision, poseBytes, handBytes] = await Promise.all([
      visionLibrary(), fetchModel(poseEntry, onProgress), hands ? fetchModel(handEntry, onProgress) : Promise.resolve(null),
    ]);
    this.vision = vision;
    this.poseBytes = poseBytes;
    this.handBytes = handBytes;
    this.backend = backend === 'CPU' || backend === 'WASM' ? 'MediaPipe CPU (WASM)' : 'MediaPipe GPU (WebGL)';
    this.delegate = backend === 'CPU' || backend === 'WASM' ? 'CPU' : 'GPU';
    onProgress?.({ phase: 'initialize', file: poseEntry.file, ratio: null });
    await this.contextFor('video-1');
    this.ready = true;
    onProgress?.({ phase: 'ready', ratio: 1, backend: this.backend, modelId: this.modelId, poseMode: this.poseMode });
    return { backend: this.backend, modelId: this.modelId, variant, relative3D: true, poseMode: this.poseMode, sharedBodyModel: this.poseMode === 'IMAGE', rescueIntervalMs: variant === 'hybrid' ? this.rescueIntervalMs : null };
  }

  async prepareSources(sourceIds) {
    if (!this.ready) throw new Error('先にモデルを読み込んでください。');
    if (this.onnx) return;
    for (const sourceId of sourceIds) await this.contextFor(sourceId);
  }

  async createPose(mode) {
    return this.vision.library.PoseLandmarker.createFromOptions(this.vision.files, {
      baseOptions: { modelAssetBuffer: this.poseBytes, delegate: this.delegate },
      runningMode: mode, numPoses: 1,
      minPoseDetectionConfidence: 0.45, minPosePresenceConfidence: 0.45, minTrackingConfidence: 0.55,
      outputSegmentationMasks: false,
    });
  }

  async sharedPoseForImages() {
    if (this.sharedPose) return this.sharedPose;
    if (this.sharedPoseInitialization) return this.sharedPoseInitialization;
    const generation = this.generation;
    const pending = this.createPose('IMAGE').then(pose => {
      if (generation !== this.generation) { pose.close(); throw new Error('人体モデルの初期化は取り消されました。'); }
      this.sharedPose = pose;
      return pose;
    });
    this.sharedPoseInitialization = pending;
    try { return await pending; }
    finally { if (this.sharedPoseInitialization === pending) this.sharedPoseInitialization = null; }
  }

  async contextFor(sourceId) {
    if (this.contexts.has(sourceId)) return this.contexts.get(sourceId);
    const generation = this.generation;
    const pending = (async () => {
      const ownsPose = this.poseMode === 'VIDEO';
      const pose = ownsPose ? await this.createPose('VIDEO') : await this.sharedPoseForImages();
      let hand = null;
      try {
        if (this.handsEnabled) {
          hand = await this.vision.library.HandLandmarker.createFromOptions(this.vision.files, {
            baseOptions: { modelAssetBuffer: this.handBytes, delegate: this.delegate },
            runningMode: 'VIDEO', numHands: 2,
            minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
          });
        }
      } catch (error) { if (ownsPose) pose.close(); throw error; }
      const context = { pose, ownsPose, hand, lastTimestamp: -1, lastMediaTime: null, lastHandAt: -Infinity, hands: [], crop: null, roiKey: null, frames: 0, lossFrames: 0, lastRescueAt: -Infinity, lastRescueMediaTimeMs: -Infinity, lastRescue: null, depthValidation: null, depthValidationTimeMs: null };
      if (generation !== this.generation) { if (ownsPose) pose.close(); hand?.close(); throw new Error('モデルの初期化は取り消されました。'); }
      return context;
    })();
    this.contexts.set(sourceId, pending);
    try { return await pending; }
    catch (error) {
      if (this.contexts.get(sourceId) === pending) this.contexts.delete(sourceId);
      throw new Error(`人体モデルの初期化に失敗しました (${this.backend}): ${error.message}`);
    }
  }

  async infer(source, { sourceId = 'video-1', mediaTimeMs = 0, frameId = 0, roi = null } = {}) {
    if (!this.ready) throw new Error('先にモデルを読み込んでください。');
    if (this.onnx) return this.onnx.infer(source, { sourceId, mediaTimeMs, frameId, roi });
    const imageSize = sourceSize(source);
    if (this.busy) return { sourceId, mediaTimeMs, frameId, imageSize, keypoints: [], hands: [], detected: false, skipped: true, skipReason: 'inference-busy', modelId: this.modelId, backend: this.backend, inferenceMs: 0, warnings: [] };
    this.busy = true;
    this.inferenceDone = new Promise(resolve => { this.finishInference = resolve; });
    const started = performance.now();
    try {
      // A lazy second-camera graph can take longer than one frame to load.
      // Capture before that await so pixels still match the supplied timestamp.
      let frameSource = source;
      const preserveFrame = () => {
        if (frameSource !== source) return;
        frameSource = makeCanvas(imageSize.width, imageSize.height);
        frameSource.getContext('2d', { alpha: false }).drawImage(source, 0, 0);
      };
      if (!this.contexts.has(sourceId)) preserveFrame();
      let context = await this.contextFor(sourceId);
      const selected = normalizeRoi(roi, imageSize);
      const roiKey = selected ? [selected.x, selected.y, selected.width, selected.height].map(v => Math.round(v)).join(',') : 'full';
      // Tracker state cannot be shared across sources, seeks, or coordinate changes.
      if ((context.lastMediaTime != null && (mediaTimeMs < context.lastMediaTime - 1 || mediaTimeMs - context.lastMediaTime > 1000)) || (context.roiKey != null && context.roiKey !== roiKey)) {
        preserveFrame();
        await this.reset(sourceId);
        context = await this.contextFor(sourceId);
      }
      context.roiKey = roiKey;
      const timestamp = Math.max(0, Number.isFinite(mediaTimeMs) ? mediaTimeMs : performance.now(), context.lastTimestamp + 0.01);
      context.lastTimestamp = timestamp;
      context.lastMediaTime = mediaTimeMs;
      let input = frameSource;
      let transform = { x: 0, y: 0, width: imageSize.width, height: imageSize.height };
      if (selected) {
        transform = selected;
        // Out-of-frame rectangles intentionally retain black padding so a
        // visible lower body is not stretched into a fictitious full body.
        const width = Math.max(2, Math.round(selected.width));
        const height = Math.max(2, Math.round(selected.height));
        if (!context.crop || context.crop.width !== width || context.crop.height !== height) context.crop = makeCanvas(width, height);
        const drawing = context.crop.getContext('2d', { alpha: false });
        drawing.fillStyle = '#000';
        drawing.fillRect(0, 0, width, height);
        drawing.drawImage(frameSource, -selected.x, -selected.y);
        input = context.crop;
      }
      const result = this.poseMode === 'IMAGE' ? context.pose.detect(input) : context.pose.detectForVideo(input, timestamp);
      const landmarks = result.landmarks?.[0] || [];
      const world = result.worldLandmarks?.[0] || [];
      const keypoints = landmarks.slice(0, 33).map((landmark, index) => {
        const u = transform.x + landmark.x * transform.width;
        const v = transform.y + landmark.y * transform.height;
        const score = clamp01(Math.min(landmark.visibility ?? 1, landmark.presence ?? 1));
        const relativeXYZm = finiteXYZ(world[index]);
        return { id: index, name: POSE_NAMES[index], u, v, relativeXYZm, relativeZm: relativeXYZm?.[2] ?? null, score, score2D: score, score3D: relativeXYZm ? score : null, score2DType: 'mediapipe-visibility-presence', modelId2D: this.modelId, modelId3D: relativeXYZm ? this.modelId : null, observable: score >= this.confidence && isInside(u, v, imageSize), source: 'monocular-model', observationTimeMs: mediaTimeMs };
      });
      const headConsistency = suppressInconsistentFace(keypoints, imageSize);
      const bilateralConsistency = suppressAmbiguousBilateralOverlap(keypoints, imageSize);
      context.frames++;
      const visibleCore = keypoints.filter((point, index) => CORE_IDS.includes(index) && point.observable).length;
      context.lossFrames = visibleCore >= 4 ? 0 : context.lossFrames + 1;
      const warnings = [];
      if (headConsistency.suppressed) warnings.push('顔の推定位置が体幹上に陥っているため、顔点を有効観測から除外しました。');
      if (bilateralConsistency.groups.length) warnings.push(`左右の識別が不確かな重複点を除外しました: ${bilateralConsistency.groups.join(', ')}`);
      if (selected) warnings.push('MediaPipe の ROI モードでも人体検出器を使用します。見切れ時は DWPose の手動 ROI 経路に切り替えられます。');
      if (keypoints.length && visibleCore < 4) warnings.push('観測できる体の関節が不足しています。非可視関節を追跡成功には数えません。');
      const wristVisible = keypoints.some(point => /^(left|right)_wrist$/.test(point.name) && point.observable);
      const handInterval = wristVisible ? this.handIntervalMs : 1000;
      let handsUpdated = false;
      if (context.hand && timestamp - context.lastHandAt >= handInterval) {
        const handResult = context.hand.detectForVideo(input, timestamp);
        context.lastHandAt = timestamp;
        context.hands = this.decodeHands(handResult, keypoints, transform, imageSize, mediaTimeMs);
        handsUpdated = true;
      }
      const hands = context.hands.map(hand => {
        const ageMs = Math.max(0, mediaTimeMs - hand.mediaTimeMs);
        const stale = ageMs > 200;
        return { ...hand, ageMs, observed: handsUpdated && !stale, landmarks: hand.landmarks.map(point => ({ ...point, observable: point.observable && !stale, source: handsUpdated ? 'monocular-model' : 'held', ageMs })) };
      }).filter(hand => hand.ageMs <= 500);
      for (const hand of hands) keypoints.push(...hand.landmarks);
      const coverage = bodyCoverage(keypoints, { hands: this.handsEnabled });
      const observation = {
        sourceId, mediaTimeMs, frameId, imageSize, keypoints, hands,
        detected: landmarks.length === 33 && visibleCore >= 4,
        visibleCore, modelId: this.modelId, modelVariant: this.variant, backend: this.backend,
        poseMode: this.poseMode, sharedBodyModel: this.poseMode === 'IMAGE',
        inferenceMs: performance.now() - started, roi: transform,
        relative3D: coverage.current3D > 0, relative3DCount: coverage.current3D,
        coordinateSystem: 'camera-x-right-y-down-z-forward-hip-relative-metres',
        coverage, headConsistency, bilateralConsistency, warnings,
      };
      if (this.variant === 'hybrid') {
        return await this.maybeRescue(frameSource, { sourceId, mediaTimeMs, frameId, roi: selected }, context, observation, started);
      }
      return observation;
    } finally { this.busy = false; this.finishInference?.(); this.finishInference = null; }
  }

  startRescueInitialization() {
    if (this.rescueOnnx || this.rescueInitialization || this.rescueError) return;
    const generation = this.generation;
    const runtime = new PoseOnnxRuntime();
    this.rescueOnnx = runtime;
    // Do not await model download from infer(): a playing video would advance
    // while its old media timestamp remained attached to a newly sampled frame.
    this.rescueInitialization = runtime.initialize({
      variant: this.rescueVariant, backend: this.backendRequested,
      manifest: this.manifest, modelBaseUrl: this.modelBaseUrl,
      confidence: this.confidence,
      onProgress: event => { if (generation === this.generation) this.onProgress?.({ ...event, role: 'rescue' }); },
    }).then(async () => {
      if (generation !== this.generation) { await runtime.dispose(); return; }
      this.rescueReady = true;
    }).catch(async error => {
      if (generation === this.generation) {
        this.rescueError = String(error.message || error);
        this.rescueReady = false;
      }
      try { await runtime.dispose(); } catch { /* The initialization error remains the reported failure. */ }
    }).finally(() => {
      if (generation === this.generation) this.rescueInitialization = null;
    });
  }

  async maybeRescue(source, options, context, observation, started) {
    const rejectedBody = context.depthValidation && observation.keypoints.some(point => CORE_IDS.includes(point.id) && point.observable && context.depthValidation.get(point.name) === false);
    const insufficient = !observation.detected || observation.coverage.missingRegions.length > 0 || rejectedBody;
    const rescue = {
      enabled: true, requested: insufficient, ran: false, intervalMs: this.rescueIntervalMs,
      rateScope: 'per-input', maximumInferenceFps: 1000 / this.rescueIntervalMs,
      modelVariant: this.rescueVariant, missingRegions: observation.coverage.missingRegions,
      state: this.rescueReady ? 'ready' : this.rescueError ? 'error' : this.rescueInitialization ? 'loading' : 'idle',
    };
    observation.runtimeMode = 'hybrid';
    observation.rescue = rescue;
    if (!insufficient) return this.applyRecentDepthValidation(observation, context);
    this.startRescueInitialization();
    rescue.state = this.rescueReady ? 'ready' : this.rescueError ? 'error' : 'loading';
    if (this.rescueError) {
      observation.warnings.push(`2D 再取得モデルを利用できません: ${this.rescueError}`);
      return this.applyRecentDepthValidation(observation, context);
    }
    if (!this.rescueReady) {
      observation.warnings.push(`部位の観測不足のため ${this.rescueLabel} を読み込み中です。`);
      return this.applyRecentDepthValidation(observation, context);
    }
    const wallTime = performance.now();
    const elapsedWall = wallTime - context.lastRescueAt;
    const elapsedMedia = options.mediaTimeMs - context.lastRescueMediaTimeMs;
    if (elapsedWall < this.rescueIntervalMs || elapsedMedia < this.rescueIntervalMs) {
      rescue.state = 'interval';
      rescue.lastObservationTimeMs = context.lastRescueMediaTimeMs;
      rescue.ageMs = Math.max(0, elapsedMedia);
      rescue.lastDetected = context.lastRescue?.detected ?? false;
      // Do not re-emit old rescued coordinates with this frame's timestamp.
      // Preview layers may hold their last display; fusion receives only the
      // current GHUM observation until the next genuine rescue inference.
      return this.applyRecentDepthValidation(observation, context);
    }
    context.lastRescueAt = wallTime;
    context.lastRescueMediaTimeMs = options.mediaTimeMs;
    let fallback;
    try {
      fallback = await this.rescueOnnx.infer(source, options);
    } catch (error) {
      rescue.state = 'error';
      observation.warnings.push(`${this.rescueLabel} の再取得に失敗しました: ${error.message}`);
      observation.inferenceMs = performance.now() - started;
      return this.applyRecentDepthValidation(observation, context);
    }
    if (fallback.skipped) return this.applyRecentDepthValidation(observation, context);
    context.lastRescue = { detected: fallback.detected, visibleCore: fallback.visibleCore, modelId: fallback.modelId };
    Object.assign(rescue, { state: 'complete', ran: true, detected: fallback.detected, visibleCore: fallback.visibleCore, modelId: fallback.modelId, backend: fallback.backend, inferenceMs: fallback.inferenceMs, observationTimeMs: fallback.mediaTimeMs });
    const timing = { primaryMs: observation.inferenceMs, rescueMs: fallback.inferenceMs };
    const fused = fuseSameFrameModels(observation, fallback, { hands: this.handsEnabled });
    context.depthValidation = new Map(fused.keypoints.map(point => [point.name, !!(point.observable && point.relativeXYZm && (point.modelAgreement === true || point.validation === 'cross-model-2d-agreement'))]));
    context.depthValidationTimeMs = observation.mediaTimeMs;
    return {
      ...observation, ...fused,
      modelVariant: 'hybrid', modelsUsed: [observation.modelId, fallback.modelId],
      backend: `${observation.backend} + ${fallback.backend} (rescue)`,
      runtimeMode: 'hybrid', source: 'hybrid-partial-refinement',
      inferenceMs: performance.now() - started, componentTimingMs: timing,
      primaryAttempt: { modelId: observation.modelId, backend: observation.backend, detected: observation.detected, visibleCore: observation.visibleCore, coverage: observation.coverage },
      rescue,
      warnings: [
        ...observation.warnings,
        `${this.rescueLabel} で不足部位を再取得しました。2D の整合した点だけ既存の単眼 3D を保持します。`,
        ...(fused.modelAgreement.conflicts ? [`モデル間で ${fused.modelAgreement.conflicts} 点の位置が一致しません。追加点の正確性を確認済みとは扱いません。`] : []),
        ...(fused.bilateralConsistency.groups.length ? [`左右識別が不確かな部位: ${fused.bilateralConsistency.groups.join(', ')}`] : []),
        ...(fused.coverage.missingRegions.length ? [`欠測部位: ${fused.coverage.missingRegions.join(', ')}`] : []),
      ],
    };
  }

  applyRecentDepthValidation(observation, context) {
    if (!context.depthValidation) return observation;
    for (const point of observation.keypoints) {
      if (context.depthValidation.get(point.name) !== false || !point.relativeXYZm) continue;
      point.depthPriorXYZm = point.relativeXYZm.slice();
      point.depthPriorScore = point.score;
      point.depthPriorRawObservable = point.observable === true && point.source !== 'held';
      point.depthPriorObservable = false;
      point.depthPriorAgreement = false;
      point.depthPriorTimeMs = point.observationTimeMs ?? observation.mediaTimeMs;
      point.depthPriorModelId = observation.modelId;
      point.relativeXYZm = null; point.relativeZm = null; point.score3D = null; point.modelId3D = null;
      point.depthStatus = 'rejected-pending-cross-model-recheck';
      point.lastDepthValidationTimeMs = context.depthValidationTimeMs;
    }
    observation.coverage = bodyCoverage(observation.keypoints, { hands: this.handsEnabled });
    observation.relative3DCount = observation.coverage.current3D;
    observation.relative3D = observation.relative3DCount > 0;
    observation.depthValidationTimeMs = context.depthValidationTimeMs;
    return observation;
  }

  decodeHands(result, body, transform, imageSize, mediaTimeMs) {
    const rawHands = result.landmarks || [];
    const bodyWrists = ['left', 'right'].map(side => body.find(point => point.name === `${side}_wrist` && point.observable));
    let assignedSides = null;
    if (rawHands.length === 2 && bodyWrists.every(Boolean)) {
      const pixels = rawHands.map(points => ({ u: transform.x + points[0].x * transform.width, v: transform.y + points[0].y * transform.height }));
      const distance = (hand, wrist) => Math.hypot(pixels[hand].u - bodyWrists[wrist].u, pixels[hand].v - bodyWrists[wrist].v);
      const direct = distance(0, 0) + distance(1, 1);
      const crossed = distance(0, 1) + distance(1, 0);
      const permutation = direct <= crossed ? [0, 1] : [1, 0];
      const maximumDistance = Math.hypot(imageSize.width, imageSize.height) * 0.15;
      if (permutation.every((wrist, hand) => distance(hand, wrist) < maximumDistance)) assignedSides = permutation.map(wrist => wrist ? 'right' : 'left');
    }
    return rawHands.map((landmarks, index) => {
      const classification = result.handedness?.[index]?.[0] || result.handednesses?.[index]?.[0];
      let side = String(classification?.categoryName || classification?.displayName || '').toLowerCase();
      const wristU = transform.x + landmarks[0].x * transform.width;
      const wristV = transform.y + landmarks[0].y * transform.height;
      // Spatial association to a body wrist disambiguates crossing hands and
      // avoids changing handedness based on a mirrored preview canvas.
      const candidates = ['left', 'right'].map(value => {
        const wrist = body.find(point => point.name === `${value}_wrist` && point.observable);
        return { side: value, wrist, distance: wrist ? Math.hypot(wrist.u - wristU, wrist.v - wristV) : Infinity };
      }).sort((a, b) => a.distance - b.distance);
      if (assignedSides) side = assignedSides[index];
      else if (candidates[0].distance < Math.hypot(imageSize.width, imageSize.height) * 0.12 && candidates[0].distance < candidates[1].distance * 0.7) side = candidates[0].side;
      if (side !== 'left' && side !== 'right') return null;
      const bodyWrist = body.find(point => point.name === `${side}_wrist`);
      const handWorld = result.worldLandmarks?.[index] || [];
      const handOrigin = finiteXYZ(handWorld[0]);
      const anchor = bodyWrist?.observable ? bodyWrist.relativeXYZm : null;
      // The category score describes left/right classification, not landmark
      // visibility. Tasks does not expose per-finger confidence. Use the
      // configured hand-presence threshold as a conservative task-level
      // weight, and retain the handedness score under its actual meaning.
      const confidence = 0.5;
      const handednessScore = clamp01(classification?.score ?? 0);
      const points = landmarks.map((point, joint) => {
        const u = transform.x + point.x * transform.width;
        const v = transform.y + point.y * transform.height;
        const local = finiteXYZ(handWorld[joint]);
        const relativeXYZm = anchor && local && handOrigin ? local.map((value, axis) => value - handOrigin[axis] + anchor[axis]) : null;
        return { id: 33 + (side === 'left' ? 0 : 21) + joint, name: `${side}_${HAND_NAMES[joint]}`, u, v, relativeXYZm, relativeZm: relativeXYZm?.[2] ?? null, score: confidence, scoreSource: 'task-presence-threshold', visibilityKnown: false, observable: confidence >= this.confidence && isInside(u, v, imageSize), handRelativeXYZm: local && handOrigin ? local.map((value, axis) => value - handOrigin[axis]) : null, source: 'monocular-model', observationTimeMs: mediaTimeMs };
      });
      return { side, score: confidence, handednessScore, landmarkVisibilityKnown: false, landmarks: points, mediaTimeMs, ageMs: 0, observed: true };
    }).filter(Boolean).sort((a, b) => b.handednessScore - a.handednessScore).filter((hand, index, all) => all.findIndex(other => other.side === hand.side) === index);
  }

  async reset(sourceId) {
    if (this.onnx) return this.onnx.reset(sourceId);
    this.rescueOnnx?.reset(sourceId);
    const selected = sourceId ? [[sourceId, this.contexts.get(sourceId)]] : [...this.contexts.entries()];
    for (const [key, pending] of selected) {
      this.contexts.delete(key);
      if (!pending) continue;
      try {
        const value = await pending;
        if (value.ownsPose) value.pose.close();
        value.hand?.close();
      } catch { /* Already failed or disposed. */ }
    }
  }

  async dispose() {
    this.ready = false;
    if (this.busy) await this.inferenceDone;
    this.generation++;
    if (this.onnx) { await this.onnx.dispose(); this.onnx = null; }
    if (this.rescueOnnx) { await this.rescueOnnx.dispose(); this.rescueOnnx = null; }
    this.rescueReady = false;
    this.rescueInitialization = null;
    this.rescueError = null;
    await this.reset();
    if (this.sharedPose) { this.sharedPose.close(); this.sharedPose = null; }
    this.sharedPoseInitialization = null;
    this.poseBytes = null;
    this.handBytes = null;
    this.busy = false;
  }
}

export { POSE_NAMES };
