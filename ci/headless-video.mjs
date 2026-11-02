/** Actions-only: actual decoded video -> real CPU ONNX -> actual VRM update.
 * Frames and per-frame coordinates stay in memory. Only aggregate JSON is saved.
 * CLI: node ci/headless-video.mjs --video PATH --vrm PATH --output PATH
 * Omit video/vrm to use pinned, verified official public fixtures. */
import { readFile, writeFile, mkdtemp, mkdir, rm, stat } from 'node:fs/promises';
import { resolve, dirname, join, sep } from 'node:path';
import { tmpdir, cpus } from 'node:os';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';
import { COCO59, cropTransform, decodeSimCC, toCoco133 } from '../dist/compact-codec.mjs';
import { faceMetrics68, faceRoiFrom59, footRoiFrom59 } from '../dist/parts.mjs';
import { evaluateStabilitySequence } from './headless-stability.mjs';
import { quaternionAngle, multiplyQuaternion } from '../dist/retarget.mjs';

const PUBLIC_VIDEO = Object.freeze({
  url: 'https://dl.fbaipublicfiles.com/pytorchvideo/projects/archery.mp4',
  sourceDocument: 'https://pytorchvideo.org/docs/tutorial_torchhub_inference',
  revision: 'sha256:8d029ab048f571b136a8c0afddbbac022606022ca95307a78655dbde9735a562',
  sha256: '8d029ab048f571b136a8c0afddbbac022606022ca95307a78655dbde9735a562', bytes: 549197,
});
const PUBLIC_VRM = Object.freeze({
  url: 'https://raw.githubusercontent.com/vrm-c/vrm-specification/94e82dd346fa6cf0337c4421728640e5252dd38e/samples/Seed-san/vrm/Seed-san.vrm',
  revision: '94e82dd346fa6cf0337c4421728640e5252dd38e', blob: 'ff4223f9b42c8f22be35dcf23abe477d93101300', bytes: 10917800,
});
const REFERENCE_133 = Object.freeze({ id: 'dwpose-t-independent-133',
  url: 'https://huggingface.co/RabbitDaisuke/RTMPose-Wholebody-onnx/resolve/9d7f318c43452baf33f1b39d39c1f8c03512aeb5/rtmpose-t-256x192-fp32.onnx',
  sha256: 'fcaae4a08ea7610b4bcc5d0b9e8a7e80fd6f2d998be41438a1b2737056a7623f',
  input: { name: 'input', shape: [1, 3, 256, 192], mean: [123.675, 116.28, 103.53], std: [58.395, 57.12, 57.375], crop: { padding: 1.25 } },
  simcc_split_ratio: 2, count: 133,
});
const LEGACY_COMMIT = 'c5983daca5dd65e2ea806186a1a1775c1630f495';
// Confirmed against the pinned GitHub tree; these match local baseline cf519e6.
const LEGACY_MODULES = Object.freeze([
  { role: 'geometry', name: 'geometry.mjs', blob: '3a20c5253212f4d23ed684e65c40abc456d7346c', bytes: 40233 },
  { role: 'retarget', name: 'retarget.mjs', blob: '0645e5e1506897e9f42763555732546d2b62cdba', bytes: 22044 },
  { role: 'calibrationDependency', name: 'calibration.mjs', blob: '75d7bed85374aeb32613c5e38cf97540ba9a650c', bytes: 12220 },
]);
const finite = Number.isFinite;
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const gitBlob = bytes => createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
const maximum = values => values.reduce((m, v) => Math.max(m, v), 0);
const percentile = (values, p) => { const sorted = values.slice().sort((a, b) => a - b); return sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))] : null; };
const summary = values => ({ count: values.length, meanMs: values.length ? values.reduce((a, b) => a + b, 0) / values.length : null, medianMs: percentile(values, .5), p95Ms: percentile(values, .95), maxMs: values.length ? maximum(values) : null });
function fail(code) { const error = new Error(code); error.safeCode = code; throw error; }
function argumentsFor(argv) {
  const options = { output: 'artifacts/headless-video.json', manifest: 'dist/models.json', fps: 15, maxSeconds: 60, maxSize: 640, minFrames: 72, threads: 2 };
  const names = { '--video': 'video', '--vrm': 'vrm', '--output': 'output', '--manifest': 'manifest', '--fps': 'fps', '--max-seconds': 'maxSeconds', '--max-size': 'maxSize', '--min-frames': 'minFrames', '--threads': 'threads' };
  for (let i = 0; i < argv.length; i += 2) { const name = names[argv[i]]; if (!name || !argv[i + 1]) fail('invalidCliArgument'); options[name] = ['fps', 'maxSeconds', 'maxSize', 'minFrames', 'threads'].includes(name) ? Number(argv[i + 1]) : argv[i + 1]; }
  if (!(options.fps >= 1 && options.fps <= 30 && options.maxSeconds > 0 && options.maxSeconds <= 180 && options.maxSize >= 128 && options.maxSize <= 640 && options.threads >= 1 && options.threads <= 8 && options.minFrames >= 72 && options.minFrames <= 900)) fail('invalidEvaluationLimits');
  return options;
}
async function fetchPublic(url, limit = 100 * 1024 * 1024) {
  if (!url.startsWith('https://')) fail('publicDownloadRequiresHttps');
  let response;
  try { response = await fetch(url, { signal: AbortSignal.timeout(120000), credentials: 'omit' }); } catch { fail('publicDownloadUnavailable'); }
  if (!response.ok || Number(response.headers.get('content-length') || 0) > limit) fail('publicDownloadRejected');
  const chunks = []; let total = 0;
  for await (const chunk of response.body) { total += chunk.length; if (total > limit) fail('publicDownloadTooLarge'); chunks.push(chunk); }
  return Buffer.concat(chunks, total);
}
async function publicFixture(spec, target) {
  const bytes = await fetchPublic(spec.url, spec.bytes + 1024);
  if (bytes.length !== spec.bytes || (spec.sha256 ? sha256(bytes) !== spec.sha256 : gitBlob(bytes) !== spec.blob)) fail('publicFixtureIntegrityMismatch');
  await writeFile(target, bytes, { mode: 0o600 });
  return target;
}
async function modelBytes(spec, manifestPath) {
  if (!/^[a-f0-9]{64}$/.test(spec?.sha256 || '')) fail('modelSha256Missing');
  let bytes;
  if (spec.url?.startsWith('https://')) bytes = await fetchPublic(spec.url);
  else {
    const base = dirname(resolve(manifestPath)), path = resolve(base, spec.url || `models/${spec.file}`);
    if (!path.startsWith(base + sep)) fail('modelPathEscapesManifestDirectory');
    try { bytes = await readFile(path); } catch { fail('packagedModelUnavailable'); }
  }
  if (sha256(bytes) !== spec.sha256 || (finite(spec.bytes) && bytes.length !== spec.bytes)) fail('modelIntegrityMismatch');
  return bytes;
}
function childResult(command, args, maxOutput = 32768) {
  return new Promise((resolveResult, reject) => {
    const child = spawn(command, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    const chunks = []; let total = 0, resolved = false;
    const timeout = setTimeout(() => child.kill('SIGKILL'), 30000);
    child.stdout.on('data', data => { total += data.length; if (total <= maxOutput) chunks.push(data); else child.kill('SIGKILL'); });
    child.stderr.resume(); // Decoder errors may contain private names; never log them.
    child.once('error', () => { clearTimeout(timeout); if (!resolved) { resolved = true; reject(Object.assign(new Error('decoderUnavailable'), { safeCode: 'decoderUnavailable' })); } });
    child.once('close', code => { clearTimeout(timeout); if (resolved) return; resolved = true; if (code !== 0 || total > maxOutput) reject(Object.assign(new Error('videoProbeFailed'), { safeCode: 'videoProbeFailed' })); else resolveResult(Buffer.concat(chunks).toString('utf8')); });
  });
}
async function probeVideo(path, options) {
  let metadata;
  try { metadata = JSON.parse(await childResult('ffprobe', ['-v', 'error', '-protocol_whitelist', 'file,pipe', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,duration,avg_frame_rate:format=duration', '-of', 'json', resolve(path)])); } catch (error) { if (error.safeCode) throw error; fail('videoProbeFailed'); }
  const stream = metadata.streams?.[0], width = Number(stream?.width), height = Number(stream?.height), duration = Number(stream?.duration || metadata.format?.duration);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 2 || height < 2 || width > 16384 || height > 16384 || !finite(duration) || duration <= 0) fail('videoMetadataInvalid');
  const ratio = Math.min(1, options.maxSize / Math.max(width, height));
  return { width: Math.max(2, Math.round(width * ratio)), height: Math.max(2, Math.round(height * ratio)), duration: Math.min(duration, options.maxSeconds) };
}
async function* decodedFrames(path, video, options) {
  const child = spawn('ffmpeg', ['-nostdin', '-hide_banner', '-loglevel', 'error', '-threads', '1', '-filter_threads', '1', '-protocol_whitelist', 'file,pipe', '-i', resolve(path), '-t', String(options.maxSeconds), '-an', '-sn', '-dn', '-vf', `fps=${options.fps},scale=${video.width}:${video.height}:flags=bilinear`, '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1'], { stdio: ['ignore', 'pipe', 'pipe'] });
  child.stderr.resume();
  const completed = new Promise(resolveDone => { child.once('close', code => resolveDone(code)); child.once('error', () => resolveDone(-1)); });
  const timeout = setTimeout(() => child.kill('SIGKILL'), 10 * 60 * 1000);
  const frameBytes = video.width * video.height * 3, queue = []; let available = 0, headOffset = 0, count = 0;
  try {
    for await (const chunk of child.stdout) {
      queue.push(chunk); available += chunk.length;
      while (available >= frameBytes) {
        const frame = Buffer.allocUnsafe(frameBytes); let position = 0;
        while (position < frameBytes) {
          const current = queue[0], bytes = Math.min(current.length - headOffset, frameBytes - position);
          current.copy(frame, position, headOffset, headOffset + bytes); position += bytes; headOffset += bytes;
          if (headOffset === current.length) { queue.shift(); headOffset = 0; }
        }
        available -= frameBytes; count++;
        if (count > Math.ceil(options.maxSeconds * options.fps) + 1) fail('decodedFrameLimitExceeded');
        yield frame;
      }
    }
    if (await completed !== 0 || available !== 0) fail('videoDecodeFailed');
  } finally { clearTimeout(timeout); if (child.exitCode === null) child.kill('SIGKILL'); }
}

/** Pixel-centre bilinear sampling matches the browser's black padded crop.
 * Same source rectangle is used for all resolutions; only input sampling differs. */
function rgbCropTensor(rgb, width, height, crop, mean, std, scale = 1) {
  const [iw, ih] = crop.inputWH, plane = iw * ih, data = new Float32Array(plane * 3);
  for (let y = 0; y < ih; y++) {
    const sy = crop.y + (y + .5) * crop.height / ih - .5, y0 = Math.floor(sy), fy = sy - y0;
    for (let x = 0; x < iw; x++) {
      const sx = crop.x + (x + .5) * crop.width / iw - .5, x0 = Math.floor(sx), fx = sx - x0, index = y * iw + x;
      for (let c = 0; c < 3; c++) {
        let value = 0;
        for (let dy = 0; dy < 2; dy++) for (let dx = 0; dx < 2; dx++) {
          const px = x0 + dx, py = y0 + dy;
          if (px >= 0 && px < width && py >= 0 && py < height) value += rgb[(py * width + px) * 3 + c] * (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy);
        }
        data[c * plane + index] = (value * scale - mean[c]) / std[c];
      }
    }
  }
  return data;
}
function usablePartRoi(candidate, video, minimum) {
  if (!candidate || candidate.confidence < .3) return null;
  const [x, y, w, h] = candidate.roi;
  if (![x, y, w, h].every(finite) || Math.min(w, h) < minimum) return null;
  const intersection = Math.max(0, Math.min(video.width, x + w) - Math.max(0, x)) * Math.max(0, Math.min(video.height, y + h) - Math.max(0, y));
  return intersection / (w * h) >= .95 ? candidate : null;
}
async function runPart(model, rgb, video, candidate, ort, side = null) {
  const spec = model.spec, padding = model.role === 'face' ? 1.1 : 1.15;
  const [x, y, width, height] = candidate.roi, expanded = Math.max(width, height) * padding;
  const crop = { x: Math.floor(x + width / 2 - expanded / 2), y: Math.floor(y + height / 2 - expanded / 2), width: Math.ceil(expanded), height: Math.ceil(expanded), inputWH: spec.inputWH };
  const prepStart = performance.now();
  const data = rgbCropTensor(rgb, video.width, video.height, crop, spec.mean || [0, 0, 0], spec.std || [1, 1, 1], spec.scale ?? 1 / 255);
  const tensor = new ort.Tensor('float32', data, [1, 3, spec.inputWH[1], spec.inputWH[0]]), preprocessingMs = performance.now() - prepStart;
  let outputs;
  try {
    if (!model.warmed) {
      for (let i = 0; i < 3; i++) { const warm = await model.session.run({ [model.session.inputNames[0]]: tensor }); Object.values(warm).forEach(t => t.dispose?.()); model.warmupCalls++; }
      model.warmed = true;
    }
    const start = performance.now(); outputs = await model.session.run({ [model.session.inputNames[0]]: tensor }); const inferenceMs = performance.now() - start;
    const values = outputs[model.session.outputNames[0]].data;
    if (!values.every(finite)) fail('nonFinitePartOutput');
    model.timings.push(inferenceMs); model.preprocessing.push(preprocessingMs);
    if (side) model.perSideCalls[side]++;
    if (model.role === 'face') {
      if (values.length !== 136) fail('faceModelOutputContract');
      let plausible = 0;
      const points = Array.from({ length: 68 }, (_, i) => { const px = Number(values[i * 2]), py = Number(values[i * 2 + 1]); if (px >= -.15 && px <= 1.15 && py >= -.15 && py <= 1.15) plausible++; return { x: crop.x + px * crop.width, y: crop.y + py * crop.height }; });
      const eyeWidth = start => Math.hypot(points[start].x - points[start + 3].x, points[start].y - points[start + 3].y);
      const valid = plausible >= 64 && eyeWidth(36) > crop.width * .035 && eyeWidth(42) > crop.width * .035;
      if (valid) model.validCalls++;
      return { valid, points, roiConfidence: candidate.confidence, ...faceMetrics68(points), computeMs: preprocessingMs + inferenceMs };
    }
    if (values.length !== 3 || Math.abs(values.reduce((a, b) => a + b, 0) - 1) > .001) fail('footModelOutputContract');
    const probabilities = Array.from(values), index = probabilities.indexOf(Math.max(...probabilities));
    const valid = probabilities[index] >= (spec.confidenceThreshold ?? .7);
    if (valid) model.validCalls++;
    model.classCounts[valid ? spec.classOrder[index] : 'unknown']++;
    return { valid, computeMs: preprocessingMs + inferenceMs };
  } finally { tensor.dispose?.(); if (outputs) Object.values(outputs).forEach(t => t.dispose?.()); }
}
function observationWithFace(keypoints, face, timestampMs, video) {
  const points = keypoints.map(p => ({ ...p }));
  if (face?.valid) {
    face.points.forEach((p, i) => { points[23 + i] = { id: 23 + i, x: p.x, y: p.y, score: face.roiConfidence, observable: true, source: 'observedFace2D' }; });
    points[0] = { id: 0, ...face.nose, score: face.roiConfidence, observable: true, source: 'observedFace2D' };
    for (const [id, start] of [[1, 42], [2, 36]]) {
      const eye = face.points.slice(start, start + 6);
      points[id] = { id, x: eye.reduce((n, p) => n + p.x, 0) / 6, y: eye.reduce((n, p) => n + p.y, 0) / 6, score: face.roiConfidence, observable: true, derived: true };
    }
  }
  return { sourceId: 'ci-video', inputMode: 'wholebody', targetVersion: 1, timestampMs, imageSize: { width: video.width, height: video.height }, keypoints: points };
}
function agreement(first, second, crop) {
  const distances = [];
  for (const id of COCO59) {
    const a = first[id], b = second[id];
    if (!a || !b || a.score < .25 || b.score < .25 || ![a.x, a.y, b.x, b.y].every(finite)) continue;
    distances.push(Math.hypot((a.x - b.x) * 192 / crop.width, (a.y - b.y) * 256 / crop.height));
  }
  return distances;
}

function rotationStatistics(rig) {
  const localAngles = [], worldAngles = [], localSpeeds = [], worldSpeeds = [];
  let previous = null, count = 0, finiteTransforms = true, maxTorsoTilt = 0, maxRootDisplacement = 0, nonAdjacentGaps = 0;
  const inverse = q => [-q[0], -q[1], -q[2], q[3]], identity = [0, 0, 0, 1];
  return {
    record(result, timestampMs) {
      if (!result) return;
      count++; const dt = previous ? (timestampMs - previous.timestampMs) / 1000 : 0;
      if (dt > .100001) nonAdjacentGaps++;
      for (const [name, bone] of Object.entries(result.bones || {})) {
        finiteTransforms &&= [bone.rotation, bone.worldRotation].every(q => Array.isArray(q) && q.length === 4 && q.every(finite));
        const old = previous?.result.bones?.[name];
        if (old && dt > 1e-6 && dt <= .100001) {
          const local = quaternionAngle(old.rotation, bone.rotation) * 180 / Math.PI, world = quaternionAngle(old.worldRotation, bone.worldRotation) * 180 / Math.PI;
          localAngles.push(local); worldAngles.push(world); localSpeeds.push(local / dt); worldSpeeds.push(world / dt);
        }
      }
      if (result.bones?.hips && rig.bones?.hips) {
        const delta = multiplyQuaternion(result.bones.hips.worldRotation, inverse(rig.bones.hips.rotation || identity));
        maxTorsoTilt = Math.max(maxTorsoTilt, 2 * Math.asin(Math.min(1, Math.hypot(delta[0], delta[1]))) * 180 / Math.PI);
      }
      const root = result.normalizedPose?.hips?.position, oldRoot = previous?.result.normalizedPose?.hips?.position;
      if (Array.isArray(root) && Array.isArray(oldRoot) && dt <= .100001) maxRootDisplacement = Math.max(maxRootDisplacement, Math.hypot(...root.map((v, i) => v - oldRoot[i])));
      previous = { result, timestampMs };
    },
    finish() {
      const angular = (angles, speeds) => ({ p95FrameAngleDeg: percentile(angles, .95), maxFrameAngleDeg: maximum(angles), p95AngularSpeedDegPerSec: percentile(speeds, .95), maxAngularSpeedDegPerSec: maximum(speeds) });
      return { renderFrames: count, finiteTransforms, nonAdjacentGapsExcludedFromFrameChanges: nonAdjacentGaps, localBones: angular(localAngles, localSpeeds), modelSpaceBones: angular(worldAngles, worldSpeeds), maxMonocularTorsoYawPitchDeg: maxTorsoTilt, maxRootFrameDisplacement: maxRootDisplacement };
    },
  };
}

async function loadLegacyComparison(workspace) {
  const root = join(workspace, 'legacy-source'), provenance = [];
  try {
    await mkdir(root);
    for (const module of LEGACY_MODULES) {
      const bytes = await fetchPublic(`https://raw.githubusercontent.com/DaisukeDaisuke/Wholebody-test/${LEGACY_COMMIT}/dist/${module.name}`, 200000);
      if (bytes.length !== module.bytes || gitBlob(bytes) !== module.blob) fail('legacySourceIntegrityMismatch');
      await writeFile(join(root, module.name), bytes, { mode: 0o600 }); provenance.push({ role: module.role, gitBlobSha: module.blob, sha256: sha256(bytes) });
    }
    const geometry = await import(pathToFileURL(join(root, 'geometry.mjs')).href), retarget = await import(pathToFileURL(join(root, 'retarget.mjs')).href);
    if (typeof geometry.estimateMonocularPose !== 'function' || typeof retarget.retargetPose !== 'function') fail('legacyModuleContractMismatch');
    return { geometry, retarget, provenance };
  } catch (error) { return { errorCode: error.safeCode || 'legacySourceUnavailable', provenance }; }
}

function compareLegacyObservations(observations, rig, current, legacy) {
  const base = { baselineCommit: LEGACY_COMMIT, sourceModules: legacy.provenance,
    protocol: 'Identical actual decoded-video inference observations and identical avatar rest rig. Pinned legacy geometry and retarget run unchanged at each inference arrival; legacy output holds between 30 Hz display samples, matching the previous app. Current guards and retarget run at 30 Hz. Both exclude initial activation and unprocessed gaps over 100 ms from adjacent-frame changes. No extra ONNX inference is performed.',
    oldNeutral: 'avatar rest pose', newNeutral: 'rig-relative relaxed upper arms, 10 degrees outward', sourceCodeRetained: false, isPoseAccuracy: false };
  if (legacy.errorCode) return { ...base, status: 'unavailable', errorCode: legacy.errorCode, current };
  try {
    if (!observations.length) fail('legacyObservationsMissing');
    const stats = rotationStatistics(rig), period = 1000 / 30;
    let priorPose = null, previousRetarget = null, nextRender = observations[0].timestampMs, lastTime = nextRender;
    for (const observation of observations) {
      while (nextRender < observation.timestampMs - 1e-6 && nextRender <= lastTime + 2000) { stats.record(previousRetarget, nextRender); nextRender += period; }
      if (nextRender < observation.timestampMs - period) nextRender = observation.timestampMs;
      const pose = legacy.geometry.estimateMonocularPose(observation, null, { bodyHeightM: 1.7, scaleConfirmed: false, minScore: .25, priorPose, allowApproximateCalibration: false });
      // Preserve the old application adapter as well as the pinned pure modules.
      for (const j of pose.joints || []) if (observation.keypoints[j.id]?.derived && j.source !== 'missing') j.source = 'inferredFinger';
      pose.inputMode = 'wholebody'; pose.sourceId = observation.sourceId; pose.targetVersion = 1; priorPose = pose;
      previousRetarget = legacy.retarget.retargetPose(pose, rig, { previous: previousRetarget, footHints: pose.footHints });
      stats.record(previousRetarget, observation.timestampMs); nextRender = Math.max(nextRender, observation.timestampMs + period); lastTime = observation.timestampMs;
    }
    const previous = stats.finish();
    return { ...base, status: previous.finiteTransforms ? 'completed' : 'legacyProducedNonFiniteTransforms', previous, current,
      change: { maxLocalFrameAngleDeg: current.localBones.maxFrameAngleDeg - previous.localBones.maxFrameAngleDeg,
        maxModelSpaceFrameAngleDeg: current.modelSpaceBones.maxFrameAngleDeg - previous.modelSpaceBones.maxFrameAngleDeg,
        maxModelSpaceAngularSpeedDegPerSec: current.modelSpaceBones.maxAngularSpeedDegPerSec - previous.modelSpaceBones.maxAngularSpeedDegPerSec,
        maxMonocularTorsoYawPitchDeg: current.maxMonocularTorsoYawPitchDeg - previous.maxMonocularTorsoYawPitchDeg } };
  } catch (error) { return { ...base, status: 'unavailable', errorCode: error.safeCode || 'legacyReplayFailed', current }; }
}

async function derivedCropScenarios(videoPath, video, options, model, ort, rig, decodedCount, legacy) {
  const selected = new Set();
  if (decodedCount <= 90) for (let i = 0; i < decodedCount; i++) selected.add(i);
  else for (const start of [0, Math.floor((decodedCount - 30) / 2), decodedCount - 30]) for (let i = start; i < start + 30; i++) selected.add(i);
  const height = Math.max(2, Math.floor(video.height * .6));
  const cases = [{ id: 'upper60Percent', top: 0 }, { id: 'lower60Percent', top: video.height - height }].map(item => ({ ...item, observations: [], timings: [], preprocessing: [], finiteOutputs: true }));
  const crop = cropTransform([0, 0, video.width, height], [model.width, model.height], model.spec.input.crop?.padding || 1.25);
  const decodeCrop = { ...crop, inputWH: model.decodeWH };
  let index = 0;
  // A separate decode pass keeps these extra selected-model calls out of the paired
  // CPU benchmark above. RGB windows never reach disk or an artifact.
  for await (const rgb of decodedFrames(videoPath, video, options)) {
    if (selected.has(index)) for (const item of cases) {
      const startRow = item.top * video.width * 3, croppedRgb = rgb.subarray(startRow, startRow + height * video.width * 3);
      const prepStart = performance.now();
      const data = rgbCropTensor(croppedRgb, video.width, height, crop, model.spec.input.mean, model.spec.input.std);
      const tensor = new ort.Tensor('float32', data, [1, 3, model.height, model.width]); item.preprocessing.push(performance.now() - prepStart);
      let outputs;
      try {
        const start = performance.now(); outputs = await model.session.run({ [model.inputName]: tensor }); item.timings.push(performance.now() - start);
        const split = model.spec.simcc_split_ratio || 2, values = Object.values(outputs);
        const x = values.find(t => t.dims.length === 3 && t.dims[1] === 59 && t.dims[2] === model.decodeWH[0] * split), y = values.find(t => t.dims.length === 3 && t.dims[1] === 59 && t.dims[2] === model.decodeWH[1] * split);
        if (!x || !y) fail('derivedCropModelOutputShapeMismatch');
        item.finiteOutputs &&= x.data.every(finite) && y.data.every(finite);
        if (!item.finiteOutputs) fail('derivedCropNonFiniteOutput');
        const points = decodeSimCC(x.data, y.data, decodeCrop, 59, split), keypoints = toCoco133(points, model.spec.coco133_indices || COCO59);
        item.observations.push({ sourceId: 'ci-derived-' + item.id, inputMode: 'wholebody', targetVersion: 1, timestampMs: index * 1000 / options.fps, imageSize: { width: video.width, height }, keypoints });
      } finally { tensor.dispose?.(); if (outputs) Object.values(outputs).forEach(t => t.dispose?.()); }
    }
    index++;
  }
  if (index !== decodedCount) fail('derivedDecodeFrameCountMismatch');
  return Object.fromEntries(cases.map(item => {
    const currentStats = rotationStatistics(rig);
    const stability = evaluateStabilitySequence(item.observations, rig, { renderFps: 30, minScore: .25, geometryOptions: { bodyHeightM: 1.7, scaleConfirmed: false }, includeChallenges: false, onActualRender: result => currentStats.record(result, result.timestampMs) });
    const safetyChecks = { actualCroppedRgbWasInferred: item.timings.length === selected.size && selected.size > 0, finiteOutputs: item.finiteOutputs,
      finiteRotations: stability.checks.finiteRotations === true, boundedAngularVelocity: stability.checks.boundedAngularVelocity === true,
      boundedRootVelocity: stability.checks.boundedRootVelocity === true, monocularYawPitchSuppressed: stability.checks.monocularYawPitchSuppressed === true,
      noCompleteMotionFreezeWhenAcceptedInputMoves: stability.checks.noCompleteMotionFreeze === true };
    return [item.id, { source: options.video ? 'derived image crop from the private input video' : 'derived image crop from the pinned official public video',
      protocol: 'Keep the upper or lower 60% of actual decoded RGB pixels, then rerun the published selected body ONNX. At most 90 frames in three contiguous 30-frame intervals (beginning/middle/end); short videos retain all frames. Original capture timestamps remain unchanged; unsampled gaps are missing inputs, not evidence that the person disappeared. Extra inference runs in a separate decoding pass after the CPU comparison.',
      naturallyCapturedCroppedVideo: false, model: model.spec.id, inferenceCalls: item.timings.length, maximumSelectedFrames: 90,
      cpu: summary(item.timings), preprocessing: summary(item.preprocessing), partsIncluded: false, frameImagesRetained: false, perFrameCoordinatesRetained: false,
      stability, legacyComparison: compareLegacyObservations(item.observations, rig, currentStats.finish(), legacy), safetyChecks, stable: Object.values(safetyChecks).every(Boolean), poseAccuracy: null,
      interpretation: 'Safety and accepted-motion retention are evaluated. Missing accepted landmarks are reported as lost/partial; their actual visibility and anatomical correctness have no ground-truth labels.' }];
  }));
}

async function main(options) {
  if (process.env.GITHUB_ACTIONS !== 'true') fail('githubActionsRequired');
  const require = createRequire(resolve(process.env.VRC_NODE_PACKAGE || '.cache/headless/package.json'));
  let ort; try { ort = require('onnxruntime-node'); } catch { fail('onnxruntimeNodeDependencyMissing'); }
  ort.env.logLevel = 'error';
  const workspace = await mkdtemp(join(tmpdir(), 'vrc-headless-')), sessions = [], previousWarn = console.warn;
  console.warn = () => {}; // Model/VRM internals may print user-controlled metadata.
  let stage = 'prepareInputs';
  try {
    const videoPath = options.video || await publicFixture(PUBLIC_VIDEO, join(workspace, 'input.mp4'));
    const vrmPath = options.vrm || await publicFixture(PUBLIC_VRM, join(workspace, 'avatar.vrm'));
    if ((await stat(videoPath)).size > 1024 * 1024 * 1024 || (await stat(vrmPath)).size > 100 * 1024 * 1024) fail('inputSizeLimitExceeded');
    const video = await probeVideo(videoPath, options);
    stage = 'prepareModels';
    const manifest = JSON.parse(await readFile(options.manifest, 'utf8'));
    const standard = manifest.bodies?.standard, fast = manifest.bodies?.fast || manifest.body;
    if (!fast || !standard || fast.sha256 === standard.sha256) fail('distinctFastAndPreviousModelsRequired');
    const models = [{ role: 'fast59', spec: fast }, { role: 'previous59', spec: standard }, { role: 'reference133', spec: REFERENCE_133 }];
    const selectedModel = models.slice(0, 2).find(model => model.spec.sha256 === manifest.body?.sha256);
    if (!selectedModel) fail('publishedBodyMissingFromComparisonModels');
    const selectedIsFast = selectedModel.role === 'fast59';
    if ((manifest.recommendedBody === 'standard' && selectedIsFast) || (manifest.recommendedBody === 'fast' && !selectedIsFast)) fail('publishedBodyRecommendationMismatch');
    for (const model of models) {
      const bytes = await modelBytes(model.spec, options.manifest);
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ['cpu'], intraOpNumThreads: options.threads, interOpNumThreads: 1, executionMode: 'sequential', graphOptimizationLevel: 'all' });
      sessions.push(session); model.session = session; model.timings = []; model.preprocessing = [];
      model.inputName = session.inputNames.includes(model.spec.input.name) ? model.spec.input.name : session.inputNames[0];
      model.count = model.spec.count || 59; model.width = model.spec.input.shape[3]; model.height = model.spec.input.shape[2];
      if (![model.width, model.height].every(n => Number.isInteger(n) && n > 0 && n <= 1024)) fail('modelInputShapeInvalid');
      model.decodeWH = model.spec.decodeWH || [model.width, model.height];
      if (!Array.isArray(model.decodeWH) || model.decodeWH.length !== 2 || !model.decodeWH.every(n => Number.isInteger(n) && n > 0 && n <= 2048)) fail('modelDecodeShapeInvalid');
      model.crop = cropTransform([0, 0, video.width, video.height], [model.width, model.height], model.spec.input.crop?.padding || 1.25);
      model.decodeCrop = { ...model.crop, inputWH: model.decodeWH };
    }
    const partModels = {};
    for (const [role, kind] of [['face', 'faceLandmarks'], ['foot', 'footView']]) {
      const spec = manifest.parts?.models?.find(m => m.kind === kind);
      if (!spec) fail('partModelManifestMissing');
      const bytes = await modelBytes(spec, options.manifest);
      const session = await ort.InferenceSession.create(bytes, { executionProviders: ['cpu'], intraOpNumThreads: options.threads, interOpNumThreads: 1, executionMode: 'sequential', graphOptimizationLevel: 'all', ...(spec.freeDimensionOverrides ? { freeDimensionOverrides: spec.freeDimensionOverrides } : {}) });
      sessions.push(session); partModels[role] = { role, spec, session, timings: [], preprocessing: [], validCalls: 0, skippedRois: 0, warmupCalls: 0, perSideCalls: { left: 0, right: 0 }, classCounts: { sole: 0, dorsum: 0, side: 0, unknown: 0 } };
    }
    stage = 'loadActualVRM';
    globalThis.self ||= globalThis;
    if (!globalThis.ProgressEvent) globalThis.ProgressEvent = class ProgressEvent extends Event { constructor(type, value = {}) { super(type); this.lengthComputable = !!value.lengthComputable; this.loaded = value.loaded || 0; this.total = value.total || 0; } };
    const [{ THREE }, { loadEmbeddedVRM, captureRig }] = await Promise.all([import('../dist/vendor/vrm-runtime.mjs'), import('../dist/vrm-view.mjs')]);
    const vrmBytes = await readFile(vrmPath), vrmBuffer = vrmBytes.buffer.slice(vrmBytes.byteOffset, vrmBytes.byteOffset + vrmBytes.byteLength);
    const vrm = await loadEmbeddedVRM(vrmBuffer, { headless: true }), group = new THREE.Group();
    group.add(vrm.scene); group.updateMatrixWorld(true); const rig = captureRig(vrm, group);
    if (!rig.bones.hips || Object.keys(rig.bones).length < 15) fail('actualVrmHumanoidMissing');
    const springCount = vrm.springBoneManager?.joints?.size ?? vrm.springBoneManager?.joints?.length ?? null;
    let springUpdates = 0;
    if (typeof vrm.springBoneManager?.update === 'function') {
      const updateSpring = vrm.springBoneManager.update.bind(vrm.springBoneManager);
      vrm.springBoneManager.update = delta => { springUpdates++; return updateSpring(delta); };
    }
    const observations = [], expressionFrames = [], partCompute = [], fastStandard = [], fastReference = [], standardReference = [];
    let decoded = 0, allOutputsFinite = true, warmupCalls = 0;
    let nextFaceAt = 0, nextFootAt = 0, latestFace = null, lastFaceTime = -Infinity;
    stage = 'actualVideoInference';
    for await (const rgb of decodedFrames(videoPath, video, options)) {
      const outputsByRole = {};
      const rotatedModels = models.slice(decoded % models.length).concat(models.slice(0, decoded % models.length));
      for (const model of rotatedModels) {
        const prepStart = performance.now();
        const data = rgbCropTensor(rgb, video.width, video.height, model.crop, model.spec.input.mean, model.spec.input.std);
        const tensor = new ort.Tensor('float32', data, [1, 3, model.height, model.width]);
        model.preprocessing.push(performance.now() - prepStart);
        if (decoded === 0) for (let i = 0; i < 3; i++) { const warm = await model.session.run({ [model.inputName]: tensor }); Object.values(warm).forEach(t => t.dispose?.()); warmupCalls++; }
        let output;
        try {
          const start = performance.now(); output = await model.session.run({ [model.inputName]: tensor }); model.timings.push(performance.now() - start);
          const split = model.spec.simcc_split_ratio || 2;
          const values = Object.values(output), x = values.find(t => t.dims.length === 3 && t.dims[1] === model.count && t.dims[2] === model.decodeWH[0] * split), y = values.find(t => t.dims.length === 3 && t.dims[1] === model.count && t.dims[2] === model.decodeWH[1] * split);
          if (!x || !y) fail('actualModelOutputShapeMismatch');
          allOutputsFinite &&= x.data.every(finite) && y.data.every(finite);
          if (!allOutputsFinite) fail('nonFiniteActualModelOutput');
          const points = decodeSimCC(x.data, y.data, model.decodeCrop, model.count, split);
          outputsByRole[model.role] = model.count === 133 ? points.map((p, id) => ({ ...p, id, source: 'observed2D' })) : toCoco133(points, model.spec.coco133_indices || COCO59);
        } finally { tensor.dispose?.(); if (output) Object.values(output).forEach(t => t.dispose?.()); }
      }
      fastStandard.push(...agreement(outputsByRole.fast59, outputsByRole.previous59, models[0].crop));
      fastReference.push(...agreement(outputsByRole.fast59, outputsByRole.reference133, models[0].crop));
      standardReference.push(...agreement(outputsByRole.previous59, outputsByRole.reference133, models[0].crop));
      const timestampMs = decoded * 1000 / options.fps, selectedPoints = outputsByRole[selectedModel.role];
      const points59 = (selectedModel.spec.coco133_indices || COCO59).map(id => selectedPoints[id]);
      let partsMs = 0;
      if (timestampMs + 1e-6 >= nextFaceAt) {
        const candidate = usablePartRoi(faceRoiFrom59(points59), video, 24);
        if (candidate) { latestFace = await runPart(partModels.face, rgb, video, candidate, ort); lastFaceTime = timestampMs; partsMs += latestFace.computeMs; }
        else { latestFace = null; partModels.face.skippedRois++; }
        do { nextFaceAt += 100; } while (nextFaceAt <= timestampMs + 1e-6);
      }
      if (timestampMs + 1e-6 >= nextFootAt) {
        for (const side of ['left', 'right']) {
          const candidate = usablePartRoi(footRoiFrom59(points59, side), video, 16);
          if (candidate) partsMs += (await runPart(partModels.foot, rgb, video, candidate, ort, side)).computeMs;
          else partModels.foot.skippedRois++;
        }
        do { nextFootAt += 200; } while (nextFootAt <= timestampMs + 1e-6);
      }
      const face = latestFace?.valid && timestampMs - lastFaceTime < 300 ? latestFace : null;
      expressionFrames.push(face ? { valid: true, blinkLeft: face.blinkLeft, blinkRight: face.blinkRight, mouthOpen: face.mouthOpen } : null);
      observations.push(observationWithFace(selectedPoints, face, timestampMs, video)); partCompute.push(partsMs);
      decoded++;
    }
    stage = 'actualVrmMotion';
    const vrmTimes = [], vrmByVideoFrame = Array(decoded).fill(0), currentReplayStats = rotationStatistics(rig); let vrmUpdates = 0, transformsFinite = true, checkedTransforms = 0, previousTime = null, expressionUpdates = 0;
    const stability = evaluateStabilitySequence(observations, rig, { renderFps: 30, minScore: .25, geometryOptions: { bodyHeightM: 1.7, scaleConfirmed: false }, onActualRender: result => {
      currentReplayStats.record(result, result.timestampMs);
      const dt = previousTime === null ? 1 / 30 : Math.min(.05, Math.max(0, (result.timestampMs - previousTime) / 1000)); previousTime = result.timestampMs;
      const index = Math.min(decoded - 1, Math.max(0, Math.floor(result.timestampMs * options.fps / 1000 + 1e-6))), face = expressionFrames[index];
      const start = performance.now(); vrm.humanoid.setNormalizedPose(result.normalizedPose);
      if (vrm.expressionManager) { vrm.expressionManager.setValue('blinkLeft', face?.valid ? face.blinkLeft : 0); vrm.expressionManager.setValue('blinkRight', face?.valid ? face.blinkRight : 0); vrm.expressionManager.setValue('aa', face?.valid ? face.mouthOpen : 0); if (face?.valid) expressionUpdates++; }
      vrm.update(dt); vrm.scene.updateMatrixWorld(true); const elapsed = performance.now() - start; vrmTimes.push(elapsed); vrmByVideoFrame[index] += elapsed; vrmUpdates++;
      vrm.scene.traverse(node => { checkedTransforms++; transformsFinite &&= node.matrixWorld.elements.every(finite) && node.quaternion.toArray().every(finite) && node.position.toArray().every(finite); });
    } });
    const modelResults = Object.fromEntries(models.map(m => [m.role, { id: m.spec.id, sha256: m.spec.sha256, inputShape: m.spec.input.shape, decodeWH: m.decodeWH, cpu: summary(m.timings), preprocessing: summary(m.preprocessing) }]));
    const partResults = Object.fromEntries(Object.entries(partModels).map(([role, m]) => [role, { id: m.spec.id, sha256: m.spec.sha256, scheduledHz: role === 'face' ? 10 : 5, frequencyUnit: role === 'face' ? 'one face' : 'per foot', actualHz: decoded ? m.timings.length / (decoded / options.fps) : 0, validCalls: m.validCalls, skippedInvalidRois: m.skippedRois, warmupCallsExcluded: m.warmupCalls, cpu: summary(m.timings), preprocessing: summary(m.preprocessing), ...(role === 'foot' ? { perSideCalls: m.perSideCalls, predictedClassCounts: m.classCounts } : {}) }]));
    const combinedSelected = selectedModel.timings.map((ms, i) => ms + selectedModel.preprocessing[i] + partCompute[i] + vrmByVideoFrame[i]);
    stage = 'legacyPoseReplay';
    const legacy = await loadLegacyComparison(workspace), legacyComparison = compareLegacyObservations(observations, rig, currentReplayStats.finish(), legacy);
    stage = 'derivedImageCropInference';
    const derivedImageScenarios = await derivedCropScenarios(videoPath, video, options, selectedModel, ort, rig, decoded, legacy);
    const comparison = values => ({ pointsCompared: values.length, meanCanonicalPixels: values.length ? values.reduce((a, b) => a + b, 0) / values.length : null, p95CanonicalPixels: percentile(values, .95), units: 'pixels in the common 192x256 source crop', isGroundTruthAccuracy: false });
    const checks = { atLeast72ActualDecodedFrames: decoded >= options.minFrames, allThreeModelsRanEveryFrame: models.every(m => m.timings.length === decoded), allOutputsFinite,
      selectedMatchesPublishedBody: selectedModel.spec.sha256 === manifest.body.sha256,
      selectedCandidateSpeedGate: !selectedIsFast || (modelResults.fast59.cpu.medianMs !== null && modelResults.fast59.cpu.medianMs < modelResults.previous59.cpu.medianMs),
      actualHumanoidLoaded: Object.keys(rig.bones).length >= 15, actualVrmUpdated: vrmUpdates >= decoded, actualSpringSystemUpdated: springUpdates >= vrmUpdates && springUpdates > 0, finiteVrmTransforms: transformsFinite, stableMotion: stability.pass,
      actualCroppedImageScenariosStable: Object.values(derivedImageScenarios).every(scenario => scenario.stable) };
    const report = { schemaVersion: 1, status: Object.values(checks).every(Boolean) ? 'passed' : 'failed', sourceCommit: process.env.GITHUB_SHA || null,
      runId: process.env.GITHUB_RUN_ID || null, checks,
      input: { kind: options.video ? 'privateEphemeralVideo' : 'pinnedOfficialPublicVideo', publicRevision: options.video ? null : PUBLIC_VIDEO.revision,
        decodedFrames: decoded, sampledFps: options.fps, sampledSeconds: decoded / options.fps, originalMediaRetained: false, frameImagesRetained: false, perFrameCoordinatesRetained: false },
      environment: { runtime: 'onnxruntime-node', runtimeVersion: ort.env.versions?.node || '1.22.0', executionProvider: 'CPUExecutionProvider', nodeVersion: process.version, intraOpThreads: options.threads, logicalCpus: cpus().length, cpuModel: cpus()[0]?.model || null, warmupCallsExcluded: warmupCalls },
      stabilityModel: { id: selectedModel.spec.id, sha256: selectedModel.spec.sha256, role: selectedModel.role, manifestRecommendedBody: manifest.recommendedBody || null, selectedByPublishedBodySha: true },
      models: modelResults, parts: partResults, combinedSelectedCompute: { ...summary(combinedSelected), definition: 'Sum of measured published-body inference/preprocessing, scheduled face/feet inference/preprocessing, and real VRM update compute per sampled video frame. Stages are timed separately; decoding, raster rendering, and candidate/reference-model comparison are excluded. Not wall-clock end-to-end latency.' }, comparison: { sameActualFrames: true, sameSourceCrop: true, modelOrderRotatedPerFrame: true, personDetectionIncluded: false,
        fastVsPreviousMedianRatio: modelResults.fast59.cpu.medianMs && modelResults.previous59.cpu.medianMs ? modelResults.fast59.cpu.medianMs / modelResults.previous59.cpu.medianMs : null,
        fastCandidateSelected: selectedIsFast, fastCandidateActuallyFaster: modelResults.fast59.cpu.medianMs !== null && modelResults.fast59.cpu.medianMs < modelResults.previous59.cpu.medianMs,
        fastVsPrevious: comparison(fastStandard), fastVsReference133: comparison(fastReference), previousVsReference133: comparison(standardReference) },
      vrm: { kind: options.vrm ? 'privateEphemeralAvatar' : 'pinnedOfficialPublicAvatar', publicRevision: options.vrm ? null : PUBLIC_VRM.revision,
        humanoidBones: Object.keys(rig.bones).length, springJointCount: springCount, springUpdateCalls: springUpdates, updateCalls: vrmUpdates, faceDrivenExpressionUpdates: expressionUpdates, updateCpu: summary(vrmTimes), transformsChecked: checkedTransforms, transformsFinite, rendererUsed: false, texturePixelsLoaded: false },
      stability, legacyComparison, derivedImageScenarios, accuracy: null, gpuUtilization: null, gpuPower: null,
      limitations: ['CPU timings are actual video inference; no user-GPU performance was measured.', 'Agreement with a reference model is not accuracy against annotations.', 'The wholebody path uses the full video rectangle as its person crop; no automatic person detector is included.', 'VRM humanoid and spring systems execute in Node; raster rendering and textures are excluded.'] };
    stage = 'writeAggregateReport'; await mkdir(dirname(resolve(options.output)), { recursive: true }); await writeFile(options.output, JSON.stringify(report, null, 2) + '\n', { mode: 0o600 });
    console.log(JSON.stringify({ status: report.status, decodedFrames: decoded, stabilityModelRole: selectedModel.role, fastMedianMs: modelResults.fast59.cpu.medianMs, previousMedianMs: modelResults.previous59.cpu.medianMs, referenceMedianMs: modelResults.reference133.cpu.medianMs, vrmUpdates, stabilityPassed: stability.pass, mediaOutput: false }));
    if (report.status !== 'passed') process.exitCode = 1;
  } catch (error) { error.evaluationStage = stage; throw error; }
  finally { console.warn = previousWarn; for (const session of sessions) try { await session.release(); } catch {} await rm(workspace, { recursive: true, force: true }); }
}

try { await main(argumentsFor(process.argv.slice(2))); }
catch (error) { console.error(JSON.stringify({ status: 'failed', stage: error.evaluationStage || 'initialization', errorCode: error.safeCode || 'headlessEvaluationFailed', privatePathsLogged: false, mediaOutput: false })); process.exitCode = 1; }
