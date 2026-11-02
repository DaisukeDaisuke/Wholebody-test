import { intrinsics4, undistortPixel, cameraAtImageSize, worldToCamera, cameraToWorld, projectWorldPoint } from './calibration.mjs';

const EPS = 1e-9;
const clamp = (v, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const mul = (a, s) => a.map(v => v * s);
const length = a => Math.hypot(...a);
const unit = a => length(a) > EPS ? mul(a, 1 / length(a)) : null;
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const lerp = (a, b, t) => a.map((v, i) => v * (1 - t) + b[i] * t);
const finite3 = a => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite);
const median = a => { const b = a.filter(Number.isFinite).sort((x, y) => x - y); return b.length ? (b[Math.floor((b.length - 1) / 2)] + b[Math.floor(b.length / 2)]) / 2 : null; };
const transposeMultiply = (m, v) => [m[0] * v[0] + m[3] * v[1] + m[6] * v[2], m[1] * v[0] + m[4] * v[1] + m[7] * v[2], m[2] * v[0] + m[5] * v[1] + m[8] * v[2]];
const visible = (p, threshold) => p && Number.isFinite(p.x ?? p.u) && Number.isFinite(p.y ?? p.v) && Number.isFinite(p.score) && p.score >= threshold && p.observable !== false;
const xy = p => [p.x ?? p.u, p.y ?? p.v];
const missing = (id, reason = 'unobserved') => ({ id, position: null, score: 0, source: 'missing', ageMs: 0, reason });
const sourceRank = { missing: 0, inferredPrior: 1, inferredKinematic: 2, predicted: 3, inferredMonocular: 4, estimatedMultiView: 5, measuredMultiView: 6 };
const weakestSource = points => points.reduce((s, p) => (sourceRank[p?.source] ?? 0) < (sourceRank[s] ?? 0) ? (p?.source || 'missing') : s, 'measuredMultiView');

export const COCO_CONNECTIONS = [
  [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [15, 19], [19, 17], [19, 18], [17, 18], [16, 22], [22, 20], [22, 21], [20, 21],
  ...[91, 112].flatMap(base => [0, 1, 2, 3, 4].flatMap(f => [[base, base + 1 + 4 * f], ...[1, 2, 3].map(j => [base + 4 * f + j, base + 4 * f + j + 1])])),
];

// Anthropometric ratios are weak priors, not measured user dimensions.
const BODY_BONES = [[5, 6, .225], [11, 12, .105], [5, 11, .29], [6, 12, .29], [5, 7, .186], [6, 8, .186], [7, 9, .146], [8, 10, .146], [11, 13, .245], [12, 14, .245], [13, 15, .246], [14, 16, .246], [0, 5, .17], [0, 6, .17], [15, 19, .04], [16, 22, .04], [19, 17, .15], [19, 18, .137], [22, 20, .15], [22, 21, .137], [17, 18, .05], [20, 21, .05]];
const HAND_RATIOS = [[.025, .020, .017, .015], [.047, .025, .016, .013], [.050, .028, .018, .014], [.046, .027, .018, .014], [.039, .021, .014, .012]];
const HAND_BONES = [91, 112].flatMap(base => HAND_RATIOS.flatMap((ratios, finger) => ratios.map((r, j) => [j === 0 ? base : base + finger * 4 + j, base + finger * 4 + j + 1, r])));
const ALL_BONES = [...BODY_BONES, ...HAND_BONES, [9, 91, .001], [10, 112, .001]];
const DEFAULTS = { minScore: .25, syncToleranceMs: 45, maxTimeUncertaintyMs: 20, minRayAngleDeg: 2, maxReprojectionErrorPx: 8, minCameraDepthM: .15, maxCameraDepthM: 30, maxSources: 8, allowApproximateCalibration: true };

function solve3(a, b) {
  const m = [[...a.slice(0, 3), b[0]], [...a.slice(3, 6), b[1]], [...a.slice(6, 9), b[2]]];
  const scale = Math.max(...a.map(Math.abs));
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;
    if (Math.abs(m[p][c]) < Math.max(1e-12, scale * 1e-8)) return null;
    [m[c], m[p]] = [m[p], m[c]];
    const d = m[c][c];
    for (let j = c; j < 4; j++) m[c][j] /= d;
    for (let r = 0; r < 3; r++) if (r !== c) { const f = m[r][c]; for (let j = c; j < 4; j++) m[r][j] -= f * m[c][j]; }
  }
  const p = m.map(r => r[3]);
  return finite3(p) ? p : null;
}

function rayForPoint(keypoint, camera) {
  const p = undistortPixel(xy(keypoint), camera.K, camera.distortion);
  if (!p) return null;
  const direction = unit(transposeMultiply(camera.Rcw, [p[0], p[1], 1]));
  return direction ? { origin: cameraToWorld([0, 0, 0], camera), direction, camera, keypoint, weight: Math.max(.001, clamp(keypoint.score) ** 2) } : null;
}

function intersectRays(rays) {
  const a = Array(9).fill(0), b = [0, 0, 0];
  for (const ray of rays) {
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      const v = ray.weight * ((i === j ? 1 : 0) - ray.direction[i] * ray.direction[j]);
      a[i * 3 + j] += v;
      b[i] += v * ray.origin[j];
    }
  }
  return solve3(a, b);
}

function rayAngle(a, b) { return Math.acos(clamp(Math.abs(dot(a.direction, b.direction)), 0, 1)) * 180 / Math.PI; }
function reprojection(point, ray, opts) {
  const cameraPoint = worldToCamera(point, ray.camera);
  if (cameraPoint[2] < opts.minCameraDepthM || cameraPoint[2] > opts.maxCameraDepthM) return Infinity;
  const pixel = projectWorldPoint(point, ray.camera), p = xy(ray.keypoint);
  return pixel ? Math.hypot(pixel[0] - p[0], pixel[1] - p[1]) : Infinity;
}

export function triangulateJoint(observations, cameras, options = {}) {
  const opts = { ...DEFAULTS, ...options }, cameraMap = new Map((Array.isArray(cameras) ? cameras : Object.values(cameras || {})).map(c => [c.id, c]));
  const rays = [], seen = new Set();
  for (const o of observations) {
    const camera = cameraMap.get(o.sourceId), kp = o.keypoint;
    if (!camera || seen.has(o.sourceId) || !visible(kp, opts.minScore)) continue;
    if (camera.calibrated !== true && !(opts.allowApproximateCalibration && camera.calibrationMethod === 'manualEstimate')) continue;
    let c;
    try { c = o.imageSize ? cameraAtImageSize(camera, o.imageSize) : camera; } catch { continue; }
    if (!c) continue;
    const ray = rayForPoint(kp, c);
    if (ray) { rays.push(ray); seen.add(o.sourceId); }
  }
  if (rays.length < 2) return { valid: false, reason: 'fewerThanTwoViews' };
  // Extrinsics only compose when their world frames agree. Legacy calibration
  // files with no IDs still share the explicit configured-world convention;
  // mixing that convention with a named board frame is not assumed safe.
  const worldCalibrationIds = [...new Set(rays.map(ray => ray.camera.worldCalibrationId || 'configured-world'))];
  if (worldCalibrationIds.length > 1) return { valid: false, reason: 'calibrationFrameMismatch', worldCalibrationIds };
  let best = null;
  for (let i = 0; i < rays.length; i++) for (let j = i + 1; j < rays.length; j++) {
    const angle = rayAngle(rays[i], rays[j]);
    if (angle < opts.minRayAngleDeg) continue;
    const point = intersectRays([rays[i], rays[j]]);
    if (!point) continue;
    const inliers = rays.filter(ray => reprojection(point, ray, opts) <= opts.maxReprojectionErrorPx);
    if (inliers.length < 2) continue;
    const quality = inliers.reduce((s, ray) => s + ray.weight / (1 + reprojection(point, ray, opts)), 0);
    if (!best || inliers.length > best.inliers.length || (inliers.length === best.inliers.length && quality > best.quality)) best = { point, inliers, quality, angle };
  }
  if (!best) return { valid: false, reason: 'geometryOrReprojectionRejected' };
  const point = intersectRays(best.inliers);
  if (!point) return { valid: false, reason: 'degenerateGeometry' };
  const errors = best.inliers.map(ray => reprojection(point, ray, opts));
  if (Math.max(...errors) > opts.maxReprojectionErrorPx) return { valid: false, reason: 'reprojectionRejected' };
  const rms = Math.sqrt(errors.reduce((s, e) => s + e * e, 0) / errors.length);
  let angle = 0;
  for (let i = 0; i < best.inliers.length; i++) for (let j = i + 1; j < best.inliers.length; j++) angle = Math.max(angle, rayAngle(best.inliers[i], best.inliers[j]));
  const depth = median(best.inliers.map(r => worldToCamera(point, r.camera)[2]));
  const focal = median(best.inliers.map(r => intrinsics4(r.camera.K)[0]));
  const source = best.inliers.every(r => r.camera.calibrated === true) ? 'measuredMultiView' : 'estimatedMultiView';
  return { valid: true, position: point, score: clamp(Math.min(...best.inliers.map(r => r.keypoint.score)) * Math.exp(-rms / opts.maxReprojectionErrorPx)), source, worldCalibrationId: worldCalibrationIds[0], reprojectionErrorPx: rms, rayAngleDeg: angle, uncertaintyM: Math.max(1, rms) * depth / (focal * Math.sin(angle * Math.PI / 180)), views: best.inliers.map(r => r.camera.id) };
}

function boneLength(a, b, ratio, height, overrides) {
  const supplied = overrides?.[`${a}-${b}`] ?? overrides?.[`${b}-${a}`];
  return Number.isFinite(supplied) && supplied > 0 && supplied < 2 ? supplied : ratio * height;
}

function fitRootDepth(points, rays, relativeDepths, height, initialDepth, previousDepth, options) {
  const hasModelDepth = relativeDepths.some(Number.isFinite);
  const bones = (hasModelDepth ? BODY_BONES.filter(([a, b]) => a >= 5 && b >= 5 && a < 17 && b < 17) : BODY_BONES.slice(0, 4)).filter(([a, b]) => rays[a] && rays[b] && (!hasModelDepth || (Number.isFinite(relativeDepths[a]) && Number.isFinite(relativeDepths[b]))));
  if (bones.length < 2) return { depth: Number.isFinite(previousDepth) ? previousDepth : initialDepth, boneCount: bones.length, priorOnly: true };
  const objective = depth => {
    let loss = 0, weightSum = 0;
    for (const [a, b, ratio] of bones) {
      const za = depth + (relativeDepths[a] || 0), zb = depth + (relativeDepths[b] || 0);
      if (za <= .05 * height || zb <= .05 * height) return Infinity;
      const expected = boneLength(a, b, ratio, height, options.boneLengths);
      const residual = length(sub(mul(rays[a], za), mul(rays[b], zb))) / expected - 1;
      const magnitude = Math.abs(residual), huber = magnitude <= .2 ? .5 * residual * residual : .2 * (magnitude - .1);
      const weight = Math.min(clamp(points[a].score), clamp(points[b].score));
      loss += weight * huber;
      weightSum += weight;
    }
    // A temporal value regularizes an ambiguous fit; it never replaces current scale evidence.
    const priorLoss = Number.isFinite(previousDepth) ? .002 * ((depth - previousDepth) / height) ** 2 : 0;
    return loss / Math.max(weightSum, EPS) + priorLoss;
  };
  const minimum = Math.max(height * .3, .05 * height - Math.min(0, ...relativeDepths.filter(Number.isFinite))), maximum = height * 15;
  const count = 48, step = (maximum - minimum) / count;
  let bestIndex = 0, bestLoss = Infinity;
  for (let i = 0; i <= count; i++) { const loss = objective(minimum + i * step); if (loss < bestLoss) { bestLoss = loss; bestIndex = i; } }
  let lo = minimum + Math.max(0, bestIndex - 1) * step, hi = minimum + Math.min(count, bestIndex + 1) * step;
  const ratio = (Math.sqrt(5) - 1) / 2;
  let a = hi - ratio * (hi - lo), b = lo + ratio * (hi - lo), fa = objective(a), fb = objective(b);
  for (let i = 0; i < 32; i++) {
    if (fa < fb) { hi = b; b = a; fb = fa; a = hi - ratio * (hi - lo); fa = objective(a); }
    else { lo = a; a = b; fa = fb; b = lo + ratio * (hi - lo); fb = objective(b); }
  }
  return { depth: (lo + hi) / 2, boneCount: bones.length, priorOnly: false, fitLoss: objective((lo + hi) / 2) };
}

/** A deliberately labelled inverse-kinematic monocular estimate.
 * No image-only method here treats unobserved depth as a measured value. */
export function estimateMonocularPose(observation, suppliedCamera = null, options = {}) {
  const opts = { ...DEFAULTS, ...options }, points = observation?.keypoints || [];
  const imageSize = observation?.imageSize || { width: 1, height: 1 };
  let camera = null;
  try { camera = suppliedCamera ? cameraAtImageSize(suppliedCamera, imageSize) : null; } catch { camera = null; }
  const previous = options.priorPose;
  const priorRoot = previous?.coordinateFrame === 'world' && previous?.metric && camera && finite3(previous.root?.position) ? worldToCamera(previous.root.position, camera) : null;
  const recentPrior = priorRoot && Number.isFinite(previous.timestampMs) && Number.isFinite(observation?.timestampMs) && observation.timestampMs >= previous.timestampMs && observation.timestampMs - previous.timestampMs <= 5000;
  const hasIntrinsics = !!camera && (camera.calibrated === true || camera.intrinsicsCalibrated === true);
  const explicitDepth = Number.isFinite(opts.rootDepthM) && opts.rootDepthM > .1 && opts.rootDepthM < 30;
  const knownScale = opts.scaleConfirmed === true && Number.isFinite(opts.bodyHeightM) && opts.bodyHeightM > .5 && opts.bodyHeightM < 2.6;
  const previousSizes = recentPrior ? BODY_BONES.filter(([, , ratio]) => ratio >= .1).flatMap(([a, b, ratio]) => finite3(previous.joints?.[a]?.position) && finite3(previous.joints?.[b]?.position) ? [length(sub(previous.joints[a].position, previous.joints[b].position)) / ratio] : []) : [];
  const previousHeight = previousSizes.length >= 3 ? clamp(median(previousSizes), .5, 2.6) : null;
  const metric = hasIntrinsics && (knownScale || explicitDepth || (recentPrior && previousHeight !== null));
  const height = knownScale ? opts.bodyHeightM : previousHeight || (explicitDepth ? (Number.isFinite(opts.bodyHeightM) ? opts.bodyHeightM : 1.7) : 1);
  // Relative skeleton units and calibrated absolute position are separate.
  // A measured body height supplies metre-scale bones even without known K;
  // RTMW3D's metre-valued relative depths must keep those same units.
  const bodyScaleInMetres = knownScale || explicitDepth || previousHeight !== null;
  const virtualFocal = Math.max(imageSize.width, imageSize.height);
  const c = camera || { K: [virtualFocal, virtualFocal, imageSize.width / 2, imageSize.height / 2], distortion: [0, 0, 0, 0, 0] };
  const rays = Array.from({ length: 133 }, (_, i) => {
    if (!visible(points[i], opts.minScore)) return null;
    const p = undistortPixel(xy(points[i]), c.K, c.distortion);
    return p ? [p[0], p[1], 1] : null;
  });
  const hipRay = rays[11] && rays[12] ? lerp(rays[11], rays[12], .5) : null;
  const shoulders = rays[5] && rays[6] ? lerp(rays[5], rays[6], .5) : null;
  // Only torso segments determine the initial scale, never a standing-height bbox.
  const depthGuesses = [[5, 6, .225], [11, 12, .105], [5, 11, .29], [6, 12, .29]].flatMap(([a, b, ratio]) => {
    if (!rays[a] || !rays[b]) return [];
    const separation = Math.hypot(rays[a][0] - rays[b][0], rays[a][1] - rays[b][1]);
    return separation > 1e-3 ? [boneLength(a, b, ratio, height, opts.boneLengths) / separation] : [];
  });
  const initialDepth = clamp(median(depthGuesses) || height * 2.5, height * .45, height * 12);
  const modelDepth = points.some(p => Number.isFinite(p?.relativeZ ?? p?.relativeZm));
  const rootRelativeZ = median([points[11]?.relativeZ ?? points[11]?.relativeZm, points[12]?.relativeZ ?? points[12]?.relativeZm]) || 0;
  const relativeDepths = Array.from({ length: 133 }, (_, i) => {
    const z = points[i]?.relativeZ ?? points[i]?.relativeZm;
    return Number.isFinite(z) ? (z - rootRelativeZ) * (bodyScaleInMetres ? 1 : 1 / 1.7) : null;
  });
  const depthFit = explicitDepth ? { depth: opts.rootDepthM, boneCount: 0, priorOnly: false } : fitRootDepth(points, rays, relativeDepths, height, initialDepth, recentPrior ? priorRoot[2] : null, opts);
  const hasScaleCue = explicitDepth || !depthFit.priorOnly || !!recentPrior;
  const rootDepth = depthFit.depth;
  const rootRay = hipRay || (recentPrior ? [priorRoot[0] / priorRoot[2], priorRoot[1] / priorRoot[2], 1] : shoulders ? [shoulders[0], shoulders[1] + .28 * height / rootDepth, 1] : [0, 0, 1]);
  const rootCamera = mul(rootRay, rootDepth);
  const depths = Array(133).fill(rootDepth), priorDepths = Array(133).fill(rootDepth);
  const previousCameraRootZ = recentPrior ? priorRoot[2] : median([previous?.cameraPositions?.[11]?.[2], previous?.cameraPositions?.[12]?.[2]]);
  for (let i = 0; i < 133; i++) {
    let prior = null;
    if (previous?.coordinateFrame === 'world' && camera && finite3(previous?.joints?.[i]?.position)) prior = worldToCamera(previous.joints[i].position, camera)[2];
    else if (previous?.sourceId === observation?.sourceId && finite3(previous?.cameraPositions?.[i])) prior = previous.cameraPositions[i][2];
    if (Number.isFinite(prior) && Number.isFinite(previousCameraRootZ)) prior += rootDepth - previousCameraRootZ;
    if (Number.isFinite(relativeDepths[i])) prior = rootDepth + relativeDepths[i];
    if (!Number.isFinite(prior)) {
      // Break the ambiguous front/back branch with a small anatomical prior.
      const relativePrior = [7, 8, 9, 10].includes(i) ? -.045 : [13, 14].includes(i) ? -.025 : [5, 6].includes(i) ? -.015 : i >= 91 ? -.045 : 0;
      prior = rootDepth + relativePrior * height;
    }
    depths[i] = priorDepths[i] = clamp(prior, .1 * height, 20 * height);
  }
  const bones = ALL_BONES.filter(([a, b]) => rays[a] && rays[b]);
  // Coordinate descent minimizes bone-length residuals along actual camera rays.
  // It preserves image reprojection and keeps the depth branch close to time/model priors.
  for (let pass = 0; pass < 14; pass++) {
    for (let i = 0; i < 133; i++) {
      if (!rays[i] || i === 11 || i === 12 || (i >= 23 && i <= 90)) continue;
      const regularizer = modelDepth ? .5 : .035;
      let gradient = regularizer * (depths[i] - priorDepths[i]), curvature = regularizer;
      for (const [a, b, ratio] of bones) {
        if (a !== i && b !== i) continue;
        const j = a === i ? b : a, delta = sub(mul(rays[i], depths[i]), mul(rays[j], depths[j]));
        const len = length(delta), expected = boneLength(a, b, ratio, height, opts.boneLengths);
        if (len < EPS) continue;
        const derivative = dot(delta, rays[i]) / len;
        const weight = Math.min(clamp(points[i].score), clamp(points[j].score));
        gradient += weight * (len - expected) * derivative;
        curvature += weight * derivative * derivative;
      }
      depths[i] = clamp(depths[i] - clamp(gradient / Math.max(curvature, EPS), -.035 * height, .035 * height), .1 * height, 20 * height);
    }
  }
  const absoluteAllowed = metric && hasScaleCue && (!!hipRay || !!recentPrior);
  const useWorld = absoluteAllowed && camera && Array.isArray(camera.Rcw) && Array.isArray(camera.tcw);
  const cameraPositions = rays.map((r, i) => r ? mul(r, depths[i]) : null);
  const transform = p => useWorld ? cameraToWorld(p, camera) : [p[0] - rootCamera[0], -(p[1] - rootCamera[1]), p[2] - rootCamera[2]];
  const joints = cameraPositions.map((p, id) => p ? { id, position: transform(p), score: clamp(points[id].score) * (modelDepth ? .65 : .4), source: 'inferredMonocular', ageMs: 0, observable2D: true, modelRelativeZ: Number.isFinite(points[id]?.relativeZ ?? points[id]?.relativeZm), depthAmbiguous: !modelDepth } : missing(id));
  const pose = { timestampMs: observation?.timestampMs ?? 0, sourceId: observation?.sourceId, joints, cameraPositions, metric: absoluteAllowed, coordinateFrame: useWorld ? 'world' : 'root-relative', units: absoluteAllowed ? 'm' : bodyScaleInMetres ? 'm-relative-estimate' : 'body-height', worldCalibrationId: useWorld ? (camera.worldCalibrationId || 'configured-world') : null, diagnostics: { method: modelDepth ? 'model-relative-depth-with-scale-fit' : 'ray-constrained-anthropometric-fit', intrinsicsCalibrated: hasIntrinsics, scaleConfirmed: knownScale, bodyScaleUnit: bodyScaleInMetres ? 'm' : 'body-height', modelRelativeDepthScale: bodyScaleInMetres ? 1 : 1 / 1.7, depthAnchored: explicitDepth, depthAnchorSource: explicitDepth ? 'providedCurrentMeasurement' : depthFit.priorOnly ? 'temporalPrior' : 'currentBoneFit', depthFitBoneCount: depthFit.boneCount, depthFitLoss: depthFit.fitLoss ?? null, depthAmbiguous: !modelDepth, scaleCueCount: depthGuesses.length, calibratedWorld: camera?.calibrated === true, observedJointCount: cameraPositions.filter(Boolean).length } };
  finishPose(pose);
  if (!absoluteAllowed) pose.root = { ...pose.root, position: null, valid: false, metric: false, distanceStatus: 'uncalibrated' };
  return pose;
}

/** Fuse time-aligned observations. An absent timestamp uncertainty is not zero. */
export function fusePose(observations, cameras = [], options = {}) {
  const opts = { ...DEFAULTS, ...options }, list = (observations || []).filter(o => o && Array.isArray(o.keypoints)).slice(0, opts.maxSources);
  const cameraList = Array.isArray(cameras) ? cameras : Object.values(cameras || {});
  const validTime = o => Number.isFinite(o.timestampMs) && Number.isFinite(o.timeUncertaintyMs ?? o.syncUncertaintyMs) && (o.timeUncertaintyMs ?? o.syncUncertaintyMs) >= 0 && (o.timeUncertaintyMs ?? o.syncUncertaintyMs) <= opts.maxTimeUncertaintyMs;
  const timed = list.filter(validTime);
  const targetTime = Number.isFinite(opts.timestampMs) ? opts.timestampMs : median(timed.map(o => o.timestampMs)) ?? median(list.map(o => o.timestampMs)) ?? 0;
  const sorted = timed.filter(o => Math.abs(o.timestampMs - targetTime) <= opts.syncToleranceMs).sort((a, b) => Math.abs(a.timestampMs - targetTime) - Math.abs(b.timestampMs - targetTime));
  const synced = [];
  for (const o of sorted) {
    if (synced.some(p => p.sourceId === o.sourceId)) continue;
    if (synced.every(p => Math.abs(p.timestampMs - o.timestampMs) + (p.timeUncertaintyMs ?? p.syncUncertaintyMs) + (o.timeUncertaintyMs ?? o.syncUncertaintyMs) <= opts.syncToleranceMs)) synced.push(o);
  }
  const worldCalibrationIds = [...new Set(synced.flatMap(o => {
    const camera = cameraList.find(c => c.id === o.sourceId);
    return camera && (camera.calibrated === true || (opts.allowApproximateCalibration && camera.calibrationMethod === 'manualEstimate')) ? [camera.worldCalibrationId || 'configured-world'] : [];
  }))];
  const calibrationFrameMismatch = worldCalibrationIds.length > 1;
  const bestObservation = list.slice().sort((a, b) => b.keypoints.reduce((s, p) => s + (visible(p, opts.minScore) ? p.score : 0), 0) - a.keypoints.reduce((s, p) => s + (visible(p, opts.minScore) ? p.score : 0), 0))[0];
  if (!bestObservation) return finishPose({ timestampMs: targetTime, joints: Array.from({ length: 133 }, (_, i) => missing(i)), metric: opts.priorPose?.metric || false, sourceId: opts.priorPose?.sourceId, coordinateFrame: opts.priorPose?.coordinateFrame || 'root-relative', units: opts.priorPose?.units || 'body-height', worldCalibrationId: opts.priorPose?.worldCalibrationId || null, diagnostics: { reason: 'noObservations', synchronizedSources: [] } });
  const bestCamera = cameraList.find(c => c.id === bestObservation.sourceId) || null;
  const joints = Array.from({ length: 133 }, (_, id) => {
    const result = calibrationFrameMismatch ? { valid: false, reason: 'calibrationFrameMismatch' } : synced.length >= 2 ? triangulateJoint(synced.map(o => ({ sourceId: o.sourceId, imageSize: o.imageSize, keypoint: o.keypoints[id] })), cameraList, opts) : { valid: false, reason: 'notSynchronized' };
    return result.valid ? { id, ...result, ageMs: 0 } : missing(id, result.reason);
  });
  const triangulatedCount = joints.filter(j => finite3(j.position)).length;
  const measuredRoot = finite3(joints[11].position) && finite3(joints[12].position) ? lerp(joints[11].position, joints[12].position, .5) : null;
  const sizes = BODY_BONES.flatMap(([a, b, ratio]) => finite3(joints[a].position) && finite3(joints[b].position) && ratio >= .1 ? [length(sub(joints[a].position, joints[b].position)) / ratio] : []);
  const learnedHeight = sizes.length >= 3 ? clamp(median(sizes), .5, 2.6) : null;
  const monoOptions = { ...opts };
  if (measuredRoot && bestCamera) {
    const depth = worldToCamera(measuredRoot, bestCamera)[2];
    if (depth > .1) monoOptions.rootDepthM = depth;
    if (learnedHeight && !opts.scaleConfirmed) { monoOptions.bodyHeightM = learnedHeight; monoOptions.scaleConfirmed = true; }
  }
  const mono = estimateMonocularPose(bestObservation, bestCamera, monoOptions);
  const diagnostics = { ...mono.diagnostics, synchronizedSources: synced.map(o => o.sourceId), rejectedTimeSources: list.filter(o => !synced.includes(o)).map(o => o.sourceId), calibrationFrameMismatch, worldCalibrationIds, maxSkewMs: synced.length > 1 ? Math.max(...synced.map(o => o.timestampMs)) - Math.min(...synced.map(o => o.timestampMs)) : null, triangulatedJointCount: triangulatedCount, learnedHeightM: learnedHeight, maxReprojectionErrorPx: opts.maxReprojectionErrorPx };
  if (!triangulatedCount) return { ...mono, timestampMs: targetTime, diagnostics };
  if (mono.coordinateFrame === 'world') for (let i = 0; i < 133; i++) if (!joints[i].position && mono.joints[i].position) joints[i] = mono.joints[i];
  return finishPose({ timestampMs: targetTime, joints, metric: true, coordinateFrame: 'world', units: 'm', worldCalibrationId: worldCalibrationIds[0] || 'configured-world', diagnostics });
}

function quaternionFromBasis(right, up, forward) {
  const m = [right[0], up[0], forward[0], right[1], up[1], forward[1], right[2], up[2], forward[2]], trace = m[0] + m[4] + m[8];
  let x, y, z, w, s;
  if (trace > 0) { s = Math.sqrt(trace + 1) * 2; w = .25 * s; x = (m[7] - m[5]) / s; y = (m[2] - m[6]) / s; z = (m[3] - m[1]) / s; }
  else if (m[0] > m[4] && m[0] > m[8]) { s = Math.sqrt(1 + m[0] - m[4] - m[8]) * 2; w = (m[7] - m[5]) / s; x = .25 * s; y = (m[1] + m[3]) / s; z = (m[2] + m[6]) / s; }
  else if (m[4] > m[8]) { s = Math.sqrt(1 + m[4] - m[0] - m[8]) * 2; w = (m[2] - m[6]) / s; x = (m[1] + m[3]) / s; y = .25 * s; z = (m[5] + m[7]) / s; }
  else { s = Math.sqrt(1 + m[8] - m[0] - m[4]) * 2; w = (m[3] - m[1]) / s; x = (m[2] + m[6]) / s; y = (m[5] + m[7]) / s; z = .25 * s; }
  const q = [x, y, z, w], n = Math.hypot(...q);
  return n > EPS && q.every(Number.isFinite) ? q.map(v => v / n) : null;
}

function footPose(joints, side) {
  const ids = side === 'left' ? [15, 17, 18, 19] : [16, 20, 21, 22], [ankle, big, small, heel] = ids.map(id => joints[id]);
  if (![ankle, big, small, heel].every(j => finite3(j?.position))) return { valid: false, source: 'missing', toes: [], instep: null };
  const toeCenter = lerp(big.position, small.position, .5), forward = unit(sub(toeCenter, heel.position));
  const lateral = side === 'left' ? sub(big.position, small.position) : sub(small.position, big.position);
  let right = unit(lateral), up = forward && right ? unit(cross(forward, right)) : null;
  if (!forward || !right || !up) return { valid: false, source: 'missing', toes: [], instep: null };
  if (dot(up, sub(ankle.position, heel.position)) < 0) up = mul(up, -1);
  right = unit(cross(up, forward));
  const rotation = quaternionFromBasis(right, up, forward), footLength = length(sub(toeCenter, heel.position));
  const toes = ['big', 'second', 'third', 'fourth', 'little'].map((name, i) => {
    const tip = lerp(big.position, small.position, i / 4);
    // Only big/little tips have COCO landmarks; other toes and all toe joints are priors.
    if (i > 0 && i < 4) for (let k = 0; k < 3; k++) tip[k] += forward[k] * footLength * [0, .025, .015, -.01, 0][i];
    const toeLength = footLength * [.23, .25, .22, .19, .16][i];
    const base = sub(tip, mul(forward, toeLength));
    const chain = [base, lerp(base, tip, .42), lerp(base, tip, .75), tip].map((p, j) => ({ position: p, source: j === 3 && (i === 0 || i === 4) ? (i === 0 ? big.source : small.source) : 'inferredAnatomy' }));
    return { name, position: tip, joints: chain, source: 'inferredAnatomy', tipObserved: i === 0 || i === 4, articulationStatus: 'unobserved', score: Math.min(big.score, small.score) * .35 };
  });
  return { valid: true, position: ankle.position.slice(), rotation, rotationXYZW: rotation, right, up, forward, source: weakestSource([ankle, big, small, heel]), score: Math.min(ankle.score, big.score, small.score, heel.score), instep: { position: lerp(ankle.position, toeCenter, .45), source: 'inferredAnatomy' }, toes };
}

function handPose(joints, side) {
  const base = side === 'left' ? 91 : 112, wrist = joints[base], index = joints[base + 5], middle = joints[base + 9], pinky = joints[base + 17];
  const names = ['thumb', 'index', 'middle', 'ring', 'pinky'];
  const fingers = names.map((name, f) => {
    const chain = [base, ...[1, 2, 3, 4].map(j => base + f * 4 + j)].map(id => joints[id]);
    const bends = [];
    for (let i = 1; i < chain.length - 1; i++) {
      if (![chain[i - 1], chain[i], chain[i + 1]].every(j => finite3(j?.position))) { bends.push(null); continue; }
      const a = unit(sub(chain[i - 1].position, chain[i].position)), b = unit(sub(chain[i + 1].position, chain[i].position));
      bends.push(a && b ? 1 - Math.acos(clamp(dot(a, b), -1, 1)) / Math.PI : null);
    }
    return { name, joints: chain.map(j => ({ id: j?.id, position: j?.position || null, source: j?.source || 'missing', score: j?.score || 0 })), bends, source: weakestSource(chain), score: Math.min(...chain.map(j => j?.score || 0)) };
  });
  if (![wrist, index, middle, pinky].every(j => finite3(j?.position))) return { valid: false, wrist: wrist?.position || null, rotation: null, fingers, source: 'missing' };
  const forward = unit(sub(middle.position, wrist.position)), lateral = side === 'left' ? sub(index.position, pinky.position) : sub(pinky.position, index.position);
  const up = forward ? unit(cross(forward, lateral)) : null, right = up && forward ? unit(cross(up, forward)) : null;
  const rotation = right && up && forward ? quaternionFromBasis(right, up, forward) : null;
  return { valid: !!rotation, wrist: wrist.position.slice(), position: wrist.position.slice(), rotation, right, up, forward, fingers, source: weakestSource([wrist, index, middle, pinky]), score: Math.min(wrist.score, index.score, middle.score, pinky.score) };
}

function finishPose(pose) {
  const [l, r] = [pose.joints[11], pose.joints[12]];
  if (finite3(l?.position) && finite3(r?.position)) {
    const source = weakestSource([l, r]), score = Math.min(l.score, r.score);
    pose.root = { position: lerp(l.position, r.position, .5), source, score, valid: score >= .05, metric: pose.metric === true, distanceStatus: !pose.metric ? 'uncalibrated' : source === 'measuredMultiView' ? 'measuredMultiView' : source === 'estimatedMultiView' ? 'estimatedMultiView' : source === 'inferredMonocular' ? 'estimatedMonocular' : 'estimatedTemporal' };
  } else pose.root = { position: null, source: 'missing', score: 0, valid: false, metric: pose.metric === true, distanceStatus: pose.metric ? 'missing' : 'uncalibrated' };
  pose.feet = { left: footPose(pose.joints, 'left'), right: footPose(pose.joints, 'right') };
  pose.hands = { left: handPose(pose.joints, 'left'), right: handPose(pose.joints, 'right') };
  const counts = {};
  for (const j of pose.joints) counts[j.source] = (counts[j.source] || 0) + 1;
  pose.quality = { counts, observed3D: counts.measuredMultiView || 0, estimated3D: pose.joints.filter(j => j.position && j.source !== 'measuredMultiView').length, missing: counts.missing || 0 };
  return pose;
}

function anchorFor(id, pose) {
  const ids = id >= 92 && id <= 111 ? [91, 9] : id >= 113 && id <= 132 ? [112, 10] : id === 91 ? [9] : id === 112 ? [10] : [17, 18, 19].includes(id) ? [15] : [20, 21, 22].includes(id) ? [16] : [];
  const part = id >= 92 && id <= 111 ? pose.hands?.left : id >= 113 && id <= 132 ? pose.hands?.right : [17, 18, 19].includes(id) ? pose.feet?.left : [20, 21, 22].includes(id) ? pose.feet?.right : null;
  const basis = part?.valid && [part.right, part.up, part.forward].every(finite3) ? [part.right, part.up, part.forward] : null;
  for (const anchorId of ids) if (finite3(pose.joints[anchorId]?.position)) return { id: anchorId, position: pose.joints[anchorId].position, source: pose.joints[anchorId].source, basis };
  const hips = [pose.joints[11], pose.joints[12]], shoulders = [pose.joints[5], pose.joints[6]];
  const root = finite3(pose.root?.position) ? pose.root.position : hips.every(p => finite3(p?.position)) ? lerp(hips[0].position, hips[1].position, .5) : null;
  if (root) {
    let torsoBasis = null;
    if ([...hips, ...shoulders].every(p => finite3(p?.position))) {
      const right = unit(sub(hips[1].position, hips[0].position)), upHint = sub(lerp(shoulders[0].position, shoulders[1].position, .5), root);
      const forward = right ? unit(cross(right, upHint)) : null, up = forward && right ? unit(cross(forward, right)) : null;
      if (right && up && forward) torsoBasis = [right, up, forward];
    }
    return { id: -1, position: root, source: pose.root?.source || weakestSource(hips), basis: torsoBasis };
  }
  return null;
}

/** Preserve motion through brief occlusion, retaining the last actual update time.
 * Predicted/held values never become fresh observations and cannot live indefinitely. */
export class TemporalPoseTracker {
  constructor({ predictionMs = 180, holdMs = 1200, priorMs = 5000, smoothingMs = 28, maxSpeed = 8 } = {}) {
    this.options = { predictionMs, holdMs, priorMs, smoothingMs, maxSpeed };
    this.reset();
  }
  reset() { this.states = Array(133).fill(null); this.lastPose = null; this.frameKey = null; this.lastTimeMs = null; this.boneLengths = {}; }
  getBoneLengths() { return { ...this.boneLengths }; }
  update(input, timestampMs = input?.timestampMs) {
    if (!input || !Array.isArray(input.joints) || !Number.isFinite(timestampMs)) return input;
    const key = `${input.coordinateFrame}:${input.worldCalibrationId || ''}:${input.coordinateFrame === 'world' ? '' : input.sourceId || ''}:${input.units}`;
    if (key !== this.frameKey || (this.lastTimeMs !== null && (timestampMs < this.lastTimeMs || timestampMs - this.lastTimeMs > this.options.priorMs))) this.reset();
    this.frameKey = key;
    if (this.lastTimeMs === timestampMs && this.lastPose) return this.lastPose;
    const pose = { ...input, timestampMs, joints: input.joints.map((j, i) => ({ ...j, id: i, position: j.position?.slice() || null })) };
    finishPose(pose);
    for (let id = 0; id < 133; id++) {
      const current = pose.joints[id] || missing(id), state = this.states[id];
      const fresh = finite3(current.position) && ['measuredMultiView', 'estimatedMultiView', 'inferredMonocular'].includes(current.source);
      if (fresh) {
        const dt = state ? Math.max(.001, (timestampMs - state.lastObservationMs) / 1000) : 0;
        let position = current.position.slice(), velocity = [0, 0, 0];
        if (state && dt < 1) {
          const rawVelocity = mul(sub(position, state.position), 1 / dt), speed = length(rawVelocity);
          const limited = speed > this.options.maxSpeed ? mul(rawVelocity, this.options.maxSpeed / speed) : rawVelocity;
          velocity = lerp(state.velocity, limited, .45);
          const adaptiveMs = this.options.smoothingMs / (1 + Math.min(6, length(velocity)));
          const alpha = 1 - Math.exp(-dt * 1000 / Math.max(1, adaptiveMs));
          position = lerp(state.position, position, alpha);
        }
        current.position = position;
        current.ageMs = 0;
        this.states[id] = { position: position.slice(), velocity, lastObservationMs: timestampMs, score: current.score, source: current.source, anchor: anchorFor(id, input) };
        continue;
      }
      if (!state) { pose.joints[id] = missing(id); continue; }
      const age = Math.max(0, timestampMs - state.lastObservationMs);
      if (age > this.options.priorMs) { pose.joints[id] = missing(id, 'priorExpired'); continue; }
      const anchor = anchorFor(id, input), storedAnchor = state.anchor;
      let position, source;
      if (anchor && storedAnchor && anchor.id === storedAnchor.id && anchor.source !== 'inferredPrior') {
        const offset = sub(state.position, storedAnchor.position);
        if (anchor.basis && storedAnchor.basis) {
          const local = storedAnchor.basis.map(axis => dot(offset, axis));
          position = add(anchor.position, add(add(mul(anchor.basis[0], local[0]), mul(anchor.basis[1], local[1])), mul(anchor.basis[2], local[2])));
        } else position = add(state.position, sub(anchor.position, storedAnchor.position));
        source = age <= this.options.holdMs ? 'inferredKinematic' : 'inferredPrior';
      } else if (age <= this.options.predictionMs) {
        const tau = .12, seconds = age / 1000;
        position = add(state.position, mul(state.velocity, tau * (1 - Math.exp(-seconds / tau))));
        source = 'predicted';
      } else {
        position = add(state.position, mul(state.velocity, .12 * (1 - Math.exp(-this.options.predictionMs / 120))));
        source = age <= this.options.holdMs ? 'inferredKinematic' : 'inferredPrior';
      }
      pose.joints[id] = { id, position, source, ageMs: age, score: state.score * (source === 'predicted' ? .7 : .5) * Math.exp(-age / Math.max(1, this.options.holdMs)), observable2D: false, reason: 'occlusion' };
    }
    // Constrain only inferred joints. A measured landmark is never moved to fit a prior.
    const isHeld = j => ['predicted', 'inferredKinematic', 'inferredPrior'].includes(j.source);
    for (let pass = 0; pass < 2; pass++) for (const [a, b] of ALL_BONES) {
      const ja = pose.joints[a], jb = pose.joints[b];
      if (!finite3(ja?.position) || !finite3(jb?.position) || isHeld(ja) === isHeld(jb)) continue;
      const stateA = this.states[a], stateB = this.states[b];
      const expected = this.boneLengths[`${a}-${b}`] || (stateA && stateB ? length(sub(stateA.position, stateB.position)) : null);
      if (!Number.isFinite(expected) || expected < .001) continue;
      const held = isHeld(ja) ? ja : jb, observed = isHeld(ja) ? jb : ja, direction = unit(sub(held.position, observed.position));
      if (direction) held.position = add(observed.position, mul(direction, expected));
    }
    finishPose(pose);
    // Adapt user bone lengths only from calibrated simultaneous measurements.
    for (const [a, b] of ALL_BONES) if (input.joints[a]?.source === 'measuredMultiView' && input.joints[b]?.source === 'measuredMultiView' && finite3(input.joints[a].position) && finite3(input.joints[b].position)) {
      const observed = length(sub(input.joints[a].position, input.joints[b].position)), key = `${a}-${b}`;
      if (observed > .001 && observed < 1.5) this.boneLengths[key] = this.boneLengths[key] ? this.boneLengths[key] * .97 + observed * .03 : observed;
    }
    if (!input.metric) pose.root = { ...pose.root, position: null, valid: false, metric: false, distanceStatus: 'uncalibrated' };
    this.lastPose = pose;
    this.lastTimeMs = timestampMs;
    return pose;
  }
}

/** NTP-style clock map. This cannot identify which video frame a side-channel packet describes. */
export class ClockSync {
  constructor({ windowSize = 40 } = {}) { this.windowSize = windowSize; this.samples = []; }
  addSample({ t0, t1, t2, t3 }) {
    if (![t0, t1, t2, t3].every(Number.isFinite) || t3 < t0 || t2 < t1) return false;
    const rttMs = (t3 - t0) - (t2 - t1);
    if (rttMs < 0 || rttMs > 2000) return false;
    this.samples.push({ remoteMs: (t1 + t2) / 2, localMs: (t0 + t3) / 2, rttMs });
    if (this.samples.length > this.windowSize) this.samples.shift();
    return true;
  }
  estimate() {
    if (!this.samples.length) return { ready: false, offsetMs: null, drift: 1, uncertaintyMs: Infinity };
    const sorted = this.samples.slice().sort((a, b) => a.rttMs - b.rttMs), chosen = sorted.slice(0, Math.max(1, Math.ceil(sorted.length * .35)));
    const remote = chosen.reduce((s, p) => s + p.remoteMs / chosen.length, 0), local = chosen.reduce((s, p) => s + p.localMs / chosen.length, 0);
    const variance = chosen.reduce((s, p) => s + (p.remoteMs - remote) ** 2, 0);
    const drift = variance > 1e6 && chosen.length >= 4 ? clamp(chosen.reduce((s, p) => s + (p.remoteMs - remote) * (p.localMs - local), 0) / variance, .999, 1.001) : 1;
    const residual = Math.sqrt(chosen.reduce((s, p) => s + (p.localMs - (local + drift * (p.remoteMs - remote))) ** 2, 0) / chosen.length);
    return { ready: this.samples.length >= 3, offsetMs: local - remote, drift, remoteOriginMs: remote, localOriginMs: local, uncertaintyMs: sorted[0].rttMs / 2 + residual, bestRttMs: sorted[0].rttMs, samples: this.samples.length };
  }
  toLocal(remoteMs) {
    const e = this.estimate();
    return e.ready && Number.isFinite(remoteMs) ? { timestampMs: e.localOriginMs + e.drift * (remoteMs - e.remoteOriginMs), timeUncertaintyMs: e.uncertaintyMs } : { timestampMs: null, timeUncertaintyMs: Infinity };
  }
}
