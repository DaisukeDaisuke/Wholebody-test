// Camera coordinates: x right, y down, z forward. Distances are metres.
// worldFromCamera.rotation is a row-major orthonormal 3x3 matrix.
const EPS = 1e-10;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const finite3 = (v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);
const add = (a, b) => a.map((x, i) => x + b[i]);
const sub = (a, b) => a.map((x, i) => x - b[i]);
const scale = (a, k) => a.map((x) => x * k);
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const norm = (a) => Math.hypot(...a);
const unit = (a) => norm(a) > EPS ? scale(a, 1 / norm(a)) : null;
const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const mv = (m, v) => [0, 1, 2].map((r) => m[r * 3] * v[0] + m[r * 3 + 1] * v[1] + m[r * 3 + 2] * v[2]);
const mtv = (m, v) => [0, 1, 2].map((r) => m[r] * v[0] + m[r + 3] * v[1] + m[r + 6] * v[2]);
const invalid = (reason, extra = {}) => ({ valid: false, xyz: null, confidence: 0, source: 'missing', reason, ...extra });

export const BODY_BONES = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
export const SKELETON_EDGES = [
  [0, 1], [0, 2], [1, 3], [2, 4], ...BODY_BONES,
  [15, 17], [15, 18], [15, 19], [17, 18], [19, 17], [19, 18],
  [16, 20], [16, 21], [16, 22], [20, 21], [22, 20], [22, 21],
  [9, 91], [10, 112],
  ...[91, 112].flatMap((base) => [1, 5, 9, 13, 17].flatMap((start) => [
    [base, base + start], [base + start, base + start + 1],
    [base + start + 1, base + start + 2], [base + start + 2, base + start + 3],
  ])),
];

// These proportions are an initialization prior, not measurements of the user.
export function anthropometricBoneLengths(heightM = 1.7) {
  if (!Number.isFinite(heightM) || heightM < 0.6 || heightM > 2.5) throw new Error('身長は 0.6〜2.5 m で指定してください。');
  const factors = { '5-6': 0.215, '5-7': 0.186, '7-9': 0.146, '6-8': 0.186, '8-10': 0.146, '5-11': 0.288, '6-12': 0.288, '11-12': 0.165, '11-13': 0.245, '13-15': 0.246, '12-14': 0.245, '14-16': 0.246 };
  return Object.fromEntries(Object.entries(factors).map(([key, value]) => [key, value * heightM]));
}

export function validateCamera(camera, requireWorld = false) {
  if (!camera || ![camera.fx, camera.fy, camera.cx, camera.cy].every(Number.isFinite) || camera.fx <= 0 || camera.fy <= 0) return false;
  if (camera.skew != null && !Number.isFinite(camera.skew)) return false;
  const d = camera.distortion || [];
  if (!Array.isArray(d) || d.length > 5 || !d.every(Number.isFinite)) return false;
  if (!requireWorld && !camera.worldFromCamera) return true;
  const transform = camera.worldFromCamera;
  if (!transform || !finite3(transform.translation) || !Array.isArray(transform.rotation) || transform.rotation.length !== 9 || !transform.rotation.every(Number.isFinite)) return false;
  const r = transform.rotation;
  const rows = [r.slice(0, 3), r.slice(3, 6), r.slice(6, 9)];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) if (Math.abs(dot(rows[i], rows[j]) - (i === j ? 1 : 0)) > 0.02) return false;
  const determinant = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
  return determinant > 0.98 && determinant < 1.02;
}

function distortion(x, y, coefficients = []) {
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = coefficients;
  const r2 = x * x + y * y;
  const radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
  return [x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y];
}

export function undistortPixel(u, v, camera) {
  if (!validateCamera(camera) || !Number.isFinite(u) || !Number.isFinite(v)) return null;
  const yd = (v - camera.cy) / camera.fy;
  const xd = (u - camera.cx - (camera.skew || 0) * yd) / camera.fx;
  let x = xd, y = yd;
  // Newton steps also handle moderate wide-angle radial distortion.
  for (let iteration = 0; iteration < 12; iteration++) {
    const [dx, dy] = distortion(x, y, camera.distortion);
    const ex = dx - xd, ey = dy - yd;
    if (Math.hypot(ex, ey) < 1e-11) break;
    const h = 1e-6;
    const px = distortion(x + h, y, camera.distortion);
    const py = distortion(x, y + h, camera.distortion);
    const a = (px[0] - dx) / h, b = (py[0] - dx) / h, c = (px[1] - dy) / h, d = (py[1] - dy) / h;
    const det = a * d - b * c;
    if (Math.abs(det) < EPS) return null;
    x -= clamp((d * ex - b * ey) / det, -0.5, 0.5);
    y -= clamp((-c * ex + a * ey) / det, -0.5, 0.5);
    if (!Number.isFinite(x + y) || Math.abs(x) > 20 || Math.abs(y) > 20) return null;
  }
  const checked = distortion(x, y, camera.distortion);
  if (Math.hypot(checked[0] - xd, checked[1] - yd) > 1e-5) return null;
  return [x, y];
}

export function projectPoint(xyz, camera) {
  if (!finite3(xyz) || !validateCamera(camera)) return null;
  const transform = camera.worldFromCamera;
  const p = transform ? mtv(transform.rotation, sub(xyz, transform.translation)) : xyz;
  if (p[2] <= 1e-6) return null;
  const [x, y] = distortion(p[0] / p[2], p[1] / p[2], camera.distortion);
  return { u: camera.fx * x + (camera.skew || 0) * y + camera.cx, v: camera.fy * y + camera.cy, depthM: p[2] };
}

export function pixelToWorldRay(u, v, camera) {
  if (!validateCamera(camera, true)) return null;
  const normalized = undistortPixel(u, v, camera);
  if (!normalized) return null;
  return { origin: [...camera.worldFromCamera.translation], direction: unit(mv(camera.worldFromCamera.rotation, [...normalized, 1])) };
}

function solve3(a, b) {
  const rows = [0, 1, 2].map((r) => [...a.slice(r * 3, r * 3 + 3), b[r]]);
  let largestPivot = 0, smallestPivot = Infinity;
  for (let column = 0; column < 3; column++) {
    let pivot = column;
    for (let row = column + 1; row < 3; row++) if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) pivot = row;
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
    const value = Math.abs(rows[column][column]);
    largestPivot = Math.max(largestPivot, value);
    smallestPivot = Math.min(smallestPivot, value);
    if (value < EPS) return null;
    const divisor = rows[column][column];
    for (let k = column; k < 4; k++) rows[column][k] /= divisor;
    for (let row = 0; row < 3; row++) {
      if (row === column) continue;
      const factor = rows[row][column];
      for (let k = column; k < 4; k++) rows[row][k] -= factor * rows[column][k];
    }
  }
  if (smallestPivot / largestPivot < 1e-7) return null;
  const xyz = rows.map((r) => r[3]);
  return finite3(xyz) ? xyz : null;
}

function intersectRays(observations) {
  const a = Array(9).fill(0), b = [0, 0, 0];
  for (const observation of observations) {
    const { direction: d, origin: c } = observation.ray;
    const weight = observation.score ** 2;
    for (let r = 0; r < 3; r++) for (let k = 0; k < 3; k++) {
      const coefficient = weight * ((r === k ? 1 : 0) - d[r] * d[k]);
      a[r * 3 + k] += coefficient;
      b[r] += coefficient * c[k];
    }
  }
  return solve3(a, b);
}

function angleBetweenRays(a, b) {
  return Math.acos(clamp(Math.abs(dot(a.direction, b.direction)), 0, 1)) * 180 / Math.PI;
}

function cameraFor(cameras, id) {
  if (cameras instanceof Map) return cameras.get(id);
  if (Array.isArray(cameras)) return cameras.find((camera) => camera.id === id);
  return cameras?.[id];
}

// Only simultaneous observations of the same joint belong in this call.
export function triangulateJoint(observations, cameras, options = {}) {
  const minimumScore = options.minimumScore ?? 0.25;
  const maxTimeDeltaMs = options.maxTimeDeltaMs ?? 25;
  const maxReprojectionErrorPx = options.maxReprojectionErrorPx ?? 8;
  const minimumRayAngleDeg = options.minimumRayAngleDeg ?? 2;
  const minimumBaselineM = options.minimumBaselineM ?? 0.02;
  const unique = new Map();
  for (const candidate of observations || []) {
    const camera = cameraFor(cameras, candidate.sourceId);
    if (!camera?.calibrated || !validateCamera(camera, true) || candidate.observable === false || candidate.visible === false) continue;
    if (![candidate.u, candidate.v, candidate.score, candidate.timeMs].every(Number.isFinite) || candidate.score < minimumScore) continue;
    if (Number.isFinite(options.referenceTimeMs) && Math.abs(candidate.timeMs - options.referenceTimeMs) > maxTimeDeltaMs) continue;
    const ray = pixelToWorldRay(candidate.u, candidate.v, camera);
    if (!ray) continue;
    const observation = { ...candidate, camera, ray, score: clamp(candidate.score, 0, 1) };
    if (!unique.has(candidate.sourceId) || unique.get(candidate.sourceId).score < observation.score) unique.set(candidate.sourceId, observation);
  }
  const sorted = [...unique.values()].sort((a, b) => a.timeMs - b.timeMs);
  if (sorted.length < 2) return invalid('insufficient-calibrated-views');
  let synchronized = [];
  for (let start = 0; start < sorted.length; start++) {
    const group = sorted.slice(start).filter((o) => o.timeMs - sorted[start].timeMs <= maxTimeDeltaMs);
    if (group.length > synchronized.length || (group.length === synchronized.length && group.reduce((s, o) => s + o.score, 0) > synchronized.reduce((s, o) => s + o.score, 0))) synchronized = group;
  }
  if (synchronized.length < 2) return invalid('unsynchronized-views');
  let best = null;
  for (let i = 0; i < synchronized.length; i++) for (let j = i + 1; j < synchronized.length; j++) {
    const pair = [synchronized[i], synchronized[j]];
    if (norm(sub(pair[0].ray.origin, pair[1].ray.origin)) < minimumBaselineM || angleBetweenRays(pair[0].ray, pair[1].ray) < minimumRayAngleDeg) continue;
    const xyz = intersectRays(pair);
    if (!xyz) continue;
    const inliers = [];
    let weightedError = 0, totalWeight = 0;
    for (const observation of synchronized) {
      const projection = projectPoint(xyz, observation.camera);
      if (!projection) continue;
      const error = Math.hypot(projection.u - observation.u, projection.v - observation.v);
      if (error > maxReprojectionErrorPx) continue;
      inliers.push(observation);
      weightedError += observation.score * error;
      totalWeight += observation.score;
    }
    if (inliers.length < 2) continue;
    const ranking = totalWeight - weightedError / (maxReprojectionErrorPx * 10);
    if (!best || ranking > best.ranking) best = { inliers, ranking };
  }
  if (!best) return invalid('degenerate-or-inconsistent-rays');
  const xyz = intersectRays(best.inliers);
  if (!xyz) return invalid('ill-conditioned-triangulation');
  let squaredError = 0, totalWeight = 0, largestAngle = 0;
  for (let i = 0; i < best.inliers.length; i++) {
    const observation = best.inliers[i];
    const projection = projectPoint(xyz, observation.camera);
    if (!projection) return invalid('behind-camera');
    const error = Math.hypot(projection.u - observation.u, projection.v - observation.v);
    if (error > maxReprojectionErrorPx) return invalid('reprojection-error');
    squaredError += error * error * observation.score;
    totalWeight += observation.score;
    for (let j = i + 1; j < best.inliers.length; j++) largestAngle = Math.max(largestAngle, angleBetweenRays(observation.ray, best.inliers[j].ray));
  }
  if (largestAngle < minimumRayAngleDeg) return invalid('insufficient-ray-angle');
  const reprojectionErrorPx = Math.sqrt(squaredError / totalWeight);
  const angleQuality = Math.min(1, Math.sin(largestAngle * Math.PI / 180) / Math.sin(20 * Math.PI / 180));
  return {
    valid: true, xyz, source: 'triangulated', confidence: clamp(Math.min(...best.inliers.map((o) => o.score)) * angleQuality * Math.exp(-reprojectionErrorPx / maxReprojectionErrorPx), 0, 1),
    reprojectionErrorPx, rayAngleDeg: largestAngle, sources: best.inliers.map((o) => o.sourceId),
    timeMs: best.inliers.reduce((s, o) => s + o.timeMs * o.score, 0) / totalWeight,
    timeSpreadMs: Math.max(...best.inliers.map((o) => o.timeMs)) - Math.min(...best.inliers.map((o) => o.timeMs)),
  };
}

function huber(residual, delta) {
  const absolute = Math.abs(residual);
  return absolute <= delta ? residual * residual / 2 : delta * (absolute - delta / 2);
}

// A two-dimensional-only model cannot supply relative depth. The optional planar
// prior is explicit and receives its own source and lower confidence.
export function estimateRootDepth(keypoints, camera, boneLengths, options = {}) {
  const failure = (reason) => ({ valid: false, depthM: null, joints: [], distanceStatus: 'uncalibrated', confidence: 0, reason });
  if (!validateCamera(camera)) return failure('invalid-intrinsics');
  const minimumScore = options.minimumScore ?? 0.25;
  const points = new Map();
  for (const point of keypoints || []) {
    if (!Number.isInteger(point.id) || ![point.u, point.v, point.score].every(Number.isFinite) || point.score < minimumScore || point.observable === false) continue;
    const ray = undistortPixel(point.u, point.v, camera);
    if (!ray) continue;
    const hasDepth = Number.isFinite(point.relativeZm);
    if (!hasDepth && !options.allowPlanarPrior) continue;
    points.set(point.id, { ...point, ray, hasDepth, dz: hasDepth ? point.relativeZm : 0, score: clamp(point.score, 0, 1) });
  }
  if (points.size < 4) return failure('insufficient-relative-depth');
  const hips = [points.get(11), points.get(12)].filter(Boolean);
  const rootOffset = hips.length === 2 ? (hips[0].dz + hips[1].dz) / 2 : 0;
  for (const point of points.values()) point.dz -= rootOffset;
  const bones = [];
  for (const [edge, length] of Object.entries(boneLengths || {})) {
    const [a, b] = edge.split('-').map(Number);
    if (!Number.isInteger(a) || !Number.isInteger(b) || !Number.isFinite(length) || length <= 0) continue;
    const pa = points.get(a), pb = points.get(b);
    if (!pa || !pb) continue;
    const slope = [pa.ray[0] - pb.ray[0], pa.ray[1] - pb.ray[1], 0];
    const offset = [pa.ray[0] * pa.dz - pb.ray[0] * pb.dz, pa.ray[1] * pa.dz - pb.ray[1] * pb.dz, pa.dz - pb.dz];
    if (norm(slope) < 1e-4) continue;
    bones.push({ a, b, length, slope, offset, weight: Math.sqrt(pa.score * pb.score), hasDepth: pa.hasDepth && pb.hasDepth });
  }
  if (bones.length < (options.minimumBones ?? 3)) return failure('insufficient-bone-constraints');
  const lower = Math.max(options.minimumDepthM ?? 0.35, 0.06 - Math.min(...[...points.values()].map((p) => p.dz)));
  const upper = options.maximumDepthM ?? 15;
  if (!(lower < upper)) return failure('invalid-depth-range');
  const delta = options.huberDeltaM ?? 0.04;
  const totalWeight = bones.reduce((sum, bone) => sum + bone.weight, 0);
  const residual = (bone, depth) => norm(add(scale(bone.slope, depth), bone.offset)) - bone.length;
  const objective = (depth) => bones.reduce((sum, bone) => sum + bone.weight * huber(residual(bone, depth), delta), 0) / totalWeight;
  const samples = 96;
  const grid = Array.from({ length: samples + 1 }, (_, i) => lower * (upper / lower) ** (i / samples));
  let bestIndex = 0;
  for (let i = 1; i < grid.length; i++) if (objective(grid[i]) < objective(grid[bestIndex])) bestIndex = i;
  let left = grid[Math.max(0, bestIndex - 1)], right = grid[Math.min(samples, bestIndex + 1)];
  const ratio = (Math.sqrt(5) - 1) / 2;
  let a = right - ratio * (right - left), b = left + ratio * (right - left);
  for (let iteration = 0; iteration < 36; iteration++) {
    if (objective(a) < objective(b)) { right = b; b = a; a = right - ratio * (right - left); }
    else { left = a; a = b; b = left + ratio * (right - left); }
  }
  const depthM = (left + right) / 2;
  if (depthM - lower < 1e-4 || upper - depthM < 1e-4) return failure('depth-at-search-boundary');
  const fitErrorM = bones.reduce((sum, bone) => sum + Math.abs(residual(bone, depthM)) * bone.weight, 0) / totalWeight;
  if (fitErrorM > (options.maximumBoneErrorM ?? 0.15)) return failure('inconsistent-bone-lengths');
  const hasRelativeDepth = bones.every((bone) => bone.hasDepth);
  const calibratedIntrinsics = camera.intrinsicsCalibrated === true || camera.calibrated === true;
  const anthropometric = options.boneLengthSource !== 'measured';
  const confidence = clamp(Math.exp(-fitErrorM / 0.08) * Math.min(1, bones.length / 8) * (hasRelativeDepth ? 0.8 : 0.3) * (calibratedIntrinsics ? 1 : 0.45) * (anthropometric ? 0.7 : 1), 0, 0.85);
  const transform = camera.worldFromCamera;
  const joints = [...points.values()].map((point) => {
    const z = depthM + point.dz;
    const cameraXYZ = [point.ray[0] * z, point.ray[1] * z, z];
    return { id: point.id, xyz: transform ? add(mv(transform.rotation, cameraXYZ), transform.translation) : cameraXYZ, cameraXYZ, confidence: point.score * confidence, source: point.hasDepth ? 'monocular' : 'planarPrior', valid: true };
  });
  return {
    valid: true, depthM, joints, fitErrorM, confidence, boneCount: bones.length,
    distanceStatus: !calibratedIntrinsics ? 'estimatedLowConfidence' : hasRelativeDepth ? 'estimatedMonocular' : 'estimatedPlanarPrior',
    coordinateFrame: transform ? 'world' : 'camera', boneLengthSource: anthropometric ? 'anthropometricPrior' : 'measured',
  };
}

export function triangulateFrame(frames, cameras, options = {}) {
  const grouped = new Map();
  for (const frame of frames || []) for (const point of frame.keypoints || []) {
    if (!grouped.has(point.id)) grouped.set(point.id, []);
    grouped.get(point.id).push({ ...point, sourceId: frame.sourceId, timeMs: frame.timeMs });
  }
  return [...grouped.entries()].map(([id, observations]) => ({ id, ...triangulateJoint(observations, cameras, options) }));
}
