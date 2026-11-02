/** Geometry contract: observations use camera x-right/y-down/z-forward metres.
 * K, R, t use OpenCV's Pcamera = R * Pworld + t (row-major matrices).
 * Public fused positions use Three.js x-right/y-up/z-backward metres.
 * Learned relative depth and estimated camera translation remain separate.
 */
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const mul = (a, s) => a.map(v => v * s);
const norm = a => Math.hypot(...a);
const unit = a => mul(a, 1 / Math.max(1e-9, norm(a)));
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const cvToView = p => [p[0], -p[1], -p[2]];
const finite3 = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
const identity = () => [1, 0, 0, 0, 1, 0, 0, 0, 1];
const mv = (m, p) => [dot(m.slice(0, 3), p), dot(m.slice(3, 6), p), dot(m.slice(6, 9), p)];
const transpose = m => [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
const median = values => {
  const a = values.filter(Number.isFinite).sort((x, y) => x - y);
  return a.length ? a[Math.floor(a.length / 2)] : NaN;
};

export function canonicalName(name) {
  return String(name || '').replace(/([a-z])([A-Z])/g, '$1_$2').toLowerCase()
    .replace(/[ .-]/g, '_').replace(/_foot_index$/, '_big_toe');
}

const isBodyAnchor = name => /^(left|right)_(shoulder|elbow|wrist|hip|knee|ankle|heel|big_toe|small_toe)$/.test(name);

export const BODY_EDGES = [
  ['left_shoulder', 'right_shoulder'], ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'], ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'], ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'], ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
  ['left_ankle', 'left_heel'], ['left_heel', 'left_big_toe'],
  ['left_ankle', 'left_big_toe'], ['right_ankle', 'right_heel'],
  ['right_heel', 'right_big_toe'], ['right_ankle', 'right_big_toe'],
  ['left_ear', 'left_eye'], ['left_eye', 'nose'], ['nose', 'right_eye'], ['right_eye', 'right_ear'],
];
const HAND_FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky'];
export const HAND_EDGES = ['left', 'right'].flatMap(side => HAND_FINGERS.flatMap(finger => {
  const joints = finger === 'thumb' ? ['cmc', 'mcp', 'ip', 'tip'] : ['mcp', 'pip', 'dip', 'tip'];
  const names = [side + '_hand_wrist', ...joints.map(joint => side + '_' + finger + '_' + joint)];
  return names.slice(1).map((name, i) => [names[i], name]);
}));

const KINEMATIC_EDGES = [...BODY_EDGES.filter(([a, b]) => isBodyAnchor(a) && isBodyAnchor(b)), ...HAND_EDGES];
const boneKey = (a, b) => [a, b].sort().join(':');
const oppositeSide = name => name.startsWith('left_') ? 'right_' + name.slice(5) : name.startsWith('right_') ? 'left_' + name.slice(6) : name;
const heightPriors = new Map();
// Loose, optional scale priors, never a claim of measured bone length.
// They are used only when the user explicitly calibrated their own height.
for (const side of ['left', 'right']) {
  for (const [a, b, ratio] of [['shoulder', 'elbow', 0.186], ['elbow', 'wrist', 0.146], ['shoulder', 'hip', 0.29],
    ['hip', 'knee', 0.245], ['knee', 'ankle', 0.246], ['ankle', 'heel', 0.045], ['heel', 'big_toe', 0.145], ['ankle', 'big_toe', 0.125]]) {
    heightPriors.set(boneKey(side + '_' + a, side + '_' + b), ratio);
  }
}

function boneReference(a, b, history, frameId, options) {
  const explicit = options.boneLengthsM?.[a + ':' + b] ?? options.boneLengthsM?.[b + ':' + a];
  const explicitLength = typeof explicit === 'number' ? explicit : explicit?.lengthM;
  if (Number.isFinite(explicitLength) && explicitLength > 0.005 && explicitLength < 2) {
    return { lengthM: explicitLength, sigmaM: Math.max(0.002, explicit?.uncertaintyM || explicitLength * 0.03), source: 'userBoneLength' };
  }
  for (const [key, mirrored] of [[boneKey(a, b), false], [boneKey(oppositeSide(a), oppositeSide(b)), true]]) {
    const record = history.get(frameId + '/' + key);
    if (!record || record.values.length < 3) continue;
    const lengthM = median(record.values);
    const spread = median(record.values.map(v => Math.abs(v - lengthM))) * 1.4826;
    return { lengthM, sigmaM: Math.max(0.003, spread, lengthM * (mirrored ? 0.07 : 0.035)), source: mirrored ? 'oppositePoseBone' : 'poseBoneHistory' };
  }
  const ratio = heightPriors.get(boneKey(a, b));
  if (options.heightCalibrated && ratio) return { lengthM: options.heightM * ratio, sigmaM: options.heightM * ratio * 0.2, source: 'estimatedAnthropometry' };
  return null;
}

function learnBoneReferences(output, history, frameId, time, confidence) {
  for (const [a, b] of KINEMATIC_EDGES) {
    const pa = output.get(a), pb = output.get(b);
    if (![pa, pb].every(p => p?.observable && finite3(p.position) && (p.source === 'measuredMultiView' || p.score >= confidence) && p.source !== 'held' && p.source !== 'boneConstrained')) continue;
    const length = norm(sub(pa.position, pb.position));
    if (length < 0.005 || length > 2) continue;
    const key = frameId + '/' + boneKey(a, b);
    const record = history.get(key) || { values: [], lastTime: -Infinity };
    if (time <= record.lastTime) continue;
    record.values.push(length); if (record.values.length > 45) record.values.shift();
    record.lastTime = time; history.set(key, record);
  }
}

function raySphereCandidates(ray, anchor, reference) {
  const projection = dot(ray, anchor), length = reference.lengthM;
  const discriminant = projection * projection - dot(anchor, anchor) + length * length;
  if (discriminant < -1e-8) return [];
  const offset = Math.sqrt(Math.max(0, discriminant));
  const uncertainty = Math.sqrt(Math.max(0, discriminant + 2 * length * reference.sigmaM + reference.sigmaM ** 2)) - offset;
  const depths = [projection - offset, projection + offset].filter(v => v > 0.1 && v < 30);
  return depths.map(distance => ({ cameraPosition: mul(ray, distance), uncertaintyM: uncertainty }));
}

function liftConstrainedPoints(source, output, root, options, history, previous, fit, failures) {
  const camera = source.camera, time = source.observation.mediaTimeMs;
  const rootCamera = add(mv(camera.R, cvToView(root.worldXYZm)), camera.t);
  const sameFrame = camera.measured ? root.worldFrameId.startsWith('calibrated-world:') :
    camera.alignmentId ? root.worldFrameId.startsWith(camera.alignmentId + ':') : root.sourceId === source.observation.sourceId;
  if (!sameFrame) return;
  const inCamera = position => add(mv(camera.R, cvToView(position)), camera.t);
  const toWorld = position => cvToView(mv(camera.inverseR, sub(position, camera.t)));
  const map = new Map(source.points.map(point => [point.name, point]));
  // A hand wrist and body wrist denote the same anatomical point. This is
  // an alias, not a new bone or an invented depth estimate.
  for (const side of ['left', 'right']) {
    const name = side + '_hand_wrist', observed = map.get(name), body = output.get(side + '_wrist');
    if (!observed || observed.source === 'held' || output.has(name) || !body?.observable || body.source === 'held' || !finite3(body.position)) continue;
    const cameraPoint = inCamera(body.position), ray = rayUV(observed, camera);
    const error = Math.hypot((cameraPoint[0] / cameraPoint[2] - ray[0]) * camera.fx, (cameraPoint[1] / cameraPoint[2] - ray[1]) * camera.fy);
    if (error <= Math.max(5, Math.min(source.observation.imageSize.width, source.observation.imageSize.height) * 0.025)) {
      output.set(name, { name, position: body.position.slice(), score: Math.min(observed.score, body.score), observable: true,
        source: 'boneConstrained', depthEvidence: 'sameAnatomicalWrist', depthInferred: true, depthAmbiguous: !!body.depthAmbiguous,
        depthUncertaintyM: body.depthUncertaintyM || 0.015, depthEvidenceTimeMs: body.depthEvidenceTimeMs ?? time,
        sourceId: source.observation.sourceId, observationTimeMs: time, score2D: observed.score2D ?? observed.score });
    }
  }
  for (let pass = 0; pass < 4; pass++) {
    let added = false;
    for (const point of source.points) {
      if (output.has(point.name) || finite3(point.relativeXYZm) || point.source === 'held') continue;
      const neighbors = KINEMATIC_EDGES.flatMap(([a, b]) => a === point.name ? [b] : b === point.name ? [a] : []);
      if (!neighbors.length) continue;
      const anchors = neighbors.map(name => ({ name, point: output.get(name), reference: boneReference(name, point.name, history, root.worldFrameId, options) }))
        .filter(a => a.reference && a.point?.observable && finite3(a.point.position) && a.point.source !== 'held' && !a.point.depthAmbiguous)
        .map(a => ({ ...a, cameraPosition: inCamera(a.point.position) }));
      if (!anchors.length) { failures.set(point.name, { reason: 'bone-length-or-3d-anchor-missing', point }); continue; }
      anchors.sort((a, b) => a.reference.sigmaM - b.reference.sigmaM);
      const ray = unit(rayUV(point, camera));
      if (!finite3(ray)) continue;
      let candidates = raySphereCandidates(ray, anchors[0].cameraPosition, anchors[0].reference)
        .map(candidate => ({ ...candidate, uncertaintyM: candidate.uncertaintyM + (anchors[0].point.depthUncertaintyM || 0) }));
      if (!candidates.length) { failures.set(point.name, { reason: 'ray-does-not-intersect-bone', point }); continue; }
      candidates = candidates.filter(candidate => norm(sub(candidate.cameraPosition, rootCamera)) < 2.5).map(candidate => {
        const residuals = anchors.slice(1).map(anchor => Math.abs(norm(sub(candidate.cameraPosition, anchor.cameraPosition)) - anchor.reference.lengthM));
        const sigma = anchor => Math.max(0.01, anchor.reference.sigmaM + (anchor.point.depthUncertaintyM || 0));
        const cost = residuals.reduce((sum, residual, i) => sum + (residual / sigma(anchors[i + 1])) ** 2, 0);
        const compatible = residuals.every((residual, i) => residual <= Math.max(0.035, 3 * sigma(anchors[i + 1])));
        return { ...candidate, cost, compatible };
      }).filter(candidate => candidate.compatible).sort((a, b) => a.cost - b.cost);
      if (!candidates.length) { failures.set(point.name, { reason: 'neighbor-bone-conflict', point }); continue; }
      let chosen = null, evidence = null, evidenceTime = Math.min(time, ...anchors.map(a => a.point.depthEvidenceTimeMs ?? time)), ambiguous = false;
      if (candidates.length === 1) { chosen = candidates[0]; evidence = 'uniquePositiveBoneIntersection'; }
      else if (anchors.length >= 2 && candidates[1].cost - candidates[0].cost > 4) { chosen = candidates[0]; evidence = 'multipleBoneConstraints'; }
      const priorAge = Math.abs((point.depthPriorTimeMs ?? time) - time);
      const strongPrior = finite3(point.depthPriorXYZm) && point.depthPriorObservable === true && point.depthPriorAgreement !== false &&
        point.depthPriorScore >= Math.max(options.confidence, 0.55) && priorAge <= options.syncToleranceMs;
      let prior = strongPrior ? add(fit?.translation || rootCamera, mul(point.depthPriorXYZm, fit?.scale || 1)) : null;
      let priorType = strongPrior ? 'currentLearnedDepthPrior' : null;
      let priorRadius = 0.18;
      const old = previous.get(point.name);
      if (!prior && old && old.source !== 'held' && time > old.time && time - old.time <= 200 && time - (old.depthEvidenceTimeMs ?? old.time) <= 600 && !old.depthAmbiguous) {
        prior = inCamera(old.position); priorType = 'shortTemporalDepthPrior'; evidenceTime = old.depthEvidenceTimeMs ?? old.time;
        priorRadius = Math.min(0.35, 0.08 + 5 * (time - old.time) / 1000);
      }
      if (!chosen && prior) {
        const ranked = candidates.map(candidate => ({ ...candidate, priorDistance: norm(sub(candidate.cameraPosition, prior)) })).sort((a, b) => a.priorDistance - b.priorDistance);
        if (ranked[0].priorDistance <= priorRadius && ranked[1].priorDistance - ranked[0].priorDistance > Math.max(0.025, ranked[0].uncertaintyM)) {
          chosen = ranked[0]; evidence = priorType;
          if (strongPrior) evidenceTime = time;
        }
      }
      if (!chosen && candidates.length === 2) {
        const gap = norm(sub(candidates[0].cameraPosition, candidates[1].cameraPosition));
        if (gap < 0.035 && Math.max(...candidates.map(c => c.uncertaintyM)) + gap / 2 < 0.06) {
          chosen = { ...candidates[0], cameraPosition: mul(add(candidates[0].cameraPosition, candidates[1].cameraPosition), 0.5),
            uncertaintyM: Math.max(...candidates.map(c => c.uncertaintyM)) + gap / 2 };
          evidence = 'boundedNearTangent'; ambiguous = true;
        }
      }
      if (!chosen || chosen.uncertaintyM > 0.25) { failures.set(point.name, { reason: 'ambiguous-bone-depth', point }); continue; }
      const worldPosition = toWorld(chosen.cameraPosition);
      if (!finite3(worldPosition)) continue;
      output.set(point.name, { name: point.name, position: worldPosition, score: Math.min(point.score, ...anchors.map(a => a.point.score)) * 0.8,
        observable: true, source: 'boneConstrained', depthInferred: true, depthEvidence: evidence, depthAmbiguous: ambiguous,
        depthUncertaintyM: chosen.uncertaintyM, depthEvidenceTimeMs: evidenceTime,
        boneLengthSource: anchors[0].reference.source, boneLengthM: anchors[0].reference.lengthM,
        observationTimeMs: time, sourceId: source.observation.sourceId, observed2D: { u: point.u, v: point.v }, score2D: point.score2D ?? point.score });
      failures.delete(point.name); added = true;
    }
    if (!added) break;
  }
}

function solve3(matrix, rhs) {
  const a = [0, 1, 2].map(i => [...matrix.slice(i * 3, i * 3 + 3), rhs[i]]);
  for (let c = 0; c < 3; c++) {
    let pivot = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(a[r][c]) > Math.abs(a[pivot][c])) pivot = r;
    if (Math.abs(a[pivot][c]) < 1e-10) return null;
    [a[c], a[pivot]] = [a[pivot], a[c]];
    const s = a[c][c];
    for (let j = c; j < 4; j++) a[c][j] /= s;
    for (let r = 0; r < 3; r++) if (r !== c) {
      const f = a[r][c];
      for (let j = c; j < 4; j++) a[r][j] -= f * a[c][j];
    }
  }
  return a.map(row => row[3]);
}

function normalizeK(value, width, height, hfov) {
  let a = Array.isArray(value) ? value.flat() : null;
  if (value && !a && Number.isFinite(value.fx)) a = [value.fx, 0, value.cx, 0, value.fy, value.cy, 0, 0, 1];
  const calibrated = !!(a && a.length === 9 && a.every(Number.isFinite) && a[0] > 0 && a[4] > 0 &&
    Math.abs(a[3]) < 1e-6 && Math.abs(a[6]) < 1e-6 && Math.abs(a[7]) < 1e-6 && Math.abs(a[8] - 1) < 1e-6);
  if (!calibrated) {
    const fov = Number.isFinite(Number(hfov)) ? clamp(Number(hfov), 20, 150) : 60;
    const f = width / (2 * Math.tan(fov * Math.PI / 360));
    a = [f, 0, width / 2, 0, f, height / 2, 0, 0, 1];
  }
  return { array: a, fx: a[0], fy: a[4], skew: a[1], cx: a[2], cy: a[5], calibrated };
}

function cameraFor(observation, options) {
  const config = (options.cameras || []).find(camera => camera.sourceId === observation.sourceId) || {};
  const width = observation.imageSize?.width || 1;
  const height = observation.imageSize?.height || 1;
  const k = normalizeK(config.K, width, height, config.hfovDeg || options.hfovDeg);
  if (k.calibrated && config.imageSize && Number.isFinite(config.imageSize.width) && config.imageSize.width > 0 && Number.isFinite(config.imageSize.height) && config.imageSize.height > 0) {
    const sx = width / config.imageSize.width, sy = height / config.imageSize.height;
    k.fx *= sx; k.skew *= sx; k.cx *= sx; k.fy *= sy; k.cy *= sy;
  }
  const r = Array.isArray(config.R) ? config.R.flat() : [];
  const validRotation = r.length === 9 && r.every(Number.isFinite) &&
    [0, 3, 6].every(i => Math.abs(norm(r.slice(i, i + 3)) - 1) < 0.03) &&
    Math.abs(dot(r.slice(0, 3), r.slice(3, 6))) < 0.03 &&
    Math.abs(dot(r.slice(0, 3), r.slice(6, 9))) < 0.03 &&
    Math.abs(dot(cross(r.slice(0, 3), r.slice(3, 6)), r.slice(6, 9)) - 1) < 0.05;
  const extrinsics = validRotation && finite3(config.t);
  const R = extrinsics ? r : identity();
  const t = extrinsics ? config.t.slice() : [0, 0, 0];
  const inverseR = transpose(R);
  const D = Array.isArray(config.D) && config.D.every(Number.isFinite) ? config.D.slice(0, 5) : [];
  return { ...k, R, t, inverseR, D, extrinsics, center: mv(inverseR, mul(t, -1)),
    measured: k.calibrated && extrinsics && config.calibrated !== false,
    calibrated: k.calibrated && config.calibrated !== false, sourceId: observation.sourceId,
    alignmentId: typeof config.alignmentId === 'string' ? config.alignmentId : null,
    alignmentSource: config.alignmentSource || null };
}

function rayUV(point, camera) {
  const py = (point.v - camera.cy) / camera.fy;
  const distorted = [(point.u - camera.cx - camera.skew * py) / camera.fx, py];
  let [x, y] = distorted;
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = camera.D;
  for (let i = 0; i < 5 && camera.D.length; i++) {
    const r2 = x * x + y * y;
    const radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    if (Math.abs(radial) < 1e-6) break;
    const nextX = (distorted[0] - 2 * p1 * x * y - p2 * (r2 + 2 * x * x)) / radial;
    const nextY = (distorted[1] - p1 * (r2 + 2 * y * y) - 2 * p2 * x * y) / radial;
    x = nextX; y = nextY;
  }
  return [x, y, 1];
}

function projectionError(positionCV, point, camera) {
  const p = add(mv(camera.R, positionCV), camera.t);
  if (p[2] <= 0.05) return Infinity;
  const ray = rayUV(point, camera);
  const dx = p[0] / p[2] - ray[0], dy = p[1] / p[2] - ray[1];
  return Math.hypot(dx * camera.fx + dy * camera.skew, dy * camera.fy);
}

function observedPoints(observation, threshold) {
  return (observation.keypoints || []).map(point => ({ ...point, name: canonicalName(point.name) }))
    .filter(point => point.observable !== false && Number.isFinite(point.u) && Number.isFinite(point.v) &&
      Number.isFinite(point.score) && point.score >= threshold);
}

function fitTranslation(points, camera, scale = 1) {
  const usable = points.filter(point => isBodyAnchor(point.name) && finite3(point.relativeXYZm));
  if (usable.length < 5) return null;
  const rays = usable.map(p => rayUV(p, camera));
  if (rays.some(p => !finite3(p))) return null;
  const weight3D = point => (Number.isFinite(point.score3D) ? point.score3D : point.score) ** 2;
  const initialWeight = usable.reduce((sum, p) => sum + weight3D(p), 0);
  if (initialWeight < 1e-8) return null;
  const rayMean = rays.reduce((sum, ray, i) => add(sum, mul(ray, weight3D(usable[i]) / initialWeight)), [0, 0, 0]);
  const angularVariance = rays.reduce((sum, ray, i) => sum + weight3D(usable[i]) * (sub(ray, rayMean).slice(0, 2).reduce((s, v) => s + v * v, 0)), 0) / initialWeight;
  // The depth normal equation becomes singular when all visible anchors
  // occupy essentially the same ray, regardless of their landmark count.
  if (angularVariance < 0.0001) return null;
  let translation = null, weights = usable.map(weight3D);
  let errors = [];
  for (let iteration = 0; iteration < 4; iteration++) {
    const normal = new Array(9).fill(0), rhs = [0, 0, 0];
    for (let i = 0; i < usable.length; i++) {
      const [u, v] = rays[i];
      const [x, y, z] = mul(usable[i].relativeXYZm, scale);
      const rows = [[1, 0, -u, u * z - x], [0, 1, -v, v * z - y]];
      for (const row of rows) for (let a = 0; a < 3; a++) {
        rhs[a] += weights[i] * row[a] * row[3];
        for (let b = 0; b < 3; b++) normal[a * 3 + b] += weights[i] * row[a] * row[b];
      }
    }
    const depthInformation = normal[8] - normal[2] ** 2 / normal[0] - normal[5] ** 2 / normal[4];
    if (!Number.isFinite(depthInformation) || depthInformation / normal[0] < 0.0001) return null;
    translation = solve3(normal, rhs);
    if (!finite3(translation) || translation[2] < 0.15 || translation[2] > 30 || Math.abs(translation[0]) > 30 || Math.abs(translation[1]) > 30) return null;
    errors = usable.map(point => {
      const p = add(mul(point.relativeXYZm, scale), translation);
      const ray = rayUV(point, camera);
      const dx = p[0] / p[2] - ray[0], dy = p[1] / p[2] - ray[1];
      return p[2] > 0.05 ? Math.hypot(dx * camera.fx + dy * camera.skew, dy * camera.fy) : Infinity;
    });
    if (errors.some(error => !Number.isFinite(error))) return null;
    const robustScale = Math.max(3, median(errors) * 1.4826);
    weights = usable.map((p, i) => weight3D(p) * Math.min(1, robustScale / Math.max(1e-6, errors[i])));
  }
  const rms = Math.sqrt(errors.reduce((s, e) => s + e * e, 0) / errors.length);
  const spread = Math.hypot(
    Math.max(...usable.map(p => p.u)) - Math.min(...usable.map(p => p.u)),
    Math.max(...usable.map(p => p.v)) - Math.min(...usable.map(p => p.v)));
  if (!Number.isFinite(rms) || rms > Math.max(18, spread * 0.18)) return null;
  return { translation, reprojectionPx: rms, count: usable.length, scale, angularSpread: Math.sqrt(angularVariance) };
}

function modelScale(points, options) {
  // An optional user height calibrates a learned skeleton; avatar height never enters this calculation.
  if (!options.heightCalibrated) return 1;
  const map = new Map(points.map(p => [p.name, p]));
  const lengths = [];
  for (const side of ['left', 'right']) {
    const chain = ['shoulder', 'hip', 'knee', 'ankle'].map(joint => map.get(side + '_' + joint));
    if (chain.every(p => finite3(p?.relativeXYZm) && p.score >= options.confidence)) {
      const total = chain.slice(1).reduce((sum, p, i) => sum + norm(sub(p.relativeXYZm, chain[i].relativeXYZm)), 0);
      if (total > 0.5) lengths.push(total / 0.82);
    }
  }
  const learnedHeight = median(lengths);
  return Number.isFinite(learnedHeight) ? clamp(options.heightM / learnedHeight, 0.55, 1.8) : null;
}

function triangulate(a, ca, b, cb, maxError) {
  const da = unit(mv(ca.inverseR, rayUV(a, ca)));
  const db = unit(mv(cb.inverseR, rayUV(b, cb)));
  const c = dot(da, db), denominator = 1 - c * c;
  const angle = Math.acos(clamp(Math.abs(c), -1, 1)) * 180 / Math.PI;
  if (denominator < 0.0012 || angle < 2 || norm(sub(ca.center, cb.center)) < 0.05) return null;
  const w = sub(ca.center, cb.center), d = dot(da, w), e = dot(db, w);
  const ta = (c * e - d) / denominator, tb = (e - c * d) / denominator;
  if (ta <= 0 || tb <= 0) return null;
  const pa = add(ca.center, mul(da, ta)), pb = add(cb.center, mul(db, tb));
  const positionCV = mul(add(pa, pb), 0.5);
  const error = Math.max(projectionError(positionCV, a, ca), projectionError(positionCV, b, cb));
  if (!finite3(positionCV) || !Number.isFinite(error) || error > maxError || norm(sub(pa, pb)) > 0.18 || norm(positionCV) > 50) return null;
  return { positionCV, reprojectionPx: error, rayAngleDeg: angle,
    score: Math.min(a.score2D ?? a.score, b.score2D ?? b.score) * Math.exp(-error / maxError) };
}

function principalQuaternion(matrix) {
  // Jacobi eigendecomposition avoids power iteration's zero projection
  // failure for some exact 180-degree rotations.
  const a = matrix.map(row => row.slice());
  const eigenvectors = Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) => +(i === j)));
  for (let iteration = 0; iteration < 80; iteration++) {
    let p = 0, q = 1;
    for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) if (Math.abs(a[i][j]) > Math.abs(a[p][q])) { p = i; q = j; }
    if (Math.abs(a[p][q]) < 1e-12) break;
    const tau = (a[q][q] - a[p][p]) / (2 * a[p][q]);
    const tangent = (tau >= 0 ? 1 : -1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
    const cosine = 1 / Math.sqrt(1 + tangent * tangent), sine = tangent * cosine;
    const off = a[p][q];
    a[p][p] -= tangent * off; a[q][q] += tangent * off; a[p][q] = 0; a[q][p] = 0;
    for (let r = 0; r < 4; r++) {
      if (r !== p && r !== q) {
        const rp = a[r][p], rq = a[r][q];
        a[r][p] = a[p][r] = cosine * rp - sine * rq;
        a[r][q] = a[q][r] = sine * rp + cosine * rq;
      }
      const vp = eigenvectors[r][p], vq = eigenvectors[r][q];
      eigenvectors[r][p] = cosine * vp - sine * vq;
      eigenvectors[r][q] = sine * vp + cosine * vq;
    }
  }
  const order = [0, 1, 2, 3].sort((i, j) => a[j][j] - a[i][i]);
  if (a[order[0]][order[0]] - a[order[1]][order[1]] < Math.max(1e-7, Math.abs(a[order[0]][order[0]]) * 1e-4)) return null;
  const q = eigenvectors.map(row => row[order[0]]);
  return q.map(v => v / Math.hypot(...q));
}

function hasAlignmentSpread(points, weights) {
  let pair = [0, 1], longest = 0;
  for (let i = 0; i < points.length; i++) for (let j = i + 1; j < points.length; j++) {
    const distance = norm(sub(points[i], points[j]));
    if (distance > longest) { longest = distance; pair = [i, j]; }
  }
  if (longest < 0.15) return false;
  const axis = unit(sub(points[pair[1]], points[pair[0]]));
  const total = weights.reduce((sum, v) => sum + v, 0);
  const transverse = points.reduce((sum, point, i) => {
    const delta = sub(point, points[pair[0]]);
    return sum + weights[i] * norm(sub(delta, mul(axis, dot(delta, axis)))) ** 2;
  }, 0) / total;
  return Math.sqrt(transverse) > Math.max(0.025, longest * 0.025);
}

function rigidAlignment(sourcePoints, targetPoints, weights) {
  const total = weights.reduce((sum, v) => sum + v, 0);
  if (sourcePoints.length < 6 || total < 1e-6) return null;
  if (!hasAlignmentSpread(sourcePoints, weights) || !hasAlignmentSpread(targetPoints, weights)) return null;
  const center = points => points.reduce((s, p, i) => add(s, mul(p, weights[i] / total)), [0, 0, 0]);
  const sourceCenter = center(sourcePoints), targetCenter = center(targetPoints);
  const covariance = new Array(9).fill(0);
  for (let i = 0; i < sourcePoints.length; i++) {
    const a = sub(sourcePoints[i], sourceCenter), b = sub(targetPoints[i], targetCenter);
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) covariance[r * 3 + c] += weights[i] * a[r] * b[c];
  }
  const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = covariance;
  const n = [
    [xx + yy + zz, yz - zy, zx - xz, xy - yx],
    [yz - zy, xx - yy - zz, xy + yx, xz + zx],
    [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
    [xy - yx, xz + zx, yz + zy, -xx - yy + zz],
  ];
  const q = principalQuaternion(n);
  if (!q) return null;
  const [w, x, y, z] = q;
  const R = [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)];
  const t = sub(targetCenter, mv(R, sourceCenter));
  const errors = sourcePoints.map((p, i) => norm(sub(add(mv(R, p), t), targetPoints[i])));
  return { R, t, errors, rmsM: Math.sqrt(errors.reduce((sum, e, i) => sum + weights[i] * e * e, 0) / total) };
}

function quaternionFromBasis(xAxis, yAxis, zAxis) {
  const m00 = xAxis[0], m01 = yAxis[0], m02 = zAxis[0];
  const m10 = xAxis[1], m11 = yAxis[1], m12 = zAxis[1];
  const m20 = xAxis[2], m21 = yAxis[2], m22 = zAxis[2];
  const trace = m00 + m11 + m22;
  let q;
  if (trace > 0) { const s = Math.sqrt(trace + 1) * 2; q = [(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, s / 4]; }
  else if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1 + m00 - m11 - m22) * 2; q = [s / 4, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s]; }
  else if (m11 > m22) { const s = Math.sqrt(1 + m11 - m00 - m22) * 2; q = [(m01 + m10) / s, s / 4, (m12 + m21) / s, (m02 - m20) / s]; }
  else { const s = Math.sqrt(1 + m22 - m00 - m11) * 2; q = [(m02 + m20) / s, (m12 + m21) / s, s / 4, (m10 - m01) / s]; }
  return q.map(v => v / Math.hypot(...q));
}

function torsoRotation(map) {
  const lh = map.get('left_hip'), rh = map.get('right_hip');
  const ls = map.get('left_shoulder'), rs = map.get('right_shoulder');
  if (![lh, rh, ls, rs].every(p => p?.observable)) return null;
  // VRM normalized +X points towards the person's anatomical left in its canonical facing pose.
  const x = unit(sub(lh.position, rh.position));
  let y = sub(mul(add(ls.position, rs.position), 0.5), mul(add(lh.position, rh.position), 0.5));
  y = unit(sub(y, mul(x, dot(x, y))));
  if (norm(cross(x, y)) < 0.5) return null;
  return quaternionFromBasis(x, y, unit(cross(x, y)));
}

function buildTrackers(keypoints, root, options) {
  const map = new Map(keypoints.map(p => [p.name, p]));
  const rotation = torsoRotation(map);
  const allowed = root.valid && (root.metricCalibrated || options.allowEstimatedTrackers);
  const waistCertain = ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder'].every(name => {
    const p = map.get(name); return p && !p.depthAmbiguous && (p.depthUncertaintyM || 0) <= 0.1;
  });
  const trackers = [{ role: 'waist', positionXYZm: root.worldXYZm, rotationXYZW: rotation || [0, 0, 0, 1],
    poseValid: !!(root.valid && rotation && waistCertain), valid: !!(allowed && rotation && waistCertain), source: root.distanceStatus }];
  for (const side of ['left', 'right']) {
    const ankle = map.get(side + '_ankle'), heel = map.get(side + '_heel');
    const toe = map.get(side + '_big_toe'), knee = map.get(side + '_knee');
    let q = null, usedKnee = false;
    if ([ankle, heel, toe].every(p => p?.observable)) {
      const z = unit(sub(toe.position, heel.position));
      let y = sub(ankle.position, heel.position);
      if (norm(sub(y, mul(z, dot(y, z)))) < 0.015 && knee?.observable) { y = sub(knee.position, ankle.position); usedKnee = true; }
      y = unit(sub(y, mul(z, dot(y, z))));
      const x = unit(cross(y, z));
      if (norm(x) > 0.9 && norm(y) > 0.9) q = quaternionFromBasis(x, y, z);
    }
    const required = usedKnee ? [ankle, heel, toe, knee] : [ankle, heel, toe];
    const certain = required.every(p => p && !p.depthAmbiguous && (p.depthUncertaintyM || 0) <= 0.1);
    const source = required.some(p => p?.source === 'boneConstrained') ? 'boneConstrained' : ankle?.source || 'missing';
    trackers.push({ role: side + '-foot', positionXYZm: ankle?.position || null,
      rotationXYZW: q || [0, 0, 0, 1], poseValid: !!(root.valid && ankle?.observable && q && certain), valid: !!(allowed && ankle?.observable && q && certain),
      source });
  }
  return trackers;
}

export class PoseFusion {
  constructor(options = {}) { this.options = {}; this.configure(options); this.reset(); }
  configure(options = {}) {
    const previous = this.options;
    this.options = { heightM: 1.7, heightCalibrated: false, hfovDeg: 60, confidence: 0.45,
      syncToleranceMs: 45, maxReprojectionPx: 18, cameras: [], smoothing: 0.035,
      allowEstimatedTrackers: false, lift2D: true, boneLengthsM: {}, ...this.options, ...options };
    for (const [key, fallback, min, max] of [['heightM', 1.7, 0.5, 2.5], ['hfovDeg', 60, 20, 150],
      ['confidence', 0.45, 0.05, 0.99], ['syncToleranceMs', 45, 0, 500], ['maxReprojectionPx', 18, 1, 100], ['smoothing', 0.035, 0, 0.3]]) {
      const value = Number(this.options[key]);
      this.options[key] = Number.isFinite(value) ? clamp(value, min, max) : fallback;
    }
    this.options.cameras = Array.isArray(this.options.cameras) ? this.options.cameras.filter(c => c && typeof c.sourceId === 'string') : [];
    this.options.heightCalibrated = this.options.heightCalibrated === true;
    if (previous.heightM !== this.options.heightM || previous.heightCalibrated !== this.options.heightCalibrated) this.scaleBySource?.clear();
    const camerasChanged = options.cameras !== undefined && JSON.stringify(previous.cameras) !== JSON.stringify(this.options.cameras);
    if (camerasChanged || previous.hfovDeg !== this.options.hfovDeg || previous.heightM !== this.options.heightM || previous.heightCalibrated !== this.options.heightCalibrated) {
      this.previous?.clear(); this.depthHistory?.clear(); this.boneHistory?.clear(); this.coordinateEpoch = (this.coordinateEpoch || 0) + 1;
    }
  }
  reset() { this.previous = new Map(); this.depthHistory = new Map(); this.scaleBySource = new Map(); this.boneHistory = new Map(); this.sequence = 0; this.lastWorldFrame = null; this.coordinateEpoch = (this.coordinateEpoch || 0) + 1; }
  getCalibration() { return { ...this.options, coordinateConvention: 'K/R/t: OpenCV world-to-camera; output: x-right,y-up,z-backward', cameraMotionCompensated: false }; }
  estimateCameraAlignment(observations = []) {
    const frames = observations.filter(o => o?.detected && Number.isFinite(o.mediaTimeMs));
    if (frames.length !== 2 || frames[0].sourceId === frames[1].sourceId) throw new Error('自動整列には別々の同時刻の検出済み動画が2つ必要です。');
    if (Math.abs(frames[0].mediaTimeMs - frames[1].mediaTimeMs) > this.options.syncToleranceMs) throw new Error('動画の同期差が許容幅を超えています。先に時刻オフセットを合わせてください。');
    const cameras = frames.map(frame => cameraFor(frame, this.options));
    const points = frames.map(frame => observedPoints(frame, this.options.confidence)
      .filter(p => finite3(p.relativeXYZm) && isBodyAnchor(p.name)));
    const fits = points.map((p, i) => fitTranslation(p, cameras[i], this.scaleBySource.get(frames[i].sourceId) ?? modelScale(p, this.options) ?? 1));
    if (fits.some(fit => !fit)) throw new Error('両方の映像で十分な相対3D関節を認識する必要があります。');
    const target = new Map(points[0].map(p => [p.name, p]));
    const common = points[1].filter(p => target.has(p.name));
    if (common.length < 8) throw new Error('共通して見える関節が8点未満のため自動整列できません。');
    const a = common.map(p => add(mul(p.relativeXYZm, fits[1].scale), fits[1].translation));
    const b = common.map(p => add(mul(target.get(p.name).relativeXYZm, fits[0].scale), fits[0].translation));
    const weights = common.map(p => Math.min(p.score, target.get(p.name).score) ** 2);
    let fit = rigidAlignment(a, b, weights);
    if (!fit) throw new Error('人体の3D関節を整列できません。');
    const cutoff = Math.max(0.08, median(fit.errors) * 2.5);
    const retained = fit.errors.map((error, i) => error <= cutoff ? i : -1).filter(i => i >= 0);
    if (retained.length < 8) throw new Error('外れ値除去後の共通関節が8点未満のため整列を採用しません。');
    fit = rigidAlignment(retained.map(i => a[i]), retained.map(i => b[i]), retained.map(i => weights[i]));
    if (!fit || fit.rmsM > 0.18) throw new Error('両映像の3D認識が整合しません。残差が18cmを超えるため整列を採用しません。');
    const alignmentId = 'estimated-' + Date.now().toString(36);
    const r2 = transpose(fit.R), t2 = mv(r2, mul(fit.t, -1));
    const configurations = cameras.map((camera, i) => ({ sourceId: frames[i].sourceId,
      K: [camera.fx, camera.skew, camera.cx, 0, camera.fy, camera.cy, 0, 0, 1], D: camera.D,
      R: i === 0 ? identity() : r2, t: i === 0 ? [0, 0, 0] : t2,
      imageSize: frames[i].imageSize, calibrated: false, alignmentId, alignmentSource: 'estimatedMultiView' }));
    const sourceIds = new Set(frames.map(frame => frame.sourceId));
    this.configure({ cameras: [...this.options.cameras.filter(c => !sourceIds.has(c.sourceId)), ...configurations] });
    this.previous.clear();
    return { calibrated: false, source: 'estimatedMultiView', cameras: configurations,
      commonJoints: retained.length, residualM: fit.rmsM,
      warning: '学習済み3D骨格からの推定整列です。実測カメラ校正・床校正ではありません。' };
  }
  update(observations = []) {
    const frames = observations.filter(o => o && Number.isFinite(o.mediaTimeMs));
    const time = frames.length ? Math.max(...frames.map(o => o.mediaTimeMs)) : 0;
    const sources = frames.filter(o => o.detected && time - o.mediaTimeMs <= this.options.syncToleranceMs)
      .map(observation => ({ observation, camera: cameraFor(observation, this.options),
        points: observedPoints(observation, this.options.confidence) }));
    const output = new Map(), fitBySource = new Map(), liftFailures = new Map();
    const diagnostics = { cameras: sources.length, inputFrames: frames.length,
      synchronizationRejected: frames.filter(o => time - o.mediaTimeMs > this.options.syncToleranceMs).length,
      undetectedSources: frames.filter(o => !o.detected).length,
      triangulated: 0, monocular: 0, lifted: 0, held: 0, missing: 0, ambiguous2D: 0, unresolved2D: 0, reprojectionPx: null, cameraMotionCompensated: false,
      warnings: [], coordinateConvention: 'right/up/backward', scaleSource: this.options.heightCalibrated ? 'user-height-and-learned-skeleton' : 'learned-skeleton' };
    let monoRoot = null, chosenCamera = null;
    // Preserve input camera priority. Landmark-count sorting would alternate
    // between unrelated uncalibrated camera frames as hands enter the image.
    for (const source of sources) {
      let scale = this.scaleBySource.get(source.observation.sourceId);
      if (scale === undefined) {
        const calibratedScale = modelScale(source.points, this.options);
        scale = calibratedScale ?? 1;
        if (this.options.heightCalibrated && calibratedScale !== null) this.scaleBySource.set(source.observation.sourceId, scale);
      }
      // Never merge independent, uncalibrated camera coordinate frames.
      const aligned = !!(monoRoot && chosenCamera.alignmentId && chosenCamera.alignmentId === source.camera.alignmentId);
      if (monoRoot && !(chosenCamera.measured && source.camera.measured) && !aligned) continue;
      const fit = fitTranslation(source.points, source.camera, scale);
      if (!fit) continue;
      fitBySource.set(source.observation.sourceId, fit);
      const worldRootCV = mv(source.camera.inverseR, sub(fit.translation, source.camera.t));
      if (!monoRoot) {
        monoRoot = { cameraXYZm: fit.translation, worldXYZm: cvToView(worldRootCV), valid: true,
          metricCalibrated: source.camera.measured && this.options.heightCalibrated && this.scaleBySource.has(source.observation.sourceId),
          distanceStatus: source.camera.calibrated ? 'estimatedMonocular' : 'estimatedUncalibrated',
          sourceId: source.observation.sourceId, relativeDepthSource: source.observation.modelId,
          worldFrameId: (source.camera.measured ? 'calibrated-world' : source.camera.alignmentId || 'camera:' + source.observation.sourceId) + ':' + this.coordinateEpoch };
        chosenCamera = source.camera;
        diagnostics.reprojectionPx = fit.reprojectionPx;
      } else if (aligned && norm(sub(monoRoot.worldXYZm, cvToView(worldRootCV))) < 0.4) {
        monoRoot.worldXYZm = mul(add(monoRoot.worldXYZm, cvToView(worldRootCV)), 0.5);
        monoRoot.cameraXYZm = add(mv(chosenCamera.R, cvToView(monoRoot.worldXYZm)), chosenCamera.t);
        monoRoot.distanceStatus = 'estimatedMultiView';
        monoRoot.metricCalibrated = false;
      }
      for (const point of source.points) {
        if (!finite3(point.relativeXYZm)) continue;
        let cameraPosition = add(mul(point.relativeXYZm, scale), fit.translation);
        if (cameraPosition[2] <= 0.05) continue;
        const refined2D = point.source !== 'held' && point.modelId2D && point.modelId3D && point.modelId2D !== point.modelId3D;
        if (refined2D) {
          // The runtime has already checked cross-model 2D association. Use
          // that accurate image ray while retaining only the valid learned Z.
          const ray = rayUV(point, source.camera);
          cameraPosition = [ray[0] * cameraPosition[2], ray[1] * cameraPosition[2], cameraPosition[2]];
        }
        const position = cvToView(mv(source.camera.inverseR, sub(cameraPosition, source.camera.t)));
        const existing = output.get(point.name);
        if (existing) {
          if (aligned && point.source !== 'held' && existing.source !== 'held' && norm(sub(position, existing.position)) < 0.25) {
            const fraction = point.score ** 2 / (point.score ** 2 + existing.score ** 2);
            existing.position = add(existing.position, mul(sub(position, existing.position), fraction));
            existing.score = Math.max(existing.score, point.score);
            existing.source = 'estimatedMultiView';
          }
          continue;
        }
        output.set(point.name, { name: point.name, position, score: point.score, observable: true,
          score2D: point.score2D ?? point.score, score3D: point.score3D ?? point.score,
          source: point.source === 'held' ? 'held' : refined2D ? 'rayAlignedMonocular' : 'estimatedMonocular', relativeXYZm: point.relativeXYZm.slice(), relativeZm: point.relativeXYZm[2],
          ageMs: point.ageMs || 0, observationTimeMs: point.observationTimeMs ?? source.observation.mediaTimeMs,
          sourceId: source.observation.sourceId });
      }
    }
    for (let i = 0; i < sources.length; i++) for (let j = i + 1; j < sources.length; j++) {
      const a = sources[i], b = sources[j];
      if (!a.camera.measured || !b.camera.measured || Math.abs(a.observation.mediaTimeMs - b.observation.mediaTimeMs) > this.options.syncToleranceMs) continue;
      const other = new Map(b.points.map(p => [p.name, p]));
      for (const point of a.points) {
        if (!other.has(point.name) || point.source === 'held' || other.get(point.name).source === 'held') continue;
        const tri = triangulate(point, a.camera, other.get(point.name), b.camera, this.options.maxReprojectionPx);
        if (!tri) continue;
        const old = output.get(point.name);
        if (old?.source === 'measuredMultiView' && old.score >= tri.score) continue;
        output.set(point.name, { name: point.name, position: cvToView(tri.positionCV), score: tri.score,
          observable: true, source: 'measuredMultiView', reprojectionPx: tri.reprojectionPx, rayAngleDeg: tri.rayAngleDeg });
      }
    }
    let root = monoRoot || { cameraXYZm: null, worldXYZm: null, distanceStatus: 'missing', valid: false, metricCalibrated: false };
    const lh = output.get('left_hip'), rh = output.get('right_hip');
    if (lh?.source === 'measuredMultiView' && rh?.source === 'measuredMultiView') {
      const world = mul(add(lh.position, rh.position), 0.5);
      const camera = chosenCamera || sources[0]?.camera;
      root = { cameraXYZm: camera ? add(mv(camera.R, cvToView(world)), camera.t) : null,
        worldXYZm: world, distanceStatus: 'measuredMultiView', valid: true, metricCalibrated: true,
        sourceId: camera?.sourceId, worldFrameId: 'calibrated-world:' + this.coordinateEpoch };
    }
    if (root.valid && this.lastWorldFrame !== root.worldFrameId) {
      this.previous.clear(); this.depthHistory.clear();
      if (this.lastWorldFrame) diagnostics.warnings.push('推定元の座標系が切り替わったため、表示原点を更新します。連続した実移動には数えません。');
      this.lastWorldFrame = root.worldFrameId;
    }
    if (root.valid && this.options.lift2D) {
      learnBoneReferences(output, this.boneHistory, root.worldFrameId, time, Math.max(this.options.confidence, 0.55));
      for (const source of sources) liftConstrainedPoints(source, output, root, this.options, this.boneHistory, this.depthHistory,
        fitBySource.get(source.observation.sourceId), liftFailures);
    }
    if (root.distanceStatus === 'estimatedMultiView') diagnostics.warnings.push('2視点は学習済み3D骨格で推定整列しています。実測のカメラ校正・三角測量ではありません。');
    else if (sources.length > 1 && !sources.every(s => s.camera.measured)) diagnostics.warnings.push('2台の内部・外部校正が未設定です。映像を2つ読み込むだけでは実測3Dにはなりません。');
    if (root.distanceStatus === 'estimatedUncalibrated') diagnostics.warnings.push('画角から求めた未校正の距離推定です。カメラ移動・ズームは補償しません。');
    if (!root.valid) diagnostics.warnings.push('有効な3D姿勢を復元できません。以前の姿勢を今回の認識結果として再利用しません。');
    const keypoints = [];
    for (const point of output.values()) {
      const previous = this.previous.get(point.name);
      const dt = previous ? (time - previous.time) / 1000 : Infinity;
      if (previous && dt > 0 && dt < 0.25 && this.options.smoothing > 0) {
        const speed = norm(sub(point.position, previous.position)) / Math.max(dt, 0.001);
        const tau = this.options.smoothing / (1 + speed * 2);
        const alpha = 1 - Math.exp(-dt / tau);
        point.position = add(previous.position, mul(sub(point.position, previous.position), alpha));
      }
      const historyPoint = { position: point.position.slice(), time, source: point.source,
        depthEvidenceTimeMs: point.depthEvidenceTimeMs ?? time, depthAmbiguous: point.depthAmbiguous || false };
      this.previous.set(point.name, historyPoint);
      if (point.source !== 'held' && !point.depthAmbiguous) this.depthHistory.set(point.name, historyPoint);
      keypoints.push(point);
      if (point.source === 'measuredMultiView') diagnostics.triangulated++;
      else if (point.source === 'held') diagnostics.held++;
      else if (point.source === 'boneConstrained') diagnostics.lifted++;
      else diagnostics.monocular++;
    }
    const expected = new Set(frames.flatMap(frame => (frame.keypoints || []).map(p => canonicalName(p.name))));
    const observations2D = new Map(sources.flatMap(source => source.points.map(point => [point.name, point])));
    for (const name of expected) if (!output.has(name)) {
      const failed = liftFailures.get(name);
      const observation2D = failed?.point || observations2D.get(name);
      keypoints.push({ name, position: null, score: observation2D?.score || 0, observable: false,
        source: observation2D ? 'observed2D' : 'missing', depthStatus: failed?.reason || (observation2D ? root.valid ? 'relative-depth-unavailable' : 'root-depth-unavailable' : 'missing'),
        depthAmbiguous: failed?.reason === 'ambiguous-bone-depth', observed2D: observation2D ? { u: observation2D.u, v: observation2D.v } : null });
      diagnostics.missing++;
      if (observation2D) diagnostics.unresolved2D++;
      if (failed?.reason === 'ambiguous-bone-depth') diagnostics.ambiguous2D++;
      this.previous.delete(name);
    }
    for (const name of this.previous.keys()) if (!output.has(name)) this.previous.delete(name);
    for (const [name, point] of this.depthHistory) if (time - point.time > 600 || time < point.time) this.depthHistory.delete(name);
    if (!sources.length) this.previous.clear();
    if (diagnostics.lifted) diagnostics.warnings.push('骨長制約による3D補完 ' + diagnostics.lifted + ' 点（紫）。深度根拠と不確かさをログへ保存します。');
    if (diagnostics.ambiguous2D) diagnostics.warnings.push('2Dで見える ' + diagnostics.ambiguous2D + ' 点は奥行きの2解を区別できず、3Dを未確定にしています。');
    const trackers = buildTrackers(keypoints, root, this.options);
    return { schemaVersion: 1, type: 'fused-pose', sequence: ++this.sequence, mediaTimeMs: time,
      referenceTimeMs: time, root, keypoints, trackers, diagnostics };
  }
}
