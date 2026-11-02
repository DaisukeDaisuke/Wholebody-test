// Causal, bounded occlusion completion. Predictions never renew observation age.
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const finite3 = (v) => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);
const add = (a, b) => a.map((x, i) => x + b[i]);
const sub = (a, b) => a.map((x, i) => x - b[i]);
const scale = (a, k) => a.map((x) => x * k);
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
const norm = (v) => Math.hypot(...v);
const unit = (v) => norm(v) > 1e-9 ? scale(v, 1 / norm(v)) : null;
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const currentSources = new Set(['triangulated', 'monocular', 'planarPrior', 'model3d']);
const missing = (id) => ({ id, xyz: null, valid: false, confidence: 0, source: 'missing', observed: false, ageMs: null });
const usable = (joint) => joint?.valid !== false && finite3(joint?.xyz) && Number.isFinite(joint.confidence) && joint.confidence > 0;
const parents = { 0: 5, 1: 0, 2: 0, 3: 1, 4: 2, 5: 11, 6: 12, 7: 5, 8: 6, 9: 7, 10: 8, 12: 11, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 15, 19: 15, 20: 16, 21: 16, 22: 16, 91: 9, 112: 10 };
for (let id = 23; id <= 90; id++) parents[id] = 0;
for (const base of [91, 112]) for (const start of [1, 5, 9, 13, 17]) for (let offset = 0; offset < 4; offset++) parents[base + start + offset] = offset ? base + start + offset - 1 : base;
const traversal = [11, 12, 5, 6, 13, 14, 7, 8, 15, 16, 9, 10, 0, 1, 2, 3, 4, ...Array.from({ length: 116 }, (_, i) => i + 17)];

function boundedVelocity(velocity, maximum) {
  const speed = norm(velocity);
  return speed > maximum ? scale(velocity, maximum / speed) : velocity;
}

function rotateBetween(vector, from, to) {
  const a = unit(from), b = unit(to);
  if (!a || !b) return [...vector];
  const cosine = clamp(dot(a, b), -1, 1);
  const axis = cross(a, b), sine = norm(axis);
  if (sine < 1e-8) {
    if (cosine > 0) return [...vector];
    const perpendicular = unit(cross(a, Math.abs(a[0]) < 0.8 ? [1, 0, 0] : [0, 1, 0]));
    return add(scale(vector, -1), scale(perpendicular, 2 * dot(perpendicular, vector)));
  }
  const k = scale(axis, 1 / sine);
  return add(add(scale(vector, cosine), scale(cross(k, vector), sine)), scale(k, dot(k, vector) * (1 - cosine)));
}

function palmBasis(joints, base) {
  const wrist = joints[base], index = joints[base + 5], middle = joints[base + 9], little = joints[base + 17];
  if (![wrist, index, middle, little].every((joint) => usable(joint) && currentSources.has(joint.source))) return null;
  const x = unit(sub(index.xyz, little.xyz));
  if (!x) return null;
  const direction = sub(middle.xyz, wrist.xyz);
  const y = unit(sub(direction, scale(x, dot(direction, x))));
  if (!y) return null;
  const z = unit(cross(x, y));
  return { origin: wrist.xyz, axes: [x, y, z], confidence: Math.min(wrist.confidence, index.confidence, middle.confidence, little.confidence) };
}

function fromLocal(local, basis) {
  return add(basis.origin, basis.axes.reduce((sum, axis, i) => add(sum, scale(axis, local[i])), [0, 0, 0]));
}

export class OcclusionTracker {
  constructor(options = {}) {
    this.options = {
      jointCount: 133, minimumConfidence: 0.025, maximumBodyPredictionMs: 600,
      maximumHandPredictionMs: 250, maximumFootPredictionMs: 450,
      maximumAnchoredPredictionMs: 1200, velocityDecaySeconds: 0.18,
      smoothingMs: 12, confidenceDecayMs: 350, ...options,
    };
    this.reset();
  }

  reset() {
    this.history = new Map();
    this.handState = new Map();
    this.lastTimeMs = null;
    this.lastResult = null;
  }

  update({ timeMs, joints = [] }) {
    if (!Number.isFinite(timeMs)) throw new Error('追跡フレームには有限の timeMs が必要です。');
    // Seeking a recorded video backwards starts a new temporal segment.
    if (this.lastTimeMs != null && timeMs < this.lastTimeMs - 1) this.reset();
    const count = this.options.jointCount;
    const output = Array.from({ length: count }, (_, id) => missing(id));
    const inputs = new Map();
    for (const joint of joints) {
      if (!Number.isInteger(joint.id) || joint.id < 0 || joint.id >= count || !usable(joint) || !currentSources.has(joint.source) || joint.confidence < this.options.minimumConfidence) continue;
      const prior = inputs.get(joint.id);
      if (!prior || joint.confidence > prior.confidence) inputs.set(joint.id, joint);
    }
    for (const [id, joint] of inputs) {
      const previous = this.history.get(id);
      const dt = previous ? (timeMs - previous.timeMs) / 1000 : 0;
      let position = [...joint.xyz], velocity = [0, 0, 0];
      if (previous && dt > 1e-6 && dt < 0.5) {
        const displacement = sub(position, previous.position);
        const speed = norm(displacement) / dt;
        // Filtering weakens as motion grows; the velocity is bounded independently.
        const tau = this.options.smoothingMs / 1000 / (1 + speed * 3);
        const alpha = tau > 0 ? 1 - Math.exp(-dt / tau) : 1;
        position = add(previous.position, scale(displacement, alpha));
        const instantaneous = boundedVelocity(scale(sub(position, previous.position), 1 / dt), id >= 91 ? 12 : id >= 17 && id <= 22 ? 8 : 6);
        velocity = add(scale(previous.velocity, 0.35), scale(instantaneous, 0.65));
      } else if (previous && dt <= 1e-6) {
        velocity = previous.velocity;
      }
      const confidence = clamp(joint.confidence, 0, 1);
      output[id] = { ...joint, id, xyz: position, confidence, valid: true, observed: joint.source === 'triangulated', ageMs: 0, lastObservedTimeMs: timeMs };
      this.history.set(id, { position, velocity, timeMs, confidence, source: joint.source, offset: previous?.offset, relativeVelocity: previous?.relativeVelocity });
    }
    // Store offsets only from simultaneous current inputs, never predicted output.
    for (const [id] of inputs) {
      const parentId = parents[id];
      if (parentId == null || !inputs.has(parentId)) continue;
      const state = this.history.get(id), parent = this.history.get(parentId);
      state.offset = sub(state.position, parent.position);
      state.relativeVelocity = sub(state.velocity, parent.velocity);
    }
    const currentPalms = new Map();
    for (const base of [91, 112]) {
      const basis = palmBasis(output, base);
      const bodyWrist = base === 91 ? 9 : 10, elbow = base === 91 ? 7 : 8;
      const forearm = usable(output[bodyWrist]) && usable(output[elbow]) ? sub(output[bodyWrist].xyz, output[elbow].xyz) : null;
      if (basis) {
        const previous = this.handState.get(base);
        const locals = new Map(previous?.locals || []);
        for (let id = base; id <= base + 20; id++) if (inputs.has(id)) {
          const difference = sub(output[id].xyz, basis.origin);
          locals.set(id, { coordinates: basis.axes.map((axis) => dot(axis, difference)), timeMs });
        }
        this.handState.set(base, { basis, forearm, locals, timeMs });
        currentPalms.set(base, basis);
      } else {
        const saved = this.handState.get(base);
        const wrist = usable(output[base]) ? output[base] : output[bodyWrist];
        if (saved && usable(wrist) && currentSources.has(wrist.source) && timeMs - saved.timeMs <= this.options.maximumAnchoredPredictionMs) {
          const axes = saved.forearm && forearm ? saved.basis.axes.map((axis) => rotateBetween(axis, saved.forearm, forearm)) : saved.basis.axes;
          currentPalms.set(base, { origin: wrist.xyz, axes, confidence: Math.min(wrist.confidence, saved.basis.confidence) * 0.7 });
        }
      }
    }
    const orderedIds = [...new Set([...traversal, ...Array.from({ length: count }, (_, id) => id)])].filter((id) => id < count);
    for (const id of orderedIds) {
      if (usable(output[id])) continue;
      const state = this.history.get(id);
      if (!state) continue;
      const ageMs = Math.max(0, timeMs - state.timeMs), ageSeconds = ageMs / 1000;
      const base = id >= 112 ? 112 : id >= 91 ? 91 : null;
      const maximumPredictionMs = base != null ? this.options.maximumHandPredictionMs : id >= 17 && id <= 22 ? this.options.maximumFootPredictionMs : this.options.maximumBodyPredictionMs;
      let xyz = null, source = 'predicted', anchorConfidence = 1;
      const savedHand = base != null ? this.handState.get(base) : null;
      const local = savedHand?.locals.get(id);
      const currentPalm = base != null ? currentPalms.get(base) : null;
      if (local && currentPalm && ageMs <= this.options.maximumAnchoredPredictionMs && timeMs - local.timeMs <= this.options.maximumAnchoredPredictionMs) {
        xyz = fromLocal(local.coordinates, currentPalm);
        source = 'kinematicPrior';
        anchorConfidence = currentPalm.confidence;
      }
      const parent = output[parents[id]];
      if (!xyz && state.offset && usable(parent) && parent.ageMs <= maximumPredictionMs && ageMs <= this.options.maximumAnchoredPredictionMs) {
        const integratedSeconds = this.options.velocityDecaySeconds * (1 - Math.exp(-ageSeconds / this.options.velocityDecaySeconds));
        let offset = add(state.offset, scale(state.relativeVelocity || [0, 0, 0], integratedSeconds));
        // Preserve the last observed bone length without moving current measurements.
        const length = norm(state.offset), direction = unit(offset);
        if (length > 1e-5 && direction) offset = scale(direction, length);
        xyz = add(parent.xyz, offset);
        source = 'kinematicPrior';
        anchorConfidence = parent.confidence;
      }
      if (!xyz && ageMs <= maximumPredictionMs) {
        const integratedSeconds = this.options.velocityDecaySeconds * (1 - Math.exp(-ageSeconds / this.options.velocityDecaySeconds));
        xyz = add(state.position, scale(state.velocity, integratedSeconds));
      }
      if (!xyz) continue;
      const confidence = state.confidence * Math.exp(-ageMs / this.options.confidenceDecayMs) * (source === 'kinematicPrior' ? Math.sqrt(anchorConfidence) * 0.8 : 0.85);
      if (confidence < 0.01) continue;
      output[id] = { id, xyz, valid: true, confidence, source, observed: false, ageMs, lastObservedTimeMs: state.timeMs, basedOn: state.source };
    }
    const leftHip = output[11], rightHip = output[12];
    const root = usable(leftHip) && usable(rightHip) ? {
      xyz: scale(add(leftHip.xyz, rightHip.xyz), 0.5),
      confidence: Math.min(leftHip.confidence, rightHip.confidence),
      source: leftHip.source === 'triangulated' && rightHip.source === 'triangulated' ? 'triangulated' : leftHip.source === rightHip.source ? leftHip.source : 'mixed',
      valid: true,
    } : { xyz: null, confidence: 0, source: 'missing', valid: false };
    const stats = { triangulated: 0, monocular: 0, planarPrior: 0, model3d: 0, predicted: 0, kinematicPrior: 0, missing: 0 };
    for (const joint of output) stats[joint.source] = (stats[joint.source] || 0) + 1;
    const result = { timeMs, joints: output, root, feet: deriveFootFrames(output), stats };
    this.lastTimeMs = timeMs;
    this.lastResult = result;
    return result;
  }
}

export function deriveFootFrames(joints) {
  const indexed = new Map((joints || []).map((joint) => [joint.id, joint]));
  const frames = {};
  for (const [side, ankleId, kneeId, bigId, smallId, heelId] of [['left', 15, 13, 17, 18, 19], ['right', 16, 14, 20, 21, 22]]) {
    const ankle = indexed.get(ankleId), big = indexed.get(bigId), small = indexed.get(smallId), heel = indexed.get(heelId), knee = indexed.get(kneeId);
    if (![ankle, big, small, heel].every(usable)) {
      frames[side] = { valid: false, source: 'missing', toeArticulationAvailable: false };
      continue;
    }
    const toeCenter = scale(add(big.xyz, small.xyz), 0.5);
    const forward = unit(sub(toeCenter, heel.xyz));
    if (!forward) { frames[side] = { valid: false, source: 'missing', toeArticulationAvailable: false }; continue; }
    const width = sub(small.xyz, big.xyz);
    let lateral = unit(sub(width, scale(forward, dot(width, forward))));
    if (!lateral) { frames[side] = { valid: false, source: 'missing', toeArticulationAvailable: false }; continue; }
    let up = unit(cross(forward, lateral));
    const towardShin = usable(knee) ? sub(knee.xyz, ankle.xyz) : sub(ankle.xyz, heel.xyz);
    if (dot(up, towardShin) < 0) { up = scale(up, -1); lateral = scale(lateral, -1); }
    frames[side] = {
      valid: true, source: 'derivedFootFrame', confidence: Math.min(ankle.confidence, big.confidence, small.confidence, heel.confidence),
      positionXYZm: add(scale(ankle.xyz, 0.5), scale(add(heel.xyz, toeCenter), 0.25)),
      positionMeaning: 'instepProxy', axes: { forward, lateral, up },
      bigToeXYZm: [...big.xyz], smallToeXYZm: [...small.xyz], heelXYZm: [...heel.xyz],
      toeArticulationAvailable: false,
      inputSources: [...new Set([ankle.source, big.source, small.source, heel.source])],
    };
  }
  return frames;
}

export function mergeJointEstimates(primary, fallback) {
  const result = new Map();
  for (const joint of fallback || []) if (usable(joint)) result.set(joint.id, joint);
  for (const joint of primary || []) if (usable(joint)) result.set(joint.id, joint);
  return [...result.values()].sort((a, b) => a.id - b.id);
}
