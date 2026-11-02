/** Causal observation gates shared by the browser and the headless evaluator.
 * Missing/implausible points stay missing. Display hold/decay belongs to retarget,
 * so no extrapolated position can become a fresh 3D observation or OSC tracker. */
import { refreshPoseMetadata } from './geometry.mjs';

const finite = Number.isFinite;
const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
const length = p => Math.hypot(...p);
const subtract = (a, b) => a.map((n, i) => n - b[i]);
const mix = (a, b, t) => a.map((n, i) => n + (b[i] - n) * t);
const median = values => { const v = values.filter(finite).sort((a, b) => a - b); return v.length ? (v[Math.floor((v.length - 1) / 2)] + v[Math.floor(v.length / 2)]) / 2 : null; };
const xy = p => [p?.x ?? p?.u, p?.y ?? p?.v];
const point = p => p && xy(p).every(finite);
const vector = p => Array.isArray(p) && p.length === 3 && p.every(finite);
const distance = (a, b) => length(subtract(a, b));
const midpoint = (a, b) => mix(a, b, .5);
const TORSO = [5, 6, 11, 12];
const BODY_SEGMENTS = [[5, 7, .186], [7, 9, .146], [6, 8, .186], [8, 10, .146], [11, 13, .245], [13, 15, .246], [12, 14, .245], [14, 16, .246]];
const NON_OBSERVATIONS = new Set(['missing', 'predicted', 'held', 'prior', 'inferredPrior', 'inferredKinematic', 'inferredAnatomy']);
const SOURCE_3D = new Set(['measuredMultiView', 'estimatedMultiView', 'inferredMonocular', 'inferredFinger']);

export const STABILITY_DEFAULTS = Object.freeze({
  minScore: .25, enterMargin: .04, borderPixels: 1, smoothingMs: 32,
  maxTorsoSpeed: 3, maxLimbSpeed: 8, reacquireFrames: 2, resetGapMs: 2500,
});

function missingPoint(p, id, reason) { return { ...p, id, rawScore: p?.rawScore ?? p?.score ?? 0, score: 0, observable: false, gateReason: reason }; }
function missingJoint(j, id, reason) { return { ...j, id, position: null, score: 0, source: 'missing', observable2D: false, ageMs: 0, reason }; }
function observationKey(o) { return `${o?.sourceId || ''}:${o?.targetVersion ?? ''}:${o?.inputMode || ''}:${o?.imageSize?.width || 0}x${o?.imageSize?.height || 0}`; }
function poseKey(p) { return `${p?.sourceId || ''}:${p?.targetVersion ?? ''}:${p?.inputMode || ''}:${p?.coordinateFrame || ''}:${p?.units || ''}:${p?.worldCalibrationId || ''}`; }
function torsoScale(points) {
  const good = id => point(points[id]) && points[id].observable !== false && points[id].score > 0;
  const values = [];
  if (good(5) && good(6)) values.push(distance(xy(points[5]), xy(points[6])) / .225);
  if (good(11) && good(12)) values.push(distance(xy(points[11]), xy(points[12])) / .105);
  for (const [a, b] of [[5, 11], [6, 12]]) if (good(a) && good(b)) values.push(distance(xy(points[a]), xy(points[b])) / .29);
  return values.length >= 2 ? median(values) : null;
}

export class PoseStabilizer {
  constructor(options = {}) { this.options = { ...STABILITY_DEFAULTS, ...options }; this.reset(); }
  reset() {
    this.observationStates = Array(133).fill(null); this.jointStates = Array(133).fill(null);
    this.observationIdentity = null; this.poseIdentity = null; this.lastObservation = null;
    this.lastObservationTime = null; this.lastPoseTime = null; this.lastPose = null; this.bodyPixels = null;
  }
  get priorPose() { return this.lastPose; }

  filterObservation(input, overrides = {}) {
    if (!input || !Array.isArray(input.keypoints) || !finite(input.timestampMs)) return input;
    const opts = { ...this.options, ...overrides }, identity = observationKey(input), time = input.timestampMs;
    if (identity !== this.observationIdentity || (this.lastObservationTime !== null && (time < this.lastObservationTime || time - this.lastObservationTime > opts.resetGapMs))) this.reset();
    this.observationIdentity = identity;
    if (this.lastObservationTime === time && this.lastObservation) return this.lastObservation;
    const width = input.imageSize?.width, height = input.imageSize?.height, imageValid = finite(width) && finite(height) && width > 2 && height > 2;
    const reasons = {}, rejected = new Set();
    const mark = (id, reason) => { if (!rejected.has(id)) { rejected.add(id); reasons[reason] = (reasons[reason] || 0) + 1; } points[id] = missingPoint(points[id], id, reason); };
    const points = Array.from({ length: 133 }, (_, id) => ({ ...input.keypoints[id], id }));
    for (let id = 0; id < points.length; id++) {
      const p = points[id], previous = this.observationStates[id], [x, y] = xy(p);
      if (!point(p) || !finite(p.score) || p.observable === false || NON_OBSERVATIONS.has(p.source)) { mark(id, 'notObserved'); continue; }
      const threshold = opts.minScore + (previous?.tracked ? 0 : opts.enterMargin);
      if (p.score < threshold) { mark(id, 'lowConfidence'); continue; }
      if (!imageValid || x < opts.borderPixels || y < opts.borderPixels || x > width - 1 - opts.borderPixels || y > height - 1 - opts.borderPixels) { mark(id, 'outsideImage'); continue; }
      points[id] = { ...p, x, y, observable: true, observedAtMs: time, rawScore: p.score };
    }
    const current = id => points[id]?.observable !== false && point(points[id]) && points[id].score > 0;
    const initialScale = torsoScale(points);
    const scale = initialScale || this.bodyPixels || (imageValid ? Math.max(width, height) * .65 : 1);
    // A collapsed/truncated torso must not manufacture a full-body orientation.
    let torsoValid = TORSO.every(current);
    if (torsoValid) {
      const shoulder = midpoint(xy(points[5]), xy(points[6])), hip = midpoint(xy(points[11]), xy(points[12]));
      const torso = distance(shoulder, hip), sw = distance(xy(points[5]), xy(points[6])), hw = distance(xy(points[11]), xy(points[12]));
      const left = distance(xy(points[5]), xy(points[11])), right = distance(xy(points[6]), xy(points[12]));
      torsoValid = torso > Math.max(8, Math.min(width, height) * .035) && sw > 3 && hw > 3 && sw / torso < 2.4 && hw / torso < 1.7 && Math.max(left, right) / Math.max(1, Math.min(left, right)) < 2.8;
      if (!torsoValid) for (const id of TORSO) mark(id, 'degenerateTorso');
    }
    // The projected length may shrink during foreshortening; only collapse and
    // impossible extension are rejected. A learned minimum would reject turns.
    for (const [a, b, ratio] of BODY_SEGMENTS) {
      // A visible elbow/wrist or knee/ankle pair still observes its own segment
      // when the shoulder/hip is outside the image. Do not erase that subchain.
      if (!current(a)) continue;
      if (!current(b)) continue;
      const d = distance(xy(points[a]), xy(points[b]));
      if (d < 1.5 || (initialScale && d > scale * ratio * 1.9)) mark(b, 'implausibleBoneLength');
    }
    // If an ankle/wrist is absent, nearby model hallucinations cannot resurrect
    // its foot/hand. The separate foot-only mode keeps its independent contract.
    for (const [anchor, ids] of [[15, [17, 18, 19]], [16, [20, 21, 22]], [9, Array.from({ length: 21 }, (_, i) => 91 + i)], [10, Array.from({ length: 21 }, (_, i) => 112 + i)]]) {
      if (!current(anchor) && input.inputMode !== 'feet') for (const id of ids) if (current(id)) mark(id, 'parentUnobserved');
    }
    for (let id = 0; id < points.length; id++) {
      const old = this.observationStates[id];
      if (!current(id)) { if (old) this.observationStates[id] = { ...old, tracked: false, pending: null }; continue; }
      const p = points[id], position = xy(p), dt = old ? Math.max(0, time - old.time) / 1000 : 0;
      const fast = !TORSO.includes(id), limit = scale * (.08 + (fast ? opts.maxLimbSpeed : opts.maxTorsoSpeed) * Math.min(dt, .25));
      const jumped = old && dt > 0 && dt < .65 && distance(position, old.position) > limit;
      const returning = old && !old.tracked;
      if (jumped || returning) {
        const pending = old.pending;
        const coherent = pending && time > pending.time && time - pending.time < 650 && distance(position, pending.position) <= Math.max(scale * .16, limit);
        const count = coherent ? pending.count + 1 : 1;
        if (count < opts.reacquireFrames) {
          mark(id, jumped ? 'temporalJump' : 'reacquiring');
          this.observationStates[id] = { ...old, tracked: false, pending: { position, time, count } };
          continue;
        }
      }
      this.observationStates[id] = { position, time, score: p.score, tracked: true, pending: null };
    }
    torsoValid = torsoValid && TORSO.every(current);
    const body = torsoScale(points);
    if (torsoValid && body && body > 16) this.bodyPixels = this.bodyPixels ? this.bodyPixels * .94 + body * .06 : body;
    const filtered = { ...input, keypoints: points, observationTimestampMs: time,
      stability: { observationTimestampMs: time, torsoValid, visibleTorsoPoints: TORSO.filter(current).length, acceptedPoints: points.filter(p => p.observable !== false && p.score > 0).length, rejectedPoints: rejected.size, rejectionReasons: reasons, bodyPixels: this.bodyPixels, state: points.some(p => p.observable !== false && p.score > 0) ? torsoValid ? 'tracking' : 'partial' : 'lost' } };
    this.lastObservation = filtered; this.lastObservationTime = time;
    return filtered;
  }

  update(input, { observation = this.lastObservation } = {}) {
    if (!input || !Array.isArray(input.joints) || !finite(input.timestampMs)) return input;
    const opts = this.options, identity = poseKey(input), time = input.timestampMs;
    if (identity !== this.poseIdentity || (this.lastPoseTime !== null && (time < this.lastPoseTime || time - this.lastPoseTime > opts.resetGapMs))) {
      this.jointStates = Array(133).fill(null); this.lastPose = null; this.lastPoseTime = null;
    }
    this.poseIdentity = identity;
    if (time === this.lastPoseTime && this.lastPose) return this.lastPose;
    const measuredAt = finite(observation?.timestampMs) ? observation.timestampMs : input.observationTimestampMs ?? time;
    const pose = { ...input, observationTimestampMs: measuredAt, joints: Array.from({ length: 133 }, (_, id) => ({ ...input.joints[id], id, position: input.joints[id]?.position?.slice() || null })) };
    const height = input.units === 'body-height' ? 1 : finite(input.diagnostics?.bodyHeight) ? input.diagnostics.bodyHeight : finite(opts.bodyHeightM) ? opts.bodyHeightM : 1.7;
    const scaleObserved = input.metric === true || input.diagnostics?.scaleCueCount >= 2;
    let rejected3D = 0;
    for (let id = 0; id < pose.joints.length; id++) {
      const j = pose.joints[id], old = this.jointStates[id];
      if (!vector(j.position) || !SOURCE_3D.has(j.source) || j.observable2D === false || !(j.score > 0)) { pose.joints[id] = missingJoint(j, id, j.reason || 'notObserved'); continue; }
      if (j.source !== 'measuredMultiView' && j.source !== 'estimatedMultiView' && observation?.keypoints?.[id]?.observable === false) { pose.joints[id] = missingJoint(j, id, 'observationRejected'); continue; }
      const dt = old ? (time - old.time) / 1000 : 0;
      const maxSpeed = TORSO.includes(id) ? opts.maxTorsoSpeed : opts.maxLimbSpeed;
      if (scaleObserved && old && dt > 0 && dt < .45 && distance(j.position, old.rawPosition) > height * (.12 + maxSpeed * dt)) {
        pose.joints[id] = missingJoint(j, id, 'implausible3DVelocity'); rejected3D++; continue;
      }
      const alpha = old && dt > 0 && dt < .25 ? 1 - Math.exp(-dt * 1000 / Math.max(1, opts.smoothingMs)) : 1;
      const rawPosition = j.position.slice();
      j.position = old ? mix(old.position, rawPosition, alpha) : rawPosition;
      if (vector(j.imagePlanePosition) && vector(old?.imagePlanePosition)) j.imagePlanePosition = mix(old.imagePlanePosition, j.imagePlanePosition, alpha);
      j.observedAtMs = measuredAt; j.ageMs = Math.max(0, time - measuredAt); j.observable2D = true;
      this.jointStates[id] = { rawPosition, position: j.position.slice(), imagePlanePosition: j.imagePlanePosition?.slice() || null, time };
    }
    for (const [a, b, ratio] of BODY_SEGMENTS) {
      const first = pose.joints[a], second = pose.joints[b];
      if (!vector(first?.position)) continue;
      if (vector(second?.position)) {
        const d = distance(first.position, second.position);
        if (d < height * .002 || (scaleObserved && d > height * ratio * 2)) { pose.joints[b] = missingJoint(second, b, 'implausible3DBoneLength'); rejected3D++; }
      }
    }
    for (const [anchor, ids] of [[15, [17, 18, 19]], [16, [20, 21, 22]], [9, Array.from({ length: 21 }, (_, i) => 91 + i)], [10, Array.from({ length: 21 }, (_, i) => 112 + i)]]) {
      if (!vector(pose.joints[anchor]?.position)) for (const id of ids) if (vector(pose.joints[id]?.position)) { pose.joints[id] = missingJoint(pose.joints[id], id, 'parentUnobserved'); rejected3D++; }
    }
    refreshPoseMetadata(pose);
    const torsoValid = TORSO.every(id => vector(pose.joints[id]?.position)) && observation?.stability?.torsoValid !== false;
    const measuredRoot = [11, 12].every(id => ['measuredMultiView', 'estimatedMultiView'].includes(pose.joints[id]?.source));
    const anchoredDepth = pose.diagnostics?.depthAnchorSource === 'providedCurrentMeasurement';
    if (!torsoValid || (!measuredRoot && !anchoredDepth)) pose.root = { ...pose.root, valid: false, position: null, reason: !torsoValid ? 'torsoUnobserved' : 'monocularRootAmbiguous' };
    pose.stability = { ...observation?.stability, observationTimestampMs: measuredAt, torsoValid, rejected3D, state: pose.joints.some(j => vector(j.position)) ? torsoValid ? 'tracking' : 'partial' : 'lost', monocularRootSuppressed: !measuredRoot && !anchoredDepth };
    pose.diagnostics = { ...pose.diagnostics, stability: pose.stability, observedJointCount: pose.joints.filter(j => vector(j.position)).length };
    this.lastPose = pose; this.lastPoseTime = time;
    return pose;
  }
}
