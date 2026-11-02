/** Aggregate real inference observations through the browser's exact pose path.
 * No image/video IO here: headless-video.mjs owns decoding and ONNX inference.
 * Challenge rows are explicitly injected observation faults, not accuracy data. */
import { PoseStabilizer } from '../dist/pose-stability.mjs';
import { estimateMonocularPose } from '../dist/geometry.mjs';
import { retargetPose, quaternionAngle, multiplyQuaternion } from '../dist/retarget.mjs';

const RAD = 180 / Math.PI;
const IDENTITY = [0, 0, 0, 1];
const finite = Number.isFinite;
const vector = (p, n) => Array.isArray(p) && p.length === n && p.every(finite);
const inverse = q => [-q[0], -q[1], -q[2], q[3]];
const distance = (a, b) => Math.hypot(...a.map((n, i) => n - b[i]));
const maximum = values => values.reduce((m, value) => finite(value) ? Math.max(m, value) : m, 0);
const percentile = (a, p) => { const sorted = a.filter(finite).sort((x, y) => x - y); return sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))] : null; };
const BODY_BONES = { leftUpperArm: [5, 7], leftLowerArm: [7, 9], rightUpperArm: [6, 8], rightLowerArm: [8, 10], leftUpperLeg: [11, 13], leftLowerLeg: [13, 15], rightUpperLeg: [12, 14], rightLowerLeg: [14, 16] };
const circularDifference = (a, b) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));

function pipeline(rig, options) {
  const stabilizer = new PoseStabilizer(options.stabilityOptions || {});
  let pose = null, previous = null;
  return {
    observe(observation) {
      const filtered = stabilizer.filterObservation(observation, { minScore: options.minScore ?? .25 });
      const raw = estimateMonocularPose(filtered, options.camera || null, { ...options.geometryOptions, minScore: options.minScore ?? .25, priorPose: stabilizer.priorPose });
      pose = stabilizer.update(raw, { observation: filtered });
      return { filtered, pose };
    },
    render(time) {
      if (!pose) return null;
      previous = retargetPose({ ...pose, timestampMs: time }, rig, { ...options.retargetOptions, previous });
      return previous;
    },
    reset() { stabilizer.reset(); pose = null; previous = null; },
  };
}

function recorder(rig) {
  const bones = {}, jointReasons = {}, angles = [], speeds = [];
  let previous = null, finiteRotations = true, renderFrames = 0, framesWithDrivenBones = 0, maxRootStep = 0, maxRootSpeed = 0, maxUnobservedTorsoTilt = 0;
  let observationFrames = 0, acceptedPoints = 0, submittedPoints = 0, torsoFrames = 0, partialFrames = 0, lostFrames = 0;
  function state(name) { return bones[name] ||= { observedFrames: 0, heldFrames: 0, returningFrames: 0, neutralFrames: 0, maxFrameRotationDeg: 0, maxAngularSpeedDegPerSec: 0, renderedWorldRangeDeg: 0, acceptedImageDirectionRangeDeg: 0, firstWorld: null, firstDirection: null }; }
  return {
    observation(filtered) {
      observationFrames++; const stability = filtered.stability || {};
      acceptedPoints += stability.acceptedPoints || 0;
      submittedPoints += (filtered.keypoints || []).filter(p => finite(p?.rawScore ?? p?.score) && (p.rawScore ?? p.score) > 0).length;
      if (stability.state === 'tracking') torsoFrames++; else if (stability.state === 'partial') partialFrames++; else lostFrames++;
      for (const [reason, count] of Object.entries(stability.rejectionReasons || {})) jointReasons[reason] = (jointReasons[reason] || 0) + count;
      for (const [name, [a, b]] of Object.entries(BODY_BONES)) {
        const x = filtered.keypoints[a], y = filtered.keypoints[b];
        if (!x || !y || x.observable === false || y.observable === false || ![x.x, x.y, y.x, y.y].every(finite)) continue;
        const angle = Math.atan2(-(y.y - x.y), y.x - x.x), current = state(name);
        current.firstDirection ??= angle;
        current.acceptedImageDirectionRangeDeg = Math.max(current.acceptedImageDirectionRangeDeg, circularDifference(angle, current.firstDirection) * RAD);
      }
    },
    render(result) {
      if (!result) return;
      renderFrames++; if (result.coverage.drivenBones > 0) framesWithDrivenBones++;
      const dt = previous ? (result.timestampMs - previous.timestampMs) / 1000 : 0;
      for (const [name, bone] of Object.entries(result.bones)) {
        const current = state(name);
        finiteRotations &&= vector(bone.rotation, 4) && vector(bone.worldRotation, 4) && Math.abs(Math.hypot(...bone.rotation) - 1) < 1e-5;
        if (bone.observed) { current.observedFrames++; current.firstWorld ||= bone.worldRotation; }
        if (bone.trackingState === 'holding') current.heldFrames++;
        if (bone.trackingState === 'returning') current.returningFrames++;
        if (bone.trackingState === 'neutral') current.neutralFrames++;
        if (current.firstWorld) current.renderedWorldRangeDeg = Math.max(current.renderedWorldRangeDeg, quaternionAngle(current.firstWorld, bone.worldRotation) * RAD);
        const old = previous?.bones?.[name];
        if (old && dt > 1e-6) {
          const angle = quaternionAngle(old.rotation, bone.rotation) * RAD, speed = angle / dt;
          current.maxFrameRotationDeg = Math.max(current.maxFrameRotationDeg, angle); current.maxAngularSpeedDegPerSec = Math.max(current.maxAngularSpeedDegPerSec, speed);
          angles.push(angle); speeds.push(speed);
        }
      }
      if (previous && vector(result.root.visualPosition, 3) && vector(previous.root.visualPosition, 3) && dt > 1e-6) {
        const d = distance(result.root.visualPosition, previous.root.visualPosition);
        maxRootStep = Math.max(maxRootStep, d); maxRootSpeed = Math.max(maxRootSpeed, d / dt);
      }
      if (result.warnings.includes('monocularTorsoYawPitchUnobserved') && result.bones.hips && rig.bones?.hips) {
        const delta = multiplyQuaternion(result.bones.hips.worldRotation, inverse(rig.bones.hips.rotation || IDENTITY));
        const unwanted = 2 * Math.asin(Math.min(1, Math.hypot(delta[0], delta[1]))) * RAD;
        maxUnobservedTorsoTilt = Math.max(maxUnobservedTorsoTilt, unwanted);
      }
      previous = result;
    },
    finish() {
      const summary = Object.fromEntries(Object.entries(bones).map(([name, { firstWorld, firstDirection, ...value }]) => [name, value]));
      const moving = Object.keys(BODY_BONES).filter(name => summary[name]?.acceptedImageDirectionRangeDeg >= 12 && summary[name]?.observedFrames >= 3);
      const retained = moving.filter(name => summary[name].renderedWorldRangeDeg >= 4);
      return { observationFrames, renderFrames, framesWithDrivenBones, torsoFrames, partialFrames, lostFrames, acceptedPoints, submittedPoints,
        rejectionReasons: jointReasons, finiteRotations, maxFrameRotationDeg: maximum(angles),
        p95FrameRotationDeg: percentile(angles, .95), maxAngularSpeedDegPerSec: maximum(speeds),
        maxRootStep, maxRootSpeed, maxUnobservedTorsoTiltDeg: maxUnobservedTorsoTilt,
        motionEvidence: { criterion: 'accepted image limb direction changes >=12 degrees; observed bone has >=3 rendered frames', movingInputBones: moving, retainedMotionBones: retained, allMovingBonesFrozen: moving.length > 0 && retained.length === 0, conclusion: moving.length ? retained.length ? 'observableMotionRetained' : 'observableMotionLost' : 'insufficientAcceptedMotionToAssess' }, bones: summary };
    },
  };
}

function faultChallenge(seed, rig, options) {
  const process = pipeline(rig, options), result = recorder(rig), period = 1000 / 30;
  let time = 0, output = null;
  const at = (kind = 'normal') => {
    const observation = { ...seed, timestampMs: time, observationTimestampMs: time,
      keypoints: seed.keypoints.map(p => p ? { ...p } : p), sourceId: seed.sourceId || 'public-challenge' };
    if (kind === 'missing') observation.keypoints = observation.keypoints.map((p, id) => ({ ...p, id, score: 0, observable: false }));
    if (kind === 'spike') observation.keypoints = observation.keypoints.map((p, id) => ({ ...p, id, x: 1e7, y: -1e7, score: 1, observable: true }));
    if (kind === 'partial') for (const id of [5, 6, 11, 12]) observation.keypoints[id] = { ...observation.keypoints[id], score: 0, observable: false };
    const update = process.observe(observation); result.observation(update.filtered);
    output = process.render(time); result.render(output); time += period;
    return { ...update, output };
  };
  for (let i = 0; i < 14; i++) at();
  const warm = output, warmDriven = warm.coverage.drivenBones;
  const hadMotion = Math.max(0, ...Object.values(warm.bones).map(b => quaternionAngle(b.rotation, b.neutralRotation || IDENTITY) * RAD));
  const spike = at('spike'), spikeJump = Math.max(0, ...Object.entries(spike.output.bones).filter(([name]) => warm.bones[name]?.observed).map(([name, b]) => quaternionAngle(warm.bones[name].rotation, b.rotation) * RAD));
  // Keep genuinely inferred visible lower segments while the proximal torso is
  // made unobserved. This catches a filter that simply freezes the whole avatar.
  for (let i = 0; i < 6; i++) at();
  const expectedPartial = Object.entries(BODY_BONES).filter(([name, ids]) => /Lower/.test(name) && ids.every(id => seed.keypoints[id]?.observable !== false && seed.keypoints[id]?.score >= .29)).map(([name]) => name);
  for (let i = 0; i < 18; i++) at('partial');
  const retainedPartial = expectedPartial.filter(name => output.bones[name]?.observed);
  const partialRootSuppressed = output.root.metric === false && !output.trackers.some(t => t.role === 'waist' && t.valid);
  for (let i = 0; i < 12; i++) at();
  const beforeLoss = output;
  at('missing'); const earlyLoss = output;
  const earlyHoldError = Math.max(0, ...Object.entries(beforeLoss.bones).filter(([, b]) => b.observed).map(([name, b]) => quaternionAngle(b.rotation, earlyLoss.bones[name].rotation) * RAD));
  const physicalTrackersDuringLoss = [output.trackers.filter(t => t.valid).length];
  for (let i = 0; i < 39; i++) { at('missing'); physicalTrackersDuringLoss.push(output.trackers.filter(t => t.valid).length); }
  const neutralError = Math.max(0, ...Object.values(output.bones).map(b => quaternionAngle(b.rotation, b.neutralRotation || IDENTITY) * RAD));
  const recovery = [];
  for (let i = 0; i < 5; i++) { at(); recovery.push(output.coverage.drivenBones); }
  const aggregate = result.finish();
  return { protocol: 'repeat one real inferred observation for 14 frames; inject one out-of-image high-confidence spike; restore for 6 frames; remove shoulders/hips for 18 frames while retaining inferred lower segments; restore for 12 frames; remove every observation for 40 frames; restore for 5 frames; 30 Hz. These are observation-fault challenges, not model-accuracy measurements.',
    baselineDrivenBones: warmDriven, baselineMaxDeviationFromNeutralDeg: hadMotion, neutralPolicy: output.neutralPolicy,
    spikeAcceptedPoints: spike.filtered.stability?.acceptedPoints || 0, spikeMaxObservedBoneRotationDeg: spikeJump,
    earlyHoldErrorDeg: earlyHoldError, neutralErrorDegAfterLoss: neutralError, recoveryDrivenBones: recovery,
    partialObservation: { applicable: expectedPartial.length > 0, expectedVisibleSubchains: expectedPartial, retainedVisibleSubchains: retainedPartial, rootSuppressed: partialRootSuppressed },
    validPhysicalTrackersDuringLoss: Math.max(...physicalTrackersDuringLoss), aggregate,
    checks: { usefulBaseline: warmDriven >= 2 && hadMotion > 3, spikeRejected: spike.filtered.stability?.acceptedPoints === 0,
      spikeDidNotJump: spikeJump < .01, briefLossHolds: earlyHoldError < .01, lostReturnsToNeutral: neutralError < .5,
      partialVisibleSubchainsRemain: expectedPartial.length === 0 || retainedPartial.length > 0, partialTorsoDoesNotMoveRoot: partialRootSuppressed,
      noFreshTrackersWhileLost: physicalTrackersDuringLoss.every(n => n === 0), firstRecoveryFrameNotFresh: recovery[0] === 0,
      sustainedRecoveryResumesMotion: Math.max(...recovery.slice(1)) >= Math.max(2, Math.floor(warmDriven * .5)), finiteRotations: aggregate.finiteRotations } };
}

export function evaluateStabilitySequence(input, rig, options = {}) {
  const observations = (Array.isArray(input) ? input : input?.observations || input?.frames || []).map(x => x?.observation || x).filter(x => x && Array.isArray(x.keypoints) && finite(x.timestampMs));
  if (!rig?.bones || !observations.length) return { schemaVersion: 1, pass: false, reason: !rig?.bones ? 'rigMissing' : 'noActualInferenceObservations', actual: null, challenge: null, checks: {} };
  const process = pipeline(rig, options), results = recorder(rig), renderPeriod = 1000 / (options.renderFps || 60);
  const render = time => { const result = process.render(time); results.render(result); if (result) options.onActualRender?.(result); };
  let nextRender = observations[0].timestampMs, lastTime = observations[0].timestampMs, seed = null, seedCount = -1;
  for (const observation of observations) {
    if (observation.timestampMs < lastTime) throw new Error('Actual observations must retain increasing capture timestamps.');
    // Limit unbounded timestamp gaps without treating redraws as measurements.
    while (nextRender < observation.timestampMs && nextRender <= lastTime + 2000) { render(nextRender); nextRender += renderPeriod; }
    if (nextRender < observation.timestampMs - renderPeriod) nextRender = observation.timestampMs;
    const { filtered } = process.observe(observation); results.observation(filtered);
    const usable = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].filter(id => filtered.keypoints[id]?.observable !== false && filtered.keypoints[id]?.score > 0).length;
    if (usable > seedCount) { seed = observation; seedCount = usable; }
    render(observation.timestampMs);
    nextRender = Math.max(nextRender, observation.timestampMs + renderPeriod); lastTime = observation.timestampMs;
  }
  const actual = results.finish(), challenge = options.includeChallenges === false ? null : faultChallenge(seed, rig, options);
  const checks = { actualInferenceFrames: observations.length > 1, finiteRotations: actual.finiteRotations,
    boundedAngularVelocity: actual.maxAngularSpeedDegPerSec <= 900.01, boundedRootVelocity: actual.maxRootSpeed <= (options.retargetOptions?.maxRootSpeed ?? 2) + .001,
    monocularYawPitchSuppressed: actual.maxUnobservedTorsoTiltDeg < .01,
    noCompleteMotionFreeze: !actual.motionEvidence.allMovingBonesFrozen,
    usefulActualTracking: actual.framesWithDrivenBones > 0,
    ...(challenge ? Object.fromEntries(Object.entries(challenge.checks).map(([key, value]) => [`challenge_${key}`, value])) : {}) };
  return { schemaVersion: 1, source: 'real ONNX inference observations; aggregate only', actual, challenge, checks, pass: Object.values(checks).every(value => value === true),
    limitations: ['These metrics evaluate stability and preservation of accepted observable motion, not 3D pose accuracy.', 'Monocular torso yaw/pitch and unanchored absolute root are intentionally unobserved.', 'The fault challenge replays inferred landmarks and injects missing/out-of-image observations; it does not claim new model accuracy.'] };
}
