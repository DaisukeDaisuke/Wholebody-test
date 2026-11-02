import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { triangulateJoint, fusePose, estimateMonocularPose, TemporalPoseTracker, ClockSync } from '../dist/geometry.mjs';
import { solvePlanarExtrinsics, validateCalibration, makeCamera } from '../dist/calibration.mjs';

// This experiment is intentionally executable only in GitHub Actions.
if (process.env.GITHUB_ACTIONS !== 'true') throw new Error('Run this evaluation in GitHub Actions.');

const output = process.argv[2] || 'evaluation/results/geometry-report.json';
const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const imageSize = { width: 1280, height: 720 };
const K = [1000, 1000, 640, 360];
const cameras = [
  { id: 'a', imageSize, K, Rcw: identity, tcw: [0, 0, 0], distortion: [0, 0, 0, 0, 0], calibrated: true },
  { id: 'b', imageSize, K, Rcw: identity, tcw: [-.6, 0, 0], distortion: [0, 0, 0, 0, 0], calibrated: true },
];
const distance = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
const missing = id => ({ id, position: null, score: 0, source: 'missing' });
const pixel = (p, cameraIndex = 0) => ({ x: 1000 * (p[0] - .6 * cameraIndex) / p[2] + 640, y: 1000 * p[1] / p[2] + 360, score: 1, observable: true });
const blankPoints = () => Array.from({ length: 133 }, () => ({ x: 0, y: 0, score: 0, observable: false }));
const observation = (sourceId, timestampMs, points, timeUncertaintyMs = 0) => ({ sourceId, timestampMs, timeUncertaintyMs, imageSize, keypoints: points });
const experiments = [];

function experiment(name, run) {
  try {
    const result = run();
    experiments.push({ name, ...result });
  } catch (error) {
    experiments.push({ name, passed: false, error: error.message });
  }
}

experiment('known-point-two-view-reconstruction', () => {
  const expected = [.1, .2, 3];
  const result = triangulateJoint(cameras.map((c, i) => ({ sourceId: c.id, keypoint: pixel(expected, i) })), cameras);
  const errorM = result.position ? distance(result.position, expected) : null;
  return { passed: result.valid && errorM < 1e-8 && result.source === 'measuredMultiView', expectedPositionM: expected, reconstructedPositionM: result.position || null, errorM, reprojectionErrorPx: result.reprojectionErrorPx ?? null };
});

experiment('subpixel-perturbation-sensitivity', () => {
  const errors = [];
  for (let i = 0; i < 30; i++) {
    const expected = [-.3 + i * .02, -.2 + .03 * (i % 7), 2 + .06 * i];
    const a = pixel(expected, 0), b = pixel(expected, 1);
    a.x += .5 * Math.sin(i * 1.7); a.y += .5 * Math.cos(i * 2.3);
    b.x -= .5 * Math.cos(i * .9); b.y -= .5 * Math.sin(i * 1.3);
    const result = triangulateJoint([{ sourceId: 'a', keypoint: a }, { sourceId: 'b', keypoint: b }], cameras);
    errors.push(result.position ? distance(result.position, expected) : Infinity);
  }
  const finite = errors.filter(Number.isFinite).sort((a, b) => a - b);
  return { passed: finite.length === 30 && finite.at(-1) < .05, sampleCount: errors.length, validCount: finite.length, noiseAmplitudePx: .5, maeM: finite.length ? finite.reduce((s, v) => s + v, 0) / finite.length : null, p95M: finite[Math.floor(.95 * (finite.length - 1))] ?? null, maximumM: finite.at(-1) ?? null, scope: 'Projection perturbation fixture; not image detector accuracy.' };
});

experiment('degenerate-and-behind-camera-rejection', () => {
  const point = [.1, .2, 3];
  const sameCenter = [{ ...cameras[0] }, { ...cameras[1], tcw: [0, 0, 0] }];
  const degenerate = triangulateJoint([{ sourceId: 'a', keypoint: pixel(point) }, { sourceId: 'b', keypoint: pixel(point) }], sameCenter);
  const behind = triangulateJoint(cameras.map((c, i) => ({ sourceId: c.id, keypoint: pixel([.1, .2, -3], i) })), cameras);
  const inconsistent = triangulateJoint([{ sourceId: 'a', keypoint: pixel(point) }, { sourceId: 'b', keypoint: { ...pixel(point, 1), y: pixel(point, 1).y + 80 } }], cameras);
  return { passed: !degenerate.valid && !behind.valid && !inconsistent.valid, degenerateReason: degenerate.reason, behindReason: behind.reason, inconsistentReason: inconsistent.reason };
});

experiment('known-planar-target-extrinsic-calibration', () => {
  const target = [[-.25, -.25], [.25, -.25], [.25, .25], [-.25, .25], [0, -.25], [0, .25]];
  const imagePoints = target.map(([x, y]) => [1000 * x / 3 + 640, 1000 * y / 3 + 360]);
  const result = solvePlanarExtrinsics({ id: 'a', imageSize, K, worldPoints: target, imagePoints, worldOrigin: [0, 0, 3], worldAxisX: [1, 0, 0], worldAxisY: [0, 1, 0], intrinsicsCalibrated: true });
  const rotationMaxError = Math.max(...result.Rcw.map((v, i) => Math.abs(v - identity[i]))), translationErrorM = distance(result.tcw, [0, 0, 0]);
  const assumed = solvePlanarExtrinsics({ id: 'a', imageSize, K, worldPoints: target, imagePoints, worldOrigin: [0, 0, 3], worldAxisX: [1, 0, 0], worldAxisY: [0, 1, 0], intrinsicsCalibrated: false });
  return { passed: result.calibrated && !assumed.calibrated && rotationMaxError < 1e-7 && translationErrorM < 1e-7, rotationMaxError, translationErrorM, reprojectionRmsPx: result.reprojectionRmsPx, guessedIntrinsicsMarkedCalibrated: assumed.calibrated };
});

experiment('manual-camera-is-explicitly-approximate', () => {
  const camera = makeCamera({ id: 'manual', imageSize, position: [0, 1.3, 0], target: [0, 1, 3], horizontalFovDeg: 65 });
  const validated = validateCalibration({ cameras: [camera] });
  return { passed: validated.valid && !camera.calibrated && camera.calibrationMethod === 'manualEstimate', validRotationAndCamera: validated.valid, errors: validated.errors, calibrationMethod: camera.calibrationMethod };
});

experiment('timestamp-and-uncertainty-rejection', () => {
  const a = blankPoints(), b = blankPoints();
  for (const [id, point] of [[11, [-.1, .2, 3]], [12, [.1, .2, 3]]]) { a[id] = pixel(point, 0); b[id] = pixel(point, 1); }
  const synchronized = fusePose([observation('a', 1000, a), observation('b', 1005, b)], cameras);
  const delayed = fusePose([observation('a', 1000, a), observation('b', 1100, b)], cameras);
  const unknown = fusePose([observation('a', 1000, a), observation('b', 1000, b, Infinity)], cameras);
  return { passed: synchronized.quality.observed3D === 2 && delayed.quality.observed3D === 0 && unknown.quality.observed3D === 0, synchronizedJointCount: synchronized.quality.observed3D, delayedJointCount: delayed.quality.observed3D, unknownTimingJointCount: unknown.quality.observed3D };
});

experiment('monocular-root-follows-forward-motion', () => {
  const relative = { 5: [-.2, -.3, -.05], 6: [.2, -.3, .05], 11: [-.1, .2, 0], 12: [.1, .2, 0] };
  const boneLengths = {};
  for (const [a, b] of [[5, 6], [11, 12], [5, 11], [6, 12]]) boneLengths[`${a}-${b}`] = distance(relative[a], relative[b]);
  const sample = (rootZ, t) => {
    const keypoints = blankPoints();
    for (const [id, p] of Object.entries(relative)) keypoints[Number(id)] = { ...pixel([p[0], p[1], p[2] + rootZ]), relativeZ: p[2] };
    return observation('a', t, keypoints);
  };
  const options = { bodyHeightM: 1.7, scaleConfirmed: true, boneLengths };
  const before = estimateMonocularPose(sample(3, 0), cameras[0], options);
  const after = estimateMonocularPose(sample(2.5, 100), cameras[0], { ...options, priorPose: before });
  const z0 = before.root?.position?.[2], z1 = after.root?.position?.[2], movementM = z0 - z1;
  return { passed: Number.isFinite(z0) && Number.isFinite(z1) && Math.abs(z0 - 3) < .01 && Math.abs(z1 - 2.5) < .02 && movementM > .45, expectedRootBeforeM: 3, expectedRootAfterM: 2.5, rootBeforeM: z0 ?? null, rootAfterM: z1 ?? null, movementM: Number.isFinite(movementM) ? movementM : null, secondFrameAnchor: after.diagnostics.depthAnchorSource, scope: 'Known bone lengths and relative depth fixture, not monocular camera accuracy.' };
});

experiment('occlusion-prediction-expires-without-refreshing-observation-age', () => {
  const tracker = new TemporalPoseTracker({ predictionMs: 180, holdMs: 1200, priorMs: 5000 });
  const pose = (t, point) => {
    const joints = Array.from({ length: 133 }, (_, i) => missing(i));
    if (point) joints[9] = { id: 9, position: point, score: 1, source: 'measuredMultiView' };
    return { timestampMs: t, joints, metric: true, coordinateFrame: 'world', units: 'm', worldCalibrationId: 'numeric-fixture' };
  };
  tracker.update(pose(0, [0, 0, 3]));
  tracker.update(pose(33.333, [.01, 0, 3]));
  const short = tracker.update(pose(66.666, null));
  const held = tracker.update(pose(500, null));
  const expired = tracker.update(pose(5500, null));
  return { passed: short.joints[9].source === 'predicted' && short.joints[9].ageMs > 30 && held.joints[9].source === 'inferredKinematic' && held.joints[9].ageMs > 400 && expired.joints[9].position === null, shortSource: short.joints[9].source, shortAgeMs: short.joints[9].ageMs, heldSource: held.joints[9].source, heldAgeMs: held.joints[9].ageMs, expiredSource: expired.joints[9].source };
});

experiment('clock-offset-with-network-delay', () => {
  const clock = new ClockSync();
  for (let i = 0; i < 5; i++) clock.addSample({ t0: i * 1000, t1: i * 1000 + 110, t2: i * 1000 + 111, t3: i * 1000 + 21 });
  const mapped = clock.toLocal(2110.5), errorMs = Math.abs(mapped.timestampMs - 2010.5);
  return { passed: clock.estimate().ready && errorMs < 1e-9 && mapped.timeUncertaintyMs >= 10, errorMs, uncertaintyMs: mapped.timeUncertaintyMs, note: 'Clock offset estimation does not by itself align a video frame to a metadata packet.' };
});

const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  gitCommit: process.env.GITHUB_SHA || null,
  runId: process.env.GITHUB_RUN_ID || null,
  scope: 'Numerical geometry/calibration/timing fixtures. These are not real-video body, finger, toe or occlusion accuracy results.',
  realVideoAccuracyMeasured: false,
  realMotionOcclusionMeasured: false,
  experiments,
  passed: experiments.every(e => e.passed),
  references: [
    'https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html',
    'https://wicg.github.io/video-rvfc/',
  ],
};
await mkdir(dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
console.log(JSON.stringify({ output, passed: report.passed, passedCount: experiments.filter(e => e.passed).length, total: experiments.length, failed: experiments.filter(e => !e.passed).map(e => e.name) }));
if (!report.passed) process.exitCode = 1;
