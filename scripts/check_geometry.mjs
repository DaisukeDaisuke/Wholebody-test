// Small physical-oracle check for Actions. This does not evaluate model accuracy.
import assert from 'node:assert/strict';
import { PoseFusion, canonicalName } from '../dist/geometry.js';

const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const K = [1000, 0, 640, 0, 1000, 480, 0, 0, 1];
const root = [0.15, -0.95, 3.2];
const anatomy = {
  left_shoulder: [0.35, -1.4, 3.19], right_shoulder: [-0.05, -1.4, 3.21],
  left_elbow: [0.58, -1.15, 3.05], right_elbow: [-0.3, -1.15, 3.28],
  left_wrist: [0.7, -0.95, 2.9], right_wrist: [-0.45, -0.92, 3.35],
  left_hip: [0.3, -0.95, 3.2], right_hip: [0, -0.95, 3.2],
  left_knee: [0.34, -0.5, 3.12], right_knee: [0, -0.5, 3.3],
  left_ankle: [0.32, -0.08, 3.1], right_ankle: [-0.02, -0.08, 3.24],
  left_heel: [0.32, -0.03, 3.18], right_heel: [-0.02, -0.03, 3.3],
  left_foot_index: [0.34, -0.04, 2.96], right_foot_index: [0, -0.04, 3.08],
};
const camera1 = { sourceId: 'video-1', K, R: I, t: [0, 1.2, 0], calibrated: true };
const c = Math.cos(Math.PI / 12), s = Math.sin(Math.PI / 12);
// Camera 2 is 1.2 m to the right, 1.2 m above the floor, looking 15° left.
const camera2 = { sourceId: 'video-2', K, R: [c, 0, s, 0, 1, 0, -s, 0, c], t: [-1.2 * c, 1.2, 1.2 * s], calibrated: true };
const transform = (p, camera) => [0, 1, 2].map(i => camera.R[3 * i] * p[0] + camera.R[3 * i + 1] * p[1] + camera.R[3 * i + 2] * p[2] + camera.t[i]);
const view = p => [p[0], -p[1], -p[2]];
const near = (actual, expected, label, tolerance = 1e-6) => {
  assert.ok(Array.isArray(actual), label + ': missing vector');
  assert.equal(actual.length, expected.length, label + ': vector shape');
  actual.forEach((value, i) => assert.ok(Number.isFinite(value) && Math.abs(value - expected[i]) < tolerance, `${label}[${i}]: ${value} versus ${expected[i]}`));
};
function observation(camera, withRelative3D, time = 1000) {
  const cameraRoot = transform(root, camera);
  return { sourceId: camera.sourceId, frameId: Math.round(time), mediaTimeMs: time, imageSize: { width: 1280, height: 960 }, detected: true,
    keypoints: Object.entries(anatomy).map(([name, position]) => {
      const p = transform(position, camera);
      assert.ok(p[2] > 0, 'Physical fixture is in front of camera');
      return { name, u: K[0] * p[0] / p[2] + K[2], v: K[4] * p[1] / p[2] + K[5],
        relativeXYZm: withRelative3D ? p.map((value, i) => value - cameraRoot[i]) : null,
        score: 0.95, score3D: 0.95, score2D: 0.9, observable: true, observationTimeMs: time };
    }) };
}
const options = { confidence: 0.4, smoothing: 0, lift2D: false, cameras: [camera1, camera2] };
const summaries = [];

const multi = new PoseFusion(options).update([observation(camera1, false), observation(camera2, false)]);
assert.equal(multi.root.distanceStatus, 'measuredMultiView');
assert.equal(multi.root.metricCalibrated, true);
near(multi.root.worldXYZm, view(root), 'triangulated world root');
near(multi.root.cameraXYZm, transform(root, camera1), 'triangulated camera root');
for (const [name, expected] of Object.entries(anatomy)) {
  const point = multi.keypoints.find(p => p.name === canonicalName(name));
  assert.equal(point?.source, 'measuredMultiView', name + ': triangulated source');
  near(point.position, view(expected), name);
}
summaries.push({ check: 'two-calibrated-camera physical reconstruction', points: multi.diagnostics.triangulated, passed: true });

const asynchronous = new PoseFusion(options).update([observation(camera1, false, 1000), observation(camera2, false, 1200)]);
assert.equal(asynchronous.root.valid, false);
assert.equal(asynchronous.diagnostics.triangulated, 0);
const uncalibrated = new PoseFusion({ ...options, cameras: [camera1, { ...camera2, calibrated: false }] })
  .update([observation(camera1, false), observation(camera2, false)]);
assert.equal(uncalibrated.root.valid, false);
assert.equal(uncalibrated.diagnostics.triangulated, 0);
const just2D = new PoseFusion({ ...options, lift2D: true }).update([observation(camera1, false)]);
assert.equal(just2D.root.valid, false);
assert.equal(just2D.root.cameraXYZm, null);
assert.ok(just2D.keypoints.every(p => p.position === null));
summaries.push({ check: 'unsynchronized, uncalibrated and 2D-only cannot invent measured depth', passed: true });

const mono = new PoseFusion(options).update([observation(camera1, true)]);
assert.equal(mono.root.distanceStatus, 'estimatedMonocular');
near(mono.root.cameraXYZm, transform(root, camera1), 'monocular learned-pose root translation');
near(mono.root.worldXYZm, view(root), 'monocular coordinate transform');
assert.equal(mono.root.metricCalibrated, false);
summaries.push({ check: 'relative 3D plus perspective recovers known root Z', rootCameraZ: mono.root.cameraXYZm[2], passed: true });

// A real 180° rigid rotation about (1,-1,0), not a matrix produced by the solver.
const reverseCamera = { sourceId: 'video-2', K, R: [0, -1, 0, -1, 0, 0, 0, 0, -1], t: [-1.2, 0, 6.4], calibrated: false };
const auto = new PoseFusion({ confidence: 0.4, smoothing: 0, cameras: [
  { sourceId: 'video-1', K, calibrated: false }, { sourceId: 'video-2', K, calibrated: false },
] });
const estimate = auto.estimateCameraAlignment([observation(camera1, true), observation(reverseCamera, true)]);
near(estimate.cameras[1].R, reverseCamera.R, '180-degree estimated rotation');
near(estimate.cameras[1].t, [0, 0, 6.4], 'estimated translation in camera-1 world');
assert.equal(estimate.calibrated, false);
assert.ok(estimate.residualM < 1e-6);
assert.ok(estimate.cameras.every(camera => camera.calibrated === false));
const autoPose = auto.update([observation(camera1, true), observation(reverseCamera, true)]);
assert.equal(autoPose.root.distanceStatus, 'estimatedMultiView');
assert.equal(autoPose.root.metricCalibrated, false);
summaries.push({ check: '180-degree auto alignment stays estimated', residualM: estimate.residualM, passed: true });

const shinLength = Math.hypot(...anatomy.left_ankle.map((v, i) => v - anatomy.left_knee[i]));
const lift = new PoseFusion({ ...options, lift2D: true, boneLengthsM: { 'left_knee:left_ankle': shinLength } });
lift.update([observation(camera1, true, 1000)]);
const partly2D = observation(camera1, true, 1033);
partly2D.keypoints.find(point => point.name === 'left_ankle').relativeXYZm = null;
const lifted = lift.update([partly2D]);
const ankle = lifted.keypoints.find(point => point.name === 'left_ankle');
assert.equal(ankle.source, 'boneConstrained');
assert.equal(ankle.depthEvidence, 'shortTemporalDepthPrior');
near(ankle.position, view(anatomy.left_ankle), 'ray/bone-constrained ankle', 1e-5);
assert.ok(lifted.diagnostics.lifted >= 1);
summaries.push({ check: 'partial 2D joint uses known bone and prior, not fixed Z', source: ankle.source, passed: true });

console.log(JSON.stringify({ purpose: 'Geometry and coordinate contracts only; not model-accuracy or real-device acceptance.', checks: summaries }, null, 2));
