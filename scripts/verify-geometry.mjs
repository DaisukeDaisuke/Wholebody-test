// A small geometry acceptance gate. Run only in GitHub Actions.
// Synthetic correspondences verify implementation, never real-camera accuracy.
import assert from 'node:assert/strict';
import {mkdir, writeFile} from 'node:fs/promises';
import {projectPoint, triangulateJoint, estimateRootDepth, BODY_BONES} from '../dist/geometry.js';
import {OcclusionTracker} from '../dist/tracking.js';

const rotation = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const camera = (id, translation, distortion = [0, 0, 0, 0, 0]) => ({
  id, width: 1280, height: 720, fx: 900, fy: 920, cx: 640, cy: 360,
  distortion, worldFromCamera: {rotation, translation}, calibrated: true, intrinsicsCalibrated: true,
});
const cameras = {a: camera('a', [0, 0, 0], [-.08, .01, .001, -.001, 0]),
                 b: camera('b', [.8, 0, 0], [-.05, .005, 0, .001, 0])};
const truth = [.12, -.24, 3.2];
const observations = Object.values(cameras).map(c => ({
  sourceId: c.id, timeMs: 1000, score: .95, ...projectPoint(truth, c),
}));
const reconstructed = triangulateJoint(observations, cameras);
assert.equal(reconstructed.valid, true, JSON.stringify(reconstructed));
const positionErrorM = Math.hypot(...truth.map((v, i) => v - reconstructed.xyz[i]));
assert.ok(positionErrorM < 1e-5, `Calibrated distorted-ray reconstruction error ${positionErrorM} m`);

const rejection = {};
rejection.unsynchronized = triangulateJoint(observations.map((o, i) => ({...o, timeMs: 1000 + i * 100})), cameras);
rejection.singleCamera = triangulateJoint([observations[0], observations[0]], cameras);
rejection.unmeasuredCalibration = triangulateJoint(observations, {a: {...cameras.a, calibrated: false}, b: cameras.b});
const plain = {a: camera('a', [0, 0, 0]), b: camera('b', [.8, 0, 0])};
rejection.parallel = triangulateJoint([
  {sourceId: 'a', timeMs: 1000, u: 640, v: 360, score: .9},
  {sourceId: 'b', timeMs: 1000, u: 640, v: 360, score: .9},
], plain);
rejection.inconsistent = triangulateJoint(observations.map((o, i) => ({...o, v: o.v + i * 120})), cameras);
rejection.behindCameras = triangulateJoint([
  {sourceId: 'a', timeMs: 1000, u: 610, v: 360, score: .9},
  {sourceId: 'b', timeMs: 1000, u: 850, v: 360, score: .9},
], plain);
for (const [name, result] of Object.entries(rejection)) assert.equal(result.valid, false, `${name}: ${JSON.stringify(result)}`);
assert.equal(projectPoint([0, 0, -1], cameras.a), null);

const knownJoints = {
  5: [-.2, -.5, 3], 6: [.2, -.5, 3], 7: [-.45, -.2, 3.1], 8: [.45, -.2, 2.9],
  9: [-.6, .05, 3.15], 10: [.6, .05, 2.85], 11: [-.15, .1, 3], 12: [.15, .1, 3],
  13: [-.15, .5, 3.05], 14: [.15, .5, 2.95], 15: [-.15, .9, 3], 16: [.15, .9, 3],
};
const points = Object.entries(knownJoints).map(([id, xyz]) => ({
  id: Number(id), ...projectPoint(xyz, plain.a), score: .95, relativeZm: xyz[2] - 3,
}));
const bones = Object.fromEntries(BODY_BONES.map(([a, b]) => [
  `${a}-${b}`, Math.hypot(...knownJoints[a].map((v, i) => v - knownJoints[b][i])),
]));
const depth = estimateRootDepth(points, plain.a, bones, {boneLengthSource: 'measured'});
assert.equal(depth.valid, true, JSON.stringify(depth));
assert.ok(Math.abs(depth.depthM - 3) < .01, `Known root depth ${depth.depthM}`);
const changedRoi = estimateRootDepth(points.map(p => ({...p, bbox: [0, 0, 10, 10]})), plain.a, bones, {boneLengthSource: 'measured'});
assert.equal(changedRoi.valid, true);
assert.ok(Math.abs(changedRoi.depthM - depth.depthM) < 1e-10, 'ROI dimensions must not set absolute distance');
const noRelativeDepth = estimateRootDepth(points.map(({relativeZm, ...p}) => p), plain.a, bones);
assert.equal(noRelativeDepth.valid, false, '2D-only observations must not silently become measured 3D');

const tracker = new OcclusionTracker({smoothingMs: 0});
const visible = tracker.update({timeMs: 0, joints: [{id: 11, xyz: [0, 0, 3], confidence: .9, source: 'triangulated'}]});
assert.equal(visible.joints[11].source, 'triangulated');
const occluded = tracker.update({timeMs: 100, joints: []});
assert.equal(occluded.joints[11].source, 'predicted');
assert.equal(occluded.joints[11].lastObservedTimeMs, 0);
assert.equal(occluded.joints[11].observed, false);
const expired = tracker.update({timeMs: 2000, joints: [occluded.joints[11]]});
assert.equal(expired.joints[11].source, 'missing', 'Prediction fed back as input must not renew observation age');
const restored = tracker.update({timeMs: 2100, joints: [{id: 11, xyz: [.1, 0, 3], confidence: .9, source: 'triangulated'}]});
assert.equal(restored.joints[11].source, 'triangulated');
assert.deepEqual(restored.joints[11].xyz, [.1, 0, 3]);

const report = {
  kind: 'synthetic-geometry-contract', passed: true,
  scope: 'Mathematical and provenance checks; not measured real-video or real-camera accuracy.',
  reconstructionErrorM: positionErrorM, knownRootDepthM: depth.depthM,
  rejected: Object.fromEntries(Object.entries(rejection).map(([name, result]) => [name, result.reason])),
  occlusion: {predictedAtMs: 100, expiredAtMs: 2000, lastObservedTimeMsDuringPrediction: 0, recoveredAtMs: 2100},
};
await mkdir('artifacts', {recursive: true});
await writeFile('artifacts/geometry.json', `${JSON.stringify(report, null, 2)}\n`, 'utf8');
console.log(JSON.stringify(report, null, 2));
