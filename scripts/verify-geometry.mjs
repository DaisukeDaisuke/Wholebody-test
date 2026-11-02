// A small geometry acceptance gate. Run only in GitHub Actions.
// Synthetic correspondences verify implementation, never real-camera accuracy.
import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {dirname} from 'node:path';
import {projectPoint, triangulateJoint, estimateRootDepth, calibratePlanarCamera, BODY_BONES} from '../dist/geometry.js';
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

const intrinsics = {width: 1280, height: 720, fx: 900, fy: 920, cx: 640, cy: 360,
  distortion: [-.04, .01, 0, 0, 0], intrinsicsCalibrated: true};
const c = Math.cos(Math.PI / 6), s = Math.sin(Math.PI / 6);
const groundCamera = {...intrinsics, id: 'floor', calibrated: true,
  worldFromCamera: {rotation: [1, 0, 0, 0, -c, -s, 0, s, -c], translation: [.5, 1.6, 3]}};
const floor = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]];
const imagePoints = floor.map(p => {const q = projectPoint(p, groundCamera); return [q.u, q.v];});
const calibrated = calibratePlanarCamera({imagePoints, worldPoints: floor, intrinsics, id: 'floor'});
assert.equal(calibrated.valid, true, JSON.stringify(calibrated));
assert.equal(calibrated.camera.calibrated, true);
const cameraCenterErrorM = Math.hypot(...groundCamera.worldFromCamera.translation.map((v, i) =>
  v - calibrated.camera.worldFromCamera.translation[i]));
assert.ok(cameraCenterErrorM < 1e-3, `Floor calibration center error ${cameraCenterErrorM} m`);
const offFloor = [.3, .5, .6];
const expectedPixel = projectPoint(offFloor, groundCamera), recoveredPixel = projectPoint(offFloor, calibrated.camera);
const offFloorReprojectionErrorPx = Math.hypot(expectedPixel.u - recoveredPixel.u, expectedPixel.v - recoveredPixel.v);
assert.ok(offFloorReprojectionErrorPx < .01, `Floor calibration off-plane error ${offFloorReprojectionErrorPx} px`);
const approximate = calibratePlanarCamera({imagePoints, worldPoints: floor, intrinsics: {...intrinsics, intrinsicsCalibrated: false}});
assert.equal(approximate.valid, true);
assert.equal(approximate.camera.calibrated, false, 'Guessed intrinsics cannot become measured calibration');
assert.equal(approximate.calibrationStatus, 'approximate');
const degenerate = calibratePlanarCamera({imagePoints: [[1, 1], [2, 1], [3, 1], [4, 1]],
  worldPoints: [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], intrinsics});
assert.equal(degenerate.valid, false);

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
  floorCalibration: {cameraCenterErrorM, offFloorReprojectionErrorPx,
    reprojectionErrorPx: calibrated.reprojectionErrorPx, approximateStatus: approximate.calibrationStatus},
  rejected: Object.fromEntries(Object.entries(rejection).map(([name, result]) => [name, result.reason])),
  occlusion: {predictedAtMs: 100, expiredAtMs: 2000, lastObservedTimeMsDuringPrediction: 0, recoveredAtMs: 2100},
};
await mkdir('artifacts', {recursive: true});
await writeFile('artifacts/geometry.json', `${JSON.stringify(report, null, 2)}\n`, 'utf8');
console.log(JSON.stringify(report, null, 2));

// Optional measured-sequence evaluation. The input stays in the download cache;
// only aggregate errors, coverage and provenance are written into the artifact.
const cli = process.argv.slice(2);
const option = name => {const at = cli.indexOf(name); return at < 0 ? null : cli[at + 1];};
const panopticInput = option('--panoptic-input');
if (panopticInput) {
  const data = JSON.parse(await readFile(panopticInput, 'utf8'));
  assert.equal(data.schemaVersion, 1);
  assert.ok(data.cameras && Array.isArray(data.conditions) && data.conditions.length > 0);
  const outputPath = option('--panoptic-out') || 'artifacts/panoptic/geometry.json';
  const measured = {
    schemaVersion: 1, kind: 'real-multiview-pose-evaluation', sequence: data.sequence, fps: data.fps,
    sourceCommit: process.env.GITHUB_SHA || null,
    methods: {triangulation: 'dist/geometry.js triangulateJoint', tracking: 'dist/tracking.js OcclusionTracker'},
    cameraIds: Object.keys(data.cameras),
    scope: 'Real images and calibrated CMU Panoptic 3D ground truth. GT boxes isolate pose estimation from detection.',
    limitations: [
      'Ground-truth coordinates are consumed privately for scoring and are not included in this report.',
      'Both-hands-drop removes observations after image inference; it is not an image-occlusion benchmark.',
      'MPJPE is conditional on an available estimate. Coverage and PCK use all valid GT joints.',
      'The body set contains 17 COCO joints. Individual toe joints and instep surface are not labeled here.',
      'This sequence does not establish performance on user cameras or every type of occlusion.',
    ],
    conditions: [],
  };

  const finite3 = v => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);
  const jointGroup = id => id >= 91 && id <= 132 ? 'hands' : id >= 0 && id <= 16 ? 'body' : null;
  const quantile = (values, p) => {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const at = (sorted.length - 1) * p, lo = Math.floor(at), hi = Math.ceil(at);
    return sorted[lo] * (hi - at || 1) + (hi === lo ? 0 : sorted[hi] * (at - lo));
  };
  const newGroup = () => ({groundTruthJoints: 0, availableJoints: 0, errorsMm: [], sources: {}});
  const newMetrics = () => ({body: newGroup(), hands: newGroup()});
  const addFrame = (metrics, gt, predictions) => {
    const byId = new Map(predictions.map(j => [j.id, j]));
    for (const point of gt) {
      const groupName = jointGroup(point.id);
      if (!groupName || point.valid === false || !finite3(point.xyz)) continue;
      const group = metrics[groupName];
      group.groundTruthJoints++;
      const predicted = byId.get(point.id);
      if (!predicted || predicted.valid === false || !finite3(predicted.xyz)) {
        group.sources.missing = (group.sources.missing || 0) + 1;
        continue;
      }
      group.availableJoints++;
      group.errorsMm.push(Math.hypot(...predicted.xyz.map((v, axis) => v - point.xyz[axis])) * 1000);
      group.sources[predicted.source || 'unknown'] = (group.sources[predicted.source || 'unknown'] || 0) + 1;
    }
  };
  const finish = metrics => Object.fromEntries(Object.entries(metrics).map(([name, m]) => [name, {
    groundTruthJoints: m.groundTruthJoints, availableJoints: m.availableJoints,
    missingJoints: m.groundTruthJoints - m.availableJoints,
    coverage: m.groundTruthJoints ? m.availableJoints / m.groundTruthJoints : null,
    mpjpeMmAvailable: m.errorsMm.length ? m.errorsMm.reduce((a, b) => a + b, 0) / m.errorsMm.length : null,
    medianErrorMmAvailable: quantile(m.errorsMm, .5), p95ErrorMmAvailable: quantile(m.errorsMm, .95),
    pck20mmAll: m.groundTruthJoints ? m.errorsMm.filter(e => e <= 20).length / m.groundTruthJoints : null,
    pck50mmAll: m.groundTruthJoints ? m.errorsMm.filter(e => e <= 50).length / m.groundTruthJoints : null,
    pck100mmAll: m.groundTruthJoints ? m.errorsMm.filter(e => e <= 100).length / m.groundTruthJoints : null,
    sources: m.sources,
  }]));

  for (const condition of data.conditions) {
    assert.ok(condition.frames.length >= 2, 'Measured sequence needs at least two frames');
    for (let i = 1; i < condition.frames.length; i++) {
      assert.ok(condition.frames[i].timeMs > condition.frames[i - 1].timeMs, 'Frame times must be strictly increasing');
    }
    const dropCount = Math.min(8, Math.max(1, Math.floor(condition.frames.length / 4)));
    const dropStart = Math.floor((condition.frames.length - dropCount) / 2);
    const result = {name: condition.name, frameCount: condition.frames.length,
      firstFrame: condition.frames[0].index, lastFrame: condition.frames.at(-1).index,
      artificialDrop: {firstFrame: condition.frames[dropStart].index,
        lastFrame: condition.frames[dropStart + dropCount - 1].index, frameCount: dropCount,
        removedJointIds: '91..132 in both cameras, after image inference'}, scenarios: {}};
    for (const drop of [false, true]) {
      const tracker = new OcclusionTracker();
      const rawAll = newMetrics(), trackedAll = newMetrics(), rawGap = newMetrics(), trackedGap = newMetrics();
      const frameMetrics = [];
      for (let i = 0; i < condition.frames.length; i++) {
        const frame = condition.frames[i];
        const inGap = i >= dropStart && i < dropStart + dropCount;
        const current = drop && inGap ? frame.observations.filter(o => jointGroup(o.id) !== 'hands') : frame.observations;
        const grouped = new Map();
        for (const observation of current) {
          if (!grouped.has(observation.id)) grouped.set(observation.id, []);
          grouped.get(observation.id).push(observation);
        }
        const reconstructed = [...grouped].map(([id, observations]) => ({
          id, ...triangulateJoint(observations, data.cameras, {referenceTimeMs: frame.timeMs}),
        }));
        const tracked = tracker.update({timeMs: frame.timeMs, joints: reconstructed}).joints;
        addFrame(rawAll, frame.groundTruth, reconstructed);
        addFrame(trackedAll, frame.groundTruth, tracked);
        if (inGap) {
          addFrame(rawGap, frame.groundTruth, reconstructed);
          addFrame(trackedGap, frame.groundTruth, tracked);
        }
        const perFrame = newMetrics();
        addFrame(perFrame, frame.groundTruth, tracked);
        frameMetrics.push({index: frame.index, timeMs: frame.timeMs, artificialHandsDropped: drop && inGap,
          tracking: finish(perFrame)});
      }
      result.scenarios[drop ? 'both_hands_observation_drop' : 'unmodified_observations'] = {
        triangulated: finish(rawAll), causalTracker: finish(trackedAll),
        middleIntervalTriangulated: finish(rawGap), middleIntervalTracker: finish(trackedGap), frameMetrics,
      };
    }
    measured.conditions.push(result);
  }
  await mkdir(dirname(outputPath), {recursive: true});
  await writeFile(outputPath, `${JSON.stringify(measured, null, 2)}\n`, 'utf8');
  console.log(JSON.stringify({kind: measured.kind, sequence: measured.sequence,
    conditions: measured.conditions.map(({scenarios, ...rest}) => ({...rest,
      scenarios: Object.fromEntries(Object.entries(scenarios).map(([name, {frameMetrics, ...metrics}]) => [name, metrics])),
    })),
  }, null, 2));
}
