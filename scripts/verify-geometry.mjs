// A small geometry acceptance gate. Run only in GitHub Actions.
// Synthetic correspondences verify implementation, never real-camera accuracy.
import assert from 'node:assert/strict';
import {mkdir, readFile, writeFile} from 'node:fs/promises';
import {basename, dirname, join} from 'node:path';
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

function selectPairFromWarmup(data) {
  const config = data.pairSelection;
  const source = data.conditions.find(condition => condition.name === config.sourceCondition);
  assert.ok(source, 'The declared pair-selection source condition must exist');
  const count = config.warmupFrameCount;
  assert.ok(Number.isInteger(count) && count >= 2 && count < source.frames.length);
  const jointIds = [...new Set(config.jointIds)];
  assert.ok(jointIds.length > 0 && jointIds.every(id => Number.isInteger(id) && id >= 0 && id < 133));
  const warmup = source.frames.slice(0, count);
  const cameraIds = [...new Set(warmup.flatMap(frame => Object.keys(frame.cameras || data.cameras || {})))].sort();
  assert.ok(cameraIds.length >= 2);
  const denominator = count * jointIds.length;
  const candidates = [];
  // Only camera calibration, times and image predictions enter this selection.
  // Reference coordinates and positional errors are never consulted.
  for (let a = 0; a < cameraIds.length; a++) for (let b = a + 1; b < cameraIds.length; b++) {
    const pair = [cameraIds[a], cameraIds[b]], selectedIds = new Set(pair);
    let confidenceSum = 0, valid = 0, angleSum = 0, reprojectionSum = 0, baselineSum = 0, baselineCount = 0;
    const invalidReasons = {};
    for (const frame of warmup) {
      const cameras = frame.cameras || data.cameras;
      const centres = pair.map(id => cameras?.[id]?.worldFromCamera?.translation);
      if (centres.every(v => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite))) {
        baselineSum += Math.hypot(...centres[0].map((v, axis) => v - centres[1][axis]));
        baselineCount++;
      }
      const grouped = new Map();
      for (const observation of frame.observations) {
        if (!selectedIds.has(String(observation.sourceId))) continue;
        if (!grouped.has(observation.id)) grouped.set(observation.id, []);
        grouped.get(observation.id).push(observation);
      }
      for (const id of jointIds) {
        const result = triangulateJoint(grouped.get(id) || [], cameras, {referenceTimeMs: frame.timeMs});
        if (result.valid) {
          valid++;
          confidenceSum += result.confidence;
          angleSum += result.rayAngleDeg;
          reprojectionSum += result.reprojectionErrorPx;
        } else {
          const reason = result.reason || 'unknown';
          invalidReasons[reason] = (invalidReasons[reason] || 0) + 1;
        }
      }
    }
    candidates.push({cameraIds: pair, score: confidenceSum / denominator,
      jointFrames: denominator, validJointFrames: valid, validFraction: valid / denominator,
      meanBaselineM: baselineCount ? baselineSum / baselineCount : null,
      meanRayAngleDegAvailable: valid ? angleSum / valid : null,
      meanReprojectionErrorPxAvailable: valid ? reprojectionSum / valid : null, invalidReasons});
  }
  candidates.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    const left = a.cameraIds.join('|'), right = b.cameraIds.join('|');
    return left < right ? -1 : left > right ? 1 : 0;
  });
  const selected = candidates[0], selectedIds = new Set(selected.cameraIds);
  const conditionName = config.name || 'selected_two_views_from_warmup';
  assert.ok(!data.conditions.some(condition => condition.name === conditionName));
  const dropStart = source.dropStartIndex ?? data.dropStartIndex;
  if (dropStart != null) assert.ok(dropStart >= count, 'Pair-selection warmup must finish before the evaluated drop');
  const prepareFrame = (frame, index, filterViews) => ({...frame,
    observations: filterViews ? frame.observations.filter(o => selectedIds.has(String(o.sourceId))) : frame.observations,
    groundTruth: index < count ? [] : frame.groundTruth,
  });
  data.conditions.push({...source, name: conditionName, scoringStartIndex: count,
    frames: source.frames.map((frame, index) => prepareFrame(frame, index, true))});
  data.conditions.push({...source, name: `${source.name}_after_warmup`, scoringStartIndex: count,
    frames: source.frames.map((frame, index) => prepareFrame(frame, index, false))});
  return {
    sourceCondition: source.name, warmupFrameCount: count,
    firstWarmupFrame: warmup[0].index, lastWarmupFrame: warmup.at(-1).index,
    firstScoredFrame: source.frames[count].index,
    objective: 'Sum of product triangulation confidence / all declared joint-frames in the warmup.',
    tieBreak: 'Lexicographic camera IDs.', referenceUsedForSelection: false,
    selectedCameraIds: selected.cameraIds, selectedScore: selected.score,
    selectedPairHasSupport: selected.validJointFrames > 0,
    scoring: 'Warmup observations initialize history; their reference points are excluded from the added comparisons.',
    candidates,
  };
}

async function evaluateSequence(inputPath, outputPath) {
  const data = JSON.parse(await readFile(inputPath, 'utf8'));
  assert.equal(data.schemaVersion, 1);
  assert.ok(Array.isArray(data.conditions) && data.conditions.length > 0);
  assert.ok(!data.coordinateUnit || ['m', 'metre', 'meter'].includes(data.coordinateUnit),
    'Evaluation coordinates must have a documented fixed conversion to metres');
  const direct3d = data.inputMode === '3d-points';
  assert.ok(!(direct3d && data.pairSelection), 'Camera-pair selection requires image observations');
  const pairSelection = data.pairSelection ? selectPairFromWarmup(data) : null;
  const dataset = data.dataset || 'CMU Panoptic';
  const metricGroups = data.metricGroups || {
    body: Array.from({length: 17}, (_, i) => i), hands: Array.from({length: 42}, (_, i) => i + 91),
  };
  const memberships = new Map();
  for (const [name, ids] of Object.entries(metricGroups)) {
    assert.ok(name && Array.isArray(ids) && ids.length > 0);
    for (const id of new Set(ids)) {
      assert.ok(Number.isInteger(id) && id >= 0 && id < 133);
      memberships.set(id, [...(memberships.get(id) || []), name]);
    }
  }
  const dropIds = new Set(data.dropJointIds || Array.from({length: 42}, (_, i) => i + 91));
  assert.ok(dropIds.size > 0 && [...dropIds].every(id => Number.isInteger(id) && id >= 0 && id < 133));
  const dropScenarioName = data.dropScenarioName || 'both_hands_observation_drop';
  const cameraIds = new Set(Object.keys(data.cameras || {}));
  for (const condition of data.conditions) for (const frame of condition.frames || []) {
    for (const id of Object.keys(frame.cameras || {})) cameraIds.add(id);
  }
  if (!direct3d) assert.ok(cameraIds.size >= 2, 'Multiview evaluation needs calibrated cameras');
  const measured = {
    schemaVersion: 1,
    kind: direct3d ? 'real-motion-occlusion-completion-evaluation' : 'real-multiview-pose-evaluation',
    dataset, referenceKind: data.referenceKind || 'provided-3d-annotation',
    sequence: data.sequence, fps: data.fps, coordinateUnit: 'm', errorUnit: 'mm',
    sourceCommit: process.env.GITHUB_SHA || null,
    methods: {triangulation: direct3d ? null : 'dist/geometry.js triangulateJoint',
      input: direct3d ? 'Published aligned 3D series; no image inference is run by this evaluator.' : 'Real-image model observations.',
      tracking: 'dist/tracking.js OcclusionTracker',
      holdLast: 'Causal last accepted input coordinate; expires at the product free-prediction age for each joint group.'},
    cameraIds: [...cameraIds], metricGroups, provenance: data.provenance || null, pairSelection,
    scope: data.scope || (direct3d
      ? `${dataset} recorded 3D observations compared with its supplied reference. Only the product causal tracker is exercised.`
      : `${dataset} real-image observations and supplied calibrated 3D reference. Reference kind is reported separately.`),
    limitations: [
      'Reference coordinates are consumed for scoring and are not included in this report.',
      'Artificial observation-drop removes observations after inference or reconstruction; it is not an image-occlusion benchmark.',
      'MPJPE is conditional on an available estimate. Coverage and PCK use all valid GT joints.',
      'This sequence does not establish performance on user cameras or every type of occlusion.',
      ...(direct3d ? [
        'The upstream recorded series may use whole-sequence alignment, interpolation and filtering; this is not camera-to-output causal latency or accuracy.',
        'Input confidence is an availability indicator supplied by the adapter, not a calibrated visibility probability.',
      ] : []),
      ...(data.limitations || []),
    ],
    conditions: [],
  };

  const finite3 = v => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);
  const quantile = (values, p) => {
    if (!values.length) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const at = (sorted.length - 1) * p, lo = Math.floor(at), hi = Math.ceil(at);
    return sorted[lo] * (hi - at || 1) + (hi === lo ? 0 : sorted[hi] * (at - lo));
  };
  const newGroup = () => ({groundTruthJoints: 0, availableJoints: 0, errorsMm: [], sources: {}});
  const newMetrics = () => Object.fromEntries(Object.keys(metricGroups).map(name => [name, newGroup()]));
  const addFrame = (metrics, gt, predictions) => {
    const byId = new Map(predictions.map(j => [j.id, j]));
    for (const point of gt) {
      if (point.valid === false || !finite3(point.xyz)) continue;
      for (const groupName of memberships.get(point.id) || []) {
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
    const dropCount = condition.dropFrameCount ?? data.dropFrameCount ?? Math.min(8, Math.max(1, Math.floor(condition.frames.length / 4)));
    const dropStart = condition.dropStartIndex ?? data.dropStartIndex ?? Math.floor((condition.frames.length - dropCount) / 2);
    assert.ok(Number.isInteger(dropCount) && dropCount >= 1 && dropCount < condition.frames.length);
    assert.ok(Number.isInteger(dropStart) && dropStart >= 1 && dropStart + dropCount <= condition.frames.length,
      'The declared drop interval needs prior history and must lie within the sequence');
    const result = {name: condition.name, frameCount: condition.frames.length,
      firstFrame: condition.frames[0].index, lastFrame: condition.frames.at(-1).index,
      cameraIds: direct3d ? [] : [...new Set(condition.frames.flatMap(frame => frame.observations.map(o => String(o.sourceId))))].sort(),
      scoringStartIndex: condition.scoringStartIndex || 0,
      artificialDrop: {firstFrame: condition.frames[dropStart].index,
        lastFrame: condition.frames[dropStart + dropCount - 1].index, frameCount: dropCount,
        removedJointIds: [...dropIds], selection: data.dropSelection || 'Fixed central interval, independent of prediction error.',
        inputStage: direct3d ? 'recorded 3D observations' : 'all camera observations after image inference'}, scenarios: {}};
    for (const drop of [false, true]) {
      const tracker = new OcclusionTracker();
      const rawAll = newMetrics(), trackedAll = newMetrics(), rawGap = newMetrics(), trackedGap = newMetrics();
      const holdAll = newMetrics(), holdGap = newMetrics(), lastSeen = new Map();
      result.holdLastPolicy = {
        maximumBodyAgeMs: tracker.options.maximumBodyPredictionMs,
        maximumHandAgeMs: tracker.options.maximumHandPredictionMs,
        maximumFootAgeMs: tracker.options.maximumFootPredictionMs,
        minimumInputConfidence: tracker.options.minimumConfidence,
        coordinates: 'Last raw accepted observation, without velocity, smoothing, or reference coordinates.',
      };
      const frameMetrics = [];
      for (let i = 0; i < condition.frames.length; i++) {
        const frame = condition.frames[i];
        const inGap = i >= dropStart && i < dropStart + dropCount;
        let reconstructed;
        if (direct3d) {
          assert.ok(Array.isArray(frame.inputJoints));
          const current = drop && inGap ? frame.inputJoints.filter(o => !dropIds.has(o.id)) : frame.inputJoints;
          reconstructed = current.filter(j => j.valid !== false && finite3(j.xyz)).map(j => ({
            id: j.id, xyz: j.xyz, confidence: Number.isFinite(j.confidence) ? j.confidence : 1,
            source: 'model3d', valid: true,
          }));
        } else {
          assert.ok(Array.isArray(frame.observations));
          const current = drop && inGap ? frame.observations.filter(o => !dropIds.has(o.id)) : frame.observations;
          const grouped = new Map();
          for (const observation of current) {
            if (!grouped.has(observation.id)) grouped.set(observation.id, []);
            grouped.get(observation.id).push(observation);
          }
          reconstructed = [...grouped].map(([id, observations]) => ({
            id, ...triangulateJoint(observations, frame.cameras || data.cameras, {referenceTimeMs: frame.timeMs}),
          }));
        }
        const tracked = tracker.update({timeMs: frame.timeMs, joints: reconstructed}).joints;
        for (const joint of reconstructed) {
          if (joint.valid === false || !finite3(joint.xyz) || !Number.isFinite(joint.confidence)
              || joint.confidence < tracker.options.minimumConfidence) continue;
          lastSeen.set(joint.id, {xyz: [...joint.xyz], confidence: joint.confidence,
            timeMs: frame.timeMs, source: joint.source});
        }
        const held = [];
        for (const [id, previous] of lastSeen) {
          const ageMs = frame.timeMs - previous.timeMs;
          const maximumAgeMs = id >= 91 ? tracker.options.maximumHandPredictionMs
            : id >= 17 && id <= 22 ? tracker.options.maximumFootPredictionMs : tracker.options.maximumBodyPredictionMs;
          if (ageMs > maximumAgeMs) continue;
          held.push({id, xyz: previous.xyz, confidence: previous.confidence, valid: true,
            source: ageMs === 0 ? previous.source : 'holdLast', ageMs});
        }
        addFrame(rawAll, frame.groundTruth, reconstructed);
        addFrame(trackedAll, frame.groundTruth, tracked);
        addFrame(holdAll, frame.groundTruth, held);
        if (inGap) {
          addFrame(rawGap, frame.groundTruth, reconstructed);
          addFrame(trackedGap, frame.groundTruth, tracked);
          addFrame(holdGap, frame.groundTruth, held);
        }
        const perFrame = newMetrics(), perFrameHold = newMetrics();
        addFrame(perFrame, frame.groundTruth, tracked);
        addFrame(perFrameHold, frame.groundTruth, held);
        frameMetrics.push({index: frame.index, timeMs: frame.timeMs, artificialObservationsDropped: drop && inGap,
          tracking: finish(perFrame), holdLast: finish(perFrameHold)});
      }
      result.scenarios[drop ? dropScenarioName : 'unmodified_observations'] = {
        [direct3d ? 'recorded3d' : 'triangulated']: finish(rawAll), causalTracker: finish(trackedAll), holdLast: finish(holdAll),
        [direct3d ? 'middleIntervalRecorded3d' : 'middleIntervalTriangulated']: finish(rawGap),
        middleIntervalTracker: finish(trackedGap), middleIntervalHoldLast: finish(holdGap), frameMetrics,
      };
    }
    measured.conditions.push(result);
  }
  await mkdir(dirname(outputPath), {recursive: true});
  await writeFile(outputPath, `${JSON.stringify(measured, null, 2)}\n`, 'utf8');
  console.log(JSON.stringify({kind: measured.kind, sequence: measured.sequence, pairSelection: measured.pairSelection,
    conditions: measured.conditions.map(({scenarios, ...rest}) => ({...rest,
      scenarios: Object.fromEntries(Object.entries(scenarios).map(([name, {frameMetrics, ...metrics}]) => [name, metrics])),
    })),
  }, null, 2));
}

const sequenceInput = option('--sequence-input') || option('--panoptic-input');
const sequenceInputReport = option('--sequence-input-report');
assert.ok(!(sequenceInput && sequenceInputReport), 'Choose one sequence input or an input report');
if (sequenceInput) {
  const outputPath = option('--sequence-out') || option('--panoptic-out') || 'artifacts/panoptic/geometry.json';
  await evaluateSequence(sequenceInput, outputPath);
}
if (sequenceInputReport) {
  const manifest = JSON.parse(await readFile(sequenceInputReport, 'utf8'));
  assert.ok(manifest.ok && Array.isArray(manifest.input_files) && manifest.input_files.length > 0);
  const outputDir = option('--sequence-out-dir') || 'artifacts/freemocap';
  for (const inputPath of manifest.input_files) {
    await evaluateSequence(inputPath, join(outputDir, `${basename(inputPath, '.json')}-metrics.json`));
  }
}
