/** Actions-only learned XYZ inference and real VRM replay.
 * Frames/landmarks remain in memory; the report contains aggregate numbers.
 * Coverage, loss duration and retained movement are mandatory companions to
 * jitter metrics. Model-relative consistency is never called GT accuracy. */
import { readFile, writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import { resolve, dirname, join } from 'node:path';
import { tmpdir, cpus } from 'node:os';
import { createRequire } from 'node:module';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';
import { PUBLIC_VIDEO, PUBLIC_VRM, REFERENCE_133, modelBytes, publicFixture, probeVideo, decodedFrames, rgbCropTensor } from './headless-video.mjs';
import { COCO59, cropTransform, decodeSimCC, toCoco133 } from '../dist/compact-codec.mjs';
import { createPose3DCrop, samplePose3DRgb, decodePose3D, mergePose3DKeypoints, fitPerspectiveRoot } from '../dist/pose3d.mjs';
import { estimateMonocularPose, estimateLearnedMonocularPose } from '../dist/geometry.mjs';
import { PoseStabilizer } from '../dist/pose-stability.mjs';
import { retargetPose, rotateVector, quaternionAngle } from '../dist/retarget.mjs';

const finite = Number.isFinite;
const BODY = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
const LIMBS = { leftUpperArm: [5, 7, 'leftLowerArm'], leftLowerArm: [7, 9, 'leftHand'], rightUpperArm: [6, 8, 'rightLowerArm'], rightLowerArm: [8, 10, 'rightHand'], leftUpperLeg: [11, 13, 'leftLowerLeg'], leftLowerLeg: [13, 15, 'leftFoot'], rightUpperLeg: [12, 14, 'rightLowerLeg'], rightLowerLeg: [14, 16, 'rightFoot'] };
const vector = p => Array.isArray(p) && p.length === 3 && p.every(finite);
const subtract = (a, b) => a.map((v, i) => v - b[i]);
const unit = v => { const n = Math.hypot(...v); return n > 1e-8 ? v.map(x => x / n) : null; };
const directionAngle = (a, b) => a && b ? Math.acos(Math.min(1, Math.max(-1, a.reduce((s, v, i) => s + v * b[i], 0)))) * 180 / Math.PI : null;
const inverse = q => [-q[0], -q[1], -q[2], q[3]];
const max = values => values.length ? Math.max(...values) : null;
const percentile = (values, p) => { const sorted = values.filter(finite).sort((a, b) => a - b); return sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))] : null; };
const stats = values => ({ count: values.length, mean: values.length ? values.reduce((a, b) => a + b, 0) / values.length : null, median: percentile(values, .5), p95: percentile(values, .95), max: max(values) });
const times = values => { const s = stats(values); return { count: s.count, meanMs: s.mean, medianMs: s.median, p95Ms: s.p95, maxMs: s.max }; };
const seen = p => p && p.observable !== false && p.score >= .25 && finite(p.x) && finite(p.y);
const fail = code => { throw Object.assign(new Error(code), { safeCode: code }); };

function parseArgs(argv) {
  const options = { manifest: 'dist/models.json', output: 'dist/evidence/headless-pose3d.json', fps: 15, maxSeconds: 10, maxSize: 640, minFrames: 72, threads: 2 };
  const names = { '--manifest': 'manifest', '--output': 'output', '--video': 'video', '--vrm': 'vrm', '--max-seconds': 'maxSeconds', '--threads': 'threads' };
  for (let i = 0; i < argv.length; i += 2) { if (!names[argv[i]] || !argv[i + 1]) fail('invalidPose3dArguments'); const key = names[argv[i]]; options[key] = ['maxSeconds', 'threads'].includes(key) ? Number(argv[i + 1]) : argv[i + 1]; }
  if (!(options.maxSeconds > 0 && options.maxSeconds <= 180 && options.threads >= 1 && options.threads <= 8)) fail('invalidPose3dLimits');
  return options;
}

function createReplay(rig, kind) {
  const filter = new PoseStabilizer(), rawCounts = [], acceptedCounts = [], drivenCounts = [], activeFrameAngles = [], targetErrors = [], depthSpreads = [];
  const boneStats = Object.fromEntries(Object.keys(LIMBS).map(name => [name, { targetFrames: 0, drivenFrames: 0, targetDirectionRangeDeg: 0, renderedDirectionRangeDeg: 0, firstTarget: null, firstRendered: null }]));
  let previous = null, pose = null, rawPose = null, longestLost = 0, lostRun = 0, lostFrames = 0, renderFrames = 0, rootValidFrames = 0;
  const roots = [[], [], []], jointRanges = Array.from({ length: 3 }, () => ({ min: Infinity, max: -Infinity }));
  let finiteTransforms = true, atLeastFour = 0, atLeastSix = 0, allEight = 0;
  function observe(keypoints, decoded, timestampMs, imageSize, sourceId) {
    const observation = { sourceId, inputMode: 'wholebody', targetVersion: kind, timestampMs, imageSize, keypoints, learned3D: decoded ? { confidence: decoded.confidence, rootFit: decoded.rootFit } : null };
    rawCounts.push(BODY.filter(id => seen(keypoints[id])).length);
    const options = { bodyHeightM: 1.7, scaleConfirmed: false, minScore: .25, priorPose: filter.priorPose };
    rawPose = kind === 'pose3d' ? estimateLearnedMonocularPose(observation, null, options) : estimateMonocularPose(observation, null, options);
    const filtered = filter.filterObservation(observation, { minScore: .25 });
    const estimated = kind === 'pose3d' ? estimateLearnedMonocularPose(filtered, null, options) : estimateMonocularPose(filtered, null, options);
    pose = filter.update(estimated, { observation: filtered });
    acceptedCounts.push(BODY.filter(id => vector(pose.joints[id]?.position)).length);
    if (pose.previewRoot?.valid) { rootValidFrames++; pose.previewRoot.position.forEach((v, i) => roots[i].push(v)); }
    const observed = BODY.map(id => pose.joints[id]).filter(j => vector(j?.position));
    if (observed.length >= 4) depthSpreads.push(Math.max(...observed.map(j => j.position[2])) - Math.min(...observed.map(j => j.position[2])));
    for (const joint of observed) joint.position.forEach((v, i) => { jointRanges[i].min = Math.min(jointRanges[i].min, v); jointRanges[i].max = Math.max(jointRanges[i].max, v); });
  }
  function render(timestampMs) {
    if (!pose) return null;
    const result = retargetPose({ ...pose, timestampMs }, rig, { previous });
    const count = Object.keys(LIMBS).filter(name => result.bones[name]?.observed).length;
    drivenCounts.push(count); renderFrames++;
    if (count >= 4) atLeastFour++; if (count >= 6) atLeastSix++; if (count === 8) allEight++;
    if (count === 0) { lostRun++; lostFrames++; longestLost = Math.max(longestLost, lostRun); } else lostRun = 0;
    for (const [name, [a, b, childName]] of Object.entries(LIMBS)) {
      const bone = result.bones[name], state = boneStats[name];
      finiteTransforms &&= !!bone && Array.isArray(bone.rotation) && bone.rotation.every(finite) && Array.isArray(bone.worldRotation) && bone.worldRotation.every(finite);
      const first = rawPose.joints[a], second = rawPose.joints[b];
      const target = vector(first?.position) && vector(second?.position) ? unit(subtract(second.position, first.position)) : null;
      if (target) { state.targetFrames++; state.firstTarget ||= target; state.targetDirectionRangeDeg = Math.max(state.targetDirectionRangeDeg, directionAngle(state.firstTarget, target)); }
      if (!bone?.observed || !rig.bones[name] || !rig.bones[childName]) continue;
      state.drivenFrames++;
      const restDirection = unit(subtract(rig.bones[childName].position, rig.bones[name].position));
      const rendered = restDirection ? unit(rotateVector(bone.worldRotation, rotateVector(inverse(rig.bones[name].rotation), restDirection))) : null;
      if (rendered) { state.firstRendered ||= rendered; state.renderedDirectionRangeDeg = Math.max(state.renderedDirectionRangeDeg, directionAngle(state.firstRendered, rendered)); }
      const error = directionAngle(target, rendered); if (error !== null) targetErrors.push(error);
      if (previous?.bones?.[name]?.observed) activeFrameAngles.push(quaternionAngle(previous.bones[name].rotation, bone.rotation) * 180 / Math.PI);
    }
    previous = result; return result;
  }
  function finish() {
    const boneDetails = Object.fromEntries(Object.entries(boneStats).map(([name, state]) => { const { firstTarget, firstRendered, ...rest } = state; return [name, { ...rest, trackingCoverageOfTargetFrames: state.targetFrames ? state.drivenFrames / state.targetFrames : null }]; }));
    const moving = Object.keys(boneDetails).filter(name => boneDetails[name].targetFrames >= 12 && boneDetails[name].targetDirectionRangeDeg >= 12);
    const retained = moving.filter(name => boneDetails[name].renderedDirectionRangeDeg >= 4 && boneDetails[name].trackingCoverageOfTargetFrames >= .3);
    const rootRanges = roots.map(values => values.length ? Math.max(...values) - Math.min(...values) : null);
    return { observationFrames: rawCounts.length, renderFrames,
      rawVisibleBody13: stats(rawCounts), acceptedBody13: stats(acceptedCounts), drivenLimbBonesOf8: stats(drivenCounts),
      tracking: { framesWithAtLeast4Limbs: atLeastFour, framesWithAtLeast6Limbs: atLeastSix, framesWithAll8Limbs: allEight,
        fractionWithAtLeast4Limbs: renderFrames ? atLeastFour / renderFrames : 0, fractionWithAtLeast6Limbs: renderFrames ? atLeastSix / renderFrames : 0, fractionWithAll8Limbs: renderFrames ? allEight / renderFrames : 0,
        completelyLostRenderFrames: lostFrames, completelyLostFraction: renderFrames ? lostFrames / renderFrames : 1, longestCompletelyLostSeconds: longestLost / 30 },
      activeBoneFrameAngleDeg: stats(activeFrameAngles), renderedVsModelTargetDirectionErrorDeg: stats(targetErrors),
      movement: { rawTargetCriterion: 'At least 12 target frames and at least 12 degrees of raw model limb-direction variation', retainedCriterion: 'Rendered direction varies at least 4 degrees and tracks at least 30% of target frames', movingInputBones: moving, retainedInputBones: retained,
        allMovingInputBonesRetained: moving.length > 0 && retained.length === moving.length, anyMovingInputBoneRetained: retained.length > 0, bones: boneDetails, isGroundTruthAccuracy: false },
      depth: { learned3D: kind === 'pose3d', jointZSpread: stats(depthSpreads), modelJointAxisRanges: jointRanges.map(v => finite(v.min) && finite(v.max) ? v.max - v.min : null),
        validRootFrames: rootValidFrames, validRootFraction: rawCounts.length ? rootValidFrames / rawCounts.length : 0, estimatedRootAxisRanges: rootRanges, absoluteDistanceGroundTruth: false },
      finiteTransforms };
  }
  return { observe, render, finish };
}

async function runModel(model, rgb, imageSize, ort, anchors, benchmark = true) {
  const start = performance.now();
  const crop = model.role === 'pose3d' ? createPose3DCrop([0, 0, imageSize.width, imageSize.height], anchors || [], imageSize) : cropTransform([0, 0, imageSize.width, imageSize.height], [model.width, model.height], model.spec.input.crop?.padding || 1.25);
  const data = model.role === 'pose3d' ? samplePose3DRgb(rgb, imageSize.width, imageSize.height, crop) : rgbCropTensor(rgb, imageSize.width, imageSize.height, crop, model.spec.input.mean, model.spec.input.std);
  const input = new ort.Tensor('float32', data, model.role === 'pose3d' ? [1, 256, 256, 3] : [1, 3, model.height, model.width]);
  const preprocessingMs = performance.now() - start;
  let outputs;
  try {
    if (!model.warmed) { for (let n = 0; n < 3; n++) { const warm = await model.session.run({ [model.session.inputNames[0]]: input }); Object.values(warm).forEach(t => t.dispose?.()); } model.warmed = true; }
    const inferStart = performance.now(); outputs = await model.session.run({ [model.session.inputNames[0]]: input });
    const inferenceMs = performance.now() - inferStart;
    if (!Object.values(outputs).every(t => t.data.every(finite))) fail('nonFinitePose3dInference');
    if (benchmark) { model.inference.push(inferenceMs); model.preprocessing.push(preprocessingMs); }
    const postStart = performance.now();
    let decoded = null, keypoints;
    if (model.role === 'pose3d') { decoded = decodePose3D(outputs, crop); keypoints = decoded.keypoints; }
    else {
      const split = model.spec.simcc_split_ratio || 2, decodeWH = model.spec.decodeWH || [model.width, model.height];
      const values = Object.values(outputs), x = values.find(t => t.data.length === model.count * decodeWH[0] * split), y = values.find(t => t.data.length === model.count * decodeWH[1] * split);
      if (!x || !y) fail('comparisonSimccContractInvalid');
      const points = decodeSimCC(x.data, y.data, crop, model.count, split, decodeWH);
      keypoints = model.count === 133 ? points : toCoco133(points, model.spec.coco133_indices || COCO59);
    }
    const postprocessingMs = performance.now() - postStart;
    if (benchmark) model.postprocessing.push(postprocessingMs);
    return { keypoints, decoded, computeMs: preprocessingMs + inferenceMs + postprocessingMs };
  } finally { input.dispose?.(); if (outputs) Object.values(outputs).forEach(t => t.dispose?.()); }
}

function rootMechanics(rig) {
  const xyz = { 0: [0, -.7, -.08], 5: [-.2, -.5, 0], 6: [.2, -.5, 0], 7: [-.25, -.2, -.1], 8: [.25, -.2, -.1], 9: [-.25, .05, -.12], 10: [.25, .05, -.12], 11: [-.1, 0, 0], 12: [.1, 0, 0], 13: [-.1, .4, -.05], 14: [.1, .4, -.05], 15: [-.1, .8, 0], 16: [.1, .8, 0] };
  const K = [600, 600, 320, 240], imageSize = { width: 640, height: 480 }, fitErrors = [], rendered = [];
  let previous = null;
  for (let i = 0; i <= 30; i++) {
    const distance = 2.5 + i * .05, timestampMs = i * 1000 / 30;
    const keypoints = Array.from({ length: 133 }, (_, id) => {
      const p = xyz[id]; return p ? { id, x: K[0] * p[0] / (p[2] + distance) + K[2], y: K[1] * p[1] / (p[2] + distance) + K[3], score: 1, observable: true, modelPosition: p, relativeZm: p[2] } : { id, x: 0, y: 0, score: 0, observable: false };
    });
    const fit = fitPerspectiveRoot({ valid: true, keypoints, imageSize }, { K });
    if (!fit.valid) fail('perspectiveRootFixtureRejected');
    fitErrors.push(Math.abs(fit.positionCamera[2] - distance));
    const observation = { sourceId: 'explicit-mathematical-fixture', inputMode: 'wholebody', targetVersion: 'pose3d', timestampMs, keypoints, imageSize, learned3D: { confidence: 1, rootFit: fit } };
    const pose = estimateLearnedMonocularPose(observation, null, { minScore: .25 });
    previous = retargetPose(pose, rig, { previous }); rendered.push(previous.root.visualPosition[2]);
  }
  const displacement = rendered.at(-1) - rendered[0];
  return { kind: 'explicit-mathematical-perspective-fixture', neuralInference: false, realVideoDepthAccuracy: false,
    cameraDistanceStart: 2.5, cameraDistanceEnd: 4, maxDistanceFitAbsoluteError: max(fitErrors), renderedZDisplacement: displacement,
    noRecentering: displacement < -1.3, backwardMovesAway: displacement < 0, passed: max(fitErrors) < 1e-6 && displacement < -1.3 };
}

async function main(options) {
  if (process.env.GITHUB_ACTIONS !== 'true') fail('githubActionsRequired');
  const require = createRequire(resolve(process.env.VRC_NODE_PACKAGE || '.cache/headless/package.json')), ort = require('onnxruntime-node'); ort.env.logLevel = 'error';
  const workspace = await mkdtemp(join(tmpdir(), 'vrc-pose3d-')), models = [];
  const originalWarn = console.warn; console.warn = () => {};
  try {
    const videoPath = options.video || await publicFixture(PUBLIC_VIDEO, join(workspace, 'input.mp4'));
    const vrmPath = options.vrm || await publicFixture(PUBLIC_VRM, join(workspace, 'avatar.vrm'));
    const video = await probeVideo(videoPath, options), manifest = JSON.parse(await readFile(options.manifest, 'utf8'));
    if (!manifest.pose3d || !manifest.bodies?.standard) fail('publishedPose3dOrStandardMissing');
    for (const [role, spec, count] of [['pose3d', manifest.pose3d, 33], ['compact59', manifest.bodies.standard, 59], ['reference133', REFERENCE_133, 133]]) {
      const session = await ort.InferenceSession.create(await modelBytes(spec, options.manifest), { executionProviders: ['cpu'], intraOpNumThreads: options.threads, interOpNumThreads: 1, executionMode: 'sequential', graphOptimizationLevel: 'all' });
      models.push({ role, spec, session, count, width: role === 'pose3d' ? 256 : spec.input.shape[3], height: role === 'pose3d' ? 256 : spec.input.shape[2], inference: [], preprocessing: [], postprocessing: [] });
    }
    globalThis.self ||= globalThis;
    globalThis.ProgressEvent ||= class ProgressEvent extends Event { constructor(type, value = {}) { super(type); Object.assign(this, value); } };
    const [{ THREE }, { loadEmbeddedVRM, captureRig }] = await Promise.all([import('../dist/vendor/vrm-runtime.mjs'), import('../dist/vrm-view.mjs')]);
    const bytes = await readFile(vrmPath), vrm = await loadEmbeddedVRM(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), { headless: true });
    const group = new THREE.Group(); group.add(vrm.scene); group.updateMatrixWorld(true); const rig = captureRig(vrm, group);
    if (Object.keys(rig.bones).length < 15) fail('actualVrmRigMissing');
    const replays = Object.fromEntries(models.map(model => [model.role, createReplay(rig, model.role)]));
    let count = 0, anchors = null, detailPoints = null, actualVrmUpdates = 0, springUpdates = 0, transformsFinite = true;
    const vrmTimes = [], productionCompute = [];
    if (typeof vrm.springBoneManager?.update === 'function') { const update = vrm.springBoneManager.update.bind(vrm.springBoneManager); vrm.springBoneManager.update = dt => { springUpdates++; return update(dt); }; }
    for await (const rgb of decodedFrames(videoPath, video, options)) {
      const output = {}, order = [...models.filter(model => model.role !== 'pose3d'), models.find(model => model.role === 'pose3d')];
      // Match the browser: the first fresh standard59 detail inference seeds
      // the 3D crop. Later frames use the previous raw 3D observation, including
      // its missing points; a loss falls back to the supplied person ROI.
      for (const model of order) output[model.role] = await runModel(model, rgb, video, ort, anchors || output.compact59?.keypoints);
      anchors = output.pose3d.keypoints;
      if (count % 3 === 0) detailPoints = output.compact59.keypoints.map(p => ({ ...p, observedAtMs: count * 1000 / options.fps }));
      const combined = mergePose3DKeypoints(detailPoints, output.pose3d.decoded), timestampMs = count * 1000 / options.fps;
      for (const model of models) replays[model.role].observe(model.role === 'pose3d' ? combined : output[model.role].keypoints, output[model.role].decoded, timestampMs, video, 'actual-' + model.role);
      let vrmCompute = 0;
      for (let half = 0; half < 2; half++) for (const model of models) {
        const result = replays[model.role].render(timestampMs + half * 1000 / 30);
        if (model.role !== 'pose3d') continue;
        const start = performance.now(); vrm.humanoid.setNormalizedPose(result.normalizedPose); vrm.update(1 / 30); vrm.scene.updateMatrixWorld(true);
        const elapsed = performance.now() - start; vrmTimes.push(elapsed); vrmCompute += elapsed; actualVrmUpdates++;
        vrm.scene.traverse(node => { transformsFinite &&= node.matrixWorld.elements.every(finite); });
      }
      productionCompute.push(output.pose3d.computeMs + (count % 3 === 0 ? output.compact59.computeMs : 0) + vrmCompute);
      count++;
    }
    const tracking = Object.fromEntries(models.map(model => [model.role, replays[model.role].finish()]));
    const partialCases = [{ name: 'upper60Percent', top: 0 }, { name: 'lower60Percent', top: video.height - Math.floor(video.height * .6) }];
    const partial = {};
    for (const item of partialCases) {
      const height = Math.floor(video.height * .6), size = { width: video.width, height }, partialReplays = Object.fromEntries(models.map(model => [model.role, createReplay(rig, model.role)]));
      let index = 0, prior = null; const elapsed = [];
      for await (const rgb of decodedFrames(videoPath, video, options)) {
        const cropped = rgb.subarray(item.top * video.width * 3, (item.top + height) * video.width * 3);
        const order = [...models.filter(model => model.role !== 'pose3d'), models.find(model => model.role === 'pose3d')];
        let initialDetails = null;
        for (const model of order) {
          const output = await runModel(model, cropped, size, ort, prior || initialDetails, false);
          if (model.role === 'compact59') initialDetails = output.keypoints;
          if (model.role === 'pose3d') { prior = output.keypoints; elapsed.push(output.computeMs); }
          partialReplays[model.role].observe(output.keypoints, output.decoded, index * 1000 / options.fps, size, item.name + '-' + model.role);
          partialReplays[model.role].render(index * 1000 / options.fps); partialReplays[model.role].render(index * 1000 / options.fps + 1000 / 30);
        }
        index++;
      }
      partial[item.name] = { actualRgbCropInferencesPerModel: index, sourceFramesRepeated: false, cropFraction: .6,
        naturallyRecordedPartialBodyVideo: false, pose3dCompute: times(elapsed), tracking: Object.fromEntries(models.map(model => [model.role, partialReplays[model.role].finish()])), groundTruthAccuracy: null };
    }
    const mechanics = rootMechanics(rig), current = tracking.pose3d;
    const executionChecks = { actualDecodedFrames: count >= options.minFrames, allThreeModelsRun: models.every(model => model.inference.length === count), actualVrmUpdates: actualVrmUpdates === count * 2, actualSpringUpdates: springUpdates >= actualVrmUpdates,
      finiteActualVrm: transformsFinite, finiteRetarget: Object.values(tracking).every(result => result.finiteTransforms), perspectiveRootMechanics: mechanics.passed };
    const readinessChecks = { acceptedBodyMeanAtLeast7Of13: current.acceptedBody13.mean >= 7,
      atLeast4TrackedLimbsOn60PercentOfRenderFrames: current.tracking.fractionWithAtLeast4Limbs >= .6,
      rawMovingInputRetained: current.movement.allMovingInputBonesRetained,
      learnedDepthNotFlattened: current.depth.jointZSpread.p95 > .05,
      rootFitAvailableOnHalfOfFrames: current.depth.validRootFraction >= .5 };
    const report = { schema: 'vrc-headless-pose3d-v1', status: Object.values(executionChecks).every(Boolean) ? 'completed' : 'failed', sourceCommit: process.env.GITHUB_SHA || null, runId: process.env.GITHUB_RUN_ID || null,
      input: { kind: options.video ? 'privateEphemeralVideo' : 'pinnedOfficialPublicVideo', publicSource: options.video ? null : PUBLIC_VIDEO.url, publicSha256: options.video ? null : PUBLIC_VIDEO.sha256,
        decodedFrames: count, sampledFps: options.fps, sampledSeconds: count / options.fps, frameImagesRetained: false, perFrameCoordinatesRetained: false },
      environment: { runtime: 'onnxruntime-node', version: ort.env.versions?.node || '1.22.0', node: process.version, executionProvider: 'CPUExecutionProvider', threads: options.threads, logicalCpus: cpus().length, cpu: cpus()[0]?.model },
      executionChecks, models: Object.fromEntries(models.map(model => [model.role, { id: model.spec.id, sha256: model.spec.sha256, bytes: model.spec.bytes || null, inference: times(model.inference), preprocessing: times(model.preprocessing), postprocessing: times(model.postprocessing), warmupCallsExcluded: 3 }])),
      tracking, partialBodyRgbCrops: partial,
      productionCompute: { ...times(productionCompute), definition: 'Sum of measured 15 Hz learned body XYZ preprocessing/inference/output decoding, every third 15 Hz frame uses standard59 detail landmarks (5 Hz), and two actual VRM updates per video frame (30 Hz). Additional reference/comparison/cropped-case calls excluded. Video decode, retarget/filter math, rendering and face/foot auxiliary inference excluded; not end-to-end latency.' },
      vrm: { humanoidBones: Object.keys(rig.bones).length, updateCalls: actualVrmUpdates, springUpdateCalls: springUpdates, updates: times(vrmTimes), finiteTransforms: transformsFinite, rendererUsed: false },
      rootMechanics: mechanics,
      deploymentReadiness: { passed: Object.values(executionChecks).every(Boolean) && Object.values(readinessChecks).every(Boolean), checks: readinessChecks,
        interpretation: 'Execution and observability screen on this fixture only. This is not labelled 3D pose accuracy, not proof for the private video, and not a Stage 4 approval.' },
      accuracy: null, gpuUtilization: null,
      limitations: ['Unlabelled video coverage is not anatomical accuracy.', 'Reference133 agreement is not ground truth.', 'The learned model emits root-relative XYZ; absolute camera distance is a perspective estimate.', 'Input is a supplied person ROI with tracking recrop; no learned person detector is included.', 'Model order is fixed to match initial detail-seeded crop dependencies; this is not a randomized balanced latency comparison.', 'No freeze or lost-frame metric can establish motion quality by itself.'] };
    await mkdir(dirname(resolve(options.output)), { recursive: true }); await writeFile(options.output, JSON.stringify(report, null, 2) + '\n', { mode: 0o600 });
    console.log(JSON.stringify({ status: report.status, decodedFrames: count, observedBody13Mean: current.acceptedBody13.mean, trackingAtLeast4LimbsFraction: current.tracking.fractionWithAtLeast4Limbs, completelyLostFraction: current.tracking.completelyLostFraction, deploymentReadiness: report.deploymentReadiness.passed, mediaOutput: false }));
    if (report.status !== 'completed') process.exitCode = 1;
  } finally { console.warn = originalWarn; for (const model of models) await model.session.release().catch(() => {}); await rm(workspace, { recursive: true, force: true }); }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try { await main(parseArgs(process.argv.slice(2))); }
  catch (error) { console.error(JSON.stringify({ status: 'failed', code: error.safeCode || 'pose3dEvaluationFailed', privatePathsLogged: false, mediaOutput: false })); process.exitCode = 1; }
}
