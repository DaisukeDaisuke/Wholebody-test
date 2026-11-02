/** Actions-only production tracker integration with the public export fixture. */
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
import { TapirTracker } from '../dist/tapir-tracker.mjs';

if (process.env.GITHUB_ACTIONS !== 'true') throw new Error('Run the shipped-tracker integration in GitHub Actions only.');
const argument = (name, fallback) => { const index = process.argv.indexOf(name); return index < 0 ? fallback : process.argv[index + 1]; };
const modelsRoot = resolve(argument('--models', '.cache/tapir-ci/models'));
const fixtureRoot = resolve(argument('--fixtures', '.cache/tapir-ci/results/codec-fixture'));
const output = resolve(argument('--output', 'evaluation/results/phase3-tapir-wasm.json'));
const optimizationPolicy = argument('--optimization', 'mixed');
if (!['mixed', 'disabled', 'basic', 'extended', 'all'].includes(optimizationPolicy)) throw new Error('Invalid graph optimization option.');
const graphOptimizationLevels = optimizationPolicy === 'mixed'
  ? { encoder: 'all', query: 'all', step: 'disabled' }
  : Object.fromEntries(['encoder', 'query', 'step'].map(id => [id, optimizationPolicy]));
const vendor = new URL('../dist/vendor/', import.meta.url);
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const requireCondition = (condition, message) => { if (!condition) throw new Error(message); };
const report = { schemaVersion: 1, passed: false, status: 'started', stage: 'fixture', gitCommit: process.env.GITHUB_SHA, workflowRunId: process.env.GITHUB_RUN_ID,
  purpose: 'Public still-image recurrent inference through the shipped tracker and ONNX Runtime Web WASM; conversion and integration, not real-video or foot accuracy.',
  privateMediaUsed: false, realVideoAccuracyMeasured: false, webgpuMeasured: false, steamVrHardwareMeasured: false, backend: 'wasm', threads: 1, optimizationPolicy, graphOptimizationLevels,
  models: [], stages: [], comparisons: [], checks: [] };
const stage = name => { report.stage = name; report.stages.push({ name, elapsedMs: performance.now() }); console.log('TAPIR_WASM_STAGE ' + name); };
let tracker = null, other = null, networkAttempts = 0, releasedSessions = 0, caught = null;
const originalFetch = globalThis.fetch;
try {
  const manifest = JSON.parse(await readFile(join(modelsRoot, 'tapir-models.json'), 'utf8'));
  const fixture = JSON.parse(await readFile(join(fixtureRoot, 'fixture.json'), 'utf8'));
  requireCondition(manifest.parityPassed === true && manifest.codecParityPassed === true && manifest.repository === 'RabbitDaisuke/Foot-Point-Tracker-onnx', 'The downloaded artifact must have passed official and codec parity.');
  requireCondition(fixture.attribution?.id === 785 && fixture.attribution?.license === 4 && fixture.attribution?.privateMedia === false && fixture.imageFile === 'public-rgb.u8', 'Only the frozen public CC BY fixture is accepted.');
  requireCondition(fixture.repeatedFrameReference?.frames?.length === 2, 'The official recurrent comparison reference is missing.');
  const rgb = new Uint8Array(await readFile(join(fixtureRoot, fixture.imageFile)));
  requireCondition(rgb.length === fixture.width * fixture.height * 3, 'The public RGB fixture has an invalid shape.');
  const rgba = new Uint8Array(fixture.width * fixture.height * 4);
  for (let i = 0; i < fixture.width * fixture.height; i++) { rgba.set(rgb.subarray(i * 3, i * 3 + 3), i * 4); rgba[i * 4 + 3] = 255; }
  const image = { width: fixture.width, height: fixture.height, data: rgba }, modelBuffers = {};
  for (const id of ['encoder', 'query', 'step']) {
    const model = manifest.models.find(value => value.id === id);
    requireCondition(model?.filename === `causal-bootstapir-${id}-256-q3-fp32.onnx`, 'Unexpected public model artifact.');
    const bytes = new Uint8Array(await readFile(join(modelsRoot, model.filename)));
    requireCondition(bytes.byteLength === model.bytes && hash(bytes) === model.sha256, `${id} artifact hash differs from the manifest.`);
    modelBuffers[id] = bytes;
    report.models.push({ id, filename: model.filename, bytes: bytes.byteLength, sha256: hash(bytes), inputs: model.inputs, outputs: model.outputs });
  }
  report.source = { imageId: 785, license: 'CC-BY-2.0', exportGitCommit: manifest.gitCommit, exportWorkflowRunId: manifest.workflowRunId,
    source: fixture.attribution.retrievedFrom, sourceImageSha256: fixture.attribution.sha256, frameKind: 'Identical public RGB bytes at three source timestamps.' };
  stage('runtime-import');
  const ort = await import(new URL('ort.webgpu.min.mjs', vendor).href);
  ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false; ort.env.wasm.wasmPaths = vendor.href;
  ort.env.wasm.wasmBinary = new Uint8Array(await readFile(new URL('ort-wasm-simd-threaded.jsep.wasm', vendor)));
  ort.env.logLevel = 'warning';
  globalThis.fetch = async () => { networkAttempts++; throw new Error('Network requests are disabled during the public tracker integration.'); };
  let createIndex = 0;
  const tracedOrt = { Tensor: ort.Tensor, InferenceSession: { create: async (bytes, options) => {
    const id = ['encoder', 'query', 'step'][createIndex++]; stage(`create-${id}`); const start = performance.now();
    requireCondition(options.graphOptimizationLevel === graphOptimizationLevels[id], `The shipped adapter did not apply the declared optimization policy to ${id}.`);
    const session = await ort.InferenceSession.create(bytes, options);
    Object.assign(report.models.find(value => value.id === id), { sessionCreateMs: performance.now() - start, graphOptimizationLevel: options.graphOptimizationLevel });
    return { inputNames: session.inputNames, outputNames: session.outputNames,
      run: async (...args) => { stage(`run-${id}`); return session.run(...args); },
      release: async () => { await session.release(); releasedSessions++; } };
  } };
  tracker = await TapirTracker.create({ ort: tracedOrt, modelBuffers, manifest, executionProviders: ['wasm'], sessionOptions: { logSeverityLevel: 2 },
    graphSessionOptions: Object.fromEntries(Object.entries(graphOptimizationLevels).map(([id, graphOptimizationLevel]) => [id, { graphOptimizationLevel }])) });
  report.checks.push({ name: 'manifest-hashes-and-three-real-sessions', passed: true, stages: createIndex });
  stage('seed');
  const seed = await tracker.seed({ image, points: fixture.pointsXY, side: 'left', timestampMs: 0 });
  requireCondition(seed.diagnostics?.modelRuns === 3 && seed.diagnostics?.seeded === true && tracker.requiresSeed === false, 'Seed did not execute encoder, query and causal step.');
  requireCondition(seed.keypoints.filter(point => point.observable).length === 3 && seed.depthKind === 'none', 'Point seeding invented depth or unrelated body landmarks.');
  report.checks.push({ name: 'production-seed-and-no-depth', passed: true, modelRuns: seed.diagnostics.modelRuns, inferenceMs: seed.diagnostics.inferenceMs });
  function compare(result, reference) {
    const expected = reference.outputs, track = result.footTracks[0], pointIds = [17, 18, 19];
    requireCondition(track?.pointVisibility?.length === 3, 'Tracker lost its point-wise uncertainty output.');
    const differences = { frame: reference.frame, coordinateMaxAbsPx: 0, logitMaxAbs: 0, confidenceMaxAbs: 0, visibilityMismatches: 0, visibleCoordinatesCompared: 0,
      inferenceMs: result.diagnostics.inferenceMs, processingMs: result.diagnostics.processingMs };
    for (let i = 0; i < 3; i++) {
      const x = expected.tracks[i * 2] * image.width / 256, y = expected.tracks[i * 2 + 1] * image.height / 256;
      const confidence = 1 / (1 + Math.exp(expected.occlusion[i])) / (1 + Math.exp(expected.expected_dist[i]));
      const visible = confidence > .5 && x >= 0 && y >= 0 && x < image.width && y < image.height;
      const actual = result.keypoints[pointIds[i]], uncertainty = track.pointVisibility[i];
      differences.logitMaxAbs = Math.max(differences.logitMaxAbs, Math.abs(uncertainty.occlusionLogit - expected.occlusion[i]), Math.abs(uncertainty.expectedDistanceLogit - expected.expected_dist[i]));
      differences.confidenceMaxAbs = Math.max(differences.confidenceMaxAbs, Math.abs(uncertainty.score - confidence));
      if (actual.observable !== visible) differences.visibilityMismatches++;
      if (visible && actual.observable) { differences.coordinateMaxAbsPx = Math.max(differences.coordinateMaxAbsPx, Math.abs(actual.x - x), Math.abs(actual.y - y)); differences.visibleCoordinatesCompared++; }
    }
    requireCondition(differences.visibleCoordinatesCompared > 0 && differences.coordinateMaxAbsPx <= .25 && differences.logitMaxAbs <= .02 && differences.confidenceMaxAbs <= .002 && differences.visibilityMismatches === 0, 'Production WASM recurrent output differs from the official PyTorch sequence: ' + JSON.stringify(differences));
    requireCondition(result.keypoints.every((point, id) => pointIds.includes(id) || point.observable === false), 'Point tracking invented unobserved body or foot-ankle coordinates.');
    report.comparisons.push(differences);
  }
  stage('first-update');
  compare(await tracker.update({ image, timestampMs: fixture.repeatedFrameReference.frames[0].timestampMs }), fixture.repeatedFrameReference.frames[0]);
  // A second stream deliberately shares stateless sessions, but owns its query
  // and recurrent tensors. Interleaving must not corrupt the first stream.
  other = new TapirTracker({ ort: tracedOrt, sessions: tracker.sessions, manifest, ownsSessions: false });
  stage('second-stream-seed');
  await other.seed({ image, points: fixture.pointsXY.map(([x, y]) => [x + 8, y + 5]), side: 'unknown', timestampMs: 0 });
  stage('first-stream-after-interleave');
  compare(await tracker.update({ image, timestampMs: fixture.repeatedFrameReference.frames[1].timestampMs }), fixture.repeatedFrameReference.frames[1]);
  report.checks.push({ name: 'official-recurrence-and-shared-session-state-isolation', passed: true, comparedFrames: report.comparisons.length });
  stage('reset'); tracker.reset();
  const unseeded = await tracker.update({ image, timestampMs: 100 });
  requireCondition(tracker.requiresSeed && unseeded.diagnostics?.modelRan === false && unseeded.keypoints.every(point => point.observable === false), 'Reset preserved an observed tracking claim or ran an unseeded model.');
  report.checks.push({ name: 'reset-removes-point-state', passed: true });
  stage('dispose'); await other.dispose(); other = null; await tracker.dispose(); tracker = null;
  requireCondition(releasedSessions === 3 && networkAttempts === 0, 'The integration did not release all three sessions or attempted network inference.');
  report.checks.push({ name: 'dispose-and-no-inference-network', passed: true, releasedSessions, networkAttempts });
  report.status = 'measured'; report.passed = true; report.stage = 'complete';
} catch (error) {
  caught = error; report.status = 'failed'; report.error = { type: error?.name || typeof error, message: String(error?.message || error), stack: error?.stack || null };
} finally {
  if (other) { try { await other.dispose(); } catch {} }
  if (tracker) { try { await tracker.dispose(); } catch {} }
  globalThis.fetch = originalFetch;
  report.networkAttempts = networkAttempts; report.releasedSessions = releasedSessions;
  await mkdir(resolve(output, '..'), { recursive: true }); await writeFile(output, JSON.stringify(report, null, 2) + '\n');
  const siteOutput = new URL('../dist/evaluation/', import.meta.url);
  await mkdir(siteOutput, { recursive: true }); await writeFile(new URL('phase3-tapir-wasm.json', siteOutput), JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify({ passed: report.passed, stage: report.stage, comparisons: report.comparisons, error: report.error || null }));
if (caught) throw caught;
