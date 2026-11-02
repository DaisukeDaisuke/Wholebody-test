/** Only public fixed Actions inputs are reachable from this isolated Worker. */
import { TapirTracker } from '/dist/tapir-tracker.mjs';
import * as ort from '/vendor/ort.webgpu.min.mjs';

const requireCondition = (condition, message) => { if (!condition) throw new Error(message); };
let started = false;
self.addEventListener('message', async event => {
  if (event.data?.command !== 'run-public-fixture' || started) return;
  started = true;
  const report = { schemaVersion: 1, passed: false, status: 'started', stage: 'public-inputs', dedicatedWorker: true,
    softwareOnly: true, physicalGpuMeasured: false, hardwareFpsMeasured: false, realVideoAccuracyMeasured: false, privateMediaUsed: false,
    purpose: 'One public-image recurrent WebGPU compatibility run on SwiftShader; not graphics-card performance or foot accuracy.',
    graphOptimizationLevels: { encoder: 'all', query: 'all', step: 'disabled' }, models: [], checks: [], locations: [], gpuErrors: [] };
  const stage = name => { report.stage = name; self.postMessage({ kind: 'stage', stage: name }); };
  const actualFetch = globalThis.fetch.bind(globalThis);
  let outsideRequests = 0, tracker = null, released = 0, watchedDevice = null;
  globalThis.fetch = (input, options) => {
    const url = new URL(typeof input === 'string' || input instanceof URL ? input : input.url, self.location.href);
    if (url.origin !== self.location.origin) { outsideRequests++; return Promise.reject(new Error('External network request blocked in the WebGPU fixture.')); }
    return actualFetch(input, options);
  };
  async function read(url, kind) {
    const response = await fetch(url);
    requireCondition(response.ok, 'Public fixture request failed: ' + new URL(url, self.location).pathname);
    return kind === 'json' ? response.json() : new Uint8Array(await response.arrayBuffer());
  }
  try {
    const manifest = await read('/models/tapir-models.json', 'json'), fixture = await read('/fixtures/fixture.json', 'json');
    requireCondition(manifest.parityPassed === true && manifest.codecParityPassed === true && manifest.gitCommit === '274407d44a7fbdcb8411f9dc78286b1f49409c6d', 'The model artifact is not the approved normalized export.');
    requireCondition(fixture.attribution?.id === 785 && fixture.attribution?.license === 4 && fixture.attribution?.privateMedia === false && fixture.imageFile === 'public-rgb.u8', 'Only the reviewed public CC BY image is accepted.');
    const reference = fixture.repeatedFrameReference?.frames?.[0];
    requireCondition(reference?.frame === 1, 'The official recurrent reference is missing.');
    const rgb = await read('/fixtures/public-rgb.u8'), rgba = new Uint8Array(fixture.width * fixture.height * 4);
    requireCondition(rgb.length === fixture.width * fixture.height * 3, 'Invalid public fixture dimensions.');
    for (let i = 0; i < fixture.width * fixture.height; i++) { rgba.set(rgb.subarray(i * 3, i * 3 + 3), i * 4); rgba[i * 4 + 3] = 255; }
    const image = { width: fixture.width, height: fixture.height, data: rgba }, modelBuffers = {};
    for (const id of ['encoder', 'query', 'step']) {
      const model = manifest.models.find(value => value.id === id);
      requireCondition(model?.filename === `causal-bootstapir-${id}-256-q3-fp32.onnx`, 'Unexpected graph filename.');
      modelBuffers[id] = await read('/models/' + model.filename);
      report.models.push({ id, bytes: model.bytes, sha256: model.sha256 });
    }
    report.source = { exportGitCommit: manifest.gitCommit, exportWorkflowRunId: manifest.workflowRunId, imageId: 785, license: 'CC-BY-2.0', sourceImageSha256: fixture.attribution.sha256,
      kind: 'Same public RGB still image at seed and update timestamps.' };
    stage('request-software-webgpu-adapter');
    requireCondition(!!navigator.gpu, 'WebGPU is unavailable in the isolated Actions Worker.');
    const adapter = await navigator.gpu.requestAdapter({ forceFallbackAdapter: true });
    requireCondition(!!adapter, 'Chromium did not expose a software WebGPU adapter.');
    const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    report.adapter = { vendor: info.vendor || '', architecture: info.architecture || '', device: info.device || '', description: info.description || '',
      isFallbackAdapter: adapter.isFallbackAdapter ?? info.isFallbackAdapter ?? null,
      features: Array.from(adapter.features).sort(), limits: Object.fromEntries(['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupStorageSize', 'maxComputeInvocationsPerWorkgroup'].map(key => [key, adapter.limits[key]])) };
    requireCondition(/swiftshader/i.test([info.architecture, info.description, info.device].join(' ')), 'This compatibility check requires the explicitly selected SwiftShader adapter.');
    ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false; ort.env.wasm.wasmPaths = new URL('/vendor/', self.location.href).href;
    ort.env.webgpu.adapter = adapter; ort.env.logLevel = 'warning';
    let index = 0;
    const tracedOrt = { Tensor: ort.Tensor, InferenceSession: { create: async (bytes, options) => {
      const id = ['encoder', 'query', 'step'][index++]; stage('create-' + id); const start = performance.now();
      requireCondition(options.executionProviders.length === 1 && options.executionProviders[0] === 'webgpu', 'The tracker changed its declared execution provider.');
      requireCondition(options.graphOptimizationLevel === report.graphOptimizationLevels[id], 'Graph optimization differs from the production policy.');
      const session = await ort.InferenceSession.create(bytes, options);
      Object.assign(report.models.find(model => model.id === id), { createMs: performance.now() - start, preferredOutputLocation: options.preferredOutputLocation });
      if (!watchedDevice && ort.env.webgpu.device) {
        watchedDevice = ort.env.webgpu.device;
        watchedDevice.addEventListener('uncapturederror', error => { report.gpuErrors.push(String(error.error?.message || error.error || 'WebGPU error')); });
      }
      return { inputNames: session.inputNames, outputNames: session.outputNames,
        run: async (...args) => {
          stage('run-' + id); const outputs = await session.run(...args);
          const locations = Object.fromEntries(Object.entries(outputs).map(([name, tensor]) => [name, tensor.location]));
          report.locations.push({ graph: id, outputs: locations });
          for (const [name, expected] of Object.entries(options.preferredOutputLocation || {})) requireCondition(locations[name] === expected, `${id}/${name} is ${locations[name]}, expected ${expected}.`);
          return outputs;
        },
        release: async () => { await session.release(); released++; } };
    } } };
    tracker = await TapirTracker.create({ ort: tracedOrt, modelBuffers, manifest, executionProviders: ['webgpu'],
      graphSessionOptions: Object.fromEntries(Object.entries(report.graphOptimizationLevels).map(([id, graphOptimizationLevel]) => [id, { graphOptimizationLevel }])) });
    report.checks.push({ name: 'three-real-webgpu-sessions-and-model-hashes', passed: true });
    stage('seed');
    const seed = await tracker.seed({ points: fixture.pointsXY, side: 'left', timestampMs: 0, image });
    requireCondition(tracker.queryLowres?.location === 'gpu-buffer' && tracker.queryHires?.location === 'gpu-buffer' && tracker.causalState?.location === 'gpu-buffer', 'The query features and causal state were not retained on GPU.');
    const queryLow = tracker.queryLowres, queryHigh = tracker.queryHires, state = tracker.causalState;
    report.seed = { inferenceMs: seed.diagnostics.inferenceMs, queryFeatureLocation: seed.diagnostics.queryFeatureLocation, causalStateLocation: seed.diagnostics.causalStateLocation };
    report.checks.push({ name: 'seed-stores-query-and-causal-state-on-gpu', passed: true });
    stage('one-recurrent-update');
    const result = await tracker.update({ image, timestampMs: reference.timestampMs });
    requireCondition(tracker.queryLowres === queryLow && tracker.queryHires === queryHigh && tracker.causalState !== state && tracker.causalState?.location === 'gpu-buffer', 'The update did not reuse GPU query features and replace recurrent GPU state.');
    const comparison = { coordinateMaxAbsPx: 0, logitMaxAbs: 0, confidenceMaxAbs: 0, visibilityMismatches: 0, visibleCoordinatesCompared: 0 };
    const expected = reference.outputs, ids = [17, 18, 19], visibility = result.footTracks[0]?.pointVisibility;
    requireCondition(visibility?.length === 3, 'The point tracker did not return three uncertainty records.');
    for (let i = 0; i < 3; i++) {
      const x = expected.tracks[i * 2] * image.width / 256, y = expected.tracks[i * 2 + 1] * image.height / 256;
      const confidence = 1 / (1 + Math.exp(expected.occlusion[i])) / (1 + Math.exp(expected.expected_dist[i]));
      const visible = confidence > .5 && x >= 0 && y >= 0 && x < image.width && y < image.height, actual = result.keypoints[ids[i]];
      comparison.logitMaxAbs = Math.max(comparison.logitMaxAbs, Math.abs(visibility[i].occlusionLogit - expected.occlusion[i]), Math.abs(visibility[i].expectedDistanceLogit - expected.expected_dist[i]));
      comparison.confidenceMaxAbs = Math.max(comparison.confidenceMaxAbs, Math.abs(visibility[i].score - confidence));
      if (actual.observable !== visible) comparison.visibilityMismatches++;
      if (visible && actual.observable) { comparison.visibleCoordinatesCompared++; comparison.coordinateMaxAbsPx = Math.max(comparison.coordinateMaxAbsPx, Math.abs(actual.x - x), Math.abs(actual.y - y)); }
    }
    report.comparison = comparison;
    requireCondition(comparison.visibleCoordinatesCompared > 0 && comparison.coordinateMaxAbsPx <= .25 && comparison.logitMaxAbs <= .02 && comparison.confidenceMaxAbs <= .002 && comparison.visibilityMismatches === 0, 'WebGPU output differs from the official PyTorch reference beyond the declared tolerance.');
    requireCondition(result.depthKind === 'none' && result.keypoints.every((point, id) => ids.includes(id) || point.observable === false), 'Point tracking invented unobserved body coordinates or depth.');
    report.checks.push({ name: 'gpu-recurrent-update-matches-official-reference', passed: true });
    report.softwareTiming = { seedInferenceMs: seed.diagnostics.inferenceMs, updateInferenceMs: result.diagnostics.inferenceMs, encoderMs: result.diagnostics.encoderMs, stepMs: result.diagnostics.stepMs,
      measuredOn: 'SwiftShader CPU implementation of WebGPU; shader compilation may be included', physicalGpuPerformance: false, hardwareFpsMeasured: false };
    requireCondition(report.gpuErrors.length === 0 && outsideRequests === 0, 'WebGPU reported a device error or attempted external network access.');
    stage('dispose'); await tracker.dispose(); tracker = null;
    requireCondition(released === 3, 'Not all three WebGPU sessions were released.');
    report.checks.push({ name: 'dispose-three-sessions', passed: true });
    report.status = 'measured'; report.stage = 'complete'; report.passed = true;
  } catch (error) {
    report.status = 'failed'; report.error = { type: error?.name || typeof error, message: String(error?.message || error), stack: error?.stack || null };
  } finally {
    if (tracker) { try { await tracker.dispose(); } catch {} }
    report.releasedSessions = released; report.outsideRequests = outsideRequests;
    self.postMessage({ kind: 'result', report });
  }
});
