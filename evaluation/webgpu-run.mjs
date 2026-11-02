/** GitHub Actions only. A disposable Chromium tests public WebGPU compute. */
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { readFile, mkdir, writeFile, stat } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { createHash } from 'node:crypto';
import { resolve, join, dirname } from 'node:path';

if (process.env.GITHUB_ACTIONS !== 'true') throw new Error('Run the isolated WebGPU compatibility check in GitHub Actions only.');
const require = createRequire(new URL('./webgpu-deps/package.json', import.meta.url));
const { chromium } = require('playwright');
const argument = (name, fallback) => { const index = process.argv.indexOf(name); return index < 0 ? fallback : process.argv[index + 1]; };
const models = resolve(argument('--models', '.cache/webgpu-ci/models'));
const fixtures = resolve(argument('--fixtures', '.cache/webgpu-ci/results/codec-fixture'));
const runtime = resolve(argument('--runtime', '.cache/webgpu-ci/runtime/vendor'));
const output = resolve(argument('--output', 'evaluation/webgpu-results/phase3-webgpu.json'));
const root = new URL('../', import.meta.url).pathname;
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const requireCondition = (condition, message) => { if (!condition) throw new Error(message); };
const report = { schemaVersion: 1, passed: false, status: 'started', gitCommit: process.env.GITHUB_SHA, workflowRunId: process.env.GITHUB_RUN_ID,
  softwareOnly: true, physicalGpuMeasured: false, hardwareFpsMeasured: false, privateMediaUsed: false,
  playwrightVersion: require('playwright/package.json').version, runtimeVersion: '1.31.1', runtimeSource: 'Same-commit evaluation/package-runtime.mjs; the core workflow validates this candidate separately.',
  publicModelArtifactRunId: '41384892970', publicModelArtifactGitCommit: '274407d44a7fbdcb8411f9dc78286b1f49409c6d',
  flags: ['--enable-gpu', '--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--use-gl=angle', '--use-angle=swiftshader', '--use-vulkan=swiftshader',
    '--disable-background-networking', '--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost, EXCLUDE 127.0.0.1'],
  references: ['https://chromium.googlesource.com/chromium/src/+/main/docs/gpu/swiftshader.md',
    'https://chromium.googlesource.com/chromium/src/+/main/docs/gpu/using-gpu-hardware-in-headless-chrome.md', 'https://playwright.dev/docs/browsers#chromium-new-headless-mode'],
  console: [], blockedExternalRequests: [], localRequests: 0 };
let browser = null, server = null, caught = null;
try {
  requireCondition(report.playwrightVersion === '1.56.1', 'The Playwright version differs from the declared dependency.');
  const runtimeManifest = JSON.parse(await readFile(join(runtime, 'runtime-manifest.json'), 'utf8'));
  requireCondition(runtimeManifest.ortWebVersion === report.runtimeVersion && runtimeManifest.gitCommit === process.env.GITHUB_SHA, 'The WebGPU runtime was not packaged by this candidate commit.');
  const runtimeFiles = ['ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm'];
  report.runtime = [];
  for (const name of runtimeFiles) {
    const record = runtimeManifest.records.find(value => value.file === name), bytes = await readFile(join(runtime, name));
    requireCondition(record?.url === `https://cdn.jsdelivr.net/npm/onnxruntime-web@${report.runtimeVersion}/dist/${name}` && bytes.byteLength === record.bytes && sha256(bytes) === record.sha256, 'Candidate ORT runtime hash mismatch: ' + name);
    report.runtime.push({ file: name, bytes: record.bytes, sha256: record.sha256 });
  }
  const routes = new Map([
    ['/entry.mjs', join(root, 'evaluation/webgpu-entry.mjs')], ['/worker.mjs', join(root, 'evaluation/webgpu-worker.mjs')],
    ['/dist/tapir-tracker.mjs', join(root, 'dist/tapir-tracker.mjs')], ['/dist/tapir-codec.mjs', join(root, 'dist/tapir-codec.mjs')],
    ['/models/tapir-models.json', join(models, 'tapir-models.json')], ['/fixtures/fixture.json', join(fixtures, 'fixture.json')], ['/fixtures/public-rgb.u8', join(fixtures, 'public-rgb.u8')],
    ...runtimeFiles.map(name => ['/vendor/' + name, join(runtime, name)]),
    ...['encoder', 'query', 'step'].map(id => { const name = `causal-bootstapir-${id}-256-q3-fp32.onnx`; return ['/models/' + name, join(models, name)]; })
  ]);
  server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url, 'http://127.0.0.1');
      response.setHeader('Cross-Origin-Opener-Policy', 'same-origin'); response.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
      response.setHeader('Cross-Origin-Resource-Policy', 'same-origin'); response.setHeader('Cache-Control', 'no-store');
      response.setHeader('Content-Security-Policy', "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'");
      if (request.method !== 'GET' || request.headers.host !== `127.0.0.1:${server.address().port}`) { response.writeHead(403); response.end(); return; }
      report.localRequests++;
      if (url.pathname === '/') { response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' }); response.end('<!doctype html><meta charset="utf-8"><title>Public WebGPU compatibility</title><script type="module" src="/entry.mjs"></script>'); return; }
      const path = routes.get(url.pathname);
      if (!path) { response.writeHead(404); response.end(); return; }
      const metadata = await stat(path), type = path.endsWith('.mjs') ? 'text/javascript' : path.endsWith('.json') ? 'application/json' : path.endsWith('.wasm') ? 'application/wasm' : 'application/octet-stream';
      response.writeHead(200, { 'Content-Type': type, 'Content-Length': metadata.size });
      createReadStream(path).on('error', () => response.destroy()).pipe(response);
    } catch { response.writeHead(500); response.end(); }
  });
  await new Promise((resolveReady, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolveReady); });
  const origin = `http://127.0.0.1:${server.address().port}`;
  browser = await chromium.launch({ headless: true, channel: 'chromium', args: report.flags, timeout: 60_000 });
  report.chromiumVersion = browser.version();
  const context = await browser.newContext({ serviceWorkers: 'block' });
  await context.route('**/*', route => {
    if (new URL(route.request().url()).origin === origin) return route.continue();
    report.blockedExternalRequests.push(new URL(route.request().url()).origin); return route.abort('blockedbyclient');
  });
  const page = await context.newPage();
  page.on('console', message => {
    const text = message.text(); if (report.console.length < 120) report.console.push({ type: message.type(), text: text.slice(0, 2400) });
    if (text.startsWith('WEBGPU_WORKER_STAGE')) console.log(text);
  });
  page.on('pageerror', error => { if (report.console.length < 120) report.console.push({ type: 'pageerror', text: error.message }); });
  await page.goto(origin, { waitUntil: 'load', timeout: 30_000 });
  await page.waitForFunction(() => !!window.__webgpuResult, null, { timeout: 30_000 });
  const outcome = await Promise.race([
    page.evaluate(() => window.__webgpuResult),
    new Promise((_, reject) => { const timeout = setTimeout(() => reject(new Error('SwiftShader compatibility run exceeded 12 minutes.')), 12 * 60_000); timeout.unref(); })
  ]);
  report.worker = outcome;
  requireCondition(outcome?.passed === true && outcome.softwareOnly === true && outcome.physicalGpuMeasured === false && report.blockedExternalRequests.length === 0, 'The isolated WebGPU compatibility run did not pass: ' + (outcome?.error?.message || outcome?.stage || 'no result'));
  report.passed = true; report.status = 'measured';
} catch (error) {
  caught = error; report.status = 'failed'; report.error = { type: error?.name || typeof error, message: String(error?.message || error), stack: error?.stack || null };
} finally {
  if (browser) { try { await browser.close(); } catch {} }
  if (server) { server.closeAllConnections(); await new Promise(resolveDone => server.close(resolveDone)); }
  await mkdir(dirname(output), { recursive: true }); await writeFile(output, JSON.stringify(report, null, 2) + '\n');
}
console.log(JSON.stringify({ passed: report.passed, status: report.status, softwareOnly: true, physicalGpuMeasured: false, stage: report.worker?.stage, comparison: report.worker?.comparison, error: report.error || null }));
if (caught) throw caught;
