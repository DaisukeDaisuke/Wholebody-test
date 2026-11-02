#!/usr/bin/env node
// Optional native bridge. Uses Node built-ins only; no inference or video upload.
import http from 'node:http';
import dgram from 'node:dgram';
import { randomBytes, timingSafeEqual } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { dirname, resolve, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
function argument(name, fallback) { const i = args.indexOf(name); return i >= 0 && args[i + 1] ? args[i + 1] : fallback; }
const port = Number(argument('--port', '8765'));
const udpPort = Number(argument('--vmt-port', '39570'));
if (![port, udpPort].every(v => Number.isInteger(v) && v > 0 && v <= 65535)) throw new Error('Port must be in 1..65535.');
const token = process.env.TRACKER_TOKEN || randomBytes(24).toString('base64url');
const allowedOrigins = new Set([`http://127.0.0.1:${port}`, `http://localhost:${port}`]);
for (let i = 0; i < args.length; i++) if (args[i] === '--origin' && args[i + 1]) allowedOrigins.add(new URL(args[++i]).origin);
const allowEstimated = args.includes('--allow-estimated');
const staticRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../dist');
const udp = dgram.createSocket('udp4');
const roleIds = new Map([['waist', 0], ['left-foot', 1], ['right-foot', 2], ['chest', 3], ['left-knee', 4], ['right-knee', 5]]);
const activeTrackers = new Map();
let lastPoseAt = 0, lastSequence = -1, lastMediaTime = null, sent = 0, shutdownStarted = false;
let lastClient = null;

function oscString(value) {
  const raw = Buffer.from(value + '\0', 'utf8');
  const padded = Buffer.alloc(Math.ceil(raw.length / 4) * 4); raw.copy(padded); return padded;
}
function oscTracker(index, enabled, position, quaternion) {
  const types = ',iiffffffff';
  const values = [index, enabled, 0, ...position, ...quaternion];
  const data = Buffer.alloc(values.length * 4);
  values.forEach((value, i) => { if (i < 2) data.writeInt32BE(value, i * 4); else data.writeFloatBE(value, i * 4); });
  return Buffer.concat([oscString('/VMT/Room/Unity'), oscString(types), data]);
}
function sendTracker(index, enabled, position = [0, 0, 0], quaternion = [0, 0, 0, 1]) {
  udp.send(oscTracker(index, enabled, position, quaternion), udpPort, '127.0.0.1'); sent++;
}
function disableAll() { for (const index of activeTrackers.keys()) sendTracker(index, 0); activeTrackers.clear(); }
function equalSecret(actual) {
  const a = Buffer.from(actual || ''), b = Buffer.from(token);
  return a.length === b.length && timingSafeEqual(a, b);
}
function finiteArray(a, count) { return Array.isArray(a) && a.length === count && a.every(Number.isFinite); }
function processPose(data) {
  if (!data || data.type !== 'fused-pose' || !Array.isArray(data.trackers) || data.trackers.length > 12) throw new Error('Expected a fused-pose with at most 12 trackers.');
  if (!Number.isSafeInteger(data.sequence) || data.sequence < 0) throw new Error('Invalid sequence.');
  if (!Number.isFinite(data.mediaTimeMs)) throw new Error('Missing media timestamp.');
  // A fresh page can explicitly reset the stream; otherwise reordered frames cannot revive old poses.
  const clientId = typeof data.clientId === 'string' ? data.clientId.slice(0, 80) : 'default';
  if (clientId !== lastClient || data.reset === true || Date.now() - lastPoseAt > 2000) { lastSequence = -1; lastMediaTime = null; lastClient = clientId; disableAll(); }
  if (data.sequence <= lastSequence) throw new Error('Stale or duplicate sequence.');
  if (lastMediaTime !== null && data.mediaTimeMs < lastMediaTime - 250 && !data.reset) { disableAll(); throw new Error('Video seek requires reset=true or a new clientId.'); }
  const measured = data.root?.metricCalibrated === true;
  const usable = data.root?.valid === true && (measured || allowEstimated);
  const updated = new Set(), commands = [], rolesSeen = new Set();
  for (const tracker of data.trackers) {
    if (!tracker || typeof tracker !== 'object') throw new Error('Invalid tracker entry.');
    const index = roleIds.get(tracker.role);
    if (index === undefined) continue;
    if (rolesSeen.has(index)) throw new Error('Duplicate tracker role.');
    rolesSeen.add(index);
    const valid = usable && (tracker.valid === true || (allowEstimated && tracker.poseValid === true)) &&
      finiteArray(tracker.positionXYZm, 3) && finiteArray(tracker.rotationXYZW, 4) &&
      tracker.positionXYZm.every(v => Math.abs(v) <= 50);
    if (!valid) { commands.push({ index, enabled: false }); continue; }
    const [x, y, z] = tracker.positionXYZm;
    const [qx, qy, qz, qw] = tracker.rotationXYZW;
    const qLength = Math.hypot(qx, qy, qz, qw);
    if (qLength < 0.9 || qLength > 1.1) throw new Error('Tracker quaternion must be normalized.');
    // Three.js RH -> Unity LH reflection: z changes sign; quaternion x/y change sign.
    const position = [x, y, -z], rotation = [-qx / qLength, -qy / qLength, qz / qLength, qw / qLength];
    commands.push({ index, enabled: true, position, rotation });
  }
  // Validate the entire pose before emitting a partial tracker update.
  for (const command of commands) {
    if (!command.enabled) { if (activeTrackers.has(command.index)) sendTracker(command.index, 0); activeTrackers.delete(command.index); continue; }
    sendTracker(command.index, 1, command.position, command.rotation);
    activeTrackers.set(command.index, Date.now()); updated.add(command.index);
  }
  for (const index of activeTrackers.keys()) if (!updated.has(index)) { sendTracker(index, 0); activeTrackers.delete(index); }
  lastPoseAt = Date.now(); lastSequence = data.sequence; lastMediaTime = data.mediaTimeMs;
  return { accepted: true, activeTrackers: activeTrackers.size, estimatedEnabled: allowEstimated };
}

const server = http.createServer(async (request, response) => {
  const origin = request.headers.origin;
  if (origin && !allowedOrigins.has(origin)) { response.writeHead(403); response.end('Origin not allowed'); return; }
  const headers = { 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' };
  if (origin) { headers['Access-Control-Allow-Origin'] = origin; headers.Vary = 'Origin'; }
  const address = new URL(request.url, `http://127.0.0.1:${port}`);
  const sendJSON = (status, body) => { response.writeHead(status, { ...headers, 'Content-Type': 'application/json; charset=utf-8' }); response.end(JSON.stringify(body)); };
  if (request.method === 'OPTIONS') {
    response.writeHead(204, { ...headers, 'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization', 'Access-Control-Max-Age': '600',
      'Access-Control-Allow-Private-Network': 'true' }); response.end(); return;
  }
  if (request.method === 'GET' && address.pathname === '/health') {
    sendJSON(200, { ok: true, activeTrackers: activeTrackers.size, receivedAgeMs: lastPoseAt ? Date.now() - lastPoseAt : null,
      packetsSent: sent, estimatedEnabled: allowEstimated, destination: '127.0.0.1:' + udpPort }); return;
  }
  if (request.method === 'POST' && address.pathname === '/pose') {
    if (!equalSecret(request.headers.authorization?.replace(/^Bearer /, ''))) { sendJSON(401, { error: 'Invalid bridge token' }); request.resume(); return; }
    if (!request.headers['content-type']?.startsWith('application/json')) { sendJSON(415, { error: 'Use application/json' }); request.resume(); return; }
    let bytes = 0; const chunks = [];
    try {
      for await (const chunk of request) { bytes += chunk.length; if (bytes > 262144) throw new Error('Pose message exceeds 256 KiB'); chunks.push(chunk); }
      sendJSON(200, processPose(JSON.parse(Buffer.concat(chunks).toString('utf8'))));
    } catch (error) { if (!response.headersSent) sendJSON(400, { error: error.message }); }
    return;
  }
  if (request.method === 'GET' && address.pathname === '/') {
    response.writeHead(302, { ...headers, Location: '/index.html' }); response.end(); return;
  }
  if (request.method === 'GET') {
    let path;
    try { path = resolve(staticRoot, '.' + decodeURIComponent(address.pathname)); }
    catch { sendJSON(400, { error: 'Invalid path' }); return; }
    if (!path.startsWith(staticRoot + sep)) { sendJSON(403, { error: 'Forbidden' }); return; }
    const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
      '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8', '.wasm': 'application/wasm' }[extname(path)];
    if (!mime) { sendJSON(404, { error: 'Not found' }); return; }
    try { const body = await readFile(path); response.writeHead(200, { ...headers, 'Content-Type': mime }); response.end(body); }
    catch { sendJSON(404, { error: 'Not found' }); }
    return;
  }
  sendJSON(404, { error: 'Not found' });
});

udp.on('error', error => { console.error('VMT UDP:', error.message); });
server.requestTimeout = 5000;
server.headersTimeout = 5000;
const watchdog = setInterval(() => { if (lastPoseAt && Date.now() - lastPoseAt > 500) disableAll(); }, 100);
function shutdown() {
  if (shutdownStarted) return; shutdownStarted = true; clearInterval(watchdog); disableAll();
  server.close(() => udp.close()); setTimeout(() => process.exit(0), 500).unref();
}
process.once('SIGINT', shutdown); process.once('SIGTERM', shutdown);
server.listen(port, '127.0.0.1', () => {
  console.log(`Video tracker: http://127.0.0.1:${port}/`);
  console.log(`Bridge token: ${token}`);
  console.log(`VMT: 127.0.0.1:${udpPort}; origins: ${[...allowedOrigins].join(', ')}`);
  console.log(`Estimated uncalibrated trackers: ${allowEstimated ? 'enabled explicitly' : 'disabled'}`);
});
