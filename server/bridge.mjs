import http from 'node:http';
import https from 'node:https';
import dgram from 'node:dgram';
import { randomBytes, randomUUID, timingSafeEqual } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';
import { WebSocketServer, WebSocket } from 'ws';
import { encodeVmtPose } from './osc.mjs';
import { ROLES, validateRoomCalibration, validatePoseFrame, mapFrameToRoom, captureRootProvenance } from './bridge-policy.mjs';

const DEFAULT_DIST = fileURLToPath(new URL('../dist/', import.meta.url));
const MIME = { '.html': 'text/html; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png', '.ico': 'image/x-icon', '.wasm': 'application/wasm', '.gz': 'application/gzip' };
function sameToken(received, expected) {
  if (typeof received !== 'string' || received.length > 256) return false;
  const a = Buffer.from(received), b = Buffer.from(expected);
  return a.length === b.length && timingSafeEqual(a, b);
}
function originValue(value) {
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol) || url.origin !== value) throw new Error('Origin must be an exact http(s) origin, without a path.');
  return value;
}

/** No external listener or arbitrary OSC destination is exposed. */
export async function createBridgeServer(options = {}) {
  const token = options.token || randomBytes(24).toString('base64url');
  if (typeof token !== 'string' || token.length < 24 || token.length > 256) throw new Error('Bridge token must contain 24–256 characters.');
  const port = options.port ?? 8787, udpPort = options.udpPort ?? 39570, staleMs = options.staleMs ?? 500;
  if (!Number.isInteger(port) || port < 0 || port > 65535 || !Number.isInteger(udpPort) || udpPort < 1 || udpPort > 65535 || !Number.isFinite(staleMs) || staleMs < 50 || staleMs > 2000) throw new Error('Invalid bridge port or stale timeout.');
  const customOrigins = (options.origins || []).map(originValue), distRoot = resolve(options.distRoot || DEFAULT_DIST);
  let calibration = options.calibration ? validateRoomCalibration(options.calibration) : null;
  let active = null, actualPort = port, udpCount = 0, closing = false, udpClosed = false;
  const roleState = new Map(ROLES.map(role => [role, { valid: false, position: [0, 0, 0], rotation: [0, 0, 0, 1] }]));
  const indices = options.indices || [0, 1, 2];
  if (!Array.isArray(indices) || indices.length !== 3 || new Set(indices).size !== 3 || indices.some(i => !Number.isInteger(i) || i < 0 || i > 57)) throw new Error('Exactly three distinct VMT indices 0–57 are required.');
  const udp = dgram.createSocket('udp4');
  let udpError = null;
  udp.on('error', error => { udpError = error.code || 'udpError'; });
  await new Promise((done, fail) => { udp.once('error', fail); udp.bind(0, '127.0.0.1', () => { udp.off('error', fail); done(); }); });
  function sendUdp(role, pose) {
    if (udpClosed) return Promise.resolve();
    const packet = encodeVmtPose(indices[ROLES.indexOf(role)], pose.valid, pose.position, pose.rotation);
    return new Promise(resolveSend => udp.send(packet, udpPort, '127.0.0.1', error => { if (error) udpError = error.code || 'udpSendError'; else udpCount++; resolveSend(); }));
  }
  async function invalidate(reason, force = false) {
    const jobs = [];
    for (const role of ROLES) {
      const previous = roleState.get(role);
      if (previous.valid || force) jobs.push(sendUdp(role, { ...previous, valid: false }));
      roleState.set(role, { ...previous, valid: false, reason });
    }
    await Promise.all(jobs);
  }
  const allowedOrigins = () => new Set([...customOrigins, `${options.tls ? 'https' : 'http'}://127.0.0.1:${actualPort}`, `${options.tls ? 'https' : 'http'}://localhost:${actualPort}`]);
  const goodHost = req => [`127.0.0.1:${actualPort}`, `localhost:${actualPort}`].includes(req.headers.host || '');
  const handler = async (req, res) => {
    res.setHeader('X-Content-Type-Options', 'nosniff'); res.setHeader('Referrer-Policy', 'no-referrer'); res.setHeader('Cache-Control', 'no-store');
    if (!goodHost(req)) { res.writeHead(421); res.end('Unrecognized loopback host.'); return; }
    if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405); res.end('Read-only static host.'); return; }
    let pathname;
    try { pathname = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname); } catch { res.writeHead(400); res.end(); return; }
    if (pathname === '/health') {
      res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify({ service: 'wholebody-vmt-bridge', schemaVersion: 1, loopbackOnly: true, steamVrVerified: false })); return;
    }
    if (pathname === '/v1/pose') { res.writeHead(426); res.end('WebSocket required.'); return; }
    const relative = pathname === '/' ? 'index.html' : pathname.replace(/^\/+/, '');
    const path = resolve(distRoot, relative);
    if (!path.startsWith(distRoot + sep) || relative.split(/[\\/]/).some(part => part.startsWith('.')) || !MIME[extname(path)]) { res.writeHead(404); res.end(); return; }
    try {
      const info = await stat(path);
      if (!info.isFile() || info.size > 32 * 1024 * 1024) { res.writeHead(404); res.end(); return; }
      res.setHeader('Content-Type', MIME[extname(path)]); res.setHeader('Content-Length', info.size);
      res.end(req.method === 'HEAD' ? undefined : await readFile(path));
    } catch { res.writeHead(404); res.end(); }
  };
  const server = options.tls ? https.createServer(options.tls, handler) : http.createServer(handler);
  const wss = new WebSocketServer({ noServer: true, maxPayload: 16 * 1024, perMessageDeflate: false });
  server.on('upgrade', (req, socket, head) => {
    let path;
    try { path = new URL(req.url, 'http://127.0.0.1').pathname; } catch { socket.destroy(); return; }
    if (closing || !goodHost(req) || path !== '/v1/pose' || !allowedOrigins().has(req.headers.origin)) { socket.end('HTTP/1.1 403 Forbidden\r\nConnection: close\r\nContent-Length: 0\r\n\r\n'); return; }
    wss.handleUpgrade(req, socket, head, ws => wss.emit('connection', ws, req));
  });
  const status = state => ({ type: 'status', authenticated: true, sessionId: state.sessionId, udpPacketsSent: udpCount,
    enabledTrackers: [...roleState.values()].filter(p => p.valid).length, trackerStates: ROLES.map(role => ({ role, valid: roleState.get(role).valid, reason: roleState.get(role).reason || null })),
    roomAlignmentId: calibration?.roomAlignmentId || null, udpError, steamVrVerified: false });
  wss.on('connection', ws => {
    const state = { ws, authenticated: false, sessionId: randomUUID(), lastSequence: -1, lastTimestamp: -Infinity, streamId: null, lastRootObservation: null, lastAcceptedAt: -Infinity, lastStatusAt: 0, clientClockOffsetMs: 0 };
    const timeout = setTimeout(() => { if (!state.authenticated) ws.close(1008, 'Authentication required.'); }, 2500);
    ws.on('error', () => {});
    ws.on('close', () => { clearTimeout(timeout); if (active === state) { active = null; void invalidate('connectionClosed'); } });
    ws.on('message', (buffer, binary) => {
      let message;
      try { if (binary) throw new Error('Text JSON is required.'); message = JSON.parse(buffer.toString('utf8')); }
      catch { ws.close(1008, 'Invalid message.'); return; }
      if (!state.authenticated) {
        if (message?.type !== 'hello' || message.schemaVersion !== 1 || !sameToken(message.token, token) || !Number.isFinite(message.clientTimeMs) || active) { ws.close(1008, active ? 'Another producer is active.' : 'Authentication failed.'); return; }
        state.authenticated = true; state.clientClockOffsetMs = Date.now() - message.clientTimeMs; active = state; clearTimeout(timeout);
        ws.send(JSON.stringify({ type: 'ready', schemaVersion: 1, sessionId: state.sessionId, serverTimeMs: Date.now(), staleMs, steamVrVerified: false }));
        return;
      }
      if (active !== state || closing) return;
      if (message?.type === 'calibration') {
        try { calibration = validateRoomCalibration(message.calibration); state.lastRootObservation = null; void invalidate('alignmentChanged', true); ws.send(JSON.stringify({ type: 'calibrated', roomAlignmentId: calibration.roomAlignmentId, worldCalibrationId: calibration.worldCalibrationId, steamVrVerified: false })); }
        catch (error) { ws.send(JSON.stringify({ type: 'rejected', reason: error.message })); }
        return;
      }
      if (message?.type === 'stop') { state.lastRootObservation = null; void invalidate('producerStopped', true); return; }
      try {
        validatePoseFrame(message, { sessionId: state.sessionId, lastSequence: state.lastSequence, clientClockOffsetMs: state.clientClockOffsetMs, maxAgeMs: staleMs });
        if (message.streamId === state.streamId && message.referenceTimeMs <= state.lastTimestamp) throw new Error('Repeated or reversed source timestamp.');
        if (performance.now() - state.lastAcceptedAt < 5) throw new Error('Pose rate exceeds the bridge limit.');
        if (message.streamId !== state.streamId) { void invalidate('streamChanged'); state.streamId = message.streamId; state.lastRootObservation = null; }
        state.lastSequence = message.sequence; state.lastTimestamp = message.referenceTimeMs; state.lastAcceptedAt = performance.now();
        const mapped = mapFrameToRoom(message, calibration, { previousRoot: state.lastRootObservation });
        // Only a permitted fresh root observation refreshes provenance. Predictions
        // retain its original source/time, and a rejected source clears it.
        if (message.pose.distanceStatus !== 'estimatedTemporal') state.lastRootObservation = captureRootProvenance(message, calibration);
        for (const target of mapped) {
          const previous = roleState.get(target.role);
          if (target.valid || previous.valid) void sendUdp(target.role, target);
          roleState.set(target.role, target);
        }
        if (performance.now() - state.lastStatusAt > 250) { state.lastStatusAt = performance.now(); ws.send(JSON.stringify(status(state))); }
      } catch (error) {
        state.lastRootObservation = null; void invalidate('frameRejected'); ws.send(JSON.stringify({ type: 'rejected', reason: error.message, steamVrVerified: false }));
      }
    });
  });
  const timer = setInterval(() => {
    if (active && performance.now() - active.lastAcceptedAt > staleMs && [...roleState.values()].some(p => p.valid)) {
      active.lastRootObservation = null; void invalidate('stale'); if (active.ws.readyState === WebSocket.OPEN) active.ws.send(JSON.stringify({ type: 'stale', staleMs, steamVrVerified: false }));
    }
  }, Math.min(50, staleMs / 2));
  try {
    await new Promise((done, fail) => { server.once('error', fail); server.listen(port, '127.0.0.1', () => { server.off('error', fail); done(); }); });
  } catch (error) { clearInterval(timer); udp.close(); throw error; }
  actualPort = server.address().port;
  return { port: actualPort, url: `${options.tls ? 'https' : 'http'}://127.0.0.1:${actualPort}`, token,
    getState: () => ({ port: actualPort, authenticated: !!active, udpPacketsSent: udpCount, enabledTrackers: [...roleState.values()].filter(p => p.valid).length, calibrationId: calibration?.roomAlignmentId || null, steamVrVerified: false }),
    async close() {
      if (closing) return; closing = true; clearInterval(timer); await invalidate('bridgeStopped', true);
      for (const ws of wss.clients) ws.terminate();
      await new Promise(done => server.close(done));
      wss.close(); udpClosed = true; await new Promise(done => udp.close(done));
    } };
}

async function main() {
  const args = process.argv.slice(2), options = { origins: [] };
  let calibrationPath, certPath, keyPath;
  for (let i = 0; i < args.length; i++) {
    const name = args[i], value = args[++i];
    if (!value) throw new Error(`Missing value for ${name}.`);
    if (name === '--origin') options.origins.push(value);
    else if (name === '--port') options.port = Number(value);
    else if (name === '--udp-port') options.udpPort = Number(value);
    else if (name === '--stale-ms') options.staleMs = Number(value);
    else if (name === '--calibration') calibrationPath = value;
    else if (name === '--tls-cert') certPath = value;
    else if (name === '--tls-key') keyPath = value;
    else throw new Error(`Unknown option ${name}.`);
  }
  if (!!certPath !== !!keyPath) throw new Error('Provide both --tls-cert and --tls-key.');
  if (certPath) options.tls = { cert: await readFile(certPath), key: await readFile(keyPath) };
  if (calibrationPath) options.calibration = JSON.parse(await readFile(calibrationPath, 'utf8'));
  options.token = process.env.WHOLEBODY_BRIDGE_TOKEN || randomBytes(24).toString('base64url');
  const bridge = await createBridgeServer(options);
  console.log(`Local tracking page: ${bridge.url}`);
  console.log(`WebSocket: ${bridge.url.replace(/^http/, 'ws')}/v1/pose`);
  console.log(`Connection token (paste into the local page; do not publish): ${bridge.token}`);
  console.log('UDP destination: 127.0.0.1:' + (options.udpPort || 39570) + '. SteamVR/HMD operation is not verified by an OSC send.');
  const shutdown = async () => { await bridge.close(); process.exitCode = 0; };
  process.once('SIGINT', shutdown); process.once('SIGTERM', shutdown);
}
if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) main().catch(error => { console.error(error.message); process.exitCode = 1; });
