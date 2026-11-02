import { poseToTrackers } from './retarget.mjs';

const randomId = () => globalThis.crypto?.randomUUID?.() || `stream-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
/** Browser client: only three tracker transforms are sent, never images or 133-point traces. */
export class BridgeClient {
  constructor({ onStatus = () => {}, maxFps = 30, WebSocketImpl = globalThis.WebSocket } = {}) {
    this.onStatus = onStatus; this.maxFps = maxFps; this.WebSocketImpl = WebSocketImpl;
    this.ws = null; this.sessionId = null; this.sequence = 0; this.streamId = randomId(); this.lastTime = -Infinity;
    this.lastSend = -Infinity; this.inputKey = null; this.lastAbsoluteMode = null; this.connected = false; this.staleMs = 500;
  }
  status(value) { this.onStatus({ ...value, steamVrVerified: false }); }
  async connect({ url = 'ws://127.0.0.1:8787', token } = {}) {
    this.disconnect();
    const target = new URL(url);
    if (!['ws:', 'wss:'].includes(target.protocol) || !['127.0.0.1', 'localhost', '[::1]'].includes(target.hostname) || target.username || target.password || target.search || target.hash) throw new Error('Bridge URL must be a loopback ws/wss URL without credentials or query parameters.');
    if (target.pathname === '/') target.pathname = '/v1/pose';
    if (target.pathname !== '/v1/pose') throw new Error('Unsupported bridge WebSocket path.');
    if (typeof token !== 'string' || token.length < 24 || token.length > 256) throw new Error('Paste the bridge token shown by the local Node process.');
    if (!this.WebSocketImpl) throw new Error('WebSocket is unavailable.');
    this.status({ state: 'connecting' });
    const ws = new this.WebSocketImpl(target.href); this.ws = ws;
    await new Promise((resolve, reject) => {
      let ready = false;
      const timeout = setTimeout(() => { ws.close(); reject(new Error('Bridge authentication timed out.')); }, 4000);
      ws.addEventListener('open', () => { ws.send(JSON.stringify({ type: 'hello', schemaVersion: 1, token, clientTimeMs: Date.now() })); token = null; });
      ws.addEventListener('message', event => {
        if (this.ws !== ws) return;
        let message; try { message = JSON.parse(event.data); } catch { return; }
        if (message.type === 'ready') {
          ready = true; clearTimeout(timeout); this.connected = true; this.sessionId = message.sessionId; this.staleMs = message.staleMs;
          this.sequence = 0; this.inputKey = null; this.lastAbsoluteMode = null; this.lastTime = -Infinity; this.lastSend = -Infinity;
          this.status({ state: 'connected', ...message }); resolve();
        } else this.status({ state: message.type, ...message });
      });
      ws.addEventListener('error', () => { clearTimeout(timeout); if (!ready) reject(new Error('Local bridge connection failed. Use its local page or a trusted wss endpoint.')); this.status({ state: 'error' }); });
      ws.addEventListener('close', event => {
        clearTimeout(timeout); token = null;
        if (this.ws === ws) { this.connected = false; this.sessionId = null; this.ws = null; this.status({ state: 'disconnected', code: event.code, reason: event.reason || '' }); }
        if (!ready) reject(new Error(event.reason || 'Bridge closed before authentication.'));
      });
    });
    return this;
  }
  calibrate(calibration) {
    if (!this.connected) throw new Error('Connect the local bridge first.');
    this.ws.send(JSON.stringify({ type: 'calibration', calibration }));
  }
  sendPose(pose, { captureAgeMs = 0, timeDomain = 'unknown' } = {}) {
    if (!this.connected || !pose || !Number.isFinite(pose.timestampMs) || this.ws.bufferedAmount > 16 * 1024) return false;
    const now = globalThis.performance?.now?.() ?? Date.now();
    if (now - this.lastSend < 1000 / this.maxFps) return false;
    if (!Number.isFinite(captureAgeMs) || captureAgeMs < 0 || captureAgeMs > this.staleMs) { this.stop(); return false; }
    const key = `${pose.sourceId || ''}:${pose.targetVersion || ''}:${pose.inputMode || ''}:${pose.coordinateFrame || ''}:${pose.units || ''}:${pose.worldCalibrationId || ''}:${timeDomain}`;
    if (key !== this.inputKey || pose.timestampMs < this.lastTime) { this.inputKey = key; this.streamId = randomId(); this.lastTime = -Infinity; this.lastAbsoluteMode = null; }
    if (pose.timestampMs <= this.lastTime) return false;
    const distanceStatus = pose.root?.distanceStatus || 'uncalibrated';
    if (['measuredMultiView', 'estimatedMonocular', 'estimatedMultiView'].includes(distanceStatus)) this.lastAbsoluteMode = distanceStatus;
    else if (distanceStatus !== 'estimatedTemporal') this.lastAbsoluteMode = null;
    const frame = { schemaVersion: 1, type: 'pose', sessionId: this.sessionId, streamId: this.streamId, sequence: ++this.sequence,
      referenceTimeMs: pose.timestampMs, sentAtMs: Date.now(), captureAgeMs, timeDomain,
      pose: { metric: pose.metric === true, rootValid: pose.root?.valid === true, coordinateFrame: pose.coordinateFrame, units: pose.units,
        worldCalibrationId: pose.worldCalibrationId || null, distanceStatus, originDistanceStatus: this.lastAbsoluteMode,
        calibratedWorld: pose.diagnostics?.calibratedWorld === true || pose.root?.distanceStatus === 'measuredMultiView' },
      trackers: poseToTrackers(pose) };
    // Do not serialize the full pose object: it may contain non-public recording data.
    this.ws.send(JSON.stringify(frame)); this.lastSend = now; this.lastTime = pose.timestampMs; return true;
  }
  stop() { if (this.connected && this.ws?.readyState === 1) this.ws.send(JSON.stringify({ type: 'stop' })); }
  disconnect() {
    this.stop(); const ws = this.ws; this.ws = null; this.connected = false; this.sessionId = null;
    if (ws && ws.readyState < 2) ws.close(1000, 'Producer disconnected.');
    if (ws) this.status({ state: 'disconnected', reason: 'Producer disconnected.' });
  }
}
