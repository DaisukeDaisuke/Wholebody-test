// Direct, non-trickle WebRTC pairing. No public signaling, STUN, or TURN service.
const SIGNAL_LIMIT = 128 * 1024;
const CLOCK_LIMIT = 40;

export function isLanAddress(address) {
  if (typeof address !== 'string') return false;
  const value = address.toLowerCase().replace(/^\[|\]$/g, '').split('%')[0];
  if (/^[a-z0-9][a-z0-9.-]*\.local\.?$/.test(value)) return true;
  if (value === '::1' || /^(?:fc|fd)[0-9a-f]{2}:/.test(value) || /^fe[89ab][0-9a-f]:/.test(value)) return true;
  if (value.startsWith('::ffff:')) return isLanAddress(value.slice(7));
  const parts = value.split('.').map(Number);
  if (parts.length !== 4 || parts.some((n, i) => !Number.isInteger(n) || n < 0 || n > 255 || !/^\d+$/.test(value.split('.')[i]))) return false;
  return parts[0] === 10 || parts[0] === 127 ||
    (parts[0] === 192 && parts[1] === 168) ||
    (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31) ||
    (parts[0] === 169 && parts[1] === 254);
}

export function lanOnlySdp(sdp) {
  if (typeof sdp !== 'string' || sdp.length > SIGNAL_LIMIT) throw new Error('接続コードのSDPが不正です。');
  let candidateCount = 0;
  const lines = sdp.split(/\r?\n/).filter((line) => {
    if (!line.startsWith('a=candidate:')) return true;
    const parts = line.slice(2).trim().split(/\s+/);
    const typ = parts.indexOf('typ');
    if (typ < 0 || parts[typ + 1] !== 'host' || !isLanAddress(parts[4])) return false;
    candidateCount++;
    return true;
  });
  if (!candidateCount) throw new Error('LAN内の接続候補が見つかりません。同じWi-Fiと端末の接続許可を確認してください。');
  return { sdp: lines.filter(Boolean).join('\r\n') + '\r\n', candidateCount };
}

export function encodeSignal(signal) {
  const bytes = new TextEncoder().encode(JSON.stringify(signal));
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replaceAll('+', '-').replaceAll('/', '_').replace(/=+$/, '');
}

export function parseSignal(input) {
  if (typeof input === 'object' && input !== null) return validateSignal(input);
  let text = String(input || '').trim();
  if (text.length > SIGNAL_LIMIT * 2) throw new Error('接続コードが長すぎます。');
  if (/^https?:\/\//i.test(text)) {
    const url = new URL(text);
    text = new URLSearchParams(url.hash.slice(1)).get('offer') || '';
  }
  if (!text) throw new Error('接続コードを貼り付けてください。');
  let value;
  try {
    if (text.startsWith('{')) value = JSON.parse(text);
    else {
      const encoded = text.replaceAll('-', '+').replaceAll('_', '/');
      const binary = atob(encoded.padEnd(Math.ceil(encoded.length / 4) * 4, '='));
      value = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(Uint8Array.from(binary, (c) => c.charCodeAt(0))));
    }
  } catch {
    throw new Error('接続コードを読み取れません。省略せずにコピーしてください。');
  }
  return validateSignal(value);
}

function validateSignal(value) {
  if (!value || value.version !== 1 || value.app !== 'wholebody-phase2' ||
      !['offer', 'answer'].includes(value.type) ||
      typeof value.sessionId !== 'string' || !/^[a-zA-Z0-9-]{12,80}$/.test(value.sessionId) ||
      !['a', 'b'].includes(value.sourceId)) throw new Error('このPoCの接続コードではありません。');
  const { sdp, candidateCount } = lanOnlySdp(value.sdp);
  return { version: 1, app: 'wholebody-phase2', type: value.type, sessionId: value.sessionId, sourceId: value.sourceId, sdp, candidateCount };
}

function uniqueId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return Array.from(crypto.getRandomValues(new Uint8Array(18)), (n) => n.toString(16).padStart(2, '0')).join('');
}

function waitForIce(pc) {
  if (pc.iceGatheringState === 'complete') return Promise.resolve();
  return new Promise((resolve, reject) => {
    const finish = () => {
      clearTimeout(timer);
      pc.removeEventListener('icegatheringstatechange', changed);
      pc.removeEventListener('connectionstatechange', closed);
      resolve();
    };
    const changed = () => { if (pc.iceGatheringState === 'complete') finish(); };
    const closed = () => {
      if (pc.connectionState !== 'closed') return;
      clearTimeout(timer);
      pc.removeEventListener('icegatheringstatechange', changed);
      pc.removeEventListener('connectionstatechange', closed);
      reject(new Error('接続処理が閉じられました。'));
    };
    const timer = setTimeout(finish, 12000);
    pc.addEventListener('icegatheringstatechange', changed);
    pc.addEventListener('connectionstatechange', closed);
  });
}

export class LanPeer {
  constructor({ sourceId = 'a', onStream = () => {}, onState = () => {}, onClock = () => {}, onAudit = () => {} } = {}) {
    if (typeof RTCPeerConnection === 'undefined') throw new Error('このブラウザはWebRTCに対応していません。');
    this.sourceId = sourceId;
    this.sessionId = uniqueId();
    this.onStream = onStream;
    this.onState = onState;
    this.onClock = onClock;
    this.onAudit = onAudit;
    this.clock = null;
    this.audit = { status: 'notConnected', iceServers: [], relay: false };
    this.samples = [];
    this.timers = new Set();
    this.closed = false;
    this.localStream = null;
    this.pc = new RTCPeerConnection({ iceServers: [], bundlePolicy: 'max-bundle', iceCandidatePoolSize: 0 });
    this.pc.ontrack = (event) => {
      if (event.track.kind === 'video') this.onStream(event.streams[0] || new MediaStream([event.track]));
    };
    this.pc.ondatachannel = (event) => this.attachChannel(event.channel);
    this.pc.onconnectionstatechange = () => {
      const state = this.pc.connectionState;
      this.onState({ state, message: this.stateLabel(state) });
      if (state === 'connected') this.inspectRoute().catch(() => {});
      if (state === 'failed') this.onState({ state, message: 'LAN接続に失敗しました。Wi-Fiの端末間通信制限と接続コードを確認してください。' });
    };
    this.pc.onicecandidateerror = (event) => {
      this.audit = { ...this.audit, lastIceError: event.errorText || String(event.errorCode || '') };
    };
    this.auditInterval = setInterval(() => this.inspectRoute().catch(() => {}), 3000);
  }

  stateLabel(state) {
    return ({ new: '接続準備中', connecting: 'LAN接続中', connected: 'LANで映像を受信', disconnected: '接続が途切れています', failed: '接続失敗', closed: '切断済み' })[state] || state;
  }

  async makeOffer() {
    this.pc.addTransceiver('video', { direction: 'recvonly' });
    this.attachChannel(this.pc.createDataChannel('wholebody-clock-v1', { ordered: false, maxRetransmits: 0 }));
    await this.pc.setLocalDescription(await this.pc.createOffer());
    await waitForIce(this.pc);
    return this.signalFromLocal();
  }

  async acceptOffer(input, stream) {
    const signal = parseSignal(input);
    if (signal.type !== 'offer') throw new Error('PCの招待コードを指定してください。');
    this.sourceId = signal.sourceId;
    this.sessionId = signal.sessionId;
    this.localStream = stream;
    await this.pc.setRemoteDescription({ type: 'offer', sdp: signal.sdp });
    for (const track of stream.getVideoTracks()) this.pc.addTrack(track, stream);
    await this.pc.setLocalDescription(await this.pc.createAnswer());
    await waitForIce(this.pc);
    return this.signalFromLocal();
  }

  async acceptAnswer(input) {
    const signal = parseSignal(input);
    if (signal.type !== 'answer' || signal.sessionId !== this.sessionId || signal.sourceId !== this.sourceId) {
      throw new Error('この入力の招待に対応した応答ではありません。新しい招待から接続し直してください。');
    }
    await this.pc.setRemoteDescription({ type: 'answer', sdp: signal.sdp });
  }

  signalFromLocal() {
    const { sdp } = lanOnlySdp(this.pc.localDescription?.sdp);
    return { version: 1, app: 'wholebody-phase2', type: this.pc.localDescription.type, sessionId: this.sessionId, sourceId: this.sourceId, sdp };
  }

  attachChannel(channel) {
    this.channel = channel;
    channel.onopen = () => {
      for (let i = 0; i < 8; i++) {
        const timer = setTimeout(() => { this.timers.delete(timer); this.ping(); }, i * 180);
        this.timers.add(timer);
      }
      this.clockInterval = setInterval(() => this.ping(), 2500);
    };
    channel.onmessage = (event) => {
      const receivedAt = performance.now();
      if (typeof event.data !== 'string' || event.data.length > 4096) return;
      let message;
      try { message = JSON.parse(event.data); } catch { return; }
      if (message.sessionId !== this.sessionId || message.version !== 1) return;
      if (message.type === 'clock-ping' && Number.isFinite(message.t0)) {
        this.send({ type: 'clock-pong', t0: message.t0, t1: receivedAt, t2: performance.now() });
      } else if (message.type === 'clock-pong' && [message.t0, message.t1, message.t2].every(Number.isFinite)) {
        this.acceptClock(message, receivedAt);
      }
    };
  }

  send(message) {
    if (this.channel?.readyState === 'open' && this.channel.bufferedAmount < 65536) {
      this.channel.send(JSON.stringify({ version: 1, sessionId: this.sessionId, ...message }));
    }
  }

  ping() { this.send({ type: 'clock-ping', t0: performance.now() }); }

  acceptClock(message, t3) {
    const rttMs = (t3 - message.t0) - (message.t2 - message.t1);
    if (rttMs < 0 || rttMs > 3000 || message.t2 < message.t1 || message.t0 > t3) return;
    const offsetMs = ((message.t1 - message.t0) + (message.t2 - t3)) / 2;
    const at = (message.t0 + t3) / 2;
    this.samples.push({ rttMs, offsetMs, at });
    if (this.samples.length > CLOCK_LIMIT) this.samples.shift();
    const sorted = [...this.samples].sort((a, b) => a.rttMs - b.rttMs);
    const best = sorted[0];
    const lowLatency = sorted.filter((sample) => sample.rttMs <= best.rttMs + Math.max(2, best.rttMs * .5)).slice(0, 12);
    const offsets = lowLatency.map((sample) => sample.offsetMs).sort((a, b) => a - b);
    const medianOffset = offsets[Math.floor(offsets.length / 2)];
    const spread = Math.max(...offsets.map((offset) => Math.abs(offset - medianOffset)), 0);
    let driftPpm = null;
    if (lowLatency.length >= 4) {
      const meanX = lowLatency.reduce((sum, sample) => sum + sample.at, 0) / lowLatency.length;
      const meanY = lowLatency.reduce((sum, sample) => sum + sample.offsetMs, 0) / lowLatency.length;
      const denominator = lowLatency.reduce((sum, sample) => sum + (sample.at - meanX) ** 2, 0);
      if (denominator > 1e8) driftPpm = 1e6 * lowLatency.reduce((sum, sample) => sum + (sample.at - meanX) * (sample.offsetMs - meanY), 0) / denominator;
    }
    this.clock = { offsetMs: medianOffset, rttMs: best.rttMs, uncertaintyMs: best.rttMs / 2 + spread, samples: this.samples.length, measuredAtMs: t3, driftPpm, convention: 'remote-minus-local' };
    this.onClock(this.clock);
  }

  async inspectRoute() {
    if (this.closed || this.pc.connectionState !== 'connected') return this.audit;
    const stats = await this.pc.getStats();
    let pair;
    let inbound;
    let outbound;
    for (const stat of stats.values()) {
      if (stat.type === 'transport' && stat.selectedCandidatePairId) pair = stats.get(stat.selectedCandidatePairId);
      if (stat.type === 'inbound-rtp' && stat.kind === 'video') inbound = stat;
      if (stat.type === 'outbound-rtp' && stat.kind === 'video') outbound = stat;
    }
    if (!pair) for (const stat of stats.values()) if (stat.type === 'candidate-pair' && stat.nominated && stat.state === 'succeeded') { pair = stat; break; }
    const local = pair ? stats.get(pair.localCandidateId) : null;
    const remote = pair ? stats.get(pair.remoteCandidateId) : null;
    const invalid = [local, remote].some((candidate) => candidate && (candidate.candidateType !== 'host' || (candidate.address && !isLanAddress(candidate.address))));
    const selectedKnown = Boolean(local && remote);
    this.audit = {
      status: invalid ? 'rejectedNonLanRoute' : selectedKnown ? 'hostCandidatesOnly' : 'routeNotDisclosed',
      iceServers: [], relay: false,
      localCandidateType: local?.candidateType ?? null,
      remoteCandidateType: remote?.candidateType ?? null,
      addressesDisclosed: Boolean(local?.address && remote?.address),
      lanAddressVerified: selectedKnown && Boolean(local.address && remote.address) && isLanAddress(local.address) && isLanAddress(remote.address),
      roundTripMs: Number.isFinite(pair?.currentRoundTripTime) ? pair.currentRoundTripTime * 1000 : null,
      framesPerSecond: inbound?.framesPerSecond ?? outbound?.framesPerSecond ?? null,
      framesDecoded: inbound?.framesDecoded ?? null,
      packetsLost: inbound?.packetsLost ?? null,
      jitterMs: Number.isFinite(inbound?.jitter) ? inbound.jitter * 1000 : null,
      frameWidth: inbound?.frameWidth ?? outbound?.frameWidth ?? null,
      frameHeight: inbound?.frameHeight ?? outbound?.frameHeight ?? null,
      atMs: performance.now()
    };
    this.onAudit(this.audit);
    if (invalid) {
      this.onState({ state: 'failed', message: 'LAN以外の接続経路が選択されたため、映像接続を停止しました。' });
      this.close();
    }
    return this.audit;
  }

  close() {
    if (this.closed) return;
    this.closed = true;
    clearInterval(this.auditInterval);
    clearInterval(this.clockInterval);
    for (const timer of this.timers) clearTimeout(timer);
    this.timers.clear();
    this.channel?.close();
    this.pc.close();
    for (const track of this.localStream?.getTracks() || []) track.stop();
  }
}
