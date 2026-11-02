import { LanPeer, encodeSignal, parseSignal } from './webrtc.mjs';

const $ = (id) => document.getElementById(id);
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const finitePosition = (p) => Array.isArray(p) && p.length >= 3 && p.slice(0, 3).every(Number.isFinite);
const numberInput = (id, fallback = 0) => {
  const value = $(id)?.value.trim();
  return value !== '' && Number.isFinite(Number(value)) ? Number(value) : fallback;
};
const median = (values) => {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  return sorted.length ? sorted[Math.floor(sorted.length / 2)] : null;
};
const COLORS = { observed: '#b9ef92', inferred: '#70d7df', predicted: '#edbd73', missing: '#617386' };
const SOURCE_LABELS = {
  measuredMultiView: '校正済み2視点の実測', estimatedMultiView: '仮配置による2視点推定',
  inferredMonocular: '単眼・人体モデルによる推定', predicted: '遮蔽中の短期予測',
  inferredKinematic: '周囲の関節から補完', inferredPrior: '過去の姿勢から補完',
  inferredAnatomy: '人体の形から補完', missing: '欠測', uncalibrated: '絶対位置は未校正'
};
let toastTimer;
function toast(message, error = false) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').classList.toggle('error', error);
  $('toast').hidden = false;
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, error ? 9000 : 4500);
}
function safe(action) {
  return async (...args) => { try { return await action(...args); } catch (error) { toast(error.message || String(error), true); } };
}
async function copyText(text) {
  if (!text) throw new Error('先に接続コードを作成してください。');
  try { await navigator.clipboard.writeText(text); }
  catch { throw new Error('自動コピーができません。コード欄を選択してコピーしてください。'); }
  toast('コピーしました。');
}
function downloadJson(filename, value) {
  const text = JSON.stringify(value, (_key, item) => typeof item === 'number' && !Number.isFinite(item) ? null : item, 2);
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json;charset=utf-8' }));
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
async function readJson(file) {
  if (!file || file.size > 2 * 1024 * 1024) throw new Error('校正JSONは2MB以内のファイルを指定してください。');
  const bytes = new Uint8Array(await file.arrayBuffer());
  if ((bytes[0] === 255 && bytes[1] === 254) || (bytes[0] === 254 && bytes[1] === 255)) throw new Error('UTF-16のファイルは読み込めません。UTF-8のJSONを指定してください。');
  return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
}

if (new URLSearchParams(location.search).get('mode') === 'camera') initSender();
else initReceiver().catch((error) => toast(error.message || String(error), true));

function initSender() {
  $('receiver-app').hidden = true;
  $('sender-app').hidden = false;
  $('camera-mode-link').textContent = 'PCの推論画面へ ↗';
  $('camera-mode-link').href = './';
  let peer = null;
  let stream = null;
  const initial = new URLSearchParams(location.hash.slice(1)).get('offer');
  if (initial) $('sender-offer').value = initial;
  const stop = () => {
    peer?.close(); peer = null;
    stream?.getTracks().forEach((track) => track.stop()); stream = null;
    $('sender-video').srcObject = null;
    $('sender-state').textContent = '停止しました';
    $('sender-start').disabled = false;
    $('sender-copy').disabled = true;
  };
  $('sender-start').onclick = safe(async () => {
    if (!isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error('カメラを使うにはHTTPSでこのページを開いてください。');
    const offer = parseSignal($('sender-offer').value);
    if (offer.type !== 'offer') throw new Error('PCが作った招待を入力してください。');
    stop();
    $('sender-start').disabled = true;
    $('sender-state').textContent = 'カメラの利用を許可してください';
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: {
        facingMode: { ideal: $('sender-facing').value },
        width: { ideal: clamp(numberInput('sender-width', 1280), 320, 3840) },
        frameRate: { ideal: clamp(numberInput('sender-fps', 60), 1, 120) }
      } });
      $('sender-video').srcObject = stream;
      await $('sender-video').play();
      const settings = stream.getVideoTracks()[0].getSettings();
      $('sender-info').textContent = `実際の撮影設定：${settings.width || '—'} × ${settings.height || '—'}、${settings.frameRate?.toFixed(1) || '—'} fps。画面を開いたまま使用してください。`;
      peer = new LanPeer({ sourceId: offer.sourceId,
        onState: ({ state, message }) => { $('sender-state').textContent = state === 'connected' ? `入力 ${offer.sourceId.toUpperCase()} へ送信中` : message; },
        onAudit: (audit) => { $('sender-audit').textContent = JSON.stringify(audit, null, 2); }
      });
      $('sender-state').textContent = 'LANの接続候補を集めています';
      const answer = await peer.acceptOffer(offer, stream);
      $('sender-answer').value = encodeSignal(answer);
      $('sender-copy').disabled = false;
      $('sender-state').textContent = '応答をPCへ貼り付けると、映像の送信を開始します';
    } catch (error) {
      stop();
      $('sender-state').textContent = error.message || String(error);
      throw error;
    }
  });
  $('sender-stop').onclick = stop;
  $('sender-copy').onclick = safe(() => copyText($('sender-answer').value));
  addEventListener('pagehide', stop);
}

async function initReceiver() {
  const state = {
    engine: null, engineModule: null, geometry: null, calibrationModule: null, tracker: null,
    running: false, loading: false, processing: false, generation: 0,
    sources: new Map(), observations: new Map(), cameras: [], fused: null,
    records: [], totalFrames: 0, recentTimes: [], latencies: [], lastObservationAt: 0,
    lastFusedTimestamp: 0, lastMissingTick: 0, sceneDirty: true,
    selectedModel: null, backend: null, offline: false, loadedOffline: null, networkAudit: null, startedAt: new Date().toISOString(),
    view: { yaw: .33, pitch: .17, zoom: 1, center: [0, 0, 0], extent: 1, initialized: false, metric: false }
  };
  let modulesPromise;
  const ensureModules = () => {
    if (!modulesPromise) modulesPromise = Promise.all([import('./inference.mjs'), import('./geometry.mjs'), import('./calibration.mjs')]).then(([engineModule, geometry, calibrationModule]) => {
      state.engineModule = engineModule; state.geometry = geometry; state.calibrationModule = calibrationModule;
      state.tracker = new geometry.TemporalPoseTracker(trackerOptions());
      const previous = $('model-select').value;
      const models = Array.isArray(engineModule.BODY_MODELS) ? engineModule.BODY_MODELS : Object.entries(engineModule.BODY_MODELS).map(([id, value]) => ({ id, ...value }));
      $('model-select').replaceChildren(...models.map((model) => {
        const option = document.createElement('option'); option.value = model.id; option.textContent = model.label || model.id; return option;
      }));
      $('model-select').value = models.some((m) => m.id === previous) ? previous : models.some((m) => m.id === 'rtmw-l') ? 'rtmw-l' : models[0]?.id || '';
    }).catch((error) => { modulesPromise = null; throw error; });
    return modulesPromise;
  };

  function engineStatus(kind, text, detail) {
    const dot = document.createElement('i'); dot.className = `status-dot${kind === 'idle' ? ' idle' : kind === 'error' ? ' error' : ''}`;
    $('engine-status').replaceChildren(dot, document.createTextNode(text));
    if (detail) $('status-message').textContent = detail;
  }
  function trackerOptions() { return { predictionMs: clamp(numberInput('prediction-ms', 180), 0, 500), holdMs: clamp(numberInput('hold-ms', 1200), 0, 5000), priorMs: Math.max(1200, clamp(numberInput('hold-ms', 1200), 0, 5000) + 1800) }; }
  function fusionOptions() {
    return { minScore: numberInput('score-threshold', .25), syncToleranceMs: numberInput('sync-tolerance-ms', 45),
      maxTimeUncertaintyMs: 20, maxReprojectionErrorPx: 8, bodyHeightM: numberInput('body-height', 1.7),
      scaleConfirmed: $('height-confirmed').checked, rootDepthM: numberInput('root-depth', undefined),
      priorPose: state.tracker?.lastPose, boneLengths: state.tracker?.getBoneLengths() || {} };
  }

  function createSource(id) {
    const source = { id, video: $(`source-${id}-video`), overlay: $(`source-${id}-overlay`), screen: $(`source-${id}-screen`),
      kind: null, epoch: 0, stream: null, peer: null, url: null, filename: null, fileIdentity: null,
      pending: null, pool: [], requestId: null, requestKind: null, roi: null, roiStart: null, selection: null,
      boardPoints: [], frameCount: 0, lastCaptureTime: null, frameIntervals: [], lastPresentationAt: 0,
      lastMediaTime: null, lastObservation: null, clock: null, audit: null, inviteLink: null, failed: false };
    state.sources.set(id, source);
    $(`source-${id}-file`).onchange = safe(async (event) => {
      const file = event.target.files?.[0];
      if (!file) return;
      resetSource(source);
      source.kind = 'video'; source.filename = file.name; source.fileIdentity = `${file.name}:${file.size}:${file.lastModified}`;
      source.url = URL.createObjectURL(file); source.video.src = source.url; source.video.controls = true;
      $(`source-${id}-state`).textContent = file.name;
      $(`source-${id}-empty`).hidden = true;
      $('video-sync-confirmed').checked = false;
      try { await source.video.play(); } catch { /* Native controls remain available. */ }
    });
    $(`source-${id}-camera`).onclick = safe(() => openLocalCamera(source));
    $(`source-${id}-device`).onchange = safe(async () => { if (source.kind === 'camera') await openLocalCamera(source); });
    $(`source-${id}-close`).onclick = () => { resetSource(source); updateDiagnostics(); };
    $(`source-${id}-remote`).onclick = () => { $(`source-${id}-pairing`).open = true; $(`source-${id}-offer`).focus(); };
    $(`source-${id}-offer`).onclick = safe(() => createInvite(source));
    $(`source-${id}-invite-copy`).onclick = safe(() => copyText(source.inviteLink));
    $(`source-${id}-answer-apply`).onclick = safe(async () => {
      if (!source.peer) throw new Error('先にこの入力で招待を作成してください。');
      await source.peer.acceptAnswer($(`source-${id}-answer-code`).value);
      $(`source-${id}-peer-state`).textContent = 'LAN接続を確立しています。';
    });
    $(`source-${id}-roi`).onclick = () => {
      if (!source.video.videoWidth) return toast('先に映像を選択してください。', true);
      source.selection = source.selection === 'roi' ? null : 'roi';
      source.screen.classList.toggle('picking', Boolean(source.selection));
      $(`source-${id}-roi`).textContent = source.selection ? 'ドラッグで指定' : '範囲を選ぶ';
    };
    $(`source-${id}-reset-roi`).onclick = () => { source.roi = null; source.selection = null; source.screen.classList.remove('picking'); $(`source-${id}-roi`).textContent = '範囲を選ぶ'; drawOverlay(source); };
    source.screen.addEventListener('pointerdown', (event) => beginSelection(source, event));
    source.screen.addEventListener('pointermove', (event) => moveSelection(source, event));
    source.screen.addEventListener('pointerup', (event) => finishSelection(source, event));
    source.screen.addEventListener('pointercancel', () => { source.roiStart = null; });
    source.video.addEventListener('loadedmetadata', () => {
      const { videoWidth: width, videoHeight: height } = source.video;
      $(`source-${id}-dimensions`).textContent = `${width} × ${height}`;
      source.overlay.width = width; source.overlay.height = height;
      startWatching(source);
      if (source.kind === 'video' && source.video.paused && state.running) captureFrame(source, performance.now(), { mediaTime: source.video.currentTime });
    });
    source.video.addEventListener('seeked', () => {
      state.tracker?.reset(); state.observations.clear();
      source.pending = null;
      if (state.running && source.kind === 'video') captureFrame(source, performance.now(), { mediaTime: source.video.currentTime });
    });
    source.video.addEventListener('error', () => {
      $(`source-${id}-state`).textContent = '映像を読み込めません';
      toast(`入力 ${id.toUpperCase()}：この動画形式をブラウザで再生できません。`, true);
    });
    return source;
  }

  function resetSource(source, preservePeer = false) {
    source.epoch++;
    if (source.requestId !== null) {
      if (source.requestKind === 'video') source.video.cancelVideoFrameCallback?.(source.requestId);
      else cancelAnimationFrame(source.requestId);
    }
    source.requestId = null;
    source.video.pause(); source.video.srcObject = null; source.video.removeAttribute('src'); source.video.load();
    source.stream?.getTracks().forEach((track) => track.stop()); source.stream = null;
    if (!preservePeer) { source.peer?.close(); source.peer = null; }
    if (source.url) URL.revokeObjectURL(source.url);
    source.url = null; source.kind = null; source.pending = null; source.pool = []; source.lastObservation = null;
    source.roi = null; source.selection = null; source.roiStart = null; source.boardPoints = [];
    source.frameCount = 0; source.lastCaptureTime = null; source.frameIntervals = []; source.lastMediaTime = null;
    source.clock = null; source.audit = null; source.inviteLink = null; source.failed = false;
    state.observations.delete(source.id); state.tracker?.reset();
    source.overlay.getContext('2d').clearRect(0, 0, source.overlay.width, source.overlay.height);
    source.screen.classList.remove('picking');
    $(`source-${source.id}-empty`).hidden = false;
    $(`source-${source.id}-state`).textContent = '未選択';
    $(`source-${source.id}-timing`).textContent = 'フレーム待機';
    $(`source-${source.id}-dimensions`).textContent = '— × —';
    $(`source-${source.id}-quality`).textContent = '観測 — / 133';
    $(`source-${source.id}-invite-copy`).disabled = true;
    if (!Array.from(state.sources.values()).some((item) => item.kind)) {
      state.fused = null; state.sceneDirty = true; state.view.initialized = false;
      for (const axis of ['x', 'y', 'z']) $(`root-${axis}`).textContent = '—';
      $('scene-empty').hidden = false; $('root-status').textContent = '絶対位置は未校正';
      $('coordinate-state').textContent = '相対ポーズ · 待機中';
    }
  }

  async function openLocalCamera(source) {
    if (!isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error('カメラはHTTPSまたはlocalhostで使用できます。');
    const deviceId = $(`source-${source.id}-device`).value;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: {
      ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      width: { ideal: clamp(numberInput('capture-width', 1920), 320, 4096) },
      height: { ideal: clamp(numberInput('capture-height', 1080), 240, 2160) },
      frameRate: { ideal: clamp(numberInput('capture-fps', 60), 1, 120) }
    } });
    resetSource(source); source.kind = 'camera'; source.stream = stream;
    source.video.srcObject = stream; source.video.controls = false;
    $(`source-${source.id}-empty`).hidden = true;
    $(`source-${source.id}-state`).textContent = stream.getVideoTracks()[0].label || 'PCカメラ';
    await source.video.play();
    await refreshDevices();
  }

  async function refreshDevices() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((device) => device.kind === 'videoinput');
    for (const source of state.sources.values()) {
      const select = $(`source-${source.id}-device`);
      const previous = source.stream?.getVideoTracks()[0]?.getSettings().deviceId || select.value;
      const automatic = document.createElement('option'); automatic.value = ''; automatic.textContent = 'カメラの自動選択';
      select.replaceChildren(automatic, ...devices.map((device, i) => {
        const option = document.createElement('option'); option.value = device.deviceId; option.textContent = device.label || `カメラ ${i + 1}`; return option;
      }));
      if (devices.some((device) => device.deviceId === previous)) select.value = previous;
    }
  }

  async function createInvite(source) {
    const button = $(`source-${source.id}-offer`);
    button.disabled = true;
    resetSource(source); source.kind = 'peer';
    $(`source-${source.id}-peer-state`).textContent = 'LANの接続候補を集めています。';
    try {
      source.peer = new LanPeer({ sourceId: source.id,
        onStream: (stream) => {
          source.stream = stream; source.video.srcObject = stream; source.video.controls = false;
          $(`source-${source.id}-empty`).hidden = true;
          $(`source-${source.id}-state`).textContent = 'スマホの映像';
          source.video.play().catch(() => { source.video.controls = true; toast('受信映像の再生ボタンを押してください。'); });
        },
        onState: ({ state: connection, message }) => { $(`source-${source.id}-peer-state`).textContent = message; if (connection === 'failed' || connection === 'closed') source.failed = true; updateDiagnostics(); },
        onClock: (clock) => { source.clock = clock; updateDiagnostics(); },
        onAudit: (audit) => { source.audit = audit; updateDiagnostics(); }
      });
      const offer = await source.peer.makeOffer();
      const code = encodeSignal(offer);
      $(`source-${source.id}-offer-code`).value = code;
      $(`source-${source.id}-answer-code`).value = '';
      const invite = new URL(location.href); invite.search = '?mode=camera'; invite.hash = `offer=${code}`;
      source.inviteLink = invite.href;
      $(`source-${source.id}-invite-copy`).disabled = false;
      $(`source-${source.id}-peer-state`).textContent = '招待をスマホで開いてください。';
    } catch (error) { source.peer?.close(); source.peer = null; throw error; }
    finally { button.disabled = false; }
  }

  function startWatching(source) {
    if (source.requestId !== null) {
      if (source.requestKind === 'video') source.video.cancelVideoFrameCallback?.(source.requestId);
      else cancelAnimationFrame(source.requestId);
    }
    const epoch = source.epoch;
    const callback = (now, metadata) => {
      if (source.epoch !== epoch || !source.kind) return;
      source.requestId = null;
      const frame = metadata || { mediaTime: source.video.currentTime, presentationTime: now };
      if (metadata || source.lastMediaTime !== frame.mediaTime) {
        source.lastPresentationAt = now; source.frameCount++;
        source.lastMediaTime = frame.mediaTime;
        if (state.running) captureFrame(source, now, frame);
      }
      queue();
    };
    const queue = () => {
      if (source.video.requestVideoFrameCallback) { source.requestKind = 'video'; source.requestId = source.video.requestVideoFrameCallback(callback); }
      else { source.requestKind = 'raf'; source.requestId = requestAnimationFrame(callback); }
    };
    queue();
  }

  function frameTiming(source, now, metadata) {
    if (source.kind === 'video') {
      const confirmed = $('video-sync-confirmed').checked;
      return { timestampMs: (metadata.mediaTime ?? source.video.currentTime) * 1000 + (source.id === 'b' ? numberInput('video-offset-ms', 0) : 0),
        timeUncertaintyMs: confirmed ? .5 : Infinity, timeDomain: 'media', timeSource: confirmed ? 'manually-aligned-media-time' : 'unaligned-media-time', mediaTimeMs: (metadata.mediaTime ?? source.video.currentTime) * 1000 };
    }
    const capture = metadata.captureTime;
    if (Number.isFinite(capture) && capture > 0 && capture <= now + 1000) {
      if (source.lastCaptureTime !== null && capture > source.lastCaptureTime) {
        const interval = capture - source.lastCaptureTime;
        if (interval < 500) { source.frameIntervals.push(interval); if (source.frameIntervals.length > 30) source.frameIntervals.shift(); }
      }
      source.lastCaptureTime = capture;
      // A frame-cadence bound is conservative; the browser does not expose a guaranteed error bound.
      const cadenceBound = (median(source.frameIntervals) || 33.34) / 2;
      const clockBound = source.kind === 'peer' ? (source.clock?.uncertaintyMs ?? Infinity) : 0;
      return { timestampMs: capture, timeUncertaintyMs: Math.max(cadenceBound, clockBound), timeDomain: 'capture',
        timeSource: 'browser-capture-time-estimate', uncertaintyBasis: 'half-frame-cadence-and-clock-rtt-not-a-measured-guarantee',
        receiveTimeMs: metadata.receiveTime ?? null, rtpTimestamp: metadata.rtpTimestamp ?? null, mediaTimeMs: (metadata.mediaTime ?? 0) * 1000 };
    }
    return { timestampMs: now, timeUncertaintyMs: Infinity, timeDomain: 'capture', timeSource: 'presentation-only',
      receiveTimeMs: metadata.receiveTime ?? null, rtpTimestamp: metadata.rtpTimestamp ?? null, mediaTimeMs: (metadata.mediaTime ?? 0) * 1000 };
  }

  function captureFrame(source, now, metadata) {
    const { videoWidth: width, videoHeight: height, readyState } = source.video;
    if (!width || !height || readyState < 2 || source.failed || source.selection === 'board') return;
    let canvas = source.pending?.canvas || source.pool.pop();
    if (!canvas) canvas = document.createElement('canvas');
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    canvas.getContext('2d', { alpha: false }).drawImage(source.video, 0, 0, width, height);
    source.pending = { canvas, epoch: source.epoch, capturedAtMs: now, frameId: source.frameCount, ...frameTiming(source, now, metadata) };
    queueMicrotask(processFrames);
  }

  async function processFrames() {
    if (!state.running || !state.engine || state.processing) return;
    const batch = Array.from(state.sources.values()).filter((source) => source.pending).map((source) => {
      const frame = source.pending; source.pending = null; return { source, frame };
    });
    if (!batch.length) return;
    state.processing = true;
    const generation = state.generation;
    const batchObservations = [];
    try {
      // Freeze both current inputs before running the single shared model sequentially.
      for (const { source, frame } of batch) {
        if (!state.running || generation !== state.generation || source.epoch !== frame.epoch) continue;
        const result = await state.engine.estimate(frame.canvas, { sourceId: source.id, timestampMs: frame.timestampMs,
          roi: source.roi?.slice() || null, handRefinement: $('hand-refinement').checked });
        if (generation !== state.generation || source.epoch !== frame.epoch) continue;
        const finishedAtMs = performance.now();
        const observation = { ...result, sourceId: source.id, timestampMs: frame.timestampMs, timeUncertaintyMs: frame.timeUncertaintyMs,
          timeDomain: frame.timeDomain, timeSource: frame.timeSource, uncertaintyBasis: frame.uncertaintyBasis,
          frameId: frame.frameId, mediaTimeMs: frame.mediaTimeMs, captureReceivedAtMs: frame.capturedAtMs,
          receiveTimeMs: frame.receiveTimeMs, rtpTimestamp: frame.rtpTimestamp, inferredAtMs: finishedAtMs,
          localPipelineLatencyMs: finishedAtMs - frame.capturedAtMs };
        source.lastObservation = observation;
        state.observations.set(source.id, observation);
        state.totalFrames++; state.recentTimes.push(finishedAtMs);
        while (state.recentTimes[0] < finishedAtMs - 3000) state.recentTimes.shift();
        if (Number.isFinite(result.timings?.totalMs)) state.latencies.push(result.timings.totalMs);
        if (state.latencies.length > 300) state.latencies.shift();
        state.lastObservationAt = finishedAtMs;
        drawOverlay(source);
        updateDetailViews(frame.canvas, observation);
        batchObservations.push(observation);
      }
      if (batchObservations.length && generation === state.generation) {
        // Commit one fused result per frozen input pair. Otherwise the later-arriving
        // view can share the first view's timestamp and be ignored by the time filter.
        const reference = batchObservations.reduce((latest, item) => item.timestampMs > latest.timestampMs ? item : latest);
        updateFusion(reference);
        batchObservations.forEach((observation, index) => {
          state.records.push({ observation, fused: index === batchObservations.length - 1 ? state.fused : null });
          if (state.records.length > 1800) state.records.shift();
        });
        $('load-progress').hidden = true;
        engineStatus('running', '推論中', '観測を更新しています。モデルは入力A・Bで共有します。');
        updateMetrics();
      }
    } catch (error) {
      state.running = false;
      $('start-inference').disabled = false; $('stop-inference').disabled = true;
      engineStatus('error', '推論停止', error.message || String(error));
      toast(error.message || String(error), true);
    } finally {
      for (const { source, frame } of batch) if (source.epoch === frame.epoch && source.pool.length < 2) source.pool.push(frame.canvas);
      state.processing = false;
      if (state.running && Array.from(state.sources.values()).some((source) => source.pending)) setTimeout(processFrames, 0);
    }
  }

  function updateFusion(observation) {
    if (!state.geometry || !state.tracker) return;
    const tolerance = numberInput('sync-tolerance-ms', 45);
    let candidates = Array.from(state.observations.values()).filter((item) => item.timeDomain === observation.timeDomain && Math.abs(item.timestampMs - observation.timestampMs) <= Math.max(150, tolerance * 3));
    const a = state.sources.get('a'); const b = state.sources.get('b');
    if (observation.timeDomain === 'media' && (!$('video-sync-confirmed').checked || (a.fileIdentity && a.fileIdentity === b.fileIdentity))) candidates = [observation];
    const fused = state.geometry.fusePose(candidates, state.cameras, { ...fusionOptions(), timestampMs: observation.timestampMs });
    state.fused = state.tracker.update(fused, observation.timestampMs);
    state.lastFusedTimestamp = observation.timestampMs;
    state.sceneDirty = true;
    updatePoseReadout();
  }

  function currentEdges() {
    if (!state.engineModule) return [];
    return [...(state.engineModule.BODY_EDGES || []), ...(state.engineModule.HAND_EDGES || []), ...(state.engineModule.FOOT_EDGES || [])];
  }
  function visible(point) { return point && Number.isFinite(point.x) && Number.isFinite(point.y) && point.score >= numberInput('score-threshold', .25) && point.observable !== false; }
  function colorFor(source) {
    if (source === 'measuredMultiView') return COLORS.observed;
    if (['inferredMonocular', 'estimatedMultiView'].includes(source)) return COLORS.inferred;
    if (source && source !== 'missing') return COLORS.predicted;
    return COLORS.missing;
  }
  function drawOverlay(source) {
    const canvas = source.overlay; const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const observation = source.lastObservation;
    const scale = Math.max(1, canvas.width / 720);
    if (observation) {
      const points = observation.keypoints;
      ctx.lineWidth = 1.7 * scale;
      for (const [a, b] of currentEdges()) {
        const p = points[a], q = points[b]; if (!visible(p) || !visible(q)) continue;
        ctx.strokeStyle = a >= 91 || b >= 91 ? COLORS.inferred : a >= 17 && a <= 22 ? COLORS.predicted : COLORS.observed;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
      }
      for (let i = 0; i < points.length; i++) {
        if (i >= 23 && i <= 90 && !$('show-face').checked) continue;
        const p = points[i]; if (!visible(p)) continue;
        ctx.fillStyle = i >= 91 ? COLORS.inferred : i >= 17 && i <= 22 ? COLORS.predicted : COLORS.observed;
        ctx.beginPath(); ctx.arc(p.x, p.y, (i >= 23 ? 1.8 : 3) * scale, 0, Math.PI * 2); ctx.fill();
      }
      const count = points.filter(visible).length;
      $(`source-${source.id}-quality`).textContent = `観測 ${count} / 133`;
      const ms = observation.timings?.totalMs;
      const timing = observation.timeSource === 'presentation-only' ? '撮影時刻なし' : observation.timeDomain === 'media' ? `${(observation.mediaTimeMs / 1000).toFixed(2)} s` : '撮影時計';
      $(`source-${source.id}-timing`).textContent = `${Number.isFinite(ms) ? ms.toFixed(0) + ' ms · ' : ''}${timing}`;
    }
    if (source.roi) {
      ctx.strokeStyle = '#edbd73'; ctx.lineWidth = 2 * scale; ctx.setLineDash([7 * scale, 5 * scale]);
      ctx.strokeRect(...source.roi); ctx.setLineDash([]);
    }
    if (source.boardPoints.length) {
      ctx.lineWidth = 2 * scale; ctx.strokeStyle = '#cba6f7'; ctx.fillStyle = '#cba6f7';
      source.boardPoints.forEach(([x, y], i) => { ctx.beginPath(); ctx.arc(x, y, 5 * scale, 0, Math.PI * 2); ctx.fill(); ctx.font = `${16 * scale}px sans-serif`; ctx.fillText(String(i + 1), x + 8 * scale, y - 8 * scale); });
      ctx.beginPath(); source.boardPoints.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
      if (source.boardPoints.length === 4) ctx.closePath(); ctx.stroke();
    }
  }

  function sourcePoint(source, event) {
    const rect = source.screen.getBoundingClientRect(), width = source.video.videoWidth, height = source.video.videoHeight;
    const scale = Math.min(rect.width / width, rect.height / height);
    return [clamp((event.clientX - rect.left - (rect.width - width * scale) / 2) / scale, 0, width), clamp((event.clientY - rect.top - (rect.height - height * scale) / 2) / scale, 0, height)];
  }
  function beginSelection(source, event) {
    if (!source.selection || !source.video.videoWidth) return;
    event.preventDefault();
    const point = sourcePoint(source, event);
    if (source.selection === 'board') {
      if (source.boardPoints.length < 4) source.boardPoints.push(point);
      $('board-state').textContent = `${source.id.toUpperCase()}：${source.boardPoints.length} / 4 点を選択`;
      if (source.boardPoints.length === 4) { source.selection = null; source.screen.classList.remove('picking'); }
      drawOverlay(source); return;
    }
    source.roiStart = point; source.screen.setPointerCapture(event.pointerId);
  }
  function moveSelection(source, event) {
    if (!source.roiStart) return;
    const end = sourcePoint(source, event), start = source.roiStart;
    source.roi = [Math.min(start[0], end[0]), Math.min(start[1], end[1]), Math.abs(end[0] - start[0]), Math.abs(end[1] - start[1])];
    drawOverlay(source);
  }
  function finishSelection(source, event) {
    if (!source.roiStart) return;
    moveSelection(source, event); source.roiStart = null;
    if (source.roi[2] < 12 || source.roi[3] < 12) { source.roi = null; toast('人物の範囲を、もう少し大きく選択してください。', true); }
    source.selection = null; source.screen.classList.remove('picking');
    $(`source-${source.id}-roi`).textContent = '範囲を選ぶ'; drawOverlay(source);
  }

  function updateDetailViews(image, observation) {
    const parts = [ ['left-hand-canvas', 'left-hand-score', 91, 112], ['right-hand-canvas', 'right-hand-score', 112, 133], ['feet-canvas', 'feet-score', 17, 23] ];
    for (const [canvasId, label, start, end] of parts) {
      const canvas = $(canvasId), ctx = canvas.getContext('2d');
      ctx.fillStyle = '#0a121a'; ctx.fillRect(0, 0, canvas.width, canvas.height);
      const pts = observation.keypoints.slice(start, end).filter(visible);
      $(label).textContent = `${pts.length} / ${end - start}`;
      if (pts.length < 2) { ctx.fillStyle = '#6e8a9e'; ctx.font = '14px sans-serif'; ctx.textAlign = 'center'; ctx.fillText('観測できません', canvas.width / 2, canvas.height / 2); continue; }
      const minX = Math.min(...pts.map((p) => p.x)), maxX = Math.max(...pts.map((p) => p.x)), minY = Math.min(...pts.map((p) => p.y)), maxY = Math.max(...pts.map((p) => p.y));
      const span = Math.max(maxX - minX, maxY - minY, 24) * 1.6;
      const cropW = span, cropH = span * canvas.height / canvas.width;
      const cropX = (minX + maxX) / 2 - cropW / 2, cropY = (minY + maxY) / 2 - cropH / 2;
      const scaleX = canvas.width / cropW, scaleY = canvas.height / cropH;
      ctx.save(); ctx.scale(scaleX, scaleY); ctx.translate(-cropX, -cropY); ctx.drawImage(image, 0, 0);
      ctx.lineWidth = 1.8 / Math.max(scaleX, .01); ctx.strokeStyle = start >= 91 ? COLORS.inferred : COLORS.predicted;
      for (const [a, b] of currentEdges()) {
        if (a < start || a >= end || b < start || b >= end) continue;
        const p = observation.keypoints[a], q = observation.keypoints[b];
        if (!visible(p) || !visible(q)) continue;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
      }
      ctx.fillStyle = start >= 91 ? COLORS.inferred : COLORS.predicted;
      for (const point of pts) { ctx.beginPath(); ctx.arc(point.x, point.y, 2.3 / Math.max(scaleX, .01), 0, Math.PI * 2); ctx.fill(); }
      ctx.restore();
    }
  }

  function updatePoseReadout() {
    const pose = state.fused; if (!pose) return;
    const valid = pose.root?.valid && pose.metric && finitePosition(pose.root.position);
    for (const [axis, i] of [['x', 0], ['y', 1], ['z', 2]]) $(`root-${axis}`).textContent = valid ? `${pose.root.position[i].toFixed(3)} m` : '—';
    const source = pose.root?.source || pose.root?.distanceStatus || 'uncalibrated';
    $('root-status').textContent = valid ? SOURCE_LABELS[source] || source : '絶対位置は未校正 · 骨盤を基準に相対表示';
    $('coordinate-state').textContent = pose.metric ? `空間座標 [m] · ${SOURCE_LABELS[source] || '推定'}` : '相対ポーズ · 絶対距離は未校正';
    $('scene-empty').hidden = pose.joints.some((joint) => finitePosition(joint.position));
    if (!state.view.initialized || state.view.metric !== Boolean(pose.metric)) initializeView();
    const diagnostic = pose.diagnostics || {};
    const synchronized = diagnostic.synchronizedSources?.length || 0;
    $('sync-state').textContent = synchronized >= 2 ? `${synchronized}視点 / 差 ${Number(diagnostic.maxSkewMs || 0).toFixed(1)} ms` : state.observations.size > 1 ? '未同期 / 単眼を使用' : '1視点';
  }

  function initializeView() {
    const pose = state.fused;
    state.view.yaw = .33; state.view.pitch = .17; state.view.zoom = 1;
    if (!pose) { state.view.center = [0, 0, 0]; state.view.extent = 1; state.view.initialized = false; return; }
    const positions = pose.joints.slice(0, 23).map((joint) => joint.position).filter(finitePosition);
    if (!positions.length) return;
    const min = [0, 1, 2].map((axis) => Math.min(...positions.map((point) => point[axis])));
    const max = [0, 1, 2].map((axis) => Math.max(...positions.map((point) => point[axis])));
    state.view.center = min.map((value, axis) => (value + max[axis]) / 2);
    state.view.extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], pose.metric ? 1.5 : .8) * 1.2;
    state.view.metric = Boolean(pose.metric); state.view.initialized = true; state.sceneDirty = true;
  }

  function drawScene() {
    const canvas = $('scene-canvas'), bounds = canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    const width = Math.round(bounds.width * dpr), height = Math.round(bounds.height * dpr);
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, width, height);
    const view = state.view, scale = Math.min(width, height) * .78 * view.zoom / view.extent;
    const project = (point) => {
      const x = point[0] - view.center[0], y = point[1] - view.center[1], z = point[2] - view.center[2];
      const rx = x * Math.cos(view.yaw) - z * Math.sin(view.yaw), rz = x * Math.sin(view.yaw) + z * Math.cos(view.yaw);
      const ry = y * Math.cos(view.pitch) - rz * Math.sin(view.pitch);
      return [width / 2 + rx * scale, height / 2 - ry * scale, rz];
    };
    const line = (from, to, color, strokeWidth = 1, dashed = false) => {
      const a = project(from), b = project(to);
      ctx.strokeStyle = color; ctx.lineWidth = strokeWidth * dpr; ctx.setLineDash(dashed ? [3 * dpr, 3 * dpr] : []);
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); ctx.setLineDash([]);
    };
    const pose = state.fused;
    if (pose?.metric) {
      const x0 = Math.floor(view.center[0]), z0 = Math.floor(view.center[2]);
      for (let i = -6; i <= 6; i++) {
        line([x0 - 3, 0, z0 + i / 2], [x0 + 3, 0, z0 + i / 2], '#28404d', .6);
        line([x0 + i / 2, 0, z0 - 3], [x0 + i / 2, 0, z0 + 3], '#28404d', .6);
      }
      line([0, 0, 0], [.6, 0, 0], '#b16c74', 1.2); line([0, 0, 0], [0, .6, 0], '#75ad7e', 1.2); line([0, 0, 0], [0, 0, .6], '#7595bd', 1.2);
    }
    if (!pose) return;
    for (const [a, b] of currentEdges()) {
      const p = pose.joints[a], q = pose.joints[b];
      if (!finitePosition(p?.position) || !finitePosition(q?.position)) continue;
      const extrapolated = !['measuredMultiView', 'estimatedMultiView', 'inferredMonocular'].includes(p.source) || !['measuredMultiView', 'estimatedMultiView', 'inferredMonocular'].includes(q.source);
      line(p.position, q.position, colorFor(extrapolated ? 'predicted' : p.source === 'measuredMultiView' && q.source === 'measuredMultiView' ? p.source : 'inferredMonocular'), a >= 91 ? 1.4 : 2.2, extrapolated);
    }
    for (const joint of pose.joints) {
      if (!finitePosition(joint.position) || (joint.id >= 23 && joint.id <= 90 && !$('show-face').checked)) continue;
      const point = project(joint.position);
      ctx.globalAlpha = Math.max(.25, Math.min(1, (joint.score || 0) * 1.5)); ctx.fillStyle = colorFor(joint.source);
      ctx.beginPath(); ctx.arc(point[0], point[1], (joint.id >= 23 ? 2.2 : 4) * dpr, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;
    for (const foot of Object.values(pose.feet || {})) {
      for (const toe of foot.toes || []) {
        for (let i = 1; i < (toe.joints?.length || 0); i++) if (finitePosition(toe.joints[i - 1].position) && finitePosition(toe.joints[i].position)) line(toe.joints[i - 1].position, toe.joints[i].position, '#edbd7399', 1, true);
      }
      if (finitePosition(foot.position) && finitePosition(foot.up)) line(foot.position, foot.position.map((value, i) => value + foot.up[i] * .13), '#cba6f7', 1.4);
    }
    if (pose.metric) for (const camera of state.cameras) {
      if (!Array.isArray(camera.Rcw) || !Array.isArray(camera.tcw)) continue;
      const position = [0, 1, 2].map((axis) => -camera.Rcw[axis] * camera.tcw[0] - camera.Rcw[3 + axis] * camera.tcw[1] - camera.Rcw[6 + axis] * camera.tcw[2]);
      const point = project(position); ctx.fillStyle = '#9db1c4'; ctx.fillRect(point[0] - 4 * dpr, point[1] - 3 * dpr, 8 * dpr, 6 * dpr);
      ctx.font = `${10 * dpr}px sans-serif`; ctx.fillText(camera.id.toUpperCase(), point[0] + 7 * dpr, point[1] + 3 * dpr);
    }
  }

  function updateMetrics() {
    const now = performance.now(), recent = state.recentTimes.filter((time) => time > now - 3000);
    const elapsed = recent.length > 1 ? (recent[recent.length - 1] - recent[0]) / 1000 : 0;
    const fps = elapsed > .1 ? (recent.length - 1) / elapsed : null;
    $('inference-fps').replaceChildren(document.createTextNode(fps === null || now - state.lastObservationAt > 1500 ? '—' : fps.toFixed(1)), Object.assign(document.createElement('small'), { textContent: ' fps' }));
    const sorted = [...state.latencies].sort((a, b) => a - b), p95 = sorted.length ? sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * .95) - 1)] : null;
    $('latency-p95').replaceChildren(document.createTextNode(p95 === null ? '—' : p95.toFixed(1)), Object.assign(document.createElement('small'), { textContent: ' ms' }));
    const joints = state.fused?.joints || [];
    const reprojection = median(joints.map((joint) => joint.reprojectionErrorPx));
    $('reprojection-error').replaceChildren(document.createTextNode(reprojection === null ? '—' : reprojection.toFixed(2)), Object.assign(document.createElement('small'), { textContent: ' px' }));
    const measured = joints.filter((joint) => joint.source === 'measuredMultiView').length;
    const inferred = joints.filter((joint) => finitePosition(joint.position) && joint.source !== 'measuredMultiView').length;
    $('joint-status-counts').textContent = `${measured} / ${inferred} / ${joints.length ? joints.length - measured - inferred : '—'}`;
    $('frame-count').textContent = `${state.totalFrames.toLocaleString()} フレーム`;
    const observations = Array.from(state.observations.values()).filter((item) => {
      const source = state.sources.get(item.sourceId);
      return source?.kind === 'video' || (now - item.inferredAtMs <= 1500 && now - source.lastPresentationAt <= 500);
    });
    for (const [id, start, end] of [['body-quality', 0, 17], ['hand-quality', 91, 133], ['foot-quality', 17, 23]]) {
      const count = Array.from({ length: end - start }, (_, i) => observations.some((item) => visible(item.keypoints[i + start]))).filter(Boolean).length;
      $(id).textContent = `${count} / ${end - start}`;
    }
    const memory = performance.memory?.usedJSHeapSize;
    $('memory-readout').textContent = `${state.backend || '未読込'}${Number.isFinite(memory) ? ` · JS ${(memory / 1024 / 1024).toFixed(0)} MB` : ''}`;
    $('recording-state').textContent = `直近 ${state.records.length.toLocaleString()} / 1,800回分を保持。映像は含みません。`;
    updateDiagnostics();
  }

  function updateDiagnostics() {
    $('diagnostics').textContent = JSON.stringify({ model: state.selectedModel, backend: state.backend, inputCount: Array.from(state.sources.values()).filter((source) => source.kind).length,
      offline: state.offline, networkAudit: state.networkAudit,
      calibration: state.cameras.map((camera) => ({ id: camera.id, calibrated: camera.calibrated, method: camera.calibrationMethod })),
      inputs: Array.from(state.sources.values()).filter((source) => source.kind).map((source) => ({ id: source.id, kind: source.kind,
        frameSize: [source.video.videoWidth, source.video.videoHeight], frameTime: source.lastObservation?.timeSource,
        timeUncertaintyMs: Number.isFinite(source.lastObservation?.timeUncertaintyMs) ? source.lastObservation.timeUncertaintyMs : null,
        clock: source.clock, route: source.audit, capture: source.stream?.getVideoTracks()[0]?.getSettings() ? (() => {
          const settings = source.stream.getVideoTracks()[0].getSettings(); return { width: settings.width, height: settings.height, frameRate: settings.frameRate };
        })() : null })), geometry: state.fused?.diagnostics || null }, null, 2);
  }

  function setCamera(camera) {
    state.cameras = [...state.cameras.filter((current) => current.id !== camera.id), camera];
    state.tracker?.reset(); state.view.initialized = false;
    $('calibration-state').textContent = `${state.cameras.filter((current) => current.calibrated).length} 台を校正`;
    updateDiagnostics();
  }
  function bindCalibration() {
    $('calibration-file').onchange = safe(async (event) => {
      await ensureModules();
      const value = await readJson(event.target.files?.[0]), result = state.calibrationModule.validateCalibration(value);
      if (!result.valid) throw new Error(result.errors.join('\n'));
      if (result.cameras.some((camera) => !['a', 'b'].includes(camera.id))) throw new Error('校正のidは入力に合わせて "a" または "b" にしてください。');
      state.cameras = result.cameras; state.tracker?.reset(); state.view.initialized = false;
      $('calibration-state').textContent = `${state.cameras.filter((camera) => camera.calibrated).length} 台を校正`;
      toast('校正ファイルを読み込みました。'); updateDiagnostics();
    });
    $('export-calibration').onclick = () => downloadJson('wholebody-calibration.json', { schemaVersion: 1, cameras: state.cameras, bodyHeightM: numberInput('body-height', 1.7), scaleConfirmed: $('height-confirmed').checked });
    $('apply-manual-camera').onclick = safe(async () => {
      await ensureModules();
      const source = state.sources.get($('manual-camera-source').value);
      if (!source.video.videoWidth) throw new Error('設定する入力の映像を先に開いてください。');
      const camera = state.calibrationModule.makeCamera({ id: source.id, imageSize: { width: source.video.videoWidth, height: source.video.videoHeight },
        horizontalFovDeg: numberInput('manual-fov', 70), position: ['manual-x', 'manual-y', 'manual-z'].map((id) => numberInput(id)),
        target: ['target-x', 'target-y', 'target-z'].map((id) => numberInput(id)), intrinsicsCalibrated: false });
      setCamera(camera); toast('仮配置を適用しました。結果は推定として表示します。');
    });
    $('pick-board').onclick = () => {
      const source = state.sources.get($('board-source').value);
      if (!source.video.videoWidth) return toast('校正する入力の映像を先に開いてください。', true);
      for (const other of state.sources.values()) { other.selection = null; other.screen.classList.remove('picking'); }
      source.video.pause(); source.selection = 'board'; source.boardPoints = []; source.screen.classList.add('picking'); source.pending = null;
      const camera = state.cameras.find((item) => item.id === source.id);
      if (camera?.K?.length === 4) ['board-fx', 'board-fy', 'board-cx', 'board-cy'].forEach((id, i) => { if (!$(id).value) $(id).value = camera.K[i].toFixed(3); });
      if (!$('board-cx').value) $('board-cx').value = String(source.video.videoWidth / 2);
      if (!$('board-cy').value) $('board-cy').value = String(source.video.videoHeight / 2);
      $('board-state').textContent = `${source.id.toUpperCase()}：左上 → 右上 → 右下 → 左下の順に選択`;
      drawOverlay(source); source.screen.scrollIntoView({ behavior: 'smooth', block: 'center' });
    };
    $('solve-board').onclick = safe(async () => {
      await ensureModules();
      const source = state.sources.get($('board-source').value);
      if (source.boardPoints.length !== 4) throw new Error('同じ長方形の4点を選択してください。');
      const width = numberInput('board-width'), height = numberInput('board-height');
      if (!(width > 0 && height > 0)) throw new Error('板の実際の幅と高さを入力してください。');
      const K = ['board-fx', 'board-fy', 'board-cx', 'board-cy'].map((id) => numberInput(id, NaN));
      if (!K.every(Number.isFinite) || K[0] <= 0 || K[1] <= 0) throw new Error('このカメラの内部パラメータfx, fy, cx, cyを入力してください。');
      const distortion = $('board-distortion').value.split(/[\s,]+/).filter(Boolean).map(Number);
      if (distortion.length !== 5 || !distortion.every(Number.isFinite)) throw new Error('歪みは5個の数値で入力してください。');
      const camera = state.calibrationModule.solvePlanarExtrinsics({ id: source.id, imageSize: { width: source.video.videoWidth, height: source.video.videoHeight }, K, distortion,
        imagePoints: source.boardPoints, worldPoints: [[0, 0], [width, 0], [width, height], [0, height]],
        worldOrigin: [0, height, 0], worldAxisX: [1, 0, 0], worldAxisY: [0, -1, 0], intrinsicsCalibrated: $('intrinsics-confirmed').checked });
      if (!Number.isFinite(camera.reprojectionRmsPx) || camera.reprojectionRmsPx > 8) throw new Error('4点と内部パラメータが整合しません。点の順序とKを確認してください。');
      if (!camera.calibrated) camera.calibrationMethod = 'manualEstimate';
      camera.worldCalibrationId = `board-${width}-${height}-vertical-v1`;
      setCamera(camera); source.selection = null; source.screen.classList.remove('picking');
      $('board-state').textContent = `${source.id.toUpperCase()}：再投影RMS ${camera.reprojectionRmsPx.toFixed(2)} px / ${camera.calibrated ? '校正適用' : '内部校正が未確認のため推定用'}`;
      await source.video.play(); toast(camera.calibrated ? '校正を適用しました。もう1台でも同じ板を指定してください。' : '推定用の配置を適用しました。');
    });
  }

  createSource('a'); createSource('b'); bindCalibration();
  $('start-inference').onclick = safe(async () => {
    if (state.loading) return;
    if (state.processing) throw new Error('前の推論処理の終了を待って、もう一度開始してください。');
    state.loading = true; state.running = false; state.generation++;
    $('start-inference').disabled = true; $('stop-inference').disabled = true;
    try {
      await ensureModules();
      const bodyModel = $('model-select').value, backend = $('backend-select').value;
      if (!state.engine || bodyModel !== state.selectedModel || backend !== state.backend || state.loadedOffline !== state.offline) {
        await state.engine?.dispose?.();
        state.engine = new state.engineModule.PoseEngine({ onProgress: ({ stage, progress, message }) => {
          engineStatus('loading', '読込中', message || stage || 'モデルを読み込んでいます');
          $('load-progress').hidden = !Number.isFinite(progress);
          if (Number.isFinite(progress)) $('load-progress').value = clamp(progress > 1 ? progress / 100 : progress, 0, 1);
        } });
        await state.engine.load({ bodyModel, handRefinement: $('hand-refinement').checked, backend, offline: state.offline });
        state.selectedModel = bodyModel; state.backend = backend; state.loadedOffline = state.offline;
      }
      state.tracker = new state.geometry.TemporalPoseTracker(trackerOptions());
      state.observations.clear(); state.latencies = []; state.recentTimes = []; state.running = true;
      $('load-progress').hidden = true; $('stop-inference').disabled = false;
      $('start-inference').textContent = '推論を再開';
      engineStatus('running', '推論中', '観測を更新しています。モデルは入力A・Bで共有します。');
      for (const source of state.sources.values()) if (source.video.readyState >= 2 && source.video.paused && source.kind === 'video' && source.selection !== 'board') captureFrame(source, performance.now(), { mediaTime: source.video.currentTime });
      processFrames(); updateDiagnostics();
    } catch (error) {
      state.running = false;
      const failedEngine = state.engine; state.engine = null; state.selectedModel = null;
      try { await failedEngine?.dispose?.(); } catch { /* Preserve the original initialization error. */ }
      engineStatus('error', '読込失敗', error.message || String(error)); throw error;
    }
    finally { state.loading = false; $('start-inference').disabled = state.running; }
  });
  $('stop-inference').onclick = () => {
    state.running = false; state.generation++;
    $('start-inference').disabled = false; $('stop-inference').disabled = true;
    engineStatus('idle', '停止中', '映像は保持しています。推論を再開できます。');
  };
  $('prepare-offline').onclick = safe(async () => {
    if (state.offline) throw new Error('オフライン準備はオンライン配布モードで実行してください。');
    const button = $('prepare-offline'); button.disabled = true;
    $('offline-state').textContent = 'アプリと実行環境を保存しています…';
    try {
      const offline = await import('./offline.mjs');
      const result = await offline.prepareOffline();
      $('offline-state').textContent = 'アプリを保存しました。使うモデルもオンラインで一度読み込んでください。';
      state.networkAudit = await offline.networkAudit();
      $('network-audit').textContent = JSON.stringify({ preparation: result, ...state.networkAudit }, null, 2);
    } finally { button.disabled = false; }
  });
  $('closed-mode').onchange = safe(async () => {
    const enabled = $('closed-mode').checked;
    $('closed-mode').disabled = true;
    if (state.running) $('stop-inference').click();
    try {
      const offline = await import('./offline.mjs');
      await offline.setClosedMode(enabled);
      state.offline = enabled;
      $('offline-state').textContent = enabled ? '閉域モード：保存済みのアプリ・モデルだけで実行します。' : 'オンライン配布モード';
      state.networkAudit = await offline.networkAudit();
      $('network-audit').textContent = JSON.stringify(state.networkAudit, null, 2);
      updateDiagnostics();
    } catch (error) { $('closed-mode').checked = !enabled; throw error; }
    finally { $('closed-mode').disabled = false; }
  });
  $('audit-network').onclick = safe(async () => {
    const offline = await import('./offline.mjs');
    state.networkAudit = await offline.networkAudit();
    $('network-audit').textContent = JSON.stringify({ ...state.networkAudit, webRtc: Array.from(state.sources.values()).filter((source) => source.peer).map((source) => ({ sourceId: source.id, route: source.audit })) }, null, 2);
    updateDiagnostics();
  });
  $('score-threshold').oninput = () => { $('score-threshold-label').value = numberInput('score-threshold', .25).toFixed(2); for (const source of state.sources.values()) drawOverlay(source); };
  $('show-face').onchange = () => { for (const source of state.sources.values()) drawOverlay(source); state.sceneDirty = true; };
  for (const id of ['prediction-ms', 'hold-ms']) $(id).onchange = () => { if (state.tracker) state.tracker.options = { ...state.tracker.options, ...trackerOptions() }; };
  for (const id of ['body-height', 'height-confirmed', 'root-depth', 'video-offset-ms', 'video-sync-confirmed']) $(id).onchange = () => { state.tracker?.reset(); state.observations.clear(); };
  $('align-video-now').onclick = () => {
    const a = state.sources.get('a'), b = state.sources.get('b');
    if (a.kind !== 'video' || b.kind !== 'video') return toast('2本の同時撮影動画を開いてください。', true);
    if (a.fileIdentity === b.fileIdentity) return toast('同じ動画の複製は2視点として使えません。', true);
    $('video-offset-ms').value = String(Math.round((a.video.currentTime - b.video.currentTime) * 1000));
    $('video-sync-confirmed').checked = true; state.tracker?.reset(); state.observations.clear();
    toast('現在表示している2フレームを同時刻として記録しました。');
  };
  $('export-session').onclick = () => {
    const { priorPose, ...options } = fusionOptions();
    downloadJson(`wholebody-session-${new Date().toISOString().replace(/[:.]/g, '-')}.json`, {
      schemaVersion: 1, application: 'Wholebody Studio / Phase 2', exportedAt: new Date().toISOString(), startedAt: state.startedAt,
      model: state.selectedModel, backend: state.backend, handRefinement: $('hand-refinement').checked, offline: state.offline, networkAudit: state.networkAudit,
      cameras: state.cameras, options, trackerOptions: trackerOptions(), totalFrames: state.totalFrames,
      timingNotes: 'captureTime is a browser estimate. Null uncertainty means synchronization was not established. mediaTime is only aligned when explicitly confirmed.',
      footNotes: 'Big toe, little toe, heel are model landmarks. Other toes and toe articulation are anatomical priors, not observed independent toe joints.',
      sources: Array.from(state.sources.values()).map((source) => ({ id: source.id, kind: source.kind, imageSize: { width: source.video.videoWidth, height: source.video.videoHeight }, roi: source.roi, clock: source.clock, networkAudit: source.audit })),
      records: state.records
    });
  };
  $('reset-3d-view').onclick = initializeView;
  const scene = $('scene-canvas'); let drag = null;
  scene.addEventListener('pointerdown', (event) => { drag = [event.clientX, event.clientY, state.view.yaw, state.view.pitch]; scene.setPointerCapture(event.pointerId); });
  scene.addEventListener('pointermove', (event) => { if (!drag) return; state.view.yaw = drag[2] + (event.clientX - drag[0]) * .007; state.view.pitch = clamp(drag[3] + (event.clientY - drag[1]) * .007, -1.35, 1.35); state.sceneDirty = true; });
  scene.addEventListener('pointerup', () => { drag = null; }); scene.addEventListener('pointercancel', () => { drag = null; });
  scene.addEventListener('wheel', (event) => { event.preventDefault(); state.view.zoom = clamp(state.view.zoom * Math.exp(-event.deltaY * .001), .25, 5); state.sceneDirty = true; }, { passive: false });
  new ResizeObserver(() => { state.sceneDirty = true; }).observe(scene);
  let lastMetrics = 0;
  function animate(now) {
    if (state.running && !state.processing && state.fused && state.lastObservationAt && now - state.lastObservationAt > 200 && now - state.lastMissingTick > 100) {
      const live = Array.from(state.sources.values()).some((source) => ['camera', 'peer'].includes(source.kind));
      if (live) {
        const missing = { ...state.fused, joints: state.fused.joints.map((joint) => ({ id: joint.id, position: null, score: 0, source: 'missing' })) };
        state.fused = state.tracker.update(missing, state.lastFusedTimestamp + now - state.lastObservationAt);
        state.lastMissingTick = now; state.sceneDirty = true; updatePoseReadout();
      }
    }
    if (state.sceneDirty) { drawScene(); state.sceneDirty = false; }
    if (now - lastMetrics > 1000) { updateMetrics(); lastMetrics = now; }
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
  refreshDevices().catch(() => {});
  navigator.mediaDevices?.addEventListener?.('devicechange', () => refreshDevices().catch(() => {}));
  addEventListener('pagehide', () => { state.running = false; for (const source of state.sources.values()) { source.peer?.close(); source.stream?.getTracks().forEach((track) => track.stop()); if (source.url) URL.revokeObjectURL(source.url); } });
  $('start-inference').disabled = true;
  try {
    const offline = await import('./offline.mjs');
    const savedMode = await offline.getClosedMode();
    state.offline = typeof savedMode === 'boolean' ? savedMode : savedMode?.closed === true;
    $('closed-mode').checked = state.offline;
    $('offline-state').textContent = state.offline ? '閉域モード：保存済みのアプリ・モデルだけで実行します。' : 'オンライン配布モード';
  } catch (error) {
    $('offline-state').textContent = `保存済みの通信設定を確認できません：${error.message || error}`;
  } finally { $('start-inference').disabled = false; }
  ensureModules().catch((error) => engineStatus('error', '初期化失敗', error.message || String(error)));
}
