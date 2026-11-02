import { PoseRuntime } from './pose-runtime.js';
import { PoseFusion, BODY_EDGES, HAND_EDGES, canonicalName } from './geometry.js';
import { readVideoFiles } from './media-input.js';

const $ = id => document.getElementById(id);
const MODEL_BASE = 'https://huggingface.co/RabbitDaisuke/Wholebody-Video-Tracking/resolve/48774ee3dc0612f5de170c7a6fbd1a86f017a3b7';
const sources = [0, 1].map(index => ({
  index, id: `video-${index + 1}`, video: $(`video-${index + 1}`), canvas: $(`overlay-${index + 1}`),
  capture: document.createElement('canvas'), name: null, url: null, roi: null, observation: null,
  frameId: 0, frames: 0, lost: 0, mediaTimeMs: 0, decodedFrames: 0,
}));
const runtime = new PoseRuntime();
const fusion = new PoseFusion({ primarySourceId: 'video-1', confidence: .45, heightM: 1.7, hfovDeg: 65, syncToleranceMs: 80 });
const state = { running: false, busy: false, loading: false, starting: false, ready: false, generation: 0, nextAt: 0,
  lastMedia: -Infinity, lastObservations: [], fused: null, performance: [], records: [], renderUntil: 0, previewStale: false, staleFrames: 0,
  startedAt: null, clientId: crypto.randomUUID(), bridgeBusy: false, bridgeLastAt: 0, bridgeReset: true };
let view = null, pickTarget = null;
const depthReadout = document.createElement('div');
depthReadout.style.display = 'block';
depthReadout.innerHTML = '<div class="metric-row"><span>カメラからの推定距離 Zroot</span><strong id="camera-z">—</strong></div><div class="metric-row"><span>関節の相対Z · 中央90%範囲</span><strong id="relative-z">—</strong></div>';
document.querySelector('.root-position').append(depthReadout);
document.querySelector('#lost-rate').previousElementSibling.textContent = '人体を検出できなかった割合';
let viewPromise = import('./vrm-view.js').then(({ VrmView }) => {
  view = new VrmView($('avatar-view')); state.renderUntil = performance.now() + 1500; return view;
}).catch(error => { message(`3D表示の初期化: ${error.message}。2Dの入力操作は利用できます。`, 'error'); return null; });

function message(text, kind = '') { $('message').textContent = text; $('message').className = `message ${kind}`; }
function status(text, active = false) { $('overall-status').textContent = text; $('overall-status').classList.toggle('active', active); }
function seconds(value) { const n = Math.max(0, Number(value) || 0); return `${Math.floor(n / 60).toString().padStart(2, '0')}:${Math.floor(n % 60).toString().padStart(2, '0')}`; }
function number(value, digits = 2) { return Number.isFinite(value) ? value.toFixed(digits) : '—'; }
function download(name, data) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json;charset=utf-8' }));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1500);
}
function options() {
  return { heightM: Number($('height').value), heightCalibrated: $('height-calibrated').checked,
    hfovDeg: Number($('hfov').value), syncToleranceMs: Number($('tolerance').value), confidence: .45 };
}
function configure() { fusion.configure(options()); state.renderUntil = performance.now() + 1000; }
function progress(event) {
  $('progress-box').hidden = false;
  const mb = value => `${(Number(value || 0) / 1e6).toFixed(1)} MB`;
  const label = event.file || event.phase || 'モデルを準備';
  $('progress-label').textContent = `${label}${event.cached ? ' · キャッシュ' : ''}${event.loaded ? ` · ${mb(event.loaded)}${event.total ? ` / ${mb(event.total)}` : ''}` : ''}`;
  if (Number.isFinite(event.ratio)) { $('model-progress').value = event.ratio; $('progress-value').textContent = `${Math.round(event.ratio * 100)}%`; }
  else { $('model-progress').removeAttribute('value'); $('progress-value').textContent = '準備中'; }
}
async function loadModel() {
  if (state.loading) return;
  pause(); state.loading = true; state.ready = false; $('load-model').disabled = true; $('play').disabled = true;
  status('モデルを読み込み中');
  try {
    while (state.busy) await new Promise(resolve => setTimeout(resolve, 25));
    const result = await runtime.initialize({ variant: $('model').value, backend: 'GPU', hands: $('hands').checked,
      poseMode: 'IMAGE', handFps: 12, confidence: .45, onProgress: progress, modelBaseUrl: MODEL_BASE,
      rescueModel: $('rescue-model')?.value || 'dwpose-m', rescueFps: Number($('rescue-fps')?.value || 5) });
    state.ready = true; $('backend-badge').textContent = result.backend;
    $('model-progress').value = 1; $('progress-value').textContent = '準備完了';
    status('準備完了'); message('モデルを読み込みました。動画を再生すると、その時刻のキーポイントを推論します。', 'success');
  } catch (error) { status('読み込み失敗'); message(error.message, 'error'); }
  finally { state.loading = false; $('load-model').disabled = false; $('play').disabled = !sources[0].name; }
}
function clearObservations() {
  state.lastObservations = []; state.fused = null; state.lastMedia = -Infinity;
  for (const source of sources) { source.observation = null; draw(source); }
  view?.update(fusion.update([])); state.renderUntil = performance.now() + 500;
  $('root-x').textContent = $('root-y').textContent = $('root-z').textContent = '—';
  $('camera-z').textContent = $('relative-z').textContent = '—';
  $('body-count').textContent = $('feet-count').textContent = $('hand-count').textContent = '—';
}
function pause() {
  state.running = false; state.generation++; sources.forEach(source => source.video.pause());
  $('play').textContent = '▶ トラッキング開始'; $('overall-status').classList.remove('active');
  if (state.ready) status('一時停止'); state.bridgeReset = true;
}
async function start() {
  if (!sources[0].name || state.loading || state.starting) return;
  if (state.running) { pause(); return; }
  if (!state.ready) await loadModel();
  if (!state.ready) return;
  if (sources[0].video.ended) await seek(0);
  configure(); state.nextAt = 0; state.lastMedia = -Infinity;
  state.starting = true; $('play').disabled = true;
  try {
    await runtime.prepareSources(sources.filter(source => source.name).map(source => source.id));
    syncSecond(true);
    await Promise.all(sources.filter(source => source.name && source.video.readyState >= 2).map(source => source.video.play()));
    state.running = true; state.generation++; state.startedAt ||= new Date().toISOString();
    $('play').textContent = 'Ⅱ 一時停止'; status('トラッキング中', true);
  } catch (error) { pause(); message(`再生を開始できません: ${error.message}`, 'error'); }
  finally { state.starting = false; $('play').disabled = false; }
}
async function seek(time) {
  const resume = state.running; pause(); clearObservations(); fusion.reset(); view?.resetOrigin();
  while (state.busy) await new Promise(resolve => setTimeout(resolve, 20));
  await runtime.reset();
  sources[0].video.currentTime = Math.max(0, Math.min(Number(time), sources[0].video.duration || 0));
  syncSecond(true); state.clientId = crypto.randomUUID(); state.bridgeReset = true;
  if (resume) await start();
}
function syncSecond(force = false) {
  const second = sources[1]; if (!second.name) return false;
  const target = sources[0].video.currentTime + Number($('offset').value);
  if (target < 0 || target >= second.video.duration) { second.video.pause(); return false; }
  if (force || Math.abs(second.video.currentTime - target) > Number($('tolerance').value) / 1000) {
    second.video.currentTime = target;
    return false;
  }
  if (state.running && second.video.paused && !second.video.seeking) second.video.play().catch(() => {});
  return !second.video.seeking && second.video.readyState >= 2;
}
async function acceptFiles(files, target = null) {
  if (!files.length) return;
  pause(); message('動画を読み込んでいます。');
  try {
    const videos = await readVideoFiles(files, text => message(text));
    if (videos.length > 2) throw new Error('1回に選べる映像は2本です。比較する組み合わせごとに読み込んでください。');
    const startIndex = videos.length === 2 ? 0 : (target ?? (sources[0].name && !sources[1].name ? 1 : 0));
    while (state.busy) await new Promise(resolve => setTimeout(resolve, 25));
    for (let i = 0; i < videos.length; i++) await assignVideo(sources[startIndex + i], videos[i]);
    fusion.configure({ cameras: [] }); fusion.reset(); clearObservations(); view?.resetOrigin();
    await runtime.reset(); state.performance = []; sources.forEach(source => { source.frames = 0; source.lost = 0; });
    state.clientId = crypto.randomUUID(); state.bridgeReset = true;
    $('calibration-status').textContent = '映像を変更しました。カメラ校正なし / 単眼の奥行きは推定値';
    $('play').disabled = !sources[0].name; $('rewind').disabled = !sources[0].name; $('timeline').disabled = !sources[0].name;
    $('timeline').max = sources[0].video.duration || 0; $('timeline').value = sources[0].video.currentTime;
    status('動画を読み込みました');
    message(`${videos.length}本の動画を読み込みました。${sources[1].name ? '2視点の開始時刻が異なる場合は「奥行き・2視点の調整」で時刻を合わせてください。' : '「トラッキング開始」で実際の認識を表示します。'}`, 'success');
  } catch (error) { message(error.message, 'error'); }
}
async function assignVideo(source, item) {
  if (!source) throw new Error('入力スロットがありません。');
  if (source.url) URL.revokeObjectURL(source.url);
  source.url = URL.createObjectURL(item.blob); source.name = item.name; source.roi = null; source.observation = null;
  source.frameId = 0; source.video.src = source.url; source.video.playbackRate = Number($('speed').value);
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => finish(new Error('動画のデコードがタイムアウトしました。MOVのHEVCはブラウザとOSの対応が必要です。')), 25000);
    function finish(error) { clearTimeout(timer); source.video.removeEventListener('loadeddata', ready); source.video.removeEventListener('error', failed); error ? reject(error) : resolve(); }
    const ready = () => finish(); const failed = () => finish(new Error(`${item.name}: この端末で動画をデコードできません。H.264のMP4で読み込んでください。`));
    source.video.addEventListener('loadeddata', ready, { once: true }); source.video.addEventListener('error', failed, { once: true });
    source.video.load();
  });
  source.canvas.width = source.video.videoWidth; source.canvas.height = source.video.videoHeight;
  source.capture.width = source.video.videoWidth; source.capture.height = source.video.videoHeight;
  $(`name-${source.index + 1}`).textContent = item.name;
  $(`empty-${source.index + 1}`).hidden = true; $(`panel-${source.index + 1}`).classList.add('loaded');
  $(`stats-${source.index + 1}`).textContent = `${source.video.videoWidth} × ${source.video.videoHeight} / ${seconds(source.video.duration)}`;
}
function snapshot(source) {
  const video = source.video;
  if (!source.name || video.readyState < 2 || video.seeking || video.ended) return null;
  // Both cameras are copied before awaiting either model. Decoding can then
  // continue without attaching camera 2's newer pixels to camera 1's timestamp.
  const mediaTimeMs = video.currentTime * 1000;
  source.capture.getContext('2d', { alpha: false }).drawImage(video, 0, 0, source.capture.width, source.capture.height);
  return { source, image: source.capture, mediaTimeMs, frameId: ++source.frameId, roi: source.roi };
}
async function processFrame(now) {
  const generation = state.generation; state.busy = true;
  const started = performance.now();
  try {
    const secondReady = syncSecond();
    const frames = [snapshot(sources[0]), secondReady ? snapshot(sources[1]) : null].filter(Boolean);
    if (!frames.length) return;
    state.lastMedia = frames[0].mediaTimeMs;
    const observations = [];
    for (const frame of frames) {
      const result = await runtime.infer(frame.image, { sourceId: frame.source.id, mediaTimeMs: frame.mediaTimeMs, frameId: frame.frameId, roi: frame.roi });
      if (state.generation !== generation) return;
      if (result.skipped) continue;
      if (Math.abs(frame.source.video.currentTime * 1000 - frame.mediaTimeMs) > 250) { state.staleFrames++; continue; }
      result.sourceMediaTimeMs = result.mediaTimeMs;
      const offset = frame.source.index === 1 ? Number($('offset').value) * 1000 : 0;
      result.mediaTimeMs -= offset;
      result.keypoints = result.keypoints.map(point => ({ ...point, sourceObservationTimeMs: point.observationTimeMs,
        observationTimeMs: Number.isFinite(point.observationTimeMs) ? point.observationTimeMs - offset : result.mediaTimeMs,
        depthPriorTimeMs: Number.isFinite(point.depthPriorTimeMs) ? point.depthPriorTimeMs - offset : point.depthPriorTimeMs,
        depthPriorObservationTimeMs: Number.isFinite(point.depthPriorObservationTimeMs) ? point.depthPriorObservationTimeMs - offset : point.depthPriorObservationTimeMs }));
      result.hands = (result.hands || []).map(hand => ({ ...hand, sourceMediaTimeMs: hand.mediaTimeMs, mediaTimeMs: hand.mediaTimeMs - offset,
        landmarks: hand.landmarks.map(point => ({ ...point, sourceObservationTimeMs: point.observationTimeMs,
          observationTimeMs: Number.isFinite(point.observationTimeMs) ? point.observationTimeMs - offset : result.mediaTimeMs })) }));
      frame.source.observation = result; frame.source.frames++; if (!result.detected) frame.source.lost++;
      observations.push(result); draw(frame.source);
      const visible = result.keypoints.filter(p => p.observable && p.source !== 'held');
      $(`stats-${frame.source.index + 1}`).textContent = `2D ${visible.length}点 / 3D ${visible.filter(p => p.relativeXYZm).length}点 · ${number(result.inferenceMs, 1)}ms`;
    }
    if (state.generation !== generation || !observations.length) return;
    state.lastObservations = observations;
    const fused = fusion.update(observations); state.fused = fused; state.previewStale = false; view?.update(fused); state.renderUntil = performance.now() + 1000;
    $('avatar-empty').hidden = fused.root.valid;
    updateQuality(observations, fused);
    const elapsed = performance.now() - started;
    state.performance.push({ at: performance.now(), elapsed, frames: observations.map(o => ({ sourceId: o.sourceId, ms: o.inferenceMs, lost: !o.detected, relative3D: !!o.relative3D })) });
    state.performance = state.performance.filter(entry => entry.at > performance.now() - 15000);
    if ($('record').checked) {
      if (state.records.length < 15000) state.records.push({ observations, fused });
      else { $('record').checked = false; message('座標ログが15,000時刻に達したため、記録を停止しました。結果を書き出してください。'); }
    }
    updatePerformance(); sendBridge(fused);
  } catch (error) { pause(); clearObservations(); view?.resetOrigin(); status('処理エラー'); message(`${error.message}。動画・モデルの設定を確認して再開してください。`, 'error'); }
  finally {
    const elapsed = performance.now() - started;
    const budget = Math.max(.05, Math.min(.95, Number($('budget').value)));
    const interval = 1000 / Number($('target-fps').value);
    state.nextAt = performance.now() + Math.min(500, Math.max(interval - elapsed, elapsed * (1 - budget) / budget, 0));
    state.busy = false;
  }
}
function draw(source, drag = null) {
  const canvas = source.canvas; const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, canvas.width, canvas.height);
  const observation = source.observation;
  const points = new Map((observation?.keypoints || []).filter(p => p.observable).map(p => [canonicalName(p.name), p]));
  const size = Math.max(2, canvas.width / 300); ctx.lineCap = 'round';
  for (const [a, b] of [...BODY_EDGES, ...HAND_EDGES]) {
    const p = points.get(a), q = points.get(b); if (!p || !q) continue;
    ctx.lineWidth = size * .8; ctx.strokeStyle = p.source === 'held' || q.source === 'held' ? '#d7a461' : '#a9dfad';
    ctx.beginPath(); ctx.moveTo(p.u, p.v); ctx.lineTo(q.u, q.v); ctx.stroke();
  }
  for (const p of points.values()) {
    if (!Number.isFinite(p.u) || !Number.isFinite(p.v)) continue;
    ctx.fillStyle = p.source === 'held' ? '#e7b56e' : p.relativeXYZm ? '#9dc9ff' : '#c1ed95';
    ctx.beginPath(); ctx.arc(p.u, p.v, /_face_/.test(p.name) ? size * .45 : size, 0, Math.PI * 2); ctx.fill();
  }
  const roi = drag || source.roi;
  if (roi) { ctx.strokeStyle = '#c6e9a4'; ctx.lineWidth = size; ctx.setLineDash([size * 3, size * 2]); ctx.strokeRect(roi.x, roi.y, roi.width, roi.height); ctx.setLineDash([]); }
}
function updateQuality(observations, fused) {
  const points = new Map();
  for (const observation of observations) for (const point of observation.keypoints) {
    if (!point.observable || point.source === 'held') continue;
    points.set(canonicalName(point.name), point);
  }
  const names = [...points.keys()];
  $('body-count').textContent = names.filter(name => /_(shoulder|elbow|wrist|hip|knee|ankle)$/.test(name) && !name.includes('hand_')).length;
  $('feet-count').textContent = names.filter(name => /_(ankle|heel|big_toe|small_toe)$/.test(name)).length;
  $('hand-count').textContent = names.filter(name => /_(hand_wrist|(?:thumb|index|middle|ring|pinky)_(?:cmc|mcp|ip|pip|dip|tip))$/.test(name)).length;
  $('quality-status').textContent = fused.root.valid ? '3Dを推定中' : points.size ? '2D観測 / 3D欠測' : '観測なし';
  const root = fused.root.worldXYZm;
  ['x', 'y', 'z'].forEach((axis, index) => { $(`root-${axis}`).textContent = fused.root.valid ? number(root?.[index]) : '—'; });
  const labels = { estimatedUncalibrated: '未校正の単眼推定', estimatedMonocular: '単眼推定', estimatedMultiView: '2視点の推定整列', measuredMultiView: '校正済み2視点の三角測量', missing: '有効な距離観測なし' };
  $('distance-status').textContent = labels[fused.root.distanceStatus] || fused.root.distanceStatus;
  const warnings = [...new Set([...(fused.diagnostics?.warnings || []), ...observations.flatMap(observation => observation.warnings || [])])];
  const regions = [['左腕', ['left_shoulder', 'left_elbow', 'left_wrist']], ['右腕', ['right_shoulder', 'right_elbow', 'right_wrist']],
    ['左脚', ['left_hip', 'left_knee', 'left_ankle']], ['右脚', ['right_hip', 'right_knee', 'right_ankle']],
    ['左足', ['left_ankle', 'left_heel', 'left_big_toe']], ['右足', ['right_ankle', 'right_heel', 'right_big_toe']]];
  const missing = regions.map(([label, joints]) => `${label}${joints.filter(joint => points.has(joint)).length}/${joints.length}`).join(' · ');
  $('quality-note').textContent = `${missing}。${warnings.join(' ') || '手の関節別の遮蔽精度はモデルから取得できません。'}`;
  const last = observations[0]; $('backend-badge').textContent = last.backend;
  if ($('camera-z')) $('camera-z').textContent = fused.root.valid ? `${number(fused.root.cameraXYZm?.[2])} m` : '—';
  const depths = observations.flatMap(observation => observation.keypoints.filter(point => point.observable && point.relativeXYZm).map(point => point.relativeXYZm[2])).filter(Number.isFinite).sort((a, b) => a - b);
  if ($('relative-z')) $('relative-z').textContent = depths.length ? `${number(depths[Math.floor(depths.length * .05)])} ～ ${number(depths[Math.min(depths.length - 1, Math.floor(depths.length * .95))])} m` : '—';
  $('sync-ms').textContent = observations.length === 2 ? `${number(Math.abs(observations[0].mediaTimeMs - observations[1].mediaTimeMs), 1)} ms` : '—';
}
function updatePerformance() {
  const entries = state.performance, count = entries.length;
  if (!count) return;
  const duration = (performance.now() - entries[0].at) / 1000;
  const frames = entries.flatMap(entry => entry.frames);
  const values = frames.map(frame => frame.ms).filter(Number.isFinite).sort((a, b) => a - b);
  $('actual-fps').textContent = `${duration > 1 ? number(frames.filter(frame => frame.sourceId === 'video-1').length / duration, 1) : '—'} fps`;
  $('inference-ms').textContent = `${number(values[Math.min(values.length - 1, Math.floor(values.length * .95))], 1)} ms`;
  const total = sources.reduce((sum, source) => sum + source.frames, 0), lost = sources.reduce((sum, source) => sum + source.lost, 0);
  $('lost-rate').textContent = total ? `${number(lost / total * 100, 1)}%` : '—';
}
async function sendBridge(fused) {
  if (!$('bridge-enabled').checked || state.bridgeBusy || performance.now() - state.bridgeLastAt < 33) return;
  let url;
  try {
    url = new URL($('bridge-url').value);
    if (!['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)) throw new Error('ブリッジはこのPCのlocalhostを指定してください。');
    if (!['http:', 'https:'].includes(url.protocol)) throw new Error('ブリッジURLが無効です。');
  } catch (error) { $('bridge-enabled').checked = false; $('bridge-status').textContent = error.message; return; }
  const token = $('bridge-token').value;
  if (!token) { $('bridge-enabled').checked = false; $('bridge-status').textContent = '接続トークンが必要です'; return; }
  state.bridgeBusy = true; state.bridgeLastAt = performance.now();
  try {
    const response = await fetch(new URL('/pose', url), { method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ ...fused, clientId: state.clientId, reset: state.bridgeReset }), signal: AbortSignal.timeout(2000) });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.bridgeReset = false; $('bridge-status').textContent = fused.root.valid ? '接続中 / 有効性はブリッジで判定' : '接続中 / 姿勢欠測';
  } catch (error) { $('bridge-status').textContent = `接続失敗: ${error.message}`; }
  finally { state.bridgeBusy = false; }
}
function raf(now) {
  if (view && (state.running || now < state.renderUntil)) view.render();
  const video = sources[0].video;
  if (sources[0].name) {
    if (document.activeElement !== $('timeline')) $('timeline').value = video.currentTime;
    $('time-label').textContent = `${seconds(video.currentTime)} / ${seconds(video.duration)}`;
    for (const source of sources) {
      if (state.running && source.observation && source.video.currentTime * 1000 - source.observation.sourceMediaTimeMs > 250) { source.observation = null; draw(source); }
    }
    if (state.running && !state.previewStale && state.fused && video.currentTime * 1000 - state.fused.mediaTimeMs > 250) {
      view?.update({ ...state.fused, root: { ...state.fused.root, valid: false }, keypoints: [] });
      state.previewStale = true; $('quality-status').textContent = '最新の観測待ち';
      $('root-x').textContent = $('root-y').textContent = $('root-z').textContent = '—';
      $('camera-z').textContent = $('relative-z').textContent = '—';
    }
  }
  if (state.running && video.ended) { pause(); status('動画の再生完了'); }
  if (state.running && !state.busy && now >= state.nextAt && video.readyState >= 2 && !video.seeking && Math.abs(video.currentTime * 1000 - state.lastMedia) >= 2) processFrame(now);
  requestAnimationFrame(raf);
}

$('load-model').addEventListener('click', loadModel);
$('play').addEventListener('click', start);
$('rewind').addEventListener('click', () => seek(0));
$('timeline').addEventListener('change', () => seek(Number($('timeline').value)));
$('speed').addEventListener('change', () => sources.forEach(source => { source.video.playbackRate = Number($('speed').value); }));
for (const id of ['model', 'hands']) $(id).addEventListener('change', () => { pause(); state.ready = false; status('モデルを再読み込み'); });
$('skeleton').addEventListener('change', () => { view?.setSkeletonVisible($('skeleton').checked); state.renderUntil = performance.now() + 500; });
for (const id of ['height', 'height-calibrated', 'hfov', 'tolerance']) $(id).addEventListener('change', configure);
$('offset').addEventListener('change', () => { syncSecond(true); fusion.reset(); clearObservations(); state.bridgeReset = true; });
const picker = $('file-input');
$('drop-zone').addEventListener('click', () => { pickTarget = null; picker.click(); });
$('drop-zone').addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); picker.click(); } });
picker.addEventListener('change', () => { acceptFiles([...picker.files], pickTarget); picker.value = ''; });
document.querySelectorAll('[data-pick]').forEach(button => button.addEventListener('click', () => { pickTarget = Number(button.dataset.pick); picker.click(); }));
for (const type of ['dragenter', 'dragover']) $('drop-zone').addEventListener(type, event => { event.preventDefault(); $('drop-zone').classList.add('drag'); });
for (const type of ['dragleave', 'drop']) $('drop-zone').addEventListener(type, event => { event.preventDefault(); $('drop-zone').classList.remove('drag'); });
$('drop-zone').addEventListener('drop', event => acceptFiles([...event.dataTransfer.files]));
for (const source of sources) {
  const canvas = source.canvas; let dragStart = null;
  function position(event) {
    const rect = canvas.getBoundingClientRect(), scale = Math.min(rect.width / canvas.width, rect.height / canvas.height);
    const left = rect.left + (rect.width - canvas.width * scale) / 2, top = rect.top + (rect.height - canvas.height * scale) / 2;
    return { x: Math.max(0, Math.min(canvas.width, (event.clientX - left) / scale)), y: Math.max(0, Math.min(canvas.height, (event.clientY - top) / scale)) };
  }
  function rectangle(end) { return { x: Math.min(dragStart.x, end.x), y: Math.min(dragStart.y, end.y), width: Math.abs(end.x - dragStart.x), height: Math.abs(end.y - dragStart.y) }; }
  canvas.addEventListener('pointerdown', event => { if (!source.name) return; dragStart = position(event); canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener('pointermove', event => { if (dragStart) draw(source, rectangle(position(event))); });
  canvas.addEventListener('pointerup', event => {
    if (!dragStart) return; const roi = rectangle(position(event)); dragStart = null;
    if (roi.width >= 20 && roi.height >= 20) { source.roi = roi; runtime.reset(source.id).catch(error => message(error.message, 'error')); }
    $(`stage-${source.index + 1}`).classList.remove('roi-active'); draw(source);
  });
  source.video.addEventListener('error', () => { if (source.name) { pause(); message(`${source.name}: 動画のデコードに失敗しました。HEVCのMOVはOS側の対応が必要な場合があります。`, 'error'); } });
}
document.querySelectorAll('[data-roi]').forEach(button => button.addEventListener('click', () => {
  const index = Number(button.dataset.roi); if (!sources[index].name) return;
  pause();
  $(`stage-${index + 1}`).classList.toggle('roi-active'); message('映像の上をドラッグして人物の範囲を指定してください。');
}));
document.querySelectorAll('[data-clear-roi]').forEach(button => button.addEventListener('click', () => {
  const source = sources[Number(button.dataset.clearRoi)]; source.roi = null; runtime.reset(source.id).catch(error => message(error.message, 'error')); draw(source);
}));
$('vrm-input').addEventListener('change', async event => {
  const file = event.target.files[0]; if (!file) return;
  try {
    const renderer = await viewPromise; if (!renderer) throw new Error('3D表示を初期化できませんでした。');
    message(`${file.name} を読み込んでいます。`); const result = await renderer.load(file);
    $('vrm-name').textContent = `${result.name} / VRM ${result.version}`;
    state.renderUntil = performance.now() + 3000; message('VRMを読み込みました。追跡した身体と人物全体の位置を反映します。', 'success');
  } catch (error) { message(`VRM: ${error.message}`, 'error'); }
});
$('avatar-view').addEventListener('pointermove', () => { state.renderUntil = performance.now() + 1500; });
$('avatar-view').addEventListener('wheel', () => { state.renderUntil = performance.now() + 1500; }, { passive: true });
window.addEventListener('resize', () => { view?.resize(); state.renderUntil = performance.now() + 500; });
$('reset-view').addEventListener('click', () => { view?.resetOrigin(); if (state.fused) view?.update(state.fused); state.renderUntil = performance.now() + 1500; });
$('auto-align').addEventListener('click', () => {
  try {
    const result = fusion.estimateCameraAlignment(state.lastObservations); view?.resetOrigin();
    $('calibration-status').textContent = `2視点の推定整列 / 共通${result.commonJoints}関節 / 残差${number(result.residualM * 100, 1)}cm / 実測校正ではありません`;
    message('2視点を推定整列しました。姿勢を変えたときにも両側の関節が一致するか確認できます。', 'success');
  } catch (error) { message(error.message, 'error'); }
});
$('calibration-input').addEventListener('change', async event => {
  try {
    const file = event.target.files[0]; if (!file) return;
    const config = JSON.parse(await file.text());
    if (!Array.isArray(config.cameras) || config.cameras.length > 2) throw new Error('cameras配列に1〜2台の校正を指定してください。');
    for (const camera of config.cameras) if (!['video-1', 'video-2'].includes(camera.sourceId)) throw new Error('sourceIdはvideo-1またはvideo-2です。');
    fusion.configure({ ...config }); fusion.reset(); view?.resetOrigin();
    if (Number.isFinite(config.heightM)) $('height').value = config.heightM;
    if (Number.isFinite(config.hfovDeg)) $('hfov').value = config.hfovDeg;
    $('height-calibrated').checked = config.heightCalibrated === true;
    $('calibration-status').textContent = `${config.cameras.length}台の設定を読み込みました。calibrated:falseの設定は実測扱いにしません。`;
    message('カメラ設定を読み込みました。', 'success');
  } catch (error) { message(`校正JSON: ${error.message}`, 'error'); }
});
$('clear-calibration').addEventListener('click', () => { fusion.configure({ cameras: [] }); fusion.reset(); view?.resetOrigin(); $('calibration-status').textContent = 'カメラ校正なし / 単眼の奥行きは推定値'; });
$('export-calibration').addEventListener('click', () => download('wholebody-camera-calibration.json', { ...fusion.getCalibration(), sourceOffsetSeconds: Number($('offset').value) }));
$('export-log').addEventListener('click', () => download('wholebody-tracking-results.json', {
  schemaVersion: 1, generatedAt: new Date().toISOString(), modelManifest: `${MODEL_BASE}/manifest.json`,
  stage4AccuracyCertified: false, gpuUtilizationMeasured: false, calibration: fusion.getCalibration(),
  settings: { variant: $('model').value, targetFps: Number($('target-fps').value), computeBudget: Number($('budget').value), handFps: 12, sourceOffsetSeconds: Number($('offset').value) },
  sources: sources.filter(source => source.name).map(source => ({ id: source.id, name: source.name, frames: source.frames, lost: source.lost, width: source.video.videoWidth, height: source.video.videoHeight, duration: source.video.duration })),
  performanceWindow: state.performance, staleFramesDiscarded: state.staleFrames, current: state.fused, records: state.records,
  note: 'Missing or held landmarks are distinct from current observations. Detection coverage alone is not anatomical accuracy.'
}));
window.addEventListener('pagehide', () => { pause(); runtime.dispose(); view?.dispose(); sources.forEach(source => { if (source.url) URL.revokeObjectURL(source.url); }); });
requestAnimationFrame(raf);
