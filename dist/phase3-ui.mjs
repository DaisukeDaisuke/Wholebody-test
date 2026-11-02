import { createVRMView } from './vrm-view.mjs';

const byId = id => document.getElementById(id);
const finitePosition = point => Array.isArray(point) && point.length >= 3 && point.slice(0, 3).every(Number.isFinite);
const hasLivePose = pose => (pose?.joints || []).some(joint => finitePosition(joint?.position) && joint.source !== 'missing') || Object.values(pose?.footHints || {}).some(foot => Array.isArray(foot?.heel) && Array.isArray(foot?.bigToe));

function setMetric(id, value, unit) {
  const node = byId(id);
  if (!node) return;
  const small = document.createElement('small'); small.textContent = ` ${unit}`;
  node.replaceChildren(document.createTextNode(Number.isFinite(value) ? value.toFixed(1) : '—'), small);
}

// Deliberately synthetic, opt-in display fixture. It only reaches this viewer;
// it never goes into inference, observations, session export, or the VR bridge.
function displayDemoPose(timeMs) {
  const t = timeMs / 1000, positions = new Map(), wave = Math.sin(t * 1.5);
  positions.set(0, [0, 1.70, 0.06]); positions.set(1, [0.035, 1.73, 0.055]); positions.set(2, [-0.035, 1.73, 0.055]);
  positions.set(3, [0.08, 1.70, 0]); positions.set(4, [-0.08, 1.70, 0]);
  for (const [side, sign, shoulder, elbow, wrist, hip, knee, ankle, footStart, handStart] of [
    ['left', 1, 5, 7, 9, 11, 13, 15, 17, 91], ['right', -1, 6, 8, 10, 12, 14, 16, 20, 112]
  ]) {
    const raise = side === 'left' ? 0.24 + wave * 0.20 : 0.10 - wave * 0.08;
    positions.set(shoulder, [sign * 0.22, 1.45, 0]);
    positions.set(elbow, [sign * 0.42, 1.14 + raise, 0.10]);
    const hand = [sign * 0.54, 0.91 + raise * 1.8, 0.15 + Math.sin(t * 1.2) * 0.07];
    positions.set(wrist, hand); positions.set(handStart, hand);
    positions.set(hip, [sign * 0.12, 0.94, 0]); positions.set(knee, [sign * 0.13, 0.49, 0.025]);
    positions.set(ankle, [sign * 0.13, 0.08, 0]);
    const toeAngle = Math.sin(t * 1.1 + (sign < 0 ? 1.1 : 0)) * 0.30;
    const toe = (x, z) => [sign * 0.13 + x * Math.cos(toeAngle) + z * Math.sin(toeAngle), 0.025, z * Math.cos(toeAngle) - x * Math.sin(toeAngle)];
    positions.set(footStart, toe(-sign * 0.035, 0.18)); positions.set(footStart + 1, toe(sign * 0.055, 0.14)); positions.set(footStart + 2, toe(0, -0.06));
    for (let finger = 0; finger < 5; finger++) {
      const spread = (finger - 2) * 0.023, length = finger === 0 ? 0.068 : finger === 4 ? 0.072 : 0.088;
      const curl = 0.5 + 0.45 * Math.sin(t * 1.7 + finger * 0.35 + (sign < 0 ? 1 : 0));
      for (let segment = 0; segment < 4; segment++) {
        const distance = length * (segment + 1) / 4, bend = curl * segment * 0.48;
        positions.set(handStart + 1 + finger * 4 + segment, [hand[0] + sign * (distance * Math.cos(bend) + (finger === 0 ? 0.005 : 0.015)), hand[1] + spread, hand[2] + distance * Math.sin(bend) + (finger === 0 ? 0.03 : 0)]);
      }
    }
  }
  return {
    timestampMs: timeMs, sourceId: 'display-demo', inputMode: 'wholebody', coordinateFrame: 'root-relative', units: 'm-relative-estimate', metric: false,
    root: { position: null, valid: false, metric: false, distanceStatus: 'demo' },
    joints: Array.from({ length: 133 }, (_, id) => ({ id, position: positions.get(id) || null, score: positions.has(id) ? 1 : 0, source: positions.has(id) ? 'demo' : 'missing', ageMs: 0 })),
    diagnostics: { syntheticDemo: true }
  };
}

export async function createPhase3UI({ onRenderMetrics = () => {}, onState = () => {}, onError = () => {} } = {}) {
  const container = byId('vrm-stage');
  if (!container || document.body.classList.contains('sender-mode')) throw new Error('VRM表示はPCの受信画面で開いてください。');
  const cleanup = [], originalCanvas = byId('scene-canvas'), wrapper = container.closest('.scene-wrap');
  let rendererNote = byId('vrm-renderer-note');
  if (!rendererNote) {
    rendererNote = document.createElement('p'); rendererNote.id = 'vrm-renderer-note';
    rendererNote.className = 'avatar-boundary avatar-renderer-note'; rendererNote.hidden = true;
    rendererNote.setAttribute('role', 'status');
    const boundary = byId('vrm-boundary');
    if (boundary) boundary.before(rendererNote); else container.after(rendererNote);
    cleanup.push(() => rendererNote.remove());
  }
  let view = null, disposed = false, demoActive = false, demoFrame = 0, lastDemoTick = 0, demoStart = 0, lastLivePose = null;
  const listen = (id, event, handler) => {
    const node = byId(id);
    if (!node) return;
    node.addEventListener(event, handler); cleanup.push(() => node.removeEventListener(event, handler));
  };
  const errorMessage = error => {
    const node = byId('vrm-status');
    if (node) { node.textContent = error?.message || String(error); node.dataset.error = 'true'; }
    onError(error);
  };
  function renderState(state) {
    const statusNode = byId('vrm-status'), badge = byId('vrm-input-badge');
    const select = byId('vrm-display-mode'), avatarOption = select?.querySelector('option[value="avatar"]');
    if (avatarOption) avatarOption.disabled = !state.hasVRM;
    if (select) select.value = state.displayMode;
    byId('vrm-unload').disabled = !state.hasVRM && !state.loading;
    byId('vrm-sample').disabled = state.loading;
    byId('vrm-file-name').textContent = state.fileName || 'VRM未選択 · 骨格だけでも使えます';
    byId('vrm-bone-count').textContent = state.hasVRM ? `${state.appliedBoneCount} / ${state.coverage?.availableBones ?? '—'} ボーン反映` : `${state.validJointCount || state.planarPointCount} 点を表示`;
    let label = '入力待ち', kind = 'waiting', message = state.hasVRM ? 'VRMの基準姿勢 · 映像の推論を待っています。' : '動画かカメラの推論を開始すると、推定骨格が動きます。';
    if (state.sourceMode === 'demo') { label = '表示確認用デモ'; kind = 'demo'; message = '作った動きで表示を確認中です。映像の推論結果ではありません。'; }
    else if (state.feetOnly && state.planarFeet) { label = '足 · 平面内の推定'; kind = 'planar'; message = '足の画像内の角度変化を反映しています。奥行きと上半身は観測していません。'; }
    else if (state.validJointCount) { label = '入力から推定'; kind = 'live'; message = state.hasVRM ? state.appliedBoneCount ? '推定した骨格を、VRMの対応するボーンへ反映しています。' : '推定点はありますが、VRMへ反映できる有効なボーンがありません。' : '入力から推定した骨格を表示しています。VRMも読み込めます。'; }
    else if (lastLivePose) { message = '現在の姿勢は欠測です。観測できない部位はVRMの基準姿勢へ戻します。'; }
    if (state.loading) message = 'VRMをこの端末内で読み込んでいます…';
    if (state.loadError) message = state.loadError;
    if (statusNode) { statusNode.textContent = message; statusNode.dataset.error = String(Boolean(state.loadError)); }
    if (badge) { badge.textContent = label; badge.dataset.kind = kind; }
    rendererNote.hidden = state.rendererBackend !== 'canvas2d';
    wrapper?.classList.toggle('cpu-renderer', state.rendererBackend === 'canvas2d');
    if (!rendererNote.hidden) {
      const stats = state.renderStats, count = number => Number(number || 0).toLocaleString('ja-JP');
      let text = 'CPU簡易描画 · 実際のボーンと変形済みメッシュを表示します。テクスチャ・MToon・影・透過は簡略化しています。';
      if (state.hasVRM && state.displayMode === 'avatar' && stats) {
        text += stats.limited
          ? ` ${count(stats.selectedFaces)} / ${count(stats.totalFaces)} 面、${count(stats.selectedVertices)} / ${count(stats.totalVertices)} 頂点を選択（上限 ${count(stats.limits.faces)} 面・${count(stats.limits.vertices)} 頂点）。`
          : ` 全 ${count(stats.totalFaces)} 面を使用。`;
        if (stats.unsupportedMeshes) text += ` 特殊なメッシュ ${count(stats.unsupportedMeshes)} 個は省略しています。`;
        text += ' 灰色の骨線は基準姿勢です。';
      }
      rendererNote.textContent = text; rendererNote.dataset.limited = String(Boolean(stats?.limited));
    }
    const boundary = byId('vrm-boundary');
    if (boundary) boundary.textContent = state.feetOnly
      ? `足だけ入力：足ボーンの平面内回転です。${state.anatomicalSidesConfirmed ? '左右は本人指定です。' : '自動で付けた左右は未確認です。'}3Dの向きと5本の足指の曲がりは未観測です。`
      : 'VRMの体型は表示に使います。本人の実寸とは別です。左右のToesは足先をまとめたボーンで、5本の足指の曲がりは未観測です。';
    onState(state);
  }
  view = createVRMView({
    container, onState: renderState, onError: errorMessage,
    onRenderMetrics(metrics) {
      setMetric('vrm-render-fps', metrics.fps, 'fps'); setMetric('vrm-render-ms', metrics.renderMs, 'ms');
      onRenderMetrics(metrics);
    }
  });
  container.hidden = false;
  if (originalCanvas) originalCanvas.hidden = true;
  wrapper?.classList.add('three-ready');
  function stopDemo(restore = true) {
    demoActive = false; cancelAnimationFrame(demoFrame);
    byId('vrm-demo-start').disabled = false; byId('vrm-demo-stop').disabled = true;
    if (restore && !disposed) view.updatePose(lastLivePose, { sourceMode: 'live' });
  }
  function startDemo() {
    if (disposed) return;
    stopDemo(false); demoActive = true; demoStart = performance.now(); lastDemoTick = -Infinity;
    byId('vrm-demo-start').disabled = true; byId('vrm-demo-stop').disabled = false;
    const tick = now => {
      if (!demoActive || disposed) return;
      if (now - lastDemoTick >= 1000 / 30) { view.updatePose(displayDemoPose(now - demoStart), { sourceMode: 'demo' }); lastDemoTick = now; }
      demoFrame = requestAnimationFrame(tick);
    };
    demoFrame = requestAnimationFrame(tick);
  }
  const loadFile = file => view.loadFile(file);
  listen('vrm-file', 'change', async event => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    try { await loadFile(file); } catch { /* createVRMView reports a specific local-file error. */ }
  });
  listen('vrm-unload', 'click', () => view.unload());
  listen('vrm-sample', 'click', async () => {
    try {
      const { createSampleVRM, SAMPLE_VRM_INFO } = await import('./vrm-fixture.mjs');
      await loadFile(new File([createSampleVRM()], SAMPLE_VRM_INFO.filename, { type: 'model/gltf-binary' }));
    } catch (error) { errorMessage(error); }
  });
  listen('vrm-display-mode', 'change', event => view.setDisplayMode(event.target.value));
  listen('reset-3d-view', 'click', () => view.resetView());
  listen('vrm-demo-start', 'click', startDemo);
  listen('vrm-demo-stop', 'click', () => stopDemo());
  return {
    loadFile, resetView: () => view.resetView(), getStatus: () => ({ ...view.getStatus(), demoActive }),
    updatePose(pose, { sourceMode = 'live' } = {}) {
      if (disposed) return;
      if (sourceMode === 'demo') { view.updatePose(pose, { sourceMode }); return; }
      lastLivePose = pose || null;
      if (demoActive) {
        if (!hasLivePose(pose)) return;
        stopDemo(false);
      }
      view.updatePose(pose, { sourceMode: 'live' });
    },
    dispose() {
      if (disposed) return;
      disposed = true; stopDemo(false); cleanup.forEach(fn => fn()); view.dispose();
      container.hidden = true; wrapper?.classList.remove('three-ready', 'cpu-renderer');
      if (originalCanvas) originalCanvas.hidden = false;
    }
  };
}
