// Model downloads are public. Video pixels never leave the browser.
export const DEFAULT_MODEL_BASE = 'https://huggingface.co/RabbitDaisuke/Wholebody-Video-Tracking/resolve/main';
const CACHE_NAME = 'wholebody-video-models-v1';
const inFlight = new Map();

function progress(callback, value) {
  if (typeof callback === 'function') callback(value);
}

export async function loadManifest({ manifest, modelBaseUrl = DEFAULT_MODEL_BASE, onProgress } = {}) {
  if (manifest) return manifest;
  progress(onProgress, { phase: 'manifest', file: 'manifest.json', loaded: 0, total: 0 });
  const response = await fetch(`${modelBaseUrl.replace(/\/$/, '')}/manifest.json`, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`モデル一覧の取得に失敗しました: HTTP ${response.status}`);
  const value = await response.json();
  if (!value.models || typeof value.models !== 'object') throw new Error('モデル一覧に models がありません。');
  return value;
}

export function resolveModel(manifest, key, fallbackFile, modelBaseUrl = DEFAULT_MODEL_BASE) {
  const entry = manifest?.models?.[key];
  const file = entry?.file || fallbackFile;
  const url = entry?.url || `${modelBaseUrl.replace(/\/$/, '')}/${file}`;
  if (new URL(url, location.href).protocol !== 'https:' && !/^https?:\/\/(127\.0\.0\.1|localhost)(:|\/)/.test(url)) {
    throw new Error('モデルは HTTPS または localhost から読み込んでください。');
  }
  return { ...entry, file, url, key };
}

async function verifyBytes(buffer, entry) {
  if (entry.bytes != null && Number(entry.bytes) > 0 && buffer.byteLength !== Number(entry.bytes)) {
    throw new Error(`${entry.file}: モデルサイズが一致しません (${buffer.byteLength} / ${entry.bytes})。`);
  }
  if (entry.sha256) {
    const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', buffer));
    const actual = Array.from(digest, value => value.toString(16).padStart(2, '0')).join('');
    if (actual !== String(entry.sha256).toLowerCase()) throw new Error(`${entry.file}: SHA-256 が一致しません。`);
  }
}

export async function fetchModel(entry, onProgress) {
  // A pinned asset URL is the cache key; a replacement model cannot silently
  // reuse the bytes of an earlier revision with the same filename.
  const requestKey = `${entry.url}#${entry.sha256 || ''}`;
  if (inFlight.has(requestKey)) return inFlight.get(requestKey);
  const pending = (async () => {
    let cache = null;
    try { cache = await caches.open(CACHE_NAME); } catch { /* Private browsing can disable Cache Storage. */ }
    if (cache) {
      const cached = await cache.match(entry.url);
      if (cached) {
        const bytes = await cached.arrayBuffer();
        try {
          await verifyBytes(bytes, entry);
          progress(onProgress, { phase: 'download', file: entry.file, loaded: bytes.byteLength, total: bytes.byteLength, ratio: 1, cached: true });
          return new Uint8Array(bytes);
        } catch {
          await cache.delete(entry.url);
        }
      }
    }
    const response = await fetch(entry.url, { mode: 'cors', credentials: 'omit', redirect: 'follow' });
    if (!response.ok) throw new Error(`${entry.file} の取得に失敗しました: HTTP ${response.status}`);
    const total = Number(entry.bytes || response.headers.get('content-length') || 0);
    const chunks = [];
    let loaded = 0;
    if (response.body?.getReader) {
      const reader = response.body.getReader();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        progress(onProgress, { phase: 'download', file: entry.file, loaded, total, ratio: total ? Math.min(1, loaded / total) : null, cached: false });
      }
    } else {
      const value = new Uint8Array(await response.arrayBuffer());
      chunks.push(value);
      loaded = value.byteLength;
    }
    const bytes = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    await verifyBytes(bytes.buffer, entry);
    if (cache) {
      try { await cache.put(entry.url, new Response(bytes, { headers: { 'Content-Type': 'application/octet-stream' } })); }
      catch { /* A full cache must not prevent local inference. */ }
    }
    progress(onProgress, { phase: 'download', file: entry.file, loaded, total: loaded, ratio: 1, cached: false });
    return bytes;
  })();
  inFlight.set(requestKey, pending);
  try { return await pending; } finally { inFlight.delete(requestKey); }
}

export function sourceSize(source) {
  const width = Number(source.videoWidth || source.naturalWidth || source.width || 0);
  const height = Number(source.videoHeight || source.naturalHeight || source.height || 0);
  if (!(width > 0 && height > 0)) throw new Error('動画フレームの幅または高さが 0 です。デコード完了を待ってください。');
  return { width, height };
}

export function normalizeRoi(roi, imageSize) {
  if (!roi) return null;
  let value;
  if (Array.isArray(roi)) value = { x: roi[0], y: roi[1], width: roi[2], height: roi[3] };
  else value = { ...roi };
  if (value.x1 != null && value.x2 != null) value = { ...value, x: value.x1, y: value.y1, width: value.x2 - value.x1, height: value.y2 - value.y1 };
  if (value.normalized) value = { ...value, x: value.x * imageSize.width, y: value.y * imageSize.height, width: value.width * imageSize.width, height: value.height * imageSize.height };
  if (![value.x, value.y, value.width, value.height].every(Number.isFinite) || value.width < 2 || value.height < 2) {
    throw new Error('ROI は元動画のピクセル座標 x, y, width, height を指定してください。');
  }
  return value;
}

export function makeCanvas(width, height) {
  const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(width, height) : document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

export const HAND_NAMES = ['hand_wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip', 'index_mcp', 'index_pip', 'index_dip', 'index_tip', 'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip', 'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip', 'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip'];

export function finiteXYZ(point) {
  return point && [point.x, point.y, point.z].every(Number.isFinite) ? [point.x, point.y, point.z] : null;
}

// Region coverage is a measurement-availability report, not an accuracy score.
// Facial points cannot compensate for a missing arm, leg, heel, or toe.
export const BODY_REGIONS = {
  torso: ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
  left_arm: ['left_shoulder', 'left_elbow', 'left_wrist'],
  right_arm: ['right_shoulder', 'right_elbow', 'right_wrist'],
  left_leg: ['left_hip', 'left_knee', 'left_ankle'],
  right_leg: ['right_hip', 'right_knee', 'right_ankle'],
  left_foot: ['left_ankle', 'left_heel', 'left_foot_index'],
  right_foot: ['right_ankle', 'right_heel', 'right_foot_index'],
};
export const BODY_CORE_NAMES = [...new Set(Object.values(BODY_REGIONS).flat())];
const FACE_NAMES = new Set(['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right']);

function currentPoint(point) {
  return point?.observable === true && point.source !== 'held' && Number.isFinite(point.u) && Number.isFinite(point.v);
}
function valid3(point) { return Array.isArray(point?.relativeXYZm) && point.relativeXYZm.length === 3 && point.relativeXYZm.every(Number.isFinite); }
function distance2(a, b) { return Math.hypot(a.u - b.u, a.v - b.v); }

export function bodyCoverage(points, { hands = false } = {}) {
  const map = new Map(points.map(point => [point.name, point]));
  const regions = {};
  for (const [name, names] of Object.entries(BODY_REGIONS)) {
    const visible = names.filter(joint => currentPoint(map.get(joint)));
    regions[name] = { visible: visible.length, expected: names.length, fraction: visible.length / names.length, complete: visible.length === names.length, missing: names.filter(joint => !visible.includes(joint)) };
  }
  if (hands) for (const side of ['left', 'right']) {
    const names = HAND_NAMES.map(name => `${side}_${name}`);
    const visible = names.filter(name => currentPoint(map.get(name)));
    const available = names.filter(name => {
      const point = map.get(name);
      return point?.observable && (point.source !== 'held' || point.ageMs <= 200);
    });
    regions[`${side}_hand`] = { visible: visible.length, available: available.length, expected: 21, fraction: visible.length / 21, complete: visible.length === 21, missing: names.filter(name => !visible.includes(name)), wristVisible: currentPoint(map.get(`${side}_wrist`)) };
  }
  const missingRegions = Object.entries(regions).filter(([name, region]) => name.endsWith('_hand') ? region.wristVisible && region.available === 0 : !region.complete).map(([name]) => name);
  return {
    regions, missingRegions, complete: missingRegions.length === 0,
    visibleCore: BODY_CORE_NAMES.filter(name => currentPoint(map.get(name))).length,
    expectedCore: BODY_CORE_NAMES.length,
    current2D: points.filter(currentPoint).length,
    current3D: points.filter(point => currentPoint(point) && valid3(point)).length,
    held: points.filter(point => point.observable && point.source === 'held').length,
    isAccuracyMeasurement: false,
  };
}

export function suppressInconsistentFace(points, imageSize) {
  const map = new Map(points.map(point => [point.name, point]));
  const anchors = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'].map(name => map.get(name));
  if (!anchors.every(currentPoint)) return { validated: false, suppressed: false, reason: 'torso-unavailable' };
  const [ls, rs, lh, rh] = anchors;
  const shoulder = { u: (ls.u + rs.u) / 2, v: (ls.v + rs.v) / 2 };
  const hip = { u: (lh.u + rh.u) / 2, v: (lh.v + rh.v) / 2 };
  const axis = [shoulder.u - hip.u, shoulder.v - hip.v];
  const length = Math.hypot(...axis);
  if (length < Math.max(12, Math.hypot(imageSize.width, imageSize.height) * 0.015)) return { validated: false, suppressed: false, reason: 'torso-foreshortened' };
  const nose = map.get('nose');
  if (!currentPoint(nose)) return { validated: false, suppressed: false, reason: 'nose-unavailable' };
  const extension = ((nose.u - shoulder.u) * axis[0] + (nose.v - shoulder.v) * axis[1]) / (length * length);
  // Orientation-independent torso geometry: this also covers rotated people.
  // A nose collapsed onto the chest is an unverified prediction; a high model
  // visibility score does not turn that contradiction into a measurement.
  if (extension >= 0.12) return { validated: false, suppressed: false, reason: 'no-geometric-contradiction', torsoExtension: extension };
  for (const point of points) {
    if (!FACE_NAMES.has(point.name) && !point.name.startsWith('face_')) continue;
    point.observable = false;
    point.suppressedReason = 'face-torso-inconsistent';
    point.visibilityValidated = false;
  }
  return { validated: false, suppressed: true, reason: 'face-torso-inconsistent', torsoExtension: extension };
}

export function suppressAmbiguousBilateralOverlap(points, imageSize) {
  const map = new Map(points.map(point => [point.name, point]));
  const widths = ['shoulder', 'hip'].map(joint => {
    const a = map.get(`left_${joint}`), b = map.get(`right_${joint}`);
    return currentPoint(a) && currentPoint(b) ? distance2(a, b) : 0;
  });
  const thresholdPx = Math.max(3, Math.min(Math.hypot(imageSize.width, imageSize.height) * 0.006, Math.max(...widths, 50) * 0.06));
  const groups = {
    arms: ['elbow', 'wrist'], legs: ['knee', 'ankle'],
    feet: ['heel', 'foot_index', 'small_toe'],
    hands: ['hand_wrist', 'index_mcp', 'middle_mcp', 'pinky_mcp'],
  };
  const ambiguousGroups = [], suppressedNames = [];
  for (const [group, joints] of Object.entries(groups)) {
    const overlaps = joints.filter(joint => {
      const left = map.get(`left_${joint}`), right = map.get(`right_${joint}`);
      return currentPoint(left) && currentPoint(right) && distance2(left, right) <= thresholdPx;
    });
    if (overlaps.length < 2) continue;
    ambiguousGroups.push(group);
    // Genuine projected overlap is also unable to independently identify both
    // sides from this view. Keep candidate pixels in the record but withhold
    // them from measured correspondences rather than copying one limb twice.
    for (const joint of overlaps) for (const side of ['left', 'right']) {
      const point = map.get(`${side}_${joint}`);
      point.observable = false;
      point.bilateralAmbiguous = true;
      point.suppressedReason = 'bilateral-overlap-ambiguous';
      suppressedNames.push(point.name);
    }
  }
  return { groups: ambiguousGroups, suppressedNames, thresholdPx };
}

export function fuseSameFrameModels(primary, refinement, { hands = false } = {}) {
  if (primary.sourceId !== refinement.sourceId || primary.frameId !== refinement.frameId || Math.abs(primary.mediaTimeMs - refinement.mediaTimeMs) > 0.01) {
    throw new Error('異なる映像または時刻のキーポイントは同時刻モデル融合に使用できません。');
  }
  const pMap = new Map(primary.keypoints.map(point => [point.name, point]));
  const rMap = new Map(refinement.keypoints.map(point => [point.name, point]));
  const pCoverage = bodyCoverage(primary.keypoints, { hands });
  const rCoverage = bodyCoverage(refinement.keypoints, { hands });
  const bodyPoints = BODY_CORE_NAMES.map(name => pMap.get(name)).filter(currentPoint);
  const extent = bodyPoints.length > 1 ? Math.hypot(
    Math.max(...bodyPoints.map(p => p.u)) - Math.min(...bodyPoints.map(p => p.u)),
    Math.max(...bodyPoints.map(p => p.v)) - Math.min(...bodyPoints.map(p => p.v)),
  ) : Math.hypot(primary.imageSize.width, primary.imageSize.height);
  const agreementPx = Math.max(4, Math.min(Math.hypot(primary.imageSize.width, primary.imageSize.height) * 0.02, extent * 0.06));
  const common = BODY_CORE_NAMES.filter(name => currentPoint(pMap.get(name)) && currentPoint(rMap.get(name)));
  const consistent = common.filter(name => distance2(pMap.get(name), rMap.get(name)) <= agreementPx);
  const bodyAgreement = consistent.length >= 3 && consistent.length / Math.max(1, common.length) >= 0.65;
  const rootMeanSquarePx = common.length ? Math.sqrt(common.reduce((sum, name) => sum + distance2(pMap.get(name), rMap.get(name)) ** 2, 0) / common.length) : null;
  const regionsFor = name => Object.entries(BODY_REGIONS).filter(([, names]) => names.includes(name)).map(([region]) => region);
  const names = new Set([...pMap.keys(), ...rMap.keys()]);
  const merged = [];
  let refined2D = 0, added2D = 0, retained3D = 0, conflicts = 0;
  for (const name of names) {
    const p = pMap.get(name), r = rMap.get(name);
    const pCurrent = currentPoint(p) && Math.abs((p.observationTimeMs ?? primary.mediaTimeMs) - primary.mediaTimeMs) < 1;
    const rCurrent = currentPoint(r);
    const agreement = pCurrent && rCurrent && distance2(p, r) <= agreementPx;
    const handSide = /^(left|right)_(?:hand_wrist|(?:thumb|index|middle|ring|pinky)_)/.exec(name)?.[1];
    const wristP = handSide ? pMap.get(`${handSide}_wrist`) : null;
    const wristR = handSide ? rMap.get(`${handSide}_wrist`) : null;
    const depthAnchorValid = !handSide || (currentPoint(wristP) && currentPoint(wristR) && valid3(wristP) && distance2(wristP, wristR) <= agreementPx);
    const regionNames = regionsFor(name);
    const betterRegion = regionNames.some(region => rCoverage.regions[region].visible > pCoverage.regions[region].visible);
    const isFace = FACE_NAMES.has(name) || name.startsWith('face_');
    const useRefinement = !isFace && rCurrent && (!pCurrent || (agreement && bodyAgreement) || betterRegion);
    const prior = p && valid3(p) ? {
      depthPriorXYZm: p.relativeXYZm.slice(), depthPriorScore: p.score,
      depthPriorObservable: !!(pCurrent && bodyAgreement && agreement && depthAnchorValid),
      depthPriorRawObservable: !!pCurrent, depthPriorAgreement: !!(bodyAgreement && agreement && depthAnchorValid),
      depthPriorTimeMs: p.observationTimeMs ?? primary.mediaTimeMs,
      depthPriorModelId: primary.modelId,
    } : {};
    if (!useRefinement) {
      if (p) {
        const keepDepth = !!(pCurrent && valid3(p) && bodyAgreement && agreement && depthAnchorValid);
        const point = {
          ...p, ...prior, modelId2D: p.modelId2D || primary.modelId,
          relativeXYZm: keepDepth ? p.relativeXYZm.slice() : null,
          relativeZm: keepDepth ? p.relativeXYZm[2] : null,
          score3D: keepDepth ? p.score : null, modelId3D: keepDepth ? primary.modelId : null,
          modelAgreement: !!(bodyAgreement && agreement),
          validation: keepDepth ? 'cross-model-2d-agreement' : 'single-model-2d-unverified',
        };
        if (pCurrent && rCurrent && !agreement) { point.modelDisagreementPx = distance2(p, r); point.refinementRejected = '2d-disagreement'; conflicts++; }
        merged.push(point);
        if (keepDepth) retained3D++;
      } else if (r && !isFace) merged.push({ ...r, relativeXYZm: null, relativeZm: null, modelId2D: refinement.modelId, modelId3D: null });
      continue;
    }
    const preserveDepth = !!(pCurrent && valid3(p) && bodyAgreement && agreement && depthAnchorValid);
    const point = {
      ...r, ...prior, name,
      // The two models' scores are not calibrated probabilities. Preserve the
      // native 3D weight when retaining its 3D and record both native scores.
      score: preserveDepth ? p.score : r.score,
      score2D: r.score, score2DType: 'simcc-peak', score3D: preserveDepth ? p.score : null,
      relativeXYZm: preserveDepth ? p.relativeXYZm.slice() : null,
      relativeZm: preserveDepth ? p.relativeXYZm[2] : null,
      modelId2D: refinement.modelId, modelId3D: preserveDepth ? primary.modelId : null,
      backend2D: refinement.backend, backend3D: preserveDepth ? primary.backend : null,
      source: p ? 'observed-2d-refinement' : 'observed-2d-rescue',
      observationTimeMs: primary.mediaTimeMs, depthStatus: preserveDepth ? 'retained-monocular-estimate' : 'unavailable',
      modelAgreement: !!(bodyAgreement && agreement),
      validation: preserveDepth ? 'cross-model-2d-agreement' : 'single-model-2d-unverified',
    };
    if (pCurrent && !agreement) { point.modelDisagreementPx = distance2(p, r); conflicts++; }
    merged.push(point);
    if (pCurrent) refined2D++; else added2D++;
    if (preserveDepth) retained3D++;
  }
  const headConsistency = suppressInconsistentFace(merged, primary.imageSize);
  const bilateralConsistency = suppressAmbiguousBilateralOverlap(merged, primary.imageSize);
  const coverage = bodyCoverage(merged, { hands });
  const map = new Map(merged.map(point => [point.name, point]));
  const outputHands = ['left', 'right'].map(side => {
    const landmarks = HAND_NAMES.map(name => map.get(`${side}_${name}`)).filter(Boolean);
    const times = landmarks.map(point => point.observationTimeMs).filter(Number.isFinite);
    const mediaTimeMs = times.length ? Math.min(...times) : primary.mediaTimeMs;
    return { side, landmarks, mediaTimeMs, ageMs: Math.max(0, primary.mediaTimeMs - mediaTimeMs), observed: landmarks.some(currentPoint), mixedModels: true };
  }).filter(hand => hand.landmarks.length);
  return {
    keypoints: merged, hands: outputHands, coverage, visibleCore: coverage.visibleCore,
    detected: coverage.visibleCore >= 4, relative3D: coverage.current3D > 0,
    relative3DCount: coverage.current3D, headConsistency, bilateralConsistency,
    modelAgreement: { bodyAgreement, commonBody: common.length, consistentBody: consistent.length, thresholdPx: agreementPx, rmsePx: rootMeanSquarePx, refined2D, added2D, retained3D, conflicts },
  };
}
