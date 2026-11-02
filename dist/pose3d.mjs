/** Learned monocular body depth, shared by the browser and Node.js.
 *
 * The upstream MediaPipe Pose weights are distributed by OpenCV Zoo under
 * Apache-2.0. This implementation adapts its RGB/NHWC crop and world-landmark
 * projection contract. The learned depth and perspective translation remain
 * estimates; neither is a calibrated physical tracker measurement.
 * Reference: opencv/opencv_zoo@48b10fa505aeb51fbd127fc60a5c660040e823b7
 * models/pose_estimation_mediapipe/mp_pose.py
 */
export const POSE3D_SPEC = Object.freeze({
  id: 'mediapipe-pose3d-lite-opencv-2023mar',
  filename: 'pose3d.onnx',
  upstreamRevision: '48b10fa505aeb51fbd127fc60a5c660040e823b7',
  upstreamUrl: 'https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx',
  sha256: '9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f',
  bytes: 5557238,
  inputShape: [1, 256, 256, 3],
  color: 'RGB', layout: 'NHWC', scale: 1 / 255,
  license: 'Apache-2.0',
  worldUnits: 'estimated-metres-relative-to-hips',
  worldAxes: 'x right, y down, z away from camera',
});

// Foot index is a toe-centre approximation; this model does not observe a
// separate big/little toe or individual finger joints.
export const POSE3D_TO_COCO = Object.freeze([
  [0, 0], [2, 1], [5, 2], [7, 3], [8, 4],
  [11, 5], [12, 6], [13, 7], [14, 8], [15, 9], [16, 10],
  [23, 11], [24, 12], [25, 13], [26, 14], [27, 15], [28, 16],
  [31, 17], [29, 19], [32, 20], [30, 22], [15, 91], [16, 112],
]);
const finite = Number.isFinite;
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
const finite3 = p => Array.isArray(p) && p.length === 3 && p.every(finite);
const mid = (p, q) => ({ x: (p.x + q.x) / 2, y: (p.y + q.y) / 2 });

/** A square hip-centred ROI, optionally aligned to the current torso.
 * Missing anchors fall back to the independently supplied person ROI. */
export function createPose3DCrop(roi, anchors = [], imageSize, options = {}) {
  const [x, y, width, height] = roi || [];
  if (![x, y, width, height, imageSize?.width, imageSize?.height].every(finite) || width <= 0 || height <= 0 || imageSize.width <= 0 || imageSize.height <= 0) throw new Error('pose3dInvalidCrop');
  const padding = options.padding ?? 1.25;
  if (!finite(padding) || padding < 1 || padding > 2) throw new Error('pose3dInvalidPadding');
  const usable = p => p && finite(p.x) && finite(p.y) && p.score >= .35 && p.observable !== false && p.x >= 0 && p.x < imageSize.width && p.y >= 0 && p.y < imageSize.height;
  const hips = usable(anchors[11]) && usable(anchors[12]) ? mid(anchors[11], anchors[12]) : null;
  const shoulders = usable(anchors[5]) && usable(anchors[6]) ? mid(anchors[5], anchors[6]) : null;
  let cx = x + width / 2, cy = y + height / 2, side = Math.max(width, height) * padding, angleRad = 0;
  if (hips && shoulders) {
    cx = hips.x; cy = hips.y;
    const torso = Math.hypot(shoulders.x - hips.x, shoulders.y - hips.y);
    if (options.rotate !== false && torso >= 8) angleRad = Math.atan2(shoulders.x - hips.x, hips.y - shoulders.y);
    const visible = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22].map(id => anchors[id]).filter(usable);
    // The minimum radius preserves room for unseen head/limbs after a partial
    // observation, rather than shrinking the crop around the few surviving points.
    const radius = Math.max(torso * 2.4, ...visible.map(p => Math.hypot(p.x - cx, p.y - cy)));
    side = Math.max(Math.max(width, height) * .65, radius * 2 * 1.08);
    side = Math.min(side, Math.max(imageSize.width, imageSize.height) * 3);
  }
  return { cx, cy, side, angleRad, inputWH: [256, 256], imageSize: { width: imageSize.width, height: imageSize.height }, source: hips && shoulders ? 'observed-hip-torso' : 'person-roi' };
}

/** Canvas setTransform(...result); drawImage(fullFrame,0,0).
 * Fill the 256-square canvas black BEFORE setting this transform. */
export function pose3dCanvasTransform(crop) {
  const s = 256 / crop.side, c = Math.cos(crop.angleRad), n = Math.sin(crop.angleRad);
  return [s * c, -s * n, s * n, s * c,
    128 - s * (c * crop.cx + n * crop.cy), 128 - s * (-n * crop.cx + c * crop.cy)];
}

export function pose3dRgbaToTensor(rgba) {
  if (rgba?.length !== 256 * 256 * 4) throw new Error('pose3dInvalidRgbaSize');
  const out = new Float32Array(256 * 256 * 3);
  for (let i = 0; i < 256 * 256; i++) { out[i * 3] = rgba[i * 4] / 255; out[i * 3 + 1] = rgba[i * 4 + 1] / 255; out[i * 3 + 2] = rgba[i * 4 + 2] / 255; }
  return out;
}

/** Node RGB24 input. Bilinear inverse sampling, black outside the source. */
export function samplePose3DRgb(rgb, width, height, crop) {
  if (rgb?.length !== width * height * 3 || !Number.isInteger(width) || !Number.isInteger(height)) throw new Error('pose3dInvalidRgbSize');
  const out = new Float32Array(256 * 256 * 3), s = crop.side / 256, c = Math.cos(crop.angleRad), n = Math.sin(crop.angleRad);
  const value = (x, y, channel) => x >= 0 && y >= 0 && x < width && y < height ? rgb[(y * width + x) * 3 + channel] : 0;
  for (let y = 0; y < 256; y++) for (let x = 0; x < 256; x++) {
    const dx = (x + .5 - 128) * s, dy = (y + .5 - 128) * s;
    const px = crop.cx + c * dx - n * dy - .5, py = crop.cy + n * dx + c * dy - .5;
    const x0 = Math.floor(px), y0 = Math.floor(py), fx = px - x0, fy = py - y0;
    for (let channel = 0; channel < 3; channel++) out[(y * 256 + x) * 3 + channel] = ((1 - fy) * ((1 - fx) * value(x0, y0, channel) + fx * value(x0 + 1, y0, channel)) + fy * ((1 - fx) * value(x0, y0 + 1, channel) + fx * value(x0 + 1, y0 + 1, channel))) / 255;
  }
  return out;
}

function outputWithLength(outputs, count) {
  const matches = Object.values(outputs || {}).filter(t => t?.data?.length === count);
  if (matches.length !== 1) throw new Error(`pose3dOutputContract${count}`);
  return matches[0].data;
}

function refineXY(x, y, id, heatmap) {
  if (!heatmap || !finite(x) || !finite(y)) return [x, y];
  const hx = Math.floor(x / 4), hy = Math.floor(y / 4);
  if (hx < 0 || hx >= 64 || hy < 0 || hy >= 64) return [x, y];
  let sum = 0, sx = 0, sy = 0, maximum = 0;
  for (let yy = Math.max(0, hy - 3); yy <= Math.min(63, hy + 3); yy++) for (let xx = Math.max(0, hx - 3); xx <= Math.min(63, hx + 3); xx++) {
    const weight = sigmoid(heatmap[(yy * 64 + xx) * 39 + id]);
    if (!finite(weight)) continue;
    sum += weight; sx += weight * xx; sy += weight * yy; maximum = Math.max(maximum, weight);
  }
  return maximum >= .5 && sum > 1e-6 ? [sx / sum * 4, sy / sum * 4] : [x, y];
}

/** Decode output by shape, not an unstable converter-generated output name. */
export function decodePose3D(outputs, crop, options = {}) {
  const screen = outputWithLength(outputs, 195), world = outputWithLength(outputs, 117), presence = outputWithLength(outputs, 1);
  const heatmap = options.refineHeatmap === false ? null : Object.values(outputs).find(t => t?.data?.length === 64 * 64 * 39)?.data;
  const confidence = finite(presence[0]) ? clamp(presence[0]) : 0;
  const valid = confidence >= (options.minPoseConfidence ?? .5), threshold = options.minPointConfidence ?? .25;
  const c = Math.cos(crop.angleRad), n = Math.sin(crop.angleRad), scale = crop.side / 256;
  const landmarks = Array.from({ length: 33 }, (_, id) => {
    const [rx, ry] = refineXY(screen[id * 5], screen[id * 5 + 1], id, heatmap);
    const dx = (rx - 128) * scale, dy = (ry - 128) * scale;
    const x = crop.cx + c * dx - n * dy, y = crop.cy + n * dx + c * dy;
    const visibility = sigmoid(screen[id * 5 + 3]), pointPresence = sigmoid(screen[id * 5 + 4]);
    const modelPosition = [c * world[id * 3] - n * world[id * 3 + 1], n * world[id * 3] + c * world[id * 3 + 1], world[id * 3 + 2]];
    const onScreen = finite(x) && finite(y) && x >= 0 && y >= 0 && x < crop.imageSize.width && y < crop.imageSize.height;
    const finiteWorld = modelPosition.every(v => finite(v) && Math.abs(v) < 5);
    const score = valid && onScreen && finiteWorld && finite(visibility) && finite(pointPresence) ? Math.min(confidence, visibility, pointPresence) : 0;
    return { id, x, y, score, visibility, presence: pointPresence, observable: score >= threshold, onScreen, modelPosition: finiteWorld ? modelPosition : null, relativeZm: finiteWorld ? modelPosition[2] : null, screenRelativeZ: screen[id * 5 + 2] * scale };
  });
  const keypoints = Array.from({ length: 133 }, (_, id) => ({ id, x: 0, y: 0, score: 0, observable: false }));
  const worldJoints = Array(133).fill(null);
  for (const [mpId, id] of POSE3D_TO_COCO) {
    const p = landmarks[mpId], approximation = mpId >= 31 ? 'foot-index-as-toe-centre' : null;
    keypoints[id] = { ...p, id, modelId: POSE3D_SPEC.id, source: 'learnedMonocular3D', depthEstimated: true, anatomicalApproximation: approximation };
    worldJoints[id] = p.observable && finite3(p.modelPosition) ? { id, positionCamera: p.modelPosition.slice(), score: p.score, source: 'inferredMonocular', modelRelativeZ: true, depthAmbiguous: false, anatomicalApproximation: approximation } : null;
  }
  const decoded = { valid, confidence, imageSize: crop.imageSize, keypoints, landmarks, worldJoints, crop, diagnostics: { method: 'learned-monocular-world-landmarks', observedBodyPoints: landmarks.filter(p => p.observable).length, worldDepthEstimated: true, absoluteDistanceMeasured: false } };
  decoded.rootFit = fitPerspectiveRoot(decoded, options);
  return decoded;
}

/** Replace the body with the current 3D model observation, including losses.
 * Old hand/face details remain separate; a stale 2D body cannot masquerade as a
 * current learned 3D observation. */
export function mergePose3DKeypoints(existing, decoded) {
  const result = Array.from({ length: 133 }, (_, id) => existing?.[id] ? { ...existing[id] } : { id, x: 0, y: 0, score: 0, observable: false });
  for (const [, id] of POSE3D_TO_COCO) result[id] = { ...decoded.keypoints[id] };
  return result;
}

/** Preserve the higher-accuracy current 2D observation. A slower world-landmark
 * model supplies relative Z only where it actually observed that joint. The
 * depth timestamp is independent of the current image-plane observation.
 * No previous fused result should be passed as decoded3D: keep the last actual
 * neural decode separate, including its confidence and original root fit. */
export function fuseBodyDepth(current133, decoded3D, timestampMs, depthTimestampMs, options = {}) {
  const threshold = options.minPointConfidence ?? .25, maxDepthAgeMs = options.maxDepthAgeMs ?? 300;
  const depthAgeMs = finite(timestampMs) && finite(depthTimestampMs) ? timestampMs - depthTimestampMs : Infinity;
  const depthFresh = depthAgeMs >= 0 && depthAgeMs <= maxDepthAgeMs && decoded3D?.valid === true;
  const imageSize = options.imageSize || decoded3D?.imageSize;
  const observed = point => point && point.observable !== false && point.score >= threshold && finite(point.x) && finite(point.y) && (!imageSize || point.x >= 0 && point.y >= 0 && point.x < imageSize.width && point.y < imageSize.height);
  const depthFields = ['modelPosition', 'relativeZm', 'relativeZ', 'depthConfidence', 'depthObservedAtMs', 'depthAgeMs', 'depthEstimated', 'depthSource', 'depthProjection'];
  const diagnostics = { method: 'current-2d-with-learned-relative-depth', primaryXYUnchanged: 0, missingXYFilledBy3D: 0, learnedDepthJoints: 0, rayProjectedJoints: 0, depthFresh, depthAgeMs: finite(depthAgeMs) ? depthAgeMs : null, absoluteDistanceMeasured: false };
  const keypoints = Array.from({ length: 133 }, (_, id) => {
    const point = current133?.[id] ? { ...current133[id] } : { id, x: 0, y: 0, score: 0, observable: false };
    // Callers may keep old details between inference frames. They cannot carry
    // old depth forward by accident through an ordinary 2D point copy.
    for (const field of depthFields) delete point[field];
    point.id = id;
    if (observed(point)) { point.observedAtMs = finite(point.observedAtMs) ? point.observedAtMs : timestampMs; diagnostics.primaryXYUnchanged++; }
    return point;
  });
  if (depthFresh) for (const [, id] of POSE3D_TO_COCO) {
    const depth = decoded3D.keypoints?.[id];
    if (!observed(depth) || !finite3(depth.modelPosition)) continue;
    let point = keypoints[id];
    if (!observed(point)) {
      point = keypoints[id] = { ...depth, id, observedAtMs: depthTimestampMs, xySource: 'learned3d-missing-2d-fallback' };
      diagnostics.missingXYFilledBy3D++;
    } else point.xySource = 'current-primary-2d';
    point.modelPosition = depth.modelPosition.slice(); point.relativeZm = depth.modelPosition[2];
    point.depthConfidence = depth.score; point.depthObservedAtMs = depthTimestampMs; point.depthAgeMs = depthAgeMs;
    point.depthEstimated = true; point.depthSource = POSE3D_SPEC.id; diagnostics.learnedDepthJoints++;
  }
  const decoded = { ...(decoded3D || {}), imageSize, valid: depthFresh, confidence: depthFresh ? decoded3D.confidence : 0, keypoints };
  decoded.rootFit = fitPerspectiveRoot(decoded, options);
  // A current successful perspective fit updates preview translation. A fresh
  // original-model fit may still project limb rays when the current fit is
  // inconsistent, but it is NOT promoted to a current root measurement.
  const projection = decoded.rootFit.valid ? decoded.rootFit : depthFresh && decoded3D.rootFit?.valid ? decoded3D.rootFit : null;
  const supplied = options.K, K = Array.isArray(supplied) && supplied.length === 9 ? [supplied[0], supplied[4], supplied[2], supplied[5]] : supplied;
  const calibrated = Array.isArray(K) && K.length === 4 && K.every(finite) && K[0] > 0 && K[1] > 0;
  const intrinsics = calibrated ? K : imageSize ? [Math.max(imageSize.width, imageSize.height), Math.max(imageSize.width, imageSize.height), imageSize.width / 2, imageSize.height / 2] : null;
  if (intrinsics) for (const point of keypoints) {
    if (!finite3(point.modelPosition)) continue;
    if (projection) {
      const [tx, ty, tz] = projection.positionCamera, [fx, fy, cx, cy] = intrinsics, z = point.relativeZm, distance = z + tz;
      const projected = [(point.x - cx) / fx * distance - tx, (point.y - cy) / fy * distance - ty, z];
      if (distance > .05 && projected.every(v => finite(v) && Math.abs(v) < 5)) {
        point.modelPosition = projected; point.depthProjection = decoded.rootFit.valid ? 'current-2d-perspective-fit' : 'fresh-depth-model-perspective-fit'; diagnostics.rayProjectedJoints++;
      } else delete point.modelPosition;
    } else if (point.xySource === 'current-primary-2d') {
      // Keep the current 2D motion and the separately labelled relative Z.
      // Without a valid projection, stale model XY must not replace current XY.
      delete point.modelPosition;
    }
  }
  decoded.worldJoints = keypoints.map((point, id) => observed(point) && finite3(point.modelPosition) ? { id, positionCamera: point.modelPosition.slice(), score: point.score, depthConfidence: point.depthConfidence, depthObservedAtMs: point.depthObservedAtMs, source: 'inferredMonocular', modelRelativeZ: true, depthAmbiguous: false } : null);
  decoded.diagnostics = { ...(decoded3D?.diagnostics || {}), ...diagnostics, rootFitUsesCurrent2D: decoded.rootFit.valid };
  return { keypoints, decoded, depthAgeMs: finite(depthAgeMs) ? depthAgeMs : null, depthFresh, diagnostics };
}

function solve3(matrix, vector) {
  const m = [0, 1, 2].map(i => [...matrix.slice(i * 3, i * 3 + 3), vector[i]]), scale = Math.max(...matrix.map(Math.abs));
  for (let col = 0; col < 3; col++) {
    let pivot = col;
    for (let row = col + 1; row < 3; row++) if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    if (Math.abs(m[pivot][col]) <= Math.max(1e-10, scale * 1e-8)) return null;
    [m[pivot], m[col]] = [m[col], m[pivot]];
    const divider = m[col][col]; for (let j = col; j < 4; j++) m[col][j] /= divider;
    for (let row = 0; row < 3; row++) if (row !== col) { const factor = m[row][col]; for (let j = col; j < 4; j++) m[row][j] -= factor * m[col][j]; }
  }
  const answer = m.map(row => row[3]); return finite3(answer) ? answer : null;
}

/** Solve translation of the learned root-relative skeleton from perspective.
 * Normalized rays give tx-u*tz=u*Z-X, ty-v*tz=v*Z-Y. Robust reweighting reduces
 * the effect of one erroneous landmark. Camera focal length is explicitly
 * assumed when K is absent; the result is a preview estimate, not measured m. */
export function fitPerspectiveRoot(decoded, options = {}) {
  const image = decoded?.imageSize, width = image?.width, height = image?.height;
  const missing = reason => ({ valid: false, positionCamera: null, metric: false, reason, method: 'learned-body-perspective-translation' });
  if (!decoded?.valid || !finite(width) || !finite(height)) return missing('noCurrentBody');
  const supplied = options.K;
  const K = Array.isArray(supplied) && supplied.length === 9 ? [supplied[0], supplied[4], supplied[2], supplied[5]] : supplied;
  const calibratedIntrinsics = Array.isArray(K) && K.length === 4 && K.every(finite) && K[0] > 0 && K[1] > 0;
  const [fx, fy, cx, cy] = calibratedIntrinsics ? K : [Math.max(width, height), Math.max(width, height), width / 2, height / 2];
  const records = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19, 22].map(id => decoded.keypoints[id]).filter(p => p?.observable && p.score >= .35 && finite3(p.modelPosition));
  if (records.length < 4) return missing('fewerThanFourVisibleJoints');
  let translation = null;
  const weights = records.map(p => Math.min(p.score, p.depthConfidence ?? p.score) ** 2), residuals = Array(records.length).fill(0);
  for (let pass = 0; pass < 3; pass++) {
    const a = Array(9).fill(0), b = Array(3).fill(0);
    records.forEach((p, index) => {
      const [X, Y, Z] = p.modelPosition, u = (p.x - cx) / fx, v = (p.y - cy) / fy;
      const weight = weights[index] * (pass ? Math.min(1, 12 / Math.max(1e-6, residuals[index])) : 1);
      for (const [row, target] of [[[1, 0, -u], u * Z - X], [[0, 1, -v], v * Z - Y]]) for (let i = 0; i < 3; i++) { b[i] += weight * row[i] * target; for (let j = 0; j < 3; j++) a[i * 3 + j] += weight * row[i] * row[j]; }
    });
    translation = solve3(a, b);
    if (!translation || translation[2] < .25 || translation[2] > 30) return missing('perspectiveFitRejected');
    records.forEach((p, index) => {
      const [X, Y, Z] = p.modelPosition, z = Z + translation[2];
      residuals[index] = z > .05 ? Math.hypot(fx * (X + translation[0]) / z + cx - p.x, fy * (Y + translation[1]) / z + cy - p.y) : Infinity;
    });
  }
  const weightSum = weights.reduce((a, b) => a + b, 0);
  const rms = Math.sqrt(residuals.reduce((sum, error, i) => sum + weights[i] * error * error, 0) / weightSum);
  const xs = records.map(p => p.x), ys = records.map(p => p.y), span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 32);
  if (!finite(rms) || rms / span > (options.maxReprojectionRatio ?? .09)) return { ...missing('reprojectionRejected'), pointCount: records.length, reprojectionRmsPx: rms };
  return { valid: true, positionCamera: translation, metric: false, calibratedIntrinsics, focalAssumed: !calibratedIntrinsics, pixelsPerMetre: Math.sqrt(fx * fy) / translation[2], pointCount: records.length, reprojectionRmsPx: rms, normalizedReprojectionError: rms / span, method: 'learned-body-perspective-translation', absoluteDistanceMeasured: false };
}
