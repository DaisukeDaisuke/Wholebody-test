/* Camera coordinates: x right, y down, z forward. World: right-handed, y up.
 * Rcw and tcw map world metres to camera metres: Pc = Rcw * Pw + tcw.
 * Image coordinates always refer to the original, unmirrored video frame. */
const EPS = 1e-10;
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const scale = (a, s) => a.map(v => v * s);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => Math.hypot(...a);
const unit = a => norm(a) > EPS ? scale(a, 1 / norm(a)) : null;
const finiteArray = (a, n) => Array.isArray(a) && a.length === n && a.every(Number.isFinite);
const mv = (a, b) => [dot(a.slice(0, 3), b), dot(a.slice(3, 6), b), dot(a.slice(6, 9), b)];
const mt = a => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
const mm = (a, b) => {
  const out = Array(9).fill(0);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) for (let k = 0; k < 3; k++) out[r * 3 + c] += a[r * 3 + k] * b[k * 3 + c];
  return out;
};

export function intrinsics4(K) {
  if (finiteArray(K, 4) && K[0] > 0 && K[1] > 0) return K.slice();
  if (finiteArray(K, 9) && K[0] > 0 && K[4] > 0 && Math.abs(K[1]) < 1e-8 && Math.abs(K[3]) < 1e-8 && Math.abs(K[6]) < 1e-8 && Math.abs(K[7]) < 1e-8 && Math.abs(K[8] - 1) < 1e-8) return [K[0], K[4], K[2], K[5]];
  throw new Error('K must be [fx, fy, cx, cy] or a zero-skew 3 × 3 camera matrix.');
}

export function undistortPixel(pixel, K, distortion = [0, 0, 0, 0, 0]) {
  const [fx, fy, cx, cy] = intrinsics4(K);
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = distortion;
  const xd = (pixel[0] - cx) / fx, yd = (pixel[1] - cy) / fy;
  if (!Number.isFinite(xd) || !Number.isFinite(yd)) return null;
  let x = xd, y = yd;
  for (let i = 0; i < 12; i++) {
    const r2 = x * x + y * y, radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    if (!Number.isFinite(radial) || Math.abs(radial) < 1e-8) return null;
    const nx = (xd - 2 * p1 * x * y - p2 * (r2 + 2 * x * x)) / radial;
    const ny = (yd - p1 * (r2 + 2 * y * y) - 2 * p2 * x * y) / radial;
    if (Math.hypot(nx - x, ny - y) < 1e-10) { x = nx; y = ny; break; }
    x = nx; y = ny;
  }
  const r2 = x * x + y * y, radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
  const ex = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) - xd;
  const ey = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y - yd;
  return Number.isFinite(x) && Number.isFinite(y) && Math.hypot(ex * fx, ey * fy) < 0.25 ? [x, y] : null;
}

export function projectCameraPoint(point, camera) {
  if (!finiteArray(point, 3) || point[2] <= EPS) return null;
  const [fx, fy, cx, cy] = intrinsics4(camera.K);
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0] = camera.distortion || [];
  const x = point[0] / point[2], y = point[1] / point[2], r2 = x * x + y * y;
  const radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
  return [fx * (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + cx, fy * (y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + cy];
}

export function worldToCamera(point, camera) { return add(mv(camera.Rcw, point), camera.tcw); }
export function cameraToWorld(point, camera) { return mv(mt(camera.Rcw), sub(point, camera.tcw)); }
export function projectWorldPoint(point, camera) { return projectCameraPoint(worldToCamera(point, camera), camera); }

export function cameraAtImageSize(camera, imageSize) {
  const from = camera.imageSize;
  if (!from || !imageSize || from.width <= 0 || from.height <= 0 || imageSize.width <= 0 || imageSize.height <= 0) return null;
  const sx = imageSize.width / from.width, sy = imageSize.height / from.height;
  // A changed aspect ratio could be sensor cropping or a rotated frame, not rescaling.
  if (Math.abs(sx / sy - 1) > 0.02) return null;
  const [fx, fy, cx, cy] = intrinsics4(camera.K);
  return { ...camera, K: [fx * sx, fy * sy, cx * sx, cy * sy], imageSize: { ...imageSize } };
}

export function validateCalibration(input) {
  const cameras = Array.isArray(input) ? input : input?.cameras;
  const errors = [], ids = new Set();
  if (!Array.isArray(cameras) || cameras.length < 1 || cameras.length > 8) return { valid: false, errors: ['cameras must contain 1–8 cameras.'], cameras: [] };
  const normalized = cameras.map((raw, index) => {
    const c = { ...raw }, tag = c.id || `camera[${index}]`;
    if (typeof c.id !== 'string' || !c.id || ids.has(c.id)) errors.push(`${tag}: camera ID is missing or duplicated.`);
    ids.add(c.id);
    try { c.K = intrinsics4(c.K); } catch (e) { errors.push(`${tag}: ${e.message}`); }
    if (!Number.isFinite(c.imageSize?.width) || !Number.isFinite(c.imageSize?.height) || c.imageSize.width <= 0 || c.imageSize.height <= 0) errors.push(`${tag}: imageSize is invalid.`);
    if (!finiteArray(c.Rcw, 9) || !finiteArray(c.tcw, 3)) errors.push(`${tag}: Rcw/tcw must contain 9/3 finite numbers.`);
    else {
      const rt = mm(c.Rcw, mt(c.Rcw));
      const e = Math.max(...rt.map((v, i) => Math.abs(v - (i % 4 === 0 ? 1 : 0))));
      const det = dot(c.Rcw.slice(0, 3), cross(c.Rcw.slice(3, 6), c.Rcw.slice(6, 9)));
      if (e > 0.015 || Math.abs(det - 1) > 0.015) errors.push(`${tag}: Rcw is not a proper orthonormal rotation.`);
    }
    c.distortion = c.distortion || [0, 0, 0, 0, 0];
    if (!finiteArray(c.distortion, 5)) errors.push(`${tag}: only the five-coefficient Brown distortion model is supported.`);
    if (c.calibrated !== true && c.calibrationMethod !== 'manualEstimate') errors.push(`${tag}: calibration must be confirmed or explicitly marked manualEstimate.`);
    return c;
  });
  return { valid: errors.length === 0, errors, cameras: normalized };
}

export function makeCamera({ id, imageSize, horizontalFovDeg = 65, position = [0, 1.3, 0], target = [0, 1, 3], K, distortion = [0, 0, 0, 0, 0], intrinsicsCalibrated = false }) {
  if (!imageSize || !Number.isFinite(imageSize.width) || !Number.isFinite(imageSize.height) || imageSize.width <= 0 || imageSize.height <= 0 || !finiteArray(position, 3) || !finiteArray(target, 3)) throw new Error('Camera image size, position and target are required.');
  if (!Number.isFinite(horizontalFovDeg) || horizontalFovDeg <= 1 || horizontalFovDeg >= 175) throw new Error('Horizontal field of view must be between 1 and 175 degrees.');
  const forward = unit(sub(target, position));
  if (!forward) throw new Error('Camera position and target must differ.');
  const right = unit(cross(forward, [0, 1, 0]));
  if (!right) throw new Error('Camera target cannot be exactly vertical.');
  const down = unit(cross(forward, right));
  const Rcw = [...right, ...down, ...forward];
  const fx = imageSize.width / (2 * Math.tan(horizontalFovDeg * Math.PI / 360));
  return { id, imageSize: { ...imageSize }, K: K ? intrinsics4(K) : [fx, fx, imageSize.width / 2, imageSize.height / 2], distortion: distortion.slice(), Rcw, tcw: scale(mv(Rcw, position), -1), calibrated: false, intrinsicsCalibrated: Boolean(K && intrinsicsCalibrated), calibrationMethod: 'manualEstimate', worldUnits: 'm' };
}

function solveLinear(matrix, rhs) {
  const n = rhs.length, a = matrix.map((r, i) => [...r, rhs[i]]);
  for (let c = 0; c < n; c++) {
    let pivot = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[pivot][c])) pivot = r;
    if (Math.abs(a[pivot][c]) < 1e-12) return null;
    [a[c], a[pivot]] = [a[pivot], a[c]];
    const d = a[c][c];
    for (let j = c; j <= n; j++) a[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) {
      const f = a[r][c];
      for (let j = c; j <= n; j++) a[r][j] -= f * a[c][j];
    }
  }
  return a.map(r => r[n]);
}

function normalizePoints(points) {
  const center = points.reduce((s, p) => [s[0] + p[0] / points.length, s[1] + p[1] / points.length], [0, 0]);
  const mean = points.reduce((s, p) => s + Math.hypot(p[0] - center[0], p[1] - center[1]) / points.length, 0);
  if (mean < EPS) throw new Error('Calibration points have no spatial extent.');
  const s = Math.SQRT2 / mean;
  return { points: points.map(p => [(p[0] - center[0]) * s, (p[1] - center[1]) * s]), T: [s, 0, -s * center[0], 0, s, -s * center[1], 0, 0, 1], inverseT: [1 / s, 0, center[0], 0, 1 / s, center[1], 0, 0, 1] };
}

/** Estimate a camera pose from a measured planar target and already-known intrinsics.
 * worldPoints are 2D coordinates in metres on that plane; they are not pixel points.
 * This does not estimate intrinsics from a single view or claim a guessed FOV is calibrated. */
export function solvePlanarExtrinsics({ id, imageSize, K, distortion = [0, 0, 0, 0, 0], imagePoints, worldPoints, worldOrigin = [0, 0, 0], worldAxisX = [1, 0, 0], worldAxisY = [0, 0, 1], intrinsicsCalibrated = false, maxReprojectionErrorPx = 3 }) {
  if (!Array.isArray(imagePoints) || !Array.isArray(worldPoints) || imagePoints.length !== worldPoints.length || imagePoints.length < 4 || imagePoints.length > 100) throw new Error('Use 4–100 matching target points.');
  if (imagePoints.some(p => !finiteArray(p, 2)) || worldPoints.some(p => !finiteArray(p, 2))) throw new Error('Target coordinates must be finite 2D pairs.');
  const axisX = unit(worldAxisX), axisY = unit(worldAxisY);
  if (!axisX || !axisY || Math.abs(dot(axisX, axisY)) > 1e-6 || !finiteArray(worldOrigin, 3)) throw new Error('The target axes must be orthogonal unit directions.');
  const normalizedPixels = imagePoints.map(p => undistortPixel(p, K, distortion));
  if (normalizedPixels.some(p => !p)) throw new Error('The target could not be undistorted with this lens model.');
  const wp = normalizePoints(worldPoints), ip = normalizePoints(normalizedPixels);
  const ata = Array.from({ length: 8 }, () => Array(8).fill(0)), atb = Array(8).fill(0);
  for (let i = 0; i < wp.points.length; i++) {
    const [x, y] = wp.points[i], [u, v] = ip.points[i];
    const rows = [[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]];
    for (let j = 0; j < 2; j++) for (let a = 0; a < 8; a++) {
      atb[a] += rows[j][a] * (j === 0 ? u : v);
      for (let b = 0; b < 8; b++) ata[a][b] += rows[j][a] * rows[j][b];
    }
  }
  const solved = solveLinear(ata, atb);
  if (!solved) throw new Error('The target points are collinear or do not constrain a camera pose.');
  const H = mm(mm(ip.inverseT, [...solved, 1]), wp.T);
  const h1 = [H[0], H[3], H[6]], h2 = [H[1], H[4], H[7]], h3 = [H[2], H[5], H[8]];
  const lambda = (h3[2] >= 0 ? 1 : -1) * 2 / (norm(h1) + norm(h2));
  const r1 = unit(scale(h1, lambda));
  if (!r1) throw new Error('The target orientation is degenerate.');
  const r2 = unit(sub(scale(h2, lambda), scale(r1, dot(r1, scale(h2, lambda)))));
  if (!r1 || !r2) throw new Error('The target orientation is degenerate.');
  const r3 = cross(r1, r2), normal = cross(axisX, axisY);
  const boardToCamera = [r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]];
  const worldBasis = [axisX[0], axisY[0], normal[0], axisX[1], axisY[1], normal[1], axisX[2], axisY[2], normal[2]];
  const Rcw = mm(boardToCamera, mt(worldBasis)), tcw = sub(scale(h3, lambda), mv(Rcw, worldOrigin));
  const camera = { id, imageSize: { ...imageSize }, K: intrinsics4(K), distortion: distortion.slice(), Rcw, tcw, calibrated: false, intrinsicsCalibrated, calibrationMethod: 'planar-known-intrinsics', worldUnits: 'm' };
  const residuals = worldPoints.map((p, i) => {
    const world = add(worldOrigin, add(scale(axisX, p[0]), scale(axisY, p[1])));
    const projected = projectWorldPoint(world, camera);
    return projected ? Math.hypot(projected[0] - imagePoints[i][0], projected[1] - imagePoints[i][1]) : Infinity;
  });
  camera.reprojectionRmsPx = Math.sqrt(residuals.reduce((s, r) => s + r * r, 0) / residuals.length);
  camera.maxReprojectionErrorPx = Math.max(...residuals);
  camera.calibrated = intrinsicsCalibrated === true && camera.reprojectionRmsPx <= maxReprojectionErrorPx && camera.maxReprojectionErrorPx <= maxReprojectionErrorPx * 2;
  camera.validation = { residualsPx: residuals, valid: camera.calibrated, reason: !intrinsicsCalibrated ? 'intrinsicsUncalibrated' : !camera.calibrated ? 'reprojectionError' : 'ok' };
  return camera;
}
