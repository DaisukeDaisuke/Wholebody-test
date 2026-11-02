/** Low-cadence learned ROI acquisition, shared by browser and Node.
 * Adapted from OpenCV Zoo's Apache-2.0 MPPersonDet preprocessing/SSD contract.
 * Virtual centre/radius points locate a crop. They are NEVER body observations.
 */
import { createPose3DCrop } from './pose3d.mjs';

export const PERSON_DETECTOR_SPEC = Object.freeze({
  id: 'mediapipe-person-detector-opencv-2023mar',
  filename: 'person-detector.onnx', inputShape: [1, 3, 224, 224],
  bytes: 11990159, sha256: '47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f',
  upstreamRevision: '48b10fa505aeb51fbd127fc60a5c660040e823b7',
  upstreamUrl: 'https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/person_detection_mediapipe/person_detection_mediapipe_2023mar.onnx',
  license: 'Apache-2.0', color: 'RGB', layout: 'NCHW', purpose: 'ROI only',
});
export const PERSON_DETECTOR_INT8_CANDIDATE = Object.freeze({
  id: 'mediapipe-person-detector-opencv-2023mar-int8-block-candidate',
  bytes: 3482053, sha256: 'c5ed8c00c028b98e5d2c55b920a6e975af6c4cd538cfeea7c054f4fbbd8b9075',
  upstreamUrl: 'https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/person_detection_mediapipe/person_detection_mediapipe_2023mar_int8bq.onnx',
  validated: false, selected: false, opset: 21, weightOnlyBlockQuantization: true,
});
const finite = Number.isFinite;
const clamp = (n, a, b) => Math.max(a, Math.min(b, n));
const sigmoid = n => n >= 0 ? 1 / (1 + Math.exp(-n)) : Math.exp(n) / (1 + Math.exp(n));
const ANCHORS = [[28, 2], [14, 2], [7, 6]].flatMap(([size, repeat]) => Array.from({ length: size * size }, (_, index) => Array.from({ length: repeat }, () => [(index % size + .5) / size, (Math.floor(index / size) + .5) / size])).flat());

export function createPersonDetectorTransform(roi, imageSize) {
  const [x, y, width, height] = roi || [];
  if (![x, y, width, height, imageSize?.width, imageSize?.height].every(finite) || width <= 0 || height <= 0) throw new Error('invalidPersonDetectorRoi');
  const ratio = 224 / Math.max(width, height), scaledWidth = Math.max(1, Math.floor(width * ratio)), scaledHeight = Math.max(1, Math.floor(height * ratio));
  return { roi: [x, y, width, height], scaledWidth, scaledHeight, padX: Math.floor((224 - scaledWidth) / 2), padY: Math.floor((224 - scaledHeight) / 2), inputWH: [224, 224], imageSize: { ...imageSize } };
}

/** The detector circle already includes a full-body margin. The SimCC model
 * gets its bounding square with no second padding multiplier; cropTransform
 * still expands one axis to the model aspect ratio. Do not clip offscreen
 * bounds: clipping changes the virtual hip centre in partial-body frames. */
export function bodyRoiFromCrop(crop, imageSize = crop?.imageSize) {
  if (![crop?.cx, crop?.cy, crop?.side, imageSize?.width, imageSize?.height].every(finite) || crop.side <= 0) throw new Error('invalidSharedBodyCrop');
  return { roi: [crop.cx - crop.side / 2, crop.cy - crop.side / 2, crop.side, crop.side], padding: 1 };
}

/** Canvas pixels after drawing source ROI into the transform's letterbox.
 * Padding is normalized zero, as in the upstream model (not black/-1). */
export function personDetectorRgbaToTensor(rgba, transform) {
  if (rgba?.length !== 224 * 224 * 4) throw new Error('invalidPersonDetectorPixels');
  const n = 224 * 224, result = new Float32Array(n * 3), t = transform;
  for (let y = t.padY; y < t.padY + t.scaledHeight; y++) for (let x = t.padX; x < t.padX + t.scaledWidth; x++) {
    const i = y * 224 + x; for (let c = 0; c < 3; c++) result[c * n + i] = rgba[i * 4 + c] / 127.5 - 1;
  }
  return result;
}

export function samplePersonDetectorRgb(rgb, width, height, transform) {
  if (rgb?.length !== width * height * 3) throw new Error('invalidPersonDetectorRgb');
  const n = 224 * 224, result = new Float32Array(n * 3), t = transform, [left, top, rw, rh] = t.roi;
  const value = (x, y, c) => x >= 0 && y >= 0 && x < width && y < height ? rgb[(y * width + x) * 3 + c] : 0;
  for (let y = t.padY; y < t.padY + t.scaledHeight; y++) for (let x = t.padX; x < t.padX + t.scaledWidth; x++) {
    const sx = left + (x - t.padX + .5) * rw / t.scaledWidth - .5, sy = top + (y - t.padY + .5) * rh / t.scaledHeight - .5;
    const x0 = Math.floor(sx), y0 = Math.floor(sy), dx = sx - x0, dy = sy - y0;
    for (let c = 0; c < 3; c++) result[c * n + y * 224 + x] = ((1 - dy) * ((1 - dx) * value(x0, y0, c) + dx * value(x0 + 1, y0, c)) + dy * ((1 - dx) * value(x0, y0 + 1, c) + dx * value(x0 + 1, y0 + 1, c))) / 127.5 - 1;
  }
  return result;
}

function iou(a, b) {
  const [ax, ay, aw, ah] = a, [bx, by, bw, bh] = b;
  const intersection = Math.max(0, Math.min(ax + aw, bx + bw) - Math.max(ax, bx)) * Math.max(0, Math.min(ay + ah, by + bh) - Math.max(ay, by));
  return intersection / Math.max(1e-6, aw * ah + bw * bh - intersection);
}

export function decodePersonDetector(outputs, transform, { minScore = .5, nmsThreshold = .3 } = {}) {
  const values = Object.values(outputs || {}), boxes = values.find(t => t?.data?.length === 2254 * 12)?.data, logits = values.find(t => t?.data?.length === 2254)?.data;
  if (!boxes || !logits || ANCHORS.length !== 2254) throw new Error('personDetectorOutputContract');
  const [left, top, width, height] = transform.roi;
  const point = (nx, ny) => [left + (nx - transform.padX) * width / transform.scaledWidth, top + (ny - transform.padY) * height / transform.scaledHeight];
  const candidates = [];
  for (let id = 0; id < 2254; id++) {
    const score = sigmoid(logits[id]); if (!finite(score) || score < minScore) continue;
    const offset = id * 12, ax = ANCHORS[id][0] * 224, ay = ANCHORS[id][1] * 224;
    const values = Array.from(boxes.slice(offset, offset + 12)); if (!values.every(finite) || values[2] <= 0 || values[3] <= 0) continue;
    const a = point(ax + values[0] - values[2] / 2, ay + values[1] - values[3] / 2), b = point(ax + values[0] + values[2] / 2, ay + values[1] + values[3] / 2);
    const [hip, fullBodyPoint, shoulder, upperBodyPoint] = [0, 1, 2, 3].map(i => point(ax + values[4 + i * 2], ay + values[5 + i * 2]));
    const radius = Math.hypot(fullBodyPoint[0] - hip[0], fullBodyPoint[1] - hip[1]);
    if (!(radius >= 8 && radius <= Math.max(width, height) * 3) || ![...hip, ...fullBodyPoint, ...shoulder].every(finite)) continue;
    const crop = { cx: hip[0], cy: hip[1], side: radius * 2, angleRad: Math.atan2(fullBodyPoint[0] - hip[0], hip[1] - fullBodyPoint[1]), inputWH: [256, 256], imageSize: { ...transform.imageSize }, source: 'learned-detector-virtual-circle' };
    candidates.push({ score, boxXYWH: [a[0], a[1], b[0] - a[0], b[1] - a[1]], hip, fullBodyPoint, shoulder, upperBodyPoint, crop, virtualPointsAreBodyObservations: false });
  }
  candidates.sort((a, b) => b.score - a.score);
  const selected = [];
  for (const candidate of candidates) { if (selected.every(old => iou(old.boxXYWH, candidate.boxXYWH) <= nmsThreshold)) selected.push(candidate); if (selected.length >= 8) break; }
  return selected;
}

/** Independent crop state, not a pose extrapolator. Current inference still
 * determines which joints exist. An offscreen pelvis can locate a crop but
 * never becomes an observed pelvis, root, tracker or skeleton joint. */
export class Pose3DRoiTracker {
  constructor({ detectorIntervalMs = 500, cropHoldMs = 1200 } = {}) { this.detectorIntervalMs = Math.max(500, detectorIntervalMs); this.cropHoldMs = cropHoldMs; this.reset(); }
  reset() { this.lastDetectionAt = -Infinity; this.lastUpdateAt = -Infinity; this.lastGoodAt = -Infinity; this.lastGoodCrop = null; this.detectedCrop = null; this.freshDetection = false; this.keypoints = null; this.poseGood = false; this.torsoGood = false; }
  shouldDetect(timestampMs) {
    if (timestampMs < this.lastUpdateAt) this.reset();
    return timestampMs - this.lastDetectionAt >= this.detectorIntervalMs && (!this.poseGood || !this.torsoGood || this.lastDetectionAt === -Infinity);
  }
  acceptDetections(detections, timestampMs) {
    this.lastDetectionAt = timestampMs;
    const prior = this.lastGoodCrop, ranked = [...(detections || [])].sort((a, b) => {
      const rank = d => d.score * Math.sqrt(d.crop.side) / (prior ? 1 + Math.hypot(d.crop.cx - prior.cx, d.crop.cy - prior.cy) / Math.max(32, prior.side) : 1);
      return rank(b) - rank(a);
    });
    this.detectedCrop = ranked[0]?.crop || null; this.freshDetection = !!this.detectedCrop;
    return ranked[0] || null;
  }
  nextCrop(roi, imageSize, timestampMs) {
    if (this.freshDetection) { this.freshDetection = false; return { ...this.detectedCrop, imageSize: { ...imageSize } }; }
    if (this.torsoGood && this.keypoints) {
      const prior = this.lastGoodCrop, trackedRoi = prior ? [prior.cx - prior.side / 2, prior.cy - prior.side / 2, prior.side, prior.side] : roi;
      const next = createPose3DCrop(trackedRoi, this.keypoints, imageSize);
      // The torso gate must agree with createPose3DCrop. If anchors disappear,
      // never feed the previous crop through person-ROI padding again: repeated
      // 1.25x expansion makes the person vanish even while detection succeeds.
      if (next.source === 'observed-hip-torso') return next;
    }
    if (this.lastGoodCrop && timestampMs - this.lastGoodAt <= this.cropHoldMs) return { ...this.lastGoodCrop, imageSize: { ...imageSize }, source: 'recent-crop-only-not-held-landmarks' };
    return createPose3DCrop(roi, [], imageSize);
  }
  update(decoded, crop, timestampMs) {
    this.lastUpdateAt = timestampMs; this.keypoints = decoded?.keypoints || null;
    const size = decoded?.imageSize || crop?.imageSize;
    const observed = (id, threshold = .25) => {
      const point = this.keypoints?.[id];
      return point?.observable !== false && point?.score >= threshold && finite(point.x) && finite(point.y) && point.x >= 0 && point.y >= 0 && point.x < size?.width && point.y < size?.height;
    };
    this.poseGood = decoded?.valid === true && [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].filter(id => observed(id)).length >= 4;
    this.torsoGood = this.poseGood && [5, 6, 11, 12].every(id => observed(id, .35));
    if (this.poseGood) { this.lastGoodCrop = { ...crop }; this.lastGoodAt = timestampMs; }
  }
}
