/** Causal BootsTAPIR 256px / three-query contract. No model or image is uploaded. */
export const TAPIR_INPUT_WH = Object.freeze([256, 256]);
export const TAPIR_QUERY_COUNT = 3;
export const TAPIR_NAMES = Object.freeze({ encoder: { inputs: ['frame'], outputs: ['lowres', 'hires'] },
  query: { inputs: ['lowres', 'hires', 'query_yx'], outputs: ['query_lowres', 'query_hires'] },
  step: { inputs: ['lowres', 'hires', 'query_lowres', 'query_hires', 'causal_state'], outputs: ['tracks', 'occlusion', 'expected_dist', 'new_causal_state'] } });

function imageSize(image) {
  const width = image?.width, height = image?.height;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) throw new Error('TAPIR requires a nonempty image.');
  return { width, height };
}
/** RGB uint8 -> float bilinear resize, align_corners=false -> NHWC [-1,1]. */
export function tapirFrameTensor(image) {
  const { width, height } = imageSize(image), input = image.data;
  const channels = input?.length === width * height * 4 ? 4 : input?.length === width * height * 3 ? 3 : 0;
  if (!channels) throw new Error('TAPIR input must contain RGB or RGBA pixels.');
  const output = new Float32Array(256 * 256 * 3);
  for (let y = 0; y < 256; y++) {
    const sy = Math.max(0, Math.min(height - 1, (y + .5) * height / 256 - .5)), y0 = Math.floor(sy), y1 = Math.min(height - 1, y0 + 1), fy = sy - y0;
    for (let x = 0; x < 256; x++) {
      const sx = Math.max(0, Math.min(width - 1, (x + .5) * width / 256 - .5)), x0 = Math.floor(sx), x1 = Math.min(width - 1, x0 + 1), fx = sx - x0;
      const offsets = [(y0 * width + x0) * channels, (y0 * width + x1) * channels, (y1 * width + x0) * channels, (y1 * width + x1) * channels];
      for (let c = 0; c < 3; c++) {
        const a = input[offsets[0] + c] * (1 - fx) + input[offsets[1] + c] * fx;
        const b = input[offsets[2] + c] * (1 - fx) + input[offsets[3] + c] * fx;
        output[(y * 256 + x) * 3 + c] = (a * (1 - fy) + b * fy) / 255 * 2 - 1;
      }
    }
  }
  return output;
}
/** Original-image [x,y] pixels -> first-frame [y,x] model pixels. */
export function tapirQueryTensor(points, image) {
  const { width, height } = imageSize(image);
  if (!Array.isArray(points) || points.length !== TAPIR_QUERY_COUNT) throw new Error('This TAPIR model requires exactly three initial query points.');
  const output = new Float32Array(TAPIR_QUERY_COUNT * 2);
  points.forEach((point, i) => {
    const x = Array.isArray(point) ? point[0] : point?.x, y = Array.isArray(point) ? point[1] : point?.y;
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x >= width || y >= height) throw new Error('A TAPIR query is outside the selected image.');
    output[i * 2] = y * 256 / height; output[i * 2 + 1] = x * 256 / width;
  });
  return output;
}
const data = tensor => ArrayBuffer.isView(tensor) || Array.isArray(tensor) ? tensor : tensor?.data;
function sigmoid(value) { return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value)); }
export function decodeTapir(outputs, image) {
  const { width, height } = imageSize(image), tracks = data(outputs?.tracks), occlusion = data(outputs?.occlusion), expected = data(outputs?.expected_dist);
  if (tracks?.length !== 6 || occlusion?.length !== 3 || expected?.length !== 3) throw new Error('Unexpected TAPIR output size; use the pinned three-point model.');
  return Array.from({ length: 3 }, (_, i) => {
    const values = [tracks[i * 2], tracks[i * 2 + 1], occlusion[i], expected[i]], finite = values.every(Number.isFinite);
    const score = finite ? (1 - sigmoid(occlusion[i])) * (1 - sigmoid(expected[i])) : 0;
    return { x: finite ? tracks[i * 2] * width / 256 : null, y: finite ? tracks[i * 2 + 1] * height / 256 : null,
      score, observable: finite && score > .5, rawOcclusion: finite ? occlusion[i] : null, rawExpectedDist: finite ? expected[i] : null,
      source: 'learnedPointTrack2D', depthKind: 'none' };
  });
}
export function createTapirCausalState(manifest) {
  const shape = manifest?.stateShape;
  if (!Array.isArray(shape) || shape.length !== 1 || !Number.isInteger(shape[0]) || shape[0] < 1 || shape[0] > 16_000_000 || manifest.queryCount !== 3) throw new Error('Invalid TAPIR causal-state manifest.');
  return new Float32Array(shape[0]);
}
