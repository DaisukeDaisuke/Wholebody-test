import { mkdir, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { createHash } from 'node:crypto';
import { TemporalPoseTracker } from '../dist/geometry.mjs';

if (process.env.GITHUB_ACTIONS !== 'true') throw new Error('Run this evaluation in GitHub Actions.');

const output = process.argv[2] || 'evaluation/results/panoptic-report.json';
const sequence = '171204_pose1_sample';
const sources = [
  `https://domedb.perception.cs.cmu.edu/webdata/dataset/${sequence}/hdPose3d_stage1_coco19.tar`,
  `http://domedb.perception.cs.cmu.edu/webdata/dataset/${sequence}/hdPose3d_stage1_coco19.tar`,
  `https://vcl.snu.ac.kr/panoptic/webdata/dataset/${sequence}/hdPose3d_stage1_coco19.tar`,
  `http://vcl.snu.ac.kr/panoptic/webdata/dataset/${sequence}/hdPose3d_stage1_coco19.tar`,
];
const references = [
  'https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox',
  'https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/blob/master/scripts/getData.sh',
  'https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/issues/18',
  'https://openaccess.thecvf.com/content_iccv_2015/html/Joo_Panoptic_Studio_A_ICCV_2015_paper.html',
];
const toCoco = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14];
const missing = id => ({ id, position: null, score: 0, source: 'missing' });
const distance = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));

async function download(url) {
  const maximumBytes = 64 * 1024 * 1024;
  const response = await fetch(url, { signal: AbortSignal.timeout(30000), redirect: 'follow' });
  if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);
  const size = Number(response.headers.get('content-length'));
  if (Number.isFinite(size) && size > maximumBytes) throw new Error(`Archive is larger than the ${maximumBytes} byte evaluation limit.`);
  const reader = response.body.getReader(), chunks = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.length;
      if (total > maximumBytes) throw new Error('Archive exceeded evaluation download limit.');
      chunks.push(value);
    }
  } catch (error) { await reader.cancel().catch(() => {}); throw error; }
  return Buffer.concat(chunks, total);
}

// Parse regular JSON entries without extracting paths or executing archive contents.
function readPoseArchive(bytes) {
  const frames = [];
  for (let offset = 0; offset + 512 <= bytes.length;) {
    const header = bytes.subarray(offset, offset + 512);
    if (header.every(v => v === 0)) break;
    const name = header.subarray(0, 100).toString('utf8').replace(/\0.*$/s, '');
    const sizeText = header.subarray(124, 136).toString('ascii').replace(/\0.*$/s, '').trim();
    const size = Number.parseInt(sizeText, 8);
    if (!Number.isSafeInteger(size) || size < 0 || offset + 512 + size > bytes.length) throw new Error('Invalid or truncated TAR archive.');
    const type = header[156], match = name.match(/body3DScene_(\d+)\.json$/);
    if ((type === 0 || type === 48) && match && size <= 1024 * 1024) {
      const frame = JSON.parse(bytes.subarray(offset + 512, offset + 512 + size).toString('utf8'));
      if (Array.isArray(frame.bodies)) frames.push({ index: Number(match[1]), frame });
    }
    offset += 512 + Math.ceil(size / 512) * 512;
  }
  if (!frames.length) throw new Error('No Panoptic body3DScene JSON frames were found.');
  frames.sort((a, b) => a.index - b.index);
  return frames;
}

function referenceSequence(frames) {
  const counts = new Map();
  for (const { frame } of frames) for (const body of frame.bodies) if (Array.isArray(body.joints19) && body.joints19.length === 76) counts.set(body.id, (counts.get(body.id) || 0) + 1);
  const selectedId = [...counts].sort((a, b) => b[1] - a[1])[0]?.[0];
  if (selectedId === undefined) throw new Error('No stable subject with joints19 labels is available.');
  const selected = frames.flatMap(({ index, frame }) => {
    const body = frame.bodies.find(b => b.id === selectedId && Array.isArray(b.joints19) && b.joints19.length === 76);
    if (!body) return [];
    const joints = Array.from({ length: 133 }, (_, i) => missing(i));
    for (let i = 0; i < 17; i++) {
      const offset = toCoco[i] * 4, values = body.joints19.slice(offset, offset + 4), [x, y, z, confidence] = values;
      if (values.every(Number.isFinite) && confidence >= .3) joints[i] = { id: i, position: [x / 100, -y / 100, z / 100], score: Math.min(1, confidence), source: 'measuredMultiView' };
    }
    return [{ index, timestampMs: index * 1001 / 30, joints }];
  }).slice(0, 900);
  if (selected.length < 90) throw new Error(`Only ${selected.length} usable frames; at least 90 are required.`);
  return { subjectId: selectedId, frames: selected, detectedSubjectCount: counts.size };
}

function summary(errors) {
  if (!errors.length) return { count: 0, mpjpeM: null, medianM: null, p95M: null, maximumM: null };
  const sorted = errors.slice().sort((a, b) => a - b);
  return { count: errors.length, mpjpeM: errors.reduce((s, v) => s + v, 0) / errors.length, medianM: sorted[Math.floor(sorted.length / 2)], p95M: sorted[Math.floor(.95 * (sorted.length - 1))], maximumM: sorted.at(-1) };
}

function evaluateGap(frames, hiddenIds, gapFrames, label) {
  const tracker = new TemporalPoseTracker(), previousObserved = new Map(), errors = [], heldErrors = [], ages = [], sourceCounts = {};
  let eligible = 0, absent = 0, maskedFrames = 0;
  for (let index = 0; index < frames.length; index++) {
    const truth = frames[index];
    const hide = index >= 30 && (index - 30) % 120 < gapFrames;
    if (hide) maskedFrames++;
    const joints = truth.joints.map((j, id) => hide && hiddenIds.includes(id) ? missing(id) : { ...j, position: j.position?.slice() || null });
    for (let id = 0; id < 17; id++) if (joints[id].position) previousObserved.set(id, joints[id].position.slice());
    const estimated = tracker.update({ timestampMs: truth.timestampMs, joints, metric: true, coordinateFrame: 'world', units: 'm', worldCalibrationId: 'panoptic-sample-reference' });
    if (!hide) continue;
    for (const id of hiddenIds) {
      if (!truth.joints[id].position || !previousObserved.has(id)) continue;
      eligible++;
      const predicted = estimated.joints[id];
      if (predicted?.position && predicted.position.every(Number.isFinite)) {
        errors.push(distance(predicted.position, truth.joints[id].position));
        heldErrors.push(distance(previousObserved.get(id), truth.joints[id].position));
        ages.push(predicted.ageMs);
        sourceCounts[predicted.source] = (sourceCounts[predicted.source] || 0) + 1;
      } else absent++;
    }
  }
  const trackerSummary = summary(errors), holdLastSummary = summary(heldErrors);
  return { label, hiddenCocoJointIds: hiddenIds, gapFrames, nominalGapMs: gapFrames * 1001 / 30, maskedFrames, eligibleJointFrames: eligible, missingPredictions: absent, coverage: eligible ? errors.length / eligible : null, predictionAgeMs: ages.length ? { min: Math.min(...ages), max: Math.max(...ages) } : null, tracker: trackerSummary, holdLastVisible: holdLastSummary, relativeImprovementOverHold: holdLastSummary.mpjpeM > 0 ? 1 - trackerSummary.mpjpeM / holdLastSummary.mpjpeM : null, predictionSourceCounts: sourceCounts };
}

const report = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  gitCommit: process.env.GITHUB_SHA || null,
  runId: process.env.GITHUB_RUN_ID || null,
  status: 'unavailable',
  sequence,
  referenceType: 'Real motion represented by Panoptic multiview-reconstructed 3D body labels; not independent motion-capture ground truth.',
  occlusionType: 'Artificially remove 3D label observations for selected time windows; the images themselves are not evaluated.',
  scope: 'Causal temporal-completion error on real recorded movement. No claim about real-camera image detection, fingers or toe articulation.',
  realVideoAccuracyMeasured: false,
  realMotionOcclusionMeasured: false,
  individualFingerAccuracyMeasured: false,
  toeArticulationAccuracyMeasured: false,
  datasetLicense: 'Panoptic Studio Dataset: non-commercial and research use only. Raw data is not redistributed by this experiment.',
  coordinateConversion: 'Official label units are centimetres. Convert to metres and negate world Y for the application display convention.',
  timeBase: 'HD frame index × (1001/30) milliseconds (29.97 fps).',
  references,
  attempts: [],
  experiments: [],
};

for (const url of sources) {
  try {
    const archive = await download(url), frames = readPoseArchive(archive), selected = referenceSequence(frames);
    report.dataset = { url, sha256: createHash('sha256').update(archive).digest('hex'), downloadedBytes: archive.length, archiveFrameCount: frames.length, evaluatedFrameCount: selected.frames.length, firstFrameIndex: selected.frames[0].index, lastFrameIndex: selected.frames.at(-1).index, subjectId: selected.subjectId, detectedSubjectCount: selected.detectedSubjectCount };
    for (const gap of [3, 9, 30]) {
      report.experiments.push(evaluateGap(selected.frames, [7, 9], gap, 'left-elbow-and-wrist-hidden'));
      report.experiments.push(evaluateGap(selected.frames, [15, 16], gap, 'both-ankles-hidden'));
      report.experiments.push(evaluateGap(selected.frames, Array.from({ length: 17 }, (_, i) => i), gap, 'whole-body-observations-hidden'));
    }
    report.status = report.experiments.every(e => e.tracker.count > 0) ? 'measured' : 'insufficient-label-coverage';
    report.realMotionOcclusionMeasured = report.status === 'measured';
    report.attempts.push({ url, status: 'downloaded' });
    break;
  } catch (error) {
    report.attempts.push({ url, status: 'failed', reason: String(error.message).slice(0, 400) });
  }
}

await mkdir(dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
console.log(JSON.stringify({ output, status: report.status, evaluatedFrames: report.dataset?.evaluatedFrameCount || 0, realVideoAccuracyMeasured: false, realMotionOcclusionMeasured: report.realMotionOcclusionMeasured, experiments: report.experiments.map(e => ({ label: e.label, gapMs: e.nominalGapMs, count: e.tracker.count, mpjpeM: e.tracker.mpjpeM, holdLastMpjpeM: e.holdLastVisible.mpjpeM })) }));
// A public dataset outage is reported as unavailable, never as a passed accuracy result.
