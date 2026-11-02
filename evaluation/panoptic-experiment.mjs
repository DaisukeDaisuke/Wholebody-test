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
const strictUtf8 = new TextDecoder('utf-8', { fatal: true });
const missing = id => ({ id, position: null, score: 0, source: 'missing' });
const distance = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));

async function download(url, maximumBytes = 64 * 1024 * 1024, timeoutMs = 30000) {
  const response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs), redirect: 'follow' });
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
      const frame = JSON.parse(strictUtf8.decode(bytes.subarray(offset + 512, offset + 512 + size)));
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
  const fpsTypes = [...new Set(frames.map(({ frame }) => frame.fpsType).filter(Boolean))];
  if (fpsTypes.some(type => type !== 'hd_29_97')) throw new Error(`Unsupported frame-time convention: ${fpsTypes.join(', ')}`);
  const selected = frames.flatMap(({ index, frame }) => {
    const body = frame.bodies.find(b => b.id === selectedId && Array.isArray(b.joints19) && b.joints19.length === 76);
    if (!body) return [];
    const joints = Array.from({ length: 133 }, (_, i) => missing(i));
    for (let i = 0; i < 17; i++) {
      const offset = toCoco[i] * 4, values = body.joints19.slice(offset, offset + 4), [x, y, z, confidence] = values;
      if (values.every(Number.isFinite) && confidence >= .3) joints[i] = { id: i, position: [x / 100, -y / 100, z / 100], score: Math.min(1, confidence), source: 'measuredMultiView' };
    }
    return [{ index, timestampMs: index * 1001 / 30, labelTime: Number.isFinite(frame.univTime) ? frame.univTime : null, joints }];
  }).slice(0, 900);
  if (selected.length < 90) throw new Error(`Only ${selected.length} usable frames; at least 90 are required.`);
  const increments = selected.slice(1).map((frame, i) => frame.index - selected[i].index);
  if (increments.some(delta => delta <= 0)) throw new Error('Frame indices are duplicated or not strictly increasing.');
  const labelIntervals = selected.slice(1).flatMap((frame, i) => Number.isFinite(frame.labelTime) && Number.isFinite(selected[i].labelTime) ? [(frame.labelTime - selected[i].labelTime) / (frame.index - selected[i].index)] : []).sort((a, b) => a - b);
  return { subjectId: selectedId, frames: selected, detectedSubjectCount: counts.size, fpsTypes, timeAudit: { labelIntervalPerFrame: labelIntervals.length ? labelIntervals[Math.floor(labelIntervals.length / 2)] : null, expectedIntervalMs: 1001 / 30, maximumFrameIndexGap: Math.max(...increments), nonconsecutiveFramePairs: increments.filter(delta => delta > 1).length, note: 'Frame-index timing is used. univTime is retained only for comparison; its absolute origin is not shared with application performance.now().' } };
}

// Independent implementation of the documented ASF/AMC equations. This small
// fallback reads recorded joint angles, not generated motion or image inference.
const identity3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];
const transpose3 = a => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
const matrixProduct = (a, b) => Array.from({ length: 9 }, (_, k) => {
  const r = Math.floor(k / 3), c = k % 3;
  return a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
});
const matrixVector = (a, v) => [0, 1, 2].map(r => a[r * 3] * v[0] + a[r * 3 + 1] * v[1] + a[r * 3 + 2] * v[2]);

function axisRotation(axis, degrees) {
  const angle = degrees * Math.PI / 180, c = Math.cos(angle), s = Math.sin(angle);
  if (axis === 'x') return [1, 0, 0, 0, c, -s, 0, s, c];
  if (axis === 'y') return [c, 0, s, 0, 1, 0, -s, 0, c];
  if (axis === 'z') return [c, -s, 0, s, c, 0, 0, 0, 1];
  throw new Error(`Unsupported rotation channel ${axis}`);
}

function orderedRotation(channels, degrees) {
  let result = identity3.slice();
  for (let i = 0; i < channels.length; i++) result = matrixProduct(axisRotation(channels[i].slice(-1).toLowerCase(), degrees[i]), result);
  return result;
}

function parseAsf(text) {
  const skeleton = { root: {}, bones: new Map(), children: new Map(), lengthScale: null, angleUnit: null };
  let section = '', bone = null;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.replace(/#.*/, '').trim();
    if (!line) continue;
    if (line[0] === ':') { section = line.slice(1).split(/\s+/)[0]; continue; }
    const [key, ...values] = line.split(/\s+/);
    if (section === 'units') {
      if (key === 'length') skeleton.lengthScale = Number(values[0]);
      if (key === 'angle') skeleton.angleUnit = values[0];
    } else if (section === 'root') {
      skeleton.root[key] = key === 'position' || key === 'orientation' ? values.map(Number) : values;
    } else if (section === 'bonedata') {
      if (key === 'begin') bone = { dof: [] };
      else if (key === 'end') {
        if (!bone?.name || skeleton.bones.has(bone.name)) throw new Error('Invalid or duplicate ASF bone');
        skeleton.bones.set(bone.name, bone); bone = null;
      } else if (bone) {
        if (key === 'name') bone.name = values[0];
        if (key === 'direction') bone.direction = values.map(Number);
        if (key === 'length') bone.length = Number(values[0]);
        if (key === 'axis') { bone.axis = values.slice(0, 3).map(Number); bone.axisOrder = values[3]; }
        if (key === 'dof') bone.dof = values.map(v => v.toLowerCase());
      }
    } else if (section === 'hierarchy' && key !== 'begin' && key !== 'end') skeleton.children.set(key, values);
  }
  if (skeleton.angleUnit !== 'deg' || !Number.isFinite(skeleton.lengthScale) || Math.abs(skeleton.lengthScale - .45) > 1e-9) throw new Error('Unsupported CMU ASF units');
  if (skeleton.root.order?.join(' ') !== 'TX TY TZ RX RY RZ' || skeleton.root.axis?.join('') !== 'XYZ') throw new Error('Unsupported ASF root channel order');
  if (!skeleton.root.position?.every(Number.isFinite) || skeleton.root.position.length !== 3 || skeleton.root.orientation?.length !== 3 || skeleton.root.orientation.some(v => !Number.isFinite(v) || Math.abs(v) > 1e-10)) throw new Error('This CMU fallback requires the documented zero root orientation');
  for (const item of skeleton.bones.values()) {
    if (item.axisOrder !== 'XYZ' || item.axis?.length !== 3 || !item.axis.every(Number.isFinite) || item.direction?.length !== 3 || !item.direction.every(Number.isFinite) || !(item.length > 0)) throw new Error(`Invalid ASF bone ${item.name}`);
    if (Math.abs(Math.hypot(...item.direction) - 1) > .001 || item.dof.some(v => !['rx', 'ry', 'rz'].includes(v))) throw new Error(`Unsupported bone direction/channels ${item.name}`);
    item.axisMatrix = orderedRotation(['rx', 'ry', 'rz'], item.axis);
  }
  const visited = new Set(['root']), order = [];
  function visit(parent) {
    for (const child of skeleton.children.get(parent) || []) {
      if (!skeleton.bones.has(child) || visited.has(child)) throw new Error('Invalid or cyclic ASF hierarchy');
      visited.add(child); order.push({ name: child, parent }); visit(child);
    }
  }
  visit('root');
  if (order.length !== skeleton.bones.size || order.length < 20) throw new Error('Incomplete ASF hierarchy');
  skeleton.order = order;
  // CMU's own FAQ: stored lengths are inches multiplied by the ASF length scale.
  skeleton.metresPerUnit = .0254 / skeleton.lengthScale;
  return skeleton;
}

function parseAmc(text, skeleton) {
  if (!text.includes(':FULLY-SPECIFIED') || !text.includes(':DEGREES')) throw new Error('Unsupported AMC header');
  const frames = [];
  let current = null;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.replace(/#.*/, '').trim();
    if (!line || line[0] === ':') continue;
    if (/^\d+$/.test(line)) { current = { index: Number(line), channels: new Map() }; frames.push(current); continue; }
    const [name, ...values] = line.split(/\s+/);
    if (!current || (name !== 'root' && !skeleton.bones.has(name)) || current.channels.has(name)) throw new Error('Invalid AMC channel line');
    const angles = values.map(Number), expected = name === 'root' ? 6 : skeleton.bones.get(name).dof.length;
    if (angles.length !== expected || !angles.every(Number.isFinite)) throw new Error(`Invalid AMC channel count for ${name}`);
    current.channels.set(name, angles);
  }
  if (frames.length < 240) throw new Error('CMU recording has less than two seconds of reference motion');
  for (let i = 0; i < frames.length; i++) {
    if (i && frames[i].index !== frames[i - 1].index + 1) throw new Error('AMC frame indices are not consecutive');
    for (const name of ['root', ...[...skeleton.bones].filter(([, bone]) => bone.dof.length).map(([name]) => name)]) if (!frames[i].channels.has(name)) throw new Error(`Missing fully-specified AMC channel ${name}`);
  }
  return frames;
}

const cmuJointMapping = [[5, 'lclavicle'], [6, 'rclavicle'], [7, 'lhumerus'], [8, 'rhumerus'],
  [9, 'lwrist'], [10, 'rwrist'], [11, 'lhipjoint'], [12, 'rhipjoint'],
  [13, 'lfemur'], [14, 'rfemur'], [15, 'ltibia'], [16, 'rtibia']];

function cmuReferenceSequence(skeleton, motion) {
  const frames = [], rootPositions = [], scale = skeleton.metresPerUnit;
  for (let frameNumber = 0; frameNumber < Math.min(motion.length, 3600); frameNumber += 4) {
    const raw = motion[frameNumber], root = raw.channels.get('root');
    const endpoints = new Map([['root', root.slice(0, 3).map((v, i) => (v + skeleton.root.position[i]) * scale)]]);
    const rotations = new Map([['root', orderedRotation(['rx', 'ry', 'rz'], root.slice(3))]]);
    for (const { name, parent } of skeleton.order) {
      const bone = skeleton.bones.get(name), C = bone.axisMatrix;
      const M = orderedRotation(bone.dof, raw.channels.get(name) || []);
      // Column-vector form of the documented row-vector ASF transform:
      // R_world = R_parent C M C^-1; the next segment begins at this endpoint.
      const rotation = matrixProduct(rotations.get(parent), matrixProduct(matrixProduct(C, M), transpose3(C)));
      const offset = matrixVector(rotation, bone.direction.map(v => v * bone.length * scale));
      const endpoint = endpoints.get(parent).map((v, i) => v + offset[i]);
      if (!endpoint.every(Number.isFinite)) throw new Error('Non-finite ASF forward kinematics');
      rotations.set(name, rotation); endpoints.set(name, endpoint);
    }
    const joints = Array.from({ length: 133 }, (_, i) => missing(i));
    for (const [id, segment] of cmuJointMapping) {
      if (!endpoints.has(segment)) throw new Error(`Missing reference segment ${segment}`);
      joints[id] = { id, position: endpoints.get(segment), score: 1, source: 'measuredMultiView' };
    }
    rootPositions.push(endpoints.get('root'));
    frames.push({ index: frameNumber / 4, sourceFrameIndex: raw.index,
                  timestampMs: (raw.index - motion[0].index) * 1000 / 120, joints });
  }
  const rootDistance = rootPositions.slice(1).reduce((sum, p, i) => sum + distance(p, rootPositions[i]), 0);
  return { frames, timeAudit: { captureFps: 120, downsampleEvery: 4, evaluationFps: 30, expectedIntervalMs: 1000 / 30,
                               firstSourceFrameIndex: frames[0].sourceFrameIndex, lastSourceFrameIndex: frames.at(-1).sourceFrameIndex,
                               note: 'Capture rate comes from the official subject 7 motion table; AMC files do not embed a rate.' },
           kinematicAudit: { metresPerStoredLength: scale, boneCount: skeleton.bones.size, rootTravelM: rootDistance,
                              rootMeanSpeedMps: rootDistance / ((frames.at(-1).timestampMs - frames[0].timestampMs) / 1000),
                              rootHeightRangeM: [Math.min(...rootPositions.map(p => p[1])), Math.max(...rootPositions.map(p => p[1]))],
                              mapping: cmuJointMapping.map(([cocoId, segment]) => ({ cocoId, reference: `${segment} distal endpoint` })),
                              interpretation: 'Twelve coarse body joints mapped to matching topology; not exact COCO surface annotations, hand fingers or individual toes.' } };
}

function summary(errors) {
  if (!errors.length) return { count: 0, mpjpeM: null, medianM: null, p95M: null, maximumM: null };
  const sorted = errors.slice().sort((a, b) => a - b);
  return { count: errors.length, mpjpeM: errors.reduce((s, v) => s + v, 0) / errors.length, medianM: sorted[Math.floor(sorted.length / 2)], p95M: sorted[Math.floor(.95 * (sorted.length - 1))], maximumM: sorted.at(-1) };
}

function evaluateGap(frames, hiddenIds, gapFrames, label, intervalMs = 1001 / 30, worldCalibrationId = 'panoptic-sample-reference') {
  const tracker = new TemporalPoseTracker(), previousObserved = new Map(), errors = [], heldErrors = [], ages = [], sourceCounts = {};
  let eligible = 0, absent = 0, maskedFrames = 0;
  for (let index = 0; index < frames.length; index++) {
    const truth = frames[index];
    const relativeFrame = truth.index - frames[0].index;
    const hide = relativeFrame >= 30 && (relativeFrame - 30) % 120 < gapFrames;
    if (hide) maskedFrames++;
    const joints = truth.joints.map((j, id) => hide && hiddenIds.includes(id) ? missing(id) : { ...j, position: j.position?.slice() || null });
    for (let id = 0; id < 17; id++) if (joints[id].position) previousObserved.set(id, joints[id].position.slice());
    const estimated = tracker.update({ timestampMs: truth.timestampMs, joints, metric: true, coordinateFrame: 'world', units: 'm', worldCalibrationId });
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
  return { label, hiddenCocoJointIds: hiddenIds, gapFrames, nominalGapMs: gapFrames * intervalMs, maskedFrames, eligibleJointFrames: eligible, missingPredictions: absent, coverage: eligible ? errors.length / eligible : null, predictionAgeMs: ages.length ? { min: Math.min(...ages), max: Math.max(...ages) } : null, tracker: trackerSummary, holdLastVisible: holdLastSummary, relativeImprovementOverHold: holdLastSummary.mpjpeM > 0 ? 1 - trackerSummary.mpjpeM / holdLastSummary.mpjpeM : null, predictionSourceCounts: sourceCounts };
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
    report.dataset = { name: 'Panoptic Studio', url, sha256: createHash('sha256').update(archive).digest('hex'), downloadedBytes: archive.length, archiveFrameCount: frames.length, evaluatedFrameCount: selected.frames.length, firstFrameIndex: selected.frames[0].index, lastFrameIndex: selected.frames.at(-1).index, subjectId: selected.subjectId, detectedSubjectCount: selected.detectedSubjectCount, fpsTypes: selected.fpsTypes, timeAudit: selected.timeAudit };
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

if (!report.realMotionOcclusionMeasured) {
  for (const scheme of ['https', 'http']) {
    const base = `${scheme}://mocap.cs.cmu.edu/subjects/07/`, asfUrl = `${base}07.asf`, amcUrl = `${base}07_01.amc`;
    try {
      const [asf, amc] = await Promise.all([download(asfUrl, 128 * 1024, 15000), download(amcUrl, 4 * 1024 * 1024, 15000)]);
      const skeleton = parseAsf(strictUtf8.decode(asf)), motion = parseAmc(strictUtf8.decode(amc), skeleton);
      const selected = cmuReferenceSequence(skeleton, motion);
      const experiments = [];
      for (const gap of [3, 9, 30]) {
        experiments.push(evaluateGap(selected.frames, [7, 9], gap, 'left-elbow-and-wrist-hidden', 1000 / 30, 'cmu-07-reference'));
        experiments.push(evaluateGap(selected.frames, [15, 16], gap, 'both-ankles-hidden', 1000 / 30, 'cmu-07-reference'));
        experiments.push(evaluateGap(selected.frames, cmuJointMapping.map(([id]) => id), gap, 'whole-body-observations-hidden', 1000 / 30, 'cmu-07-reference'));
      }
      report.sequence = 'CMU-07_01-walk';
      report.referenceType = 'Real optical motion-capture recording represented by the CMU Vicon ASF/AMC skeletal model and original joint angles.';
      report.datasetLicense = 'CMU Graphics Lab FAQ permits motion data copying, modification and redistribution. This recording was created with funding from NSF EIA-0196217. Raw recordings are not redistributed here.';
      report.coordinateConversion = 'CMU ASF and AMC length values × (0.0254 / ASF length scale). Original right-handed Y-up coordinates are retained.';
      report.timeBase = 'Official capture rate 120 Hz; every fourth original frame is evaluated at 30 Hz. Source frame indices are preserved.';
      report.references.push('https://mocap.cs.cmu.edu/faqs.php', 'https://mocap.cs.cmu.edu/info.php',
                             'https://mocap.cs.cmu.edu/search.php?subjectnumber=7',
                             'https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html');
      report.dataset = { name: 'CMU Graphics Lab motion capture', fallbackFrom: 'Panoptic Studio',
                         files: [{ url: asfUrl, bytes: asf.length, sha256: createHash('sha256').update(asf).digest('hex') },
                                 { url: amcUrl, bytes: amc.length, sha256: createHash('sha256').update(amc).digest('hex') }],
                         originalFrameCount: motion.length, evaluatedFrameCount: selected.frames.length,
                         subjectId: '07', timeAudit: selected.timeAudit, kinematicAudit: selected.kinematicAudit };
      report.experiments = experiments;
      report.status = experiments.every(e => e.tracker.count > 0) ? 'measured' : 'insufficient-label-coverage';
      report.realMotionOcclusionMeasured = report.status === 'measured';
      report.attempts.push({ url: base, dataset: 'CMU ASF/AMC', status: 'downloaded' });
      break;
    } catch (error) {
      report.attempts.push({ url: base, dataset: 'CMU ASF/AMC', status: 'failed', reason: String(error.message).slice(0, 400) });
    }
  }
}

await mkdir(dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`, 'utf8');
console.log(JSON.stringify({ output, status: report.status, evaluatedFrames: report.dataset?.evaluatedFrameCount || 0, realVideoAccuracyMeasured: false, realMotionOcclusionMeasured: report.realMotionOcclusionMeasured, experiments: report.experiments.map(e => ({ label: e.label, gapMs: e.nominalGapMs, count: e.tracker.count, mpjpeM: e.tracker.mpjpeM, holdLastMpjpeM: e.holdLastVisible.mpjpeM })) }));
// A public dataset outage is reported as unavailable, never as a passed accuracy result.
