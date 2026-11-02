/** Actions-only numerical retarget and real loopback WebSocket -> UDP experiment. */
import assert from 'node:assert/strict';
import dgram from 'node:dgram';
import { randomBytes } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { WebSocket } from 'ws';
import { createBridgeServer } from './bridge.mjs';
import { validateRoomCalibration, mapFrameToRoom, captureRootProvenance } from './bridge-policy.mjs';
import { retargetPose, footObservationToPose, multiplyQuaternion, rotateVector } from '../dist/retarget.mjs';

if (process.env.GITHUB_ACTIONS !== 'true') throw new Error('Run this experiment in GitHub Actions; local execution is disabled.');
const cases = [], vectorError = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
const unit = v => v.map(x => x / Math.hypot(...v));
const subtract = (a, b) => a.map((x, i) => x - b[i]);
const identity = [0, 0, 0, 1];
const record = (name, values = {}) => cases.push({ name, passed: true, ...values });
function fixture() {
  const bones = {};
  const add = (name, parent, position) => { bones[name] = { parent, position, rotation: identity.slice(), localRotation: identity.slice() }; };
  add('hips', null, [0, 1, 0]); add('spine', 'hips', [0, 1.16, 0]); add('chest', 'spine', [0, 1.4, 0]); add('neck', 'chest', [0, 1.65, 0]); add('head', 'neck', [0, 1.8, 0]);
  const pose = { timestampMs: 1000, sourceId: 'synthetic-fixture', metric: true, units: 'm', coordinateFrame: 'world', worldCalibrationId: 'fixture-world',
    diagnostics: { calibratedWorld: true }, root: { position: [0, 1, 0], valid: true, metric: true, distanceStatus: 'measuredMultiView' },
    joints: Array.from({ length: 133 }, (_, id) => ({ id, position: null, score: 0, source: 'missing', ageMs: 0 })), feet: {} };
  const set = (id, position) => { pose.joints[id] = { id, position, score: .9, source: 'measuredMultiView', ageMs: 0 }; };
  set(1, [-.03, 1.82, 0]); set(2, [.03, 1.82, 0]);
  for (const side of ['left', 'right']) {
    const left = side === 'left', sign = left ? -1 : 1, ids = left ? [5, 7, 9, 11, 13, 15, 17, 18, 19] : [6, 8, 10, 12, 14, 16, 20, 21, 22];
    const shoulder = [sign * .3, 1.5, 0], elbow = [sign * .6, 1.5, 0], wrist = [sign * .9, 1.5, 0], hip = [sign * .11, 1, 0], knee = [sign * .11, .55, 0], ankle = [sign * .11, .1, 0];
    [shoulder, elbow, wrist, hip, knee, ankle].forEach((p, i) => set(ids[i], p));
    add(side + 'UpperArm', 'chest', shoulder); add(side + 'LowerArm', side + 'UpperArm', elbow); add(side + 'Hand', side + 'LowerArm', wrist);
    add(side + 'UpperLeg', 'hips', hip); add(side + 'LowerLeg', side + 'UpperLeg', knee); add(side + 'Foot', side + 'LowerLeg', ankle); add(side + 'Toes', side + 'Foot', [sign * .11, .025, .2]);
    set(ids[6], [sign * .07, 0, .2]); set(ids[7], [sign * .15, 0, .2]); set(ids[8], [sign * .11, 0, -.07]);
    pose.feet[side] = { valid: true, position: ankle, rotation: identity.slice(), up: [0, 1, 0] };
    const base = left ? 91 : 112; set(base, wrist);
    for (const [finger, f] of [['Thumb', 0], ['Index', 1], ['Middle', 2], ['Ring', 3], ['Little', 4]]) {
      const suffixes = finger === 'Thumb' ? ['Metacarpal', 'Proximal', 'Distal'] : ['Proximal', 'Intermediate', 'Distal'];
      for (let k = 0; k < 4; k++) {
        const p = [sign * (1 + k * .035), 1.5, (.08 - f * .035)]; set(base + f * 4 + 1 + k, p);
        if (k < 3) add(side + finger + suffixes[k], k ? side + finger + suffixes[k - 1] : side + 'Hand', p);
      }
    }
  }
  return { rig: { bones }, pose };
}
const { rig, pose } = fixture();
const idle = retargetPose(null, rig, { smoothingMs: 0 });
assert.equal(idle.coverage.drivenBones, 0); assert.deepEqual(idle.normalizedPose.hips.position, [0, 0, 0]);
record('empty-input-does-not-generate-a-pose');
const changed = structuredClone(pose); changed.joints[9].position = [-.65, 1.15, .15]; changed.joints[102].position = [-1.075, 1.46, .045];
const driven = retargetPose(changed, rig, { smoothingMs: 0 });
const forearmDirection = unit(subtract(changed.joints[9].position, changed.joints[7].position));
const actualForearm = rotateVector(driven.bones.leftLowerArm.worldRotation, [-1, 0, 0]);
assert.ok(vectorError(actualForearm, forearmDirection) < 1e-7);
const fingerDirection = unit(subtract(changed.joints[102].position, changed.joints[101].position));
const actualFinger = rotateVector(driven.bones.leftMiddleIntermediate.worldRotation, [-1, 0, 0]);
assert.ok(vectorError(actualFinger, fingerDirection) < 1e-7);
assert.equal(driven.bones.leftToes.articulationStatus, 'unobserved');
record('observed-elbow-and-finger-directions-drive-real-bone-rotations', { forearmDirectionError: vectorError(actualForearm, forearmDirection), fingerDirectionError: vectorError(actualFinger, fingerDirection), drivenBones: driven.coverage.drivenBones });

// Noncommuting rest rotation and an external parent rotation catch wrong order
// in three-vrm's poseDelta * restLocal contract (including a VRM0 parent turn).
const localRest = [0, Math.sin(.35), 0, Math.cos(.35)], external = [0, 0, Math.sin(.2), Math.cos(.2)];
const restWorld = multiplyQuaternion(external, localRest);
const rotatedRig = { bones: {
  leftUpperArm: { parent: null, position: [-.3, 1.5, 0], rotation: restWorld, localRotation: localRest },
  leftLowerArm: { parent: 'leftUpperArm', position: [-.6, 1.5, 0], rotation: restWorld, localRotation: identity },
  leftHand: { parent: 'leftLowerArm', position: [-.9, 1.5, 0], rotation: restWorld, localRotation: identity }
} };
const rotatedPose = structuredClone(pose); rotatedPose.joints[7].position = [-.4, 1.2, .2];
const rotationResult = retargetPose(rotatedPose, rotatedRig, { smoothingMs: 0 });
const reconstructedWorld = multiplyQuaternion(external, multiplyQuaternion(rotationResult.normalizedPose.leftUpperArm.rotation, localRest));
const inverseRest = [-restWorld[0], -restWorld[1], -restWorld[2], restWorld[3]];
const boneLocalAxis = rotateVector(inverseRest, [-1, 0, 0]);
const observedDirection = unit(subtract(rotatedPose.joints[7].position, rotatedPose.joints[5].position));
assert.ok(vectorError(rotateVector(reconstructedWorld, boneLocalAxis), observedDirection) < 1e-7);
record('normalized-quaternion-multiplication-and-parent-frame', { directionError: vectorError(rotateVector(reconstructedWorld, boneLocalAxis), observedDirection) });

const footObservation = { sourceId: 'foot-fixture', targetVersion: 1, timestampMs: 1000, keypoints: Array.from({ length: 133 }, () => ({ x: 0, y: 0, score: 0, observable: false })) };
for (const [id, point] of [[17, [150, 190]], [18, [150, 210]], [19, [100, 200]]]) footObservation.keypoints[id] = { x: point[0], y: point[1], score: .9, observable: true };
const foot0 = footObservationToPose(footObservation), retarget0 = retargetPose(foot0, rig, { smoothingMs: 0 });
const movedObservation = structuredClone(footObservation); movedObservation.timestampMs += 33;
movedObservation.keypoints[17] = { x: 90, y: 150, score: .9, observable: true }; movedObservation.keypoints[18] = { x: 110, y: 150, score: .9, observable: true };
const foot1 = footObservationToPose(movedObservation), retarget1 = retargetPose(foot1, rig, { previous: retarget0, smoothingMs: 0 });
assert.ok(Math.abs(retarget1.normalizedPose.leftFoot.rotation[2]) > .7);
assert.equal(retarget1.coverage.drivenBones, 1); assert.ok(retarget1.trackers.every(t => !t.valid)); assert.ok(foot1.joints.every(j => j.position === null));
const reset = retargetPose(footObservationToPose({ ...movedObservation, targetVersion: 2 }), rig, { previous: retarget1, smoothingMs: 0 });
assert.ok(vectorError(reset.normalizedPose.leftFoot.rotation, identity) < 1e-7);
record('foot-only-relative-angle-without-invented-root-and-seed-reset', { drivenBones: retarget1.coverage.drivenBones, invented3DJoints: 0, validSteamVrTrackers: 0 });

const relative = retargetPose({ ...pose, metric: false, coordinateFrame: 'root-relative', units: 'm-relative-estimate', root: { ...pose.root, valid: false, metric: false, position: [5, 2, 3] } }, rig, { smoothingMs: 0 });
assert.deepEqual(relative.normalizedPose.hips.position, [0, 0, 0]); assert.ok(relative.trackers.every(t => !t.valid));
record('uncalibrated-relative-pose-never-enables-metric-trackers');

const packets = [], receiver = dgram.createSocket('udp4');
function decode(packet) {
  let offset = 0;
  const text = () => { const end = packet.indexOf(0, offset); assert.ok(end >= 0); const value = packet.toString('utf8', offset, end); offset = Math.ceil((end + 1) / 4) * 4; return value; };
  const address = text(), types = text(), values = [];
  for (const type of types.slice(1)) { values.push(type === 'i' ? packet.readInt32BE(offset) : packet.readFloatBE(offset)); offset += 4; }
  assert.equal(offset, packet.length); return { address, types, values };
}
receiver.on('message', packet => packets.push(decode(packet)));
await new Promise(done => receiver.bind(0, '127.0.0.1', done));
const sleep = ms => new Promise(done => setTimeout(done, ms));
async function until(predicate, label, timeout = 1800) {
  const started = Date.now();
  while (Date.now() - started < timeout) { const result = predicate(); if (result) return result; await sleep(8); }
  throw new Error('Timed out: ' + label);
}
function waitMessage(ws, type) {
  return new Promise((done, fail) => {
    const timeout = setTimeout(() => { ws.off('message', receive); fail(new Error('No WebSocket ' + type)); }, 1800);
    function receive(data) { const value = JSON.parse(data.toString()); if (value.type === type) { clearTimeout(timeout); ws.off('message', receive); done(value); } }
    ws.on('message', receive);
  });
}
let bridge, ws;
try {
  const token = randomBytes(24).toString('base64url');
  bridge = await createBridgeServer({ port: 0, udpPort: receiver.address().port, staleMs: 160, token });
  const url = `ws://127.0.0.1:${bridge.port}/v1/pose`, origin = `http://127.0.0.1:${bridge.port}`;
  const deniedOrigin = await new Promise(done => {
    const bad = new WebSocket(url, { origin: 'https://untrusted.invalid' }); bad.on('error', () => {});
    bad.once('unexpected-response', (_req, response) => { done(response.statusCode); response.resume(); bad.terminate(); });
  });
  assert.equal(deniedOrigin, 403);
  const wrong = new WebSocket(url, { origin }); wrong.on('error', () => {});
  await new Promise(done => wrong.once('open', done));
  const wrongClosed = new Promise(done => wrong.once('close', code => done(code)));
  wrong.send(JSON.stringify({ type: 'hello', schemaVersion: 1, token: 'incorrect-token-xxxxxxxxxxxxxx', clientTimeMs: Date.now() }));
  assert.equal(await wrongClosed, 1008); assert.equal(bridge.getState().authenticated, false);
  record('real-websocket-origin-and-token-rejection', { originStatus: deniedOrigin, wrongTokenCloseCode: 1008 });
  ws = new WebSocket(url, { origin }); ws.on('error', () => {}); await new Promise(done => ws.once('open', done));
  const readyPromise = waitMessage(ws, 'ready'); ws.send(JSON.stringify({ type: 'hello', schemaVersion: 1, token, clientTimeMs: Date.now() }));
  const ready = await readyPromise;
  const calibration = { schemaVersion: 1, confirmed: true, worldCalibrationId: 'fixture-world', roomAlignmentId: 'fixture-room', rotationXYZW: [0, Math.SQRT1_2, 0, Math.SQRT1_2], translationM: [1, 0, 2] };
  const calibrated = waitMessage(ws, 'calibrated'); ws.send(JSON.stringify({ type: 'calibration', calibration })); await calibrated;
  let sequence = 0;
  const frame = () => ({ type: 'pose', schemaVersion: 1, sessionId: ready.sessionId, streamId: 'fixture-stream', sequence: ++sequence, referenceTimeMs: 1000 + sequence * 33,
    sentAtMs: Date.now(), captureAgeMs: 0, pose: { metric: true, rootValid: true, coordinateFrame: 'world', units: 'm', calibratedWorld: true, worldCalibrationId: 'fixture-world', distanceStatus: 'measuredMultiView' },
    trackers: ['waist', 'left-foot', 'right-foot'].map(role => ({ role, valid: true, position: [1, 1, 3], rotation: identity, source: 'measuredMultiView', score: .9, ageMs: 0 })) });
  let start = packets.length; ws.send(JSON.stringify(frame()));
  const packet = await until(() => packets.slice(start).find(p => p.values[0] === 0 && p.values[1] === 1), 'actual VMT UDP packet');
  assert.equal(packet.address, '/VMT/Room/Unity'); assert.equal(packet.types, ',iiffffffff');
  assert.ok(vectorError(packet.values.slice(3, 6), [4, 1, -1]) < 1e-6);
  assert.ok(vectorError(packet.values.slice(6, 10), [0, -Math.SQRT1_2, 0, Math.SQRT1_2]) < 1e-6);
  record('real-websocket-to-udp-with-calibrated-coordinate-conversion', { destination: '127.0.0.1 ephemeral CI receiver', oscAddress: packet.address, oscTypes: packet.types,
    positionErrorM: vectorError(packet.values.slice(3, 6), [4, 1, -1]), quaternionError: vectorError(packet.values.slice(6, 10), [0, -Math.SQRT1_2, 0, Math.SQRT1_2]) });
  const enabledIndex = packets.indexOf(packet);
  await until(() => packets.slice(enabledIndex + 1).filter(p => p.values[1] === 0).length >= 3 && bridge.getState().enabledTrackers === 0, 'stale tracker disable');
  record('stale-input-emits-real-disable-packets', { staleMs: 160, disabledTrackers: 3 });
  const wrongWorld = frame(); wrongWorld.pose.worldCalibrationId = 'another-world';
  assert.ok(mapFrameToRoom(wrongWorld, validateRoomCalibration(calibration)).every(t => !t.valid));
  const relativeFrame = frame(); relativeFrame.pose.metric = false; relativeFrame.pose.coordinateFrame = 'root-relative';
  assert.ok(mapFrameToRoom(relativeFrame, validateRoomCalibration(calibration)).every(t => !t.valid));
  record('world-id-mismatch-and-relative-input-stay-disabled');
  const currentRoot = frame(), room = validateRoomCalibration(calibration), provenance = captureRootProvenance(currentRoot, room);
  const predicted = structuredClone(currentRoot); predicted.referenceTimeMs += 100; predicted.pose.distanceStatus = 'estimatedTemporal'; predicted.pose.originDistanceStatus = 'measuredMultiView';
  assert.ok(mapFrameToRoom(predicted, room).every(t => !t.valid));
  assert.ok(mapFrameToRoom(predicted, room, { previousRoot: provenance }).every(t => t.valid));
  assert.ok(mapFrameToRoom({ ...predicted, referenceTimeMs: currentRoot.referenceTimeMs + 181 }, room, { previousRoot: provenance }).every(t => !t.valid));
  assert.ok(mapFrameToRoom(predicted, room, { previousRoot: { ...provenance, streamId: 'different-stream' } }).every(t => !t.valid));
  assert.ok(mapFrameToRoom(predicted, room, { previousRoot: { ...provenance, worldCalibrationId: 'different-world' } }).every(t => !t.valid));
  const monocular = structuredClone(currentRoot); monocular.pose.distanceStatus = 'estimatedMonocular';
  assert.equal(captureRootProvenance(monocular, room), null);
  predicted.pose.originDistanceStatus = 'estimatedMonocular';
  assert.ok(mapFrameToRoom(predicted, room, { previousRoot: { ...provenance, distanceStatus: 'estimatedMonocular' } }).every(t => !t.valid));
  record('temporal-root-preserves-permitted-source-world-and-original-observation-time', { maximumPredictionAgeMs: 180, disabledMonocularBypassRejected: true });
  start = packets.length; ws.send(JSON.stringify(frame()));
  await until(() => packets.slice(start).some(p => p.values[1] === 1), 'tracker recovery');
  await sleep(10);
  const close = new Promise(done => ws.once('close', done)); ws.close(); await close;
  await until(() => bridge.getState().enabledTrackers === 0 && packets.slice(start).filter(p => p.values[1] === 0).length >= 3, 'disconnect invalidation');
  record('disconnect-emits-real-disable-packets', { disabledTrackers: 3 });
} finally {
  if (ws && ws.readyState < 2) ws.terminate();
  if (bridge) await bridge.close();
  await new Promise(done => receiver.close(done));
}
const report = { schemaVersion: 1, status: 'measured', passed: true, runId: process.env.GITHUB_RUN_ID || null, gitCommit: process.env.GITHUB_SHA || null,
  scope: 'Synthetic numerical retarget fixtures and actual loopback WebSocket/OSC UDP transport in GitHub Actions.',
  privateMediaUsed: false, realCameraAccuracyMeasured: false, actualVrmAssetMeasured: false, steamVrHmdVerified: false, vrChatFbtVerified: false,
  cases, count: cases.length, udpPacketsReceived: packets.length,
  limitations: ['No SteamVR, VMT driver, HMD, or VRChat ran in this experiment.', 'No camera image, hidden finger, or individual toe accuracy is established.', 'UDP receipt verifies bytes and lifecycle, not driver acceptance or end-to-end latency.'] };
const outIndex = process.argv.indexOf('--out'), output = resolve(outIndex >= 0 ? process.argv[outIndex + 1] : 'evaluation/results/phase3-bridge-report.json');
await mkdir(dirname(output), { recursive: true }); await writeFile(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ passed: true, cases: cases.length, udpPacketsReceived: packets.length, steamVrHmdVerified: false, output }));
