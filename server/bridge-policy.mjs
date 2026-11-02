import { multiplyQuaternion, rotateVector } from '../dist/retarget.mjs';
import { rightHandedToUnity } from './osc.mjs';

export const ROLES = Object.freeze(['waist', 'left-foot', 'right-foot']);
const vec = (v, n) => Array.isArray(v) && v.length === n && v.every(Number.isFinite);
const id = v => typeof v === 'string' && /^[\w.:-]{1,120}$/.test(v);
const normalize = q => q.map(v => v / Math.hypot(...q));
const add = (a, b) => a.map((v, i) => v + b[i]);
function transform(rotation, position) {
  return vec(rotation, 4) && Math.hypot(...rotation) > .5 && Math.hypot(...rotation) < 1.5 && vec(position, 3) && position.every(v => Math.abs(v) <= 100);
}
export function validateRoomCalibration(input) {
  if (!input || input.schemaVersion !== 1 || input.confirmed !== true || !id(input.worldCalibrationId) || !id(input.roomAlignmentId) || !transform(input.rotationXYZW, input.translationM)) throw new Error('Calibration requires confirmed metric world/room IDs and a finite rigid transform.');
  const trackerOffsets = {};
  for (const role of ROLES) {
    const offset = input.trackerOffsets?.[role] || { rotationXYZW: [0, 0, 0, 1], positionM: [0, 0, 0] };
    if (!transform(offset.rotationXYZW, offset.positionM) || offset.positionM.some(v => Math.abs(v) > 1)) throw new Error('Invalid local tracker offset.');
    trackerOffsets[role] = { rotationXYZW: normalize(offset.rotationXYZW), positionM: offset.positionM.slice() };
  }
  return { schemaVersion: 1, confirmed: true, worldCalibrationId: input.worldCalibrationId, roomAlignmentId: input.roomAlignmentId,
    rotationXYZW: normalize(input.rotationXYZW), translationM: input.translationM.slice(), trackerOffsets, allowEstimatedMonocular: input.allowEstimatedMonocular === true };
}
export function validatePoseFrame(frame, { sessionId, lastSequence = -1, clientClockOffsetMs = 0, nowUnixMs = Date.now(), maxAgeMs = 500 } = {}) {
  if (!frame || frame.type !== 'pose' || frame.schemaVersion !== 1 || frame.sessionId !== sessionId || !id(frame.streamId)) throw new Error('Invalid pose envelope.');
  if (!Number.isSafeInteger(frame.sequence) || frame.sequence <= lastSequence || !Number.isFinite(frame.referenceTimeMs)) throw new Error('Stale pose sequence.');
  const age = nowUnixMs - ((frame.sentAtMs ?? NaN) + clientClockOffsetMs) + (frame.captureAgeMs ?? NaN);
  if (!Number.isFinite(age) || age < -250 || age > maxAgeMs || frame.captureAgeMs < 0) throw new Error('Pose is too old or has no capture-age metadata.');
  if (!frame.pose || !Array.isArray(frame.trackers) || frame.trackers.length > ROLES.length) throw new Error('Invalid tracker list.');
  const seen = new Set();
  for (const tracker of frame.trackers) {
    if (!ROLES.includes(tracker?.role) || seen.has(tracker.role) || typeof tracker.valid !== 'boolean') throw new Error('Invalid or duplicate tracker role.');
    seen.add(tracker.role);
    if (tracker.valid && (!transform(tracker.rotation, tracker.position) || !Number.isFinite(tracker.score) || tracker.score < 0 || tracker.score > 1 || !Number.isFinite(tracker.ageMs) || tracker.ageMs < 0 || tracker.ageMs > 180)) throw new Error('Invalid or stale tracker transform.');
  }
  return frame;
}
function worldMatches(meta, calibration) {
  return !!calibration && meta.metric === true && meta.rootValid === true && meta.coordinateFrame === 'world' && meta.units === 'm' && meta.calibratedWorld === true && meta.worldCalibrationId === calibration.worldCalibrationId;
}
export function captureRootProvenance(frame, calibration) {
  const meta = frame.pose, modeAllowed = meta.distanceStatus === 'measuredMultiView' || (calibration?.allowEstimatedMonocular && meta.distanceStatus === 'estimatedMonocular');
  if (!worldMatches(meta, calibration) || !modeAllowed) return null;
  return { streamId: frame.streamId, worldCalibrationId: meta.worldCalibrationId, roomAlignmentId: calibration.roomAlignmentId, distanceStatus: meta.distanceStatus, referenceTimeMs: frame.referenceTimeMs };
}
export function mapFrameToRoom(frame, calibration, { previousRoot = null } = {}) {
  const meta = frame.pose, world = worldMatches(meta, calibration);
  const temporalProvenance = meta.distanceStatus === 'estimatedTemporal' && previousRoot && previousRoot.streamId === frame.streamId && previousRoot.worldCalibrationId === meta.worldCalibrationId && previousRoot.roomAlignmentId === calibration?.roomAlignmentId
    && meta.originDistanceStatus === previousRoot.distanceStatus && frame.referenceTimeMs >= previousRoot.referenceTimeMs && frame.referenceTimeMs - previousRoot.referenceTimeMs <= 180;
  const effectiveMode = meta.distanceStatus === 'estimatedTemporal' ? temporalProvenance ? previousRoot.distanceStatus : null : meta.distanceStatus;
  const modeAllowed = effectiveMode === 'measuredMultiView' || (calibration?.allowEstimatedMonocular && effectiveMode === 'estimatedMonocular');
  return ROLES.map(role => {
    const input = frame.trackers.find(t => t.role === role);
    const valid = !!(world && modeAllowed && input?.valid && input.score >= .15 && input.ageMs <= 180);
    const reason = !calibration ? 'roomAlignmentMissing' : !world ? 'worldCalibrationMissingOrMismatch' : !modeAllowed ? 'distanceModeNotEnabled' : !input?.valid ? input?.reason || 'trackerMissing' : null;
    if (!valid) return { role, valid: false, reason, position: [0, 0, 0], rotation: [0, 0, 0, 1] };
    const offset = calibration.trackerOffsets[role], sourceRotation = normalize(input.rotation);
    const physicalPosition = add(input.position, rotateVector(sourceRotation, offset.positionM));
    const roomPosition = add(rotateVector(calibration.rotationXYZW, physicalPosition), calibration.translationM);
    const roomRotation = multiplyQuaternion(multiplyQuaternion(calibration.rotationXYZW, sourceRotation), offset.rotationXYZW);
    return { role, valid: true, source: input.source, ...rightHandedToUnity(roomPosition, roomRotation), reason: null };
  });
}
