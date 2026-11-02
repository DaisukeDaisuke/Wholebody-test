/** Pure pose-to-humanoid math. No camera, network, Three.js, or synthetic motion. */
const EPS = 1e-8;
const IDENTITY = [0, 0, 0, 1];
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const vec = (v, n = 3) => Array.isArray(v) && v.length === n && v.every(Number.isFinite);
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const mul = (a, s) => a.map(v => v * s);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = a => Math.hypot(...a);
const unit = a => a && norm(a) > EPS ? mul(a, 1 / norm(a)) : null;
const mid = (a, b) => mul(add(a, b), .5);
const inverse = q => [-q[0], -q[1], -q[2], q[3]];
const normalizeQ = q => vec(q, 4) && norm(q) > EPS ? mul(q, 1 / norm(q)) : IDENTITY.slice();

export function multiplyQuaternion(a, b) {
  return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
}
export function rotateVector(q, v) {
  const t = mul(cross(q.slice(0, 3), v), 2);
  return add(v, add(mul(t, q[3]), cross(q.slice(0, 3), t)));
}
function axisAngle(axis, angle) { const u = unit(axis); return u ? [...mul(u, Math.sin(angle / 2)), Math.cos(angle / 2)] : IDENTITY.slice(); }
function between(a, b) {
  const x = unit(a), y = unit(b);
  if (!x || !y) return IDENTITY.slice();
  const d = clamp(dot(x, y), -1, 1);
  if (d < -1 + 1e-7) return axisAngle(cross(x, Math.abs(x[0]) < .8 ? [1, 0, 0] : [0, 1, 0]), Math.PI);
  return normalizeQ([...cross(x, y), 1 + d]);
}
function slerp(a, b, t) {
  let q = b, d = dot(a, b);
  if (d < 0) { q = mul(b, -1); d = -d; }
  if (d > .9995) return normalizeQ(a.map((v, i) => v + (q[i] - v) * t));
  const angle = Math.acos(clamp(d, -1, 1)), sine = Math.sin(angle);
  return a.map((v, i) => (Math.sin((1 - t) * angle) * v + Math.sin(t * angle) * q[i]) / sine);
}
export function quaternionAngle(a, b = IDENTITY) { const cosine = clamp(Math.abs(dot(normalizeQ(a), normalizeQ(b))), 0, 1); return cosine > 1 - 1e-14 ? 0 : 2 * Math.acos(cosine); }
function limitRotation(previous, desired, maxAngle) {
  const angle = quaternionAngle(previous, desired);
  return angle > maxAngle && angle > EPS ? slerp(previous, desired, maxAngle / angle) : desired;
}
const smoothstep = t => { const x = clamp(t, 0, 1); return x * x * (3 - 2 * x); };
const wrapAngle = angle => Math.atan2(Math.sin(angle), Math.cos(angle));
const freshSource = source => ['measuredMultiView', 'estimatedMultiView', 'inferredMonocular', 'inferredFinger', 'observed3D'].includes(source);
const observedDepth = js => js.every(j => ['measuredMultiView', 'estimatedMultiView', 'observed3D'].includes(j?.source) && j?.depthAmbiguous !== true);
const relativeDepth = js => observedDepth(js) || js.every(j => j?.modelRelativeZ === true && j?.depthAmbiguous !== true);
const planePosition = j => vec(j?.imagePlanePosition) ? j.imagePlanePosition : [j.position[0], j.position[1], 0];
const segmentDirection = js => relativeDepth(js) ? sub(js[1].position, js[0].position) : sub(planePosition(js[1]), planePosition(js[0]));
function basisQuaternion(x, y, z) {
  const m = [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]], trace = m[0] + m[4] + m[8];
  let q, s;
  if (trace > 0) { s = 2 * Math.sqrt(trace + 1); q = [(m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s, s / 4]; }
  else if (m[0] > m[4] && m[0] > m[8]) { s = 2 * Math.sqrt(1 + m[0] - m[4] - m[8]); q = [s / 4, (m[1] + m[3]) / s, (m[2] + m[6]) / s, (m[7] - m[5]) / s]; }
  else if (m[4] > m[8]) { s = 2 * Math.sqrt(1 + m[4] - m[0] - m[8]); q = [(m[1] + m[3]) / s, s / 4, (m[5] + m[7]) / s, (m[2] - m[6]) / s]; }
  else { s = 2 * Math.sqrt(1 + m[8] - m[0] - m[4]); q = [(m[2] + m[6]) / s, (m[5] + m[7]) / s, s / 4, (m[3] - m[1]) / s]; }
  return normalizeQ(q);
}
function frame(primary, secondary) {
  const x = unit(primary), y = x && unit(sub(secondary, mul(x, dot(x, secondary))));
  const z = x && y && unit(cross(x, y));
  return z ? { x, y, z, rotation: basisQuaternion(x, y, z) } : null;
}
function torsoFrame(left, right, root, top) { return frame(sub(right, left), sub(top, root)); }
const SOURCE_ORDER = ['measuredMultiView', 'estimatedMultiView', 'observed3D', 'inferredMonocular', 'inferredFinger', 'predicted', 'inferredKinematic', 'held', 'prior', 'inferredPrior', 'inferredAnatomy', 'missing'];
function quality(joints) {
  if (!joints.length) return { source: 'missing', score: 0, ageMs: Infinity };
  const rank = s => SOURCE_ORDER.includes(s) ? SOURCE_ORDER.indexOf(s) : SOURCE_ORDER.length;
  const source = joints.map(j => j?.source || 'missing').sort((a, b) => rank(b) - rank(a))[0];
  const observationTimes = joints.map(j => j?.observedAtMs);
  return { source, score: Math.min(...joints.map(j => Number.isFinite(j?.score) ? clamp(j.score, 0, 1) : 0)), ageMs: Math.max(...joints.map(j => Number.isFinite(j?.ageMs) ? j.ageMs : 0)),
    observedAtMs: observationTimes.every(Number.isFinite) ? Math.min(...observationTimes) : null,
    continuityEligible: joints.every(j => j?.continuityEligible === true) };
}
function jointReader(pose, options) {
  return id => {
    const j = pose?.joints?.[id];
    return j && vec(j.position) && freshSource(j.source) && j.observable2D !== false && j.valid !== false && (j.score ?? 0) >= options.minScore && (j.ageMs ?? 0) <= options.maxJointAgeMs ? j : null;
  };
}
function rigData(rig) {
  const bones = {}, order = [], visiting = new Set(), visited = new Set();
  for (const [name, item] of Object.entries(rig?.bones || {})) {
    if (!item || !vec(item.position)) continue;
    bones[name] = { ...item, rotation: normalizeQ(item.rotation || IDENTITY), localRotation: item.localRotation ? normalizeQ(item.localRotation) : null };
  }
  function visit(name) {
    if (visited.has(name)) return;
    if (visiting.has(name)) throw new Error('Retarget rig contains a parent cycle.');
    visiting.add(name);
    const bone = bones[name];
    if (bone.parent && bones[bone.parent]) visit(bone.parent); else bone.parent = null;
    const parentQ = bone.parent ? bones[bone.parent].rotation : IDENTITY;
    bone.restRelative = multiplyQuaternion(inverse(parentQ), bone.rotation);
    bone.localRotation ||= bone.restRelative.slice();
    bone.intermediateRotation = multiplyQuaternion(bone.restRelative, inverse(bone.localRotation));
    visiting.delete(name); visited.add(name); order.push(name);
  }
  Object.keys(bones).forEach(visit);
  return { bones, order };
}

function relaxedNeutralRotations(rest, armAngleDeg = 10) {
  const neutral = Object.fromEntries(Object.keys(rest).map(name => [name, IDENTITY.slice()]));
  const hips = [rest.leftUpperLeg?.position, rest.rightUpperLeg?.position], shoulders = [rest.leftUpperArm?.position, rest.rightUpperArm?.position];
  const torso = [...hips, ...shoulders].every(p => vec(p)) ? torsoFrame(hips[0], hips[1], mid(...hips), mid(...shoulders)) : null;
  const up = torso?.y || [0, 1, 0], lateral = torso?.x || [-1, 0, 0], angle = clamp(armAngleDeg, 0, 30) * Math.PI / 180;
  for (const side of ['left', 'right']) {
    const name = side + 'UpperArm', bone = rest[name], child = rest[side + 'LowerArm'];
    if (!bone || !child) continue;
    const direction = sub(child.position, bone.position), outward = mul(lateral, side === 'left' ? -1 : 1);
    const down = add(mul(up, -Math.cos(angle)), mul(outward, Math.sin(angle)));
    const desired = multiplyQuaternion(between(direction, down), bone.rotation);
    const parent = bone.parent ? rest[bone.parent].rotation : IDENTITY;
    const actualLocal = multiplyQuaternion(inverse(bone.intermediateRotation), multiplyQuaternion(inverse(parent), desired));
    neutral[name] = normalizeQ(multiplyQuaternion(actualLocal, inverse(bone.localRotation)));
  }
  return neutral;
}

/** Explicit display fallback, not an observation: rest pose except upper arms
 * directed down, ten degrees outward, expressed in each rig bone's local frame. */
export function neutralPoseForRig(rig, { relaxedArmAngleDeg = 10 } = {}) {
  const { bones } = rigData(rig), rotations = relaxedNeutralRotations(bones, relaxedArmAngleDeg);
  return Object.fromEntries(Object.entries(rotations).map(([name, rotation]) => [name, { rotation }]));
}

/**
 * rig bones: model-space rest position/world rotation, actual node localRotation,
 * and nearest humanoid parent. Output matches three-vrm setNormalizedPose:
 * actualLocalQuaternion = outputDelta * restLocalQuaternion, not the reverse.
 */
export function retargetPose(pose, rig, options = {}) {
  const opts = { minScore: .1, maxJointAgeMs: 180, smoothingMs: 45, holdMs: 140, neutralMs: 650, reacquireFrames: 2, resumeGapMs: 180, maxRootSpeed: 2, relaxedArmAngleDeg: 10, ...options };
  const { bones: rest, order } = rigData(rig), read = jointReader(pose, opts);
  const neutralRotations = relaxedNeutralRotations(rest, opts.relaxedArmAngleDeg);
  const inputKey = `${pose?.inputMode || ''}:${pose?.sourceId || ''}:${pose?.targetVersion ?? ''}:${pose?.coordinateFrame || ''}:${pose?.worldCalibrationId || ''}:${pose?.units || ''}`;
  const previous = opts.previous?.inputKey === inputKey && (pose?.timestampMs ?? 0) >= (opts.previous.timestampMs ?? 0) ? opts.previous : null;
  const now = pose?.timestampMs ?? 0, measuredAt = Math.min(now, pose?.observationTimestampMs ?? now);
  const dt = previous ? Math.max(0, now - previous.timestampMs) : 1000 / 60;
  const observationFresh = now - measuredAt <= opts.maxJointAgeMs;
  const targets = {}, solved = {}, normalizedPose = {}, outputBones = {}, warnings = [];
  const reference = name => rest[name]?.position || null;
  const get = ids => ids.map(read), all = js => js.every(Boolean);
  const setTarget = (name, rotation, js, extra = {}) => { if (rest[name] && rotation) targets[name] = { rotation: normalizeQ(rotation), ...quality(js), valid: true, ...extra }; };
  const restLeftHip = reference('leftUpperLeg'), restRightHip = reference('rightUpperLeg');
  const restLeftShoulder = reference('leftUpperArm'), restRightShoulder = reference('rightUpperArm');
  const restTorso = [restLeftHip, restRightHip, restLeftShoulder, restRightShoulder].every(Boolean)
    ? torsoFrame(restLeftHip, restRightHip, mid(restLeftHip, restRightHip), mid(restLeftShoulder, restRightShoulder)) : null;
  const hips = get([11, 12]), shoulders = get([5, 6]);
  let hipFrame = null, chestFrame = null, hipDelta = null, chestDelta = null;
  const torsoJoints = [...hips, ...shoulders], fullTorsoDepth = all(torsoJoints) && relativeDepth(torsoJoints);
  if (all(torsoJoints) && restTorso && pose?.stability?.torsoValid !== false) {
    const hipCenter = mid(hips[0].position, hips[1].position), shoulderCenter = mid(shoulders[0].position, shoulders[1].position);
    const torsoLength = norm(sub(shoulderCenter, hipCenter)), hipWidth = norm(sub(hips[0].position, hips[1].position)), shoulderWidth = norm(sub(shoulders[0].position, shoulders[1].position));
    const usableTorso = torsoLength > EPS && hipWidth > torsoLength * .05 && hipWidth < torsoLength * 1.7 && shoulderWidth > torsoLength * .08 && shoulderWidth < torsoLength * 2.4;
    if (usableTorso && fullTorsoDepth) {
      hipFrame = torsoFrame(hips[0].position, hips[1].position, hipCenter, shoulderCenter);
      chestFrame = torsoFrame(shoulders[0].position, shoulders[1].position, hipCenter, shoulderCenter);
      hipDelta = hipFrame && multiplyQuaternion(hipFrame.rotation, inverse(restTorso.rotation));
      chestDelta = chestFrame && multiplyQuaternion(chestFrame.rotation, inverse(restTorso.rotation));
    } else if (usableTorso) {
      // The shoulder/hip line cannot resolve monocular yaw or front/back pitch.
      // Its visible upright-axis roll remains useful, including genuine leaning.
      const up = sub(mid(...shoulders.map(planePosition)), mid(...hips.map(planePosition)));
      if (up[1] > norm(up) * .15) {
        const angle = wrapAngle(Math.atan2(-up[0], up[1]) - Math.atan2(-restTorso.y[0], restTorso.y[1]));
        hipDelta = chestDelta = axisAngle([0, 0, 1], clamp(angle, -Math.PI / 4, Math.PI / 4));
      }
      warnings.push('monocularTorsoYawPitchUnobserved');
    }
    if (hipDelta) setTarget('hips', multiplyQuaternion(hipDelta, rest.hips?.rotation || IDENTITY), [...hips, ...shoulders]);
    const spineNames = ['spine', 'chest', 'upperChest'].filter(name => rest[name]);
    if (hipDelta && chestDelta) spineNames.forEach((name, i) => setTarget(name, multiplyQuaternion(slerp(hipDelta, chestDelta, (i + 1) / spineNames.length), rest[name].rotation), [...hips, ...shoulders], { source: 'inferredTorsoDistribution' }));
  }
  const eyes = get([1, 2]), ears = get([3, 4]);
  const facePair = all(eyes) ? eyes : all(ears) ? ears : null;
  if (facePair && all(shoulders) && restTorso) {
    const headFrame = relativeDepth([...facePair, ...shoulders]) ? frame(sub(facePair[1].position, facePair[0].position), sub(mid(facePair[0].position, facePair[1].position), mid(shoulders[0].position, shoulders[1].position))) : null;
    let headDelta = headFrame ? multiplyQuaternion(headFrame.rotation, inverse(restTorso.rotation)) : null;
    if (!headDelta && !relativeDepth([...facePair, ...shoulders])) {
      const line = sub(planePosition(facePair[1]), planePosition(facePair[0]));
      if (norm(line) > EPS) {
        let angle = wrapAngle(Math.atan2(line[1], line[0]) - Math.atan2(restTorso.x[1], restTorso.x[0]));
        if (angle > Math.PI / 2) angle -= Math.PI;
        if (angle < -Math.PI / 2) angle += Math.PI;
        headDelta = axisAngle([0, 0, 1], clamp(angle, -Math.PI * .28, Math.PI * .28));
      }
    }
    if (headDelta) {
      if (rest.neck) setTarget('neck', multiplyQuaternion(chestDelta ? slerp(chestDelta, headDelta, .35) : headDelta, rest.neck.rotation), [...facePair, ...shoulders], { source: 'inferredNeck' });
      if (rest.head) setTarget('head', multiplyQuaternion(headDelta, rest.head.rotation), [...facePair, ...shoulders]);
    }
  }
  const segments = {}, planarFeet = {};
  for (const side of ['left', 'right']) {
    const left = side === 'left', ids = left ? [5, 7, 9, 11, 13, 15] : [6, 8, 10, 12, 14, 16];
    for (const [suffix, a, b, child] of [['UpperArm', ids[0], ids[1], 'LowerArm'], ['LowerArm', ids[1], ids[2], 'Hand'], ['UpperLeg', ids[3], ids[4], 'LowerLeg'], ['LowerLeg', ids[4], ids[5], 'Foot']]) {
      const name = side + suffix, js = get([a, b]), restChild = reference(side + child);
      if (rest[name] && restChild && all(js)) {
        const direction = segmentDirection(js), height = pose?.diagnostics?.bodyHeight || (pose?.units === 'body-height' ? 1 : 1.7);
        const ratio = suffix === 'UpperArm' ? .186 : suffix === 'LowerArm' ? .146 : .246;
        const scaleObserved = pose?.metric === true || pose?.diagnostics?.scaleCueCount >= 2;
        if (norm(direction) > height * .002 && (!scaleObserved || norm(direction) < height * ratio * 2)) segments[name] = { js, direction, restDirection: sub(restChild, reference(name)), secondary: null };
      }
    }
    const base = left ? 91 : 112, palm = get([base, base + 5, base + 9, base + 17]);
    const palmRestNames = [side + 'Hand', side + 'IndexProximal', side + 'MiddleProximal', side + 'LittleProximal'];
    const palmRest = palmRestNames.map(reference);
    let palmFrame = null, restPalmFrame = null;
    if (all(palm) && palmRest.every(Boolean)) {
      palmFrame = observedDepth(palm) ? frame(sub(palm[2].position, palm[0].position), sub(palm[1].position, palm[3].position)) : null;
      restPalmFrame = frame(sub(palmRest[2], palmRest[0]), sub(palmRest[1], palmRest[3]));
      if (palmFrame && restPalmFrame) setTarget(side + 'Hand', multiplyQuaternion(multiplyQuaternion(palmFrame.rotation, inverse(restPalmFrame.rotation)), rest[side + 'Hand'].rotation), palm);
      else if (rest[side + 'Hand']) segments[side + 'Hand'] = { js: palm, direction: segmentDirection([palm[0], palm[2]]), restDirection: sub(palmRest[2], palmRest[0]), secondary: null };
    }
    // Each observed finger segment controls its matching bone; distal direction
    // uses the measured fingertip, with the avatar's rest distal axis retained.
    for (const [finger, f] of [['Thumb', 0], ['Index', 1], ['Middle', 2], ['Ring', 3], ['Little', 4]]) {
      const suffixes = finger === 'Thumb' ? ['Metacarpal', 'Proximal', 'Distal'] : ['Proximal', 'Intermediate', 'Distal'];
      suffixes.forEach((suffix, k) => {
        const name = side + finger + suffix, js = get([base + f * 4 + 1 + k, base + f * 4 + 2 + k]);
        if (!rest[name] || !all(js)) return;
        const next = k < 2 ? reference(side + finger + suffixes[k + 1]) : null;
        const previousBone = k > 0 ? reference(side + finger + suffixes[k - 1]) : reference(side + 'Hand');
        const restDirection = next ? sub(next, reference(name)) : previousBone ? sub(reference(name), previousBone) : null;
        if (restDirection && norm(restDirection) > EPS) segments[name] = { js, direction: segmentDirection(js), restDirection,
          secondary: palmFrame && restPalmFrame ? { current: palmFrame.z, rest: restPalmFrame.z } : null, finger: true };
      });
    }
    const foot = pose?.feet?.[side], footJs = get(left ? [15, 17, 18, 19] : [16, 20, 21, 22]);
    const footName = side + 'Foot', toeName = side + 'Toes';
    if (foot?.valid && vec(foot.rotationXYZW || foot.rotation, 4) && all(footJs) && observedDepth(footJs) && rest[footName]) {
      const forward = unit(sub(mid(footJs[1].position, footJs[2].position), footJs[3].position));
      const up = unit(foot.up || [0, 1, 0]);
      const restUp = restTorso?.y || [0, 1, 0];
      const restDirection = reference(toeName) ? sub(reference(toeName), reference(footName)) : restTorso?.z;
      const restForward = restDirection && unit(sub(restDirection, mul(restUp, dot(restDirection, restUp))));
      const observed = forward && frame(forward, up), restFrame = restForward && frame(restForward, restUp);
      if (observed && restFrame) setTarget(footName, multiplyQuaternion(multiplyQuaternion(observed.rotation, inverse(restFrame.rotation)), rest[footName].rotation), footJs, { twistStatus: 'estimatedFromFootLandmarks' });
      // COCO supplies no toe articulation. Toes inherit the foot in their rest
      // configuration; no invented curl is presented as a measured rotation.
      if (rest[toeName]) outputBones[toeName] = { valid: false, source: 'inferredAnatomy', score: 0, articulationStatus: 'unobserved' };
    } else if (all([footJs[0], footJs[1], footJs[3]]) && relativeDepth([footJs[0], footJs[1], footJs[3]]) && rest[footName]) {
      // The learned body observes an ankle, heel and foot-index point. These
      // define a foot plane without inventing a separately observed little toe.
      const [ankle, toe, , heel] = footJs, forward = unit(sub(toe.position, heel.position));
      const upHint = sub(ankle.position, heel.position), restUp = restTorso?.y || [0, 1, 0];
      const restForward = reference(toeName) ? sub(reference(toeName), reference(footName)) : restTorso?.z;
      const currentFrame = forward && frame(forward, upHint), restFrame = restForward && frame(restForward, restUp);
      if (currentFrame && restFrame) setTarget(footName, multiplyQuaternion(multiplyQuaternion(currentFrame.rotation, inverse(restFrame.rotation)), rest[footName].rotation), [ankle, toe, heel], { twistStatus: 'learnedFootPlaneEstimate', toeIndexIsApproximation: true });
      if (rest[toeName]) outputBones[toeName] = { valid: false, source: 'inferredAnatomy', score: 0, articulationStatus: 'unobserved' };
    } else if (all(footJs) && rest[footName]) {
      const direction = sub(mid(planePosition(footJs[1]), planePosition(footJs[2])), planePosition(footJs[3]));
      if (norm(direction) > EPS) planarFeet[side] = { angle: Math.atan2(direction[1], direction[0]), js: footJs };
    }
  }
  const footReferenceAngles = { ...(previous?.footReferenceAngles || {}) }, hints = opts.footHints || pose?.footHints || {};
  for (const side of ['left', 'right']) {
    if (targets[side + 'Foot']) continue;
    const hint = hints[side], name = side + 'Foot', planar = planarFeet[side];
    if (!rest[name]) continue;
    let angle = planar?.angle;
    if (!Number.isFinite(angle)) {
      if (!hint || ![hint.heel, hint.bigToe, hint.littleToe].every(p => vec(p, 2)) || (hint.score ?? 0) < opts.minScore || (hint.ageMs || 0) > opts.maxJointAgeMs) continue;
      const toe = mid(hint.bigToe, hint.littleToe), direction = [toe[0] - hint.heel[0], -(toe[1] - hint.heel[1])];
      if (norm(direction) < 2) continue;
      angle = Math.atan2(direction[1], direction[0]);
    }
    if (!Number.isFinite(footReferenceAngles[side])) footReferenceAngles[side] = angle;
    const difference = Math.atan2(Math.sin(angle - footReferenceAngles[side]), Math.cos(angle - footReferenceAngles[side]));
    setTarget(name, multiplyQuaternion(axisAngle([0, 0, 1], difference), rest[name].rotation), planar?.js || [], { source: 'estimatedPlanarFoot', inputSource: hint?.source || 'observed2D', score: planar ? quality(planar.js).score : clamp(hint.score, 0, 1), ageMs: hint?.ageMs || 0, depthAmbiguous: true, anatomicalSideConfirmed: hint?.anatomicalSideConfirmed === true });
  }
  for (const name of order) {
    const bone = rest[name], parent = bone.parent ? solved[bone.parent] : IDENTITY;
    const inherited = multiplyQuaternion(parent, bone.restRelative), segment = segments[name];
    let target = targets[name];
    if (!target && segment && unit(segment.direction)) {
      const localDirection = rotateVector(inverse(bone.rotation), segment.restDirection);
      const currentDirection = unit(segment.direction);
      let rotation = multiplyQuaternion(between(rotateVector(inherited, localDirection), currentDirection), inherited);
      if (segment.secondary) {
        const referenceLocal = rotateVector(inverse(bone.rotation), segment.secondary.rest);
        const a = unit(sub(rotateVector(rotation, referenceLocal), mul(currentDirection, dot(rotateVector(rotation, referenceLocal), currentDirection))));
        const b = unit(sub(segment.secondary.current, mul(currentDirection, dot(segment.secondary.current, currentDirection))));
        if (a && b) rotation = multiplyQuaternion(axisAngle(currentDirection, Math.atan2(dot(currentDirection, cross(a, b)), clamp(dot(a, b), -1, 1))), rotation);
      }
      target = { rotation: normalizeQ(rotation), ...quality(segment.js), valid: true, twistStatus: segment.secondary ? 'estimatedFromPalm' : 'inheritedUnobservedTwist' };
    }
    const old = previous?.bones?.[name], neutralRotation = neutralRotations[name], oldDelta = vec(old?.rotation, 4) ? old.rotation : neutralRotation;
    const fresh = !!target && observationFresh && target.ageMs <= opts.maxJointAgeMs;
    const actualMeasurementAt = Number.isFinite(target?.observedAtMs) ? Math.min(measuredAt, target.observedAtMs) : measuredAt - (target?.ageMs || 0);
    const currentMeasurement = fresh && Math.abs(actualMeasurementAt - measuredAt) < 1e-6;
    const newMeasurement = fresh && (!Number.isFinite(old?.lastCandidateMs) || actualMeasurementAt > old.lastCandidateMs);
    const wasTracking = old?.trackingState === 'tracking' && Number.isFinite(old.lastObservedMs) && now - old.lastObservedMs <= opts.maxJointAgeMs;
    const priorActualAt = old?.lastActualObservedMs ?? old?.lastObservedMs, actualGap = actualMeasurementAt - priorActualAt;
    // Image gates already confirmed normal displacement for every endpoint.
    // Resume a short gap once, instead of adding a second two-frame wait here.
    // Repeated cached depth XY cannot satisfy currentMeasurement or advance
    // newMeasurement, and a jump/long gap retains the ordinary confirmation.
    const shortGapResumed = !wasTracking && currentMeasurement && target.continuityEligible === true &&
      Number.isFinite(priorActualAt) && actualGap > 0 && actualGap <= Math.min(180, opts.resumeGapMs) && vec(old?.lastObservedRotation, 4);
    const pendingRecovery = (old?.recoveryCount || 0) < opts.reacquireFrames ? old?.recoveryCount || 0 : 0;
    const recoveryCount = fresh ? wasTracking || shortGapResumed ? opts.reacquireFrames : pendingRecovery + (newMeasurement ? 1 : 0) : 0;
    const observed = fresh && recoveryCount >= opts.reacquireFrames && (wasTracking || currentMeasurement);
    const lastObservedMs = observed ? actualMeasurementAt : old?.lastObservedMs ?? null;
    const age = Number.isFinite(lastObservedMs) ? Math.max(0, now - lastObservedMs) : Infinity;
    let desired, trackingState, neutralWeight = 0;
    if (observed) {
      const relativeDesired = multiplyQuaternion(inverse(parent), target.rotation);
      const actualLocal = multiplyQuaternion(inverse(bone.intermediateRotation), relativeDesired);
      desired = normalizeQ(multiplyQuaternion(actualLocal, inverse(bone.localRotation)));
      trackingState = 'tracking';
    } else if (vec(old?.lastObservedRotation, 4) && age <= opts.holdMs) {
      desired = old.lastObservedRotation; trackingState = 'holding';
    } else {
      neutralWeight = smoothstep((age - opts.holdMs) / Math.max(1, opts.neutralMs));
      desired = vec(old?.lastObservedRotation, 4) ? slerp(old.lastObservedRotation, neutralRotation, neutralWeight) : neutralRotation;
      trackingState = fresh && !Number.isFinite(lastObservedMs) ? 'acquiring' : neutralWeight >= 1 ? 'neutral' : 'returning';
    }
    // Filter in local bone coordinates. Holding an old world rotation while its
    // parent moves creates a compensating twist; local hold preserves the limb.
    const alpha = opts.smoothingMs > 0 && dt > 0 && dt < 250 ? 1 - Math.exp(-dt / opts.smoothingMs) : 1;
    let delta = observed ? slerp(oldDelta, desired, alpha) : desired;
    if (!observed && !vec(old?.lastObservedRotation, 4)) delta = slerp(oldDelta, desired, 1 - Math.exp(-Math.max(dt, 0) / 120));
    const speedDeg = /^(hips|spine|chest|upperChest)$/.test(name) ? 120 : /^(head|neck)$/.test(name) ? 210 : /Foot$/.test(name) ? 360 : /Thumb|Index|Middle|Ring|Little/.test(name) ? 900 : 720;
    const maxStep = Math.min(Math.PI / 4, (observed ? speedDeg : Math.min(speedDeg, 360)) * Math.PI / 180 * Math.min(Math.max(dt, 0), 120) / 1000);
    delta = normalizeQ(limitRotation(oldDelta, delta, maxStep));
    const actualLocal = multiplyQuaternion(delta, bone.localRotation);
    solved[name] = normalizeQ(multiplyQuaternion(parent, multiplyQuaternion(bone.intermediateRotation, actualLocal)));
    normalizedPose[name] = { rotation: delta };
    outputBones[name] = { ...(outputBones[name] || {}), ...(target || {}), valid: observed, observed,
      source: observed ? target.source : trackingState === 'holding' ? 'held' : trackingState === 'returning' ? 'returningToNeutral' : 'missing',
      observationSource: observed ? target.source : old?.observationSource || null,
      score: observed ? target.score : 0, ageMs: Number.isFinite(age) ? age : null,
      trackingState, recoveryCount, lastCandidateMs: fresh ? actualMeasurementAt : old?.lastCandidateMs ?? null,
      shortGapResumed: !!shortGapResumed && observed, shortGapElapsedMs: shortGapResumed && observed ? actualGap : null,
      lastActualObservedMs: observed ? actualMeasurementAt : old?.lastActualObservedMs ?? old?.lastObservedMs ?? null,
      lastObservedMs, lastObservedRotation: observed ? delta.slice() : old?.lastObservedRotation || null,
      neutralWeight, neutralRotation, rotation: delta, worldRotation: solved[name] };
  }
  const currentRootObserved = ['measuredMultiView', 'estimatedMultiView'].includes(pose?.root?.source) || pose?.diagnostics?.depthAnchorSource === 'providedCurrentMeasurement';
  const metric = observationFresh && currentRootObserved && pose?.metric === true && pose?.coordinateFrame === 'world' && pose?.units === 'm' && pose?.root?.valid === true && vec(pose.root.position);
  const rootPosition = metric ? pose.root.position.slice() : null;
  const oldRoot = previous?.root, rootRest = rest.hips?.position || [0, 0, 0];
  const preview = observationFresh && pose?.previewRoot?.valid === true && vec(pose.previewRoot.position);
  const previewOrigin = vec(oldRoot?.previewOrigin) ? oldRoot.previewOrigin : preview ? pose.previewRoot.position.slice() : null;
  // A fixed first-observation origin keeps backwards motion visible. Never
  // recenter at every frame or label this monocular estimate as metric tracking.
  const previewPosition = preview ? add(rootRest, sub(pose.previewRoot.position, previewOrigin)) : null;
  const rootFresh = metric || preview;
  const rootAge = rootFresh ? Math.max(0, now - measuredAt) : Number.isFinite(oldRoot?.lastObservedMs) ? now - oldRoot.lastObservedMs : Infinity;
  const rootNeutralWeight = rootFresh ? 0 : smoothstep((rootAge - opts.holdMs) / Math.max(1, opts.neutralMs));
  const desiredRoot = metric ? rootPosition : preview ? previewPosition : vec(oldRoot?.lastObservedPosition) ? oldRoot.lastObservedPosition.map((v, i) => v * (1 - rootNeutralWeight) + rootRest[i] * rootNeutralWeight) : rootRest;
  let visualPosition = vec(oldRoot?.visualPosition) ? oldRoot.visualPosition.slice() : rootRest.slice();
  const rootDifference = sub(desiredRoot, visualPosition), rootDistance = norm(rootDifference), rootStep = opts.maxRootSpeed * Math.min(Math.max(dt, 0), 120) / 1000;
  visualPosition = rootDistance > rootStep && rootDistance > EPS ? add(visualPosition, mul(rootDifference, rootStep / rootDistance)) : desiredRoot.slice();
  if (rest.hips && normalizedPose.hips) {
    const worldDelta = sub(visualPosition, rest.hips.position);
    const parentRotation = multiplyQuaternion(rest.hips.rotation, inverse(rest.hips.localRotation));
    const scale = vec(rest.hips.parentScale) ? rest.hips.parentScale : [1, 1, 1];
    normalizedPose.hips.position = rotateVector(inverse(parentRotation), worldDelta).map((v, i) => v / (Math.abs(scale[i]) > EPS ? scale[i] : 1));
  }
  const validNames = Object.keys(outputBones).filter(name => outputBones[name].valid);
  if (!metric) warnings.push('absoluteRootUncalibrated');
  if (validNames.some(name => outputBones[name].depthAmbiguous)) warnings.push('footDirectionIsImagePlaneOnly');
  warnings.push('individualToeArticulationUnobserved');
  return { schemaVersion: 1, timestampMs: pose?.timestampMs ?? 0, inputKey, normalizedPose, bones: outputBones, neutralPolicy: { upperArms: 'rig-relative down with outward angle', outwardAngleDeg: opts.relaxedArmAngleDeg, otherBones: 'rest-local rotation', isObservation: false },
    root: { position: rootPosition, visualPosition, previewOrigin, lastObservedPosition: rootFresh ? visualPosition.slice() : oldRoot?.lastObservedPosition || null, lastObservedMs: rootFresh ? measuredAt : oldRoot?.lastObservedMs ?? null, metric, mode: metric ? 'world-estimate' : preview ? 'monocular-relative-motion' : 'relative-only', distanceStatus: preview ? 'estimatedMonocular' : pose?.root?.distanceStatus || 'uncalibrated' },
    trackers: poseToTrackers(pose), footReferenceAngles,
    coverage: { availableBones: order.length, drivenBones: validNames.length, drivenBoneNames: validNames, heldBones: Object.values(outputBones).filter(b => b.trackingState === 'holding').length, returningBones: Object.values(outputBones).filter(b => b.trackingState === 'returning').length, missingBones: order.length - validNames.length }, warnings: [...new Set(warnings)] };
}

/** Preserve the 2D-only foot contract; never invent 3D joints or a pelvis. */
export function footObservationToPose(observation = {}) {
  const footHints = {}, keypoints = observation.keypoints || [];
  for (const side of ['left', 'right']) {
    const ids = side === 'left' ? [15, 17, 18, 19] : [16, 20, 21, 22], values = ids.map(id => keypoints[id]);
    const track = (observation.footTracks || []).find(t => (t.modelSide || t.side) === side && t.source !== 'missing');
    const point = p => p && Number.isFinite(p.x) && Number.isFinite(p.y) && p.observable !== false && (p.score ?? 0) > 0 ? [p.x, p.y] : null;
    if (values.slice(1).every(p => point(p))) footHints[side] = { ankle: point(values[0]), bigToe: point(values[1]), littleToe: point(values[2]), heel: point(values[3]),
      score: Math.min(...values.slice(1).map(p => clamp(p.score, 0, 1))), timestampMs: observation.timestampMs ?? 0, imageSize: observation.imageSize,
      modelSide: side, anatomicalSideConfirmed: track?.anatomicalSideConfirmed === true || observation.anatomicalSideConfirmed === true,
      source: track?.source || values[1]?.source || 'observed2D', ageMs: Math.max(0, track?.ageMs || 0), depthKind: 'none' };
  }
  return { schemaVersion: 1, timestampMs: observation.timestampMs ?? 0, sourceId: observation.sourceId || 'feet', targetVersion: observation.targetVersion ?? null,
    footHintsSourceId: `${observation.sourceId || 'feet'}:${observation.targetVersion ?? ''}`, inputMode: 'feet',
    metric: false, units: 'unknown', coordinateFrame: 'image-plane', worldCalibrationId: null, depthKind: 'none',
    joints: Array.from({ length: 133 }, (_, id) => ({ id, position: null, score: 0, source: 'missing', ageMs: 0 })),
    root: { valid: false, position: null, metric: false, source: 'missing', score: 0, distanceStatus: 'uncalibrated' },
    feet: { left: { valid: false, source: 'observed2D', toes: [] }, right: { valid: false, source: 'observed2D', toes: [] } }, hands: {},
    footHints, footTracks: observation.footTracks || [], diagnostics: { ...observation.diagnostics, method: 'foot-only-image-plane', fullBodyObserved: false, anatomicalSideConfirmed: Object.values(footHints).length > 0 && Object.values(footHints).every(h => h.anatomicalSideConfirmed) },
    quality: { observed3D: 0, estimated3D: 0, missing: 133, observed2DFootPoints: keypoints.filter(p => pointFinite(p) && p.observable !== false && (p.score ?? 0) > 0).length } };
}
function pointFinite(p) { return !!p && Number.isFinite(p.x) && Number.isFinite(p.y); }

/** Physical tracker targets use the user's estimated skeleton, never avatar scale. */
export function poseToTrackers(pose, options = {}) {
  const opts = { minScore: .15, maxJointAgeMs: 180, ...options }, read = jointReader(pose, opts);
  const hips = [read(11), read(12)], shoulders = [read(5), read(6)];
  const world = pose?.metric === true && pose?.coordinateFrame === 'world' && pose?.units === 'm' && pose?.root?.valid === true && vec(pose.root.position) && typeof pose.worldCalibrationId === 'string' && pose.worldCalibrationId.length > 0;
  const rootObserved = ['measuredMultiView', 'estimatedMultiView'].includes(pose?.root?.source) || pose?.diagnostics?.depthAnchorSource === 'providedCurrentMeasurement';
  const calibrated = world && rootObserved && (pose.diagnostics?.calibratedWorld === true || pose.root?.distanceStatus === 'measuredMultiView');
  const targets = [];
  const torso = [...hips, ...shoulders].every(Boolean) && observedDepth([...hips, ...shoulders]) ? torsoFrame(hips[0].position, hips[1].position, mid(hips[0].position, hips[1].position), mid(shoulders[0].position, shoulders[1].position)) : null;
  const rootQuality = quality([...hips, ...shoulders]);
  targets.push({ role: 'waist', position: world ? pose.root.position.slice() : null, rotation: torso?.rotation || null, ...rootQuality,
    valid: !!(calibrated && torso && rootQuality.score >= opts.minScore && rootQuality.ageMs <= opts.maxJointAgeMs), reason: !world ? 'absoluteRootUncalibrated' : !calibrated ? 'worldCalibrationUnconfirmed' : !torso ? 'torsoMissing' : null });
  for (const side of ['left', 'right']) {
    const foot = pose?.feet?.[side], ids = side === 'left' ? [15, 17, 18, 19] : [16, 20, 21, 22], js = ids.map(read), q = quality(js);
    const rotation = foot?.rotationXYZW || foot?.rotation;
    targets.push({ role: side + '-foot', position: vec(foot?.position) ? foot.position.slice() : null, rotation: vec(rotation, 4) ? normalizeQ(rotation) : null, ...q,
      valid: !!(calibrated && foot?.valid && js.every(Boolean) && observedDepth(js) && q.score >= opts.minScore && q.ageMs <= opts.maxJointAgeMs),
      reason: !world ? 'absoluteRootUncalibrated' : !calibrated ? 'worldCalibrationUnconfirmed' : !foot?.valid || !js.every(Boolean) ? 'footMissing' : !observedDepth(js) ? 'footOrientationUnobserved' : null });
  }
  return targets;
}
