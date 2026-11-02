/** GitHub Actions only: load the actual browser VRM runtime without WebGL. */
import { mkdir, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { THREE } from '../dist/vendor/vrm-runtime.mjs';
import { loadEmbeddedVRM, captureRig } from '../dist/vrm-view.mjs';
import { createSampleVRM, SAMPLE_VRM_INFO } from '../dist/vrm-fixture.mjs';
import { retargetPose, footObservationToPose } from '../dist/retarget.mjs';

const report = { schemaVersion: 1, status: 'running', scope: 'Generated VRM 1.0 GLB → production loader → normalized humanoid → raw bones and skinned mesh',
  syntheticFixture: true, realVideoAccuracyMeasured: false, privateDataUsed: false, webGLUsed: false, startedAt: new Date().toISOString(), checks: [] };
const requireCondition = (condition, message) => { if (!condition) throw new Error(message); };
const distance = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));
const pos = node => node.getWorldPosition(new THREE.Vector3()).toArray();
const quat = node => node.getWorldQuaternion(new THREE.Quaternion());
const upperBody = ['hips', 'spine', 'chest', 'upperChest', 'neck', 'head', 'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand', 'rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand'];
const requiredBones = ['hips', 'spine', 'head', 'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'leftUpperArm', 'leftLowerArm', 'leftHand', 'rightUpperArm', 'rightLowerArm', 'rightHand'];
const fetchBefore = globalThis.fetch;
let networkAttempts = 0;
globalThis.fetch = async () => { networkAttempts++; throw new Error('The generated VRM integration must not request a network resource.'); };

try {
  const buffer = createSampleVRM();
  const vrm = await loadEmbeddedVRM(buffer), group = new THREE.Group();
  group.add(vrm.scene); group.updateMatrixWorld(true);
  const rig = captureRig(vrm, group), meshes = [];
  vrm.scene.traverse(node => { if (node.isSkinnedMesh) meshes.push(node); });
  requireCondition(requiredBones.every(name => vrm.humanoid.getRawBoneNode(name) && vrm.humanoid.getNormalizedBoneNode(name)), 'A required VRM humanoid bone was not loaded.');
  requireCondition(meshes.length > 0 && meshes.every(mesh => mesh.geometry.attributes.skinIndex && mesh.geometry.attributes.skinWeight), 'Fixture did not load as a mesh with skin indices and weights.');
  const mesh = meshes[0], hand = vrm.humanoid.getRawBoneNode('leftHand'), foot = vrm.humanoid.getRawBoneNode('leftFoot');
  const rawUpperRest = Object.fromEntries(upperBody.map(name => [name, quat(vrm.humanoid.getRawBoneNode(name))]));
  const restHand = pos(hand), restFootRotation = quat(foot);
  const skinPoint = index => mesh.applyBoneTransform(index, new THREE.Vector3().fromBufferAttribute(mesh.geometry.attributes.position, index)).applyMatrix4(mesh.matrixWorld).toArray();
  const upperArmJoint = mesh.skeleton.bones.indexOf(vrm.humanoid.getRawBoneNode('leftUpperArm'));
  requireCondition(upperArmJoint >= 0, 'The actual upper-arm bone is missing from the loaded mesh skin.');
  const weightedVertices = [];
  for (let i = 0; i < mesh.geometry.attributes.position.count; i++) {
    if (mesh.geometry.attributes.skinIndex.getX(i) === upperArmJoint && mesh.geometry.attributes.skinWeight.getX(i) > 0.99) weightedVertices.push(i);
  }
  requireCondition(weightedVertices.length > 0, 'No actual mesh vertices are influenced by the test bone.');
  const restVertices = weightedVertices.map(skinPoint);
  const initialBindErrorM = Math.max(...weightedVertices.map((index, i) => distance(restVertices[i], new THREE.Vector3().fromBufferAttribute(mesh.geometry.attributes.position, index).applyMatrix4(mesh.matrixWorld).toArray())));
  requireCondition(initialBindErrorM < 1e-5, 'The mesh inverse bind matrices do not match the bone rest pose.');
  report.fixture = { ...SAMPLE_VRM_INFO, bytes: buffer.byteLength, sha256: createHash('sha256').update(new Uint8Array(buffer)).digest('hex'), humanoidBones: Object.keys(rig.bones).length, skinnedMeshes: meshes.length, vertices: mesh.geometry.attributes.position.count };
  report.checks.push({ name: 'load-generated-vrm-and-validate-skin', passed: true, requiredBones: requiredBones.length, weightedUpperArmVertices: weightedVertices.length, initialBindErrorM });

  const apply = result => {
    vrm.humanoid.resetNormalizedPose(); vrm.humanoid.setNormalizedPose(result.normalizedPose); vrm.update(0); group.updateMatrixWorld(true);
    meshes.forEach(item => item.skeleton.update());
  };
  function wholebodyPose(timestampMs) {
    const positions = new Map(), root = rig.bones.hips.position;
    const mapping = [[0, 'head'], [5, 'leftUpperArm'], [6, 'rightUpperArm'], [7, 'leftLowerArm'], [8, 'rightLowerArm'], [9, 'leftHand'], [10, 'rightHand'], [11, 'leftUpperLeg'], [12, 'rightUpperLeg'], [13, 'leftLowerLeg'], [14, 'rightLowerLeg'], [15, 'leftFoot'], [16, 'rightFoot']];
    for (const [id, name] of mapping) positions.set(id, rig.bones[name].position.map((v, i) => v - root[i]));
    return { timestampMs, sourceId: 'generated-ci-pose', inputMode: 'wholebody', metric: false, units: 'm-relative-estimate', coordinateFrame: 'root-relative',
      root: { valid: false, position: null, metric: false, distanceStatus: 'uncalibrated' },
      joints: Array.from({ length: 133 }, (_, id) => ({ id, position: positions.get(id) || null, score: positions.has(id) ? 1 : 0, source: positions.has(id) ? 'inferredMonocular' : 'missing', ageMs: 0 })) };
  }
  const raised = wholebodyPose(1000), shoulder = raised.joints[5].position, angle = Math.PI / 3;
  for (const id of [7, 9]) {
    const delta = raised.joints[id].position.map((v, i) => v - shoulder[i]);
    raised.joints[id].position = [shoulder[0] + delta[0] * Math.cos(angle) - delta[1] * Math.sin(angle), shoulder[1] + delta[0] * Math.sin(angle) + delta[1] * Math.cos(angle), shoulder[2] + delta[2]];
  }
  const armResult = retargetPose(raised, rig, { smoothingMs: 0 });
  apply(armResult);
  const rawHandDisplacementM = distance(restHand, pos(hand));
  const maxSkinnedVertexDisplacementM = Math.max(...weightedVertices.map((index, i) => distance(restVertices[i], skinPoint(index))));
  const rawUpperArmRotationRad = rawUpperRest.leftUpperArm.angleTo(quat(vrm.humanoid.getRawBoneNode('leftUpperArm')));
  requireCondition(armResult.bones.leftUpperArm.valid && rawHandDisplacementM > 0.1 && rawUpperArmRotationRad > 0.5 && maxSkinnedVertexDisplacementM > 0.05, 'Normalized-pose changes did not reach both the actual raw arm and its skinned mesh.');
  report.checks.push({ name: 'normalized-to-raw-arm-and-skinned-mesh', passed: true, rawHandDisplacementM, rawUpperArmRotationRad, maxSkinnedVertexDisplacementM, rootIsMetric: armResult.root.metric, source: 'synthetic input pose, not image inference' });

  const missing = { ...wholebodyPose(1100), joints: Array.from({ length: 133 }, (_, id) => ({ id, position: null, score: 0, source: 'missing', ageMs: 0 })) };
  const missingResult = retargetPose(missing, rig, { previous: armResult, smoothingMs: 0 });
  apply(missingResult);
  const missingDrivenBones = Object.values(missingResult.bones).filter(bone => bone.valid).length;
  const returnedHandErrorM = distance(restHand, pos(hand));
  requireCondition(missingDrivenBones === 0 && missingResult.root.metric === false && returnedHandErrorM < 1e-5, 'Missing input retained observed bone claims or did not restore the rest pose.');
  report.checks.push({ name: 'missing-is-unobserved-and-restored', passed: true, drivenBones: missingDrivenBones, returnedHandErrorM, validTrackers: missingResult.trackers.filter(tracker => tracker.valid).length });

  function footObservation(timestampMs, rotation) {
    const keypoints = Array.from({ length: 133 }, () => ({ x: 0, y: 0, score: 0, observable: false })), pivot = [100, 190];
    for (const [id, p] of [[15, [100, 165]], [17, [90, 80]], [18, [110, 80]], [19, pivot]]) {
      const x = p[0] - pivot[0], y = p[1] - pivot[1];
      keypoints[id] = { x: pivot[0] + x * Math.cos(rotation) - y * Math.sin(rotation), y: pivot[1] + x * Math.sin(rotation) + y * Math.cos(rotation), score: 1, observable: true };
    }
    return { timestampMs, sourceId: 'generated-foot-plane', imageSize: { width: 320, height: 240 }, anatomicalSideConfirmed: true, keypoints };
  }
  const firstFoot = retargetPose(footObservationToPose(footObservation(1200, 0)), rig, { smoothingMs: 0 });
  const planarResult = retargetPose(footObservationToPose(footObservation(1233, -0.65)), rig, { previous: firstFoot, smoothingMs: 0 });
  apply(planarResult);
  const upperBodyClaimed = upperBody.filter(name => planarResult.bones[name]?.valid);
  const maxUpperBodyRotationRad = Math.max(...upperBody.map(name => rawUpperRest[name].angleTo(quat(vrm.humanoid.getRawBoneNode(name)))));
  const planarFootRotationRad = restFootRotation.angleTo(quat(foot));
  const footOnlyHandDisplacementM = distance(restHand, pos(hand));
  const validPlanarTrackers = planarResult.trackers.filter(tracker => tracker.valid).length;
  requireCondition(upperBodyClaimed.length === 0 && maxUpperBodyRotationRad < 1e-5 && footOnlyHandDisplacementM < 1e-5, 'A 2D foot-only input moved or claimed an observed upper body.');
  requireCondition(planarResult.bones.leftFoot?.valid && planarResult.bones.leftFoot.source === 'estimatedPlanarFoot' && planarResult.bones.leftFoot.depthAmbiguous === true && planarFootRotationRad > 0.3, 'Foot-only image-plane rotation did not reach the actual foot bone.');
  requireCondition(planarResult.root.metric === false && validPlanarTrackers === 0, 'A 2D foot-only input emitted valid physical-space tracking.');
  report.checks.push({ name: 'foot-only-moves-foot-without-observing-upper-body', passed: true, planarFootRotationRad, maxUpperBodyRotationRad, footOnlyHandDisplacementM, observedUpperBodyBones: upperBodyClaimed.length, validTrackers: validPlanarTrackers, footSource: planarResult.bones.leftFoot.source, depthAmbiguous: true });
  requireCondition(networkAttempts === 0, 'Loading the embedded generated VRM attempted network access.');
  report.status = 'passed'; report.passed = true;
} catch (error) {
  report.status = 'failed'; report.passed = false;
  report.error = { name: error.name, message: error.message, stack: error.stack };
  process.exitCode = 1;
} finally {
  globalThis.fetch = fetchBefore;
  report.networkAttempts = networkAttempts; report.finishedAt = new Date().toISOString();
  await mkdir(new URL('./results/', import.meta.url), { recursive: true });
  await writeFile(new URL('./results/phase3-vrm-integration.json', import.meta.url), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
}
