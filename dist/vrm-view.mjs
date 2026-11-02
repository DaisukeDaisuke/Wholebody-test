import { THREE, OrbitControls, GLTFLoader, VRMLoaderPlugin, VRMUtils } from './vendor/vrm-runtime.mjs';
import { COCO_CONNECTIONS } from './geometry.mjs';
import { retargetPose } from './retarget.mjs';

const COLORS = { measured: 0xb8ee96, inferred: 0x68d8df, predicted: 0xf0bb77, demo: 0xcc9fff };
const finite3 = p => Array.isArray(p) && p.length >= 3 && p.slice(0, 3).every(Number.isFinite);
const finite2 = p => Array.isArray(p) && p.length >= 2 && p.slice(0, 2).every(Number.isFinite);
const colorFor = source => source === 'measuredMultiView' ? COLORS.measured : source === 'demo' ? COLORS.demo : /predict|held|prior|temporal/i.test(source || '') ? COLORS.predicted : COLORS.inferred;
const usableJoint = joint => finite3(joint?.position) && joint.source !== 'missing';
const visibleJoint = id => id < 23 || id >= 91;

// A .vrm is a binary glTF container. Reject URI references before a loader can
// resolve them; all images and buffers must be embedded in this selected file.
function readEmbeddedGLB(buffer) {
  if (buffer.byteLength < 20) throw new Error('VRMファイルが短すぎます。元の .vrm を選択してください。');
  const header = new DataView(buffer);
  if (header.getUint32(0, true) !== 0x46546c67 || header.getUint32(4, true) !== 2) {
    throw new Error('バイナリ形式の .vrm / .glb に対応しています。外部ファイルを参照する glTF は読み込めません。');
  }
  if (header.getUint32(8, true) !== buffer.byteLength) throw new Error('VRMファイルの長さが一致しません。ダウンロードし直した元ファイルを選択してください。');
  let document = null;
  for (let offset = 12; offset + 8 <= buffer.byteLength;) {
    const length = header.getUint32(offset, true), type = header.getUint32(offset + 4, true);
    if (length % 4 !== 0 || offset + 8 + length > buffer.byteLength) throw new Error('VRMファイルのデータが途中で切れているか、形式が一致しません。');
    if (type === 0x4e4f534a) {
      if (document) throw new Error('VRMのモデル情報が重複しています。配布元の元ファイルを選択してください。');
      document = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, offset + 8, length)).replace(/\0+$/g, '').trim());
    }
    offset += 8 + length;
  }
  if (!document) throw new Error('VRM内のモデル情報を読み取れません。');
  const visit = value => {
    if (!value || typeof value !== 'object') return;
    if (Array.isArray(value)) { value.forEach(visit); return; }
    for (const [key, child] of Object.entries(value)) {
      if (key.toLowerCase() === 'uri' && typeof child === 'string' && !child.startsWith('data:')) {
        throw new Error('外部の画像・データを参照するVRMです。テクスチャを埋め込んだ .vrm を選択してください。外部通信は行っていません。');
      }
      visit(child);
    }
  };
  visit(document);
  return document;
}

function disposeObject(object) {
  if (!object) return;
  const geometries = new Set(), materials = new Set(), textures = new Set();
  object.traverse(node => {
    if (node.geometry) geometries.add(node.geometry);
    const list = Array.isArray(node.material) ? node.material : node.material ? [node.material] : [];
    list.forEach(material => materials.add(material));
  });
  for (const material of materials) {
    for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
    for (const uniform of Object.values(material.uniforms || {})) if (uniform?.value?.isTexture) textures.add(uniform.value);
  }
  for (const texture of textures) {
    texture.dispose();
    const image = texture.source?.data || texture.image;
    if (typeof image?.close === 'function') image.close();
  }
  materials.forEach(material => material.dispose());
  geometries.forEach(geometry => geometry.dispose());
  object.removeFromParent();
}

export function captureRig(vrm, displayGroup) {
  displayGroup.updateMatrixWorld(true);
  const bones = {}, nodes = new Map(), inverseGroup = displayGroup.matrixWorld.clone().invert();
  const inverseRotation = displayGroup.getWorldQuaternion(new THREE.Quaternion()).invert();
  const groupScale = displayGroup.getWorldScale(new THREE.Vector3());
  for (const name of Object.keys(vrm.humanoid.normalizedHumanBones)) {
    const node = vrm.humanoid.getNormalizedBoneNode(name);
    if (node) nodes.set(node, name);
  }
  for (const [node, name] of nodes) {
    let parentNode = node.parent;
    while (parentNode && !nodes.has(parentNode)) parentNode = parentNode.parent;
    const position = node.getWorldPosition(new THREE.Vector3()).applyMatrix4(inverseGroup);
    const rotation = inverseRotation.clone().multiply(node.getWorldQuaternion(new THREE.Quaternion())).normalize();
    bones[name] = { parent: parentNode ? nodes.get(parentNode) : null, position: position.toArray(), rotation: rotation.toArray(), localRotation: node.quaternion.toArray() };
    if (name === 'hips' && node.parent) {
      const scale = node.parent.getWorldScale(new THREE.Vector3());
      bones[name].parentScale = scale.divide(groupScale).toArray();
    }
  }
  return { bones, hipsPosition: bones.hips?.position?.slice() };
}

function updateGeometry(geometry, positions, colors) {
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();
}

// Shared by the browser viewer and the Node integration in GitHub Actions.
// No renderer or DOM is required for this local-buffer loading path.
export async function loadEmbeddedVRM(buffer) {
  readEmbeddedGLB(buffer);
  const blobURLs = new Set();
  let pendingScene = null;
  try {
    const manager = new THREE.LoadingManager();
    manager.setURLModifier(url => {
      if (url.startsWith('data:')) return url;
      // File JSON cannot contain blob URIs. These are the embedded image blobs
      // generated by this instance of GLTFLoader after the container check.
      if (url.startsWith('blob:')) { blobURLs.add(url); return url; }
      throw new Error('VRMの外部リソース参照を拒否しました。テクスチャを埋め込んだファイルを選択してください。');
    });
    const loader = new GLTFLoader(manager);
    loader.register(parser => new VRMLoaderPlugin(parser));
    const gltf = await loader.parseAsync(buffer, '');
    pendingScene = gltf.scene;
    const candidate = gltf.userData?.vrm;
    if (!candidate?.humanoid) throw new Error('VRMの人体ボーンが見つかりません。VRM 0.x / 1.0 のモデルを選択してください。');
    VRMUtils.rotateVRM0(candidate);
    candidate.humanoid.resetNormalizedPose();
    candidate.update(0);
    pendingScene = null;
    return candidate;
  } catch (error) {
    disposeObject(pendingScene);
    throw error;
  } finally {
    for (const url of blobURLs) URL.revokeObjectURL(url);
  }
}

export function createVRMView({ container, onState = () => {}, onRenderMetrics = () => {}, onError = () => {} } = {}) {
  if (!container) throw new Error('3D表示領域が見つかりません。');
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x101b24);
  const camera = new THREE.PerspectiveCamera(36, 1, 0.01, 100);
  camera.position.set(2.3, 1.65, 4.3);
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 1.75));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.setAttribute('aria-label', 'VRMと推定骨格の3D表示。ドラッグで回転、ホイールで拡大。');
  container.append(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.9, 0);
  controls.enableDamping = true;
  controls.minDistance = 0.1;
  controls.maxDistance = 40;
  const hemisphere = new THREE.HemisphereLight(0xdceeff, 0x405145, 2.1);
  const keyLight = new THREE.DirectionalLight(0xfff6e8, 2.4);
  keyLight.position.set(-2, 4, 3);
  scene.add(hemisphere, keyLight);
  const grid = new THREE.GridHelper(10, 40, 0x426169, 0x263b45);
  grid.material.transparent = true;
  grid.material.opacity = 0.48;
  scene.add(grid);
  const avatarGroup = new THREE.Group();
  scene.add(avatarGroup);
  const skeletonGroup = new THREE.Group();
  const lineGeometry = new THREE.BufferGeometry(), pointGeometry = new THREE.BufferGeometry();
  const lines = new THREE.LineSegments(lineGeometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 }));
  const points = new THREE.Points(pointGeometry, new THREE.PointsMaterial({ vertexColors: true, size: 0.018, sizeAttenuation: true }));
  skeletonGroup.add(lines, points);
  scene.add(skeletonGroup);
  let vrm = null, rig = null, fileName = '', pose = null, sourceMode = 'live', displayMode = 'skeleton';
  let disposed = false, loading = false, generation = 0, lastRetarget = null, poseIdentity = null, framed = false;
  let frameId = 0, lastTick = performance.now(), metricStart = lastTick, renderFrames = 0, renderMsTotal = 0;
  let lastStateKey = '', loadError = '', lastPoseError = '';

  const poseSummary = () => {
    const validJointCount = (pose?.joints || []).filter((joint, id) => visibleJoint(id) && usableJoint(joint)).length;
    const footHints = pose?.footHints || {};
    const planarHints = Object.values(footHints).filter(foot => finite2(foot?.bigToe) && finite2(foot?.littleToe) && finite2(foot?.heel));
    const planarFeet = planarHints.length;
    const planarPointCount = planarHints.reduce((sum, foot) => sum + [foot.ankle, foot.bigToe, foot.littleToe, foot.heel].filter(finite2).length, 0);
    const anatomicalSidesConfirmed = planarFeet > 0 && planarHints.every(foot => foot.anatomicalSideConfirmed === true);
    const boneResults = Object.values(lastRetarget?.bones || {});
    const appliedBoneCount = boneResults.filter(bone => bone.valid).length;
    const feetOnly = pose?.inputMode === 'feet' || pose?.feetOnly === true || pose?.coordinateFrame === 'image-plane';
    return { validJointCount, planarFeet, planarPointCount, anatomicalSidesConfirmed, appliedBoneCount, feetOnly, coverage: lastRetarget?.coverage || null, warnings: lastRetarget?.warnings || [] };
  };
  const status = () => ({ ready: !disposed, loading, hasVRM: Boolean(vrm), fileName, sourceMode, displayMode, loadError, ...poseSummary() });
  function notify(force = false) {
    const value = status(), key = JSON.stringify(value);
    if (force || key !== lastStateKey) { lastStateKey = key; onState(value); }
  }
  function visibility() {
    avatarGroup.visible = Boolean(vrm) && displayMode === 'avatar';
    skeletonGroup.visible = !vrm || displayMode === 'skeleton';
  }
  function resetView() {
    const object = avatarGroup.visible ? avatarGroup : skeletonGroup;
    object.updateMatrixWorld(true);
    object.traverse(node => { if (node.isSkinnedMesh) node.computeBoundingBox(); });
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) { camera.position.set(2.3, 1.65, 4.3); controls.target.set(0, 0.9, 0); }
    else {
      const center = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
      const extent = Math.max(size.x, size.y, size.z, 0.35);
      const distance = extent / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.4;
      controls.target.copy(center);
      camera.position.copy(center).add(new THREE.Vector3(0.28, 0.12, 1).normalize().multiplyScalar(distance));
    }
    controls.update();
  }
  function drawSkeleton(nextPose) {
    const positions = [], colors = [], jointPositions = [], jointColors = [];
    const color = new THREE.Color();
    const line = (a, b, source) => {
      color.setHex(colorFor(source));
      positions.push(...a, ...b); colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    };
    const dot = (p, source) => { color.setHex(colorFor(source)); jointPositions.push(...p); jointColors.push(color.r, color.g, color.b); };
    const joints = nextPose?.joints || [];
    for (const [a, b] of COCO_CONNECTIONS) {
      if (visibleJoint(a) && visibleJoint(b) && usableJoint(joints[a]) && usableJoint(joints[b])) {
        const source = /predict|held|prior|temporal/i.test(joints[a].source || '') ? joints[a].source : joints[b].source;
        line(joints[a].position, joints[b].position, sourceMode === 'demo' ? 'demo' : source);
      }
    }
    joints.forEach((joint, id) => { if (visibleJoint(id) && usableJoint(joint)) dot(joint.position, sourceMode === 'demo' ? 'demo' : joint.source); });
    if (!jointPositions.length && nextPose?.footHints) {
      for (const foot of Object.values(nextPose.footHints)) {
        if (!finite2(foot?.heel) || !finite2(foot?.bigToe) || !finite2(foot?.littleToe)) continue;
        const all = [foot.heel, foot.bigToe, foot.littleToe, foot.ankle].filter(finite2);
        const width = foot.imageSize?.width || Math.max(...all.map(p => p[0])) * 1.1 || 1;
        const height = foot.imageSize?.height || Math.max(...all.map(p => p[1])) * 1.1 || 1;
        const scale = 2 / Math.max(width, height);
        const project = p => [(p[0] - width / 2) * scale, 1 + (height / 2 - p[1]) * scale, 0];
        const heel = project(foot.heel), bigToe = project(foot.bigToe), littleToe = project(foot.littleToe);
        line(heel, bigToe, 'estimatedPlanarFoot'); line(heel, littleToe, 'estimatedPlanarFoot'); line(bigToe, littleToe, 'estimatedPlanarFoot');
        for (const p of all) dot(project(p), 'estimatedPlanarFoot');
        if (finite2(foot.ankle)) line(project(foot.ankle), heel, 'estimatedPlanarFoot');
      }
    }
    updateGeometry(lineGeometry, positions, colors);
    updateGeometry(pointGeometry, jointPositions, jointColors);
  }
  function updatePose(nextPose, options = {}) {
    if (disposed) return;
    pose = nextPose || null;
    sourceMode = options.sourceMode === 'demo' ? 'demo' : 'live';
    const identity = [sourceMode, pose?.inputMode || 'wholebody', pose?.sourceId || pose?.footHintsSourceId || '', pose?.targetVersion || '', pose?.coordinateFrame || '', pose?.worldCalibrationId || '', pose?.units || ''].join('|');
    if (identity !== poseIdentity) { poseIdentity = identity; lastRetarget = null; framed = false; }
    drawSkeleton(pose);
    if (vrm && rig) {
      try {
        vrm.humanoid.resetNormalizedPose();
        lastRetarget = retargetPose(pose, rig, { previous: lastRetarget, footHints: pose?.footHints });
        vrm.humanoid.setNormalizedPose(lastRetarget.normalizedPose || {});
        // retargetPose includes calibrated root displacement in hips.position.
        // The display group never adds root.position a second time.
        vrm.update(0);
        lastPoseError = '';
      } catch (error) {
        lastRetarget = null;
        if (lastPoseError !== error.message) { lastPoseError = error.message; onError(error); }
      }
    }
    visibility();
    const info = poseSummary();
    if (!framed && (info.validJointCount || info.planarFeet || vrm)) { resetView(); framed = true; }
    notify();
  }
  async function loadFile(file) {
    if (disposed) throw new Error('3D表示を閉じています。');
    if (!file || typeof file.arrayBuffer !== 'function') throw new Error('端末内の .vrm ファイルを選択してください。');
    const myGeneration = ++generation;
    loading = true; loadError = ''; notify(true);
    let candidate = null;
    try {
      const buffer = await file.arrayBuffer();
      if (disposed || myGeneration !== generation) return null;
      candidate = await loadEmbeddedVRM(buffer);
      if (disposed || myGeneration !== generation) { disposeObject(candidate.scene); return null; }
      candidate.scene.traverse(node => { if (node.isMesh) node.frustumCulled = false; });
      const previousVRM = vrm;
      avatarGroup.add(candidate.scene);
      const nextRig = captureRig(candidate, avatarGroup);
      if (!nextRig.bones.hips) { disposeObject(candidate.scene); candidate = null; throw new Error('VRMに hips ボーンがありません。別のVRMを選択してください。'); }
      if (previousVRM) disposeObject(previousVRM.scene);
      vrm = candidate; rig = nextRig; fileName = file.name || 'ローカルVRM'; displayMode = 'avatar';
      lastRetarget = null; framed = false;
      updatePose(pose, { sourceMode });
      return { fileName, boneCount: Object.keys(rig.bones).length };
    } catch (error) {
      if (candidate && candidate !== vrm) disposeObject(candidate.scene);
      if (myGeneration === generation && !disposed) { loadError = error.message || String(error); onError(error); }
      throw error;
    } finally {
      if (myGeneration === generation && !disposed) { loading = false; notify(true); }
    }
  }
  function unload() {
    generation++; loading = false; loadError = '';
    if (vrm) disposeObject(vrm.scene);
    vrm = null; rig = null; fileName = ''; lastRetarget = null; displayMode = 'skeleton'; framed = false;
    updatePose(pose, { sourceMode }); notify(true);
  }
  function setDisplayMode(mode) {
    displayMode = mode === 'avatar' && vrm ? 'avatar' : 'skeleton';
    visibility(); resetView(); notify();
  }
  function updateExpressions(face) {
    if (!vrm?.expressionManager) return;
    const clamp = value => Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
    vrm.expressionManager.setValue('blinkLeft', face?.valid ? clamp(face.blinkLeft) : 0);
    vrm.expressionManager.setValue('blinkRight', face?.valid ? clamp(face.blinkRight) : 0);
    vrm.expressionManager.setValue('aa', face?.valid ? clamp(face.mouthOpen) : 0);
    // A contour-width proxy is not a cheek-puff blendshape observation.
    vrm.expressionManager.update();
  }
  const resize = () => {
    if (disposed) return;
    const width = Math.max(container.clientWidth, 1), height = Math.max(container.clientHeight, 1);
    renderer.setSize(width, height, false);
    camera.aspect = width / height; camera.updateProjectionMatrix();
  };
  const observer = new ResizeObserver(resize);
  observer.observe(container);
  resize();
  function tick(now) {
    if (disposed) return;
    const dt = Math.min(Math.max((now - lastTick) / 1000, 0), 0.05); lastTick = now;
    const start = performance.now();
    controls.update();
    vrm?.update(dt);
    renderer.render(scene, camera);
    renderMsTotal += performance.now() - start; renderFrames++;
    if (now - metricStart >= 1000) {
      onRenderMetrics({ fps: renderFrames * 1000 / (now - metricStart), renderMs: renderMsTotal / renderFrames, sourceMode, measurement: 'cpu-submit-and-vrm-update' });
      metricStart = now; renderFrames = 0; renderMsTotal = 0;
    }
    frameId = requestAnimationFrame(tick);
  }
  frameId = requestAnimationFrame(tick);
  notify(true);
  return {
    updatePose, updateExpressions, loadFile, unload, resetView, setDisplayMode, getStatus: status,
    dispose() {
      if (disposed) return;
      disposed = true; generation++; cancelAnimationFrame(frameId); observer.disconnect(); controls.dispose();
      disposeObject(avatarGroup); disposeObject(skeletonGroup); disposeObject(grid);
      renderer.dispose(); renderer.domElement.remove();
    }
  };
}
