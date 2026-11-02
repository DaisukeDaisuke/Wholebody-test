import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { BODY_EDGES, HAND_EDGES, canonicalName } from './geometry.js';

const v3 = value => new THREE.Vector3(...value);
const limbSegments = [
  ['leftUpperArm', 'leftLowerArm', 'left_shoulder', 'left_elbow'],
  ['leftLowerArm', 'leftHand', 'left_elbow', 'left_wrist'],
  ['rightUpperArm', 'rightLowerArm', 'right_shoulder', 'right_elbow'],
  ['rightLowerArm', 'rightHand', 'right_elbow', 'right_wrist'],
  ['leftUpperLeg', 'leftLowerLeg', 'left_hip', 'left_knee'],
  ['leftLowerLeg', 'leftFoot', 'left_knee', 'left_ankle'],
  ['rightUpperLeg', 'rightLowerLeg', 'right_hip', 'right_knee'],
  ['rightLowerLeg', 'rightFoot', 'right_knee', 'right_ankle'],
];

function point(map, name) {
  const p = map.get(name);
  return p?.observable && Array.isArray(p.position) && p.position.every(Number.isFinite) ? v3(p.position) : null;
}
function midpoint(a, b) { return a && b ? a.clone().add(b).multiplyScalar(0.5) : null; }
function basisRotation(horizontal, vertical) {
  const x = horizontal.clone().normalize();
  const y = vertical.clone().addScaledVector(x, -vertical.dot(x)).normalize();
  if (x.lengthSq() < 0.9 || y.lengthSq() < 0.9) return null;
  const z = new THREE.Vector3().crossVectors(x, y).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
}

export class VrmView {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#101820');
    this.scene.fog = new THREE.Fog('#101820', 10, 30);
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.02, 80);
    this.camera.position.set(2.7, 1.7, 4.2);
    this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false, powerPreference: 'low-power' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.domElement.setAttribute('aria-label', '3D骨格とVRMプレビュー。マウスで回転、ホイールで拡大できます。');
    this.renderer.domElement.style.cssText = 'width:100%;height:100%;display:block;touch-action:none';
    container.appendChild(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.95, 0);
    this.controls.enableDamping = false;
    this.controls.minDistance = 0.3;
    this.controls.maxDistance = 20;
    this.scene.add(new THREE.HemisphereLight(0xe9f3ff, 0x334452, 2.2));
    const light = new THREE.DirectionalLight(0xffffff, 2.1);
    light.position.set(1, 3, 2); this.scene.add(light);
    const grid = new THREE.GridHelper(12, 24, 0x386b76, 0x263840);
    grid.position.y = -0.015; this.scene.add(grid);
    this.scene.add(new THREE.AxesHelper(0.35));
    this.linesGeometry = new THREE.BufferGeometry();
    this.linesGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(1024 * 3), 3).setUsage(THREE.DynamicDrawUsage));
    this.linesGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(1024 * 3), 3).setUsage(THREE.DynamicDrawUsage));
    this.linesGeometry.setDrawRange(0, 0);
    this.lines = new THREE.LineSegments(this.linesGeometry, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 }));
    this.lines.frustumCulled = false; this.scene.add(this.lines);
    this.pointsGeometry = new THREE.BufferGeometry();
    this.pointsGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(256 * 3), 3).setUsage(THREE.DynamicDrawUsage));
    this.pointsGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(256 * 3), 3).setUsage(THREE.DynamicDrawUsage));
    this.pointsGeometry.setDrawRange(0, 0);
    this.points = new THREE.Points(this.pointsGeometry, new THREE.PointsMaterial({ size: 0.018, vertexColors: true, sizeAttenuation: true }));
    this.points.frustumCulled = false; this.scene.add(this.points);
    this.rootMarker = new THREE.Mesh(new THREE.SphereGeometry(0.035, 8, 6), new THREE.MeshBasicMaterial({ color: 0xffc567 }));
    this.rootMarker.visible = false; this.scene.add(this.rootMarker);
    this.loader = new GLTFLoader();
    this.loader.register(parser => new VRMLoaderPlugin(parser));
    this.vrm = null; this.nodes = new Map(); this.rest = new Map();
    this.origin = null; this.initialRoot = null; this.lastTime = null;
    this.fused = null; this.skeletonVisible = true; this.lastRender = 0;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
  }

  async load(file) {
    const url = file instanceof Blob ? URL.createObjectURL(file) : String(file);
    let gltf;
    try { gltf = await this.loader.loadAsync(url); }
    finally { if (file instanceof Blob) URL.revokeObjectURL(url); }
    const vrm = gltf.userData.vrm;
    if (!vrm?.humanoid) { VRMUtils.deepDispose(gltf.scene); throw new Error('このファイルにはVRMのHumanoid定義がありません。VRM 0.x または 1.0 を選択してください。'); }
    if (this.vrm) { this.scene.remove(this.vrm.scene); VRMUtils.deepDispose(this.vrm.scene); }
    VRMUtils.rotateVRM0(vrm);
    vrm.scene.traverse(object => { if (object.isMesh) { object.frustumCulled = false; object.castShadow = false; object.receiveShadow = false; } });
    // Display normalization only: this value never calibrates the tracked person's dimensions.
    const bounds = new THREE.Box3().setFromObject(vrm.scene);
    const displayHeight = Math.max(0.1, bounds.max.y - bounds.min.y);
    vrm.scene.scale.multiplyScalar(1.7 / displayHeight);
    vrm.scene.updateMatrixWorld(true);
    const adjusted = new THREE.Box3().setFromObject(vrm.scene);
    vrm.scene.position.y -= adjusted.min.y;
    this.scene.add(vrm.scene); this.vrm = vrm;
    this.nodes.clear(); this.rest.clear();
    const boneNames = ['hips', 'spine', 'chest', 'upperChest', 'neck', 'head',
      ...['left', 'right'].flatMap(side => ['Shoulder', 'UpperArm', 'LowerArm', 'Hand', 'UpperLeg', 'LowerLeg', 'Foot', 'Toes',
        'ThumbMetacarpal', 'ThumbProximal', 'ThumbDistal', 'IndexProximal', 'IndexIntermediate', 'IndexDistal',
        'MiddleProximal', 'MiddleIntermediate', 'MiddleDistal', 'RingProximal', 'RingIntermediate', 'RingDistal',
        'LittleProximal', 'LittleIntermediate', 'LittleDistal'].map(name => side + name))];
    vrm.scene.updateMatrixWorld(true);
    for (const name of boneNames) {
      const bone = vrm.humanoid.getNormalizedBoneNode(name);
      if (!bone) continue;
      this.nodes.set(name, bone);
      this.rest.set(name, { position: bone.position.clone(), quaternion: bone.quaternion.clone(),
        worldPosition: bone.getWorldPosition(new THREE.Vector3()), worldQuaternion: bone.getWorldQuaternion(new THREE.Quaternion()),
        parentWorldPosition: bone.parent.getWorldPosition(new THREE.Vector3()) });
    }
    this.initialRoot = this.fused?.root?.valid ? v3(this.fused.root.worldXYZm) : null;
    this.avatarBaseHips = this.nodes.get('hips')?.getWorldPosition(new THREE.Vector3()) || new THREE.Vector3(0, 0.9, 0);
    this.avatarSceneBase = vrm.scene.position.clone();
    this.lastTime = null;
    return { version: vrm.meta?.metaVersion || '0', name: vrm.meta?.name || vrm.meta?.title || file.name || 'VRM', bones: this.nodes.size };
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth), height = Math.max(160, this.container.clientHeight);
    this.camera.aspect = width / height; this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  setSkeletonVisible(visible) { this.skeletonVisible = !!visible; this.lines.visible = !!visible; this.points.visible = !!visible; }
  resetOrigin() {
    this.origin = null; this.initialRoot = null; this.lastTime = null; this.worldFrameId = null;
    this.fused = null; this.linesGeometry.setDrawRange(0, 0); this.pointsGeometry.setDrawRange(0, 0); this.rootMarker.visible = false;
    if (this.vrm) { this.vrm.scene.visible = false; this.vrm.scene.position.copy(this.avatarSceneBase); }
  }

  update(fused) {
    if (fused?.root?.valid && fused.root.worldFrameId && this.worldFrameId !== fused.root.worldFrameId) {
      this.resetOrigin(); this.worldFrameId = fused.root.worldFrameId;
    }
    this.fused = fused;
    const visible = (fused?.keypoints || []).filter(p => p.observable && p.position?.every(Number.isFinite));
    const map = new Map(visible.map(p => [canonicalName(p.name), p]));
    if (!this.origin && visible.length) {
      const root = fused.root?.valid ? fused.root.worldXYZm : visible[0].position;
      const footY = visible.filter(p => /ankle|heel|big_toe/.test(p.name)).map(p => p.position[1]);
      const floorY = footY.length ? Math.min(...footY) : root[1] - 0.9;
      this.origin = new THREE.Vector3(root[0], floorY, root[2]);
    }
    const origin = this.origin || new THREE.Vector3();
    const linePos = this.linesGeometry.getAttribute('position'), lineColor = this.linesGeometry.getAttribute('color');
    let lineCount = 0;
    for (const [aName, bName] of [...BODY_EDGES, ...HAND_EDGES]) {
      const a = map.get(aName), b = map.get(bName);
      if (!a || !b || lineCount + 2 > linePos.count) continue;
      const measured = a.source === 'measuredMultiView' && b.source === 'measuredMultiView';
      const color = measured ? [0.24, 0.92, 0.56] : a.source === 'held' || b.source === 'held' ? [1, 0.67, 0.25] :
        a.source === 'boneConstrained' || b.source === 'boneConstrained' ? [0.8, 0.55, 1] : [0.25, 0.76, 1];
      for (const p of [a, b]) { linePos.setXYZ(lineCount, ...v3(p.position).sub(origin).toArray()); lineColor.setXYZ(lineCount, ...color); lineCount++; }
    }
    linePos.needsUpdate = true; lineColor.needsUpdate = true; this.linesGeometry.setDrawRange(0, lineCount);
    const positions = this.pointsGeometry.getAttribute('position'), colors = this.pointsGeometry.getAttribute('color');
    let count = 0;
    for (const p of visible) {
      if (count >= positions.count) break;
      positions.setXYZ(count, ...v3(p.position).sub(origin).toArray());
      colors.setXYZ(count, ...(p.source === 'measuredMultiView' ? [0.25, 1, 0.5] : p.source === 'held' ? [1, 0.67, 0.25] :
        p.source === 'boneConstrained' ? [0.8, 0.55, 1] : [0.4, 0.82, 1])); count++;
    }
    positions.needsUpdate = true; colors.needsUpdate = true; this.pointsGeometry.setDrawRange(0, count);
    this.rootMarker.visible = !!fused?.root?.valid;
    if (this.rootMarker.visible) this.rootMarker.position.copy(v3(fused.root.worldXYZm).sub(origin));
    if (!this.vrm) return;
    // Missing poses disappear; a frozen avatar must never masquerade as a new detection.
    this.vrm.scene.visible = !!(fused?.root?.valid && visible.length >= 5);
    if (!this.vrm.scene.visible) return;
    const time = fused.mediaTimeMs / 1000;
    const dt = this.lastTime === null ? 1 / 30 : Math.max(0.001, Math.min(0.1, time - this.lastTime));
    this.lastTime = time;
    if (!this.initialRoot) this.initialRoot = v3(fused.root.worldXYZm);
    const delta = v3(fused.root.worldXYZm).sub(this.initialRoot);
    this.vrm.scene.position.copy(this.avatarSceneBase).add(delta);
    for (const [name, node] of this.nodes) node.quaternion.copy(this.rest.get(name).quaternion);
    this.vrm.scene.updateMatrixWorld(true);
    const lh = point(map, 'left_hip'), rh = point(map, 'right_hip');
    const ls = point(map, 'left_shoulder'), rs = point(map, 'right_shoulder');
    const hipCenter = midpoint(lh, rh), shoulderCenter = midpoint(ls, rs);
    const torso = lh && rh && hipCenter && shoulderCenter ? basisRotation(lh.clone().sub(rh), shoulderCenter.clone().sub(hipCenter)) : null;
    if (torso) this.setWorldRotation('hips', torso);
    if (ls && rs && shoulderCenter && hipCenter) {
      const chest = basisRotation(ls.clone().sub(rs), shoulderCenter.clone().sub(hipCenter));
      if (chest) {
        const chestName = this.nodes.has('upperChest') ? 'upperChest' : this.nodes.has('chest') ? 'chest' : 'spine';
        this.setWorldRotation(chestName, chest);
      }
    }
    for (const [boneName, childName, start, end] of limbSegments) {
      const a = point(map, start), b = point(map, end);
      if (a && b) this.aimBone(boneName, childName, b.sub(a));
    }
    const leftEar = point(map, 'left_ear'), rightEar = point(map, 'right_ear');
    const nose = point(map, 'nose');
    if (leftEar && rightEar && nose) {
      const x = leftEar.clone().sub(rightEar).normalize();
      const face = nose.clone().sub(midpoint(leftEar, rightEar)).normalize();
      const up = new THREE.Vector3().crossVectors(face, x).normalize();
      const q = basisRotation(x, up);
      if (q) this.setWorldRotation('head', q);
    }
    for (const side of ['left', 'right']) {
      const heel = point(map, side + '_heel'), toe = point(map, side + '_big_toe');
      const ankle = point(map, side + '_ankle'), knee = point(map, side + '_knee');
      if (heel && toe && ankle) {
        const forward = toe.clone().sub(heel).normalize();
        let up = ankle.clone().sub(heel);
        if (up.clone().addScaledVector(forward, -up.dot(forward)).length() < 0.015 && knee) up = knee.clone().sub(ankle);
        up.addScaledVector(forward, -up.dot(forward)).normalize();
        const right = new THREE.Vector3().crossVectors(up, forward).normalize();
        const footQ = basisRotation(right, up);
        if (footQ) this.setWorldRotation(side + 'Foot', footQ);
      }
      const wrist = point(map, side + '_hand_wrist') || point(map, side + '_wrist');
      const index = point(map, side + '_index_mcp'), little = point(map, side + '_pinky_mcp');
      const middle = point(map, side + '_middle_mcp');
      if (wrist && index && little && middle) {
        const length = middle.clone().sub(wrist).normalize();
        const across = index.clone().sub(little).normalize();
        const palmNormal = new THREE.Vector3().crossVectors(across, length).normalize();
        const hand = this.nodes.get(side + 'Hand');
        const referenceChild = side + 'MiddleProximal';
        if (hand) this.aimBone(side + 'Hand', referenceChild, length, palmNormal);
      }
      for (const finger of ['thumb', 'index', 'middle', 'ring', 'pinky']) {
        const vrmFinger = finger === 'pinky' ? 'Little' : finger[0].toUpperCase() + finger.slice(1);
        const joints = finger === 'thumb' ? ['cmc', 'mcp', 'ip', 'tip'] : ['mcp', 'pip', 'dip', 'tip'];
        const bones = finger === 'thumb' ? ['Metacarpal', 'Proximal', 'Distal'] : ['Proximal', 'Intermediate', 'Distal'];
        for (let i = 0; i < 3; i++) {
          const a = point(map, side + '_' + finger + '_' + joints[i]);
          const b = point(map, side + '_' + finger + '_' + joints[i + 1]);
          if (a && b) this.aimBone(side + vrmFinger + bones[i], i < 2 ? side + vrmFinger + bones[i + 1] : null, b.sub(a));
        }
      }
    }
    this.vrm.update(dt);
  }

  setWorldRotation(name, desired) {
    const bone = this.nodes.get(name); if (!bone) return;
    bone.parent.updateWorldMatrix(true, false);
    const parent = bone.parent.getWorldQuaternion(new THREE.Quaternion());
    bone.quaternion.copy(parent.invert().multiply(desired));
    bone.updateWorldMatrix(false, true);
  }

  aimBone(name, childName, direction, targetNormal = null) {
    const bone = this.nodes.get(name), rest = this.rest.get(name);
    if (!bone || !rest || direction.lengthSq() < 1e-8) return;
    bone.parent.updateWorldMatrix(true, false);
    const parentQ = bone.parent.getWorldQuaternion(new THREE.Quaternion());
    const baseWorld = parentQ.clone().multiply(rest.quaternion);
    let localAxis;
    const childRest = this.rest.get(childName);
    if (childRest) localAxis = childRest.worldPosition.clone().sub(rest.worldPosition).applyQuaternion(rest.worldQuaternion.clone().invert()).normalize();
    else {
      // Terminal fingers have no humanoid tip bone. Continue the model's own
      // rest segment instead of assuming thumbs and all fingers point on X.
      localAxis = rest.worldPosition.clone().sub(rest.parentWorldPosition).applyQuaternion(rest.worldQuaternion.clone().invert());
      if (localAxis.lengthSq() < 1e-10) return;
      localAxis.normalize();
    }
    const currentAxis = localAxis.applyQuaternion(baseWorld).normalize();
    const delta = new THREE.Quaternion().setFromUnitVectors(currentAxis, direction.clone().normalize());
    const desired = delta.multiply(baseWorld);
    if (targetNormal && name.endsWith('Hand')) {
      const side = name.startsWith('left') ? 'left' : 'right';
      const index = this.rest.get(side + 'IndexProximal'), little = this.rest.get(side + 'LittleProximal');
      if (index && little && childRest) {
        const restAcross = index.worldPosition.clone().sub(little.worldPosition);
        const restLength = childRest.worldPosition.clone().sub(rest.worldPosition);
        const localNormal = new THREE.Vector3().crossVectors(restAcross, restLength).normalize().applyQuaternion(rest.worldQuaternion.clone().invert());
        const axis = direction.clone().normalize();
        const currentNormal = localNormal.applyQuaternion(desired);
        currentNormal.addScaledVector(axis, -currentNormal.dot(axis)).normalize();
        const wanted = targetNormal.clone().addScaledVector(axis, -targetNormal.dot(axis)).normalize();
        if (currentNormal.lengthSq() > 0.9 && wanted.lengthSq() > 0.9) {
          const angle = Math.atan2(new THREE.Vector3().crossVectors(currentNormal, wanted).dot(axis), currentNormal.dot(wanted));
          desired.premultiply(new THREE.Quaternion().setFromAxisAngle(axis, angle));
        }
      }
    }
    bone.quaternion.copy(parentQ.invert().multiply(desired));
    bone.updateWorldMatrix(false, true);
  }

  render() {
    const now = performance.now();
    // Rendering is capped independently from actual model inference throughput.
    if (now - this.lastRender < 1000 / 30) return;
    this.lastRender = now; this.controls.update(); this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.resizeObserver.disconnect(); this.controls.dispose();
    if (this.vrm) VRMUtils.deepDispose(this.vrm.scene);
    this.linesGeometry.dispose(); this.lines.material.dispose();
    this.pointsGeometry.dispose(); this.points.material.dispose();
    this.rootMarker.geometry.dispose(); this.rootMarker.material.dispose();
    this.scene.traverse(object => { if (object.type === 'GridHelper' || object.type === 'AxesHelper') { object.geometry.dispose(); if (Array.isArray(object.material)) object.material.forEach(m => m.dispose()); else object.material.dispose(); } });
    this.renderer.dispose(); this.renderer.domElement.remove();
  }
}
