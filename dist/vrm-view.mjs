import { THREE, OrbitControls, GLTFLoader, VRMLoaderPlugin, VRMUtils } from './vendor/vrm-runtime.mjs';
import { COCO_CONNECTIONS } from './geometry.mjs';
import { retargetPose } from './retarget.mjs';

const COLORS = { measured: 0xb8ee96, inferred: 0x68d8df, predicted: 0xf0bb77, demo: 0xcc9fff };
const finite3 = p => Array.isArray(p) && p.length >= 3 && p.slice(0, 3).every(Number.isFinite);
const finite2 = p => Array.isArray(p) && p.length >= 2 && p.slice(0, 2).every(Number.isFinite);
const colorFor = source => source === 'measuredMultiView' ? COLORS.measured : source === 'demo' ? COLORS.demo : /predict|held|prior|temporal/i.test(source || '') ? COLORS.predicted : COLORS.inferred;
const usableJoint = joint => finite3(joint?.position) && joint.source !== 'missing';
const visibleJoint = id => id < 23 || id >= 91;
const CPU_RENDER_LIMITS = Object.freeze({ faces: 6000, vertices: 18000, maxFps: 24 });

// This renderer projects the same animated Three.js objects as the WebGL path.
// Mesh.getVertexPosition includes morph targets and SkinnedMesh bone weights;
// no replacement avatar, animation, or inferred coordinates are introduced.
function createCanvas2DRenderer() {
  const canvas = document.createElement('canvas'), context = canvas.getContext('2d', { alpha: false });
  if (!context) throw new Error('WebGLとCanvas 2Dの描画を初期化できません。別のブラウザで表示を開いてください。');
  let width = 1, height = 1, pixelRatio = 1, avatar = null, meshPlans = [], boneEdges = [], boneStates = {}, demo = false;
  let summary = { totalFaces: 0, totalVertices: 0, selectedFaces: 0, selectedVertices: 0, unsupportedMeshes: 0, limited: false };
  const viewProjection = new THREE.Matrix4(), clipMatrix = new THREE.Matrix4();
  const vertex = new THREE.Vector3(), aWorld = new THREE.Vector3(), bWorld = new THREE.Vector3();
  const edgeA = new THREE.Vector3(), edgeB = new THREE.Vector3(), normal = new THREE.Vector3();
  const tint = new THREE.Color(), baseTint = new THREE.Color(), attributeTint = new THREE.Color();
  const lightDirection = new THREE.Vector3(-0.3, 0.75, 0.6).normalize();
  const planes = [[1, 0, 0, 1], [-1, 0, 0, 1], [0, 1, 0, 1], [0, -1, 0, 1], [0, 0, 1, 1], [0, 0, -1, 1]];
  const distanceToPlane = (p, plane) => p[0] * plane[0] + p[1] * plane[1] + p[2] * plane[2] + p[3];
  const between = (a, b, t) => a.map((value, i) => value + (b[i] - value) * t);
  const clipPoint = (point, matrix) => {
    const e = matrix.elements, { x, y, z } = point;
    return [e[0] * x + e[4] * y + e[8] * z + e[12], e[1] * x + e[5] * y + e[9] * z + e[13],
      e[2] * x + e[6] * y + e[10] * z + e[14], e[3] * x + e[7] * y + e[11] * z + e[15]];
  };
  const screen = point => [(point[0] / point[3] + 1) * width / 2, (1 - point[1] / point[3]) * height / 2];
  const visible = node => { for (let current = node; current; current = current.parent) if (!current.visible) return false; return true; };
  function clipPolygon(polygon) {
    for (const plane of planes) {
      if (polygon.every(point => distanceToPlane(point, plane) >= 0)) continue;
      const clipped = [];
      for (let i = 0; i < polygon.length; i++) {
        const a = polygon[i], b = polygon[(i + 1) % polygon.length], da = distanceToPlane(a, plane), db = distanceToPlane(b, plane);
        if (da >= 0) clipped.push(a);
        if ((da >= 0) !== (db >= 0)) clipped.push(between(a, b, da / (da - db)));
      }
      polygon = clipped;
      if (polygon.length < 3) return [];
    }
    return polygon.filter(point => point.every(Number.isFinite) && point[3] > 1e-9);
  }
  function drawSegment(a, b, color, opacity = 1, lineWidth = 1.4) {
    let start = clipPoint(a, viewProjection), end = clipPoint(b, viewProjection);
    for (const plane of planes) {
      const da = distanceToPlane(start, plane), db = distanceToPlane(end, plane);
      if (da < 0 && db < 0) return;
      if ((da >= 0) !== (db >= 0)) {
        const intersection = between(start, end, da / (da - db));
        if (da < 0) start = intersection; else end = intersection;
      }
    }
    if (start[3] <= 1e-9 || end[3] <= 1e-9) return;
    const p = screen(start), q = screen(end);
    if (![...p, ...q].every(Number.isFinite)) return;
    context.globalAlpha = opacity; context.strokeStyle = color; context.lineWidth = lineWidth;
    context.beginPath(); context.moveTo(...p); context.lineTo(...q); context.stroke();
  }
  function drawLines(node, grid = false) {
    const position = node.geometry?.attributes?.position, colors = node.geometry?.attributes?.color;
    if (!position) return;
    const material = Array.isArray(node.material) ? node.material[0] : node.material;
    if (material?.visible === false) return;
    const count = Math.min(position.count, 4096);
    for (let i = 0; i + 1 < count; i += 2) {
      aWorld.fromBufferAttribute(position, i).applyMatrix4(node.matrixWorld);
      bWorld.fromBufferAttribute(position, i + 1).applyMatrix4(node.matrixWorld);
      if (material?.vertexColors && colors) tint.fromBufferAttribute(colors, i); else tint.copy(material?.color || baseTint.setHex(0x68d8df));
      drawSegment(aWorld, bWorld, tint.getStyle(), material?.opacity ?? 1, grid ? 0.7 : 1.8);
    }
  }
  function drawPoints(node) {
    const position = node.geometry?.attributes?.position, colors = node.geometry?.attributes?.color;
    if (!position) return;
    for (let i = 0; i < Math.min(position.count, 1024); i++) {
      vertex.fromBufferAttribute(position, i).applyMatrix4(node.matrixWorld);
      const point = clipPoint(vertex, viewProjection);
      if (point[3] <= 1e-9 || planes.some(plane => distanceToPlane(point, plane) < 0)) continue;
      if (colors) tint.fromBufferAttribute(colors, i); else tint.setHex(COLORS.inferred);
      const p = screen(point);
      context.globalAlpha = 1; context.fillStyle = tint.getStyle();
      context.beginPath(); context.arc(p[0], p[1], 2.6, 0, Math.PI * 2); context.fill();
    }
  }
  function setAvatar(nextAvatar) {
    avatar = nextAvatar; meshPlans = []; boneEdges = []; boneStates = {};
    summary = { totalFaces: 0, totalVertices: 0, selectedFaces: 0, selectedVertices: 0, unsupportedMeshes: 0, limited: false };
    if (!avatar) return;
    const ranges = [];
    avatar.scene.traverse(mesh => {
      const geometry = mesh.geometry, position = geometry?.attributes?.position;
      if (!mesh.isMesh || !position) return;
      if (mesh.isInstancedMesh || typeof mesh.getVertexPosition !== 'function') { summary.unsupportedMeshes++; return; }
      const count = geometry.index?.count ?? position.count, start = Math.max(0, geometry.drawRange.start || 0);
      const end = Math.min(count, start + geometry.drawRange.count);
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      const groups = Array.isArray(mesh.material) ? geometry.groups : [{ start: 0, count, materialIndex: 0 }];
      summary.totalVertices += position.count;
      for (const group of groups) {
        const material = materials[group.materialIndex || 0], first = Math.max(start, group.start), last = Math.min(end, group.start + group.count);
        const faceCount = Math.max(0, Math.floor((last - first) / 3));
        if (!material || !faceCount) continue;
        ranges.push({ mesh, material, first, faceCount }); summary.totalFaces += faceCount;
      }
    });
    // Sample across the entire model, not only its first mesh. At most three
    // distinct vertices per selected triangle guarantees the vertex bound too.
    const stride = Math.max(1, Math.ceil(summary.totalFaces / CPU_RENDER_LIMITS.faces)), plans = new Map();
    let offset = 0;
    for (const range of ranges) {
      let plan = plans.get(range.mesh);
      if (!plan) { plan = { mesh: range.mesh, faces: [], vertices: new Map() }; plans.set(range.mesh, plan); }
      for (let i = (stride - offset % stride) % stride; i < range.faceCount; i += stride) {
        const first = range.first + i * 3, index = range.mesh.geometry.index;
        const ids = [0, 1, 2].map(k => index ? index.getX(first + k) : first + k);
        if (ids.some(id => !Number.isInteger(id) || id < 0 || id >= range.mesh.geometry.attributes.position.count)) continue;
        const added = new Set(ids.filter(id => !plan.vertices.has(id))).size;
        if (summary.selectedFaces >= CPU_RENDER_LIMITS.faces || summary.selectedVertices + added > CPU_RENDER_LIMITS.vertices) continue;
        for (const id of ids) if (!plan.vertices.has(id)) plan.vertices.set(id, { world: new THREE.Vector3(), clip: null });
        plan.faces.push({ ids, material: range.material }); summary.selectedFaces++; summary.selectedVertices += added;
      }
      offset += range.faceCount;
    }
    meshPlans = Array.from(plans.values()).filter(plan => plan.faces.length);
    const boneNames = new Map();
    for (const name of Object.keys(avatar.humanoid.normalizedHumanBones)) {
      const node = avatar.humanoid.getRawBoneNode(name); if (node) boneNames.set(node, name);
    }
    for (const [node, name] of boneNames) {
      let parent = node.parent;
      while (parent && !boneNames.has(parent)) parent = parent.parent;
      if (parent) boneEdges.push({ a: parent, b: node, name: boneNames.get(parent), childName: name });
    }
    summary.limited = summary.selectedFaces < summary.totalFaces || summary.unsupportedMeshes > 0;
  }
  function updateMeshVertices(plan, camera = null) {
    if (camera) clipMatrix.multiplyMatrices(viewProjection, plan.mesh.matrixWorld);
    for (const [id, record] of plan.vertices) {
      // This method is overridden by SkinnedMesh to apply its actual skin.
      plan.mesh.getVertexPosition(id, vertex);
      if (camera) record.clip = clipPoint(vertex, clipMatrix);
      record.world.copy(vertex).applyMatrix4(plan.mesh.matrixWorld);
    }
  }
  function render(scene, camera) {
    scene.updateMatrixWorld(true); camera.updateMatrixWorld(true);
    viewProjection.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0); context.globalAlpha = 1;
    context.fillStyle = scene.background?.isColor ? scene.background.getStyle() : '#101b24'; context.fillRect(0, 0, width, height);
    const lineObjects = [], pointObjects = [];
    scene.traverseVisible(node => {
      if (node.isLineSegments) { if (node.isGridHelper || node.type === 'GridHelper') drawLines(node, true); else lineObjects.push(node); }
      if (node.isPoints) pointObjects.push(node);
    });
    const faces = [];
    for (const plan of meshPlans) {
      if (!visible(plan.mesh)) continue;
      updateMeshVertices(plan, camera);
      const colors = plan.mesh.geometry.attributes.color;
      for (const face of plan.faces) {
        const material = face.material;
        if (material.visible === false || material.opacity <= 0) continue;
        const vertices = face.ids.map(id => plan.vertices.get(id));
        if (vertices.some(record => !record.clip.every(Number.isFinite))) continue;
        const polygon = clipPolygon(vertices.map(record => record.clip));
        if (polygon.length < 3) continue;
        edgeA.subVectors(vertices[1].world, vertices[0].world); edgeB.subVectors(vertices[2].world, vertices[0].world);
        normal.crossVectors(edgeA, edgeB).normalize();
        // Flat material/vertex colors and a simple two-sided light. MToon,
        // texture sampling, shadows, and per-pixel transparency are not emulated.
        tint.copy(material.color?.isColor ? material.color : baseTint.setHex(0xc0ced8));
        if (material.vertexColors && colors) {
          attributeTint.setRGB(0, 0, 0);
          for (const id of face.ids) { baseTint.fromBufferAttribute(colors, id); attributeTint.add(baseTint); }
          tint.multiply(attributeTint.multiplyScalar(1 / 3));
        }
        tint.multiplyScalar(0.48 + 0.52 * Math.abs(normal.dot(lightDirection)));
        const depth = polygon.reduce((sum, point) => sum + point[2] / point[3], 0) / polygon.length;
        faces.push({ polygon, depth, color: tint.getStyle(), opacity: Math.min(1, Math.max(0, material.opacity ?? 1)) });
      }
    }
    // Painter ordering is intentionally approximate, unlike a GPU depth buffer.
    faces.sort((a, b) => b.depth - a.depth);
    for (const face of faces) {
      const points = face.polygon.map(screen);
      context.globalAlpha = face.opacity; context.fillStyle = face.color;
      context.beginPath(); context.moveTo(...points[0]);
      for (let i = 1; i < points.length; i++) context.lineTo(...points[i]);
      context.closePath(); context.fill();
    }
    if (avatar && visible(avatar.scene)) for (const bone of boneEdges) {
      bone.a.getWorldPosition(aWorld); bone.b.getWorldPosition(bWorld);
      const state = boneStates[bone.name], color = state?.valid ? colorFor(demo ? 'demo' : state.source) : 0x8b9aa8;
      tint.setHex(color); drawSegment(aWorld, bWorld, tint.getStyle(), state?.valid ? 0.85 : 0.30, state?.valid ? 1.5 : 1);
    }
    lineObjects.forEach(node => drawLines(node)); pointObjects.forEach(drawPoints); context.globalAlpha = 1;
  }
  return {
    domElement: canvas, isCanvas2DRenderer: true,
    setPixelRatio(value) { pixelRatio = Math.min(1.5, Math.max(1, value)); },
    setSize(w, h) { width = w; height = h; canvas.width = Math.round(w * pixelRatio); canvas.height = Math.round(h * pixelRatio); },
    setAvatar, setBoneStates(states, isDemo) { boneStates = states || {}; demo = Boolean(isDemo); },
    getSummary: () => ({ ...summary, limits: CPU_RENDER_LIMITS }),
    getAvatarBounds() {
      const box = new THREE.Box3();
      avatar?.scene.updateWorldMatrix(true, true);
      for (const plan of meshPlans) { updateMeshVertices(plan); for (const record of plan.vertices.values()) box.expandByPoint(record.world); }
      for (const bone of boneEdges) { box.expandByPoint(bone.a.getWorldPosition(aWorld)); box.expandByPoint(bone.b.getWorldPosition(bWorld)); }
      return box;
    },
    render,
    dispose() { setAvatar(null); canvas.width = 1; canvas.height = 1; }
  };
}

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
  let renderer, rendererFallbackReason = null;
  try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' }); }
  catch (error) { rendererFallbackReason = error?.message || String(error); renderer = createCanvas2DRenderer(); }
  const rendererBackend = renderer.isCanvas2DRenderer ? 'canvas2d' : 'webgl';
  renderer.setPixelRatio(Math.min(globalThis.devicePixelRatio || 1, 1.75));
  if (!renderer.isCanvas2DRenderer) renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.dataset.renderer = rendererBackend;
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
  let frameId = 0, lastTick = performance.now(), metricStart = lastTick, renderFrames = 0, renderMsTotal = 0, lastRenderTick = -Infinity;
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
  const status = () => ({ ready: !disposed, loading, hasVRM: Boolean(vrm), fileName, sourceMode, displayMode, loadError,
    rendererBackend, rendererFallbackReason, renderStats: renderer.getSummary?.() || null, ...poseSummary() });
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
    let box;
    if (avatarGroup.visible && renderer.isCanvas2DRenderer) box = renderer.getAvatarBounds();
    else {
      object.traverse(node => { if (node.isSkinnedMesh) node.computeBoundingBox(); });
      box = new THREE.Box3().setFromObject(object);
    }
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
    renderer.setBoneStates?.(lastRetarget?.bones, sourceMode === 'demo');
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
      renderer.setAvatar?.(vrm);
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
    renderer.setAvatar?.(null);
    updatePose(pose, { sourceMode }); notify(true);
  }
  function setDisplayMode(mode) {
    displayMode = mode === 'avatar' && vrm ? 'avatar' : 'skeleton';
    visibility(); resetView(); notify();
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
    if (renderer.isCanvas2DRenderer && now - lastRenderTick < 1000 / CPU_RENDER_LIMITS.maxFps) { frameId = requestAnimationFrame(tick); return; }
    lastRenderTick = now;
    const dt = Math.min(Math.max((now - lastTick) / 1000, 0), 0.05); lastTick = now;
    const start = performance.now();
    controls.update();
    vrm?.update(dt);
    renderer.render(scene, camera);
    renderMsTotal += performance.now() - start; renderFrames++;
    if (now - metricStart >= 1000) {
      onRenderMetrics({ fps: renderFrames * 1000 / (now - metricStart), renderMs: renderMsTotal / renderFrames, sourceMode, rendererBackend,
        measurement: renderer.isCanvas2DRenderer ? 'cpu-skin-project-and-canvas2d' : 'cpu-submit-and-vrm-update' });
      metricStart = now; renderFrames = 0; renderMsTotal = 0;
    }
    frameId = requestAnimationFrame(tick);
  }
  frameId = requestAnimationFrame(tick);
  notify(true);
  return {
    updatePose, loadFile, unload, resetView, setDisplayMode, getStatus: status,
    dispose() {
      if (disposed) return;
      disposed = true; generation++; cancelAnimationFrame(frameId); observer.disconnect(); controls.dispose();
      disposeObject(avatarGroup); disposeObject(skeletonGroup); disposeObject(grid);
      renderer.dispose(); renderer.domElement.remove();
    }
  };
}
