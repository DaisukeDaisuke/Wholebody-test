/** Original procedural VRM fixture. No downloaded mesh, texture, or user data. */
export const SAMPLE_VRM_INFO = Object.freeze({ name: 'Wholebody generated sample', filename: 'wholebody-generated-sample.vrm', license: 'Unlicense', synthetic: true, trackingAccuracyEvidence: false });
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const mul = (a, n) => a.map(v => v * n);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = a => mul(a, 1 / Math.max(Math.hypot(...a), 1e-9));
const align4 = n => Math.ceil(n / 4) * 4;

export function createSampleVRM() {
  const nodes = [{ name: 'Generated sample root', children: [] }], humanBones = {}, rest = {}, names = [];
  function bone(name, parent, translation) {
    const index = nodes.length, parentIndex = parent ? humanBones[parent].node : 0;
    nodes.push({ name, translation, children: [] }); nodes[parentIndex].children.push(index);
    humanBones[name] = { node: index }; rest[name] = add(parent ? rest[parent] : [0, 0, 0], translation); names.push(name);
  }
  bone('hips', null, [0, 0.95, 0]); bone('spine', 'hips', [0, 0.18, 0]);
  bone('chest', 'spine', [0, 0.20, 0]); bone('upperChest', 'chest', [0, 0.12, 0]);
  bone('neck', 'upperChest', [0, 0.09, 0]); bone('head', 'neck', [0, 0.14, 0]);
  for (const [side, sign] of [['left', 1], ['right', -1]]) {
    bone(side + 'Shoulder', 'upperChest', [sign * 0.06, 0.02, 0]);
    bone(side + 'UpperArm', side + 'Shoulder', [sign * 0.14, 0, 0]);
    bone(side + 'LowerArm', side + 'UpperArm', [sign * 0.26, 0, 0]);
    bone(side + 'Hand', side + 'LowerArm', [sign * 0.24, 0, 0]);
    bone(side + 'UpperLeg', 'hips', [sign * 0.10, -0.04, 0]);
    bone(side + 'LowerLeg', side + 'UpperLeg', [0, -0.42, 0]);
    bone(side + 'Foot', side + 'LowerLeg', [0, -0.42, 0]);
    bone(side + 'Toes', side + 'Foot', [0, -0.045, 0.16]);
    for (const [finger, spread] of [['Thumb', 0.045], ['Index', 0.024], ['Middle', 0.003], ['Ring', -0.018], ['Little', -0.038]]) {
      const segments = finger === 'Thumb' ? ['Metacarpal', 'Proximal', 'Distal'] : ['Proximal', 'Intermediate', 'Distal'];
      let parent = side + 'Hand';
      for (let part = 0; part < segments.length; part++) {
        const name = side + finger + segments[part];
        bone(name, parent, [sign * (part === 0 ? (finger === 'Thumb' ? 0.025 : 0.06) : part === 1 ? 0.025 : 0.019), 0, part === 0 ? spread : 0]);
        parent = name;
      }
    }
  }
  const position = [], normal = [], colors = [], joints = [], weights = [], indices = [];
  const corners = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]];
  const faces = [
    [[0, 3, 2, 1], [0, 0, -1]], [[4, 5, 6, 7], [0, 0, 1]],
    [[0, 4, 7, 3], [-1, 0, 0]], [[1, 2, 6, 5], [1, 0, 0]],
    [[3, 7, 6, 2], [0, 1, 0]], [[0, 1, 5, 4], [0, -1, 0]]
  ];
  function box(start, end, width, depth, joint, tint) {
    const y = unit(sub(end, start)), x = unit(cross(y, Math.abs(y[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0])), z = cross(x, y);
    const center = mul(add(start, end), 0.5), half = [width / 2, Math.hypot(...sub(end, start)) / 2, depth / 2];
    const transform = v => add(add(mul(x, v[0]), mul(y, v[1])), mul(z, v[2]));
    for (const [face, faceNormal] of faces) {
      const base = position.length / 3, n = transform(faceNormal);
      for (const corner of face) {
        const p = add(center, transform(corners[corner].map((v, i) => v * half[i])));
        position.push(...p); normal.push(...n); colors.push(...tint, 1); joints.push(joint, 0, 0, 0); weights.push(1, 0, 0, 0);
      }
      indices.push(base, base + 1, base + 2, base, base + 2, base + 3);
    }
  }
  for (const [joint, name] of names.entries()) {
    const node = nodes[humanBones[name].node], start = rest[name], sign = name.startsWith('right') ? -1 : 1;
    const firstChild = node.children[0], childName = firstChild === undefined ? null : nodes[firstChild].name;
    let end = childName ? rest[childName] : add(start, [sign * 0.017, 0, 0]), width = 0.054, depth = 0.054;
    if (['hips', 'spine', 'chest', 'upperChest'].includes(name)) { width = name === 'hips' ? 0.22 : 0.24; depth = 0.13; }
    else if (name === 'neck') { width = 0.065; depth = 0.065; }
    else if (name === 'head') { end = add(start, [0, 0.17, 0]); width = 0.17; depth = 0.14; }
    else if (/Leg$/.test(name)) { width = 0.075; depth = 0.075; }
    else if (/Foot$/.test(name)) { width = 0.085; depth = 0.055; }
    else if (/Toes$/.test(name)) { end = add(start, [0, 0, 0.055]); width = 0.075; depth = 0.04; }
    else if (/Hand$/.test(name)) { end = add(start, [sign * 0.065, 0, 0]); width = 0.075; depth = 0.035; }
    else if (/Thumb|Index|Middle|Ring|Little/.test(name)) { width = 0.011; depth = 0.011; }
    const tint = name === 'head' ? [0.90, 0.78, 0.58] : name.startsWith('left') ? [0.25, 0.72, 0.78] : name.startsWith('right') ? [0.58, 0.82, 0.38] : [0.72, 0.79, 0.86];
    box(start, end, width, depth, joint, tint);
  }
  const bufferViews = [], accessors = [], chunks = [];
  let binaryLength = 0;
  function accessor(array, type, componentType, components, target, limits) {
    const byteOffset = align4(binaryLength), bufferView = bufferViews.length;
    bufferViews.push({ buffer: 0, byteOffset, byteLength: array.byteLength, ...(target ? { target } : {}) });
    chunks.push({ byteOffset, bytes: new Uint8Array(array.buffer, array.byteOffset, array.byteLength) });
    binaryLength = byteOffset + array.byteLength;
    const index = accessors.length;
    accessors.push({ bufferView, componentType, count: array.length / components, type, ...limits });
    return index;
  }
  const minima = [Infinity, Infinity, Infinity], maxima = [-Infinity, -Infinity, -Infinity];
  position.forEach((v, i) => { minima[i % 3] = Math.min(minima[i % 3], v); maxima[i % 3] = Math.max(maxima[i % 3], v); });
  const attributes = {
    POSITION: accessor(new Float32Array(position), 'VEC3', 5126, 3, 34962, { min: minima, max: maxima }),
    NORMAL: accessor(new Float32Array(normal), 'VEC3', 5126, 3, 34962),
    COLOR_0: accessor(new Float32Array(colors), 'VEC4', 5126, 4, 34962),
    JOINTS_0: accessor(new Uint16Array(joints), 'VEC4', 5123, 4, 34962),
    WEIGHTS_0: accessor(new Float32Array(weights), 'VEC4', 5126, 4, 34962)
  };
  const indexAccessor = accessor(new Uint16Array(indices), 'SCALAR', 5123, 1, 34963);
  const inverseBindMatrices = new Float32Array(names.length * 16);
  names.forEach((name, joint) => inverseBindMatrices.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...rest[name].map(v => -v), 1], joint * 16));
  const inverseAccessor = accessor(inverseBindMatrices, 'MAT4', 5126, 16);
  const meshNode = nodes.length;
  nodes.push({ name: 'Generated skinned mesh', mesh: 0, skin: 0 }); nodes[0].children.push(meshNode);
  for (const node of nodes) if (node.children?.length === 0) delete node.children;
  const document = {
    asset: { version: '2.0', generator: 'Wholebody Studio procedural fixture' }, scene: 0, scenes: [{ nodes: [0] }], nodes,
    buffers: [{ byteLength: binaryLength }], bufferViews, accessors,
    meshes: [{ name: 'Original rigid-weighted humanoid primitives', primitives: [{ attributes, indices: indexAccessor, material: 0, mode: 4 }] }],
    materials: [{ name: 'Generated vertex colors', pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 0.8 }, doubleSided: false }],
    skins: [{ joints: names.map(name => humanBones[name].node), skeleton: humanBones.hips.node, inverseBindMatrices: inverseAccessor }],
    extensionsUsed: ['VRMC_vrm'],
    extensions: { VRMC_vrm: {
      specVersion: '1.0', meta: { name: SAMPLE_VRM_INFO.name, version: '1', authors: ['Wholebody Studio'], copyrightInformation: 'Original procedural fixture. Released under the Unlicense.',
        licenseUrl: 'https://vrm.dev/licenses/1.0/', avatarPermission: 'everyone', allowExcessivelyViolentUsage: true, allowExcessivelySexualUsage: true,
        commercialUsage: 'corporation', allowPoliticalOrReligiousUsage: true, allowAntisocialOrHateUsage: true,
        creditNotation: 'unnecessary', allowRedistribution: true, modification: 'allowModificationRedistribution', otherLicenseUrl: 'https://unlicense.org/' },
      humanoid: { humanBones }
    } },
    extras: { ...SAMPLE_VRM_INFO, purpose: 'Display integration verification, never tracking accuracy evidence' }
  };
  const json = new TextEncoder().encode(JSON.stringify(document)), jsonLength = align4(json.length), binLength = align4(binaryLength);
  const result = new ArrayBuffer(12 + 8 + jsonLength + 8 + binLength), view = new DataView(result), bytes = new Uint8Array(result);
  view.setUint32(0, 0x46546c67, true); view.setUint32(4, 2, true); view.setUint32(8, result.byteLength, true);
  view.setUint32(12, jsonLength, true); view.setUint32(16, 0x4e4f534a, true);
  bytes.fill(0x20, 20, 20 + jsonLength); bytes.set(json, 20);
  const binaryOffset = 20 + jsonLength;
  view.setUint32(binaryOffset, binLength, true); view.setUint32(binaryOffset + 4, 0x004e4942, true);
  for (const chunk of chunks) bytes.set(chunk.bytes, binaryOffset + 8 + chunk.byteOffset);
  return result;
}
