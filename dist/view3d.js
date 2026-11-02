import { SKELETON_EDGES } from './geometry.js';

const COLORS = {
  triangulated: '#50dfd2', estimatedMultiView: '#70c3d8', monocular: '#b4a0ff',
  model3d: '#b4a0ff', planarPrior: '#998bb5', predicted: '#ffd281',
  kinematicPrior: '#ffd281', mixed: '#b8c9df', missing: '#62748e',
};
const clamp = (value, minimum, maximum) => Math.min(maximum, Math.max(minimum, value));
const finite3 = (value) => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
const add = (a, b) => a.map((value, index) => value + b[index]);
const sub = (a, b) => a.map((value, index) => value - b[index]);
const scale = (value, factor) => value.map((component) => component * factor);
const validJoint = (joint) => joint?.valid !== false && finite3(joint?.xyz) && joint.confidence > 0;
const isPrediction = (source) => source === 'predicted' || source === 'kinematicPrior';

/** A dependency-free orthographic pose inspector. Input coordinates are never mutated. */
export class PoseView {
  constructor(canvas) {
    if (!canvas?.getContext) throw new Error('PoseView requires a canvas.');
    this.canvas = canvas;
    this.context = canvas.getContext('2d');
    if (!this.context) throw new Error('3D骨格表示のCanvasを初期化できません。');
    this.yaw = -0.42;
    this.pitch = 0.17;
    this.spanM = 3.3;
    this.target = null;
    this.floorCenter = [0, 0, 0];
    this.coordinateFrame = 'world';
    this.result = null;
    this.cameras = null;
    this.options = {};
    this.emptyText = '3D姿勢はまだ推論されていません';
    this.pointer = null;
    this.listeners = [];
    this.canvas.style.touchAction = 'none';
    if (!canvas.hasAttribute('tabindex')) canvas.tabIndex = 0;
    this.bind('pointerdown', (event) => {
      if (event.button !== 0 && event.pointerType !== 'touch') return;
      this.pointer = { id: event.pointerId, x: event.clientX, y: event.clientY };
      this.canvas.setPointerCapture?.(event.pointerId);
      this.canvas.style.cursor = 'grabbing';
      event.preventDefault();
    });
    this.bind('pointermove', (event) => {
      if (!this.pointer || this.pointer.id !== event.pointerId) return;
      this.yaw += (event.clientX - this.pointer.x) * 0.008;
      this.pitch = clamp(this.pitch + (event.clientY - this.pointer.y) * 0.006, -1.25, 1.25);
      this.pointer.x = event.clientX; this.pointer.y = event.clientY;
      this.draw();
    });
    const release = (event) => {
      if (this.pointer?.id !== event.pointerId) return;
      this.pointer = null;
      this.canvas.style.cursor = 'grab';
    };
    this.bind('pointerup', release);
    this.bind('pointercancel', release);
    this.bind('lostpointercapture', release);
    this.bind('wheel', (event) => {
      event.preventDefault();
      this.spanM = clamp(this.spanM * Math.exp(clamp(event.deltaY, -300, 300) * 0.0015), 0.25, 40);
      this.draw();
    }, { passive: false });
    this.bind('dblclick', () => this.resetView());
    this.bind('keydown', (event) => {
      if (event.key === 'ArrowLeft') this.yaw -= 0.12;
      else if (event.key === 'ArrowRight') this.yaw += 0.12;
      else if (event.key === 'ArrowUp') this.pitch = clamp(this.pitch - 0.1, -1.25, 1.25);
      else if (event.key === 'ArrowDown') this.pitch = clamp(this.pitch + 0.1, -1.25, 1.25);
      else if (event.key === '+' || event.key === '=') this.spanM = Math.max(0.25, this.spanM / 1.15);
      else if (event.key === '-') this.spanM = Math.min(40, this.spanM * 1.15);
      else if (event.key === 'Home') { event.preventDefault(); this.resetView(); return; }
      else return;
      event.preventDefault(); this.draw();
    });
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.draw());
      this.resizeObserver.observe(canvas);
    }
    this.draw();
  }

  bind(event, listener, options) {
    this.canvas.addEventListener(event, listener, options);
    this.listeners.push([event, listener, options]);
  }

  dispose() {
    this.resizeObserver?.disconnect();
    for (const [event, listener, options] of this.listeners) this.canvas.removeEventListener(event, listener, options);
    this.listeners = [];
  }

  clear(text = '3D姿勢はまだ推論されていません') {
    this.result = null;
    this.cameras = null;
    this.target = null;
    this.emptyText = text;
    this.draw();
  }

  render(result, cameras = null, options = {}) {
    const frame = options.coordinateFrame || 'world';
    if (frame !== this.coordinateFrame) this.target = null;
    this.coordinateFrame = frame;
    this.result = result;
    this.cameras = cameras;
    this.options = options;
    this.emptyText = result ? '有効な3D点がありません（欠測・未校正）' : '3D姿勢はまだ推論されていません';
    this.draw();
  }

  resetView() {
    this.yaw = -0.42; this.pitch = 0.17; this.target = null; this.spanM = 3.3;
    this.draw();
  }

  displayPoint(xyz) {
    // A camera's down-positive y is inverted for display only.
    return this.coordinateFrame === 'camera' ? [xyz[0], -xyz[1], xyz[2]] : [...xyz];
  }

  orient(vector) {
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const x = cy * vector[0] - sy * vector[2];
    const z = sy * vector[0] + cy * vector[2];
    return [x, cp * vector[1] - sp * z, sp * vector[1] + cp * z];
  }

  project(xyz) {
    const view = this.orient(sub(this.displayPoint(xyz), this.target || [0, 0, 0]));
    return [this.width / 2 + view[0] * this.pixelsPerM, this.height * 0.48 - view[1] * this.pixelsPerM, view[2]];
  }

  initializeView(joints) {
    const body = joints.filter((joint) => joint.id < 17);
    const selected = body.length >= 3 ? body : joints;
    const points = selected.map((joint) => this.displayPoint(joint.xyz));
    const minimum = [0, 1, 2].map((axis) => Math.min(...points.map((point) => point[axis])));
    const maximum = [0, 1, 2].map((axis) => Math.max(...points.map((point) => point[axis])));
    const midpoint = scale(add(minimum, maximum), 0.5);
    this.target = finite3(this.result?.root?.xyz) ? this.displayPoint(this.result.root.xyz) : midpoint;
    if (this.coordinateFrame === 'world') {
      this.target[1] = Math.max(0.65, midpoint[1]);
      this.floorCenter = [Math.round(this.target[0]), 0, Math.round(this.target[2])];
    }
    const vertical = maximum[1] - minimum[1], horizontal = Math.max(maximum[0] - minimum[0], maximum[2] - minimum[2]);
    this.spanM = clamp(Math.max(2.7, vertical * 1.6, horizontal * 1.8 * this.height / this.width), 0.5, 25);
  }

  line(a, b, color, width = 1, alpha = 1, dash = []) {
    const ctx = this.context;
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.globalAlpha = alpha; ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    ctx.globalAlpha = 1; ctx.setLineDash([]);
  }

  worldLine(a, b, color, width = 1, alpha = 1, dash = []) {
    this.line(this.project(a), this.project(b), color, width, alpha, dash);
  }

  text(text, x, y, color = '#95a9c2', font = '11px system-ui', maxWidth) {
    const ctx = this.context;
    ctx.fillStyle = color; ctx.font = font; ctx.textAlign = 'left';
    let display = String(text);
    if (maxWidth) {
      while (display.length > 1 && ctx.measureText(display).width > maxWidth) display = `${display.slice(0, -2)}…`;
    }
    ctx.fillText(display, x, y);
  }

  drawFloor() {
    if (this.coordinateFrame !== 'world') return;
    const center = this.floorCenter, half = Math.max(3, Math.ceil(this.spanM * 1.3));
    const step = half > 12 ? 2 : 1;
    for (let offset = -half; offset <= half; offset += step) {
      const x = center[0] + offset, z = center[2] + offset;
      this.worldLine([x, 0, center[2] - half], [x, 0, center[2] + half], x === 0 ? '#477591' : '#27394c', x === 0 ? 1.2 : 0.7, 0.7);
      this.worldLine([center[0] - half, 0, z], [center[0] + half, 0, z], z === 0 ? '#825c64' : '#27394c', z === 0 ? 1.2 : 0.7, 0.7);
    }
    const anchor = this.project(center);
    this.text(`床 y = 0 · 目盛 ${step} m`, clamp(anchor[0] + 8, 12, this.width - 180), clamp(anchor[1] + 20, 60, this.height - 42), '#657d98', '10px system-ui');
  }

  drawCameras() {
    if (this.coordinateFrame !== 'world' || !this.cameras) return;
    const cameras = this.cameras instanceof Map ? [...this.cameras.values()] : Array.isArray(this.cameras) ? this.cameras : Object.values(this.cameras);
    for (const camera of cameras) {
      const transform = camera?.worldFromCamera;
      if (!finite3(transform?.translation) || !Array.isArray(transform?.rotation) || transform.rotation.length !== 9) continue;
      const r = transform.rotation, origin = transform.translation;
      const right = [r[0], r[3], r[6]], down = [r[1], r[4], r[7]], forward = [r[2], r[5], r[8]];
      const center = add(origin, scale(forward, 0.24));
      const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([x, y]) => add(center, add(scale(right, x * 0.13), scale(down, y * 0.09))));
      const color = camera.calibrated ? '#6689ac' : '#8c7b66';
      for (let i = 0; i < 4; i++) {
        this.worldLine(origin, corners[i], color, 1, 0.8);
        this.worldLine(corners[i], corners[(i + 1) % 4], color, 1, 0.8);
      }
      const screen = this.project(origin);
      this.text(camera.id || 'camera', screen[0] + 6, screen[1] - 8, color, '10px system-ui', 100);
    }
  }

  drawSkeleton(joints) {
    const byId = new Map(joints.map((joint) => [joint.id, joint]));
    const edges = SKELETON_EDGES.filter(([a, b]) => byId.has(a) && byId.has(b)).map(([a, b]) => {
      const first = byId.get(a), second = byId.get(b);
      const pa = this.project(first.xyz), pb = this.project(second.xyz);
      return { first, second, pa, pb, depth: (pa[2] + pb[2]) / 2 };
    }).sort((a, b) => a.depth - b.depth);
    for (const { first, second, pa, pb } of edges) {
      const predicted = isPrediction(first.source) || isPrediction(second.source);
      const source = predicted ? 'predicted' : first.source === 'triangulated' && second.source === 'triangulated' ? 'triangulated' : first.source === 'triangulated' ? second.source : first.source;
      this.line(pa, pb, COLORS[source] || COLORS.mixed, first.id >= 91 ? 1.2 : 2.2, clamp(Math.min(first.confidence, second.confidence) * 0.75 + 0.3, 0.3, 1), predicted ? [3, 3] : []);
    }
    const ordered = joints.map((joint) => ({ joint, screen: this.project(joint.xyz) })).sort((a, b) => a.screen[2] - b.screen[2]);
    const ctx = this.context;
    for (const { joint, screen } of ordered) {
      const face = joint.id >= 23 && joint.id <= 90;
      ctx.beginPath(); ctx.arc(screen[0], screen[1], face ? 0.7 : joint.id >= 91 ? 1.5 : 2.6, 0, Math.PI * 2);
      ctx.fillStyle = COLORS[joint.source] || COLORS.mixed;
      ctx.globalAlpha = clamp(joint.confidence * 0.8 + 0.2, 0.2, 1) * (face ? 0.5 : 1);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (this.coordinateFrame === 'world' && this.result?.root?.valid && finite3(this.result.root.xyz)) {
      const root = this.result.root.xyz;
      this.worldLine([root[0], 0, root[2]], root, '#8194ad', 1, 0.5, [2, 4]);
      const floor = this.project([root[0], 0, root[2]]);
      this.line([floor[0] - 4, floor[1]], [floor[0] + 4, floor[1]], '#a8bad0');
      this.line([floor[0], floor[1] - 4], [floor[0], floor[1] + 4], '#a8bad0');
    }
  }

  drawFeet() {
    for (const foot of Object.values(this.result?.feet || {})) {
      if (!foot?.valid || !finite3(foot.positionXYZm) || !finite3(foot.axes?.forward)) continue;
      const from = this.project(foot.positionXYZm), to = this.project(add(foot.positionXYZm, scale(foot.axes.forward, 0.2)));
      const predicted = foot.inputSources?.some(isPrediction);
      const color = predicted ? COLORS.predicted : '#c3d9ed';
      this.line(from, to, color, 1.5, 0.8, predicted ? [3, 2] : []);
      const angle = Math.atan2(to[1] - from[1], to[0] - from[0]);
      this.line(to, [to[0] - 5 * Math.cos(angle - 0.5), to[1] - 5 * Math.sin(angle - 0.5)], color, 1.5);
      this.line(to, [to[0] - 5 * Math.cos(angle + 0.5), to[1] - 5 * Math.sin(angle + 0.5)], color, 1.5);
      const ctx = this.context;
      ctx.strokeStyle = color; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(from[0], from[1] - 3); ctx.lineTo(from[0] + 3, from[1]);
      ctx.lineTo(from[0], from[1] + 3); ctx.lineTo(from[0] - 3, from[1]); ctx.closePath(); ctx.stroke();
    }
  }

  drawAxes() {
    const origin = [this.width - 45, 43];
    for (const [label, direction, color] of [['X', [1, 0, 0], '#d99595'], ['Y', [0, 1, 0], '#94c89c'], ['Z', [0, 0, 1], '#8eaed8']]) {
      const rotated = this.orient(this.displayPoint(direction));
      const end = [origin[0] + rotated[0] * 24, origin[1] - rotated[1] * 24];
      this.line(origin, end, color, 1.3, 0.9);
      this.text(label, end[0] + 3, end[1] + 3, color, '9px ui-monospace, monospace');
    }
  }

  draw() {
    const rectangle = this.canvas.getBoundingClientRect();
    this.width = rectangle.width || this.width || 640;
    this.height = rectangle.height || this.height || 310;
    const ratio = clamp(globalThis.devicePixelRatio || 1, 1, 2);
    const pixelWidth = Math.round(this.width * ratio), pixelHeight = Math.round(this.height * ratio);
    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) { this.canvas.width = pixelWidth; this.canvas.height = pixelHeight; }
    const ctx = this.context;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.globalAlpha = 1; ctx.setLineDash([]);
    const background = ctx.createRadialGradient(this.width * 0.48, this.height * 0.5, 5, this.width * 0.5, this.height * 0.5, Math.max(this.width, this.height) * 0.7);
    background.addColorStop(0, '#172334'); background.addColorStop(1, '#0b111b');
    ctx.fillStyle = background; ctx.fillRect(0, 0, this.width, this.height);
    const joints = (this.result?.joints || []).filter(validJoint);
    if (!joints.length) {
      ctx.fillStyle = '#8193ab'; ctx.font = '12px system-ui'; ctx.textAlign = 'center';
      String(this.emptyText).split('\n').forEach((line, index) => ctx.fillText(line, this.width / 2, this.height / 2 + index * 20, Math.max(50, this.width - 40)));
      if (this.options.statusText && this.result) { ctx.font = '10px system-ui'; ctx.fillStyle = '#63788f'; ctx.fillText(this.options.statusText, this.width / 2, this.height / 2 + 30, Math.max(50, this.width - 40)); }
      return;
    }
    if (!this.target) this.initializeView(joints);
    // Only the explicitly relative view follows the moving root. World/camera
    // views retain the initial target, making translation visible against the axes.
    if (this.coordinateFrame === 'rootRelative' && finite3(this.result.root?.xyz)) this.target = this.displayPoint(this.result.root.xyz);
    this.pixelsPerM = this.height / this.spanM;
    this.drawFloor(); this.drawCameras(); this.drawSkeleton(joints); this.drawFeet(); this.drawAxes();
    const title = this.coordinateFrame === 'world' ? '部屋座標 · 床固定' : this.coordinateFrame === 'camera' ? 'カメラ座標 · 床未校正' : '骨盤に追従する相対表示';
    this.text(title, 14, 22, '#aac0d7', '10px system-ui', this.width - 110);
    if (this.options.statusText) this.text(this.options.statusText, 14, 39, '#7188a2', '10px system-ui', this.width - 110);
    const root = this.result.root;
    if (root?.valid && finite3(root.xyz)) {
      this.text(`骨盤 X ${root.xyz[0].toFixed(2)} / Y ${root.xyz[1].toFixed(2)} / Z ${root.xyz[2].toFixed(2)} m`, 14, this.height - 15, '#c2d2e5', '10px ui-monospace, monospace', this.width - 28);
    } else {
      this.text(`3D点 ${joints.length} / 骨盤は欠測`, 14, this.height - 15, '#8193ab', '10px system-ui');
    }
  }
}
