/** OSC 1.0 binary encoding; VMT's documented Room/Unity endpoint. */
const finite = (a, n) => Array.isArray(a) && a.length === n && a.every(Number.isFinite);
function oscString(value) {
  if (typeof value !== 'string' || value.includes('\0')) throw new Error('Invalid OSC string.');
  const text = Buffer.from(value, 'utf8'), buffer = Buffer.alloc(Math.ceil((text.length + 1) / 4) * 4);
  text.copy(buffer); return buffer;
}
export function encodeOscMessage(address, types, values) {
  if (!address.startsWith('/') || types.length !== values.length) throw new Error('Invalid OSC message.');
  const body = Buffer.alloc(values.length * 4);
  for (let i = 0; i < types.length; i++) {
    const value = values[i];
    if (types[i] === 'i' && Number.isInteger(value) && value >= -2147483648 && value <= 2147483647) body.writeInt32BE(value, i * 4);
    else if (types[i] === 'f' && Number.isFinite(value) && Math.abs(value) <= 3.4e38) body.writeFloatBE(value, i * 4);
    else throw new Error('OSC type/value mismatch.');
  }
  return Buffer.concat([oscString(address), oscString(',' + types), body]);
}
export function rightHandedToUnity(position, rotation) {
  if (!finite(position, 3) || !finite(rotation, 4)) throw new Error('Invalid tracker transform.');
  // Reflection S=diag(1,1,-1): R_unity=S R_right S, q=(-qx,-qy,qz,qw).
  return { position: [position[0], position[1], -position[2]], rotation: [-rotation[0], -rotation[1], rotation[2], rotation[3]] };
}
export function encodeVmtPose(index, enabled, position = [0, 0, 0], rotation = [0, 0, 0, 1]) {
  if (!Number.isInteger(index) || index < 0 || index > 57 || !finite(position, 3) || !finite(rotation, 4)) throw new Error('Invalid VMT tracker pose.');
  const n = Math.hypot(...rotation);
  if (n < 1e-8) throw new Error('Zero tracker quaternion.');
  return encodeOscMessage('/VMT/Room/Unity', 'iiffffffff', [index, enabled ? 1 : 0, 0, ...position, ...rotation.map(v => v / n)]);
}
