// Distribution-only decoding. The restored bytes must match the published
// runtime; compression never changes the model, execution provider or graph.
export const ORT_WEB_VERSION = '1.31.1';
export const RUNTIME_CACHE_NAME = 'wholebody-app-20261102-v6';
const ORT_SOURCE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_WEB_VERSION}/dist/`;
const WASM_FILE = 'ort-wasm-simd-threaded.asyncify.wasm';
const WASM_BYTES = 28747738;
const WASM_SHA256 = 'ca319e9539cd0f1e291cf56074760121eaf39650f5dbf84b6e52c07f4b2944ec';
const MAX_RUNTIME_BYTES = 64 * 1024 * 1024;
const validHash = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const validSize = value => Number.isSafeInteger(value) && value > 0 && value <= MAX_RUNTIME_BYTES;
const isGzip = bytes => bytes[0] === 0x1f && bytes[1] === 0x8b;
const isWasm = bytes => bytes[0] === 0 && bytes[1] === 0x61 && bytes[2] === 0x73 && bytes[3] === 0x6d;

async function digest(bytes) {
  if (!globalThis.crypto?.subtle) throw new Error('ランタイムのSHA-256を検証できません。HTTPSまたはlocalhostで開いてください。');
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), value => value.toString(16).padStart(2, '0')).join('');
}
async function verify(bytes, size, sha256, label) {
  if (!validSize(size) || !validHash(sha256) || bytes.byteLength !== size || await digest(bytes) !== sha256) throw new Error(`${label}のサイズまたはSHA-256が配布仕様と一致しません。`);
}
async function readBounded(stream, limit, signal) {
  const reader = stream.getReader(), parts = []; let length = 0;
  try {
    while (true) {
      signal?.throwIfAborted();
      const { value, done } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > limit) throw new Error('ランタイムのサイズが配布仕様の上限を超えました。');
      parts.push(value);
    }
  } catch (error) { await reader.cancel().catch(() => {}); throw error; }
  finally { reader.releaseLock(); }
  const bytes = new Uint8Array(length); let offset = 0;
  for (const part of parts) { bytes.set(part, offset); offset += part.byteLength; }
  return bytes;
}

/** Pure byte input for both the browser and Node Actions/CLI. Legacy raw
 * records retain their own recorded version/hash; browser version pinning is
 * done separately by loadWasmBinary(). No fetch or persistent writes here. */
export async function decodeWasmPayload(payload, record) {
  const bytes = payload instanceof Uint8Array ? payload : payload instanceof ArrayBuffer ? new Uint8Array(payload) : null;
  if (!bytes || !record) throw new Error('ランタイムの配布データが不正です。');
  if (!record.encoding) {
    await verify(bytes, record.bytes, record.sha256, 'WASM');
    if (!isWasm(bytes)) throw new Error('WASMのファイル形式が不正です。');
    return bytes;
  }
  if (record.encoding !== 'gzip' || !validSize(record.decodedBytes) || !validHash(record.decodedSha256)) throw new Error('ランタイムの圧縮仕様が不正です。');
  let binary;
  if (isGzip(bytes)) {
    await verify(bytes, record.bytes, record.sha256, '圧縮WASM');
    if (typeof DecompressionStream !== 'function') throw new Error('このブラウザはgzipランタイムの展開に対応していません。');
    binary = await readBounded(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip')), record.decodedBytes);
  } else if (isWasm(bytes)) {
    // A host may set Content-Encoding:gzip on the .gz URL. Fetch then exposes
    // already-decoded bytes: verify the raw hash without claiming a packed hash.
    binary = bytes;
  } else throw new Error('圧縮WASMのファイル形式が不正です。');
  await verify(binary, record.decodedBytes, record.decodedSha256, '復元WASM');
  if (!isWasm(binary)) throw new Error('復元WASMのファイル形式が不正です。');
  return binary;
}

async function runtimeResponse(url, { offline, signal }) {
  signal?.throwIfAborted();
  if (offline) {
    if (!globalThis.caches) throw new Error('閉域モード：保存済みランタイムを取得できません。');
    const cache = await caches.open(RUNTIME_CACHE_NAME), response = await cache.match(url.href);
    if (!response) throw new Error('閉域モード：この版のランタイムが未保存です。オンラインでオフライン準備を行ってください。');
    return response;
  }
  const response = await fetch(url, { signal, credentials: 'same-origin', cache: 'no-cache', referrerPolicy: 'no-referrer' });
  if (!response.ok) throw new Error(`ランタイム取得 HTTP ${response.status} : ${url.pathname}`);
  return response;
}
async function responseBytes(response, limit, signal) {
  if (response.body) return readBounded(response.body, limit, signal);
  const bytes = new Uint8Array(await response.arrayBuffer());
  signal?.throwIfAborted();
  if (bytes.byteLength > limit) throw new Error('ランタイムのサイズが配布仕様の上限を超えました。');
  return bytes;
}

/** Browser production and WebGPU CI share the same origin-local gzip path.
 * record.url is provenance only and is never used as a download fallback. */
export async function loadWasmBinary(baseURL = new URL('./vendor/', import.meta.url), { offline = false, signal } = {}) {
  const base = new URL(baseURL, import.meta.url);
  if (!base.pathname.endsWith('/')) base.pathname += '/';
  if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password || base.search || base.hash
    || (globalThis.location && base.origin !== globalThis.location.origin)) throw new Error('ランタイムはこのアプリと同じ配布元から読み込んでください。');
  const options = { offline, signal };
  const manifestBytes = await responseBytes(await runtimeResponse(new URL('runtime-manifest.json', base), options), 1024 * 1024, signal);
  const manifest = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(manifestBytes));
  if (manifest.schemaVersion !== 1 || manifest.ortWebVersion !== ORT_WEB_VERSION || !Array.isArray(manifest.records)) throw new Error('ランタイムの固定バージョンが配布仕様と一致しません。');
  const candidates = manifest.records.filter(record => record.file === `${WASM_FILE}.gz`);
  const record = candidates[0];
  if (candidates.length !== 1 || record.encoding !== 'gzip' || record.decodedFile !== WASM_FILE || record.decodedBytes !== WASM_BYTES
    || record.decodedSha256 !== WASM_SHA256 || record.url !== ORT_SOURCE + WASM_FILE || !validSize(record.bytes)
    || record.bytes > 25 * 1024 * 1024 || !validHash(record.sha256)) throw new Error('ランタイムの復元元が固定したWASMと一致しません。');
  for (const file of ['ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.asyncify.mjs']) {
    const entries = manifest.records.filter(item => item.file === file);
    if (entries.length !== 1 || entries[0].url !== ORT_SOURCE + file || !validSize(entries[0].bytes) || !validHash(entries[0].sha256)) throw new Error('ランタイムのJavaScript配布仕様が不正です。');
  }
  const payload = await responseBytes(await runtimeResponse(new URL(record.file, base), options), Math.max(record.bytes, record.decodedBytes), signal);
  signal?.throwIfAborted();
  const binary = await decodeWasmPayload(payload, record);
  signal?.throwIfAborted();
  return { binary, record, manifest, delivery: { transport: isGzip(payload) ? 'gzip' : 'http-decoded',
    compressedHashVerified: isGzip(payload), decodedHashVerified: true, decodedBytes: binary.byteLength, decodedSha256: record.decodedSha256 } };
}
