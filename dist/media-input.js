const VIDEO = /\.(mp4|mov|m4v|webm)$/i;
const SPLIT = /^(.*\.tar(?:\.gz)?)\.(\d{3,})$/i;
const decoder = new TextDecoder('utf-8', { fatal: true });
function text(bytes) { return decoder.decode(bytes).replace(/\0.*$/s, '').trim(); }
function number(bytes) {
  if (bytes[0] & 0x80) {
    let result = bytes[0] & 0x7f;
    for (const byte of bytes.slice(1)) result = result * 256 + byte;
    if (!Number.isSafeInteger(result)) throw new Error('アーカイブ内のファイルが大きすぎます。');
    return result;
  }
  return parseInt(text(bytes) || '0', 8);
}
async function unpackTar(blob, label, onProgress) {
  const videos = [];
  let offset = 0, longName = null;
  while (offset + 512 <= blob.size) {
    const bytes = new Uint8Array(await blob.slice(offset, offset + 512).arrayBuffer());
    if (bytes.every(value => value === 0)) break;
    const expected = number(bytes.slice(148, 156));
    const checksum = bytes.reduce((sum, value, i) => sum + (i >= 148 && i < 156 ? 32 : value), 0);
    if (expected !== checksum) throw new Error(`${label}: TARの検査値が一致しません。全分割ファイルを選択してください。`);
    const prefix = text(bytes.slice(345, 500));
    let name = longName || `${prefix ? `${prefix}/` : ''}${text(bytes.slice(0, 100))}`;
    longName = null;
    const size = number(bytes.slice(124, 136));
    const type = String.fromCharCode(bytes[156]);
    const start = offset + 512;
    if (!Number.isFinite(size) || size < 0 || start + size > blob.size) throw new Error(`${label}: 分割ファイルが不足しています。`);
    if (type === 'L') longName = text(new Uint8Array(await blob.slice(start, start + size).arrayBuffer()));
    else if (type === 'x') {
      const extended = decoder.decode(await blob.slice(start, start + size).arrayBuffer());
      const found = extended.match(/(?:^|\n)\d+ path=([^\n]+)/);
      if (found) longName = found[1];
    } else if ((type === '0' || type === '\0') && VIDEO.test(name)) {
      name = name.split(/[\\/]/).pop();
      videos.push({ name, blob: blob.slice(start, start + size, /\.mov$/i.test(name) ? 'video/quicktime' : 'video/mp4') });
    }
    offset = start + Math.ceil(size / 512) * 512;
    onProgress?.(`${label}: ${Math.min(100, offset / blob.size * 100).toFixed(0)}%`);
  }
  if (!videos.length) throw new Error(`${label}: MOV/MP4/WebMが見つかりません。`);
  return videos;
}
export async function readVideoFiles(files, onProgress) {
  const groups = new Map(), blobs = [], videos = [];
  for (const file of files) {
    const match = file.name.match(SPLIT);
    if (match) {
      if (!groups.has(match[1])) groups.set(match[1], []);
      groups.get(match[1]).push({ file, index: Number(match[2]) });
    } else if (VIDEO.test(file.name)) videos.push({ name: file.name, blob: file });
    else if (/\.tar(?:\.gz)?$/i.test(file.name)) blobs.push({ name: file.name, blob: file });
    else throw new Error(`${file.name}: MOV、MP4、WebM、TARまたはTARの全分割ファイルを選択してください。`);
  }
  for (const [name, parts] of groups) {
    parts.sort((a, b) => a.index - b.index);
    if (parts.some((part, i) => part.index !== i + 1)) throw new Error(`${name}: .001から連続した全ファイルが必要です。`);
    blobs.push({ name, blob: new Blob(parts.map(part => part.file)) });
  }
  for (const item of blobs) {
    let blob = item.blob;
    if (/\.gz$/i.test(item.name)) {
      if (!globalThis.DecompressionStream) throw new Error('このブラウザはgzip展開に対応していません。');
      blob = await new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).blob();
    }
    videos.push(...await unpackTar(blob, item.name, onProgress));
  }
  return videos;
}
