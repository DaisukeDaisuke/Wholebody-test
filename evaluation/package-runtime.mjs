// Build dependencies in GitHub Actions; this script never reads local media.
import {build} from '../build/node_modules/esbuild/lib/main.js';
import {mkdir,writeFile,copyFile,readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const vendor=new URL('../dist/vendor/',import.meta.url);
await mkdir(vendor,{recursive:true});
await build({entryPoints:[new URL('../build/vrm-entry.mjs',import.meta.url).pathname],bundle:true,format:'esm',platform:'browser',target:'es2022',minify:true,legalComments:'eof',outfile:new URL('vrm-runtime.mjs',vendor).pathname});
const runtime='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
const files=['ort.webgpu.min.mjs','ort-wasm-simd-threaded.jsep.mjs','ort-wasm-simd-threaded.jsep.wasm'];
const records=[];
for(const name of files){
  const response=await fetch(runtime+name);if(!response.ok)throw new Error(`ORT runtime HTTP ${response.status}: ${name}`);
  const bytes=new Uint8Array(await response.arrayBuffer());await writeFile(new URL(name,vendor),bytes);
  records.push({file:name,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex'),url:runtime+name});
}
for(const [name,url] of [
  ['ONNX-LICENSE.txt','https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/LICENSE'],
  ['ONNX-ThirdPartyNotices.txt','https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/ThirdPartyNotices.txt']
]){const response=await fetch(url);if(!response.ok)throw new Error(`License HTTP ${response.status}`);await writeFile(new URL(name,vendor),await response.text());}
await copyFile(new URL('../build/node_modules/three/LICENSE',import.meta.url),new URL('THREE-LICENSE.txt',vendor));
await copyFile(new URL('../build/node_modules/@pixiv/three-vrm/LICENSE',import.meta.url),new URL('VRM-LICENSE.txt',vendor));
const bundled=await readFile(new URL('vrm-runtime.mjs',vendor));records.push({file:'vrm-runtime.mjs',bytes:bundled.length,sha256:createHash('sha256').update(bundled).digest('hex'),dependencies:{three:'0.180.0',threeVrm:'3.5.4',esbuild:'0.25.0'}});
await writeFile(new URL('runtime-manifest.json',vendor),JSON.stringify({schemaVersion:1,records},null,2)+'\n');
console.log(JSON.stringify({vendorFiles:records.map(({file,bytes,sha256})=>({file,bytes,sha256}))}));
