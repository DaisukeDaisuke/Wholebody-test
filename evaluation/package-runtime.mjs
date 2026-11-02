// Build dependencies in GitHub Actions; this script never reads local media.
import {mkdir,writeFile,copyFile,readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
const onnxOnly=process.argv.includes('--onnx-only');
const outputIndex=process.argv.indexOf('--output');
if(outputIndex>=0&&!process.argv[outputIndex+1])throw new Error('--output needs a directory.');
const vendor=outputIndex<0?new URL('../dist/vendor/',import.meta.url):pathToFileURL(resolve(process.argv[outputIndex+1])+'/');
await mkdir(vendor,{recursive:true});
if(!onnxOnly){
  const {build}=await import('../build/node_modules/esbuild/lib/main.js');
  await build({entryPoints:[new URL('../build/vrm-entry.mjs',import.meta.url).pathname],bundle:true,format:'esm',platform:'browser',target:'es2022',minify:true,legalComments:'eof',outfile:new URL('vrm-runtime.mjs',vendor).pathname});
}
const ortWebVersion='1.31.1';
const runtime=`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortWebVersion}/dist/`;
const files=['ort.webgpu.min.mjs','ort-wasm-simd-threaded.jsep.mjs','ort-wasm-simd-threaded.jsep.wasm'];
const records=[];
for(const name of files){
  const response=await fetch(runtime+name);if(!response.ok)throw new Error(`ORT runtime HTTP ${response.status}: ${name}`);
  const bytes=new Uint8Array(await response.arrayBuffer());await writeFile(new URL(name,vendor),bytes);
  records.push({file:name,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex'),url:runtime+name});
}
for(const [name,url] of [
  ['ONNX-LICENSE.txt',`https://raw.githubusercontent.com/microsoft/onnxruntime/v${ortWebVersion}/LICENSE`],
  ['ONNX-ThirdPartyNotices.txt',`https://raw.githubusercontent.com/microsoft/onnxruntime/v${ortWebVersion}/ThirdPartyNotices.txt`]
]){const response=await fetch(url);if(!response.ok)throw new Error(`License HTTP ${response.status}`);await writeFile(new URL(name,vendor),await response.text());}
if(!onnxOnly){
  await copyFile(new URL('../build/node_modules/three/LICENSE',import.meta.url),new URL('THREE-LICENSE.txt',vendor));
  await copyFile(new URL('../build/node_modules/@pixiv/three-vrm/LICENSE',import.meta.url),new URL('VRM-LICENSE.txt',vendor));
  const bundled=await readFile(new URL('vrm-runtime.mjs',vendor));records.push({file:'vrm-runtime.mjs',bytes:bundled.length,sha256:createHash('sha256').update(bundled).digest('hex'),dependencies:{three:'0.180.0',threeVrm:'3.5.4',esbuild:'0.25.0'}});
}
await writeFile(new URL('runtime-manifest.json',vendor),JSON.stringify({schemaVersion:1,ortWebVersion,gitCommit:process.env.GITHUB_SHA||null,records},null,2)+'\n');
console.log(JSON.stringify({vendorFiles:records.map(({file,bytes,sha256})=>({file,bytes,sha256}))}));
