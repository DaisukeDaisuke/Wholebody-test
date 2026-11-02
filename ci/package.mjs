import {readFile,writeFile,mkdir,copyFile,rm} from 'node:fs/promises';
const readJSON=async path=>JSON.parse(await readFile(path,'utf8'));
const manifest=await readJSON('artifacts/hf/models.json');
for(const spec of [manifest.body,...Object.values(manifest.bodies),manifest.pose3d,...manifest.parts.models]){
  const url=new URL(spec.url);
  if(url.origin!=='https://huggingface.co'||!/^\/[^/]+\/[^/]+\/resolve\/[a-f0-9]{40}\/.+\.onnx$/.test(url.pathname)||url.search||! /^[a-f0-9]{64}$/.test(spec.sha256))throw new Error('Expected a pinned, anonymous Hugging Face model URL');
}
// The deployment contains only code and aggregate evidence. All inference
// weights are downloaded anonymously from the Actions-published HF revision.
await rm('dist/models',{recursive:true,force:true});
await mkdir('dist/models/parts',{recursive:true});await mkdir('dist/evidence',{recursive:true});await mkdir('dist/vendor',{recursive:true});
await writeFile('dist/models.json',JSON.stringify(manifest,null,2)+'\n','utf8');
await writeFile('dist/models/parts/parts-manifest.json',JSON.stringify(manifest.parts,null,2)+'\n','utf8');
for(const [source,target] of [
  ['artifacts/compact59/metrics.json','compact59-metrics.json'],
  ['artifacts/compact59/public-split.json','compact59-split.json'],
  ['artifacts/fast59/metrics.json','fast59-metrics.json'],
  ['artifacts/fast59/public-split.json','fast59-split.json'],
  ['artifacts/fast59/selection.json','fast59-selection.json'],
  ['artifacts/fast59/training-curve.json','fast59-training-curve.json'],
  ['artifacts/parts/parts-report.json','parts-report.json'],
  ['artifacts/parts/foot-split.json','foot-split.json'],
  ['artifacts/hf/publication.json','hf-publication.json'],
])await copyFile(source,'dist/evidence/'+target);
await rm('dist/evidence/browser-inference.json',{force:true});
await copyFile('RESULTS.md','dist/evidence/RESULTS.md');
const vendor=[['ort.webgpu.min.mjs','https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs'],['ort-wasm-simd-threaded.jsep.mjs','https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort-wasm-simd-threaded.jsep.mjs'],['ort-wasm-simd-threaded.jsep.wasm','https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort-wasm-simd-threaded.jsep.wasm'],['ort-LICENSE.txt','https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/LICENSE']];
for(const [name,url] of vendor){const r=await fetch(url);if(!r.ok)throw new Error(`Runtime fetch failed ${r.status}: ${name}`);const data=Buffer.from(await r.arrayBuffer());if(data.length>60*1024*1024)throw new Error('Oversized runtime');await writeFile('dist/vendor/'+name,data);}
await writeFile('dist/evidence/README.md',`# VRC Compact 実測資料\n\n採用モデルはmodels.jsonのbodyです。精度基準に届かない軽量候補は比較記録として残し、既定へ昇格させません。候補選択用のvalidationと、選択固定後に開くtestは分離しています。\n\n全モデルはGitHub ActionsからHugging Faceへ配布し、このアプリは公開resolve APIの固定revisionからSHA-256照合付きで取得します。\n\n精度はCOCO-WholeBodyの人物正解矩形で計測したPCKです。headless-video.jsonはNode.jsでの実動画推論とVRM更新の集計です。観測欠測を注入するchallenge、実RGBの部分切り抜き、通常の動画入力は別の結果として記録します。\n\n非公開動画、画像、切り抜き、観測座標列は配布しません。headless GPU描画とVRC実機は測定に含みません。\n\n顔の目・口は68点ランドマーク由来、頬は輪郭幅変化のみで頬の膨らみは未学習です。足はFootNet CC-BY-4.0による足裏・甲・側面の分類です。\n`,'utf8');
console.log(JSON.stringify({prepared:true,modelSource:'huggingface-public-pinned-resolve',bundledModelWeights:0,privateInputsIncluded:false}));
