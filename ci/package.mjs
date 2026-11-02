import {readFile,writeFile,mkdir,copyFile,rm} from 'node:fs/promises';
const readJSON=async path=>JSON.parse(await readFile(path,'utf8'));
const manifest=await readJSON('artifacts/hf/models.json');
for(const spec of [manifest.body,...Object.values(manifest.bodies),manifest.pose3d,manifest.personDetector,...manifest.parts.models]){
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
// Retired rotation-only interpretation must not remain the default evidence.
// Its original result remains in Git history and the earlier Actions artifact.
await rm('dist/evidence/headless-video.json',{force:true});
await copyFile('RESULTS.md','dist/evidence/RESULTS.md');
const vendor=[['ort.webgpu.min.mjs','https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs'],['ort-wasm-simd-threaded.jsep.mjs','https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort-wasm-simd-threaded.jsep.mjs'],['ort-wasm-simd-threaded.jsep.wasm','https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort-wasm-simd-threaded.jsep.wasm'],['ort-LICENSE.txt','https://raw.githubusercontent.com/microsoft/onnxruntime/v1.22.0/LICENSE']];
for(const [name,url] of vendor){const r=await fetch(url);if(!r.ok)throw new Error(`Runtime fetch failed ${r.status}: ${name}`);const data=Buffer.from(await r.arrayBuffer());if(data.length>60*1024*1024)throw new Error('Oversized runtime');await writeFile('dist/vendor/'+name,data);}
await writeFile('dist/evidence/README.md',`# VRC Compact 実測資料\n\n既定構成はCompact59の画像座標と低頻度の学習済み3D奥行きです。models.jsonは各モデルの固定revisionとSHA-256を記録します。精度基準に届かない軽量候補は既定へ昇格させません。\n\nheadless-pose3d.jsonはNode.jsで実動画をデコードし、ONNX推論とVRM更新を実行した集計です。全身と実RGBの部分切り抜きについて、可視関節、駆動した四肢、ロスト時間、入力動作の保持を別々に記録します。回転量の減少だけでは追跡の改善を示せません。status: completedは実行完了であり、deploymentReadiness: passedとは別です。\n\nlabelled-body13.jsonはCOCO-WholeBodyの固定160画像・身体13点の正解ラベルとの比較です。棄却した点も誤りに数え、人物の正解矩形を使用します。構成の決定に使った最初の32画像はこの集計から除きます。これは2Dの精度であり、3Dの正解精度ではありません。\n\nモデルはActionsからHugging Faceへ配布し、公開resolve APIで匿名取得します。非公開動画、画像、切り抜き、観測座標列は配布しません。GPU描画、VRC実機、非公開動画の未実行部分を検証済みとは扱いません。\n\n顔の目・口は68点ランドマーク由来、頬は輪郭幅変化のみで頬の膨らみは未学習です。足はFootNet CC-BY-4.0による足裏・甲・側面の分類です。\n`,'utf8');
console.log(JSON.stringify({prepared:true,modelSource:'huggingface-public-pinned-resolve',bundledModelWeights:0,privateInputsIncluded:false}));
