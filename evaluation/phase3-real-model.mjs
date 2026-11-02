// Runs on GitHub Actions with a public COCO reference only. No user media input.
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {MODEL_SPECS} from '../dist/models.mjs';
import {cropTransform,rgbaToTensor,decodeSimCC} from '../dist/pose-codec.mjs';
import {footTransform,footTensor,decodeFootCandidates} from '../dist/foot-codec.mjs';
import {fusePose} from '../dist/geometry.mjs';
const cache=new URL('../.cache/phase3-public/',import.meta.url),output=new URL('./results/',import.meta.url),vendor=new URL('../dist/vendor/',import.meta.url);
await mkdir(cache,{recursive:true});await mkdir(output,{recursive:true});
const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
async function acquire(url,path,expected){
  let bytes;try{bytes=await readFile(path);}catch{const response=await fetch(url);if(!response.ok)throw new Error(`Download HTTP ${response.status}`);bytes=new Uint8Array(await response.arrayBuffer());await writeFile(path,bytes);}
  if(expected&&hash(bytes)!==expected)throw new Error('Public input SHA256 mismatch');return bytes;
}
const source={id:785,url:'https://raw.githubusercontent.com/open-mmlab/mmpose/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/tests/data/coco/000000000785.jpg',sha256:'83981537a7baeafbeb9c8cb67b3484dc26433f574b3685d021fa537e277e4726',license:'CC-BY-2.0',original:'https://farm8.staticflickr.com/7015/6795644157_f019453ae7_z.jpg'};
const imagePath=new URL('public-785.jpg',cache);await acquire(source.url,imagePath,source.sha256);
const imageSize={width:640,height:425};
const decoded=spawnSync('ffmpeg',['-v','error','-i',imagePath.pathname,'-frames:v','1','-f','rawvideo','-pix_fmt','rgba','pipe:1'],{maxBuffer:8*1024*1024});
if(decoded.status!==0)throw new Error(decoded.stderr?.toString()||'Cannot decode public frame');
if(decoded.stdout.length!==imageSize.width*imageSize.height*4)throw new Error('Unexpected reference dimensions');
const image={...imageSize,data:new Uint8Array(decoded.stdout.buffer,decoded.stdout.byteOffset,decoded.stdout.length)};
const ids=['rtmw3d-x','focus-foot8-one2many'],weights={};
for(const id of ids)weights[id]=await acquire(MODEL_SPECS[id].url,new URL(id+'.onnx',cache),MODEL_SPECS[id].sha256);
const ort=await import(new URL('ort.webgpu.min.mjs',vendor).href);
ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=vendor.href;
ort.env.wasm.wasmBinary=new Uint8Array(await readFile(new URL('ort-wasm-simd-threaded.jsep.wasm',vendor)));
// Download is finished; no inference step may initiate a network request.
globalThis.fetch=async()=>{throw new Error('Network disabled during inference');};
const measured=[];
let bodyObservation;
for(const id of ids){
  const spec=MODEL_SPECS[id],session=await ort.InferenceSession.create(new Uint8Array(weights[id]),{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  const bbox=[281.1622971754808,43.81200937124399,215.15948955829327,336.5215242826022];
  const start=performance.now(),transform=id==='rtmw3d-x'?cropTransform(bbox,spec.inputWH):footTransform(image,[0,0,image.width,image.height]);
  const tensor=new ort.Tensor('float32',id==='rtmw3d-x'?rgbaToTensor(image,transform):footTensor(image,transform),[1,3,spec.inputWH[1],spec.inputWH[0]]);
  const preprocessMs=performance.now()-start;
  const inferenceStart=performance.now(),result=await session.run({[session.inputNames[0]]:tensor}),modelMs=performance.now()-inferenceStart;
  const outputs=Object.entries(result).map(([name,value])=>({name,shape:value.dims,type:value.type,finite:Array.from(value.data).every(Number.isFinite)}));
  if(!outputs.every(item=>item.finite))throw new Error('Non-finite actual ONNX output');
  let checks,summary;
  if(id==='rtmw3d-x'){
    const points=decodeSimCC(result,transform,spec);
    checks={threeHeads:outputs.length===3,points133:points.length===133,finiteXY:points.every(p=>Number.isFinite(p.x+p.y)),finiteRelativeZ:points.every(p=>Number.isFinite(p.relativeZ))};
    bodyObservation={schemaVersion:1,sourceId:'public-reference',timestampMs:0,timeUncertaintyMs:Infinity,imageSize,bbox,keypoints:points,modelId:id,depthKind:'rootRelativeModelEstimate',inputMode:'wholebody'};
    const pose=fusePose([bodyObservation],[],{timestampMs:0,scaleConfirmed:false});
    checks.relativeOnly=pose.metric===false&&pose.root?.valid!==true;
    summary={visiblePoints:points.filter(p=>p.observable&&p.score>=.25).length,relativeZRange:[Math.min(...points.map(p=>p.relativeZ)),Math.max(...points.map(p=>p.relativeZ))],poseUnits:pose.units,rootValid:pose.root?.valid};
  }else{
    const candidates=decodeFootCandidates(Object.values(result)[0],spec,transform);
    checks={footHead:outputs.some(value=>JSON.stringify(value.shape)==='[1,29,8400]'),finiteCandidates:candidates.every(c=>Number.isFinite(c.detectionScore))};
    summary={candidateCount:candidates.length,note:'Candidate count is not an accuracy metric.'};
  }
  if(!Object.values(checks).every(Boolean))throw new Error(`${id}: ${JSON.stringify(checks)}`);
  measured.push({id,sha256:hash(weights[id]),backend:'wasm',threads:1,inputNames:session.inputNames,outputs,preprocessMs,modelMs,checks,summary});
  tensor.dispose();for(const value of Object.values(result))value.dispose();await session.release();
}
const report={schemaVersion:1,passed:true,purpose:'Actual shipped Node/WASM model and codec dataflow on a public still image; not WebGPU FPS or 3D accuracy.',source,boxOrigin:'Frozen output of YOLOX on the same public image from phase 2; not a ground-truth box.',measured,userMediaUsed:false,webgpuMeasured:false,steamVrHardwareMeasured:false};
await writeFile(new URL('phase3-real-model.json',output),JSON.stringify(report,null,2)+'\n');
await mkdir(new URL('../dist/evaluation/',import.meta.url),{recursive:true});
await writeFile(new URL('../dist/evaluation/phase3-real-model.json',import.meta.url),JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report,null,2));
