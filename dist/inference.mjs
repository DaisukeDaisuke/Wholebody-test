import {cropTransform,rgbaToTensor,decodeSimCC,predictedHandBox,acceptHandRefinement} from './pose-codec.mjs';
import {MODEL_SPECS,REFINEMENT_MANIFEST} from './models.mjs';
export {BODY_MODELS} from './models.mjs';
export {BODY_EDGES,HAND_EDGES,FOOT_EDGES} from './pose-codec.mjs';

const ORT_BASE='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
const CACHE_NAME='wholebody-phase2-models-v1';
let runtimePromise;
async function loadRuntime(offline) {
  if (!runtimePromise) runtimePromise=(async()=>{
    let ort;
    try {ort=await import('./vendor/ort.webgpu.min.mjs');ort.env.wasm.wasmPaths=new URL('./vendor/',import.meta.url).href;}
    catch(error) {if(offline&&!navigator.serviceWorker?.controller)throw new Error('先にオンラインでオフライン準備を行ってください。');ort=await import(ORT_BASE+'ort.webgpu.min.mjs');ort.env.wasm.wasmPaths=ORT_BASE;}
    ort.env.wasm.numThreads=1;
    ort.env.wasm.proxy=false;
    return ort;
  })().catch(e=>{runtimePromise=null;throw e;});
  return runtimePromise;
}

async function openCache(){try{return await caches.open(CACHE_NAME);}catch{return null;}}
async function sha256(bytes){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),v=>v.toString(16).padStart(2,'0')).join('');}
async function readBytes(url,{progress,signal,offline,expectedBytes}) {
  const cache=await openCache(),cached=await cache?.match(url);
  if(cached){progress?.(1,'保存済みモデルを読み込み');return new Uint8Array(await cached.arrayBuffer());}
  if(offline)throw new Error('閉域モード：選択モデルのキャッシュがありません。');
  const response=await fetch(url,{signal,credentials:'omit',referrerPolicy:'no-referrer'});
  if(!response.ok)throw new Error('モデル取得 HTTP '+response.status+' : '+new URL(url).pathname);
  const total=Number(response.headers.get('content-length'))||expectedBytes||0;
  const chunks=[];let received=0;
  if(response.body){const reader=response.body.getReader();while(true){const {done,value}=await reader.read();if(done)break;chunks.push(value);received+=value.length;if(received>600*1024*1024){await reader.cancel();throw new Error('モデルサイズが600MiBの上限を超えました。');}progress?.(total?Math.min(.99,received/total):null,Math.round(received/1048576)+' MiB');}}
  else{const b=new Uint8Array(await response.arrayBuffer());chunks.push(b);received=b.length;}
  const bytes=new Uint8Array(received);let at=0;for(const part of chunks){bytes.set(part,at);at+=part.length;}
  try{await cache?.put(url,new Response(bytes,{headers:{'Content-Type':'application/octet-stream'}}));}catch{}
  progress?.(1,'取得完了');return bytes;
}

async function resolveSpec(id,offline) {
  let spec={...MODEL_SPECS[id]};
  if(!spec.id)throw new Error('Unknown model '+id);
  if(spec.filename){
    try{
      const bytes=await readBytes(REFINEMENT_MANIFEST,{offline,signal:AbortSignal.timeout(8000)});
      const manifest=JSON.parse(new TextDecoder().decode(bytes)), found=manifest.models?.find(m=>m.id===id);
      if(found?.sha256&&found?.url&&Array.isArray(found.inputWH))spec={...spec,...found};
    }catch{}
  }
  return spec;
}

export class PoseEngine {
  constructor({onProgress=()=>{}}={}){this.onProgress=onProgress;this.sessions=new Map();this.trackers=new Map();this.busy=false;this.offline=false;this.abort=new AbortController();}
  notify(stage,progress,message){this.onProgress({stage,progress,message});}
  async load({bodyModel='rtmw-l',handRefinement=true,backend='webgpu',offline=false}={}){
    this.offline=offline;this.backend=backend;this.bodyId=bodyModel;this.handRefinement=handRefinement;this.abort=new AbortController();
    if(backend==='webgpu'&&!navigator.gpu)throw new Error('この推論画面にはWebGPU対応のPCブラウザが必要です。WASMは比較用として明示的に選択できます。');
    this.notify('runtime',null,'推論ランタイムを読み込み');this.ort=await loadRuntime(offline);
    await this.loadModel(bodyModel);
    if(handRefinement)await this.loadModel('hand5');
    this.notify('ready',1,'モデル準備完了');return this;
  }
  async loadModel(id){
    if(this.sessions.has(id))return this.sessions.get(id);
    const spec=await resolveSpec(id,this.offline);
    let bytes;
    const progress=(fraction,message)=>this.notify(id,fraction,spec.label+' · '+message);
    try{bytes=await readBytes(spec.url,{progress,signal:this.abort.signal,offline:this.offline,expectedBytes:spec.bytes});}
    catch(error){
      if(!spec.archiveUrl||this.offline||this.abort.signal.aborted)throw error;
      this.notify(id,null,'公式配布からモデルを取得');
      const zip=await readBytes(spec.archiveUrl,{progress,signal:this.abort.signal,offline:false,expectedBytes:spec.bytes});
      const {unzipSync}=await import('https://cdn.jsdelivr.net/npm/fflate@0.8.2/esm/browser.js');
      const files=unzipSync(zip,{filter:file=>file.name.endsWith('.onnx')});
      const names=Object.keys(files),preferred=names.filter(n=>/(^|\/)end2end\.onnx$/.test(n));
      const name=preferred.length===1?preferred[0]:names.length===1?names[0]:null;
      if(!name)throw new Error('公式ZIP内のONNXを一意に特定できません。');bytes=files[name];
      try{await (await openCache())?.put(spec.url,new Response(bytes));}catch{}
    }
    this.notify(id,null,'モデルの整合性を確認');
    const actualHash=await sha256(bytes);
    if(spec.sha256&&actualHash!==spec.sha256){await (await openCache())?.delete(spec.url);throw new Error(id+' のSHA-256が配布仕様と一致しません。');}
    this.notify(id,null,'推論セッションを初期化');
    const session=await this.ort.InferenceSession.create(bytes,{executionProviders:[this.backend],graphOptimizationLevel:'all'});
    const record={session,spec:{...spec,actualHash,hashVerified:!!spec.sha256}};
    this.sessions.set(id,record);return record;
  }
  readFrame(source){
    const width=source.videoWidth||source.naturalWidth||source.width,height=source.videoHeight||source.naturalHeight||source.height;
    if(!(width>0&&height>0))throw new Error('映像フレームがまだ利用できません。');
    if(!this.canvas){this.canvas=typeof OffscreenCanvas!=='undefined'?new OffscreenCanvas(width,height):Object.assign(document.createElement('canvas'),{width,height});this.context=this.canvas.getContext('2d',{willReadFrequently:true,alpha:false});}
    if(this.canvas.width!==width||this.canvas.height!==height){this.canvas.width=width;this.canvas.height=height;}
    this.context.drawImage(source,0,0,width,height);
    return this.context.getImageData(0,0,width,height);
  }
  async run(id,image,box){
    const record=this.sessions.get(id)||await this.loadModel(id),{session,spec}=record;
    const transform=cropTransform(box,spec.inputWH),pixels=rgbaToTensor(image,transform);
    const tensor=new this.ort.Tensor('float32',pixels,[1,3,spec.inputWH[1],spec.inputWH[0]]);
    let result;
    try{result=await session.run({[session.inputNames[0]]:tensor});return decodeSimCC(result,transform,spec);}
    finally{tensor.dispose?.();for(const value of Object.values(result||{}))value.dispose?.();}
  }
  trackedBox(sourceId,width,height,roi,timestampMs){
    if(roi)return roi.slice();
    const tracker=this.trackers.get(sourceId);
    if(!tracker||timestampMs<tracker.timestampMs||timestampMs-tracker.timestampMs>700||tracker.frames%30===0)return [0,0,width,height];
    return tracker.box;
  }
  updateTracker(sourceId,points,image,timestampMs){
    const reliable=points.slice(0,23).filter(p=>p.score>.4&&Number.isFinite(p.x+p.y));
    const prev=this.trackers.get(sourceId);
    if(reliable.length<8){this.trackers.delete(sourceId);return;}
    const xs=reliable.map(p=>p.x),ys=reliable.map(p=>p.y),left=Math.min(...xs),right=Math.max(...xs),top=Math.min(...ys),bottom=Math.max(...ys);
    // Wide margin includes head and toes omitted by sparse low-confidence predictions.
    const w=Math.max(right-left,image.width*.08),h=Math.max(bottom-top,image.height*.18);
    const fresh=[left-w*.25,top-h*.15,w*1.5,h*1.3];
    const box=prev?fresh.map((v,i)=>v*.7+prev.box[i]*.3):fresh;
    this.trackers.set(sourceId,{box,timestampMs,frames:(prev?.frames||0)+1});
  }
  async estimate(source,{sourceId='a',timestampMs=performance.now(),timeUncertaintyMs=0,roi=null,handRefinement=this.handRefinement}={}){
    if(this.busy)throw new Error('PoseEngine executes one frame at a time. Keep the newest frame per input.');
    if(!this.sessions.has(this.bodyId))throw new Error('先にモデルを読み込んでください。');
    this.busy=true;const started=performance.now();
    try{
      const image=this.readFrame(source),bbox=this.trackedBox(sourceId,image.width,image.height,roi,timestampMs),bodyStart=performance.now();
      const points=await this.run(this.bodyId,image,bbox),bodyMs=performance.now()-bodyStart;
      const handsStart=performance.now(),refinements=[];
      if(handRefinement){
        for(const [side,offset] of [['left',91],['right',112]]){
          const handBox=predictedHandBox(points,side,bbox);if(!handBox){refinements.push({side,accepted:false,reason:'no_roi'});continue;}
          const hand=await this.run('hand5',image,handBox),accepted=acceptHandRefinement(points,hand,handBox,side);
          if(accepted){for(let j=0;j<21;j++){const previous=points[offset+j];points[offset+j]={...previous,...hand[j],id:offset+j,source:'handRefined2D',relativeZ:previous.relativeZ,relativeZScore:previous.relativeZScore};}}
          refinements.push({side,roi:handBox,accepted});
        }
      }
      for(const point of points){point.observable=point.observable&&point.x>=0&&point.y>=0&&point.x<image.width&&point.y<image.height;}
      this.updateTracker(sourceId,points,image,timestampMs);
      return {schemaVersion:1,sourceId,timestampMs,timeUncertaintyMs,imageSize:{width:image.width,height:image.height},bbox,keypoints:points,refinements,modelId:this.bodyId,modelHash:this.sessions.get(this.bodyId).spec.actualHash,backend:this.backend,timings:{bodyMs,handsMs:performance.now()-handsStart,totalMs:performance.now()-started},depthKind:this.sessions.get(this.bodyId).spec.dimensions===3?'rootRelativeModelEstimate':'none'};
    }finally{this.busy=false;}
  }
  async dispose(){this.abort.abort();for(const {session} of this.sessions.values())await session.release();this.sessions.clear();this.trackers.clear();}
  resetSource(sourceId){this.trackers.delete(sourceId);}
}
