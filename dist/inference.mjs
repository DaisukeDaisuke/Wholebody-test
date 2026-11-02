import {cropTransform,rgbaToTensor,decodeSimCC,predictedHandBox,handRefinementDecision} from './pose-codec.mjs';
import {MODEL_SPECS,REFINEMENT_MANIFEST} from './models.mjs';
import {detectorTransform,detectorTensor,decodePersons,boxIou} from './detector-codec.mjs';
import {footTransform,footTensor,decodeFootCandidates,associateFeet,applyFootAssociation} from './foot-codec.mjs';
import {FootTracker} from './foot-track.mjs';
import {TapirTracker,seedFootSquareRoi,TAPIR_SEED_ROI_POLICY} from './tapir-tracker.mjs';
import {TAPIR_REPOSITORY,TAPIR_MANIFEST_URL} from './tapir-models.mjs';
export {BODY_MODELS} from './models.mjs';
export {BODY_EDGES,HAND_EDGES,FOOT_EDGES} from './pose-codec.mjs';

const ORT_BASE='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
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
  constructor({onProgress=()=>{}}={}){this.onProgress=onProgress;this.sessions=new Map();this.trackers=new Map();this.footTrackers=new Map();this.seedVersions=new Map();this.footDetectionTimes=new Map();this.manualFootSources=new Set();this.tapirMaster=null;this.busy=false;this.offline=false;this.abort=new AbortController();}
  notify(stage,progress,message){this.onProgress({stage,progress,message});}
  async load({bodyModel='rtmw3d-x',bodyEnabled=true,footTrackingEnabled=false,footTrackingMethod='bootstapir',handRefinement=true,personDetection=true,footRefinement=false,backend='webgpu',offline=false}={}){
    this.offline=offline;this.backend=backend;this.bodyId=bodyModel;this.handRefinement=handRefinement;this.personDetection=personDetection;this.footRefinement=footRefinement;this.abort=new AbortController();
    if(!['bootstapir','image-flow'].includes(footTrackingMethod))throw new Error('足の追跡方式を選択してください。');
    this.footTrackingMethod=footTrackingMethod;
    if(backend==='webgpu'&&!navigator.gpu)throw new Error('この推論画面にはWebGPU対応のPCブラウザが必要です。WASMは比較用として明示的に選択できます。');
    this.notify('runtime',null,'推論ランタイムを読み込み');this.ort=await loadRuntime(offline);
    if(bodyEnabled)await this.loadModel(bodyModel);
    if(bodyEnabled&&personDetection)await this.loadModel('person-yolox-tiny-416');
    if(bodyEnabled&&handRefinement)await this.loadModel('hand5');
    if(footTrackingEnabled||(bodyEnabled&&footRefinement))await this.loadModel('focus-foot8-one2many');
    if(footTrackingEnabled&&footTrackingMethod==='bootstapir')await this.loadTapir();
    this.notify('ready',1,'モデル準備完了');return this;
  }
  async loadTapir(){
    if(this.tapirMaster)return this.tapirMaster;
    if(!TAPIR_MANIFEST_URL)throw new Error('学習済み足追跡モデルはActionsで変換・検証中です。公開リビジョンの確定後に利用できます。');
    this.notify('bootstapir',null,'学習済み点追跡の配布仕様を確認');
    const manifest=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(await readBytes(TAPIR_MANIFEST_URL,{offline:this.offline,signal:this.abort.signal})));
    if(manifest.repository!==TAPIR_REPOSITORY||manifest.license!=='apache-2.0'||manifest.parityPassed!==true||manifest.codecParityPassed!==true||manifest.models?.length!==3)throw new Error('学習済み追跡モデルの検証済み配布仕様と一致しません。');
    const modelBuffers={};
    for(const id of ['encoder','query','step']){
      const model=manifest.models.find(item=>item.id===id),expected=`causal-bootstapir-${id}-256-q3-fp32.onnx`;
      if(!model||model.filename!==expected||!/^[a-f0-9]{40}$/.test(model.revision||'')||model.url!==`https://huggingface.co/${TAPIR_REPOSITORY}/resolve/${model.revision}/${expected}`)throw new Error('追跡モデルの固定リビジョンが不正です。');
      modelBuffers[id]=await readBytes(model.url,{offline:this.offline,signal:this.abort.signal,expectedBytes:model.bytes,progress:(fraction,message)=>this.notify('bootstapir-'+id,fraction,`BootsTAPIR ${id} · ${message}`)});
    }
    this.notify('bootstapir',null,'3つの追跡セッションを初期化');
    this.tapirMaster=await TapirTracker.create({ort:this.ort,modelBuffers,manifest,executionProviders:[this.backend],sessionOptions:{graphOptimizationLevel:'all'},graphSessionOptions:{step:{graphOptimizationLevel:'disabled'}}});
    return this.tapirMaster;
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
    const provider=spec.kind==='personDetector'?'wasm':this.backend;
    const session=await this.ort.InferenceSession.create(bytes,{executionProviders:[provider],graphOptimizationLevel:'all'});
    const record={session,spec:{...spec,actualHash,hashVerified:!!spec.sha256,executionProvider:provider}};
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
    const prepStart=performance.now();
    const transform=cropTransform(box,spec.inputWH),pixels=rgbaToTensor(image,transform);
    const tensor=new this.ort.Tensor('float32',pixels,[1,3,spec.inputWH[1],spec.inputWH[0]]);
    this.frameTiming.preprocessMs+=performance.now()-prepStart;
    let result;
    try{const start=performance.now();result=await session.run({[session.inputNames[0]]:tensor});this.frameTiming.modelMs+=performance.now()-start;this.frameTiming.modelRuns++;return decodeSimCC(result,transform,spec);}
    finally{tensor.dispose?.();for(const value of Object.values(result||{}))value.dispose?.();}
  }
  async detectPersons(image){
    const {session,spec}=this.sessions.get('person-yolox-tiny-416')||await this.loadModel('person-yolox-tiny-416');
    const prepStart=performance.now();
    const transform=detectorTransform(image,spec.inputWH),tensor=new this.ort.Tensor('float32',detectorTensor(image,transform),[1,3,spec.inputWH[1],spec.inputWH[0]]);
    this.frameTiming.preprocessMs+=performance.now()-prepStart;
    let result;try{const start=performance.now();result=await session.run({[session.inputNames[0]]:tensor});this.frameTiming.modelMs+=performance.now()-start;this.frameTiming.modelRuns++;return decodePersons(result,transform,spec);}
    finally{tensor.dispose?.();for(const value of Object.values(result||{}))value.dispose?.();}
  }
  async selectPersonBox(sourceId,image,roi,timestampMs){
    if(roi)return {bbox:roi.slice(),personDetection:{enabled:this.personDetection,detected:null,selectedBox:roi.slice(),candidateCount:0,selection:'manual-roi'}};
    const previous=this.trackers.get(sourceId),recent=previous&&timestampMs>=previous.timestampMs&&timestampMs-previous.timestampMs<=700;
    if(!this.personDetection){const bbox=this.trackedBox(sourceId,image.width,image.height,null,timestampMs);return {bbox,personDetection:{enabled:false,detected:null,selectedBox:bbox,candidateCount:0,selection:recent?'tracked-person':'full-frame'}};}
    if(recent&&previous.frames%30!==0)return {bbox:previous.box,personDetection:{enabled:true,detected:true,selectedBox:previous.box,candidateCount:previous.candidateCount||1,selection:'tracked-person'}};
    if(!recent)this.trackers.delete(sourceId);
    const candidates=await this.detectPersons(image);
    if(!candidates.length){this.trackers.delete(sourceId);return {bbox:null,personDetection:{enabled:true,detected:false,selectedBox:null,candidateCount:0,selection:'not-detected',backend:'wasm'}};}
    let selected=candidates[0],selection='largest-person',targetReset=!!previous&&!recent;
    if(recent){const overlap=candidates.map(candidate=>({candidate,iou:boxIou(previous.box,candidate.bboxXYWH)})).sort((a,b)=>b.iou-a.iou);if(overlap[0].iou>=.1){selected=overlap[0].candidate;selection='overlap-with-previous';}}
    if(recent&&selection==='largest-person'){this.trackers.delete(sourceId);targetReset=true;}
    return {bbox:selected.bboxXYWH,personDetection:{enabled:true,detected:true,selectedBox:selected.bboxXYWH,candidateCount:candidates.length,selection,targetReset,score:selected.score,backend:'wasm'}};
  }
  async refineFeet(image,box,points){
    const {session,spec}=this.sessions.get('focus-foot8-one2many')||await this.loadModel('focus-foot8-one2many');
    const prepStart=performance.now();
    const [x,y,w,h]=box,transform=footTransform(image,[x-w*.125,y-h*.125,w*1.25,h*1.25]);
    const tensor=new this.ort.Tensor('float32',footTensor(image,transform),[1,3,640,640]);
    this.frameTiming.preprocessMs+=performance.now()-prepStart;
    let result;try{
      const start=performance.now();result=await session.run({[session.inputNames[0]]:tensor});this.frameTiming.modelMs+=performance.now()-start;this.frameTiming.modelRuns++;
      const association=associateFeet(decodeFootCandidates(Object.values(result)[0],spec,transform),points);
      const acceptedCocoIndices=applyFootAssociation(points,association,{blend:true});
      return {modelId:spec.id,policyId:association.policyId,blendWeight:.5,roi:transform.cropXYWH,acceptedCocoIndices,candidateCount:association.candidateCountEvaluated,selectedCandidateIndex:association.selected?.candidate.candidateIndex??null};
    }finally{tensor.dispose?.();for(const value of Object.values(result||{}))value.dispose?.();}
  }
  async detectFeet(image,box){
    const {session,spec}=this.sessions.get('focus-foot8-one2many')||await this.loadModel('focus-foot8-one2many');
    const prepStart=performance.now(),transform=footTransform(image,box||[0,0,image.width,image.height]);
    const tensor=new this.ort.Tensor('float32',footTensor(image,transform),[1,3,640,640]);
    this.frameTiming.preprocessMs+=performance.now()-prepStart;
    let result;
    try{const start=performance.now();result=await session.run({[session.inputNames[0]]:tensor});this.frameTiming.modelMs+=performance.now()-start;this.frameTiming.modelRuns++;return decodeFootCandidates(Object.values(result)[0],spec,transform);}
    finally{tensor.dispose?.();for(const value of Object.values(result||{}))value.dispose?.();}
  }
  async estimateFeet(image,{sourceId,timestampMs,timeUncertaintyMs,roi,footSeed,frameMediaTimeMs},started){
    let tracker=this.footTrackers.get(sourceId);
    const learned=this.footTrackingMethod==='bootstapir';
    if(!tracker){tracker=learned?new TapirTracker({ort:this.ort,sessions:this.tapirMaster.sessions,manifest:this.tapirMaster.manifest}):new FootTracker();this.footTrackers.set(sourceId,tracker);}
    let result=null,manualSeedApplied=false;
    if(footSeed&&this.seedVersions.get(sourceId)!==footSeed.version){
      if(Number.isFinite(footSeed.mediaTimeMs)&&Number.isFinite(frameMediaTimeMs)&&Math.abs(footSeed.mediaTimeMs-frameMediaTimeMs)>75)throw new Error('3点を選んだフレームから動画が進んでいます。動画を停止して3点を再指定してください。');
      // The optional crop uses only the initial three visible points. Keep it
      // fixed for this identity; an explicit user region always takes priority.
      const trackingRoi=roi||(learned&&footSeed.cropPolicy==='seed-square'?seedFootSquareRoi(footSeed.points,image):null);
      result=await tracker.seed({...footSeed,timestampMs,image,roi:trackingRoi,seedSource:'manual'});
      this.seedVersions.set(sourceId,footSeed.version);this.manualFootSources.add(sourceId);manualSeedApplied=true;
    }
    if(learned){
      if(!result&&tracker.requiresSeed&&!this.manualFootSources.has(sourceId)){
        const last=this.footDetectionTimes.get(sourceId),detectionDue=!Number.isFinite(last)||timestampMs<last||timestampMs-last>=500;
        if(detectionDue){
          const candidates=await this.detectFeet(image,roi);this.footDetectionTimes.set(sourceId,timestampMs);
          // Use the detector's existing geometric gates only to start a new
          // identity. It cannot replace an explicitly selected foot after loss.
          const proposals=new FootTracker({maxTracks:1}).update({image,timestampMs,candidates,roi});
          // Detector coordinates may extend beyond its crop. Automatic seeds
          // must fit the same clipped region that the learned tracker will use;
          // an outside proposal is a missing detection, not an inference crash.
          const bounds=roi?[Math.max(0,Math.floor(roi[0])),Math.max(0,Math.floor(roi[1])),Math.min(image.width,Math.ceil(roi[0]+roi[2])),Math.min(image.height,Math.ceil(roi[1]+roi[3]))]:[0,0,image.width,image.height];
          const insideRegion=point=>Array.isArray(point)&&point.length===2&&point.every(Number.isFinite)&&point[0]>=bounds[0]&&point[0]<bounds[2]&&point[1]>=bounds[1]&&point[1]<bounds[3];
          const found=proposals.footTracks.find(track=>[track.bigToe,track.littleToe,track.heel].every(insideRegion));
          if(found)result=await tracker.seed({points:[found.bigToe,found.littleToe,found.heel],side:found.modelSide||'unknown',timestampMs,image,roi,seedSource:'detector',seedScores:[found.score,found.score,found.score]});
        }
      }
      if(!result)result=await tracker.update({image,timestampMs});
      if(result.diagnostics?.modelRan){
        const timing=result.diagnostics;this.frameTiming.modelMs+=timing.inferenceMs||0;this.frameTiming.modelRuns+=timing.modelRuns||0;
        this.frameTiming.preprocessMs+=Math.max(0,(timing.processingMs||0)-(timing.inferenceMs||0));
      }
    }else{
      const detectionRan=tracker.needsDetection(timestampMs),candidates=detectionRan?await this.detectFeet(image,roi):undefined;
      result=tracker.update({image,timestampMs,candidates,roi});
    }
    const spec=this.sessions.get('focus-foot8-one2many').spec,modelId=learned?'google/causal-bootstapir':spec.id;
    const trackingBox=result.diagnostics?.fixedRoi||roi||null;
    const regionPolicy=roi?'explicit-user-roi':learned&&trackingBox&&footSeed?.cropPolicy==='seed-square'?TAPIR_SEED_ROI_POLICY.id:'full-frame';
    return {schemaVersion:1,sourceId,timestampMs,timeUncertaintyMs,imageSize:{width:image.width,height:image.height},
      bbox:trackingBox||[0,0,image.width,image.height],...result,diagnostics:{...result.diagnostics,manualSeedApplied,regionPolicy},inputMode:'feet',refinements:[],
      personDetection:{enabled:false,detected:null,selection:'foot-only',selectedBox:trackingBox,candidateCount:0},
      modelId,modelHash:learned?null:spec.actualHash,modelHashes:learned?Object.fromEntries(this.tapirMaster.manifest.models.map(model=>[model.id,model.sha256])):null,
      backend:this.backend,footTrackingMethod:this.footTrackingMethod,scoreKind:learned?'uncalibratedLearnedVisibility':'uncalibratedModelAndFlowQuality',
      timings:{...this.frameTiming,detectorMs:0,bodyMs:0,handsMs:0,feetMs:performance.now()-started,neuralFrame:this.frameTiming.modelRuns>0,totalMs:performance.now()-started}};
  }
  trackedBox(sourceId,width,height,roi,timestampMs){
    if(roi)return roi.slice();
    const tracker=this.trackers.get(sourceId);
    if(!tracker||timestampMs<tracker.timestampMs||timestampMs-tracker.timestampMs>700||tracker.frames%30===0)return [0,0,width,height];
    return tracker.box;
  }
  updateTracker(sourceId,points,image,timestampMs){
    const reliable=points.slice(0,23).filter(p=>p.score>.4&&Number.isFinite(p.x+p.y));
    const stored=this.trackers.get(sourceId),prev=stored&&timestampMs>=stored.timestampMs&&timestampMs-stored.timestampMs<=700?stored:null;
    if(reliable.length<8){this.trackers.delete(sourceId);return;}
    const xs=reliable.map(p=>p.x),ys=reliable.map(p=>p.y),left=Math.min(...xs),right=Math.max(...xs),top=Math.min(...ys),bottom=Math.max(...ys);
    // Wide margin includes head and toes omitted by sparse low-confidence predictions.
    const w=Math.max(right-left,image.width*.08),h=Math.max(bottom-top,image.height*.18);
    const fresh=[left-w*.25,top-h*.15,w*1.5,h*1.3];
    const box=prev?fresh.map((v,i)=>v*.7+prev.box[i]*.3):fresh;
    this.trackers.set(sourceId,{box,timestampMs,frames:(prev?.frames||0)+1});
  }
  async estimate(source,{sourceId='a',timestampMs=performance.now(),timeUncertaintyMs=0,roi=null,mode='wholebody',footSeed=null,frameMediaTimeMs=null,handRefinement=this.handRefinement,footRefinement=this.footRefinement}={}){
    if(this.busy)throw new Error('PoseEngine executes one frame at a time. Keep the newest frame per input.');
    if(!this.sessions.has(mode==='feet'?'focus-foot8-one2many':this.bodyId))throw new Error('先に選択した入力方式のモデルを読み込んでください。');
    this.busy=true;const started=performance.now();
    this.frameTiming={preprocessMs:0,modelMs:0,modelRuns:0};
    try{
      const image=this.readFrame(source);this.frameTiming.preprocessMs=performance.now()-started;
      if(mode==='feet')return await this.estimateFeet(image,{sourceId,timestampMs,timeUncertaintyMs,roi,footSeed,frameMediaTimeMs},started);
      const detectorStart=performance.now(),{bbox,personDetection}=await this.selectPersonBox(sourceId,image,roi,timestampMs),detectorMs=performance.now()-detectorStart;
      if(!bbox)return {schemaVersion:1,sourceId,timestampMs,timeUncertaintyMs,imageSize:{width:image.width,height:image.height},bbox:[0,0,image.width,image.height],keypoints:Array.from({length:133},(_,id)=>({id,x:0,y:0,score:0,rawScore:0,observable:false,source:'missing'})),refinements:[],personDetection,modelId:this.bodyId,modelHash:this.sessions.get(this.bodyId).spec.actualHash,backend:this.backend,timings:{...this.frameTiming,neuralFrame:this.frameTiming.modelRuns>0,detectorMs,bodyMs:0,handsMs:0,feetMs:0,totalMs:performance.now()-started},depthKind:'none'};
      const bodyStart=performance.now();
      const points=await this.run(this.bodyId,image,bbox),bodyMs=performance.now()-bodyStart;
      const handsStart=performance.now(),refinements=[];
      if(handRefinement){
        for(const [side,offset] of [['left',91],['right',112]]){
          const handBox=predictedHandBox(points,side,bbox);if(!handBox){refinements.push({side,accepted:false,reason:'no_roi'});continue;}
          const hand=await this.run('hand5',image,handBox),decision=handRefinementDecision(points,hand,handBox,side),{accepted}=decision;
          if(accepted){for(let j=0;j<21;j++){const previous=points[offset+j];points[offset+j]={...previous,...hand[j],id:offset+j,source:'handRefined2D',relativeZ:previous.relativeZ,relativeZScore:previous.relativeZScore};}}
          refinements.push({side,roi:handBox,...decision});
        }
      }
      const handsMs=performance.now()-handsStart,feetStart=performance.now();
      const footRefinements=footRefinement?await this.refineFeet(image,bbox,points):null;
      const feetMs=performance.now()-feetStart;
      for(const point of points){point.observable=point.observable&&point.x>=0&&point.y>=0&&point.x<image.width&&point.y<image.height;}
      this.updateTracker(sourceId,points,image,timestampMs);
      if(this.trackers.has(sourceId))this.trackers.get(sourceId).candidateCount=personDetection.candidateCount;
      return {schemaVersion:1,sourceId,timestampMs,timeUncertaintyMs,imageSize:{width:image.width,height:image.height},bbox,keypoints:points,refinements,footRefinements,personDetection,inputMode:'wholebody',scoreKind:'uncalibratedClippedModelPeak',modelId:this.bodyId,modelHash:this.sessions.get(this.bodyId).spec.actualHash,backend:this.backend,timings:{...this.frameTiming,neuralFrame:true,detectorMs,bodyMs,handsMs,feetMs,totalMs:performance.now()-started},depthKind:this.sessions.get(this.bodyId).spec.dimensions===3?'rootRelativeModelEstimate':'none'};
    }finally{this.busy=false;}
  }
  async dispose(){this.abort.abort();for(const tracker of this.footTrackers.values())tracker.reset();await this.tapirMaster?.dispose();this.tapirMaster=null;for(const {session} of this.sessions.values())await session.release();this.sessions.clear();this.trackers.clear();this.footTrackers.clear();this.seedVersions.clear();this.footDetectionTimes.clear();this.manualFootSources.clear();}
  resetSource(sourceId){this.trackers.delete(sourceId);this.footTrackers.get(sourceId)?.reset();this.footTrackers.delete(sourceId);this.seedVersions.delete(sourceId);this.footDetectionTimes.delete(sourceId);this.manualFootSources.delete(sourceId);}
}
