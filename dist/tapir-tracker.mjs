import {tapirFrameTensor,tapirQueryTensor,decodeTapir} from './tapir-codec.mjs';

export const TAPIR_TRACK_POLICY=Object.freeze({id:'causal-bootstapir-three-points-v1',queryCount:3,inputWH:[256,256],visibilityThreshold:.5,maxGapMs:2000});
export const TAPIR_GPU_OUTPUT_LOCATIONS=Object.freeze({
  encoder:Object.freeze({lowres:'gpu-buffer',hires:'gpu-buffer'}),
  query:Object.freeze({query_lowres:'gpu-buffer',query_hires:'gpu-buffer'}),
  step:Object.freeze({tracks:'cpu',occlusion:'cpu',expected_dist:'cpu',new_causal_state:'gpu-buffer'}),
});
const IDS={left:[17,18,19],right:[20,21,22]};
const missingKeypoints=()=>Array.from({length:133},(_,id)=>({id,x:NaN,y:NaN,score:0,rawScore:0,observable:false,source:'missing'}));
const finitePoint=p=>Array.isArray(p)&&p.length===2&&p.every(Number.isFinite);
const inImage=(p,image)=>finitePoint(p)&&p[0]>=0&&p[1]>=0&&p[0]<image.width&&p[1]<image.height;
const product=shape=>shape.reduce((a,b)=>a*b,1);
const dimsEqual=(a,b)=>Array.isArray(a)&&a.length===b.length&&a.every((x,i)=>x===b[i]);
function assertImage(image){if(!Number.isInteger(image?.width)||!Number.isInteger(image?.height)||image.width<2||image.height<2||image.data?.length!==image.width*image.height*4)throw new Error('Learned foot tracking requires a complete RGBA frame.');}
function croppedImage(image,box){
  if(!box)return image;const [x,y,width,height]=box,data=new Uint8Array(width*height*4);
  for(let row=0;row<height;row++){const start=((y+row)*image.width+x)*4;data.set(image.data.subarray(start,start+width*4),row*width*4);}
  return {width,height,data};
}
function fixedCrop(image,roi){
  if(!roi)return null;if(!Array.isArray(roi)||roi.length!==4||!roi.every(Number.isFinite)||roi[2]<=0||roi[3]<=0)throw new Error('A fixed tracking region must be [x,y,width,height] in source pixels.');
  const x=Math.max(0,Math.floor(roi[0])),y=Math.max(0,Math.floor(roi[1])),right=Math.min(image.width,Math.ceil(roi[0]+roi[2])),bottom=Math.min(image.height,Math.ceil(roi[1]+roi[3]));
  if(right-x<2||bottom-y<2)throw new Error('Tracking region is outside the image.');return [x,y,right-x,bottom-y];
}
function tensorPresent(outputs,name){
  const tensor=outputs[name];
  if(!tensor||tensor.type!=='float32'||!Array.isArray(tensor.dims)||tensor.dims.some(x=>!Number.isInteger(x)||x<1)||tensor.size!==product(tensor.dims))throw new Error(`Invalid learned-tracker tensor metadata: ${name}`);
  if(tensor.location==='gpu-buffer')return tensor; // .data would force/require a CPU readback.
  if(!['cpu','cpu-pinned'].includes(tensor.location)||!tensor.data||tensor.data.length!==tensor.size)throw new Error(`Invalid learned-tracker tensor storage: ${name}`);
  return tensor;
}
const disposedTensors=new WeakSet();
function dispose(tensor){if(tensor&&typeof tensor.dispose==='function'&&!disposedTensors.has(tensor)){disposedTensors.add(tensor);tensor.dispose();}}
function disposeOutputs(outputs,keep=[]){if(outputs)for(const [name,tensor] of Object.entries(outputs))if(!keep.includes(name))dispose(tensor);}

/** Stateful online point tracker, initialized by three explicitly selected points.
 * This is not a foot detector. It cannot assign toe semantics before seeding.
 * All inference runs in the supplied local ORT sessions. No network operation is
 * performed by seed/update, and no frame or trace is transmitted by this module.
 */
export class TapirTracker{
  constructor({ort,sessions,manifest,ownsSessions=false,policy={}}){
    if(!ort?.Tensor||!sessions?.encoder||!sessions?.query||!sessions?.step)throw new Error('Encoder, query, and causal-step sessions are required.');
    if(manifest?.queryCount!==3||!dimsEqual(manifest?.inputWH,[256,256])||manifest?.stateType!=='float32'||!Array.isArray(manifest?.stateShape)||!manifest.stateShape.length||manifest.stateShape.some(n=>!Number.isInteger(n)||n<=0))throw new Error('Unsupported learned foot tracker manifest.');
    this.ort=ort;this.sessions=sessions;this.manifest=manifest;this.ownsSessions=ownsSessions;this.policy={...TAPIR_TRACK_POLICY,...policy};this.epoch=0;this.busy=false;this.deferredDisposals=[];this.reset();
  }
  static async create({ort,modelBuffers,manifest,executionProviders=['wasm'],sessionOptions={},...options}){
    const sessions={},useGpu=executionProviders.some(provider=>(typeof provider==='string'?provider:provider.name)==='webgpu');
    try{
      for(const id of ['encoder','query','step']){
        const record=manifest?.models?.find(x=>x.id===id),value=modelBuffers?.[id];
        if(!record||!value)throw new Error(`Missing local learned-tracker graph: ${id}`);
        const bytes=value instanceof Uint8Array?value:new Uint8Array(value);
        if(record.bytes&&record.bytes!==bytes.byteLength)throw new Error(`Unexpected learned-tracker size: ${id}`);
        if(!/^[a-f0-9]{64}$/i.test(record.sha256||''))throw new Error(`A pinned SHA-256 is required: ${id}`);
        const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
        if(hash!==record.sha256.toLowerCase())throw new Error(`Learned-tracker checksum mismatch: ${id}`);
        sessions[id]=await ort.InferenceSession.create(bytes,{...sessionOptions,executionProviders,enableGraphCapture:false,...(useGpu?{preferredOutputLocation:TAPIR_GPU_OUTPUT_LOCATIONS[id]}:{})});
      }
      return new TapirTracker({ort,sessions,manifest,ownsSessions:true,...options});
    }catch(error){await Promise.allSettled(Object.values(sessions).map(s=>s.release?.()));throw error;}
  }
  reset(){
    this.epoch++;for(const tensor of [this.queryLowres,this.queryHires,this.causalState])if(tensor){if(this.busy)this.deferredDisposals.push(tensor);else dispose(tensor);}
    this.queryLowres=null;this.queryHires=null;this.causalState=null;this.initialized=false;this.originalSize=null;this.crop=null;this.lastTimestampMs=null;this.trackId=null;this.side='unknown';this.slot='left';this.seedSource='manual';this.semanticSeedScores=null;this.modelFrames=0;this.lastResult={keypoints:missingKeypoints(),footTracks:[],diagnostics:{trackingModel:'causal-bootstapir',requiresSeed:true,depthObserved:false},depthKind:'none'};
  }
  needsDetection(){return false;}
  get requiresSeed(){return !this.initialized;}
  async seed({points,side='unknown',timestampMs,image,roi=null,seedSource='manual',seedScores=null}){
    if(this.busy)throw new Error('Wait for the current learned-tracker frame before selecting new seed points.');
    assertImage(image);if(!Number.isFinite(timestampMs)||!Array.isArray(points)||points.length!==3||!points.every(p=>inImage(p,image)))throw new Error('Select three visible source-image points: big toe, little toe, heel.');
    if(!['left','right','unknown'].includes(side))throw new Error('Foot side must be left, right, or unknown.');
    if(!['manual','detector'].includes(seedSource))throw new Error('Seed source must be manual or detector.');
    if(seedScores!==null&&(!Array.isArray(seedScores)||seedScores.length!==3||seedScores.some(x=>!Number.isFinite(x)||x<0||x>1)))throw new Error('Optional semantic seed scores must contain three values in [0,1].');
    if(points.some((p,i)=>points.some((q,j)=>i!==j&&Math.hypot(p[0]-q[0],p[1]-q[1])<2)))throw new Error('Seed points must be distinct.');
    this.reset();this.busy=true;const epoch=this.epoch,started=performance.now();let encoderOutputs=null,queryOutputs=null,queryTensor=null,frameTensor=null;
    try{
      this.crop=fixedCrop(image,roi);const cropped=croppedImage(image,this.crop),offset=this.crop||[0,0],local=points.map(p=>[p[0]-offset[0],p[1]-offset[1]]);
      if(!local.every(p=>inImage(p,cropped)))throw new Error('All selected points must lie inside the fixed tracking region.');
      this.originalSize=[image.width,image.height];this.side=side;this.slot=side==='right'?'right':'left';this.trackId=`learned-foot-${epoch}`;this.seedSource=seedSource;this.semanticSeedScores=seedScores;
      frameTensor=new this.ort.Tensor('float32',tapirFrameTensor(cropped),[1,1,256,256,3]);let stageStarted=performance.now();
      encoderOutputs=await this.sessions.encoder.run({frame:frameTensor});const encoderMs=performance.now()-stageStarted;
      if(epoch!==this.epoch)return this.lastResult;
      const lowres=tensorPresent(encoderOutputs,'lowres'),hires=tensorPresent(encoderOutputs,'hires');
      queryTensor=new this.ort.Tensor('float32',tapirQueryTensor(local,cropped),[1,3,2]);stageStarted=performance.now();
      queryOutputs=await this.sessions.query.run({lowres,hires,query_yx:queryTensor});const queryMs=performance.now()-stageStarted;
      if(epoch!==this.epoch)return this.lastResult;
      this.queryLowres=tensorPresent(queryOutputs,'query_lowres');this.queryHires=tensorPresent(queryOutputs,'query_hires');
      this.causalState=new this.ort.Tensor('float32',new Float32Array(product(this.manifest.stateShape)),this.manifest.stateShape);
      const output=await this.runStep({lowres,hires,cropped,image,timestampMs,epoch});if(!output)return this.lastResult;
      this.initialized=true;this.lastTimestampMs=timestampMs;
      // The selected coordinates are user observations. The seed frame's step
      // updates temporal state but is excluded from tracking accuracy metrics.
      const source=seedSource==='manual'?'manualSeed2D':'modelSeed2D',seeded=points.map((p,i)=>({x:p[0],y:p[1],score:seedSource==='manual'?1:seedScores?.[i]??output.points[i].score,observable:seedSource==='manual'||output.points[i].observable,source}));
      return this.publish(seeded,timestampMs,{seeded:true,encoderMs,queryMs,stepMs:output.stepMs,inferenceMs:encoderMs+queryMs+output.stepMs,processingMs:performance.now()-started},source);
    }catch(error){if(epoch===this.epoch)this.reset();throw error;}
    finally{dispose(frameTensor);dispose(queryTensor);disposeOutputs(encoderOutputs);disposeOutputs(queryOutputs,Object.keys(queryOutputs||{}).filter(name=>queryOutputs[name]===this.queryLowres||queryOutputs[name]===this.queryHires));this.busy=false;this.flushDeferred();}
  }
  async update({image,timestampMs}){
    if(this.busy)throw new Error('Learned tracking frames must be processed sequentially.');
    assertImage(image);if(!Number.isFinite(timestampMs))throw new Error('A finite source-frame timestamp is required.');
    if(!this.initialized)return this.missing(timestampMs,'requires-three-point-seed');
    const dt=timestampMs-this.lastTimestampMs;
    if(this.originalSize[0]!==image.width||this.originalSize[1]!==image.height||dt<0||dt>this.policy.maxGapMs){this.reset();return this.missing(timestampMs,dt<0?'seek-backward':dt>this.policy.maxGapMs?'timestamp-discontinuity':'image-size-change');}
    if(dt===0)return {...this.lastResult,diagnostics:{...this.lastResult.diagnostics,modelRan:false,modelRuns:0,encoderMs:0,queryMs:0,stepMs:0,inferenceMs:0,processingMs:0,repeatFrame:true,seeded:false}};
    this.busy=true;const epoch=this.epoch,started=performance.now();let encoderOutputs=null,frameTensor=null;
    try{
      const cropped=croppedImage(image,this.crop);frameTensor=new this.ort.Tensor('float32',tapirFrameTensor(cropped),[1,1,256,256,3]);const stageStarted=performance.now();
      encoderOutputs=await this.sessions.encoder.run({frame:frameTensor});const encoderMs=performance.now()-stageStarted;
      if(epoch!==this.epoch)return this.lastResult;
      const output=await this.runStep({lowres:tensorPresent(encoderOutputs,'lowres'),hires:tensorPresent(encoderOutputs,'hires'),cropped,image,timestampMs,epoch});
      if(!output)return this.lastResult;this.lastTimestampMs=timestampMs;
      return this.publish(output.points,timestampMs,{seeded:false,encoderMs,queryMs:0,stepMs:output.stepMs,inferenceMs:encoderMs+output.stepMs,processingMs:performance.now()-started});
    }catch(error){if(epoch===this.epoch)this.reset();throw error;}
    finally{dispose(frameTensor);disposeOutputs(encoderOutputs);this.busy=false;this.flushDeferred();}
  }
  async runStep({lowres,hires,cropped,image,epoch}){
    const started=performance.now(),outputs=await this.sessions.step.run({lowres,hires,query_lowres:this.queryLowres,query_hires:this.queryHires,causal_state:this.causalState});
    if(epoch!==this.epoch){disposeOutputs(outputs);return null;}
    let retainedState=false;try{
      const state=tensorPresent(outputs,'new_causal_state');if(!dimsEqual(state.dims,this.manifest.stateShape))throw new Error('Causal state shape changed during tracking.');
      const tracks=tensorPresent(outputs,'tracks'),occlusion=tensorPresent(outputs,'occlusion'),expected_dist=tensorPresent(outputs,'expected_dist');
      if(!dimsEqual(tracks.dims,[1,3,1,2])||!dimsEqual(occlusion.dims,[1,3,1])||!dimsEqual(expected_dist.dims,[1,3,1]))throw new Error('Unexpected learned point-tracker output layout.');
      if([tracks,occlusion,expected_dist].some(tensor=>!['cpu','cpu-pinned'].includes(tensor.location)))throw new Error('Final learned point outputs must be on CPU; use the per-graph output-location contract.');
      if(!tracks.data.every(Number.isFinite)||!occlusion.data.every(Number.isFinite)||!expected_dist.data.every(Number.isFinite))throw new Error('Non-finite learned tracking output.');
      const offset=this.crop||[0,0],decoded=decodeTapir({tracks,occlusion,expected_dist},cropped),points=decoded.map(p=>{
        const x=p.x+offset[0],y=p.y+offset[1],observable=!!p.observable&&p.score>this.policy.visibilityThreshold&&inImage([p.x,p.y],cropped)&&inImage([x,y],image);
        return {...p,x,y,observable,source:observable?'learnedPointTrack2D':'missing'};
      });
      dispose(this.causalState);this.causalState=state;retainedState=true;this.modelFrames++;return {points,stepMs:performance.now()-started};
    }finally{disposeOutputs(outputs,retainedState?['new_causal_state']:[]);}
  }
  publish(points,timestampMs,timing,source='learnedPointTrack2D'){
    const confirmed=this.seedSource==='manual'&&this.side!=='unknown',anatomicalSide=confirmed?this.side:'unknown',modelSide=this.seedSource==='detector'&&this.side!=='unknown'?this.side:null;
    const keypoints=missingKeypoints(),named=['bigToe','littleToe','heel'],track={trackId:this.trackId,slot:this.slot,anatomicalSide,anatomicalSideConfirmed:confirmed,modelSide,ankle:null,seededManually:this.seedSource==='manual',semanticSeedSource:this.seedSource,semanticSeedScores:this.semanticSeedScores,source,depthObserved:false,imageSize:[...this.originalSize],timestampMs};
    const visibleScores=[];
    for(let i=0;i<3;i++){
      const p=points[i],id=IDS[this.slot][i],visible=p.observable===true;track[named[i]]=visible?[p.x,p.y]:null;
      if(visible){visibleScores.push(p.score);keypoints[id]={id,x:p.x,y:p.y,score:p.score,rawScore:p.score,observable:true,source,trackId:this.trackId,anatomicalSide,anatomicalSideConfirmed:confirmed,modelSide,semanticSeedSource:this.seedSource,depthObserved:false,rawOcclusion:p.rawOcclusion??null,rawExpectedDist:p.rawExpectedDist??null};}
    }
    track.visiblePointCount=visibleScores.length;track.state=visibleScores.length?'observed2D':'missing';track.score=visibleScores.length?Math.min(...visibleScores):0;track.lossReason=visibleScores.length===3?null:'model-visibility-or-uncertainty';
    track.pointVisibility=points.map(p=>({visible:p.observable===true,score:p.score,occlusionLogit:p.rawOcclusion??null,expectedDistanceLogit:p.rawExpectedDist??null}));
    const diagnostics={policyId:this.policy.id,trackingModel:'causal-bootstapir',modelFrames:this.modelFrames,modelRuns:timing.seeded?3:2,modelRan:true,flowFrames:0,requiresSeed:false,queryCount:3,visiblePointCount:visibleScores.length,visibilityThreshold:this.policy.visibilityThreshold,visibilityScoreCalibrated:false,causalStateValues:product(this.manifest.stateShape),causalStateLocation:this.causalState?.location??'none',queryFeatureLocation:this.queryLowres?.location??'none',fixedRoi:this.crop,depthObserved:false,...timing};
    this.lastResult={keypoints,footTracks:[track],diagnostics,depthKind:'none',timestampMs};return this.lastResult;
  }
  missing(timestampMs,reason){
    this.lastResult={keypoints:missingKeypoints(),footTracks:[],diagnostics:{policyId:this.policy.id,trackingModel:'causal-bootstapir',modelRan:false,modelRuns:0,modelFrames:this.modelFrames,encoderMs:0,queryMs:0,stepMs:0,inferenceMs:0,processingMs:0,requiresSeed:!this.initialized,reason,depthObserved:false},depthKind:'none',timestampMs};return this.lastResult;
  }
  async dispose(){this.reset();if(this.ownsSessions)await Promise.allSettled(Object.values(this.sessions).map(session=>session.release?.()));}
  flushDeferred(){for(const tensor of this.deferredDisposals)dispose(tensor);this.deferredDisposals=[];}
}
