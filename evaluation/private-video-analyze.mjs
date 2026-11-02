#!/usr/bin/env node
// Explicitly local private-video analysis. This CLI has no upload operation.
// Inputs, annotations, frames, and raw traces must remain outside this checkout.
// Runtime/model dependencies are read from local paths before fetch is disabled.
import {readFile,writeFile,mkdir,realpath} from 'node:fs/promises';
import {spawn,execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {resolve,relative,dirname,sep} from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {FootTracker,FOOT_TRACK_POLICY} from '../dist/foot-track.mjs';
import {footTransform,footTensor,decodeFootCandidates} from '../dist/foot-codec.mjs';
import {MODEL_SPECS} from '../dist/models.mjs';

const args={};for(let i=2;i<process.argv.length;i++){
  const key=process.argv[i];if(!key.startsWith('--'))throw new Error('Use named command-line arguments.');
  if(key==='--flow-only'){args.flowOnly=true;continue;}
  args[key.slice(2)]=process.argv[++i];
}
if(!args.video||!args['output-dir']||!args['private-root'])throw new Error('Required: --video PATH --output-dir PATH --private-root PATH; optional --annotations PATH --runtime-dir PATH --model PATH --start SECONDS --duration SECONDS --stride N --flow-only.');
const checkout=await realpath(resolve(dirname(fileURLToPath(import.meta.url)),'..'));
await mkdir(resolve(args['private-root']),{recursive:true});const privateRoot=await realpath(resolve(args['private-root']));
await mkdir(resolve(args['output-dir']),{recursive:true});const outputDir=await realpath(resolve(args['output-dir']));
const inside=(path,root)=>{const r=relative(root,path);return r===''||(!r.startsWith('..'+sep)&&r!=='..'&&!resolve(r).startsWith(sep+'..'+sep)&&!r.startsWith(sep));};
if(inside(privateRoot,checkout)||inside(outputDir,checkout)||!inside(outputDir,privateRoot)||!privateRoot.split(sep).includes('private-phase3'))throw new Error('Private derivatives must be inside an explicit private-phase3 directory outside the source checkout.');
const video=await realpath(resolve(args.video));
const metadata=JSON.parse((await promisify(execFile)('ffprobe',['-v','error','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_frames,duration','-show_entries','format=duration','-of','json',video],{encoding:'utf8'})).stdout);
const stream=metadata.streams[0],[numerator,denominator]=stream.avg_frame_rate.split('/').map(Number),fps=numerator/denominator;
if(!(fps>0)||!stream.width||!stream.height)throw new Error('Video geometry and a finite frame rate are required.');
const startFrame=Math.round(Number(args.start||0)*fps),duration=Number(args.duration||Math.min(Number(stream.duration||metadata.format?.duration||5),5)),stride=Math.max(1,Number(args.stride||1));
if(!Number.isInteger(stride)||!Number.isFinite(duration)||duration<=0)throw new Error('Invalid duration or stride.');
const annotations=args.annotations?JSON.parse(await readFile(resolve(args.annotations),'utf8')):{seeds:[],references:[]};
if(args.annotations){const path=await realpath(resolve(args.annotations));if(!inside(path,privateRoot))throw new Error('Manual references must remain inside the private analysis root.');}
const seeds=new Map((annotations.seeds||[]).map(x=>[x.frame,x])),references=new Map((annotations.references||[]).map(x=>[x.frame,x]));
const tracker=new FootTracker(),records=[],inferenceTimes=[],flowTimes=[],baselineScores=[];
let session=null,ort=null,modelHash=null;const spec=MODEL_SPECS['focus-foot8-one2many'];
if(!args.flowOnly){
  if(!args.model||!args['runtime-dir'])throw new Error('Provide local --model and --runtime-dir, or use --flow-only.');
  const runtime=await realpath(resolve(args['runtime-dir']));
  ort=await import(pathToFileURL(resolve(runtime,'ort.webgpu.min.mjs')).href);
  ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=pathToFileURL(runtime+sep).href;
  ort.env.wasm.wasmBinary=new Uint8Array(await readFile(resolve(runtime,'ort-wasm-simd-threaded.jsep.wasm')));
  const model=new Uint8Array(await readFile(resolve(args.model)));modelHash=createHash('sha256').update(model).digest('hex');
  if(modelHash!==(args.sha256||spec.sha256))throw new Error('The local model does not match its pinned public SHA-256.');
  globalThis.fetch=async()=>{throw new Error('Network is disabled during private video analysis.');};
  session=await ort.InferenceSession.create(model,{executionProviders:['wasm']});
}else globalThis.fetch=async()=>{throw new Error('Network is disabled during private video analysis.');};

const ffmpegArgs=['-hide_banner','-loglevel','error','-ss',String(startFrame/fps),'-i',video,'-an','-t',String(duration),'-vf',`select=not(mod(n\\,${stride}))`,'-vsync','0','-pix_fmt','rgba','-f','rawvideo','pipe:1'];
const ffmpeg=spawn('ffmpeg',ffmpegArgs,{stdio:['ignore','pipe','pipe']});let stderr='';ffmpeg.stderr.on('data',chunk=>stderr+=chunk.toString());
const ffmpegDone=new Promise((resolvePromise,reject)=>{ffmpeg.once('error',reject);ffmpeg.once('close',code=>code===0?resolvePromise():reject(new Error('ffmpeg failed: '+stderr.slice(-1200))));});
const frameBytes=stream.width*stream.height*4;let pending=Buffer.alloc(0),decoded=0,lastBaseline=-Infinity,lastSeed=null;
const started=performance.now();
for await(const chunk of ffmpeg.stdout){
  pending=pending.length?Buffer.concat([pending,chunk]):chunk;
  while(pending.length>=frameBytes){
    const data=new Uint8Array(pending.subarray(0,frameBytes));pending=pending.subarray(frameBytes);
    const frameIndex=startFrame+decoded*stride,timestampMs=frameIndex/fps*1000,image={width:stream.width,height:stream.height,data};decoded++;
    const seed=seeds.get(frameIndex),reference=references.get(frameIndex);
    if(seed){tracker.seed({points:seed.points,side:seed.side||'unknown',timestampMs,image});lastSeed=seed;}
    let candidates,baseline=null,inferenceMs=null,maxDetectionScore=null;
    const modelNeeded=session&&(tracker.needsDetection(timestampMs)||reference||timestampMs-lastBaseline>=Number(args['baseline-every-ms']||500));
    if(modelNeeded){
      const transform=footTransform(image,[0,0,image.width,image.height]),input=footTensor(image,transform),modelStarted=performance.now();
      const outputs=await session.run({[session.inputNames[0]]:new ort.Tensor('float32',input,[1,3,640,640])}),output=Object.values(outputs)[0];
      inferenceMs=performance.now()-modelStarted;inferenceTimes.push(inferenceMs);candidates=decodeFootCandidates(output,spec,transform);
      maxDetectionScore=0;for(let i=0;i<output.dims[2];i++)maxDetectionScore=Math.max(maxDetectionScore,output.data[4*output.dims[2]+i]);
      baselineScores.push(maxDetectionScore);const independent=new FootTracker();baseline=independent.update({image,timestampMs,candidates});lastBaseline=timestampMs;
    }
    const prediction=tracker.update({image,timestampMs,candidates:tracker.needsDetection(timestampMs)?candidates:undefined});flowTimes.push(prediction.diagnostics.processingMs);
    records.push({frame:frameIndex,timestampMs,seeded:!!seed,reference:reference??null,prediction,baseline,seedHoldBaseline:lastSeed?.points??null,inferenceMs,maxDetectionScore});
  }
}
await ffmpegDone;if(pending.length)throw new Error('Incomplete RGBA frame from ffmpeg.');
const percentile=(values,p)=>{if(!values.length)return null;const a=[...values].sort((x,y)=>x-y),index=p*(a.length-1),low=Math.floor(index),high=Math.ceil(index);return a[low]+(a[high]-a[low])*(index-low);};
const average=values=>values.length?values.reduce((a,b)=>a+b,0)/values.length:null;
function metrics(kind){
  let count=0,missing=0,within10px=0,withinFoot10pct=0,frames=0,allThreeWithin10pct=0;const errors=[];
  for(const record of records){
    if(!record.reference||record.seeded)continue;frames++;
    const ref=record.reference,visible=ref.visible||[true,true,true],points=ref.points;
    const length=Math.hypot((points[0][0]+points[1][0])*.5-points[2][0],(points[0][1]+points[1][1])*.5-points[2][1]);
    let predicted=null;
    if(kind==='seedHold')predicted=record.seedHoldBaseline;
    else{
      const result=kind==='flow'?record.prediction:record.baseline;
      const present=(result?.footTracks||[]).filter(t=>t.state!=='missing');
      // Diagnostic matching only: it never chooses runtime detections or changes
      // a tracker. A named track can be used to avoid this optimistic association.
      const track=ref.trackId?present.find(t=>t.trackId===ref.trackId):present.sort((a,b)=>{
        const err=t=>[t.bigToe,t.littleToe,t.heel].reduce((sum,p,i)=>sum+(p?Math.hypot(p[0]-points[i][0],p[1]-points[i][1]):1e6),0);return err(a)-err(b);
      })[0];
      if(track)predicted=[track.bigToe,track.littleToe,track.heel];
    }
    let all=true;
    for(let i=0;i<3;i++){
      if(!visible[i])continue;count++;
      const p=predicted?.[i];if(!p||!Number.isFinite(p[0]+p[1])){missing++;all=false;continue;}
      const error=Math.hypot(p[0]-points[i][0],p[1]-points[i][1]);errors.push(error);if(error<=10)within10px++;if(error<=length*.1)withinFoot10pct++;else all=false;
    }
    if(all)allThreeWithin10pct++;
  }
  return {referenceFrames:frames,visibleReferencePoints:count,predictedPoints:errors.length,missingPoints:missing,meanErrorPx:average(errors),medianErrorPx:percentile(errors,.5),p95ErrorPx:percentile(errors,.95),within10px:count?within10px/count:null,pckFootLength10pct:count?withinFoot10pct/count:null,allThreeWithinFootLength10pctFrames:allThreeWithin10pct};
}
const summary={schemaVersion:1,privacy:'Local only: never upload frames, annotations, or raw traces.',policy:FOOT_TRACK_POLICY,model:{id:spec.id,sha256:modelHash,provider:session?'wasm-single-thread':null},video:{width:stream.width,height:stream.height,fps},analysis:{processedFrames:records.length,elapsedSeconds:(performance.now()-started)/1000,manualSeedCount:records.filter(r=>r.seeded).length,referenceFramesExcludingSeeds:records.filter(r=>r.reference&&!r.seeded).length,manualReferenceUncertaintyPx:annotations.uncertaintyPx??null,manualReferenceMeaning:annotations.meaning??null,modelRuns:inferenceTimes.length,modelInferenceMs:{mean:average(inferenceTimes),median:percentile(inferenceTimes,.5),p95:percentile(inferenceTimes,.95)},flowProcessingMs:{mean:average(flowTimes),median:percentile(flowTimes,.5),p95:percentile(flowTimes,.95)},modelDetectionThreshold:.25,modelFramesPassingDetectionThreshold:baselineScores.filter(x=>x>=.25).length,flowFramesWithImageSupport:records.filter(r=>r.prediction.footTracks.some(t=>t.source==='opticalFlow2D')).length},metrics:{uninitializedModel:metrics('model'),stationarySeed:metrics('seedHold'),manualSeedAndOpticalFlow:metrics('flow')},limitations:['Diagnostic nearest-reference matching is optimistic when multiple model tracks exist.','Manual seeds and references are not an independent benchmark. Seed frames are excluded from position metrics.','Missing output remains a PCK failure; finite-only mean error is reported with its missing count.','These are 2D image coordinates; depth, hidden joints, and anatomical side for an unknown seed are unobserved.','Processing timings include Node WASM and this host only, not browser or 60 Hz display performance.']};
await writeFile(resolve(outputDir,'trace-private.json'),JSON.stringify({metadata,annotations,records}));
await writeFile(resolve(outputDir,'summary-private.json'),JSON.stringify(summary,null,2));
console.log(JSON.stringify(summary,null,2));
