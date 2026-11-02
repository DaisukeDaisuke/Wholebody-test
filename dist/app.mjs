import * as ort from './vendor/ort.webgpu.min.mjs';
import { COCO59, COMPACT_EDGES, cropTransform, rgbaToTensor, decodeSimCC, toCoco133, percentile } from './compact-codec.mjs';
import { PartModels, faceRoiFrom59, footRoiFrom59 } from './parts.mjs';
import { estimateMonocularPose, estimateLearnedMonocularPose, fusePose } from './geometry.mjs';
import { validateCalibration } from './calibration.mjs';
import { BridgeClient } from './bridge-client.mjs';
import { PoseStabilizer } from './pose-stability.mjs';
import { createPose3DCrop, pose3dCanvasTransform, pose3dRgbaToTensor, decodePose3D, mergePose3DKeypoints } from './pose3d.mjs';

const $=id=>document.getElementById(id), finite=v=>Number.isFinite(v), clamp=(x,a=0,b=1)=>Math.min(b,Math.max(a,x));
ort.env.wasm.wasmPaths=new URL('./vendor/',import.meta.url).href;
ort.env.wasm.numThreads=1;
ort.env.logLevel='warning';
let manifest=null,body=null,body3D=null,parts=null,running=false,loading=false,generation=0,selected=null,view=null,viewPromise=null;
let cameraCalibration=[],neutralFace={},lastFace=null,lastPose=null,sourceCounter=0,frameCounter=0,lastMetricsAt=performance.now();
let actualBackend='—',lastBodyMs=null,bodyCalls=0,activeBody=null,inFlight=null,sourceSync={};
const sources=[],timings=[],bodyTimes=[],partTimes=[],readTimes=[];
const preprocess=document.createElement('canvas');preprocess.width=192;preprocess.height=256;
const preprocessContext=preprocess.getContext('2d',{willReadFrequently:true});
const preprocess3D=document.createElement('canvas');preprocess3D.width=preprocess3D.height=256;
const preprocess3DContext=preprocess3D.getContext('2d',{willReadFrequently:true});
const bridge=new BridgeClient({onStatus:s=>{$('bridge-status').textContent=({connected:'接続済み。校正データを読み込んでください。',calibrated:'校正済み。新しい有効姿勢を送出します。',stale:'観測が古くなったためトラッカーを無効化しました。',disconnected:'未接続',connecting:'接続中…'}[s.state])||s.reason||(s.udpPacketsSent!=null?`送信 ${s.udpPacketsSent} パケット・有効 ${s.enabledTrackers} 台` : s.state);}});
function notify(message,error=false){$('notice').textContent=message;$('notice').classList.toggle('error',error);}
function status(message){$('model-status').textContent=message;}
function fail(error){notify(error?.message||String(error),true);console.error(error);}
function limitedPush(a,v){if(finite(v)){a.push(v);if(a.length>600)a.shift();}}
function resetReadouts(){for(const id of ['eye-left','eye-right','mouth','cheek']){$(id).value=0;$(id+'-text').textContent='—';}for(const side of ['left','right']){$(side+'-foot-state').textContent='未観測';$(side+'-foot-score').textContent='対象の足領域が必要です';}lastFace=null;view?.updateExpressions?.(null);}
async function sha256(buffer){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))).map(x=>x.toString(16).padStart(2,'0')).join('');}
async function modelBytes(url,hash){
  const target=new URL(url,location.href);let cache=null;
  try{cache=await caches.open('vrc-compact-models-v1');}catch{}
  const response=cache?await cache.match(target.href):null;
  if(response){const bytes=await response.arrayBuffer();if(!hash||await sha256(bytes)===hash)return bytes;await cache.delete(target.href);}
  const r=await fetch(target,{credentials:target.origin===location.origin?'same-origin':'omit',cache:'no-cache'});if(!r.ok)throw new Error(`モデル取得に失敗しました (${r.status})`);
  const bytes=await r.arrayBuffer();if(hash&&await sha256(bytes)!==hash)throw new Error('モデルのSHA-256が一致しません。');
  if(cache)try{await cache.put(target.href,new Response(bytes));}catch{}
  return bytes;
}
async function prepareModels(){
  if(loading)return;loading=true;$('start').disabled=true;
  try{
    if(!manifest){
      const response=await fetch('./models.json',{cache:'no-cache'});if(!response.ok)throw new Error('学習済みモデルのマニフェストを取得できません。');manifest=await response.json();
      const choice=$('model-choice'),fast=choice.querySelector('option[value="fast"]');
      if(manifest.recommendedBody==='standard'){fast.disabled=true;fast.textContent='軽量候補（精度基準未達）';choice.value=manifest.pose3d?'pose3d':'standard';}
      else if(manifest.recommendedBody==='fast')choice.value='fast';
    }
    const learned3D=$('model-choice').value==='pose3d';
    const requested=$('mode').value==='wholebody'&&!learned3D?$('backend').value:'wasm', spec=manifest.bodies?.[$('model-choice').value]||manifest.body;
    if(actualBackend!==requested||activeBody?.sha256!==spec.sha256){if(body)await body.release();body=null;timings.length=bodyTimes.length=partTimes.length=0;bodyCalls=0;lastBodyMs=null;}
    activeBody=spec;
    actualBackend=requested;
    if(requested==='webgpu'&&!navigator.gpu)throw new Error('このブラウザではWebGPUを使用できません。推論エンジンをCPU / WASMへ切り替えてください。');
    if($('mode').value==='wholebody'&&!body){
      status('59点モデルを取得しています…');const bytes=await modelBytes(spec.url,spec.sha256);
      status(`${requested==='webgpu'?'WebGPU':'CPU'}セッションを準備しています…`);
      body=await ort.InferenceSession.create(bytes,{executionProviders:[requested],graphOptimizationLevel:'all'});
      const names=spec.outputs.map(x=>x.name);if(!names.every(n=>body.outputNames.includes(n)))throw new Error('モデルの出力名が一致しません。');
      $('model-detail').textContent=`${spec.id} · ${(spec.bytes/1048576).toFixed(2)} MB · 入力${spec.input.shape[3]}×${spec.input.shape[2]} · 59点`;
    }
    if($('mode').value==='wholebody'&&learned3D){
      if(!manifest.pose3d)throw new Error('3D身体モデルが配布されていません。');
      if(!body3D){status('3D身体モデルを取得しています…');body3D=await ort.InferenceSession.create(await modelBytes(manifest.pose3d.url,manifest.pose3d.sha256),{executionProviders:['wasm'],graphOptimizationLevel:'all'});}
      $('model-detail').textContent=`3D身体 ${(manifest.pose3d.bytes/1048576).toFixed(2)} MB · 15 Hz上限 · 手の詳細 5 Hz`;
    }
    if(!parts)parts=new PartModels(ort,manifest.parts,{executionProviders:['wasm']});
    const mode=$('mode').value;
    if(mode==='face'){status('表情モデルを取得しています…');await parts.load('face');}
    if(mode==='foot'){status('足モデルを取得しています…');await parts.load('foot');}
    status(`${requested==='webgpu'?'WebGPU':'CPU / WASM'} · 準備完了`);return true;
  }catch(e){fail(e);status('モデルの準備に失敗しました。');return false;}
  finally{loading=false;$('start').disabled=false;}
}
function chooseSource(source){selected=source;for(const s of sources)s.root.classList.toggle('selected',s===source);resetReadouts();clearPose();view?.setPaused(!source.stream&&source.video.paused);}
function clearPose(){lastPose=null;bridge.stop();view?.updatePose(null);$('pose-status').textContent='姿勢の観測待ち';}
function removeSource(source){
  const i=sources.indexOf(source);if(i<0)return;sources.splice(i,1);source.stream?.getTracks().forEach(t=>t.stop());source.video.pause();source.video.removeAttribute('src');source.video.srcObject=null;if(source.url)URL.revokeObjectURL(source.url);source.root.remove();
  if(selected===source)selected=sources[0]||null;if(selected)chooseSource(selected);else{stop();clearPose();}
}
function createSource({file,stream,label}){
  if(sources.length>=3){stream?.getTracks().forEach(t=>t.stop());throw new Error('入力は3件までです。不要な入力を閉じてください。');}
  const id=`camera-${++sourceCounter}`,root=document.createElement('section');root.className='source';
  const bar=document.createElement('div');bar.className='source-title';const title=document.createElement('span');title.textContent=`${id} · ${label||file?.name||'カメラ'}`;const close=document.createElement('button');close.textContent='閉じる';bar.append(title,close);
  const media=document.createElement('div');media.className='source-media';const video=document.createElement('video');video.muted=true;video.playsInline=true;video.preload='metadata';const overlay=document.createElement('canvas');media.append(video,overlay);root.append(bar,media);
  const frame=document.createElement('canvas');
  const source={id,root,video,frame,frameContext:frame.getContext('2d'),overlay,context:overlay.getContext('2d'),stream,roi:null,points:null,observation:null,face:null,feet:{},stabilizer:new PoseStabilizer(),lastFace:-Infinity,lastFoot:-Infinity,lastDetails:-Infinity,details:null,decoded3D:null,rawKeypoints:null,lastMediaTime:-Infinity,lastFrameAt:-Infinity,frameCaptureAt:0};
  if(file){source.url=URL.createObjectURL(file);video.src=source.url;const controls=document.createElement('div');controls.className='source-video-controls';const play=document.createElement('button');play.className='secondary';play.textContent='再生';const seek=document.createElement('input');seek.type='range';seek.min=0;seek.max=1;seek.step=.01;seek.value=0;seek.setAttribute('aria-label','動画の再生位置');const time=document.createElement('output');time.textContent='0:00';controls.append(play,seek,time);root.append(controls);play.onclick=()=>video.paused?video.play().catch(fail):video.pause();video.addEventListener('play',()=>play.textContent='停止');video.addEventListener('pause',()=>play.textContent='再生');video.addEventListener('loadedmetadata',()=>seek.max=video.duration);video.addEventListener('timeupdate',()=>{seek.value=video.currentTime;time.textContent=`${Math.floor(video.currentTime/60)}:${String(Math.floor(video.currentTime%60)).padStart(2,'0')}`;});seek.oninput=()=>{video.currentTime=Number(seek.value);source.lastMediaTime=-Infinity;source.observation=null;clearPose();};}
  else video.srcObject=stream;
  video.addEventListener('pause',()=>{if(source===selected)view?.setPaused(true);});
  video.addEventListener('play',()=>{if(source===selected)view?.setPaused(false);});
  video.addEventListener('seeking',()=>{source.stabilizer.reset();source.observation=null;source.lastMediaTime=source.lastDetails=-Infinity;source.details=source.decoded3D=source.rawKeypoints=null;});
  video.addEventListener('loadedmetadata',()=>{overlay.width=video.videoWidth;overlay.height=video.videoHeight;draw(source);if(stream||running)video.play().catch(fail);});
  video.addEventListener('error',()=>fail(new Error('この動画をデコードできません。ブラウザ対応のMP4/WebMを選択してください。')));
  if(video.requestVideoFrameCallback){const mark=(now,meta)=>{source.frameCaptureAt=now;source.presentedMediaTime=meta.mediaTime;if(sources.includes(source))video.requestVideoFrameCallback(mark);};video.requestVideoFrameCallback(mark);}
  close.onclick=()=>removeSource(source);root.onclick=()=>{if(selected!==source)chooseSource(source);};
  const imagePoint=e=>{const r=overlay.getBoundingClientRect(),scale=Math.min(r.width/overlay.width,r.height/overlay.height),w=overlay.width*scale,h=overlay.height*scale;return [clamp((e.clientX-r.left-(r.width-w)/2)/scale,0,overlay.width),clamp((e.clientY-r.top-(r.height-h)/2)/scale,0,overlay.height)];};
  let anchor=null;
  overlay.onpointerdown=e=>{chooseSource(source);anchor=imagePoint(e);overlay.setPointerCapture(e.pointerId);};
  overlay.onpointermove=e=>{if(!anchor)return;const p=imagePoint(e);source.roi=[Math.min(anchor[0],p[0]),Math.min(anchor[1],p[1]),Math.abs(anchor[0]-p[0]),Math.abs(anchor[1]-p[1])];draw(source);};
  overlay.onpointerup=()=>{anchor=null;if(source.roi?.[2]<12||source.roi?.[3]<12)source.roi=null;source.stabilizer.reset();source.observation=null;source.lastMediaTime=-Infinity;source.lastFace=source.lastFoot=-Infinity;clearPose();draw(source);};
  sources.push(source);$('sources').append(root);chooseSource(source);return source;
}
function draw(source){
  const {context:ctx,overlay,points,roi}=source;ctx.clearRect(0,0,overlay.width,overlay.height);const scale=Math.max(1,overlay.width/700),threshold=Number($('confidence').value);
  if(roi){ctx.strokeStyle='#d6f69e';ctx.lineWidth=2*scale;ctx.setLineDash([7*scale,5*scale]);ctx.strokeRect(...roi);ctx.setLineDash([]);}
  if(points&&$('mode').value==='wholebody'){
    ctx.lineWidth=2*scale;ctx.strokeStyle='#9adb8c';
    for(const [a,b] of COMPACT_EDGES){const p=points[a],q=points[b];if(p?.score<threshold||q?.score<threshold)continue;ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();}
    for(const p of points){if(p.score<threshold)continue;ctx.fillStyle=p.id<13?'#d5f69c':p.id<19?'#65d7bf':p.id<49?'#e3b686':'#b7b1ff';ctx.beginPath();ctx.arc(p.x,p.y,2.5*scale,0,Math.PI*2);ctx.fill();}
  }
  if(source.face?.valid&&performance.now()-source.lastFace<400){ctx.fillStyle='#d5c8ff';for(const p of source.face.points){ctx.beginPath();ctx.arc(p.x,p.y,1.5*scale,0,Math.PI*2);ctx.fill();}}
  for(const foot of Object.values(source.feet)){if(!foot?.roi||performance.now()-source.lastFoot>600)continue;ctx.strokeStyle='#77e7ca';ctx.lineWidth=2*scale;ctx.strokeRect(...foot.roi);ctx.font=`${14*scale}px sans-serif`;ctx.fillStyle='#a3f8df';ctx.fillText(`${foot.label} ${(foot.confidence*100).toFixed(0)}%`,foot.roi[0],Math.max(18*scale,foot.roi[1]-6*scale));}
}
async function runBody(source,roi){
  const spec=activeBody,h=spec.input.shape[2],w=spec.input.shape[3],crop=cropTransform(roi,[w,h],spec.input.crop.padding);
  if(preprocess.width!==w||preprocess.height!==h){preprocess.width=w;preprocess.height=h;}
  preprocessContext.fillStyle='#000';preprocessContext.fillRect(0,0,w,h);preprocessContext.drawImage(source.frame,crop.x,crop.y,crop.width,crop.height,0,0,w,h);
  const data=rgbaToTensor(preprocessContext.getImageData(0,0,w,h).data,w,h,spec.input.mean,spec.input.std);
  const input=new ort.Tensor('float32',data,[1,3,h,w]),start=performance.now();let output;
  try{output=await body.run({[spec.input.name]:input});lastBodyMs=performance.now()-start;limitedPush(bodyTimes,lastBodyMs);bodyCalls++;return decodeSimCC(output.simcc_x.data,output.simcc_y.data,crop,59,spec.simcc_split_ratio,spec.decodeWH||crop.inputWH);}
  finally{input.dispose?.();if(output)Object.values(output).forEach(t=>t.dispose?.());}
}
async function runBody3D(source,roi){
  const size={width:source.video.videoWidth,height:source.video.videoHeight};
  const anchors=source.rawKeypoints||source.details||[];
  const crop=createPose3DCrop(roi,anchors,size),ctx=preprocess3DContext;
  ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#000';ctx.fillRect(0,0,256,256);
  ctx.setTransform(...pose3dCanvasTransform(crop));ctx.drawImage(source.frame,0,0);ctx.setTransform(1,0,0,1,0,0);
  const input=new ort.Tensor('float32',pose3dRgbaToTensor(ctx.getImageData(0,0,256,256).data),[1,256,256,3]);
  const start=performance.now();let output;
  try{output=await body3D.run({[body3D.inputNames[0]]:input});lastBodyMs=performance.now()-start;limitedPush(bodyTimes,lastBodyMs);bodyCalls++;return decodePose3D(output,crop);}
  finally{input.dispose?.();if(output)Object.values(output).forEach(t=>t.dispose?.());}
}
function faceReadout(value){
  if(!value?.valid){resetFaceOnly();return;}
  lastFace=value;const map={'eye-left':value.blinkLeft,'eye-right':value.blinkRight,mouth:value.mouthOpen,cheek:value.cheekWidthChange==null?null:Math.abs(value.cheekWidthChange)*5};
  for(const [id,v] of Object.entries(map)){$(id).value=finite(v)?clamp(v):0;$(id+'-text').textContent=finite(v)?`${Math.round(clamp(v)*100)}%`:'—';}
  view?.updateExpressions?.(value);
}
function resetFaceOnly(){for(const id of ['eye-left','eye-right','mouth','cheek']){$(id).value=0;$(id+'-text').textContent='—';}lastFace=null;view?.updateExpressions?.(null);}
function footReadout(side,value){$(side+'-foot-state').textContent=value?.label||'未観測';$(side+'-foot-score').textContent=value?`${Math.round(value.confidence*100)}% · ${value.valid?'足領域の分類':'不確か'}`:'対象の足領域が必要です';}
function makeObservation(source,stamp){
  const keypoints=source.rawKeypoints?source.rawKeypoints.map(p=>({...p})):toCoco133(source.points,activeBody.coco133_indices||COCO59);
  if(source.face?.valid&&performance.now()-source.lastFace<300){const score=source.faceRoiConfidence||.3;source.face.points.forEach((p,i)=>keypoints[23+i]={id:23+i,x:p.x,y:p.y,score,observable:true,source:'observedFace2D'});if(!keypoints[0]?.modelPosition)keypoints[0]={id:0,x:source.face.nose.x,y:source.face.nose.y,score,observable:true};for(const [id,lo,hi] of [[1,42,48],[2,36,42]]){if(keypoints[id]?.modelPosition)continue;const ps=source.face.points.slice(lo,hi);keypoints[id]={id,x:ps.reduce((s,p)=>s+p.x,0)/ps.length,y:ps.reduce((s,p)=>s+p.y,0)/ps.length,score,observable:true,derived:true};}}
  const sync=sourceSync[source.id];
  return {sourceId:source.id,inputMode:'wholebody',targetVersion:$('model-choice').value,timestampMs:stamp+(finite(sync?.offsetMs)?sync.offsetMs:0),imageSize:{width:source.video.videoWidth,height:source.video.videoHeight},keypoints,learned3D:source.decoded3D?{confidence:source.decoded3D.confidence,rootFit:source.decoded3D.rootFit}:null,timeUncertaintyMs:sync?.confirmed===true&&finite(sync.uncertaintyMs)?sync.uncertaintyMs:1000};
}
function poseFor(source){
  const o=source.observation,options={bodyHeightM:Number($('height').value),scaleConfirmed:$('height-confirm').checked,minScore:Number($('confidence').value),priorPose:source.stabilizer.priorPose,allowApproximateCalibration:false};
  const all=sources.filter(s=>s.observation&&Math.abs(s.observation.timestampMs-o.timestampMs)<=45).map(s=>s.observation);
  const camera=cameraCalibration.find(c=>c.id===source.id)||null;
  const pose=cameraCalibration.length>=2?fusePose(all,cameraCalibration,{...options,timestampMs:o.timestampMs}):o.learned3D?estimateLearnedMonocularPose(o,camera,options):estimateMonocularPose(o,camera,options);
  pose.inputMode='wholebody';pose.sourceId=source.id;pose.targetVersion=o.targetVersion;return source.stabilizer.update(pose,{observation:o});
}
function usableRoi(candidate,source,minSize){
  if(!candidate||candidate.confidence<.3)return null;
  const [x,y,w,h]=candidate.roi,iw=source.video.videoWidth,ih=source.video.videoHeight;
  if(![x,y,w,h].every(finite)||Math.min(w,h)<minSize)return null;
  const visible=Math.max(0,Math.min(iw,x+w)-Math.max(0,x))*Math.max(0,Math.min(ih,y+h)-Math.max(0,y));
  return visible/(w*h)>=.95?candidate:null;
}
function learnedFootRoi(source,side){
  const ids=side==='left'?[15,19,17]:[16,22,20],points=ids.map(id=>source.rawKeypoints?.[id]);
  if(!points.every(p=>p?.observable!==false&&p?.score>=.3&&finite(p.x)&&finite(p.y)))return null;
  const xs=points.map(p=>p.x),ys=points.map(p=>p.y),w=Math.max(...xs)-Math.min(...xs),h=Math.max(...ys)-Math.min(...ys),size=Math.max(w,h)*1.6;
  if(size<16)return null;
  return {roi:[(Math.min(...xs)+Math.max(...xs)-size)/2,(Math.min(...ys)+Math.max(...ys)-size)/2,size,size],confidence:Math.min(...points.map(p=>p.score))};
}
async function processFrame(source,token){
  const start=performance.now(),mode=$('mode').value,roi=source.roi||[0,0,source.video.videoWidth,source.video.videoHeight],stamp=source.stream?start:source.video.currentTime*1000;
  let didInference=false;
  const captured=source.frameCaptureAt||start;
  if(source.frame.width!==source.video.videoWidth||source.frame.height!==source.video.videoHeight){source.frame.width=source.video.videoWidth;source.frame.height=source.video.videoHeight;}
  source.frameContext.drawImage(source.video,0,0);
  if(mode==='wholebody'){
    if($('model-choice').value==='pose3d'){
      // The body has its own model and cannot be blocked by a lost 2D student.
      if(start-source.lastDetails>=200){const details=await runBody(source,roi);source.details=toCoco133(details,activeBody.coco133_indices||COCO59).map(p=>({...p,observedAtMs:stamp}));source.lastDetails=start;}
      source.decoded3D=await runBody3D(source,roi);
      const details=start-source.lastDetails<250?source.details:[];
      source.rawKeypoints=mergePose3DKeypoints(details||[],source.decoded3D);
      source.points=COCO59.map((id,index)=>({...source.rawKeypoints[id],id:index}));
    }else{source.decoded3D=source.rawKeypoints=null;source.points=await runBody(source,roi);}
    didInference=true;if(!running||generation!==token||!sources.includes(source))return;
  }
  else{source.points=null;source.observation=null;}
  if((mode==='face'||mode==='wholebody'&&$('face-enabled').checked)&&start-source.lastFace>=100){
    const candidate=usableRoi(mode==='face'?{roi,confidence:1}:faceRoiFrom59(source.points),source,24);
    if(candidate){source.face=await parts.runFace(source.frame,candidate.roi,{roiConfidence:candidate.confidence,calibration:neutralFace});didInference=true;source.faceRoiConfidence=candidate.confidence;limitedPush(partTimes,source.face.inferenceMs);source.lastFace=performance.now();}else source.face=null;
  }
  if((mode==='foot'||mode==='wholebody'&&$('foot-enabled').checked)&&start-source.lastFoot>=200){
    for(const side of (mode==='foot'?['left']:['left','right'])){
      const candidate=usableRoi(mode==='foot'?{roi,confidence:1}:source.decoded3D?learnedFootRoi(source,side):footRoiFrom59(source.points,side),source,16);
      if(candidate){source.feet[side]={...await parts.runFoot(source.frame,candidate.roi,{roiConfidence:candidate.confidence}),roi:candidate.roi};didInference=true;limitedPush(partTimes,source.feet[side].inferenceMs);}else delete source.feet[side];
    }
    source.lastFoot=performance.now();
  }
  if(!running||generation!==token||!sources.includes(source))return;
  if(mode==='wholebody'){
    source.observation=source.stabilizer.filterObservation(makeObservation(source,stamp),{minScore:Number($('confidence').value)});
    source.points=source.points.map((p,i)=>({...p,score:source.observation.keypoints[(activeBody.coco133_indices||COCO59)[i]]?.score||0}));
    for(const [side,ankle] of [['left',15],['right',16]])if(source.observation.keypoints[ankle]?.observable===false)delete source.feet[side];
  }
  if(source===selected){
    faceReadout(performance.now()-source.lastFace<400?source.face:null);
    for(const side of ['left','right'])footReadout(side,performance.now()-source.lastFoot<600?source.feet[side]:null);
    if(source.observation){lastPose=poseFor(source);view?.updatePose(lastPose);const state=lastPose.stability?.state;$('tracking-state').textContent=({tracking:'追跡中',partial:'部分追跡',lost:'観測なし'})[state]||'追跡中';$('pose-status').textContent=lastPose.metric?`位置 ${lastPose.root?.distanceStatus} · ${lastPose.diagnostics?.triangulatedJointCount||0} 点を多視点で計測`:lastPose.diagnostics?.method==='learned-xyz-and-perspective-root'?`3D身体 ${lastPose.diagnostics.learned3DJointCount} 点 · 前後移動 ${lastPose.previewRoot?.valid?'推定中':'未取得'} · ${lastPose.stability.acceptedPoints||0} 点受理` :state==='lost'?'観測なし':state==='partial'?'見えている関節を反映中':'2D相対姿勢';bridge.sendPose(lastPose,{captureAgeMs:Math.max(0,performance.now()-captured),timeDomain:source.stream?'browser-camera-receipt':'recorded-media'});}
    else{clearPose();$('pose-status').textContent=mode==='foot'?'足領域の見え方を分類中。絶対位置と足首回転は未観測です。':'顔領域から表情を読み取り中。';}
    $('measure-points').innerHTML=`${source.points?.filter(p=>p.score>=Number($('confidence').value)).length??'—'}<small> / 59</small>`;
  }
  draw(source);if(didInference){limitedPush(timings,performance.now()-start);limitedPush(readTimes,Math.max(0,performance.now()-captured));frameCounter++;}
}
async function loop(token){
  let cursor=0,next=0;
  while(running&&generation===token){
    const now=performance.now();if(now<next){await new Promise(r=>setTimeout(r,Math.min(16,next-now)));continue;}
    const available=sources.filter(s=>s.video.readyState>=2&&!s.video.ended&&(!s.video.paused||s.lastMediaTime!==s.video.currentTime));
    if(!available.length){await new Promise(r=>setTimeout(r,30));continue;}
    const source=available[cursor++%available.length];
    if(source.lastMediaTime===source.video.currentTime){await new Promise(r=>setTimeout(r,8));continue;}
    source.lastMediaTime=source.video.currentTime;source.lastFrameAt=now;
    try{inFlight=processFrame(source,token);await inFlight;}catch(e){stop();fail(e);break;}finally{inFlight=null;}
    next=now+1000/Math.min($('model-choice').value==='pose3d'?15:60,Number($('target-fps').value));
    if(performance.now()-lastMetricsAt>=1000){const dt=(performance.now()-lastMetricsAt)/1000;$('measure-fps').innerHTML=`${(frameCounter/dt).toFixed(1)}<small> fps</small>`;$('measure-body').innerHTML=`${lastBodyMs?.toFixed(1)??'—'}<small> ms</small>`;$('measure-p95').innerHTML=`${percentile(timings,.95)?.toFixed(1)??'—'}<small> ms</small>`;frameCounter=0;lastMetricsAt=performance.now();}
  }
}
async function start(){
  if(!sources.length){notify('動画またはカメラを選択してください。');return;}
  stop();if(inFlight)await inFlight.catch(()=>{});if(!await prepareModels())return;running=true;const token=++generation;$('start').disabled=true;$('stop').disabled=false;$('status-dot').classList.add('on');$('top-status').textContent=`${actualBackend==='webgpu'?'WebGPU':'CPU'} · 推論中`;
  await initView();view?.setPaused(false);for(const s of sources){s.stabilizer.reset();s.observation=null;s.lastMediaTime=s.lastDetails=-Infinity;s.details=s.decoded3D=s.rawKeypoints=null;await s.video.play().catch(()=>{});}frameCounter=0;lastMetricsAt=performance.now();notify('');void loop(token);
}
function stop(){running=false;generation++;for(const s of sources)if(!s.stream)s.video.pause();$('start').disabled=loading;$('stop').disabled=true;$('status-dot').classList.remove('on');$('top-status').textContent='一時停止';bridge.stop();}
async function initView(){if(view)return view;if(viewPromise)return viewPromise;viewPromise=(async()=>{const {createVRMView}=await import('./vrm-view.mjs');view=createVRMView({container:$('avatar-canvas'),onState:s=>$('avatar-status').textContent=s.hasVRM?`${s.fileName} · ${s.appliedBoneCount} ボーン反映`:'VRM未選択',onError:fail});if(lastPose)view.updatePose(lastPose);return view;})();try{return await viewPromise;}catch(e){viewPromise=null;throw e;}}
async function enumerate(){if(!navigator.mediaDevices?.enumerateDevices)return;const selected=$('camera-select').value,devices=await navigator.mediaDevices.enumerateDevices();$('camera-select').replaceChildren(new Option('標準カメラ',''));for(const d of devices.filter(d=>d.kind==='videoinput'))$('camera-select').append(new Option(d.label||`カメラ ${$('camera-select').options.length}`,d.deviceId));$('camera-select').value=selected;}
$('video-file').onchange=e=>{try{for(const file of e.target.files)createSource({file});}catch(error){fail(error);}e.target.value='';};
$('camera-add').onclick=async()=>{try{if(sources.length>=3)throw new Error('入力は3件までです。');const device=$('camera-select').value;const stream=await navigator.mediaDevices.getUserMedia({video:{...(device?{deviceId:{exact:device}}:{}),width:{ideal:1280},height:{ideal:720},frameRate:{ideal:60,max:60}},audio:false});createSource({stream,label:stream.getVideoTracks()[0].label});await enumerate();}catch(e){fail(e);}};
$('start').onclick=()=>void start().catch(fail);$('stop').onclick=stop;
$('mode').onchange=()=>{stop();for(const s of sources){s.stabilizer.reset();s.points=null;s.face=null;s.feet={};s.observation=null;s.lastFace=s.lastFoot=-Infinity;draw(s);}resetReadouts();clearPose();notify('対象を囲み、開始してください。');};
$('backend').onchange=stop;$('model-choice').onchange=()=>{stop();for(const s of sources)s.stabilizer.reset();clearPose();};$('face-enabled').onchange=stop;$('foot-enabled').onchange=stop;
$('target-fps').oninput=e=>$('fps-value').value=e.target.value;
$('confidence').oninput=()=>sources.forEach(draw);
$('reset-roi').onclick=()=>{if(selected){selected.stabilizer.reset();selected.roi=null;selected.observation=null;selected.lastMediaTime=-Infinity;selected.lastFace=selected.lastFoot=-Infinity;clearPose();draw(selected);}};
$('neutral-face').onclick=()=>{if(!lastFace?.valid){notify('顔が検出されてから基準を設定してください。');return;}neutralFace={eyeLeftOpen:lastFace.eyeLeftEAR,eyeRightOpen:lastFace.eyeRightEAR,cheekWidthNormalized:lastFace.cheekWidthNormalized};notify('現在の目の開きと頬幅を基準にしました。');};
$('vrm-file').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;await initView();await view.loadFile(file);view.setPaused(Boolean(selected&&!selected.stream&&selected.video.paused));if(lastFace)view.updateExpressions(lastFace);}catch(error){fail(error);}e.target.value='';};
$('reset-view').onclick=()=>view?.resetView();
$('calibration-file').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const input=JSON.parse(await file.text()),result=validateCalibration(input);if(!result.valid)throw new Error(result.errors.join(' / '));cameraCalibration=result.cameras;sourceSync=input.sourceSync||{};clearPose();$('calibration-status').textContent=`${cameraCalibration.length} 台の校正を読み込みました。入力IDは camera-1 / camera-2 / camera-3 です。多視点の統合にはsourceSyncの確認済み時刻補正が必要です。`;}catch(error){fail(error);}e.target.value='';};
$('bridge-connect').onclick=async()=>{try{await bridge.connect({url:$('bridge-url').value,token:$('bridge-token').value});$('bridge-token').value='';}catch(e){fail(e);}};
$('room-file').onchange=async e=>{try{const file=e.target.files[0];if(file)bridge.calibrate(JSON.parse(await file.text()));}catch(error){fail(error);}e.target.value='';};
$('export-metrics').onclick=()=>{
  const result={schema:'compact-vrc-device-metrics-v2',createdAt:new Date().toISOString(),backend:actualBackend,bodyModel:activeBody?.id,modelSha256:activeBody?.sha256,bodyCalls,bodyMedianMs:percentile(bodyTimes,.5),bodyP95Ms:percentile(bodyTimes,.95),fullFrameMedianMs:percentile(timings,.5),fullFrameP95Ms:percentile(timings,.95),partsMedianMs:percentile(partTimes,.5),partsProvider:'wasm',faceMaxHz:10,footMaxHz:5,inputCount:sources.length,inputMode:$('mode').value,settingMaxCallsPerSecond:Number($('target-fps').value),containsVideoOrLandmarks:false,gpuUtilizationMeasured:false,scope:'Observed browser elapsed time. No frame content, file names or landmark traces.'};
  const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='vrc-compact-device-metrics.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
window.addEventListener('pagehide',()=>{stop();bridge.disconnect();for(const s of sources){s.stream?.getTracks().forEach(t=>t.stop());if(s.url)URL.revokeObjectURL(s.url);}view?.dispose();});
void enumerate().catch(()=>{});
