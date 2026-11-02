import {PoseEngine} from './inference.js';
import {BODY_MODELS} from './models.js';
import {setOffline,NETWORK_LOG,cachedFetch} from './model-cache.js';
import {EDGES,clamp} from './skeleton.js';
import {triangulateFrame,estimateRootDepth,anthropometricBoneLengths,calibratePlanarCamera,validateCamera} from './geometry.js';
import {OcclusionTracker,mergeJointEstimates} from './tracking.js';
import {PoseView} from './view3d.js';
import {CameraPeer,SignalClient,CONNECTION_LOG} from './lan.js';

const $=id=>document.getElementById(id);
const number=(id,fallback=0)=>{const v=Number($(id).value);return Number.isFinite(v)?v:fallback;};
const engine=new PoseEngine(),tracker=new OcclusionTracker(),world=new PoseView($('world'));
const sources=['a','b'].map(id=>({id,video:$('video-'+id),canvas:$('view-'+id),snapshot:document.createElement('canvas'),kind:null,media:null,ready:false,version:0,serial:0,result:null,box:null,manualBox:null,calibration:null,corners:[],lastMeta:null}));
const state={running:false,processing:false,loading:false,needsReload:false,records:[],logs:[],timings:[],runBegan:0,processed:0,peer:{},signal:null,coordinateFrame:'world',coordinateId:'world',fusion:null,lastFreshWall:0,lastFusionTime:0,lastDecayWall:0};
for(const s of sources){s.video.hidden=false;s.video.className='source-video';s.video.setAttribute('aria-hidden','true');}
for(const m of BODY_MODELS){const option=document.createElement('option');option.value=m.id;option.textContent=m.name;$('model').append(option);}
$('model').value='rtmw-l';
function status(message,kind=''){ $('status').textContent=message;$('status-dot').className='dot '+kind; }
function log(message){state.logs.push(new Date().toLocaleTimeString('ja-JP')+' '+message);state.logs=state.logs.slice(-80);$('event-log').textContent=state.logs.join('\n');}
function fail(error){const message=error?.message||String(error);$('alert').hidden=false;$('alert').textContent=message;status('処理を確認してください','error');log(message);}
function clearError(){$('alert').hidden=true;}
const guarded=fn=>async event=>{try{await fn(event);}catch(error){fail(error);}};
function resetHistory(){tracker.reset();state.fusion=null;world.clear('新しいフレームを待っています');}
function saveJSON(name,value){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function setRunControls(){const ready=!!engine.session&&!state.loading&&!state.needsReload;const input=sources.some(s=>s.ready);$('run').disabled=!ready||!input||state.running;$('single').disabled=!ready||!input||state.processing||state.running;$('stop').disabled=!state.running;$('load').disabled=state.loading||state.processing;}
function stop(){state.running=false;for(const s of sources)if(s.kind==='video')s.video.pause();setRunControls();if(engine.session)status('停止中','ready');}
async function waitIdle(){stop();while(state.processing)await new Promise(r=>setTimeout(r,30));}

for(const button of document.querySelectorAll('[data-tab]'))button.onclick=()=>{
  document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b===button));
  document.querySelectorAll('[data-panel]').forEach(p=>p.hidden=p.dataset.panel!==button.dataset.tab);
  if(button.dataset.tab==='calibration')sources.forEach(drawCalibration);
  if(button.dataset.tab==='tracking')world.render(state.fusion,sources.map(s=>s.calibration).filter(Boolean),{coordinateFrame:state.coordinateFrame});
};
$('confidence').oninput=()=>{$('confidence-value').textContent=number('confidence').toFixed(2);sources.forEach(drawSource);};
$('offline').onchange=()=>setOffline($('offline').checked);
$('load').onclick=guarded(async()=>{
  await waitIdle();clearError();state.loading=true;setRunControls();$('progress').hidden=false;$('progress').value=0;
  try{await engine.load({modelId:$('model').value,backend:$('backend').value,hands:$('hands').checked,detector:$('detector').checked,onProgress:p=>{
    const mb=n=>(n/1048576).toFixed(1);$('load-detail').textContent=p.name+' · '+({runtime:'準備中',compile:'GPU/CPUセッションを作成中',cache:'保存済み',verify:'照合中',download:'取得中'}[p.phase]||p.phase)+(p.received?` ${mb(p.received)} / ${mb(p.total||p.received)} MB`:'');
    if(p.total)$('progress').value=p.received/p.total;
  }});state.needsReload=false;status('推論の準備ができました','ready');$('load-detail').textContent=`${engine.model.name} · ${engine.backend.toUpperCase()} · 映像を選んで推論を開始`;resetHistory();log('モデル読み込み完了: '+engine.model.id);}
  finally{state.loading=false;$('progress').hidden=true;setRunControls();}
});

function watchVideo(s,version){
  if(!s.video.requestVideoFrameCallback)return;
  s.video.requestVideoFrameCallback((now,meta)=>{if(version!==s.version)return;s.lastMeta={...meta,callbackTime:now};s.serial++;watchVideo(s,version);});
}
async function releaseSource(s){
  s.version++;s.ready=false;s.video.pause();s.video.srcObject?.getTracks().forEach(t=>t.stop());s.video.srcObject=null;s.video.removeAttribute('src');s.video.load();
  if(s.objectURL)URL.revokeObjectURL(s.objectURL);s.objectURL=null;s.media=null;s.kind=null;s.result=null;s.box=null;s.manualBox=null;s.lastMeta=null;s.calibration=null;s.corners=[];
  $('empty-'+s.id).hidden=false;$('source-'+s.id+'-info').textContent='未選択';$('seek-'+s.id).disabled=true;s.canvas.getContext('2d').clearRect(0,0,s.canvas.width,s.canvas.height);
}
function snapshotSource(s){
  const w=s.media?.videoWidth||s.media?.naturalWidth||s.media?.width,h=s.media?.videoHeight||s.media?.naturalHeight||s.media?.height;
  if(!w||!h)return false;if(s.width&&s.height&&(s.width!==w||s.height!==h)){s.corners=[];s.calibration=null;resetHistory();log(`入力${s.id}: 映像寸法が変わったため校正を無効化しました`);}s.width=w;s.height=h;
  if(s.snapshot.width!==w||s.snapshot.height!==h){s.snapshot.width=w;s.snapshot.height=h;}
  s.snapshot.getContext('2d').drawImage(s.media,0,0,w,h);return true;
}
async function fileSource(s,file){
  await waitIdle();await releaseSource(s);clearError();
  s.objectURL=URL.createObjectURL(file);
  if(file.type.startsWith('image/')){const image=new Image();image.src=s.objectURL;await image.decode();s.media=image;s.kind='image';}
  else{s.video.src=s.objectURL;await new Promise((resolve,reject)=>{s.video.onloadeddata=resolve;s.video.onerror=()=>reject(Error('この動画をデコードできません'));});s.video.onloadeddata=null;s.video.onerror=null;s.media=s.video;s.kind='video';watchVideo(s,s.version);
    const seek=$('seek-'+s.id);seek.max=Number.isFinite(s.video.duration)?s.video.duration:1;seek.disabled=false;}
  s.ready=true;s.serial++;snapshotSource(s);$('empty-'+s.id).hidden=true;$('source-'+s.id+'-info').textContent=`${s.width} × ${s.height}`;drawSource(s);drawCalibration(s);resetHistory();setRunControls();log(`入力${s.id.toUpperCase()}: ${file.name}`);
}
async function streamSource(s,stream,kind='camera'){
  await waitIdle();await releaseSource(s);s.video.srcObject=stream;await s.video.play();s.media=s.video;s.kind=kind;
  if(!s.video.videoWidth)await new Promise(resolve=>s.video.addEventListener('loadedmetadata',resolve,{once:true}));
  s.ready=true;watchVideo(s,s.version);snapshotSource(s);$('empty-'+s.id).hidden=true;$('source-'+s.id+'-info').textContent=kind==='remote'?'LANカメラ':'PCカメラ';drawSource(s);resetHistory();setRunControls();
}
for(const s of sources){
  $('file-'+s.id).onchange=guarded(async e=>{if(e.target.files[0])await fileSource(s,e.target.files[0]);});
  $('clear-'+s.id).onclick=guarded(async()=>{await waitIdle();state.peer[s.id]?.close();delete state.peer[s.id];await releaseSource(s);resetHistory();setRunControls();});
  $('camera-'+s.id).onclick=guarded(async()=>{
    if(!navigator.mediaDevices?.getUserMedia)throw Error('カメラはHTTPSまたはlocalhostで使用してください');
    await waitIdle();const devices=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
    const used=new Set(sources.filter(x=>x!==s).flatMap(x=>x.video.srcObject?.getVideoTracks().map(t=>t.getSettings().deviceId)||[]));
    const device=devices.find(d=>!used.has(d.deviceId));
    if(!device&&used.size)throw Error('別のPCカメラが見つかりません。入力Bへスマホを接続することもできます。');
    const stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{...(device?{deviceId:{exact:device.deviceId}}:{}),width:{ideal:1280},height:{ideal:720},frameRate:{ideal:60,max:60}}});await streamSource(s,stream);
  });
  $('roi-'+s.id).onclick=()=>{s.manualBox=null;s.box=null;drawSource(s);};
  $('seek-'+s.id).oninput=guarded(async e=>{await waitIdle();s.video.currentTime=Number(e.target.value);resetHistory();});
  s.video.onseeked=()=>{if(s.ready){s.result=null;s.lastMeta=null;s.serial++;snapshotSource(s);drawSource(s);drawCalibration(s);}};
  s.video.onended=()=>{if(sources.filter(x=>x.kind==='video').every(x=>x.video.ended))stop();};
  let start=null;
  s.canvas.onpointerdown=event=>{if(!s.ready)return;stop();start=canvasPoint(event,s.canvas,s.width,s.height);s.canvas.setPointerCapture(event.pointerId);};
  s.canvas.onpointermove=event=>{if(!start)return;const end=canvasPoint(event,s.canvas,s.width,s.height);drawSource(s,[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])]);};
  s.canvas.onpointerup=event=>{if(!start)return;const end=canvasPoint(event,s.canvas,s.width,s.height);const b=[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])];start=null;if(b[2]-b[0]>20&&b[3]-b[1]>20){s.manualBox=b;s.box=b;}drawSource(s);resetHistory();};
  s.canvas.onpointercancel=()=>{start=null;drawSource(s);};
}
function canvasPoint(event,canvas,width,height){const r=canvas.getBoundingClientRect();return[clamp((event.clientX-r.left)/r.width*width,0,width),clamp((event.clientY-r.top)/r.height*height,0,height)];}
function drawSource(s,dragBox){
  if(!s.ready||!s.width)return;const scale=Math.min(1,1000/s.width),c=s.canvas;c.width=Math.round(s.width*scale);c.height=Math.round(s.height*scale);
  const ctx=c.getContext('2d');ctx.drawImage(s.snapshot,0,0,c.width,c.height);ctx.save();ctx.scale(scale,scale);
  const pts=s.result?.points||[],threshold=number('confidence');
  const valid=p=>p&&p.score>=threshold&&p.inFrame!==false;
  ctx.lineWidth=2/scale;
  for(const[a,b]of EDGES){const p=pts[a],q=pts[b];if(!valid(p)||!valid(q))continue;ctx.strokeStyle=a>=91?'#b4a0ff':a>=15&&b>=17?'#50dfd2':'#e9f3ff';ctx.beginPath();ctx.moveTo(p.u,p.v);ctx.lineTo(q.u,q.v);ctx.stroke();}
  for(const p of pts){if(!valid(p))continue;ctx.fillStyle=p.id>=91?'#b4a0ff':p.id>=23?'#a6b6c688':'#50dfd2';ctx.beginPath();ctx.arc(p.u,p.v,(p.id>=23&&p.id<91?1:2.2)/scale,0,Math.PI*2);ctx.fill();}
  const box=dragBox||s.manualBox||s.result?.box;if(box){ctx.strokeStyle=dragBox?'#ffd281':'#50dfd299';ctx.setLineDash([6/scale,5/scale]);ctx.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);}
  ctx.restore();
}
function stamp(s){
  if(s.kind==='video'){const t=s.lastMeta?.mediaTime??s.video.currentTime;return{timeMs:(t-(s.id==='b'?number('time-offset'):0))*1000,timeBasis:'media',synchronized:$('files-synchronized').checked,mediaTime:t};}
  if(s.kind==='image')return{timeMs:performance.now(),timeBasis:'static-image',synchronized:false};
  const capture=s.lastMeta?.captureTime;
  const age=performance.now()-capture,clock=state.peer[s.id]?.clock;
  if(Number.isFinite(capture)&&age>=-2&&age<1500)return{timeMs:capture,timeBasis:'webrtc-capture-estimate',synchronized:false,captureAgeMs:age,clockUncertaintyMs:clock?.uncertaintyMs??null,clockAgeMs:clock?performance.now()-clock.measuredAt:null};
  return{timeMs:s.lastMeta?.expectedDisplayTime??performance.now(),timeBasis:'presentation',synchronized:false};
}
function adaptedCamera(s){
  const c=s.calibration;if(!c)return null;
  const sx=s.width/c.width,sy=s.height/c.height;
  if(!Number.isFinite(sx)||!Number.isFinite(sy)||Math.abs(sx/sy-1)>.015)return null;
  return{...c,id:s.id,width:s.width,height:s.height,fx:c.fx*sx,fy:c.fy*sy,cx:c.cx*sx,cy:c.cy*sy,skew:(c.skew||0)*sx};
}
function approximateCamera(s){const fov=clamp(number('mono-fov',70),20,150)*Math.PI/180,fx=s.width/(2*Math.tan(fov/2));return{id:s.id,width:s.width,height:s.height,fx,fy:fx,cx:s.width/2,cy:s.height/2,distortion:[],calibrated:false,intrinsicsCalibrated:false};}
function fuse(batch){
  const frames=batch.filter(b=>b.result?.points?.length).map(b=>({sourceId:b.s.id,timeMs:b.stamp.timeMs,keypoints:b.result.points}));
  const cameras=Object.fromEntries(batch.map(b=>[b.s.id,adaptedCamera(b.s)]).filter(([,c])=>c));
  const compatibleClocks=batch.length===2&&batch.every(b=>b.stamp.synchronized)&&batch.every(b=>b.stamp.timeBasis==='media');
  const captureEstimates=batch.length===2&&batch.every(b=>b.stamp.timeBasis==='webrtc-capture-estimate');
  const allowApproximate=$('allow-approximate').checked;
  let triangulated=[],mono=null,frame=state.coordinateFrame,message='2視点の校正が必要です';
  if(frames.length===2&&Object.keys(cameras).length===2){
    if(compatibleClocks||captureEstimates||allowApproximate){
      const inputs=compatibleClocks||captureEstimates?cameras:Object.fromEntries(Object.entries(cameras).map(([id,c])=>[id,{...c,calibrated:false,calibrationStatus:'approximate'}]));
      triangulated=triangulateFrame(frames,inputs,{minimumScore:number('confidence'),maxTimeDeltaMs:25,allowApproximateCalibration:allowApproximate});
      if(!compatibleClocks)triangulated=triangulated.map(j=>j.valid?{...j,source:'estimatedMultiView',distanceStatus:'estimatedMultiView',confidence:j.confidence*.7,observed:false,timingEstimated:true}:j);
      message=compatibleClocks?'共通関節を2視点から復元':captureEstimates?'WebRTCの推定撮影時刻による2視点3D':'表示時刻による暫定3D · 撮影同期は未確認';
    }else message='撮影時刻の対応が未確認です。録画の時間差または接続の時刻品質を確認してください。';
  }
  const primary=batch.find((b,index)=>b.fresh&&(index===0||adaptedCamera(b.s)?.worldFromCamera)&&b.result?.points?.some(p=>Number.isFinite(p.relativeZm)));
  if(primary){const camera=adaptedCamera(primary.s)||approximateCamera(primary.s);
    mono=estimateRootDepth(primary.result.points,camera,anthropometricBoneLengths(number('height',170)/100),{minimumScore:number('confidence'),boneLengthSource:'anthropometricPrior'});
    if(mono.valid&&(!triangulated.some(j=>j.valid)||mono.coordinateFrame==='world')){frame=mono.coordinateFrame;
      if(!triangulated.some(j=>j.valid))message=`単眼の骨長推定 · 骨盤まで約${mono.depthM.toFixed(2)}m · ${camera.intrinsicsCalibrated?'内部校正あり':'画角は仮定値'}`;
    }else if(!triangulated.some(j=>j.valid))message='単眼の距離を確定できる関節が不足しています';
  }
  const primaryValid=triangulated.filter(j=>j.valid);let joints=primaryValid;
  if(mono?.valid&&(!primaryValid.length||mono.coordinateFrame==='world'))joints=mergeJointEstimates(primaryValid,mono.joints);
  if(primaryValid.length)frame='world';
  const sameDomain=batch.every(b=>b.stamp.timeBasis==='media')||batch.every(b=>b.stamp.timeBasis!=='media');
  const timeMs=sameDomain?Math.max(...batch.map(b=>b.stamp.timeMs)):batch[0].stamp.timeMs;
  const coordinateId=frame==='camera'?`camera:${primary?.s.id||state.coordinateId.split(':')[1]||'a'}`:'world';
  if(joints.length&&coordinateId!==state.coordinateId)tracker.reset();state.coordinateId=coordinateId;
  const fused=tracker.update({timeMs,joints});state.coordinateFrame=frame;state.fusion=fused;
  $('depth-status').textContent=message;$('calibration-badge').textContent=primaryValid.length?`${primaryValid.length}点を復元`:mono?.valid?'単眼推定':'3D未確定';
  world.render(fused,Object.values(cameras),{coordinateFrame:frame,statusText:message});
  $('metric-missing').textContent=`${fused.stats.predicted+fused.stats.kinematicPrior} / ${fused.stats.missing}`;
  return{...fused,coordinateFrame:frame,synchronization:{compatibleClocks,stamps:batch.map(b=>({sourceId:b.s.id,...b.stamp}))},monocular:mono?{valid:mono.valid,depthM:mono.depthM,distanceStatus:mono.distanceStatus,fitErrorM:mono.fitErrorM}:null};
}
async function inferBatch(force=false){
  if(state.processing||state.loading)return;const active=sources.filter(s=>s.ready);if(!active.length)throw Error('入力映像を選択してください');
  state.processing=true;setRunControls();
  try{
    // Freeze both views before running the shared GPU session. Body and hand crops use this exact image.
    const batch=active.map(s=>{const key=frameKey(s);const fresh=force||!s.result||key!==s.lastFrameKey;
      if(fresh&&!snapshotSource(s))return null;return{s,fresh,frameKey:key,stamp:fresh?stamp(s):s.lastStamp,result:fresh?null:s.result};}).filter(Boolean);
    if(!batch.length)return;
    for(const b of batch){const s=b.s;if(!b.fresh)continue;
      b.result=await engine.infer(s.snapshot,{box:s.manualBox||s.box,automatic:$('detector').checked&&!s.manualBox,redetect:!s.box||(s.inferenceCount||0)%12===0,hands:$('hands').checked});
      s.result=b.result;s.lastStamp=b.stamp;s.lastFrameKey=b.frameKey;s.inferenceCount=(s.inferenceCount||0)+1;if(b.result.box)s.box=b.result.box;drawSource(s);drawCalibration(s);
      $('source-'+s.id+'-info').textContent=b.result.missing?'人物を再検出中':`${s.width}×${s.height} · ${b.result.poseMs.toFixed(1)}ms`;
      if(s.kind==='video')$('seek-'+s.id).value=s.video.currentTime;
      state.processed++;
    }
    const fused=fuse(batch);state.lastFreshWall=performance.now();state.lastFusionTime=fused.timeMs;const freshBatch=batch.filter(b=>b.fresh);const total=key=>freshBatch.reduce((sum,b)=>sum+(b.result[key]||0),0)/Math.max(1,freshBatch.length);
    $('metric-pose').innerHTML=total('poseMs').toFixed(1)+'<small> ms</small>';$('metric-hand').innerHTML=total('handMs').toFixed(1)+'<small> ms</small>';
    state.timings.push({at:performance.now(),count:freshBatch.length});state.timings=state.timings.filter(x=>performance.now()-x.at<3000);
    const elapsed=Math.min(3000,performance.now()-state.runBegan);const fps=state.timings.reduce((n,t)=>n+t.count,0)/(Math.max(1,elapsed)/1000);$('metric-fps').innerHTML=fps.toFixed(1)+'<small> 枚/秒</small>';
    const record={sequence:state.processed,at:new Date().toISOString(),inputs:batch.map(b=>({sourceId:b.s.id,...b.stamp,width:b.s.width,height:b.s.height,box:b.result.box,roi:b.result.roi,points:b.result.points,hands:b.result.hands,poseMs:b.result.poseMs,handMs:b.result.handMs})),fused};
    state.records.push(record);if(state.records.length>3000)state.records.shift();$('record-count').textContent=`${state.records.length}組のフレーム（最大3000組を保持）`;
    status(state.running?'推論中':'1フレームを処理しました','ready');
  }finally{state.processing=false;setRunControls();}
}
async function loop(){
  if(!state.running)return;
  const changed=sources.some(s=>s.ready&&(s.kind==='image'?!s.result:frameKey(s)!==s.lastFrameKey));
  if(!changed){decayLiveStall();requestAnimationFrame(loop);return;}
  try{await inferBatch();}catch(error){stop();fail(error);return;}
  if(sources.filter(s=>s.ready).every(s=>s.kind==='image')){stop();return;}
  if(state.running)requestAnimationFrame(loop);
}
function frameKey(s){if(s.kind==='image')return'image';if(Number.isFinite(s.lastMeta?.presentedFrames))return s.lastMeta.presentedFrames;const decoded=s.video.getVideoPlaybackQuality?.().totalVideoFrames;return Number.isFinite(decoded)&&decoded>0?decoded:'unknown-frame';}
function decayLiveStall(){
  const now=performance.now();if(!state.fusion||!sources.some(s=>s.ready&&['camera','remote'].includes(s.kind))||now-state.lastFreshWall<120||now-state.lastDecayWall<50)return;
  state.lastDecayWall=now;state.fusion=tracker.update({timeMs:state.lastFusionTime+now-state.lastFreshWall,joints:[]});world.render(state.fusion,sources.map(s=>s.calibration).filter(Boolean),{coordinateFrame:state.coordinateFrame});
  $('metric-missing').textContent=`${state.fusion.stats.predicted+state.fusion.stats.kinematicPrior} / ${state.fusion.stats.missing}`;$('depth-status').textContent='映像の新しいフレームが届かないため、短時間予測から欠測へ移行しています。';
}
$('run').onclick=guarded(async()=>{clearError();if(!engine.session)throw Error('モデルを読み込んでください');for(const s of sources)if(s.ready&&s.kind==='video'){if(s.video.ended)s.video.currentTime=0;await s.video.play();}state.running=true;state.runBegan=performance.now();state.processed=0;state.timings=[];setRunControls();loop();});
$('stop').onclick=stop;
for(const id of['model','backend','hands','detector'])$(id).onchange=()=>{if(id==='model')$('hands').checked=$('model').value==='rtmpose-s';if(engine.session){stop();state.needsReload=true;$('load-detail').textContent='設定を変更しました。モデルを読み込み直してください。';setRunControls();}};
$('single').onclick=guarded(async()=>{clearError();await waitIdle();state.runBegan=performance.now();state.timings=[];await inferBatch(true);});
$('sample').onclick=guarded(async()=>{const url='https://raw.githubusercontent.com/open-mmlab/mmpose/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/tests/data/coco/000000000785.jpg';const{bytes}=await cachedFetch(url,{cacheName:'wholebody-sample-v1'});await fileSource(sources[0],new File([bytes],'COCO-000000000785.jpg',{type:'image/jpeg'}));});

// Intrinsics and four measured floor corners are explicit user inputs.
for(const s of sources){
  const card=document.createElement('div');card.className='calibration-view';
  card.innerHTML=`<h3>入力${s.id.toUpperCase()}</h3><canvas id="cal-view-${s.id}" width="640" height="480"></canvas><p id="corner-count-${s.id}" class="small muted">四隅を0 / 4点選択</p><button id="mark-${s.id}">4点を選び直す</button><div class="form-grid"><label>fx（pixel）<input id="fx-${s.id}" type="number" min="1" placeholder="未入力なら画角から仮定"></label><label>fy（pixel）<input id="fy-${s.id}" type="number" min="1"></label><label>cx（pixel）<input id="cx-${s.id}" type="number" placeholder="画像中心"></label><label>cy（pixel）<input id="cy-${s.id}" type="number" placeholder="画像中心"></label></div><label>横画角の仮定（度）<input id="fov-${s.id}" type="number" value="70" min="20" max="150"></label><label class="check"><input id="intrinsics-${s.id}" type="checkbox"><span>fx・fyは実測した内部校正値</span></label><button id="solve-${s.id}" class="primary">この視点を校正</button>`;
  $('calibration-controls').append(card);
  $('cal-view-'+s.id).onclick=event=>{if(!s.ready)return;if(s.corners.length===4)return;s.corners.push(canvasPoint(event,event.currentTarget,event.currentTarget.width,event.currentTarget.height));drawCalibration(s);};
  $('mark-'+s.id).onclick=()=>{s.corners=[];s.calibration=null;drawCalibration(s);resetHistory();};
  $('solve-'+s.id).onclick=guarded(()=>{
    if(s.corners.length!==4)throw Error('両視点で同じ順に床の四隅を選択してください');
    const w=number('floor-width'),d=number('floor-depth');if(w<=0||d<=0)throw Error('床の幅・奥行をメートルで入力してください');
    const measured=$('intrinsics-'+s.id).checked;
    if(measured&&(!number('fx-'+s.id)||!number('fy-'+s.id)))throw Error('実測したfxとfyを両方入力してください');
    const assumed=s.width/(2*Math.tan(number('fov-'+s.id,70)*Math.PI/360));
    const intrinsics={width:s.width,height:s.height,fx:number('fx-'+s.id)||assumed,fy:number('fy-'+s.id)||number('fx-'+s.id)||assumed,cx:$('cx-'+s.id).value===''?s.width/2:number('cx-'+s.id),cy:$('cy-'+s.id).value===''?s.height/2:number('cy-'+s.id),distortion:[],intrinsicsCalibrated:measured};
    const result=calibratePlanarCamera({id:s.id,imagePoints:s.corners,worldPoints:[[0,0,0],[w,0,0],[w,0,d],[0,0,d]],intrinsics});
    if(!result.valid)throw Error('床の校正に失敗: '+result.reason+'。4点の順番と寸法を確認してください。');
    s.calibration=result.camera;resetHistory();clearError();$('calibration-result').textContent=JSON.stringify({camera:s.id,status:result.calibrationStatus,reprojectionErrorPx:result.reprojectionErrorPx,position:result.camera.worldFromCamera.translation},null,2);log(`入力${s.id}校正: ${result.calibrationStatus}, 再投影 ${result.reprojectionErrorPx.toFixed(2)}px`);
  });
}
function drawCalibration(s){
  const c=$('cal-view-'+s.id);if(!c||!s.ready)return;c.width=s.width;c.height=s.height;const ctx=c.getContext('2d');ctx.drawImage(s.snapshot,0,0);
  ctx.lineWidth=3;ctx.strokeStyle='#50dfd2';ctx.fillStyle='#ffd281';ctx.font=`bold ${Math.max(16,s.width/40)}px sans-serif`;
  s.corners.forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0],p[1],s.width/100,0,Math.PI*2);ctx.fill();ctx.fillText(String(i+1),p[0]+s.width/75,p[1]);if(i){ctx.beginPath();ctx.moveTo(...s.corners[i-1]);ctx.lineTo(...p);ctx.stroke();}});
  $('corner-count-'+s.id).textContent=`四隅を ${s.corners.length} / 4点選択`;
}
$('export-calibration').onclick=()=>saveJSON('wholebody-calibration.json',{schemaVersion:1,cameras:sources.map(s=>s.calibration).filter(Boolean),heightCm:number('height'),videoOffsetSeconds:number('time-offset')});
$('import-calibration').onchange=guarded(async event=>{
  const file=event.target.files[0];if(!file)return;if(file.size>1000000)throw Error('校正ファイルが大きすぎます');const data=JSON.parse(await file.text());
  if(!Array.isArray(data.cameras)||data.cameras.length>2||data.cameras.some(c=>!['a','b'].includes(c.id)||!validateCamera(c,true)||!(c.width>0&&c.height>0)))throw Error('校正JSONの形式が不正です');
  if(new Set(data.cameras.map(c=>c.id)).size!==data.cameras.length)throw Error('カメラIDが重複しています');
  await waitIdle();for(const c of data.cameras)sources.find(s=>s.id===c.id).calibration=c;
  if(data.heightCm>=80&&data.heightCm<=230)$('height').value=data.heightCm;if(Number.isFinite(data.videoOffsetSeconds))$('time-offset').value=data.videoOffsetSeconds;
  $('calibration-result').textContent=`${data.cameras.length}視点の校正を読み込みました。入力映像と解像度・向きを確認してください。`;resetHistory();
});
for(const id of['height','time-offset','files-synchronized','allow-approximate','mono-fov'])$(id).onchange=resetHistory;

async function makePeer(slot){
  state.peer[slot]?.close();const s=sources.find(x=>x.id===slot);
  const peer=new CameraPeer({slot,onStream:stream=>{if(state.peer[slot]===peer)streamSource(s,stream,'remote').catch(fail);},onStatus:message=>{if(state.peer[slot]!==peer)return;$('signal-state').textContent=`カメラ${slot.toUpperCase()}: ${message}`;if(['failed','disconnected','closed'].includes(message)){s.ready=false;resetHistory();setRunControls();}},onClock:clock=>{if(state.peer[slot]===peer)s.clock=clock;}});
  state.peer[slot]=peer;return peer;
}
for(const slot of['a','b'])$('offer-'+slot).onclick=guarded(async()=>{const peer=await makePeer(slot);$('signal-text').value=JSON.stringify(await peer.offer());});
$('apply-answer').onclick=guarded(async()=>{const answer=JSON.parse($('signal-text').value);const peer=state.peer[answer.slot];if(!peer)throw Error('先にPC側の接続情報を作成してください');await peer.accept(answer);});
$('copy-signal').onclick=guarded(async()=>{await navigator.clipboard.writeText($('signal-text').value);$('signal-state').textContent='コピーしました';});
$('connect-signal').onclick=guarded(async()=>{
  state.signal?.stop();const client=new SignalClient({url:$('signal-url').value,token:$('signal-token').value,room:$('room').value,peer:'pc',onStatus:text=>{$('signal-state').textContent=text;},onMessage:async m=>{
    const slot=m.from==='camera-a'?'a':m.from==='camera-b'?'b':null;if(!slot)return;
    if(m.payload.type==='join'){const peer=await makePeer(slot);await client.send(m.from,{type:'offer',offer:await peer.offer()});}
    else if(m.payload.type==='answer'&&state.peer[slot])await state.peer[slot].accept(m.payload.answer);
  }});state.signal=client;await client.start();
});
$('export-records').onclick=()=>saveJSON('wholebody-observations.json',{schemaVersion:1,sessionId:'732e048e-dba4-468e-bdd2-6026afc8218d',model:engine.provenance,cameras:sources.map(s=>s.calibration).filter(Boolean),heightCm:number('height'),scoreMeaning:'model confidence; not a visibility probability',records:state.records});
$('export-network').onclick=()=>saveJSON('wholebody-connections.json',{externalFetches:NETWORK_LOG,connections:CONNECTION_LOG,note:'Application-level requests and selected WebRTC candidates. Not an OS packet capture or sandbox certification.'});
$('clear-records').onclick=()=>{state.records=[];$('record-count').textContent='0フレーム';};
async function loadEvidence(){
  try{const response=await fetch('./evidence/latest.json',{cache:'no-cache'});if(!response.ok)throw Error();const report=await response.json();const container=$('evaluation-report');container.replaceChildren();
    const text=document.createElement('p');text.textContent=report.description||'正解座標付き画像での実測。2Dの検出誤差と3Dの幾何検証は別々に評価します。';container.append(text);
    if(Array.isArray(report.rows)){const table=document.createElement('table');const head=document.createElement('tr');for(const name of['構成','体の平均誤差','足の平均誤差','手の平均誤差','手PCK20']){const th=document.createElement('th');th.textContent=name;head.append(th);}table.append(head);
      for(const row of report.rows){const tr=document.createElement('tr');for(const v of[row.name,row.bodyPx,row.footPx,row.handPx,row.handPck20]){const td=document.createElement('td');td.textContent=v;tr.append(td);}table.append(tr);}container.append(table);}
    const note=document.createElement('p');note.textContent=report.limitations||'';container.append(note);
    if(report.runUrl){const link=document.createElement('a');link.href=report.runUrl;link.textContent='この結果のActions実行 ↗';link.target='_blank';link.rel='noopener';container.append(link);}
  }catch{$('evaluation-report').textContent='評価結果を準備しています。Actionsの実行履歴から最新の記録を確認できます。';}
}
loadEvidence();world.clear('3Dの座標は推論後に表示されます');
if('serviceWorker'in navigator)navigator.serviceWorker.register('./sw.js').catch(error=>log('オフライン画面キャッシュ: '+error.message));
window.addEventListener('beforeunload',()=>{state.signal?.stop();Object.values(state.peer).forEach(p=>p.close());sources.forEach(s=>s.video.srcObject?.getTracks().forEach(t=>t.stop()));});
setRunControls();
