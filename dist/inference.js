import {initRuntime,RUNTIME_VERSION} from './runtime.js';
import {loadModel} from './model-cache.js';
import {BODY_MODELS,HAND_MODEL,DETECTOR} from './models.js';
import {createHandRefiner} from './hand-refinement.js';
import {clamp} from './skeleton.js';

const MEAN=[123.675,116.28,103.53],STD=[58.395,57.12,57.375];
const makeCanvas=(w=1,h=1)=>{const c=document.createElement('canvas');c.width=w;c.height=h;return c;};
export function makeROI(box,width,height,padding=1.25){
  const[x1,y1,x2,y2]=box;
  if(!box.every(Number.isFinite)||x2<=x1||y2<=y1)throw Error('人物範囲が不正です');
  let w=(x2-x1)*padding,h=(y2-y1)*padding;
  if(w/h>width/height)h=w*height/width;else w=h*width/height;
  return{x:(x1+x2-w)/2,y:(y1+y2-h)/2,width:w,height:h};
}
export function decodeBody(outputs,roi,model,sourceWidth,sourceHeight){
  const points=[];const values=Object.values(outputs);
  let tx=outputs.simcc_x||values.find(t=>t.dims.join()===`1,133,${model.width*2}`);
  let ty=outputs.simcc_y||values.find(t=>t.dims.join()===`1,133,${model.height*2}`);
  const xyz=outputs.keypoints,scores=outputs.scores;
  const peak=(tensor,id)=>{const bins=tensor.dims[2];let index=0,value=-Infinity;for(let i=0;i<bins;i++){const q=tensor.data[id*bins+i];if(!Number.isFinite(q))throw Error('全身SimCCに非有限値があります');if(q>value){value=q;index=i;}}return[index/2,value];};
  if(model.dimensions===3){
    if(xyz?.dims.join()!=='1,133,3'||scores?.dims.join()!=='1,133')throw Error('RTMW3D出力の形状が契約と一致しません');
  }else if(!tx||!ty||tx.dims.join()!==`1,133,${model.width*2}`||ty.dims.join()!==`1,133,${model.height*2}`)throw Error('全身SimCC出力の形状が契約と一致しません');
  for(let id=0;id<133;id++){
    let px,py,score,relativeZm=null;
    if(model.dimensions===3){px=xyz.data[id*3];py=xyz.data[id*3+1];relativeZm=xyz.data[id*3+2];score=scores.data[id];}
    else{const ax=peak(tx,id),ay=peak(ty,id);px=ax[0];py=ay[0];score=Math.min(ax[1],ay[1]);}
    const u=roi.x+px/model.width*roi.width,v=roi.y+py/model.height*roi.height;
    const finite=[u,v,score].every(Number.isFinite)&&(relativeZm===null||Number.isFinite(relativeZm));
    if(!finite)throw Error('全身モデルが非有限座標を返しました');
    const inFrame=u>=0&&v>=0&&u<sourceWidth&&v<sourceHeight;
    points.push({id,u,v,score:inFrame?clamp(score,0,1):0,rawScore:score,relativeZm,inFrame,observable:inFrame?null:false,source:'wholebody-image',visibility:'unknown'});
  }
  return points;
}
function chooseBox(boxes,previous){
  if(!boxes.length)return null;
  const area=b=>Math.max(0,b[2]-b[0])*Math.max(0,b[3]-b[1]);
  if(!previous)return boxes.sort((a,b)=>area(b)-area(a))[0];
  const iou=b=>{const n=Math.max(0,Math.min(b[2],previous[2])-Math.max(b[0],previous[0]))*Math.max(0,Math.min(b[3],previous[3])-Math.max(b[1],previous[1]));return n/(area(b)+area(previous)-n||1);};
  return boxes.sort((a,b)=>iou(b)-iou(a))[0];
}
export class PoseEngine{
  constructor(){this.session=null;this.detector=null;this.hand=null;this.model=null;this.busy=false;this.snapshot=makeCanvas();this.crop=makeCanvas();this.detCanvas=makeCanvas(640,640);this.backend='webgpu';}
  async load({modelId='rtmpose-s',backend='webgpu',hands=true,detector=true,onProgress=()=>{}}={}){
    if(this.busy||this.loading)throw Error('推論・読み込みの完了を待ってからモデルを変更してください');
    this.loading=true;try{
    if(backend==='webgpu'&&!navigator.gpu)throw Error('このブラウザにはWebGPUがありません。PCのChrome/Edgeを使うか、WASM診断を明示選択してください。');
    await this.dispose();this.backend=backend;this.model=BODY_MODELS.find(m=>m.id===modelId);
    if(!this.model)throw Error('未知のモデルです');
    onProgress({name:'推論エンジン',phase:'runtime'});this.ort=await initRuntime();
    const body=await loadModel(this.model,p=>onProgress({name:this.model.name,...p}));
    onProgress({name:this.model.name,phase:'compile'});
    this.session=await this.ort.InferenceSession.create(body.bytes,{executionProviders:[backend],graphOptimizationLevel:'all'});
    this.provenance={body:{...this.model,sha256:body.sha256},backend,runtime:RUNTIME_VERSION};
    if(hands){const h=await loadModel(HAND_MODEL,p=>onProgress({name:'手の拡大推論',...p}));this.hand=await createHandRefiner({ort:this.ort,modelBytes:h.bytes,backend,minPointScore:.15});this.provenance.hand=HAND_MODEL;}
    if(detector){const d=await loadModel(DETECTOR,p=>onProgress({name:'人物検出',...p}));
      // Exported NonMaxSuppression runs explicitly on WASM; pose and hands retain the chosen backend.
      this.detector=await this.ort.InferenceSession.create(d.bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});this.provenance.detector={...DETECTOR,backend:'wasm'};}
    return this.provenance;
    }finally{this.loading=false;}
  }
  async detect(image,previous){
    if(!this.detector)return null;
    const w=image.width,h=image.height,scale=Math.min(640/w,640/h),ctx=this.detCanvas.getContext('2d',{willReadFrequently:true});
    ctx.fillStyle='rgb(114,114,114)';ctx.fillRect(0,0,640,640);ctx.drawImage(image,0,0,w,h,0,0,Math.round(w*scale),Math.round(h*scale));
    const rgba=ctx.getImageData(0,0,640,640).data,n=640*640,data=new Float32Array(3*n);
    for(let i=0;i<n;i++)for(let c=0;c<3;c++)data[c*n+i]=(rgba[i*4+2-c]-MEAN[2-c])/STD[2-c];
    const input=new this.ort.Tensor('float32',data,[1,3,640,640]);let out;
    try{
      out=await this.detector.run({[this.detector.inputNames[0]]:input});const values=Object.values(out),det=values.find(t=>t.dims.at(-1)===5);
      if(!det)throw Error('人物検出器の出力形式が不正です');
      const labels=values.find(t=>t!==det&&t.data.length===det.data.length/5),boxes=[];
      for(let i=0;i<det.data.length/5;i++){const a=det.data;if(a[i*5+4]<.35||(labels&&Number(labels.data[i])!==0))continue;
        const row=Array.from(a.slice(i*5,i*5+5));if(!row.every(Number.isFinite)||row[2]<=row[0]||row[3]<=row[1])continue;
        boxes.push([clamp(a[i*5]/scale,0,w),clamp(a[i*5+1]/scale,0,h),clamp(a[i*5+2]/scale,0,w),clamp(a[i*5+3]/scale,0,h)]);}
      return chooseBox(boxes,previous);
    }finally{input.dispose();Object.values(out||{}).forEach(t=>t.dispose());}
  }
  async infer(source,{box=null,automatic=false,redetect=false,hands=true,padding=1.25}={}){
    if(!this.session||this.loading)throw Error('モデルの読み込みを完了してください');
    if(this.busy)throw Error('共有推論セッションは同時実行できません');
    this.busy=true;const began=performance.now();
    try{
      const w=source.videoWidth||source.naturalWidth||source.width,h=source.videoHeight||source.naturalHeight||source.height;
      if(!w||!h)throw Error('映像フレームがまだありません');
      if(this.snapshot.width!==w||this.snapshot.height!==h){this.snapshot.width=w;this.snapshot.height=h;}
      this.snapshot.getContext('2d').drawImage(source,0,0,w,h);
      let detectionMs=0;
      if(automatic&&(!box||redetect)){const t=performance.now();const b=await this.detect(this.snapshot,box);detectionMs=performance.now()-t;if(b)box=b;else if(!box)return{missing:true,points:[],ms:performance.now()-began,width:w,height:h};}
      const roi=makeROI(box||[0,0,w,h],this.model.width,this.model.height,padding);
      this.crop.width=this.model.width;this.crop.height=this.model.height;
      const ctx=this.crop.getContext('2d',{willReadFrequently:true});ctx.fillStyle='#000';ctx.fillRect(0,0,this.crop.width,this.crop.height);
      ctx.save();ctx.setTransform(this.crop.width/roi.width,0,0,this.crop.height/roi.height,-roi.x*this.crop.width/roi.width,-roi.y*this.crop.height/roi.height);ctx.drawImage(this.snapshot,0,0);ctx.restore();
      const rgba=ctx.getImageData(0,0,this.crop.width,this.crop.height).data,n=this.crop.width*this.crop.height,data=new Float32Array(3*n);
      for(let i=0;i<n;i++)for(let c=0;c<3;c++)data[c*n+i]=(rgba[4*i+c]-MEAN[c])/STD[c];
      const input=new this.ort.Tensor('float32',data,[1,3,this.model.height,this.model.width]);let out,points,poseMs;
      try{const t=performance.now();out=await this.session.run({[this.session.inputNames[0]]:input});poseMs=performance.now()-t;points=decodeBody(out,roi,this.model,w,h);}
      finally{input.dispose();Object.values(out||{}).forEach(t=>t.dispose());}
      let handResult=null;if(hands&&this.hand){handResult=await this.hand.refine(this.snapshot,points,{imageWidth:w,imageHeight:h});points=handResult.points;}
      return{points,roi,box:box||[0,0,w,h],hands:handResult?.hands||[],width:w,height:h,ms:performance.now()-began,poseMs,handMs:handResult?.ms||0,detectionMs};
    }finally{this.busy=false;}
  }
  async dispose(){
    if(this.busy)throw Error('推論中です');
    await this.hand?.dispose();this.hand=null;await this.session?.release();this.session=null;await this.detector?.release();this.detector=null;
  }
}
