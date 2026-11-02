/** Real Node ONNX inference on public, labelled body13 regression images.
 * Missing predictions count as failures. Media and per-image landmarks are
 * runner-local only; the persisted report contains aggregate errors/coverage. */
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {createRequire} from 'node:module';
import {spawnSync} from 'node:child_process';
import {performance} from 'node:perf_hooks';
import {COCO59,cropTransform,decodeSimCC,toCoco133} from '../dist/compact-codec.mjs';
import {createPose3DCrop,samplePose3DRgb,decodePose3D} from '../dist/pose3d.mjs';
import {REFERENCE_133,modelBytes,rgbCropTensor} from './headless-video.mjs';

if(process.env.GITHUB_ACTIONS!=='true')throw new Error('GitHub Actions required');
const require=createRequire(resolve('.cache/headless/package.json')),ort=require('onnxruntime-node');ort.env.logLevel='error';
const input=JSON.parse(await readFile('.cache/labelled-pose3d/input.json','utf8')),manifest=JSON.parse(await readFile('dist/models.json','utf8'));
const BODY=[0,5,6,7,8,9,10,11,12,13,14,15,16],models=[];
const accumulator=()=>({labelled:0,visible:0,accepted:0,correct:0,visibleCorrect:0,acceptedCorrect:0,errorSum:0,finitePredictions:0,imagesWithAtLeast8Correct:0,imagesWithAtLeast8Accepted:0,latencyMs:[]});
try{
  for(const [role,spec,count] of [['compact59',manifest.bodies.standard,59],['reference133',REFERENCE_133,133],['pose3d',manifest.pose3d,33]]){
    models.push({role,spec,count,result:accumulator(),session:await ort.InferenceSession.create(await modelBytes(spec,'dist/models.json'),{executionProviders:['cpu'],intraOpNumThreads:2,interOpNumThreads:1,graphOptimizationLevel:'all'})});
  }
  for(const record of input.records){
    const decoded=spawnSync('ffmpeg',['-nostdin','-v','error','-threads','1','-i',record.path,'-frames:v','1','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],{maxBuffer:record.width*record.height*3+65536});
    if(decoded.status!==0||decoded.stdout.length!==record.width*record.height*3)throw new Error('Public JPEG decode failed');
    const rgb=decoded.stdout,size={width:record.width,height:record.height};let anchors=[];
    const common=cropTransform(record.bbox,[192,256],1.25);
    for(const model of models){
      const learned=model.role==='pose3d',shape=learned?[1,256,256,3]:model.spec.input.shape;
      const crop=learned?createPose3DCrop(record.bbox,anchors,size):cropTransform(record.bbox,[shape[3],shape[2]],1.25);
      const data=learned?samplePose3DRgb(rgb,record.width,record.height,crop):rgbCropTensor(rgb,record.width,record.height,crop,model.spec.input.mean,model.spec.input.std);
      const tensor=new ort.Tensor('float32',data,shape);let outputs;
      try{
        if(!model.warmed){for(let j=0;j<3;j++){const warm=await model.session.run({[model.session.inputNames[0]]:tensor});Object.values(warm).forEach(t=>t.dispose?.());}model.warmed=true;}
        const start=performance.now();outputs=await model.session.run({[model.session.inputNames[0]]:tensor});model.result.latencyMs.push(performance.now()-start);
        if(!Object.values(outputs).every(t=>t.data.every(Number.isFinite)))throw new Error('Nonfinite public model output');
        let points;
        if(learned)points=decodePose3D(outputs,crop).keypoints;
        else{const decodeWH=model.spec.decodeWH||[shape[3],shape[2]],split=model.spec.simcc_split_ratio||2,values=Object.values(outputs);
          const x=values.find(t=>t.data.length===model.count*decodeWH[0]*split),y=values.find(t=>t.data.length===model.count*decodeWH[1]*split);
          const raw=decodeSimCC(x.data,y.data,crop,model.count,split,decodeWH);points=model.count===59?toCoco133(raw,model.spec.coco133_indices||COCO59):raw;
        }
        if(model.role==='compact59')anchors=points;
        let accepted=0,correct=0;
        for(const id of BODY){
          const visibility=record.keypoints[id*3+2];if(!visibility)continue;
          const r=model.result,p=points[id];r.labelled++;if(visibility===2)r.visible++;
          const finite=p&&Number.isFinite(p.x)&&Number.isFinite(p.y),usable=finite&&p.score>=.25&&p.observable!==false;
          const error=finite?Math.hypot((p.x-record.keypoints[id*3])*192/common.width,(p.y-record.keypoints[id*3+1])*256/common.height):Infinity;
          if(finite){r.errorSum+=error;r.finitePredictions++;}
          if(usable){accepted++;r.accepted++;}
          if(usable&&error<=12.8){correct++;r.correct++;r.acceptedCorrect++;if(visibility===2)r.visibleCorrect++;}
        }
        if(accepted>=8)model.result.imagesWithAtLeast8Accepted++;if(correct>=8)model.result.imagesWithAtLeast8Correct++;
      }finally{tensor.dispose?.();if(outputs)Object.values(outputs).forEach(t=>t.dispose?.());}
    }
  }
  const results=Object.fromEntries(models.map(m=>{const r=m.result,t=r.latencyMs.slice().sort((a,b)=>a-b);return[m.role,{id:m.spec.id,sha256:m.spec.sha256,images:input.records.length,labelledBodyPoints:r.labelled,visibleLabelledPoints:r.visible,
    acceptedPointCoverage:r.accepted/r.labelled,pck005CountingRejectedAsIncorrect:r.correct/r.labelled,visiblePck005CountingRejectedAsIncorrect:r.visibleCorrect/r.visible,
    pck005AmongAcceptedOnly:r.accepted?r.acceptedCorrect/r.accepted:null,imagesWithAtLeast8Accepted:r.imagesWithAtLeast8Accepted,imagesWithAtLeast8Correct:r.imagesWithAtLeast8Correct,
    inferenceMedianMs:t[Math.floor((t.length-1)*.5)],inferenceP95Ms:t[Math.floor((t.length-1)*.95)]}];}));
  const report={schema:'vrc-labelled-body13-v1',status:'completed',sourceCommit:process.env.GITHUB_SHA,runId:process.env.GITHUB_RUN_ID,
    protocol:{dataset:'COCO-WholeBody val2017',selection:'first32IdsOfPreviouslyFixedFast59TestSplit',freshHoldout:false,modelSelectionPerformed:false,groundTruthPersonBoxesSupplied:true,pose3dCropAnchors:'compact59 predictions, no ground-truth joint coordinates',points:BODY,canonicalCrop:[192,256],thresholdCanonicalPixels:12.8,confidenceThreshold:.25,rejectedPredictionsCountAsIncorrect:true,sourceAnnotationSha256:input.sourceAnnotationSha256,privateInputsUsed:false},models:results,accuracy3D:null,
    limitations:['2D labelled regression only; not a 3D ground truth evaluation.','Only 32 existing heldout images, not full COCO AP.','Public image identity split was fixed previously; current model/crop was not selected on these outputs.']};
  const target='dist/evidence/labelled-body13.json';await mkdir(dirname(target),{recursive:true});await writeFile(target,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({status:report.status,images:input.records.length,models:results,mediaOutput:false}));
}finally{for(const model of models)await model.session.release();}
