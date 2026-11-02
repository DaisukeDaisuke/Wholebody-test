import fs from 'node:fs/promises';
import path from 'node:path';
import {footTransform,footTensor,decodeFoot,decodeFootCandidates,associateFeet,applyFootAssociation} from '../dist/foot-codec.mjs';
if(process.env.GITHUB_ACTIONS!=='true')throw new Error('Run this real-image parity measurement in GitHub Actions.');
const arg=name=>{const i=process.argv.indexOf(name);return i<0?null:process.argv[i+1];};
const root=arg('--fixtures')||'evaluation/foot-results/codec-fixtures',output=arg('--output')||'evaluation/foot-results/foot-codec-parity.json';
const f32=async p=>{const b=await fs.readFile(p);return new Float32Array(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
function difference(a,b){
  if(a.length!==b.length)throw new Error('Array size mismatch.');
  let maxAbs=0,total=0,nonFinite=0;for(let i=0;i<a.length;i++){if(!Number.isFinite(a[i]+b[i])){nonFinite++;continue;}const value=Math.abs(a[i]-b[i]);maxAbs=Math.max(maxAbs,value);total+=value;}
  return {elements:a.length,maxAbs,meanAbs:total/a.length,nonFinite};
}
const experiments=[];
for(const directory of await fs.readdir(root,{withFileTypes:true})){
  if(!directory.isDirectory())continue;
  const folder=path.join(root,directory.name),fixture=JSON.parse(await fs.readFile(path.join(folder,'fixture.json'),'utf8'));
  const rgba=await fs.readFile(path.join(folder,fixture.image.file)),image={width:fixture.image.width,height:fixture.image.height,data:rgba};
  const transform=footTransform(image,fixture.transform.cropXYWH),actual=footTensor(image,transform);
  const expected=await f32(path.join(folder,fixture.expectedInput.file)),tensor=difference(actual,expected);
  const modelOutput={data:await f32(path.join(folder,fixture.output.file)),dims:fixture.output.shape};
  const decoded=decodeFoot(modelOutput,fixture.model,transform);
  const coordinates=difference(decoded.points.flatMap(p=>[p.x,p.y]),fixture.expected.points.flat()),scores=difference(decoded.points.map(p=>p.rawScore),fixture.expected.keypointScores);
  const boxes=difference(decoded.bboxXYXY,fixture.expected.bboxXYXY),detector=Math.abs(decoded.detectionScore-fixture.expected.detectionScore);
  // OpenCV's uint8 resize uses fixed-point coefficients. One intensity-level
  // rounding difference is the stated tolerance, not an image accuracy claim.
  let associationCheck=null;
  if(fixture.associationExpected){
    const expectedAssociation=fixture.associationExpected;
    const baseline=expectedAssociation.baseline.map(([x,y],id)=>({id,x,y,rawScore:expectedAssociation.baselineScores[id],score:Math.max(0,Math.min(1,expectedAssociation.baselineScores[id]))}));
    const association=associateFeet(decodeFootCandidates(modelOutput,fixture.model,transform),baseline);
    const blended=baseline.map(p=>({...p})),direct=baseline.map(p=>({...p}));
    const replaced=applyFootAssociation(blended,association,{blend:true});applyFootAssociation(direct,association,{blend:false});
    const blendError=difference(blended.flatMap(p=>[p.x,p.y]),expectedAssociation.associatedBlend.flat()),directError=difference(direct.flatMap(p=>[p.x,p.y]),expectedAssociation.associated.flat());
    associationCheck={candidateMatches:(association.selected?.candidate.candidateIndex??null)===expectedAssociation.selectedCandidateIndex,jointsMatch:JSON.stringify(replaced)===JSON.stringify(expectedAssociation.replacedCocoIndices),blendError,directError};
  }
  const associationPassed=associationCheck&&associationCheck.candidateMatches&&associationCheck.jointsMatch&&associationCheck.blendError.maxAbs<=.001&&associationCheck.directError.maxAbs<=.001;
  const passed=tensor.nonFinite===0&&tensor.maxAbs<=1/255+1e-7&&tensor.meanAbs<=.001&&coordinates.maxAbs<=.001&&scores.maxAbs<=1e-6&&boxes.maxAbs<=.001&&detector<=1e-6&&associationPassed;
  experiments.push({modelId:fixture.model.id,passed,tensor,coordinates,scores,boxes,detectionScoreAbsError:detector,association:associationCheck});
}
const report={schemaVersion:1,createdAt:new Date().toISOString(),gitCommit:process.env.GITHUB_SHA,workflowRunId:process.env.GITHUB_RUN_ID,passed:experiments.length>0&&experiments.every(e=>e.passed),experiments,scope:'Real-image preprocessing and real ONNX output decoder comparison. This does not measure browser execution, WebGPU speed or generalization.',tolerances:{inputMaxAbs:1/255+1e-7,inputMeanAbs:.001,coordinatesPx:.001,rawScores:1e-6}};
await fs.mkdir(path.dirname(output),{recursive:true});await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify(report));if(!report.passed)process.exitCode=1;
