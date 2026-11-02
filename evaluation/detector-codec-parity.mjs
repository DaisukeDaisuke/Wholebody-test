import fs from 'node:fs/promises';
import path from 'node:path';
import {detectorTransform,detectorTensor,decodePersons} from '../dist/detector-codec.mjs';
if(process.env.GITHUB_ACTIONS!=='true')throw new Error('Run this measurement in GitHub Actions.');
const folder='evaluation/results/detector-codec-fixture';
const fixture=JSON.parse(await fs.readFile(path.join(folder,'manifest.json'),'utf8'));
const binary=async item=>{
  const b=await fs.readFile(path.join(folder,item.file)),view=new DataView(b.buffer,b.byteOffset,b.byteLength);
  const dtype=item.dtype,bytes=dtype.includes('64')?8:4,length=b.length/bytes,out=new Float64Array(length);
  for(let i=0;i<length;i++)out[i]=dtype==='int64'?Number(view.getBigInt64(i*bytes,true)):dtype==='int32'?view.getInt32(i*bytes,true):dtype==='float64'?view.getFloat64(i*bytes,true):view.getFloat32(i*bytes,true);
  return out;
};
function difference(a,b){if(a.length!==b.length)throw new Error('Detector parity array length mismatch.');let maxAbs=0,total=0;for(let i=0;i<a.length;i++){const d=Math.abs(a[i]-b[i]);if(!Number.isFinite(d))throw new Error('Non-finite comparison.');maxAbs=Math.max(maxAbs,d);total+=d;}return {elements:a.length,maxAbs,meanAbs:total/Math.max(1,a.length)};}
const image={width:fixture.image.width,height:fixture.image.height,data:await fs.readFile(path.join(folder,fixture.image.file))};
const transform=detectorTransform(image,fixture.transform.inputWH),tensor=difference(detectorTensor(image,transform),await binary(fixture.input));
const outputs={};for(const item of fixture.outputs)outputs[item.name]={dims:item.shape,data:await binary(item)};
const actual=decodePersons(outputs,transform,fixture.decoding),expected=fixture.expectedDetections;
const boxes=difference(actual.flatMap(d=>d.bboxXYWH),expected.flatMap(d=>d.bboxXYWH)),scores=difference(actual.map(d=>d.score),expected.map(d=>d.score));
const passed=tensor.maxAbs<=1&&tensor.meanAbs<=.2&&boxes.maxAbs<=.001&&scores.maxAbs<=1e-6;
const report={schemaVersion:1,createdAt:new Date().toISOString(),gitCommit:process.env.GITHUB_SHA,workflowRunId:process.env.GITHUB_RUN_ID,passed,detectedPeople:actual.length,tensor,boxes,scores,tolerances:{tensorMax:1,tensorMean:.2,boxPx:.001,score:1e-6},scope:'One real image and actual detector tensors. Numerical parity only; not browser or hardware timing.'};
await fs.writeFile('evaluation/results/detector-codec-parity.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));if(!passed)process.exitCode=1;
