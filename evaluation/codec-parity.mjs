// One real-image comparison, executed only by GitHub Actions.
// The independent reference is OpenCV + ONNX Runtime from benchmark.py.
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {resolve, dirname, basename} from 'node:path';
import {cropTransform, rgbaToTensor, decodeSimCC} from '../dist/pose-codec.mjs';

const argumentsList=process.argv.slice(2);
function option(name,fallback){
  const index=argumentsList.indexOf(name);
  if(index<0)return fallback;
  if(index+1>=argumentsList.length||argumentsList[index+1].startsWith('--'))throw new Error(`Missing value for ${name}`);
  return argumentsList[index+1];
}
const fixtureDirectory=resolve(option('--fixture','evaluation/results/codec-fixture'));
const outputPath=resolve(option('--output','evaluation/results/codec-parity.json'));
const tolerances={
  // A uint8 interpolation tie can change a channel by one intensity level.
  // 1 / min(STD) = 0.01751. This permits that rounding difference, not an RGB
  // swap, a half-pixel coordinate shift, or a different crop convention.
  tensorMaxAbs:Number(option('--tensor-max-abs','0.02')),
  tensorMeanAbs:Number(option('--tensor-mean-abs','0.0001')),
  transformMaxAbs:Number(option('--transform-max-abs','0.0001')),
  decodedPixelMaxAbs:Number(option('--decoded-pixel-max-abs','0.0001')),
  rawScoreMaxAbs:Number(option('--raw-score-max-abs','0.000001'))
};
if(Object.values(tolerances).some(v=>!Number.isFinite(v)||v<0))throw new Error('Tolerances must be finite and non-negative');

async function bytes(filename){
  if(basename(filename)!==filename)throw new Error('Fixture files must be direct children of the fixture directory');
  return readFile(resolve(fixtureDirectory,filename));
}
async function float32(filename){
  const buffer=await bytes(filename);
  if(buffer.byteLength%4!==0)throw new Error(`Invalid float32 byte count: ${filename}`);
  // Decode explicitly as little endian; do not depend on Buffer alignment.
  const view=new DataView(buffer.buffer,buffer.byteOffset,buffer.byteLength);
  const values=new Float32Array(buffer.byteLength/4);
  for(let i=0;i<values.length;i++)values[i]=view.getFloat32(i*4,true);
  return values;
}
function compare(actual,expected){
  if(actual.length!==expected.length)throw new Error(`Length mismatch ${actual.length} != ${expected.length}`);
  let maxAbs=0,sumAbs=0,sumSquared=0,different=0,nonFinite=0,maxIndex=-1;
  for(let i=0;i<actual.length;i++){
    const delta=Math.abs(actual[i]-expected[i]);
    if(!Number.isFinite(delta)){nonFinite++;continue;}
    if(delta>maxAbs){maxAbs=delta;maxIndex=i;}
    if(delta>1e-6)different++;
    sumAbs+=delta;sumSquared+=delta*delta;
  }
  return {elements:actual.length,maxAbs,meanAbs:sumAbs/Math.max(1,actual.length),
    rms:Math.sqrt(sumSquared/Math.max(1,actual.length)),differentAbove1eMinus6:different,
    nonFinite,maxIndex,actualAtMax:maxIndex<0?null:actual[maxIndex],expectedAtMax:maxIndex<0?null:expected[maxIndex]};
}

const fixture=JSON.parse(await readFile(resolve(fixtureDirectory,'manifest.json'),'utf8'));
if(fixture.schemaVersion!==1||fixture.image.channels!=='RGBA')throw new Error('Unsupported fixture schema');
const imageBytes=await bytes(fixture.image.file);
if(imageBytes.length!==fixture.image.width*fixture.image.height*4)throw new Error('RGBA frame byte count mismatch');
const image={data:new Uint8Array(imageBytes.buffer,imageBytes.byteOffset,imageBytes.byteLength),
  width:fixture.image.width,height:fixture.image.height};
const transform=cropTransform(fixture.boxXYWH,fixture.inputWH,fixture.padding);
const transformComparison=compare([...transform.center,...transform.scale],[...fixture.transform.center,...fixture.transform.scale]);
const tensor=rgbaToTensor(image,transform);
const expectedTensor=await float32(fixture.expectedTensor.file);
const tensorComparison=compare(tensor,expectedTensor);
const outputs={};
for(const record of fixture.outputs){
  const data=await float32(record.file);
  if(data.length!==record.shape.reduce((total,value)=>total*value,1))throw new Error(`Output size mismatch: ${record.name}`);
  outputs[record.name]={data,dims:record.shape};
}
const decoded=decodeSimCC(outputs,transform,{keypointCount:fixture.keypointCount});
if(decoded.length!==fixture.expectedPoints.length)throw new Error('Decoded keypoint count mismatch');
const coordinates=compare(decoded.flatMap(p=>[p.x,p.y]),fixture.expectedPoints.flatMap(p=>[p.x,p.y]));
const rawScores=compare(decoded.map(p=>p.rawScore),fixture.expectedPoints.map(p=>p.rawScore));
const displayScores=compare(decoded.map(p=>p.score),fixture.expectedPoints.map(p=>p.score));
const parts={transform:transformComparison,tensor:tensorComparison,coordinates,rawScores,displayScores};
const checks={
  finite:Object.values(parts).every(part=>part.nonFinite===0),
  cropTransform:transformComparison.maxAbs<=tolerances.transformMaxAbs,
  inputMaxError:tensorComparison.maxAbs<=tolerances.tensorMaxAbs,
  inputMeanError:tensorComparison.meanAbs<=tolerances.tensorMeanAbs,
  decodedCoordinates:coordinates.maxAbs<=tolerances.decodedPixelMaxAbs,
  rawConfidence:rawScores.maxAbs<=tolerances.rawScoreMaxAbs,
  displayConfidence:displayScores.maxAbs<=tolerances.rawScoreMaxAbs
};
const report={schemaVersion:1,createdAt:new Date().toISOString(),passed:Object.values(checks).every(Boolean),
  gitCommit:process.env.GITHUB_SHA??null,workflowRunId:process.env.GITHUB_RUN_ID??null,
  imageId:fixture.imageId,annotationId:fixture.annotationId,modelSha256:fixture.modelSha256,
  reference:{runtime:'OpenCV + ONNX Runtime Python',opencvVersion:fixture.opencvVersion},
  implementation:'dist/pose-codec.mjs: cropTransform / rgbaToTensor / decodeSimCC',
  tolerances,checks,comparison:parts,
  limitation:'One real-image numerical comparison. Does not execute a browser, prove WebGPU support, or establish tracking accuracy.'};
await mkdir(dirname(outputPath),{recursive:true});
await writeFile(outputPath,JSON.stringify(report,null,2)+'\n','utf8');
console.log(JSON.stringify(report,null,2));
if(!report.passed)process.exitCode=1;
