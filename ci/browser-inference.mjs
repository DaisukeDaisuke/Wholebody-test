// One Actions-only execution of the actual shipped browser runtime on public data.
// This is a deployment-contract check, not an additional accuracy benchmark.
import {createServer} from 'node:http';
import {readFile, writeFile} from 'node:fs/promises';
import {resolve, extname, sep} from 'node:path';
import {createRequire} from 'node:module';

const require = createRequire(resolve('.cache/browser/package.json'));
const {chromium} = require('playwright');
const split = JSON.parse(await readFile('artifacts/compact59/public-split.json', 'utf8'));
const publicImageId = split.evaluation_image_ids[0];
if (!Number.isSafeInteger(publicImageId)) throw new Error('Missing held-out public image ID');
const imageUrl = `https://s3.amazonaws.com/images.cocodataset.org/val2017/${String(publicImageId).padStart(12,'0')}.jpg`;
const response = await fetch(imageUrl, {signal: AbortSignal.timeout(30000)});
if (!response.ok) throw new Error(`Public image fetch: ${response.status}`);
const publicImage = Buffer.from(await response.arrayBuffer());
const root = resolve('dist');
const mime = {'.html':'text/html; charset=utf-8','.mjs':'text/javascript','.js':'text/javascript',
  '.json':'application/json','.css':'text/css','.wasm':'application/wasm','.onnx':'application/octet-stream'};
const server = createServer(async (req, res) => {
  try {
    const path = new URL(req.url, 'http://localhost').pathname;
    if (path === '/__public_ci_input.jpg') {
      res.writeHead(200, {'Content-Type':'image/jpeg'}); res.end(publicImage); return;
    }
    const file = resolve(root, '.' + (path === '/' ? '/index.html' : path));
    if (!file.startsWith(root + sep)) {res.writeHead(403); res.end(); return;}
    const bytes = await readFile(file);
    res.writeHead(200, {'Content-Type':mime[extname(file)] || 'application/octet-stream'});
    res.end(bytes);
  } catch {res.writeHead(404); res.end();}
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
let browser;
try {
  browser = await chromium.launch({headless:true});
  const page = await browser.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto(origin, {waitUntil:'networkidle'});
  const result = await page.evaluate(async () => {
    const ort = await import('./vendor/ort.webgpu.min.mjs');
    const codec = await import('./compact-codec.mjs');
    const {PartModels, faceRoiFrom59, footRoiFrom59} = await import('./parts.mjs');
    ort.env.wasm.wasmPaths = new URL('./vendor/', location.href).href;
    ort.env.wasm.numThreads = 1;
    const manifest = await (await fetch('./models.json')).json();
    const inputImage = new Image();
    inputImage.src = './__public_ci_input.jpg';
    await inputImage.decode();
    const box = [0,0,inputImage.width,inputImage.height];
    const crop = codec.cropTransform(box,[192,256],manifest.body.input.crop.padding);
    const canvas = document.createElement('canvas'); canvas.width=192; canvas.height=256;
    const context = canvas.getContext('2d',{willReadFrequently:true});
    context.fillStyle='#000'; context.fillRect(0,0,192,256);
    context.drawImage(inputImage,crop.x,crop.y,crop.width,crop.height,0,0,192,256);
    const tensor = new ort.Tensor('float32',codec.rgbaToTensor(context.getImageData(0,0,192,256).data,192,256,manifest.body.input.mean,manifest.body.input.std),[1,3,256,192]);
    const session = await ort.InferenceSession.create(await (await fetch(manifest.body.url)).arrayBuffer(),{executionProviders:['wasm']});
    const outputs = await session.run({[manifest.body.input.name]:tensor});
    const shapes = Object.fromEntries(Object.entries(outputs).map(([name,value])=>[name,value.dims]));
    if (JSON.stringify(shapes.simcc_x)!=='[1,59,384]' || JSON.stringify(shapes.simcc_y)!=='[1,59,512]') throw new Error('Shipped browser model output shape');
    if (!Object.values(outputs).every(output=>output.data.every(Number.isFinite))) throw new Error('Non-finite browser model output');
    const points = codec.decodeSimCC(outputs.simcc_x.data,outputs.simcc_y.data,crop,59,manifest.body.simcc_split_ratio);
    if (points.length!==59 || !points.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y))) throw new Error('Browser keypoint decode');
    const parts = new PartModels(ort,manifest.parts,{executionProviders:['wasm']});
    const face = await parts.runFace(inputImage,faceRoiFrom59(points)?.roi || box);
    const foot = await parts.runFoot(inputImage,footRoiFrom59(points,'left')?.roi || box);
    if (face.points.length!==68 || !face.points.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y))) throw new Error('Browser face output');
    if (foot.probabilities.length!==3 || !foot.probabilities.every(Number.isFinite) || Math.abs(foot.probabilities.reduce((a,b)=>a+b,0)-1)>.0001) throw new Error('Browser foot output');
    const summary = {backend:'wasm',bodyOutputShapes:shapes,decodedBodyPoints:points.length,
      facePoints:face.points.length,footProbabilities:foot.probabilities.length,
      imageWidth:inputImage.width,imageHeight:inputImage.height,
      accuracyMeasured:false,hardwareGpuMeasured:false,privateInputsUsed:false};
    for (const value of Object.values(outputs)) value.dispose?.();
    tensor.dispose?.(); await session.release(); await parts.dispose();
    return summary;
  });
  if (pageErrors.length) throw new Error('Shipped page errors: '+pageErrors.join(' / '));
  const report = {status:'passed',runUrl:`https://github.com/${process.env.GITHUB_REPOSITORY}/actions/runs/${process.env.GITHUB_RUN_ID}`,
    sourceCommit:process.env.GITHUB_SHA,publicImageId,...result,
    scope:'One real public image through the shipped ORT Web/WASM. Output contracts only; accuracy is evaluated separately. No media is stored in the artifact.'};
  await writeFile('dist/evidence/browser-inference.json',JSON.stringify(report,null,2)+'\n','utf8');
  console.log(JSON.stringify(report));
} finally {
  await browser?.close();
  await new Promise(resolve => server.close(resolve));
}
