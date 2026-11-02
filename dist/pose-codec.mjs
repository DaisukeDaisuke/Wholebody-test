// Original preprocessing/decoding shared by the browser and the Actions experiment.
export const MEAN = [123.675, 116.28, 103.53];
export const STD = [58.395, 57.12, 57.375];
export const BODY_EDGES = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
const HAND_LOCAL = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]];
export const HAND_EDGES = [91,112].flatMap(offset => HAND_LOCAL.map(([a,b]) => [a+offset,b+offset]));
export const FOOT_EDGES = [[15,17],[17,18],[18,19],[19,15],[16,20],[20,21],[21,22],[22,16]];

export function cropTransform(box, inputWH, padding = 1.25) {
  if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite) || box[2] <= 0 || box[3] <= 0) throw new Error('ROI must be [x,y,width,height] with positive dimensions.');
  const [iw, ih] = inputWH;
  const center = [box[0] + box[2] / 2, box[1] + box[3] / 2];
  let width = box[2] * padding, height = box[3] * padding;
  if (width / height > iw / ih) height = width * ih / iw;
  else width = height * iw / ih;
  return {center, scale:[width,height], inputWH:[iw,ih], left:center[0]-width/2, top:center[1]-height/2};
}

export function rgbaToTensor(image, transform) {
  const {data,width,height} = image;
  const [iw,ih] = transform.inputWH;
  const output = new Float32Array(3 * iw * ih);
  const area = iw * ih;
  // Match OpenCV warpAffine INTER_LINEAR's integer-pixel convention and 1/32 interpolation table.
  // Original RGB frame is sampled once; there is no full-frame downscaling before the hand crops.
  const sample = (x,y,c) => x < 0 || y < 0 || x >= width || y >= height ? 0 : data[(y*width+x)*4+c];
  for (let y=0; y<ih; y++) {
    const sy = transform.top + y * transform.scale[1] / ih;
    let y0 = Math.floor(sy), fy = Math.round((sy-y0)*32)/32;
    for (let x=0; x<iw; x++) {
      const sx = transform.left + x * transform.scale[0] / iw;
      const x0 = Math.floor(sx), fx = Math.round((sx-x0)*32)/32;
      const index = y*iw+x;
      for (let c=0; c<3; c++) {
        const pixel = Math.round(sample(x0,y0,c)*(1-fx)*(1-fy) + sample(x0+1,y0,c)*fx*(1-fy) + sample(x0,y0+1,c)*(1-fx)*fy + sample(x0+1,y0+1,c)*fx*fy);
        output[c*area+index] = (pixel-MEAN[c])/STD[c];
      }
    }
  }
  return output;
}

function peak(data, offset, count) {
  let index=0, value=-Infinity;
  for (let j=0;j<count;j++) { const v=data[offset+j]; if (v>value) {value=v;index=j;} }
  return {index,value};
}

export function decodeSimCC(outputs, transform, {keypointCount=133,zRange=2.1744869,scoreThreshold=0.2}={}) {
  const entries = Object.entries(outputs);
  const [iw,ih] = transform.inputWH;
  const byName = axis => entries.find(([name]) => new RegExp('(?:simcc_|_)(?:'+axis+')$','i').test(name))?.[1];
  const x = byName('x') || entries[0]?.[1], y = byName('y') || entries[1]?.[1], z = byName('z') || (entries.length===3?entries[2]?.[1]:null);
  if (!x || !y || x.dims.length!==3 || y.dims.length!==3 || x.dims[1]!==keypointCount || y.dims[1]!==keypointCount || x.dims[2]!==iw*2 || y.dims[2]!==ih*2) throw new Error('Unexpected ONNX SimCC output shape. Check the selected model contract.');
  if (z && (z.dims.length!==3 || z.dims[1]!==keypointCount || z.dims[2]!==iw*2)) throw new Error('Unexpected relative-depth output shape.');
  const points = [];
  for (let i=0;i<keypointCount;i++) {
    const px=peak(x.data,i*x.dims[2],x.dims[2]), py=peak(y.data,i*y.dims[2],y.dims[2]);
    const rawScore = Math.min(px.value,py.value);
    const point = {id:i,x:transform.left+(px.index/2)*transform.scale[0]/iw,y:transform.top+(py.index/2)*transform.scale[1]/ih,score:Number.isFinite(rawScore)?Math.max(0,Math.min(1,rawScore)):0,rawScore,observable:Number.isFinite(rawScore)&&rawScore>=scoreThreshold,source:'model2D'};
    if (z) {
      const pz=peak(z.data,i*z.dims[2],z.dims[2]);
      point.relativeZ=(pz.index/2/iw*2-1)*zRange;
      point.relativeZScore=Number.isFinite(pz.value)?Math.max(0,Math.min(1,pz.value)):0;
      point.source='modelRelative3D';
    }
    points.push(point);
  }
  return points;
}

export function predictedHandBox(points, side, personBox) {
  const [offset,wrist,elbow] = side==='left'?[91,9,7]:[112,10,8];
  const hand = points.slice(offset,offset+21).filter(p=>p.score>=0.2 && Number.isFinite(p.x+p.y));
  const armValid=points[wrist]?.score>=0.2 && points[elbow]?.score>=0.2;
  const forearm=armValid?Math.hypot(points[wrist].x-points[elbow].x,points[wrist].y-points[elbow].y):0;
  let center,size;
  if (hand.length>=5) {
    const xs=hand.map(p=>p.x),ys=hand.map(p=>p.y),lo=[Math.min(...xs),Math.min(...ys)],hi=[Math.max(...xs),Math.max(...ys)];
    center=[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2];
    size=Math.max(hi[0]-lo[0],hi[1]-lo[1],forearm*0.5,personBox[3]*0.045,10);
  } else if (armValid) {
    center=[points[wrist].x+(points[wrist].x-points[elbow].x)*0.25,points[wrist].y+(points[wrist].y-points[elbow].y)*0.25];
    size=Math.max(forearm*0.8,personBox[3]*0.08,12);
  } else return null;
  size=Math.min(size,Math.max(personBox[2],personBox[3])*0.55);
  return [center[0]-size/2,center[1]-size/2,size,size];
}

export function acceptHandRefinement(points, hand, box, side) {
  const wrist=side==='left'?9:10;
  const mean=hand.reduce((a,p)=>a+(p.rawScore??p.score),0)/hand.length;
  const distance=Math.hypot(hand[0].x-points[wrist].x,hand[0].y-points[wrist].y);
  return hand.length===21 && hand.every(p=>Number.isFinite(p.x+p.y+p.score)) && mean>=0.2 && (points[wrist].score<0.3 || distance<=box[2]*0.8);
}
