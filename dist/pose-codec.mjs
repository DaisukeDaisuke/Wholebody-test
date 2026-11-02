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
  const f = Math.fround;
  const center = [f(box[0] + box[2] / 2), f(box[1] + box[3] / 2)];
  let width = f(f(box[2]) * padding), height = f(f(box[3]) * padding);
  const aspect=iw/ih;
  if (width > f(height*aspect)) height = f(width/aspect);
  else width = f(height*aspect);
  // The independent reference creates an FP32 affine matrix, then OpenCV
  // inverts that matrix in double precision. Preserve the same pixel map.
  const ax=f(iw/width), ay=f(ih/height);
  const bx=f(iw*f(.5-f(center[0]/width))), by=f(ih*f(.5-f(center[1]/height)));
  const inverse=[1/ax,0,-bx/ax,0,1/ay,-by/ay];
  return {center, scale:[width,height], inputWH:[iw,ih], left:center[0]-width/2, top:center[1]-height/2, inverse};
}

function roundEven(value) {
  const base=Math.floor(value), fraction=value-base;
  return fraction<.5?base:fraction>.5?base+1:base+(base&1);
}

export function rgbaToTensor(image, transform) {
  const {data,width,height} = image;
  const [iw,ih] = transform.inputWH;
  const output = new Float32Array(3 * iw * ih);
  const area = iw * ih;
  // Match OpenCV warpAffine's separately rounded 10-bit affine terms and
  // subsequent 5-bit interpolation-table coordinates, including negative ROI.
  // Original RGB frame is sampled once; there is no full-frame downscaling before the hand crops.
  const sample = (x,y,c) => x < 0 || y < 0 || x >= width || y >= height ? 0 : data[(y*width+x)*4+c];
  const inverse=transform.inverse||[transform.scale[0]/iw,0,transform.left,0,transform.scale[1]/ih,transform.top];
  const xDelta=Int32Array.from({length:iw},(_,x)=>roundEven(inverse[0]*x*1024));
  const xBase=roundEven(inverse[2]*1024)+16;
  for (let y=0; y<ih; y++) {
    const iy=(roundEven((inverse[4]*y+inverse[5])*1024)+16)>>5;
    const y0=iy>>5, fy=(iy&31)/32;
    for (let x=0; x<iw; x++) {
      const ix=(xBase+xDelta[x])>>5;
      const x0=ix>>5, fx=(ix&31)/32;
      const index = y*iw+x;
      for (let c=0; c<3; c++) {
        const pixel = Math.round(sample(x0,y0,c)*(1-fx)*(1-fy) + sample(x0+1,y0,c)*fx*(1-fy) + sample(x0,y0+1,c)*(1-fx)*fy + sample(x0+1,y0+1,c)*fx*fy);
        output[c*area+index] = Math.fround(pixel-Math.fround(MEAN[c]))/Math.fround(STD[c]);
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

export function decodeSimCC(outputs, transform, {keypointCount=133,zRange=2.1744869,scoreThreshold=0.2,dimensions,simccOutputNames=null,simccInputSize=null,simccSplitRatio=2}={}) {
  const entries = Object.entries(outputs);
  const [iw,ih] = transform.inputWH;
  const describe = () => entries.map(([name,value])=>`${name}:[${value?.dims?.join(',')||'?'}]/${value?.type||'float32'}`).join(', ');
  if (!Number.isFinite(simccSplitRatio) || simccSplitRatio<=0) throw new Error('Invalid SimCC split ratio.');
  if (simccInputSize && (!Array.isArray(simccInputSize) || simccInputSize.length!==3 || simccInputSize[0]!==iw || simccInputSize[1]!==ih || !simccInputSize.every(value=>Number.isInteger(value)&&value>0))) throw new Error('SimCC input-size contract does not match the selected crop.');
  const depthSize=simccInputSize?.[2]??iw;
  const named = axis => entries.find(([name]) => new RegExp('(?:^|_)'+axis+'$','i').test(name));
  let axes;
  if (simccOutputNames) {
    axes=Object.fromEntries(['x','y','z'].map(axis=>[axis,simccOutputNames[axis]===undefined?null:[simccOutputNames[axis],outputs[simccOutputNames[axis]]]]));
  } else {
    axes={x:named('x'),y:named('y'),z:named('z')};
    // Retain the two-output legacy 2D contract. An unnamed 3D graph must bind
    // axes explicitly: X and Z can have identical shapes, so shape guessing
    // cannot disambiguate them, and numeric object keys reorder themselves.
    if ((!axes.x || !axes.y) && entries.length===2 && dimensions!==3) axes={x:entries[0],y:entries[1],z:null};
    if ((!axes.x || !axes.y) || (entries.length===3 && !axes.z)) throw new Error(`Explicit SimCC axis names are required for this graph: ${describe()}`);
  }
  const x=axes.x?.[1],y=axes.y?.[1],z=axes.z?.[1]||null;
  if (dimensions===3 && !z) throw new Error(`The selected 3D model has no bound Z head: ${describe()}`);
  if (dimensions===2 && z) throw new Error(`A 2D model unexpectedly exposes a Z head: ${describe()}`);
  for (const [axis,tensor,size] of [['x',x,iw],['y',y,ih],...(z?[['z',z,depthSize]]:[])]) {
    const bins=size*simccSplitRatio;
    if (!tensor || !Number.isInteger(bins) || tensor.dims?.length!==3 || tensor.dims[0]!==1 || tensor.dims[1]!==keypointCount || tensor.dims[2]!==bins || tensor.data?.length!==keypointCount*bins || (tensor.type && tensor.type!=='float32')) {
      throw new Error(`Unexpected ONNX SimCC ${axis.toUpperCase()} head (${axes[axis]?.[0]??'unbound'}); expected float32 [1,${keypointCount},${bins}]. Actual outputs: ${describe()}`);
    }
  }
  if (x===y || (z && (z===x || z===y))) throw new Error('SimCC axes must refer to distinct output tensors.');
  const points = [];
  for (let i=0;i<keypointCount;i++) {
    const px=peak(x.data,i*x.dims[2],x.dims[2]), py=peak(y.data,i*y.dims[2],y.dims[2]);
    const rawScore = Math.min(px.value,py.value);
    if(rawScore<=0){px.index=-1;py.index=-1;}
    const point = {id:i,x:transform.left+(px.index/simccSplitRatio)*transform.scale[0]/iw,y:transform.top+(py.index/simccSplitRatio)*transform.scale[1]/ih,score:Number.isFinite(rawScore)?Math.max(0,Math.min(1,rawScore)):0,rawScore,observable:Number.isFinite(rawScore)&&rawScore>=scoreThreshold,source:'model2D'};
    if (z) {
      const pz=peak(z.data,i*z.dims[2],z.dims[2]);
      if(rawScore<=0)pz.index=-1;
      // MMPose SimCC3DLabel uses the third codec input-size component (288 for
      // this RTMW3D-X), not the image height and not a fixed absolute root Z.
      point.relativeZ=(pz.index/simccSplitRatio/depthSize*2-1)*zRange;
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

// These are diagnostics of this candidate's own Hand5 output, not calibrated
// visibility probabilities or a comparison between different model score scales.
export function handRefinementDecision(points, hand, box, side) {
  const wrist=side==='left'?9:10,raw=hand.map(p=>p.rawScore??p.score);
  const mean=raw.reduce((sum,value)=>sum+value,0)/hand.length;
  const distance=hand[0]?Math.hypot(hand[0].x-points[wrist].x,hand[0].y-points[wrist].y):Infinity;
  const initial=hand.length===21 && hand.every((p,i)=>Number.isFinite(p.x+p.y+raw[i])) && mean>=0.2 && (points[wrist].score<0.3 || distance<=box[2]*0.8);
  if(!initial)return {accepted:false,policy:'hand_selective',rejectedBy:['initial_candidate_gate']};
  const padded=box[2]*1.25,left=box[0]-box[2]*.125,top=box[1]-box[3]*.125;
  const boundary=hand.filter(p=>{const u=(p.x-left)/padded*256,v=(p.y-top)/padded*256;return u<=1||v<=1||u>=255||v>=255;}).length;
  const confident=raw.filter(value=>value>=.2).length,wristShift=distance/box[2];
  const checks={wrist_confidence:raw[0]>=.2,wrist_agreement:wristShift<=.35,majority_confident:confident>=11,not_boundary_collapsed:boundary<=10};
  const rejectedBy=Object.entries(checks).filter(([,passed])=>!passed).map(([name])=>name);
  return {accepted:rejectedBy.length===0,policy:'hand_selective',rejectedBy,rawWristScore:raw[0],wristShiftFraction:wristShift,confidentJointCount:confident,cropBoundaryJointCount:boundary};
}

export function acceptHandRefinement(points,hand,box,side){return handRefinementDecision(points,hand,box,side).accepted;}
