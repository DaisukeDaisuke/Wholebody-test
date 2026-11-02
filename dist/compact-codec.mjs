export const COCO59 = [0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
  ...[91,112].flatMap(base => [1,2,4,5,6,8,9,10,12,13,14,16,17,18,20].map(i=>base+i)),
  ...[37,41,44,46,48,54,62,66,2,14].map(i=>23+i)];
export const COMPACT_EDGES = [[1,2],[1,3],[3,5],[2,4],[4,6],[1,7],[2,8],[7,8],[7,9],[9,11],[8,10],[10,12],
  [11,15],[15,13],[15,14],[13,14],[12,18],[18,16],[18,17],[16,17],
  ...[19,34].flatMap((base,side)=>Array.from({length:5},(_,f)=>[[side?6:5,base+f*3],[base+f*3,base+f*3+1],[base+f*3+1,base+f*3+2]]).flat()),
  [49,50],[51,52],[53,55],[55,54],[54,56],[56,53],[57,58]];
export function cropTransform(box, inputWH=[192,256], padding=1.25) {
  const [x,y,w,h]=box, [iw,ih]=inputWH;
  if(![x,y,w,h,iw,ih,padding].every(Number.isFinite)||w<=0||h<=0||iw<=0||ih<=0||padding<=0) throw new Error('切り出し範囲が不正です。');
  let width=w*padding,height=h*padding;
  if(width/height>iw/ih)height=width*ih/iw;else width=height*iw/ih;
  return {x:x+w/2-width/2,y:y+h/2-height/2,width,height,inputWH};
}
export function rgbaToTensor(rgba,width,height,mean=[123.675,116.28,103.53],std=[58.395,57.12,57.375]) {
  const n=width*height, data=new Float32Array(n*3);
  for(let i=0;i<n;i++)for(let c=0;c<3;c++)data[c*n+i]=(rgba[i*4+c]-mean[c])/std[c];
  return data;
}
export function decodeSimCC(x,y,crop,count=59,split=2,decodeWH=crop.inputWH) {
  const [iw,ih]=decodeWH, nx=iw*split,ny=ih*split;
  if(![iw,ih,split].every(v=>Number.isFinite(v)&&v>0)||!Number.isInteger(nx)||!Number.isInteger(ny))throw new Error('モデルの座標出力サイズが不正です。');
  if(x.length!==count*nx||y.length!==count*ny)throw new Error('モデルと59点の出力契約が一致しません。');
  return Array.from({length:count},(_,k)=>{
    let ix=0,iy=0,mx=-Infinity,my=-Infinity;
    for(let i=0;i<nx;i++){const v=x[k*nx+i];if(v>mx){mx=v;ix=i;}}
    for(let i=0;i<ny;i++){const v=y[k*ny+i];if(v>my){my=v;iy=i;}}
    const score=Math.min(mx,my),valid=Number.isFinite(score)&&score>0;
    return {id:k,x:crop.x+(valid?ix/split:-.5)*crop.width/iw,y:crop.y+(valid?iy/split:-.5)*crop.height/ih,score:valid?score:0,observable:valid};
  });
}
export function toCoco133(points,mapping=COCO59) {
  const out=Array.from({length:133},(_,id)=>({id,x:0,y:0,score:0,observable:false}));
  points.forEach((p,i)=>{if(Number.isInteger(mapping[i]))out[mapping[i]]={...p,id:mapping[i],source:'observed2D'};});
  // Display-only interpolation for the one omitted phalanx; it is never a new observation.
  for(const [base,wrist] of [[91,9],[112,10]]){
    out[base]={...out[wrist],id:base};
    for(let f=0;f<5;f++){
      const a=out[base+f*4+2],b=out[base+f*4+4],id=base+f*4+3;
      out[id]={id,x:(a.x+b.x)/2,y:(a.y+b.y)/2,score:Math.min(a.score,b.score)*.75,observable:a.observable&&b.observable,derived:true,source:'interpolatedFinger'};
    }
  }
  for(const [id,a,b] of [[1,67,69],[2,60,64]]){
    const p=out[a],q=out[b];out[id]={id,x:(p.x+q.x)/2,y:(p.y+q.y)/2,score:Math.min(p.score,q.score),observable:p.observable&&q.observable,derived:true};
  }
  return out;
}
export function boxFromPoints(points,ids,{threshold=.2,padding=1.5,minSize=24,square=true}={}) {
  const valid=ids.map(i=>points[i]).filter(p=>p?.score>=threshold&&Number.isFinite(p.x)&&Number.isFinite(p.y));
  if(valid.length<Math.min(3,ids.length))return null;
  const xs=valid.map(p=>p.x),ys=valid.map(p=>p.y),x0=Math.min(...xs),y0=Math.min(...ys),x1=Math.max(...xs),y1=Math.max(...ys);
  let w=Math.max(minSize,(x1-x0)*padding),h=Math.max(minSize,(y1-y0)*padding);if(square)w=h=Math.max(w,h);
  return [(x0+x1-w)/2,(y0+y1-h)/2,w,h];
}
export function percentile(a,p){if(!a.length)return null;const b=[...a].sort((x,y)=>x-y);return b[Math.min(b.length-1,Math.floor(p*(b.length-1)))];}
