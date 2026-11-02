import {resizeRgbaNchw} from './resize-codec.mjs';
export function detectorTransform(image,inputWH=[416,416]){
  const ratio=Math.min(inputWH[0]/image.width,inputWH[1]/image.height);
  return {cropXYWH:[0,0,image.width,image.height],ratio,left:0,top:0,resizedWH:[Math.max(1,Math.floor(image.width*ratio)),Math.max(1,Math.floor(image.height*ratio))],inputWH};
}
export function detectorTensor(image,transform){return resizeRgbaNchw(image,transform,{bgr:true});}
export function boxIou(a,b){
  const lowX=Math.max(a[0],b[0]),lowY=Math.max(a[1],b[1]),highX=Math.min(a[0]+a[2],b[0]+b[2]),highY=Math.min(a[1]+a[3],b[1]+b[3]);
  const overlap=Math.max(0,highX-lowX)*Math.max(0,highY-lowY);
  return overlap/Math.max(a[2]*a[3]+b[2]*b[3]-overlap,1e-12);
}
export function decodePersons(outputs,transform,{scoreThreshold=.3,nmsThreshold=.45}={}){
  const tensors=Object.values(outputs),packed=tensors.find(t=>t.dims.length===3&&t.dims[0]===1&&t.dims[2]===5),rows=[];
  if(packed){
    const labels=tensors.find(t=>t!==packed&&t.data.length===packed.dims[1]);
    if(!labels)throw new Error('Person detector labels are missing.');
    for(let i=0;i<packed.dims[1];i++)if(Number(labels.data[i])===0&&packed.data[i*5+4]>=scoreThreshold)rows.push(Array.from(packed.data.subarray(i*5,i*5+5)));
  }else{
    const [iw,ih]=transform.inputWH,expected=[8,16,32].reduce((n,s)=>n+Math.floor(iw/s)*Math.floor(ih/s),0);
    const raw=tensors.find(t=>t.dims.length===3&&t.dims[0]===1&&t.dims[1]===expected&&t.dims[2]>=6);
    if(!raw)throw new Error('Unsupported person detector output.');
    let at=0;const channels=raw.dims[2];
    for(const stride of [8,16,32])for(let y=0;y<Math.floor(ih/stride);y++)for(let x=0;x<Math.floor(iw/stride);x++,at++){
      const base=at*channels,p=raw.data,score=p[base+4]*p[base+5];if(score<scoreThreshold)continue;
      const cx=(p[base]+x)*stride,cy=(p[base+1]+y)*stride,w=Math.exp(Math.max(-20,Math.min(20,p[base+2])))*stride,h=Math.exp(Math.max(-20,Math.min(20,p[base+3])))*stride;
      rows.push([cx-w/2,cy-h/2,cx+w/2,cy+h/2,score]);
    }
  }
  const [,,width,height]=transform.cropXYWH,clip=(v,max)=>Math.max(0,Math.min(max,v/transform.ratio));
  const candidates=rows.map(row=>{const [left,top,right,bottom]=[clip(row[0],width),clip(row[1],height),clip(row[2],width),clip(row[3],height)];return {bboxXYWH:[left,top,right-left,bottom-top],score:row[4],classId:0,area:(right-left)*(bottom-top)};}).filter(p=>p.bboxXYWH.every(Number.isFinite)&&p.bboxXYWH[2]>0&&p.bboxXYWH[3]>0).sort((a,b)=>b.score-a.score);
  const kept=[];for(const candidate of candidates)if(!kept.some(other=>boxIou(other.bboxXYWH,candidate.bboxXYWH)>nmsThreshold))kept.push(candidate);
  return kept.sort((a,b)=>b.area-a.area||b.score-a.score||a.bboxXYWH[0]-b.bboxXYWH[0]||a.bboxXYWH[1]-b.bboxXYWH[1]);
}
