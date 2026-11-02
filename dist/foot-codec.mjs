import {clippedCrop,roundEven,resizeRgbaNchw} from './resize-codec.mjs';
export const FOOT_COCO_INDICES=[17,18,19,15,20,21,22,16];

export function predictedFootBox(points,personBox){
  const selected=FOOT_COCO_INDICES.map(id=>points[id]).filter(p=>p.score>=.3&&Number.isFinite(p.x+p.y));
  const [x,y,w,h]=personBox;
  if(selected.length<3)return [x-w*.125,y+h*.5,w*1.25,h*.65];
  const xs=selected.map(p=>p.x),ys=selected.map(p=>p.y),loX=Math.min(...xs),hiX=Math.max(...xs),loY=Math.min(...ys),hiY=Math.max(...ys);
  const width=Math.max((hiX-loX)*1.6,w*.3,32),height=Math.max((hiY-loY)*1.8,h*.18,32);
  return [(loX+hiX-width)/2,(loY+hiY-height)/2,width,height];
}

export function footTransform(image,box,size=640){
  const crop=clippedCrop(box,image),ratio=Math.min(size/crop[2],size/crop[3]);
  const resizedWH=[roundEven(crop[2]*ratio),roundEven(crop[3]*ratio)];
  return {cropXYWH:crop,ratio,left:roundEven((size-resizedWH[0])/2-.1),top:roundEven((size-resizedWH[1])/2-.1),resizedWH,inputWH:[size,size]};
}
export function footTensor(image,transform){return resizeRgbaNchw(image,transform,{divisor:255});}

export function decodeFoot(output,spec,transform){
  const {dims,data}=output;
  if(dims.length!==3||dims[0]!==1||!data.every(Number.isFinite))throw new Error('Invalid foot model output.');
  const many=spec.head==='one2many',count=many?dims[2]:dims[1];
  if((many&&dims[1]!==29)||(!many&&(spec.head!=='one2one'||dims[2]!==30)))throw new Error('Unexpected foot output layout.');
  const value=(row,col)=>many?data[col*count+row]:data[row*30+col];
  let best=0;for(let index=1;index<count;index++)if(value(index,4)>value(best,4))best=index;
  const detectionScore=value(best,4),offset=many?5:6;
  if(!many&&roundEven(value(best,5))!==0)throw new Error('Unexpected foot detector class.');
  const toX=x=>(x-transform.left)/transform.ratio+transform.cropXYWH[0],toY=y=>(y-transform.top)/transform.ratio+transform.cropXYWH[1];
  const points=FOOT_COCO_INDICES.map((id,k)=>{
    const rawScore=value(best,offset+k*3+2);
    return {id,x:toX(value(best,offset+k*3)),y:toY(value(best,offset+k*3+1)),rawScore,score:Math.max(0,Math.min(1,detectionScore*rawScore)),observable:detectionScore>=.25&&rawScore>=.35,source:'footRefined2D'};
  });
  let box=Array.from({length:4},(_,i)=>value(best,i));
  if(many){const [cx,cy,w,h]=box;box=[cx-w/2,cy-h/2,cx+w/2,cy+h/2];}
  return {found:detectionScore>=.05,detectionScore,points,bboxXYXY:[toX(box[0]),toY(box[1]),toX(box[2]),toY(box[3])],candidateIndex:best,candidateCount:count,transform};
}

export function mergeFootRefinement(points,result){
  const accepted=[];
  if(result.detectionScore>=.25)for(const candidate of result.points){
    if(candidate.rawScore<.35)continue;
    const previous=points[candidate.id];
    points[candidate.id]={...previous,...candidate,relativeZ:previous.relativeZ,relativeZScore:previous.relativeZScore};
    accepted.push(candidate.id);
  }
  return accepted;
}

export function decodeFootCandidates(output,spec,transform){
  decodeFoot(output,spec,transform); // Verify the measured tensor contract first.
  const many=spec.head==='one2many',count=many?output.dims[2]:output.dims[1];
  const read=(row,col)=>many?output.data[col*count+row]:output.data[row*30+col],offset=many?5:6;
  const indices=Array.from({length:count},(_,i)=>i).filter(i=>read(i,4)>=.25).sort((a,b)=>read(b,4)-read(a,4)).slice(0,64);
  return indices.map(index=>({candidateIndex:index,detectionScore:read(index,4),points:FOOT_COCO_INDICES.map((id,k)=>({id,x:(read(index,offset+k*3)-transform.left)/transform.ratio+transform.cropXYWH[0],y:(read(index,offset+k*3+1)-transform.top)/transform.ratio+transform.cropXYWH[1],rawScore:read(index,offset+k*3+2)}))}));
}

export function associateFeet(candidates,baseline){
  const distance=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
  const anchors=[0,1].map(side=>{
    const knee=baseline[13+side],ankle=baseline[15+side];
    const valid=knee?.score>=.25&&ankle?.score>=.25&&Number.isFinite(knee.x+knee.y+ankle.x+ankle.y),shin=valid?distance(knee,ankle):0;
    return {ankle,shin,valid:valid&&shin>=8};
  });
  let selected=null;
  for(const candidate of candidates){
    if(candidate.detectionScore<.25)continue;
    const accepted=[],sides=[];let quality=0;
    for(let side=0;side<2;side++){
      const anchor=anchors[side],offset=side*4,ankle=candidate.points[offset+3];
      if(!anchor.valid||ankle.rawScore<.35){sides.push(false);continue;}
      const shift=distance(ankle,anchor.ankle),gate=Math.max(3,.3*anchor.shin);
      if(shift>gate){sides.push(false);continue;}
      const usable=[0,1,2].filter(i=>candidate.points[offset+i].rawScore>=.35&&distance(candidate.points[offset+i],ankle)<=Math.max(4,anchor.shin*(i===2?.45:.9)));
      if(usable.length<2){sides.push(false);continue;}
      accepted.push(...usable.map(i=>offset+i));sides.push(true);
      quality+=ankle.rawScore*Math.exp(-.5*(shift/gate)**2)*usable.length/3;
    }
    if(sides.every(Boolean)){
      const separation=distance(anchors[0].ankle,anchors[1].ankle),candidateSeparation=distance(candidate.points[3],candidate.points[7]);
      if(separation>=6&&candidateSeparation<separation*.35)continue;
    }
    quality*=candidate.detectionScore;
    if(accepted.length&&(!selected||quality>selected.quality))selected={candidate,accepted,quality};
  }
  return {selected,candidateCountEvaluated:candidates.length,policyId:'ankle-associated-v1'};
}

export function applyFootAssociation(points,association,{blend=true}={}){
  if(!association.selected)return [];
  const {candidate,accepted}=association.selected,weight=blend?.5:1,ids=[];
  for(const index of accepted){
    const refined=candidate.points[index],previous=points[refined.id];
    points[refined.id]={...previous,x:previous.x*(1-weight)+refined.x*weight,y:previous.y*(1-weight)+refined.y*weight,score:blend?previous.score:candidate.detectionScore*refined.rawScore,rawScore:blend?previous.rawScore:refined.rawScore,observable:blend?previous.observable:true,source:blend?'footAssociatedBlend2D':'footAssociated2D',footDetectionScore:candidate.detectionScore,footKeypointScore:refined.rawScore};
    ids.push(refined.id);
  }
  return ids;
}
