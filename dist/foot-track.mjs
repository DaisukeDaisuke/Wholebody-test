// Image-supported 2D foot tracking. No depth, hidden toe joints, or ankle is
// manufactured from a three-point seed. All thresholds are fixed heuristics;
// scores are not calibrated probabilities.
export const FOOT_TRACK_POLICY=Object.freeze({
  id:'foot-patch-flow-v1',maxImageEdge:640,pyramidLevels:3,patchRadius:3,
  featureLimit:24,minFeatures:5,minCorrelation:.72,forwardBackwardLimit:1.5,
  inlierDistance:2.5,maxGapMs:250,detectionIntervalMs:250,minDetectionScore:.25,
  minKeypointScore:.35,maxTracks:2,reacquireIntervalMs:150,reacquireMaxAgeMs:2000,
});
const SIDES=['left','right'];
const IDS={left:[17,18,19,15],right:[20,21,22,16]};
const finitePoint=p=>p&&Number.isFinite(p[0])&&Number.isFinite(p[1]);
const dist=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
const center=points=>[points.reduce((s,p)=>s+p[0],0)/points.length,points.reduce((s,p)=>s+p[1],0)/points.length];
const median=values=>{const a=[...values].sort((x,y)=>x-y);return a.length?a[Math.floor(a.length/2)]:0;};

function grayFrame(image,maxEdge,levels){
  if(!image||!Number.isInteger(image.width)||!Number.isInteger(image.height)||image.width<8||image.height<8||image.data?.length!==image.width*image.height*4)throw new Error('FootTracker requires a complete RGBA image.');
  const scale=Math.min(1,maxEdge/Math.max(image.width,image.height));
  const width=Math.max(8,Math.round(image.width*scale)),height=Math.max(8,Math.round(image.height*scale));
  const data=new Float32Array(width*height),sx=image.width/width,sy=image.height/height;
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const px=clamp((x+.5)*sx-.5,0,image.width-1),py=clamp((y+.5)*sy-.5,0,image.height-1),x0=Math.floor(px),y0=Math.floor(py),x1=Math.min(x0+1,image.width-1),y1=Math.min(y0+1,image.height-1),ax=px-x0,ay=py-y0;
    const luminance=(ix,iy)=>{const o=(iy*image.width+ix)*4;return image.data[o]*.299+image.data[o+1]*.587+image.data[o+2]*.114;};
    data[y*width+x]=(luminance(x0,y0)*(1-ax)+luminance(x1,y0)*ax)*(1-ay)+(luminance(x0,y1)*(1-ax)+luminance(x1,y1)*ax)*ay;
  }
  const pyramid=[{width,height,data}];
  while(pyramid.length<levels){
    const p=pyramid.at(-1),w=Math.floor(p.width/2),h=Math.floor(p.height/2);if(w<20||h<20)break;
    const d=new Float32Array(w*h);
    for(let y=0;y<h;y++)for(let x=0;x<w;x++){const o=y*2*p.width+x*2;d[y*w+x]=(p.data[o]+p.data[o+1]+p.data[o+p.width]+p.data[o+p.width+1])*.25;}
    pyramid.push({width:w,height:h,data:d});
  }
  return {width,height,originalWidth:image.width,originalHeight:image.height,scaleX:width/image.width,scaleY:height/image.height,pyramid};
}

function sample(frame,x,y){
  const ix=Math.floor(x),iy=Math.floor(y);if(ix<0||iy<0||ix+1>=frame.width||iy+1>=frame.height)return NaN;
  const ax=x-ix,ay=y-iy,o=iy*frame.width+ix;
  return (frame.data[o]*(1-ax)+frame.data[o+1]*ax)*(1-ay)+(frame.data[o+frame.width]*(1-ax)+frame.data[o+frame.width+1]*ax)*ay;
}
function template(frame,point,radius){
  const [x,y]=point;if(x<radius+1||y<radius+1||x>=frame.width-radius-2||y>=frame.height-radius-2)return null;
  const values=[],n=(radius*2+1)**2;let sum=0,squares=0;
  for(let dy=-radius;dy<=radius;dy++)for(let dx=-radius;dx<=radius;dx++){const v=sample(frame,x+dx,y+dy);values.push(v);sum+=v;squares+=v*v;}
  const mean=sum/n,variance=squares-sum*sum/n;if(variance/n<12)return null;
  return {values:values.map(v=>v-mean),variance,radius,n};
}
function correlation(frame,t,x,y){
  const r=t.radius;if(x<r+1||y<r+1||x>=frame.width-r-2||y>=frame.height-r-2)return -1;
  let sum=0,sq=0,dot=0,index=0;
  for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){const value=sample(frame,x+dx,y+dy);sum+=value;sq+=value*value;dot+=t.values[index++]*value;}
  const variance=sq-sum*sum/t.n;return variance>1?dot/Math.sqrt(t.variance*variance):-1;
}
function patchMatch(previous,current,point,predicted,policy){
  let destination=[...predicted],score=-1;
  for(let level=Math.min(previous.pyramid.length,current.pyramid.length)-1;level>=0;level--){
    const a=previous.pyramid[level],b=current.pyramid[level],scale=2**level;
    const source=point.map(v=>v/scale),t=template(a,source,policy.patchRadius);if(!t)return null;
    const around=destination.map(v=>v/scale),radius=level===previous.pyramid.length-1?(policy.coarseSearchRadius??5):2;
    let best=-2,bestPoint=null;
    for(let dy=-radius;dy<=radius;dy++)for(let dx=-radius;dx<=radius;dx++){
      const x=Math.round(around[0])+dx,y=Math.round(around[1])+dy,value=correlation(b,t,x,y);
      if(value>best){best=value;bestPoint=[x,y];}
    }
    if(!bestPoint||best<.35)return null;
    // Subpixel peak fit; not a motion interpolation between detections.
    const [x,y]=bestPoint,cx0=correlation(b,t,x-1,y),cx1=correlation(b,t,x+1,y),cy0=correlation(b,t,x,y-1),cy1=correlation(b,t,x,y+1);
    const parabola=(minus,mid,plus)=>{const denominator=minus-2*mid+plus;return denominator<-.00001?clamp(.5*(minus-plus)/denominator,-.5,.5):0;};
    destination=[(x+parabola(cx0,best,cx1))*scale,(y+parabola(cy0,best,cy1))*scale];score=best;
  }
  return score>=policy.minCorrelation?{point:destination,score}:null;
}

function pointInTriangle(p,a,b,c){
  const v0=[c[0]-a[0],c[1]-a[1]],v1=[b[0]-a[0],b[1]-a[1]],v2=[p[0]-a[0],p[1]-a[1]];
  const d00=v0[0]**2+v0[1]**2,d01=v0[0]*v1[0]+v0[1]*v1[1],d02=v0[0]*v2[0]+v0[1]*v2[1],d11=v1[0]**2+v1[1]**2,d12=v1[0]*v2[0]+v1[1]*v2[1],den=d00*d11-d01*d01;
  if(Math.abs(den)<1e-8)return false;const u=(d11*d02-d01*d12)/den,v=(d00*d12-d01*d02)/den;return u>=-.08&&v>=-.08&&u+v<=1.08;
}
function featurePoints(frame,points,limit){
  const f=frame.pyramid[0],p=points.map(p=>[p[0]*frame.scaleX,p[1]*frame.scaleY]),minX=Math.max(6,Math.floor(Math.min(...p.map(q=>q[0])))),maxX=Math.min(f.width-7,Math.ceil(Math.max(...p.map(q=>q[0])))),minY=Math.max(6,Math.floor(Math.min(...p.map(q=>q[1])))),maxY=Math.min(f.height-7,Math.ceil(Math.max(...p.map(q=>q[1]))));
  const candidates=[];
  for(let y=minY;y<=maxY;y+=2)for(let x=minX;x<=maxX;x+=2){
    if(!pointInTriangle([x,y],...p.slice(0,3)))continue;
    let xx=0,xy=0,yy=0;
    for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){
      const o=(y+dy)*f.width+x+dx,gx=(f.data[o+1]-f.data[o-1])*.5,gy=(f.data[o+f.width]-f.data[o-f.width])*.5;xx+=gx*gx;xy+=gx*gy;yy+=gy*gy;
    }
    const quality=(xx+yy-Math.sqrt((xx-yy)**2+4*xy*xy))*.5;
    if(quality>30)candidates.push({point:[x,y],quality});
  }
  candidates.sort((a,b)=>b.quality-a.quality);
  const selected=[];
  // Include semantic boundaries, but do not rely exclusively on aperture-prone edges.
  for(const q of p.slice(0,3))if(template(f,q,3))selected.push(q);
  for(const candidate of candidates){if(selected.length>=limit)break;if(selected.every(q=>dist(q,candidate.point)>=5))selected.push(candidate.point);}
  return selected;
}

function solve3(matrix,rhs){
  const a=matrix.map((row,i)=>[...row,rhs[i]]);
  for(let col=0;col<3;col++){
    let pivot=col;for(let r=col+1;r<3;r++)if(Math.abs(a[r][col])>Math.abs(a[pivot][col]))pivot=r;
    if(Math.abs(a[pivot][col])<1e-8)return null;[a[col],a[pivot]]=[a[pivot],a[col]];
    const div=a[col][col];for(let j=col;j<4;j++)a[col][j]/=div;
    for(let r=0;r<3;r++)if(r!==col){const mul=a[r][col];for(let j=col;j<4;j++)a[r][j]-=mul*a[col][j];}
  }
  return a.map(row=>row[3]);
}
function fitAffine(matches){
  const m=[[0,0,0],[0,0,0],[0,0,0]],bx=[0,0,0],by=[0,0,0];
  for(const {from,to} of matches){const v=[from[0],from[1],1];for(let i=0;i<3;i++){bx[i]+=v[i]*to[0];by[i]+=v[i]*to[1];for(let j=0;j<3;j++)m[i][j]+=v[i]*v[j];}}
  const x=solve3(m,bx),y=solve3(m,by);return x&&y?[...x,...y]:null;
}
const applyAffine=(a,p)=>[a[0]*p[0]+a[1]*p[1]+a[2],a[3]*p[0]+a[4]*p[1]+a[5]];
function robustAffine(matches,policy){
  if(matches.length<policy.minFeatures)return null;
  const plausible=a=>a&&a[0]*a[4]-a[1]*a[3]>.55&&a[0]*a[4]-a[1]*a[3]<1.7&&Math.hypot(a[0],a[3])>.65&&Math.hypot(a[0],a[3])<1.5&&Math.hypot(a[1],a[4])>.65&&Math.hypot(a[1],a[4])<1.5;
  let best=null,iterations=0;
  for(let i=0;i<matches.length-2;i++)for(let j=i+1;j<matches.length-1;j++)for(let k=j+1;k<matches.length;k++){
    if(++iterations>180)break;
    const a=fitAffine([matches[i],matches[j],matches[k]]);if(!plausible(a))continue;
    const inliers=matches.filter(m=>dist(applyAffine(a,m.from),m.to)<=policy.inlierDistance);
    if(!best||inliers.length>best.inliers.length)best={a,inliers};
  }
  if(!best||best.inliers.length<policy.minFeatures||best.inliers.length<matches.length*.45)return null;
  const a=fitAffine(best.inliers);if(!plausible(a))return null;
  return {a,inliers:best.inliers,residual:median(best.inliers.map(m=>dist(applyAffine(a,m.from),m.to)))};
}

function imageContains(image,points){return points.every(p=>finitePoint(p)&&p[0]>=0&&p[1]>=0&&p[0]<image.width&&p[1]<image.height);}
function footGeometry(points){
  const toes=[(points[0][0]+points[1][0])*.5,(points[0][1]+points[1][1])*.5],length=dist(toes,points[2]),width=dist(points[0],points[1]);
  const twiceArea=Math.abs((points[1][0]-points[0][0])*(points[2][1]-points[0][1])-(points[1][1]-points[0][1])*(points[2][0]-points[0][0]));
  return {length,width,valid:length>=8&&width>=2&&width/length>=.04&&width/length<=1.5&&twiceArea/(length*length)>.008};
}
function observations(candidates,image,policy){
  const proposals=[];
  for(const candidate of candidates){
    if(!(candidate.detectionScore>=policy.minDetectionScore))continue;
    for(let side=0;side<2;side++){
      const group=candidate.points.slice(side*4,side*4+4),points=group.slice(0,3).map(p=>[p.x,p.y]);
      if(group.length!==4||group.slice(0,3).some(p=>p.rawScore<policy.minKeypointScore)||!imageContains(image,points)||!footGeometry(points).valid)continue;
      const quality=candidate.detectionScore*Math.min(...group.slice(0,3).map(p=>p.rawScore)),ankle=group[3];
      proposals.push({points,ankle:ankle.rawScore>=policy.minKeypointScore&&imageContains(image,[[ankle.x,ankle.y]])?[ankle.x,ankle.y]:null,modelSide:SIDES[side],quality,candidateIndex:candidate.candidateIndex,rawScores:group.map(p=>p.rawScore),detectionScore:candidate.detectionScore});
    }
  }
  proposals.sort((a,b)=>b.quality-a.quality);const selected=[];
  for(const candidate of proposals){if(selected.length>=policy.maxTracks)break;const c=center(candidate.points),length=footGeometry(candidate.points).length;if(selected.every(p=>dist(c,center(p.points))>Math.max(length,footGeometry(p.points).length)*.45))selected.push(candidate);}
  return selected;
}

export class FootTracker{
  constructor(options={}){this.policy={...FOOT_TRACK_POLICY,...options};this.reset();}
  reset(){this.tracks=[];this.previous=null;this.lastTimestampMs=null;this.lastDetectionMs=-Infinity;this.nextId=1;this.modelRuns=0;this.flowFrames=0;this.lastDiagnostics={};}
  needsDetection(timestampMs){
    // A manual identity survives a tracking loss. Unrelated detector output must
    // never replace it silently; reset or another explicit seed starts detection.
    if(this.tracks.some(t=>t.manual))return false;
    return !Number.isFinite(this.lastDetectionMs)||timestampMs-this.lastDetectionMs>=this.policy.detectionIntervalMs;
  }
  seed({points,side='unknown',timestampMs,image}){
    if(!Array.isArray(points)||points.length!==3||!points.every(finitePoint)||!imageContains(image,points)||!footGeometry(points).valid)throw new Error('Select three distinct visible points in order: big toe, little toe, heel.');
    if(!Number.isFinite(timestampMs))throw new Error('A frame timestamp is required for a foot seed.');
    if(!['left','right','unknown'].includes(side))throw new Error('Foot side must be left, right, or unknown.');
    const frame=grayFrame(image,this.policy.maxImageEdge,this.policy.pyramidLevels);
    const slot=side==='unknown'?(SIDES.find(s=>!this.tracks.some(t=>t.slot===s&&t.state!=='missing'))||'left'):side;
    this.tracks=this.tracks.filter(t=>t.slot!==slot);
    const track={id:`foot-${this.nextId++}`,slot,anatomicalSide:side,anatomicalSideConfirmed:side!=='unknown',modelSide:null,manual:true,points:points.map(p=>[...p]),ankle:null,quality:1,source:'manualSeed2D',state:'observed2D',lastImageSupportMs:timestampMs,lastModelObservationMs:null,lastSeedMs:timestampMs,features:featurePoints(frame,points,this.policy.featureLimit),velocity:[0,0],modelMetadata:null};
    track.reference={frame,points:track.points.map(p=>[...p]),features:track.features.map(p=>[...p]),timestampMs};track.lastReacquireMs=-Infinity;
    this.tracks.push(track);this.previous=frame;this.lastTimestampMs=timestampMs;
    return {...this.snapshot(timestampMs,{seeded:true}),trackId:track.id};
  }
  update({image,timestampMs,candidates,roi=null}){
    if(!Number.isFinite(timestampMs))throw new Error('FootTracker requires a finite frame timestamp.');
    const started=performance.now(),frame=grayFrame(image,this.policy.maxImageEdge,this.policy.pyramidLevels),dt=this.lastTimestampMs===null?0:timestampMs-this.lastTimestampMs;
    let flowAttempted=0,flowSucceeded=0,flowInliers=0,reacquired=0;
    const sameImageSize=this.previous?.originalWidth===frame.originalWidth&&this.previous?.originalHeight===frame.originalHeight;
    if(this.previous&&dt>0&&dt<=this.policy.maxGapMs&&sameImageSize){
      for(const track of this.tracks){
        if(track.state==='missing')continue;flowAttempted++;
        const matches=[];
        for(const point of track.features){
          const forward=patchMatch(this.previous,frame,point,[point[0]+track.velocity[0],point[1]+track.velocity[1]],this.policy);if(!forward)continue;
          const backward=patchMatch(frame,this.previous,forward.point,point,this.policy);if(!backward||dist(backward.point,point)>this.policy.forwardBackwardLimit)continue;
          matches.push({from:point,to:forward.point,score:Math.min(forward.score,backward.score)});
        }
        const motion=robustAffine(matches,this.policy);
        if(!motion){track.state='missing';track.source='missing';track.quality=0;track.lossReason='insufficient-image-support';continue;}
        const oldScaled=track.points.map(p=>[p[0]*frame.scaleX,p[1]*frame.scaleY]),nextScaled=oldScaled.map(p=>applyAffine(motion.a,p)),localGate=Math.max(4,footGeometry(oldScaled).length*.18),landmarkImageSupport=[false,false,false];
        // A boundary can refine the shared affine only when it independently
        // passes forward/backward appearance checks near that image-supported fit.
        for(let i=0;i<3;i++){
          const local=patchMatch(this.previous,frame,oldScaled[i],nextScaled[i],this.policy);if(!local||dist(local.point,nextScaled[i])>localGate)continue;
          const backward=patchMatch(frame,this.previous,local.point,oldScaled[i],this.policy);
          if(backward&&dist(backward.point,oldScaled[i])<=this.policy.forwardBackwardLimit){nextScaled[i]=local.point;landmarkImageSupport[i]=true;}
        }
        const next=nextScaled.map(p=>[p[0]/frame.scaleX,p[1]/frame.scaleY]);
        if(!imageContains(image,next)||!footGeometry(next).valid){track.state='missing';track.source='missing';track.quality=0;track.lossReason='outside-image-or-collapsed';continue;}
        const oldCenter=center(oldScaled),newCenter=center(nextScaled);track.velocity=[newCenter[0]-oldCenter[0],newCenter[1]-oldCenter[1]];
        track.points=next;
        if(track.ankle){const ankle=applyAffine(motion.a,[track.ankle[0]*frame.scaleX,track.ankle[1]*frame.scaleY]);track.ankle=[ankle[0]/frame.scaleX,ankle[1]/frame.scaleY];}
        track.features=featurePoints(frame,next,this.policy.featureLimit);track.state='flow2D';track.source='opticalFlow2D';track.lastImageSupportMs=timestampMs;track.lossReason=null;
        track.quality=clamp(median(motion.inliers.map(m=>m.score))*motion.inliers.length/Math.max(this.policy.minFeatures,matches.length),0,1);
        track.flow={inliers:motion.inliers.length,matches:matches.length,residualWorkingPixels:motion.residual,landmarkImageSupport};flowSucceeded++;flowInliers+=motion.inliers.length;
      }
      this.flowFrames++;
    }else if(this.previous&&(dt<0||dt>this.policy.maxGapMs||!sameImageSize)){
      for(const t of this.tracks){t.state='missing';t.source='missing';t.quality=0;t.lossReason=!sameImageSize?'image-size-change':'timestamp-discontinuity';}
    }
    for(const track of this.tracks){
      if(track.state!=='missing'||!track.reference||!sameImageSize||dt<0||timestampMs-track.lastImageSupportMs>this.policy.reacquireMaxAgeMs||timestampMs-track.lastReacquireMs<this.policy.reacquireIntervalMs)continue;
      track.lastReacquireMs=timestampMs;
      const reference=track.reference,policy={...this.policy,minCorrelation:.8,minFeatures:7,coarseSearchRadius:12},matches=[];
      for(const point of reference.features){
        const forward=patchMatch(reference.frame,frame,point,point,policy);if(!forward)continue;
        const backward=patchMatch(frame,reference.frame,forward.point,point,policy);if(!backward||dist(backward.point,point)>policy.forwardBackwardLimit)continue;
        matches.push({from:point,to:forward.point,score:Math.min(forward.score,backward.score)});
      }
      const motion=robustAffine(matches,policy);if(!motion)continue;
      const source=reference.points.map(p=>[p[0]*frame.scaleX,p[1]*frame.scaleY]),predicted=source.map(p=>applyAffine(motion.a,p)),localGate=Math.max(4,footGeometry(source).length*.18);let supported=true;
      for(let i=0;i<3;i++){
        const local=patchMatch(reference.frame,frame,source[i],predicted[i],policy);
        if(!local||dist(local.point,predicted[i])>localGate){supported=false;break;}
        const backward=patchMatch(frame,reference.frame,local.point,source[i],policy);
        if(!backward||dist(backward.point,source[i])>policy.forwardBackwardLimit){supported=false;break;}
        predicted[i]=local.point;
      }
      const points=predicted.map(p=>[p[0]/frame.scaleX,p[1]/frame.scaleY]);
      if(!supported||!imageContains(image,points)||!footGeometry(points).valid)continue;
      Object.assign(track,{points,ankle:null,features:featurePoints(frame,points,this.policy.featureLimit),velocity:[0,0],source:'opticalFlow2D',state:'flow2D',lastImageSupportMs:timestampMs,lossReason:null,quality:median(motion.inliers.map(m=>m.score)),flow:{inliers:motion.inliers.length,matches:matches.length,residualWorkingPixels:motion.residual,reacquiredFromSeed:true}});reacquired++;
    }
    if(candidates!==undefined){
      if(!Array.isArray(candidates))throw new Error('Foot candidates must be an array or undefined for a flow-only frame.');
      this.lastDetectionMs=timestampMs;this.modelRuns++;
      const proposals=observations(candidates,image,this.policy),assigned=new Set();
      for(const observation of proposals){
        if(this.tracks.some(t=>t.manual))break;
        const length=footGeometry(observation.points).length,c=center(observation.points);
        let track=this.tracks.filter(t=>!assigned.has(t.id)&&t.state!=='missing').sort((a,b)=>dist(center(a.points),c)-dist(center(b.points),c))[0];
        if(track&&dist(center(track.points),c)>Math.max(length,footGeometry(track.points).length)*.8)track=null;
        if(track?.manual){assigned.add(track.id);continue;}
        if(!track){
          const slot=SIDES.find(s=>!this.tracks.some(t=>t.slot===s&&t.state!=='missing'));
          if(!slot)continue;this.tracks=this.tracks.filter(t=>t.slot!==slot);
          track={id:`foot-${this.nextId++}`,slot,manual:false,anatomicalSide:'unknown',anatomicalSideConfirmed:false,velocity:[0,0]};this.tracks.push(track);
        }
        Object.assign(track,{points:observation.points,ankle:observation.ankle,modelSide:observation.modelSide,quality:observation.quality,source:'footModel2D',state:'observed2D',lastImageSupportMs:timestampMs,lastModelObservationMs:timestampMs,features:featurePoints(frame,observation.points,this.policy.featureLimit),lossReason:null,modelMetadata:{candidateIndex:observation.candidateIndex,detectionScore:observation.detectionScore,rawScores:observation.rawScores}});assigned.add(track.id);
      }
    }
    this.previous=frame;this.lastTimestampMs=timestampMs;
    const diagnostics={policyId:this.policy.id,modelRan:candidates!==undefined,modelCandidateCount:candidates?.length??null,modelRuns:this.modelRuns,flowFrames:this.flowFrames,flowAttempted,flowSucceeded,flowInliers,reacquired,processingMs:performance.now()-started,workingImageSize:[frame.width,frame.height],roi,depthObserved:false};
    this.lastDiagnostics=diagnostics;return this.snapshot(timestampMs,diagnostics);
  }
  snapshot(timestampMs,diagnostics=this.lastDiagnostics){
    const keypoints=Array.from({length:133},(_,id)=>({id,x:NaN,y:NaN,score:0,rawScore:0,observable:false,source:'missing'}));
    const footTracks=this.tracks.map(track=>{
      const present=track.state!=='missing',named={bigToe:present?[...track.points[0]]:null,littleToe:present?[...track.points[1]]:null,heel:present?[...track.points[2]]:null,ankle:present&&track.ankle?[...track.ankle]:null};
      if(present)for(let i=0;i<4;i++){
        const p=i===3?track.ankle:track.points[i];if(!p)continue;const id=IDS[track.slot][i];
        keypoints[id]={id,x:p[0],y:p[1],score:track.quality,rawScore:track.modelMetadata?.rawScores?.[i]??null,observable:true,source:track.source,trackId:track.id,anatomicalSide:track.anatomicalSide,anatomicalSideConfirmed:track.anatomicalSideConfirmed,modelSide:track.modelSide,depthObserved:false};
      }
      return {trackId:track.id,slot:track.slot,anatomicalSide:track.anatomicalSide,anatomicalSideConfirmed:track.anatomicalSideConfirmed,modelSide:track.modelSide,state:track.state,source:track.source,...named,score:track.quality,seededManually:track.manual,lastImageSupportMs:track.lastImageSupportMs,modelObservationAgeMs:track.lastModelObservationMs===null||track.lastModelObservationMs===undefined?null:timestampMs-track.lastModelObservationMs,lossReason:track.lossReason??null,flow:track.flow??null,depthObserved:false};
    });
    return {keypoints,footTracks,diagnostics,depthKind:'none',timestampMs};
  }
}
