// Original, deterministic RGB-frame sampling. Used by model-specific codecs.
export function roundEven(value){const low=Math.floor(value),fraction=value-low;return fraction<.5?low:fraction>.5?low+1:low+(low&1);}

export function clippedCrop(box,image){
  if(box.length!==4||!box.every(Number.isFinite)||box[2]<=0||box[3]<=0)throw new Error('Invalid image crop.');
  const x=Math.max(0,Math.min(image.width-1,Math.floor(box[0]))),y=Math.max(0,Math.min(image.height-1,Math.floor(box[1])));
  const right=Math.max(x+1,Math.min(image.width,Math.ceil(box[0]+box[2]))),bottom=Math.max(y+1,Math.min(image.height,Math.ceil(box[1]+box[3])));
  return [x,y,right-x,bottom-y];
}

export function resizeRgbaNchw(image,transform,{bgr=false,divisor=1}={}){
  const {cropXYWH:[ox,oy,width,height],resizedWH:[rw,rh],inputWH:[iw,ih],left,top}=transform;
  const area=iw*ih,output=new Float32Array(3*area);output.fill(114/divisor);
  const sample=(x,y,c)=>image.data[((oy+y)*image.width+ox+x)*4+c];
  for(let y=0;y<rh;y++){
    const sy=(y+.5)*height/rh-.5,lowY=Math.floor(sy),y0=Math.max(0,Math.min(height-1,lowY)),y1=Math.max(0,Math.min(height-1,lowY+1)),fy=sy-lowY;
    for(let x=0;x<rw;x++){
      const sx=(x+.5)*width/rw-.5,lowX=Math.floor(sx),x0=Math.max(0,Math.min(width-1,lowX)),x1=Math.max(0,Math.min(width-1,lowX+1)),fx=sx-lowX;
      for(let channel=0;channel<3;channel++){
        const c=bgr?2-channel:channel;
        const value=Math.round(sample(x0,y0,c)*(1-fx)*(1-fy)+sample(x1,y0,c)*fx*(1-fy)+sample(x0,y1,c)*(1-fx)*fy+sample(x1,y1,c)*fx*fy);
        output[channel*area+(top+y)*iw+left+x]=value/divisor;
      }
    }
  }
  return output;
}
