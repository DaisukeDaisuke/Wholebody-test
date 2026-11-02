import {BODY_MODELS} from './models.mjs';
export {BODY_MODELS};
export {BODY_EDGES,HAND_EDGES,FOOT_EDGES} from './pose-codec.mjs';

// One inference worker and one model session are shared by all input cameras.
// The main thread keeps the video, controls, and VRM renderer responsive while
// WASM preprocessing / fallback inference runs in the worker.
export class PoseEngine {
  constructor({onProgress=()=>{}}={}) {
    this.onProgress=onProgress;this.requests=new Map();this.sequence=0;this.closed=false;
    this.worker=new Worker(new URL('./engine-worker.mjs',import.meta.url),{type:'module',name:'wholebody-inference'});
    this.abort={abort:()=>{if(!this.closed)this.worker.postMessage({type:'abort'});}};
    this.worker.onmessage=({data})=>{
      if(data.type==='progress'){this.onProgress(data.value);return;}
      const request=this.requests.get(data.id);if(!request)return;
      this.requests.delete(data.id);
      if(data.error){const error=new Error(data.error.message);error.name=data.error.name||'Error';request.reject(error);}
      else request.resolve(data.value);
    };
    this.worker.onerror=(event)=>this.fail(new Error(event.message||'推論Workerを起動できません。ページを再読み込みしてください。'));
    this.worker.onmessageerror=()=>this.fail(new Error('推論Workerとのデータ転送に失敗しました。'));
  }
  fail(error){this.failure=error;for(const request of this.requests.values())request.reject(error);this.requests.clear();}
  call(type,value,transfer=[]){
    if(this.closed)return Promise.reject(new Error('推論Workerは終了しています。'));
    if(this.failure)return Promise.reject(this.failure);
    const id=++this.sequence;
    return new Promise((resolve,reject)=>{this.requests.set(id,{resolve,reject});try{this.worker.postMessage({id,type,value},transfer);}catch(error){this.requests.delete(id);reject(error);}});
  }
  async load(options){await this.call('load',options);return this;}
  async estimate(source,options={}){
    const started=performance.now();
    const bitmap=await createImageBitmap(source);
    try{const result=await this.call('estimate',{bitmap,options},[bitmap]);result.timings.workerRoundTripMs=performance.now()-started;return result;}
    catch(error){bitmap.close();throw error;}
  }
  resetSource(sourceId){if(!this.closed)this.worker.postMessage({type:'reset',value:sourceId});}
  async dispose(){
    if(this.closed)return;
    this.abort.abort();
    try{await this.call('dispose');}finally{this.closed=true;this.worker.terminate();this.fail(new Error('推論Workerを終了しました。'));}
  }
}
