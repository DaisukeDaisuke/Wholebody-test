import {modelURL} from './models.js';
export const NETWORK_LOG=[];
let offline=false;
export function setOffline(value){offline=!!value;}
export function isOffline(){return offline;}
export async function hashBytes(bytes){return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');}
export async function cachedFetch(url,{expectedHash,size,onProgress=()=>{},cacheName='wholebody-models-v1'}={}){
  let cache; try{cache=await caches.open(cacheName);}catch{}
  const valid=async bytes=>{
    if(size&&bytes.byteLength!==size)throw Error(`ファイル長が一致しません: ${bytes.byteLength} / ${size}`);
    const hash=await hashBytes(bytes);if(expectedHash&&hash!==expectedHash)throw Error('SHA-256が一致しません');return hash;
  };
  if(cache){const hit=await cache.match(url);if(hit){const bytes=new Uint8Array(await hit.arrayBuffer());try{const sha256=await valid(bytes);onProgress({phase:'cache',received:bytes.length,total:bytes.length});return{bytes,sha256,source:'cache'};}catch{await cache.delete(url);}}}
  if(offline)throw Error('閉域モード: このファイルはキャッシュにありません。事前にオンラインで読み込んでください。');
  NETWORK_LOG.push({time:new Date().toISOString(),kind:'model',url});
  const res=await fetch(url,{credentials:'omit',referrerPolicy:'no-referrer',signal:AbortSignal.timeout(300000)});
  if(!res.ok)throw Error(`モデル取得 HTTP ${res.status}`);
  const total=Number(res.headers.get('content-length'))||size||0,chunks=[];let received=0;
  const reader=res.body.getReader();for(;;){const{done,value}=await reader.read();if(done)break;chunks.push(value);received+=value.length;onProgress({phase:'download',received,total});}
  const bytes=new Uint8Array(received);let offset=0;for(const c of chunks){bytes.set(c,offset);offset+=c.length;}
  onProgress({phase:'verify',received,total:received});const sha256=await valid(bytes);
  if(cache)try{await cache.put(url,new Response(bytes,{headers:{'Content-Type':'application/octet-stream'}}));}catch{}
  return{bytes,sha256,source:'network'};
}
export function loadModel(model,onProgress){return cachedFetch(modelURL(model),{expectedHash:model.sha256,size:model.size,onProgress});}
