import {patchOrtWebGpuF32Accumulate}from './ort-patch.js';
import {hashBytes,isOffline,NETWORK_LOG}from './model-cache.js';
export const RUNTIME_VERSION='1.27.0-JSEP-f32-accumulation-patch';
const CDN='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';
const FILES={
  'ort.all.min.mjs':'a2030ec9a95584a3773d889dd2514ce42e5dcd0e6dfb8cf0c9b2e5647f1d7607',
  'ort-wasm-simd-threaded.jsep.mjs':'3ee381d20a80f51a788a1c4a5872f6f1d047538dd4342f4af00062de5f9ea4c6',
  'ort-wasm-simd-threaded.jsep.wasm':'78feeeb3d08f6bcee94d938ed322f69073bb8076b5f9d34697a574ffba8deb48'};
let runtimePromise;
export async function runtimeAsset(name){
  const url=CDN+name,expected=FILES[name];let cache;
  try{cache=await caches.open('vrc-mmpose-runtime-1.27.0');const hit=await cache.match(url);if(hit){const bytes=new Uint8Array(await hit.arrayBuffer());if(await hashBytes(bytes)===expected)return bytes;await cache.delete(url);}}catch{}
  if(isOffline())throw Error(`閉域モード: 推論エンジン ${name} が未保存です`);
  NETWORK_LOG.push({time:new Date().toISOString(),kind:'runtime',url});
  const response=await fetch(url,{credentials:'omit',signal:AbortSignal.timeout(120000)});if(!response.ok)throw Error(`推論エンジン ${name}: HTTP ${response.status}`);
  const bytes=new Uint8Array(await response.arrayBuffer());if(await hashBytes(bytes)!==expected)throw Error(`推論エンジン ${name}: SHA-256が一致しません`);
  if(cache)try{await cache.put(url,new Response(bytes));}catch{}
  return bytes;
}
export async function initRuntime(){
  if(runtimePromise)return runtimePromise;
  runtimePromise=(async()=>{
    const[js,glue,wasm]=await Promise.all(['ort.all.min.mjs','ort-wasm-simd-threaded.jsep.mjs','ort-wasm-simd-threaded.jsep.wasm'].map(runtimeAsset));
    const patched=patchOrtWebGpuF32Accumulate(new TextDecoder().decode(js));
    if(await hashBytes(new TextEncoder().encode(patched))!=='b399bba00f2e1328758b3f57d6b43ec76e1c2f49440268fe85f6894f61fc0b0a')throw Error('FP32累積パッチの照合に失敗しました');
    const url=(bytes,type)=>URL.createObjectURL(new Blob([bytes],{type}));
    const ort=await import(url(patched,'text/javascript'));
    ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
    ort.env.wasm.wasmPaths={mjs:url(glue,'text/javascript'),wasm:url(wasm,'application/wasm')};
    ort.env.webgpu.powerPreference='high-performance';ort.env.logLevel='warning';
    return ort;
  })().catch(e=>{runtimePromise=null;throw e;});
  return runtimePromise;
}
