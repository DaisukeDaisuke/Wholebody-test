const VERSION='wholebody-app-20261102-v6';
const SETTINGS='wholebody-settings-v1';
const PREPARED=new URL('./__app_prepared__',self.registration.scope);
const APP_FILES=['./','./index.html','./styles.css','./app.mjs','./evaluation-ui.mjs','./webrtc.mjs','./video-stepper.mjs','./inference.mjs','./wasm-loader.mjs','./engine-client.mjs','./engine-worker.mjs','./foot-track.mjs','./tapir-tracker.mjs','./tapir-codec.mjs','./tapir-models.mjs','./retarget.mjs','./bridge-client.mjs','./phase3-ui.mjs','./vrm-view.mjs','./vrm-fixture.mjs','./pose-codec.mjs','./resize-codec.mjs','./foot-codec.mjs','./detector-codec.mjs','./models.mjs','./geometry.mjs','./calibration.mjs','./offline.mjs','./vendor/vrm-runtime.mjs'];
const RUNTIME_FILES=['ort.webgpu.min.mjs','ort-wasm-simd-threaded.asyncify.mjs','ort-wasm-simd-threaded.asyncify.wasm.gz','runtime-manifest.json'];
let closedPromise;
let preparing=false;
function closedMode(){return closedPromise||(closedPromise=caches.open(SETTINGS).then(c=>c.match(new URL('./__closed_mode__',self.registration.scope))).then(r=>r?r.json():false));}
async function readableAppCache(closed){
  const current=await caches.open(VERSION);
  if(!closed)return current;
  const index=new URL('./index.html',self.registration.scope);
  const previous=(await caches.keys()).filter(name=>name.startsWith('wholebody-app-')&&name!==VERSION).sort().reverse();
  // Prefer a completed prepare over a new network-first cache that may contain
  // only part of the app. The marker is written after all required files arrive.
  for(const name of [VERSION,...previous]){const cache=name===VERSION?current:await caches.open(name);if(await cache.match(PREPARED)&&await cache.match(index))return cache;}
  // Earlier releases did not write a marker. Preserve their closed-session
  // cache as a compatibility fallback; an index alone is not a completion proof.
  for(const name of previous){const cache=await caches.open(name);if(await cache.match(index))return cache;}
  return current;
}
self.addEventListener('install',event=>{event.waitUntil(self.skipWaiting());});
self.addEventListener('activate',event=>{event.waitUntil(self.clients.claim());});
self.addEventListener('message',event=>{
  event.waitUntil((async()=>{
    try{
      if(event.data?.type==='status'){
        event.ports[0]?.postMessage({closed:await closedMode()});
      }else if(event.data?.type==='closed-mode'){
        if(preparing)throw new Error('オフライン準備の完了後に切り替えてください。');
        const closed=!!event.data.closed,cache=await caches.open(SETTINGS);
        await cache.put(new URL('./__closed_mode__',self.registration.scope),new Response(JSON.stringify(closed)));
        closedPromise=Promise.resolve(closed);event.ports[0]?.postMessage({closed});
      }else if(event.data?.type==='prepare'){
        if(await closedMode())throw new Error('先に閉域モードを解除して準備してください。');
        if(preparing)throw new Error('オフライン準備は実行中です。');
        preparing=true;
        try{
        const cache=await caches.open(VERSION);
        await cache.delete(PREPARED);
        await cache.addAll(APP_FILES.map(p=>new URL(p,self.registration.scope).href));
        for(const name of RUNTIME_FILES){
          const local=new URL('./vendor/'+name,self.registration.scope).href;
          const localResponse=await fetch(local,{credentials:'same-origin'});
          if(!localResponse.ok)throw new Error('Runtime preparation failed: '+name);
          await cache.put(local,localResponse);
        }
        for(const name of ['report.json','codec-parity.json','geometry-report.json','panoptic-report.json','foot-report.json','detector-report.json','holdout-report.json','phase3-real-model.json','phase3-tapir-wasm.json','phase3-vrm-integration.json','phase3-bridge-report.json','phase3-webgpu.json']){
          const url=new URL('./evaluation/'+name,self.registration.scope).href;
          try{const response=await fetch(url);if(response.ok)await cache.put(url,response);}catch{}
        }
        await cache.put(PREPARED,new Response(JSON.stringify({version:VERSION,preparedAt:new Date().toISOString()}),{headers:{'Content-Type':'application/json'}}));
        event.ports[0]?.postMessage({prepared:true,models:'Models are cached separately when loaded.'});
        }finally{preparing=false;}
      }
    }catch(error){event.ports[0]?.postMessage({error:error.message});}
  })());
});
self.addEventListener('fetch',event=>{
  const request=event.request,url=new URL(request.url);
  if(!['http:','https:'].includes(url.protocol))return;
  event.respondWith((async()=>{
    const closed=await closedMode(),cache=await readableAppCache(closed);
    if(closed){const cached=await cache.match(request);if(cached)return cached;if(request.mode==='navigate'&&url.origin===self.location.origin){const index=await cache.match(new URL('./index.html',self.registration.scope));if(index)return index;}return new Response('Closed mode blocked an uncached request.',{status:503,headers:{'Content-Type':'text/plain'}});}
    // Network-first source files let a new deployment replace an older cached UI.
    const cacheable=request.method==='GET'&&url.origin===self.location.origin;
    try{const response=await fetch(request);if(cacheable&&response.ok&&!/\.onnx(?:$|\?)/.test(url.pathname))await cache.put(request,response.clone());return response;}
    catch(error){const cached=await cache.match(request);if(cached)return cached;throw error;}
  })());
});
