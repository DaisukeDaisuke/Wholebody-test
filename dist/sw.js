const VERSION='wholebody-app-20261102-v4';
const SETTINGS='wholebody-settings-v1';
const APP_FILES=['./','./index.html','./styles.css','./app.mjs','./evaluation-ui.mjs','./webrtc.mjs','./inference.mjs','./engine-client.mjs','./engine-worker.mjs','./foot-track.mjs','./tapir-tracker.mjs','./tapir-codec.mjs','./tapir-models.mjs','./retarget.mjs','./bridge-client.mjs','./phase3-ui.mjs','./vrm-view.mjs','./vrm-fixture.mjs','./pose-codec.mjs','./resize-codec.mjs','./foot-codec.mjs','./detector-codec.mjs','./models.mjs','./geometry.mjs','./calibration.mjs','./offline.mjs','./vendor/vrm-runtime.mjs'];
const RUNTIME='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
const RUNTIME_FILES=['ort.webgpu.min.mjs','ort-wasm-simd-threaded.jsep.mjs','ort-wasm-simd-threaded.jsep.wasm'];
let closedPromise;
let preparing=false;
function closedMode(){return closedPromise||(closedPromise=caches.open(SETTINGS).then(c=>c.match(new URL('./__closed_mode__',self.registration.scope))).then(r=>r?r.json():false));}
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
        await cache.addAll(APP_FILES.map(p=>new URL(p,self.registration.scope).href));
        for(const name of RUNTIME_FILES){
          const local=new URL('./vendor/'+name,self.registration.scope).href;
          const localResponse=await fetch(local,{credentials:'same-origin'});
          if(localResponse.ok)await cache.put(local,localResponse);
          else {const url=RUNTIME+name,response=await fetch(url,{credentials:'omit'});if(!response.ok)throw new Error('Runtime preparation failed: '+name);await cache.put(url,response);}
        }
        for(const name of ['report.json','codec-parity.json','geometry-report.json','panoptic-report.json','foot-report.json','detector-report.json','holdout-report.json']){
          const url=new URL('./evaluation/'+name,self.registration.scope).href;
          try{const response=await fetch(url);if(response.ok)await cache.put(url,response);}catch{}
        }
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
    const closed=await closedMode(),cache=await caches.open(VERSION);
    if(closed){const cached=await cache.match(request);if(cached)return cached;if(request.mode==='navigate'&&url.origin===self.location.origin){const index=await cache.match(new URL('./index.html',self.registration.scope));if(index)return index;}return new Response('Closed mode blocked an uncached request.',{status:503,headers:{'Content-Type':'text/plain'}});}
    // Network-first source files let a new deployment replace an older cached UI.
    const cacheable=request.method==='GET'&&(url.origin===self.location.origin||url.href.startsWith(RUNTIME));
    try{const response=await fetch(request);if(cacheable&&response.ok&&!/\.onnx(?:$|\?)/.test(url.pathname))await cache.put(request,response.clone());return response;}
    catch(error){const cached=await cache.match(request);if(cached)return cached;throw error;}
  })());
});
