const CACHE='wholebody-shell-20261102-v1';
const SHELL=['./','./index.html','./style.css','./app.js','./models.js','./model-cache.js','./runtime.js','./ort-patch.js','./inference.js','./hand-refinement.js','./skeleton.js','./geometry.js','./tracking.js','./view3d.js','./lan.js','./capture.html','./capture.js'];
self.addEventListener('install',event=>event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(SHELL)).then(()=>self.skipWaiting())));
self.addEventListener('activate',event=>event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(key=>key.startsWith('wholebody-shell-')&&key!==CACHE).map(key=>caches.delete(key)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',event=>{
  const request=event.request,url=new URL(request.url);if(request.method!=='GET'||url.origin!==self.location.origin||url.pathname.startsWith('/api/'))return;
  event.respondWith(fetch(request).then(response=>{if(response.ok){const copy=response.clone();caches.open(CACHE).then(cache=>cache.put(request,copy));}return response;}).catch(()=>caches.match(request).then(response=>response||new Response('Offline asset missing',{status:503}))));
});
