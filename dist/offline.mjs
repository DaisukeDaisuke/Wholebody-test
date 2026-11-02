const permittedHosts = new Set(['huggingface.co','cdn.jsdelivr.net','download.openmmlab.com']);
async function workerMessage(message) {
  const registration=await navigator.serviceWorker.ready;
  const target=navigator.serviceWorker.controller||registration.active;
  if(!target)throw new Error('オフライン制御がまだ有効ではありません。');
  return new Promise((resolve,reject)=>{
    const channel=new MessageChannel();
    const timeout=setTimeout(()=>reject(new Error('オフライン制御から応答がありません。')),10000);
    channel.port1.onmessage=event=>{clearTimeout(timeout);event.data?.error?reject(new Error(event.data.error)):resolve(event.data);};
    target.postMessage(message,[channel.port2]);
  });
}
export async function prepareOffline(){
  if(!('serviceWorker' in navigator))throw new Error('このブラウザはService Workerに対応していません。');
  await navigator.serviceWorker.register('./sw.js',{scope:'./'});
  await navigator.serviceWorker.ready;
  return workerMessage({type:'prepare'});
}
export async function setClosedMode(closed){
  if(!('serviceWorker' in navigator)){if(closed)throw new Error('閉域モードにはService Workerが必要です。');return;}
  if(!navigator.serviceWorker.controller){
    if(!closed)return;
    await prepareOffline();
    if(!navigator.serviceWorker.controller)await new Promise((resolve,reject)=>{const timeout=setTimeout(()=>reject(new Error('オフライン準備後にこのページを再読込してください。')),5000);navigator.serviceWorker.addEventListener('controllerchange',()=>{clearTimeout(timeout);resolve();},{once:true});});
  }
  return workerMessage({type:'closed-mode',closed:!!closed});
}
export async function getClosedMode(){
  if(!('serviceWorker' in navigator)||!navigator.serviceWorker.controller)return false;
  const result=await workerMessage({type:'status'});return !!result.closed;
}
export function networkAudit(){
  const resources=performance.getEntriesByType('resource').map(e=>{
    const url=new URL(e.name,location.href);
    return {origin:url.origin,path:url.pathname,initiator:e.initiatorType,durationMs:Math.round(e.duration),transferBytes:e.transferSize,expected:url.origin===location.origin||permittedHosts.has(url.hostname)||url.hostname.endsWith('.hf.co')||url.hostname.endsWith('.xethub.hf.co')};
  }).filter(e=>!e.origin.startsWith('blob:'));
  return {scope:'This page resource timing and Service Worker only; not an OS/firewall packet audit.',resources,unexpected:resources.filter(e=>!e.expected),userMediaUpload:false};
}
