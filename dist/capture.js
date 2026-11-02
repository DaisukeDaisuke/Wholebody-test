import {CameraPeer,SignalClient} from './lan.js';
const $=id=>document.getElementById(id);let stream=null,peer=null,client=null,wake=null;
const safe=fn=>async event=>{try{$('capture-error').hidden=true;await fn(event);}catch(error){$('capture-error').hidden=false;$('capture-error').textContent=error.message;}};
function stop(){client?.stop();client=null;peer?.close();peer=null;stream?.getTracks().forEach(t=>t.stop());stream=null;$('preview').srcObject=null;wake?.release().catch(()=>{});wake=null;$('capture-status').textContent='停止しました。';}
$('start-camera').onclick=safe(async()=>{
  if(!navigator.mediaDevices?.getUserMedia)throw Error('撮影にはHTTPSが必要です。Sitesまたは信頼済みHTTPSのローカルページを開いてください。');
  stop();stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:$('capture-device').value},width:{ideal:1280},height:{ideal:720},frameRate:{ideal:Number($('capture-fps').value)}}});
  $('preview').srcObject=stream;await $('preview').play();const settings=stream.getVideoTracks()[0].getSettings();$('capture-status').textContent=`撮影中 · ${settings.width} × ${settings.height} · ${settings.frameRate?.toFixed(1)||'?'} fps`;
  if(navigator.wakeLock)wake=await navigator.wakeLock.request('screen').catch(()=>null);
});
$('stop-camera').onclick=stop;
function requireStream(){if(!stream)throw Error('先にカメラを開いてください。');}
function newPeer(slot){peer?.close();peer=new CameraPeer({slot,onStatus:status=>{$('capture-status').textContent=`カメラ ${slot.toUpperCase()} · ${status}`;}});return peer;}
$('make-answer').onclick=safe(async()=>{requireStream();const offer=JSON.parse($('offer').value);$('capture-slot').value=offer.slot;$('answer').value=JSON.stringify(await newPeer(offer.slot).answer(offer,stream));});
$('copy-answer').onclick=safe(async()=>{await navigator.clipboard.writeText($('answer').value);$('capture-status').textContent='応答をコピーしました。PCへ貼り付けてください。';});
$('join').onclick=safe(async()=>{
  requireStream();client?.stop();const slot=$('capture-slot').value;
  client=new SignalClient({url:$('capture-url').value,token:$('capture-token').value,room:$('capture-room').value,peer:'camera-'+slot,onStatus:status=>{$('capture-status').textContent=status;},onMessage:async message=>{
    if(message.from==='pc'&&message.payload.type==='offer'){const answer=await newPeer(slot).answer(message.payload.offer,stream);await client.send('pc',{type:'answer',answer});}
  });await client.start();await client.send('pc',{type:'join',slot});$('capture-status').textContent='PCの応答を待っています。';
});
if(['localhost','127.0.0.1'].includes(location.hostname)||/^192\.168\.|^10\.|^172\.(1[6-9]|2\d|3[01])\./.test(location.hostname))$('capture-url').value=location.origin;
window.addEventListener('beforeunload',stop);
