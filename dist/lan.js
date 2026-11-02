// Video uses a direct WebRTC connection. The signalling server exchanges SDP only.
export const CONNECTION_LOG=[];
const MAX_SDP=150000;
function validateDescription(description,type){
  if(!description||description.type!==type||typeof description.sdp!=='string'||description.sdp.length>MAX_SDP)throw Error('接続情報の形式が不正です');
  for(const line of description.sdp.split(/\r?\n/))if(line.startsWith('a=candidate:')&&!/ typ host(?: |$)/.test(line))throw Error('LAN用のhost候補以外が含まれています');
  return{type,sdp:description.sdp};
}
async function gathered(pc){
  if(pc.iceGatheringState==='complete')return;
  await new Promise((resolve,reject)=>{const timer=setTimeout(()=>{cleanup();reject(Error('LAN候補の収集が時間内に完了しませんでした'));},10000);
    const listener=()=>{if(pc.iceGatheringState==='complete'){cleanup();resolve();}};
    function cleanup(){clearTimeout(timer);pc.removeEventListener('icegatheringstatechange',listener);}pc.addEventListener('icegatheringstatechange',listener);listener();});
}
export class CameraPeer{
  constructor({slot='a',onStream=()=>{},onStatus=()=>{},onClock=()=>{}}={}){
    this.slot=slot;this.onStatus=onStatus;this.onClock=onClock;this.clock=null;this.samples=[];
    this.pc=new RTCPeerConnection({iceServers:[],bundlePolicy:'max-bundle'});
    this.pc.ontrack=event=>{if(event.track.kind==='video')onStream(event.streams[0]||new MediaStream([event.track]));};
    this.pc.onconnectionstatechange=()=>{onStatus(this.pc.connectionState);CONNECTION_LOG.push({time:new Date().toISOString(),slot,state:this.pc.connectionState});if(this.pc.connectionState==='connected')this.inspectRoute().catch(()=>{});};
    this.pc.ondatachannel=event=>this.attachClock(event.channel);
  }
  attachClock(channel){
    this.channel=channel;
    channel.onmessage=event=>{if(typeof event.data!=='string'||event.data.length>4096)return;let m;try{m=JSON.parse(event.data);}catch{return;}
      const now=performance.now();if(m.type==='clock-ping'&&Number.isFinite(m.t0)){const t1=now;channel.send(JSON.stringify({type:'clock-pong',t0:m.t0,t1,t2:performance.now()}));}
      else if(m.type==='clock-pong'&&[m.t0,m.t1,m.t2].every(Number.isFinite)){
        const rtt=(now-m.t0)-(m.t2-m.t1);if(rtt<0||rtt>2000)return;
        this.samples.push({offsetMs:((m.t1-m.t0)+(m.t2-now))/2,uncertaintyMs:rtt/2,measuredAt:now});this.samples=this.samples.slice(-20);
        this.clock=this.samples.reduce((a,b)=>a.uncertaintyMs<b.uncertaintyMs?a:b);this.onClock(this.clock);
      }};
    channel.onopen=()=>{this.clockTimer=setInterval(()=>{if(channel.readyState==='open')channel.send(JSON.stringify({type:'clock-ping',t0:performance.now()}));},1000);};
  }
  async offer(){
    this.pc.addTransceiver('video',{direction:'recvonly'});this.attachClock(this.pc.createDataChannel('clock',{ordered:false,maxRetransmits:0}));
    await this.pc.setLocalDescription(await this.pc.createOffer());await gathered(this.pc);
    return{schemaVersion:1,slot:this.slot,description:validateDescription(this.pc.localDescription,'offer')};
  }
  async answer(offer,stream){
    if(!['a','b'].includes(offer?.slot))throw Error('入力A/Bが指定されていません');this.slot=offer.slot;
    await this.pc.setRemoteDescription(validateDescription(offer.description,'offer'));
    for(const track of stream.getVideoTracks())this.pc.addTrack(track,stream);
    await this.pc.setLocalDescription(await this.pc.createAnswer());await gathered(this.pc);
    return{schemaVersion:1,slot:this.slot,description:validateDescription(this.pc.localDescription,'answer')};
  }
  async accept(answer){if(answer.slot!==this.slot)throw Error('カメラA/Bが一致しません');await this.pc.setRemoteDescription(validateDescription(answer.description,'answer'));}
  async inspectRoute(){
    const stats=await this.pc.getStats();let pair;
    for(const s of stats.values())if(s.type==='transport'&&s.selectedCandidatePairId)pair=stats.get(s.selectedCandidatePairId);
    if(!pair)for(const s of stats.values())if(s.type==='candidate-pair'&&s.state==='succeeded'&&s.nominated)pair=s;
    if(pair){const local=stats.get(pair.localCandidateId),remote=stats.get(pair.remoteCandidateId);
      CONNECTION_LOG.push({time:new Date().toISOString(),slot:this.slot,local:{type:local?.candidateType,address:local?.address,protocol:local?.protocol},remote:{type:remote?.candidateType,address:remote?.address,protocol:remote?.protocol},roundTripTime:pair.currentRoundTripTime});
      if([local,remote].some(c=>c?.candidateType&&c.candidateType!=='host')){this.close();throw Error('LAN直接接続以外を検出したため停止しました');}}
  }
  close(){clearInterval(this.clockTimer);this.channel?.close();this.pc.close();}
}
export class SignalClient{
  constructor({url,token,room='wholebody',peer='pc',onMessage=()=>{},onStatus=()=>{}}){
    const parsed=new URL(url||location.origin);if(parsed.protocol!=='https:'&&!(parsed.protocol==='http:'&&['localhost','127.0.0.1','[::1]'].includes(parsed.hostname)))throw Error('スマホ接続はHTTPSのローカルサーバーを指定してください');
    if(!/^[a-zA-Z0-9_-]{1,48}$/.test(room))throw Error('ルーム名には半角英数・_・-を使ってください');
    this.url=parsed.origin;this.token=token;this.room=room;this.peer=peer;this.onMessage=onMessage;this.onStatus=onStatus;this.stopped=false;this.controller=new AbortController();
  }
  async request(path,body){
    const response=await fetch(this.url+path,{method:body?'POST':'GET',headers:{Authorization:`Bearer ${this.token}`,...(body?{'Content-Type':'application/json'}:{})},body:body?JSON.stringify(body):undefined,credentials:'omit',signal:AbortSignal.any([this.controller.signal,AbortSignal.timeout(12000)])});
    if(!response.ok)throw Error(`接続サーバー HTTP ${response.status}`);return response.json();
  }
  async start(){await this.request('/api/health');this.onStatus('接続情報を待っています');this.poll();}
  send(to,payload){return this.request('/api/signal',{room:this.room,to,from:this.peer,payload});}
  async poll(){while(!this.stopped){try{const data=await this.request(`/api/signal?room=${encodeURIComponent(this.room)}&peer=${encodeURIComponent(this.peer)}`);for(const m of data.messages||[])await this.onMessage(m);}
      catch(error){if(this.stopped)return;this.onStatus(error.message);}await new Promise(r=>{this.timer=setTimeout(r,700);});}}
  stop(){this.stopped=true;this.controller.abort();clearTimeout(this.timer);}
}
