import http from 'node:http';
import https from 'node:https';
import {readFile,stat} from 'node:fs/promises';
import {resolve,dirname,sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {randomBytes,timingSafeEqual} from 'node:crypto';

const args=process.argv.slice(2);
const option=(name,fallback)=>{const i=args.indexOf(name);return i<0?fallback:args[i+1];};
const root=resolve(dirname(fileURLToPath(import.meta.url)),'dist');
const cert=option('--cert'),key=option('--key');
if(Boolean(cert)!==Boolean(key))throw Error('--cert と --key を両方指定してください');
const tls=Boolean(cert&&key),host=option('--host',tls?'0.0.0.0':'127.0.0.1'),port=Number(option('--port',tls?'8443':'8080'));
if(!tls&&!['127.0.0.1','localhost','::1'].includes(host))throw Error('LANへ公開する場合は信頼済みHTTPS証明書を指定してください');
if(!Number.isInteger(port)||port<1||port>65535)throw Error('port が不正です');
const token=process.env.WHOLEBODY_SIGNAL_TOKEN||randomBytes(24).toString('base64url');
const allowedOrigins=new Set(['https://wholebody-phase2-732e048e.daisukedaisuke389137.chatgpt.site',...(process.env.WHOLEBODY_ALLOWED_ORIGINS||'').split(',').filter(Boolean)]);
const queues=new Map();
const mime={'.html':'text/html; charset=utf-8','.js':'text/javascript; charset=utf-8','.css':'text/css; charset=utf-8','.json':'application/json; charset=utf-8','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.mp4':'video/mp4','.wasm':'application/wasm'};
const allowedName=name=>typeof name==='string'&&/^[a-zA-Z0-9_-]{1,48}$/.test(name);
function authenticated(req){const value=(req.headers.authorization||'').replace(/^Bearer /,'');const a=Buffer.from(value),b=Buffer.from(token);return a.length===b.length&&timingSafeEqual(a,b);}
async function body(req){const chunks=[];let n=0;for await(const chunk of req){n+=chunk.length;if(n>200000)throw Object.assign(Error('接続情報が大きすぎます'),{status:413});chunks.push(chunk);}try{return JSON.parse(Buffer.concat(chunks).toString('utf8'));}catch{throw Object.assign(Error('JSON形式が不正です'),{status:400});}}
function json(res,status,data){res.writeHead(status,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'});res.end(JSON.stringify(data));}
async function handler(req,res){
  res.setHeader('X-Content-Type-Options','nosniff');res.setHeader('Referrer-Policy','no-referrer');
  try{
    const ownOrigin=`${tls?'https':'http'}://${req.headers.host}`;
    const origin=req.headers.origin;
    if(origin&&origin!==ownOrigin&&!allowedOrigins.has(origin)){json(res,403,{error:'許可されていないOriginです'});return;}
    if(origin){res.setHeader('Access-Control-Allow-Origin',origin);res.setHeader('Vary','Origin');}
    if(req.method==='OPTIONS'){res.writeHead(204,{'Access-Control-Allow-Methods':'GET, POST, OPTIONS','Access-Control-Allow-Headers':'Authorization, Content-Type','Access-Control-Max-Age':'600','Access-Control-Allow-Private-Network':'true'});res.end();return;}
    const url=new URL(req.url,ownOrigin);
    if(url.pathname.startsWith('/api/')){
      if(!authenticated(req)){json(res,401,{error:'接続トークンが一致しません'});return;}
      if(url.pathname==='/api/health'&&req.method==='GET'){json(res,200,{ok:true,videoRelay:false,iceServers:[],tls});return;}
      if(url.pathname!=='/api/signal'){json(res,404,{error:'not found'});return;}
      const now=Date.now();for(const[k,v]of queues)if(v.every(m=>now-m.time>60000))queues.delete(k);
      if(req.method==='POST'){
        const m=await body(req);if(![m.room,m.from,m.to].every(allowedName)||!m.payload||typeof m.payload!=='object'){json(res,400,{error:'接続メッセージが不正です'});return;}
        if(queues.size>=128){json(res,429,{error:'ルーム上限に達しました'});return;}
        const name=`${m.room}/${m.to}`,q=(queues.get(name)||[]).filter(x=>now-x.time<60000);if(q.length>=32){json(res,429,{error:'メッセージ上限に達しました'});return;}
        q.push({from:m.from,payload:m.payload,time:now});queues.set(name,q);json(res,200,{ok:true});return;
      }
      if(req.method==='GET'){
        const room=url.searchParams.get('room'),peer=url.searchParams.get('peer');if(![room,peer].every(allowedName)){json(res,400,{error:'ルーム名が不正です'});return;}
        const name=`${room}/${peer}`,messages=(queues.get(name)||[]).filter(x=>now-x.time<60000);queues.delete(name);json(res,200,{messages});return;
      }
      json(res,405,{error:'method not allowed'});return;
    }
    if(!['GET','HEAD'].includes(req.method)){json(res,405,{error:'method not allowed'});return;}
    const pathname=decodeURIComponent(url.pathname),target=resolve(root,'.'+(pathname==='/'?'/index.html':pathname));
    if(target!==root&&!target.startsWith(root+sep)){json(res,403,{error:'forbidden'});return;}
    const info=await stat(target);if(!info.isFile()){json(res,404,{error:'not found'});return;}
    const ext=target.slice(target.lastIndexOf('.')),data=await readFile(target);
    res.writeHead(200,{'Content-Type':mime[ext]||'application/octet-stream','Cache-Control':'no-cache','Content-Length':data.length});res.end(req.method==='HEAD'?undefined:data);
  }catch(error){if(!res.headersSent)json(res,error.status|| (error.code==='ENOENT'?404:500),{error:error.code==='ENOENT'?'not found':error.message});else res.end();}
}
const server=tls?https.createServer({cert:await readFile(cert),key:await readFile(key),minVersion:'TLSv1.2'},handler):http.createServer(handler);
server.requestTimeout=15000;server.headersTimeout=10000;
server.listen(port,host,()=>{
  const base=`${tls?'https':'http'}://${['0.0.0.0','::'].includes(host)?'localhost':host}:${port}`;
  console.log(`全身トラッキング: ${base}/`);
  console.log(`接続トークン: ${token}`);
  console.log('映像はブラウザ間を直接流れます。このサーバーは接続情報だけを一時保持します。');
  if(tls)console.log('スマホではlocalhostをPCのLANアドレスへ変更し、信頼済み証明書のホスト名に合わせてください。');
});
