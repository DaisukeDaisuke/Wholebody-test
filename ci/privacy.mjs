import {readdir,readFile,stat} from 'node:fs/promises';
const forbidden=/\.(mp4|mov|avi|webm|mkv|jpg|jpeg|png|pt|pth|npy|npz)$/i;
const seen=[];
async function visit(dir){for(const item of await readdir(dir)){if(item==='node_modules')continue;const path=dir+'/'+item,s=await stat(path);if(s.isDirectory()){if(/(?:^|\/)(private|upload|cache|frames)(?:\/|$)/i.test(path))throw new Error('Private/cache directory in package: '+path);await visit(path);}else{if(forbidden.test(path))throw new Error('Media/training file in publication: '+path);seen.push(path);}}}
await visit('dist');await visit('server');
for(const path of seen.filter(p=>p.endsWith('.mjs')&&!p.includes('/vendor/'))){const text=await readFile(path,'utf8');if(/sendBeacon\s*\(|FormData\s*\(/.test(text))throw new Error('Unexpected upload facility: '+path);}
console.log(JSON.stringify({publicationFiles:seen.length,videoOrImageFiles:0,privateDirectoryFiles:0,status:'passed'}));
