// A private URL must be configured as an Actions secret, never a workflow input,
// source file, artifact or console message. No media/trace leaves RUNNER_TEMP.
import {mkdtemp,writeFile,rm,readFile} from 'node:fs/promises';
import {createWriteStream} from 'node:fs';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {Readable} from 'node:stream';
import {pipeline} from 'node:stream/promises';
import {spawn} from 'node:child_process';
if(process.env.GITHUB_ACTIONS!=='true')throw new Error('Private evaluation requires GitHub Actions.');
const secret=process.env.PRIVATE_VIDEO_URL;
if(!secret){
  await writeFile('artifacts/private-video-status.json',JSON.stringify({status:'not-run',reason:'PRIVATE_VIDEO_URL is not configured',privateVideoEvaluated:false})+'\n');
  console.log('Private video evaluation: not run; private fetch secret is not configured.');
}else{
  const directory=await mkdtemp(join(process.env.RUNNER_TEMP||tmpdir(),'vrc-private-'));
  try{
    const source=new URL(secret);if(source.protocol!=='https:')throw new Error('secure-source-required');
    const response=await fetch(source,{signal:AbortSignal.timeout(120000)});
    if(!response.ok||!response.body)throw new Error('private-fetch-failed');
    const path=join(directory,'input.mp4');
    await pipeline(Readable.fromWeb(response.body),createWriteStream(path,{mode:0o600}));
    const output=join(directory,'aggregate.json');
    const env={...process.env};delete env.PRIVATE_VIDEO_URL;
    await new Promise((resolve,reject)=>{const child=spawn(process.execPath,['ci/headless-pose3d.mjs','--video',path,'--max-seconds','180','--output',output],{env,stdio:['ignore','ignore','ignore']});child.once('error',()=>reject(new Error('private-evaluation-failed')));child.once('exit',code=>code===0?resolve():reject(new Error('private-evaluation-failed')));});
    const report=JSON.parse(await readFile(output,'utf8'));
    await writeFile('artifacts/private-video-status.json',JSON.stringify({status:'completed',privateVideoEvaluated:true,trackingReadiness:report.deploymentReadiness?.passed===true,report},null,2)+'\n');
    console.log('Private video inference completed. See aggregate tracking coverage and loss metrics; execution alone does not establish accuracy.');
  }catch{throw new Error('Private video evaluation failed; source URL and media are suppressed.');}
  finally{await rm(directory,{recursive:true,force:true});}
}
