import {PoseEngine} from './inference.mjs';
let engine=null;
let operation=Promise.resolve();
const reply=(id,value)=>postMessage({id,value});
const fail=(id,error)=>postMessage({id,error:{name:error.name||'Error',message:error.message||String(error)}});
self.onmessage=({data})=>{
  if(data.type==='abort'){engine?.abort.abort();return;}
  // Serialize reset/dispose with inference so a completed old frame cannot
  // restore the tracker state after a seek or a change of selected person.
  operation=operation.catch(()=>{}).then(async()=>{
    try{
      if(data.type==='load'){
        await engine?.dispose();
        engine=new PoseEngine({onProgress:value=>postMessage({type:'progress',value})});
        await engine.load(data.value);reply(data.id,{ready:true});
      }else if(data.type==='estimate'){
        if(!engine)throw new Error('先にモデルを読み込んでください。');
        try{reply(data.id,await engine.estimate(data.value.bitmap,data.value.options));}
        finally{data.value.bitmap.close();}
      }else if(data.type==='reset'){engine?.resetSource(data.value);}
      else if(data.type==='dispose'){await engine?.dispose();engine=null;reply(data.id,{disposed:true});}
    }catch(error){if(data.id)fail(data.id,error);}
  });
};
