import {spawn} from 'node:child_process';
import {mkdir,writeFile} from 'node:fs/promises';
import {createServer} from 'node:net';
const root='G:/CoT/nine_layer_membership_20261102';
const port=await new Promise(resolve=>{const s=createServer();s.listen(0,'127.0.0.1',()=>{const p=s.address().port;s.close(()=>resolve(p))})});
await mkdir(root+'/qa_profile',{recursive:true});
const proc=spawn('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',['--headless=new','--disable-gpu','--no-first-run','--remote-debugging-port='+port,'--user-data-dir='+root+'/qa_profile','about:blank'],{windowsHide:true,stdio:'ignore'});
let ws;
try {
 let tabs;
 for(let i=0;i<50;i++){try{tabs=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();if(tabs.length)break}catch{}await new Promise(r=>setTimeout(r,200))}
 ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
 let seq=0;const pending=new Map(),errors=[];
 ws.onmessage=e=>{const d=JSON.parse(e.data);if(d.id){const p=pending.get(d.id);pending.delete(d.id);d.error?p.reject(d.error):p.resolve(d.result)}else if(d.method==='Runtime.exceptionThrown')errors.push(d.params.exceptionDetails)};
 const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
 const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
 await call('Runtime.enable');await call('Page.enable');await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1050,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:'http://127.0.0.1:8765/nine_layer_membership_20261102/report.html'});
 for(let i=0;i<50;i++){if(await evaluate('document.readyState==="complete" && document.querySelectorAll("table").length>=6'))break;await new Promise(r=>setTimeout(r,100))}
 const result=await evaluate('(async()=>{for(const im of document.images)await im.decode();return {title:document.title,headings:document.querySelectorAll("h2").length,tables:document.querySelectorAll("table").length,images:document.images.length,has720:document.body.innerText.includes("720"),hasRoute:document.body.innerText.includes("路线对齐"),sourceText:document.body.innerText.startsWith("<!doctype")}})()');
 if(result.headings!==9||result.tables!==7||!result.has720||!result.hasRoute||result.sourceText||errors.length)throw Error(JSON.stringify({result,errors}));
 const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});await writeFile(root+'/report_preview.png',Buffer.from(shot.data,'base64'));
 await writeFile(root+'/browser_validation.json',JSON.stringify({passed:true,result,errors},null,2));console.log(JSON.stringify({passed:true,result}));
}finally{ws?.close();proc.kill()}
