import {spawn} from 'node:child_process';
import {writeFile,mkdir} from 'node:fs/promises';
import {createServer} from 'node:net';
import {pathToFileURL} from 'node:url';
const root='G:/CoT/attention_success_20261030';
const port=await new Promise(resolve=>{const s=createServer();s.listen(0,'127.0.0.1',()=>{const p=s.address().port;s.close(()=>resolve(p))})});
const profile=root+'/browser_qa_profile';
await mkdir(profile,{recursive:true});
const proc=spawn('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',[
  '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
  '--remote-debugging-port='+port,'--user-data-dir='+profile,'about:blank'
],{windowsHide:true,stdio:'ignore'});
let ws;
try{
 let tabs;
 for(let i=0;i<60;i++){try{tabs=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();if(tabs.length)break;}catch{}await new Promise(r=>setTimeout(r,200))}
 if(!tabs)throw Error('Headless Edge unavailable');
 ws=new WebSocket(tabs.find(t=>t.type==='page').webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});
 let seq=0;const pending=new Map(),errors=[];
 ws.onmessage=e=>{const d=JSON.parse(e.data);if(d.id){const p=pending.get(d.id);pending.delete(d.id);if(d.error)p.reject(d.error);else p.resolve(d.result)}else if(d.method==='Runtime.exceptionThrown')errors.push(d.params.exceptionDetails)};
 const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});ws.send(JSON.stringify({id,method,params}))});
 const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 await call('Runtime.enable');await call('Page.enable');
 await call('Emulation.setDeviceMetricsOverride',{width:1500,height:1160,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url:pathToFileURL(root+'/search_animation.html').href});
 for(let i=0;i<40;i++){if(await evaluate('Boolean(window.animationTest)'))break;await new Promise(r=>setTimeout(r,100))}
 const results=[];
 results.push(await evaluate(`(()=>{const t=animationTest;let count=0;for(const mode of ['reference']){document.getElementById('mode').value=mode;document.getElementById('mode').dispatchEvent(new Event('change'));for(let f=0;f<t.data[mode].frames.length;f++){t.setFrame(f);for(let l=0;l<6;l++){document.getElementById('layer').value=l;document.getElementById('layer').dispatchEvent(new Event('change'));if(document.querySelectorAll('#attention > rect').length!==9)throw Error('Missing key cards');if(document.querySelectorAll('#masses .layer-row').length!==6)throw Error('Missing layer rows');if(document.querySelectorAll('#edge-encoding rect').length!==576)throw Error('Edge matrix missing cells');if(document.querySelectorAll('#edge-reads td').length!==96)throw Error('Edge read weights missing');const groups=t.data[mode].frames[f].masses[l],edge=t.data[mode].frames[f].edge_weights[l].flat().reduce((a,b)=>a+b,0);if(Math.abs(groups[0]-edge)>1e-6)throw Error('Graph mass accounting mismatch');count++;}}}return {allFrameLayerViews:count,edgeMatrixCells:576,edgeReadCells:96};})()`));
 results.push(await evaluate(`(()=>{document.getElementById('mode').value='reference';document.getElementById('mode').dispatchEvent(new Event('change'));animationTest.setFrame(0);animationTest.advance(1);if(animationTest.getFrame()===0)throw Error('Step does not advance');document.getElementById('detail').value='token';animationTest.setFrame(0);animationTest.advance(1);if(animationTest.getFrame()!==1)throw Error('Token step incorrect');document.getElementById('scrub').value=20;document.getElementById('scrub').dispatchEvent(new Event('input'));if(animationTest.getFrame()!==20)throw Error('Slider broken');document.querySelector('#records button:nth-child(4)').click();if(animationTest.data.reference.frames[animationTest.getFrame()].record!==3)throw Error('Record navigation broken');document.getElementById('speed').value='500';animationTest.setFrame(0);document.getElementById('play').click();return {stepSliderRecordNavigation:true};})()`));
 await new Promise(r=>setTimeout(r,650));
 results.push(await evaluate(`(()=>{const frame=animationTest.getFrame();animationTest.stop();if(frame<1)throw Error('Autoplay did not advance');return {autoplayAdvancedTo:frame};})()`));
 results.push(await evaluate(`(()=>{document.querySelector('#edge-encoding rect[data-i="10"][data-j="8"]').dispatchEvent(new Event('click'));if(!document.getElementById('edge-pair-title').textContent.includes('11 A→C'))throw Error('Edge drilldown failed');if(document.querySelectorAll('#edge-pair td').length!==9)throw Error('Missing 3x3 details');document.getElementById('graph-view').value='full';document.getElementById('graph-view').dispatchEvent(new Event('change'));if(document.querySelectorAll('#graph circle').length!==16)throw Error('Full DAG missing nodes');document.getElementById('graph-view').value='active';document.getElementById('graph-view').dispatchEvent(new Event('change'));if(document.querySelectorAll('#graph circle').length!==3)throw Error('Active graph wrong');return {edgeDrilldown:true,fullAndActiveGraph:true};})()`));
 await evaluate(`document.getElementById('layer').value=3;animationTest.setFrame(animationTest.data.reference.frames.findIndex(f=>f.record===2&&f.phase==='argument'));`);
 await call('Runtime.evaluate',{expression:'document.fonts.ready',awaitPromise:true});
 await new Promise(r=>setTimeout(r,500));
 const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});
 await writeFile(root+'/interactive_preview.png',Buffer.from(shot.data,'base64'));
 await evaluate(`document.getElementById('edge-encoding').scrollIntoView();`);
 await new Promise(r=>setTimeout(r,300));
 const edgeShot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
 await writeFile(root+'/edge_browser_preview.png',Buffer.from(edgeShot.data,'base64'));
 await call('Emulation.setDeviceMetricsOverride',{width:430,height:932,deviceScaleFactor:1,mobile:true});
 results.push(await evaluate(`(()=>{animationTest.setFrame(0);return {mobileBodyWidth:document.body.scrollWidth,viewportWidth:innerWidth};})()`));
 if(errors.length)throw Error(JSON.stringify(errors));
 await writeFile(root+'/animation_verification.json',JSON.stringify({passed:true,results,browserErrors:errors},null,2));
 console.log(JSON.stringify({passed:true,results}));
}finally{if(ws)ws.close();proc.kill();}
