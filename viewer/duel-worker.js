const PYODIDE='https://cdn.jsdelivr.net/pyodide/v0.28.3/full/';
let runtime;
const ready=(async()=>{
  self.postMessage({status:'加载浏览器运行环境…首次打开稍慢，之后会使用浏览器缓存。'});
  const {loadPyodide}=await import(`${PYODIDE}pyodide.mjs`);
  runtime=await loadPyodide({indexURL:PYODIDE});
  self.postMessage({status:'加载 NumPy 与冻结模型权重…'});
  await runtime.loadPackage('numpy');
  const response=await fetch('duel/manifest.json');
  if(!response.ok)throw new Error('无法读取模型清单');
  const manifest=await response.json();
  runtime.FS.mkdirTree('/duel');
  runtime.FS.writeFile('/duel/manifest.json',JSON.stringify(manifest));
  await Promise.all(Object.entries(manifest.files).map(async([name,expected])=>{
    if(!['runtime.zip','model.npz','catalog.json','parity.npz'].includes(name))throw new Error('未知模型文件');
    const response=await fetch(`duel/${name}`);
    if(!response.ok)throw new Error(`模型文件下载失败：${name}`);
    const data=new Uint8Array(await response.arrayBuffer());
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),b=>b.toString(16).padStart(2,'0')).join('');
    if(hash!==expected)throw new Error(`模型完整性检查失败：${name}。请重新加载页面。`);
    runtime.FS.writeFile(`/duel/${name}`,data);
  }));
  const metadata=JSON.parse(runtime.runPython(`
import sys, zipfile
with zipfile.ZipFile('/duel/runtime.zip') as archive:
    archive.extractall('/duel')
sys.path.insert(0, '/duel')
from sap_web.duel import initialize, dispatch
initialize('/duel')
`));
  self.postMessage({metadata});
})();
ready.catch(error=>self.postMessage({fatal:true,error:String(error)}));
let queue=ready;
self.onmessage=event=>{
  queue=queue.then(()=>{
    runtime.globals.set('request_json',JSON.stringify(event.data));
    const state=JSON.parse(runtime.runPython('dispatch(request_json)'));
    self.postMessage({state,id:event.data.id});
  }).catch(error=>self.postMessage({error:String(error),id:event.data.id}));
};
