// The Actions page has no UI, camera, file picker, or external resource.
window.__webgpuResult = new Promise(resolve => {
  const worker = new Worker('/worker.mjs', { type: 'module', name: 'wholebody-public-webgpu-fixture' });
  worker.addEventListener('message', event => {
    if (event.data?.kind === 'stage') console.log('WEBGPU_WORKER_STAGE ' + event.data.stage);
    if (event.data?.kind === 'result') { resolve(event.data.report); worker.terminate(); }
  });
  worker.addEventListener('error', event => { resolve({ passed: false, status: 'worker-error', error: { message: event.message } }); worker.terminate(); });
  worker.postMessage({ command: 'run-public-fixture' });
});
