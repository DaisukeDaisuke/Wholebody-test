/** Paused-video, fixed-media-time sampling with explicit inference backpressure.
 * onFrame receives one owned canvas. Only advance(frameToken) requests another.
 * This samples a 30 Hz media-time grid by default; it does not claim native
 * frame-accurate decoding, real-time playback, or a faster inference backend.
 */
let nextStepperId = 0;

export class VideoStepper {
  constructor({ video, stepSeconds = 1 / 30, seekTimeoutMs = 15000,
    onFrame, onStop = () => {}, onError = () => {} } = {}) {
    if (!video || typeof video.pause !== 'function' || typeof video.addEventListener !== 'function') throw new Error('順送りする動画が見つかりません。');
    if (typeof onFrame !== 'function') throw new Error('順送りフレームの受け取り先が必要です。');
    if (!Number.isFinite(stepSeconds) || stepSeconds <= 0 || stepSeconds > 1) throw new Error('順送りの時間間隔が不正です。');
    if (!Number.isFinite(seekTimeoutMs) || seekTimeoutMs < 1000) throw new Error('動画の待ち時間が不正です。');
    this.video = video; this.stepSeconds = stepSeconds; this.seekTimeoutMs = seekTimeoutMs;
    this.onFrame = onFrame; this.onStop = onStop; this.onError = onError;
    this._id = ++nextStepperId; this._runId = 0; this._active = false; this._disposed = false;
    this._pending = null; this._waitingToken = null; this._timeout = null; this._captureTimer = null; this._retiredSeek = null; this._retiredTimer = null;
    this._source = null; this._origin = 0; this._frameIndex = 0; this._deliveredFrames = 0; this._lastMediaTime = null;
    this._timeTolerance = Math.min(0.005, stepSeconds / 4);
    this._listeners = {
      play: () => { if (this.active && !video.paused) this.stop('play'); },
      seeking: () => { if (this.active && !this.ownsSeek(video.currentTime)) this.stop('user-seek'); },
      seeked: () => {
        const retired = this._retiredSeek;
        if (retired) {
          clearTimeout(this._retiredTimer);
          this._retiredTimer = setTimeout(() => { if (this._retiredSeek === retired) this._retiredSeek = null; }, 0);
        }
        const pending = this._pending;
        if (!this.active || !pending || pending.kind !== 'seek' || video.seeking) return;
        if (!this.ownsSeek(video.currentTime)) { this.stop('user-seek'); return; }
        pending.ready = true; this._scheduleCapture(pending);
      },
      loadeddata: () => this._resumeReadyCapture(),
      canplay: () => this._resumeReadyCapture(),
      ended: () => { if (this.active) this.stop('ended'); },
      emptied: () => { if (this.active) this.stop('source-closed'); },
      abort: () => { if (this.active) this.stop('source-changed'); },
      error: () => this._fail(new Error('動画のフレームを読み込めません。'), this._runId)
    };
    // Capture listeners see an internal seek before ordinary source handlers.
    // Ownership remains valid throughout seeked dispatch (capture is a task,
    // not a microtask), irrespective of the app's listener registration order.
    for (const [event, listener] of Object.entries(this._listeners)) video.addEventListener(event, listener, true);
  }

  get active() { return this._active && !this._disposed; }
  get runId() { return this._runId; }

  /** Start from the currently displayed time; the first capture keeps manual
   * seed coordinates on the paused frame on which they were selected. */
  start() {
    if (this._disposed) throw new Error('この動画の順送りは閉じられています。');
    if (this.active) return false;
    const video = this.video;
    if (video.srcObject || !Number.isFinite(video.duration) || video.duration <= 0) throw new Error('順送りには再生時間が確定した動画ファイルを選んでください。');
    if (video.seeking) throw new Error('動画の再生位置の変更が終わってから順送りを開始してください。');
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight || !Number.isFinite(video.currentTime)) throw new Error('動画のフレームが表示されてから順送りを開始してください。');
    this._clearTimers(); this._runId++; this._active = true;
    this._source = this._sourceIdentity(); this._frameIndex = 0;
    this._waitingToken = null; this._deliveredFrames = 0; this._lastMediaTime = null;
    video.pause();
    this._origin = video.currentTime;
    if (this._origin >= video.duration - 1e-7) { this.stop('ended'); return false; }
    const pending = { runId: this._runId, frameIndex: 0, targetTime: this._origin, kind: 'current', ready: true };
    this._pending = pending; this._armTimeout(pending); this._scheduleCapture(pending);
    return true;
  }

  /** Acknowledge exactly the delivered frame, then request its next time step.
   * A token from an old run, a duplicate ack, or an in-flight seek is ignored. */
  advance(frameToken) {
    if (!this.active || !this._waitingToken || frameToken !== this._waitingToken || this._pending) return false;
    if (!this._sameSource()) { this.stop('source-changed'); return false; }
    if (!this.video.paused) { this.stop('play'); return false; }
    if (this.video.seeking || Math.abs(this.video.currentTime - this._lastMediaTime) > this._timeTolerance) { this.stop('user-seek'); return false; }
    const nextIndex = this._frameIndex + 1, targetTime = this._origin + nextIndex * this.stepSeconds;
    if (!Number.isFinite(this.video.duration) || targetTime >= this.video.duration - 1e-7) { this.stop('ended'); return false; }
    this._waitingToken = null; this._frameIndex = nextIndex;
    const pending = { runId: this._runId, frameIndex: nextIndex, targetTime, kind: 'seek', ready: false };
    this._pending = pending; this._armTimeout(pending);
    try { this.video.currentTime = targetTime; }
    catch (error) { this._fail(error, pending.runId); return false; }
    return true;
  }

  /** Call from the app's normal seeking/seeked reset handler. A cancelled
   * internal seek keeps event ownership until its late seeked is dispatched,
   * but can never capture or advance another frame after stop(). */
  ownsSeek(currentTime = this.video.currentTime) {
    if (this._disposed || !Number.isFinite(currentTime)) return false;
    const pending = this._pending, retired = this._retiredSeek;
    return Boolean((this.active && pending?.kind === 'seek' && pending.runId === this._runId && this._sameSource()
      && Math.abs(currentTime - pending.targetTime) <= this._timeTolerance)
      || (retired && performance.now() < retired.expiresAt && this._matchesSource(retired.source)
        && Math.abs(currentTime - retired.targetTime) <= this._timeTolerance));
  }

  stop(reason = 'stopped') {
    const wasActive = this.active, runId = this._runId;
    const detail = { reason, runId, deliveredFrames: this._deliveredFrames, mediaTime: this._lastMediaTime, stepSeconds: this.stepSeconds };
    if (this._pending?.kind === 'seek' && !this._pending.ready) this._retiredSeek = { targetTime: this._pending.targetTime, source: this._source, expiresAt: performance.now() + this.seekTimeoutMs };
    this._active = false; this._runId++; this._pending = null; this._waitingToken = null; this._clearTimers();
    // Leave the video at its last position. A normal user play event continues
    // normally; stopping the sampler must not pause or restart that playback.
    if (wasActive && !this._disposed) this.onStop(detail);
    return wasActive;
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true; this.stop('disposed');
    clearTimeout(this._retiredTimer); this._retiredSeek = null;
    for (const [event, listener] of Object.entries(this._listeners)) this.video.removeEventListener(event, listener, true);
    this.onFrame = () => {}; this.onStop = () => {}; this.onError = () => {};
  }

  _sourceIdentity() { return { src: this.video.getAttribute('src'), currentSrc: this.video.currentSrc, srcObject: this.video.srcObject }; }
  _sameSource() { return this._matchesSource(this._source); }
  _matchesSource(source) {
    const value = this._sourceIdentity();
    return source && value.src === source.src && value.currentSrc === source.currentSrc && value.srcObject === source.srcObject;
  }
  _isCurrent(pending) { return this.active && pending === this._pending && pending.runId === this._runId; }
  _clearTimers() { clearTimeout(this._timeout); clearTimeout(this._captureTimer); this._timeout = null; this._captureTimer = null; }
  _armTimeout(pending) {
    clearTimeout(this._timeout);
    this._timeout = setTimeout(() => {
      if (this._isCurrent(pending)) this._fail(new Error('動画の次のフレームを待つ時間を超えました。再生位置と動画の読込状態を確認してください。'), pending.runId);
    }, this.seekTimeoutMs);
  }
  _resumeReadyCapture() { if (this._pending?.ready) this._scheduleCapture(this._pending); }
  _scheduleCapture(pending) {
    if (!this._isCurrent(pending) || !pending.ready || this._captureTimer !== null) return;
    this._captureTimer = setTimeout(() => {
      if (!this._isCurrent(pending)) return;
      this._captureTimer = null;
      const video = this.video;
      if (!this._sameSource()) { this.stop('source-changed'); return; }
      if (!video.paused) { this.stop('play'); return; }
      if (video.seeking) return;
      if (Math.abs(video.currentTime - pending.targetTime) > this._timeTolerance) { this.stop('user-seek'); return; }
      if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) return;
      try {
        const canvas = video.ownerDocument.createElement('canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        const context = canvas.getContext('2d', { alpha: false });
        if (!context) throw new Error('動画フレームの取り込みを初期化できません。');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        if (!this._isCurrent(pending)) return;
        const capturedAtMs = performance.now(), frameToken = `video-stepper-${this._id}:${pending.runId}:${pending.frameIndex}`;
        const frame = { canvas, mediaTime: video.currentTime, requestedTime: pending.targetTime, capturedAtMs,
          frameIndex: pending.frameIndex, frameToken, runId: pending.runId, stepSeconds: this.stepSeconds,
          sampling: 'fixed-media-time', nativeFrameAccurate: false,
          timingBasis: pending.kind === 'seek' ? 'paused-currentTime-after-seeked' : 'paused-current-displayed-frame' };
        this._pending = null; this._waitingToken = frameToken; this._lastMediaTime = frame.mediaTime; this._deliveredFrames++;
        clearTimeout(this._timeout); this._timeout = null;
        // The callback may enqueue async inference. It does not advance this
        // clock: only the matching completion token can release backpressure.
        const result = this.onFrame(frame);
        if (result && typeof result.then === 'function') Promise.resolve(result).catch(error => this._fail(error, pending.runId));
      } catch (error) { this._fail(error, pending.runId); }
    }, 0);
  }
  _fail(error, runId) {
    if (!this.active || runId !== this._runId) return;
    const nextRunId = this._runId + 1;
    this.stop('error');
    if (!this._disposed && this._runId === nextRunId) this.onError(error instanceof Error ? error : new Error(String(error)));
  }
}
