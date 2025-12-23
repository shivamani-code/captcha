(function () {
  const root = document.getElementById('smartcaptcha-root');
  const statusEl = document.getElementById('smartcaptcha-status');
  const resetBtn = document.getElementById('smartcaptcha-reset');
  const validateBtn = document.getElementById('smartcaptcha-validate');
  const exportHumanBtn = document.getElementById('smartcaptcha-export-human');
  const exportBotBtn = document.getElementById('smartcaptcha-export-bot');

  const VERIFY_ENDPOINT = 'http://127.0.0.1:8000/verify';
  const VALIDATE_ENDPOINT = 'http://127.0.0.1:8000/validate';

  const FEATURE_COLUMNS = [
    'avg_mouse_speed',
    'mouse_path_entropy',
    'click_delay',
    'task_completion_time',
    'idle_time',
    'micro_jitter_variance',
    'acceleration_curve',
    'curvature_variance',
    'overshoot_correction_ratio',
    'timing_entropy',
  ];

  function setStatus(text) {
    if (!statusEl) return;
    statusEl.textContent = text;
  }

  function nowMs() {
    return (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
  }

  function clamp(n, min, max) {
    return Math.min(max, Math.max(min, n));
  }

  function getClientX(evt) {
    if (evt.touches && evt.touches.length) return evt.touches[0].clientX;
    if (evt.changedTouches && evt.changedTouches.length) return evt.changedTouches[0].clientX;
    return evt.clientX;
  }

  function safeDivide(n, d) {
    if (!Number.isFinite(n) || !Number.isFinite(d) || d === 0) return 0;
    return n / d;
  }

  function mean(values) {
    if (!values.length) return 0;
    let s = 0;
    for (const v of values) s += v;
    return s / values.length;
  }

  function variance(values) {
    if (values.length < 2) return 0;
    const m = mean(values);
    let s = 0;
    for (const v of values) {
      const d = v - m;
      s += d * d;
    }
    return s / (values.length - 1);
  }

  function shannonEntropy(probs) {
    let h = 0;
    for (const p of probs) {
      if (p <= 0) continue;
      h -= p * Math.log2(p);
    }
    return h;
  }

  function normalizeEntropyBits(hBits, binCount) {
    if (binCount <= 1) return 0;
    const max = Math.log2(binCount);
    return safeDivide(hBits, max);
  }

  function setExportEnabled(enabled) {
    if (exportHumanBtn) exportHumanBtn.disabled = !enabled;
    if (exportBotBtn) exportBotBtn.disabled = !enabled;
  }

  function toCsvRow(features, label) {
    const values = FEATURE_COLUMNS.map((k) => {
      const v = features && Object.prototype.hasOwnProperty.call(features, k) ? features[k] : 0;
      return Number.isFinite(v) ? String(v) : '0';
    });
    values.push(String(label));
    return values.join(',');
  }

  function downloadTextFile(filename, content) {
    const blob = new Blob([content], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  function exportCurrentFeatures(label) {
    const features = window.__smartcaptcha_features;
    if (!features) {
      setStatus('No features available to export. Complete the slider first.');
      setExportEnabled(false);
      return;
    }

    const header = `${FEATURE_COLUMNS.join(',')},label`;
    const row = toCsvRow(features, label);
    const csv = `${header}\n${row}\n`;
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `smartcaptcha_row_${label === 1 ? 'human' : 'bot'}_${ts}.csv`;
    downloadTextFile(filename, csv);
    setStatus('Exported one labeled CSV row file. Append it into dataset/behavior_data.csv.');
  }

  function computeFeatures(session) {
    const events = Array.isArray(session?.events) ? session.events : [];
    const moveEvents = events.filter((e) => e && e.type === 'move' && Number.isFinite(e.t_ms));
    if (moveEvents.length < 3) {
      return null;
    }

    const click_delay = safeDivide(
      (session.interaction_started_at_ms ?? 0) - (session.widget_shown_at_ms ?? 0),
      1000
    );
    const task_completion_time = safeDivide(
      (session.interaction_ended_at_ms ?? 0) - (session.interaction_started_at_ms ?? 0),
      1000
    );

    let totalDistance = 0;
    let totalTime = 0;

    const segmentSpeeds = [];
    const segmentDt = [];
    const dxList = [];
    const dyList = [];
    const angles = [];

    for (let i = 1; i < moveEvents.length; i++) {
      const a = moveEvents[i - 1];
      const b = moveEvents[i];
      const dtMs = b.t_ms - a.t_ms;
      if (!Number.isFinite(dtMs) || dtMs <= 0) continue;

      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.hypot(dx, dy);

      totalDistance += dist;
      totalTime += dtMs;

      const speed = dist / (dtMs / 1000);
      segmentSpeeds.push(speed);
      segmentDt.push(dtMs);
      dxList.push(dx);
      dyList.push(dy);

      angles.push(Math.atan2(dy, dx));
    }

    const avg_mouse_speed = safeDivide(totalDistance, totalTime / 1000);

    const idleGapThresholdMs = 120;
    let idle_time_ms = 0;
    for (let i = 1; i < moveEvents.length; i++) {
      const dtMs = moveEvents[i].t_ms - moveEvents[i - 1].t_ms;
      if (dtMs > idleGapThresholdMs) idle_time_ms += dtMs;
    }
    const idle_time = idle_time_ms / 1000;

    const directionBins = 12;
    const directionCounts = new Array(directionBins).fill(0);
    for (const a of angles) {
      const normalized = (a + Math.PI) / (2 * Math.PI);
      const idx = Math.min(directionBins - 1, Math.max(0, Math.floor(normalized * directionBins)));
      directionCounts[idx] += 1;
    }
    const directionTotal = directionCounts.reduce((s, c) => s + c, 0);
    const directionProbs = directionCounts.map((c) => safeDivide(c, directionTotal));
    const mouse_path_entropy = normalizeEntropyBits(shannonEntropy(directionProbs), directionBins);

    const micro_jitter_variance = variance(dxList) + variance(dyList);

    const accelerationsAbs = [];
    for (let i = 1; i < segmentSpeeds.length; i++) {
      const dv = segmentSpeeds[i] - segmentSpeeds[i - 1];
      const dtS = (segmentDt[i] ?? 0) / 1000;
      if (!Number.isFinite(dtS) || dtS <= 0) continue;
      accelerationsAbs.push(Math.abs(dv / dtS));
    }
    const acceleration_curve = mean(accelerationsAbs);

    const curvatures = [];
    for (let i = 1; i < angles.length; i++) {
      const da = angles[i] - angles[i - 1];
      const wrapped = Math.atan2(Math.sin(da), Math.cos(da));
      const segLen = Math.hypot(dxList[i] ?? 0, dyList[i] ?? 0);
      if (segLen <= 0) continue;
      curvatures.push(Math.abs(wrapped) / segLen);
    }
    const curvature_variance = variance(curvatures);

    let forward = 0;
    let backward = 0;
    for (const dx of dxList) {
      if (dx >= 0) forward += dx;
      else backward += Math.abs(dx);
    }
    const overshoot_correction_ratio = safeDivide(backward, forward);

    const timingBins = 10;
    const timingCounts = new Array(timingBins).fill(0);
    const dtValues = [];
    for (const dtMs of segmentDt) {
      if (Number.isFinite(dtMs) && dtMs > 0) dtValues.push(dtMs);
    }
    let timing_entropy = 0;
    if (dtValues.length) {
      const minDt = Math.min(...dtValues);
      const maxDt = Math.max(...dtValues);
      const range = maxDt - minDt;
      for (const dtMs of dtValues) {
        const normalized = range > 0 ? (dtMs - minDt) / range : 0;
        const idx = Math.min(timingBins - 1, Math.max(0, Math.floor(normalized * timingBins)));
        timingCounts[idx] += 1;
      }
      const timingTotal = timingCounts.reduce((s, c) => s + c, 0);
      const timingProbs = timingCounts.map((c) => safeDivide(c, timingTotal));
      timing_entropy = normalizeEntropyBits(shannonEntropy(timingProbs), timingBins);
    }

    const featureVector = {
      avg_mouse_speed,
      mouse_path_entropy,
      click_delay,
      task_completion_time,
      idle_time,
      micro_jitter_variance,
      acceleration_curve,
      curvature_variance,
      overshoot_correction_ratio,
      timing_entropy,
    };

    return featureVector;
  }

  async function verifyWithBackend(features) {
    const res = await fetch(VERIFY_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(features),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Verify failed (${res.status}): ${text}`);
    }

    return res.json();
  }

  async function validateTokenWithBackend(token) {
    const res = await fetch(VALIDATE_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Validate failed (${res.status}): ${text}`);
    }

    return res.json();
  }

  function render() {
    if (!root) return;
    root.innerHTML = '';

    const widget = document.createElement('section');
    widget.className = 'sc-widget';
    widget.setAttribute('role', 'group');
    widget.setAttribute('aria-label', 'SmartCAPTCHA slider');

    const label = document.createElement('div');
    label.className = 'sc-label';
    label.textContent = 'Slide to verify';

    const track = document.createElement('div');
    track.className = 'sc-track';
    track.setAttribute('role', 'presentation');

    const fill = document.createElement('div');
    fill.className = 'sc-fill';

    const handle = document.createElement('button');
    handle.type = 'button';
    handle.className = 'sc-handle';
    handle.setAttribute('aria-label', 'Drag slider');

    track.appendChild(fill);
    track.appendChild(handle);
    widget.appendChild(label);
    widget.appendChild(track);
    root.appendChild(widget);

    const session = {
      widget_shown_at_ms: nowMs(),
      interaction_started_at_ms: null,
      interaction_ended_at_ms: null,
      events: [],
    };

    window.__smartcaptcha_session = session;

    function toLocalPoint(evt) {
      const trackRect = track.getBoundingClientRect();
      const x = getClientX(evt) - trackRect.left;
      let y;
      if (evt.touches && evt.touches.length) y = evt.touches[0].clientY - trackRect.top;
      else if (evt.changedTouches && evt.changedTouches.length) y = evt.changedTouches[0].clientY - trackRect.top;
      else y = evt.clientY - trackRect.top;
      return {
        x,
        y,
      };
    }

    function recordEvent(evtType, evt) {
      const p = toLocalPoint(evt);
      session.events.push({
        type: evtType,
        t_ms: nowMs(),
        x: p.x,
        y: p.y,
      });
    }

    const state = {
      dragging: false,
      verified: false,
      startX: 0,
      currentX: 0,
      maxX: 0,
      rafId: 0,
      needsRender: false,
    };

    function measure() {
      const trackRect = track.getBoundingClientRect();
      const handleRect = handle.getBoundingClientRect();
      const maxX = Math.max(0, trackRect.width - handleRect.width - 6);
      state.maxX = maxX;
      state.currentX = clamp(state.currentX, 0, state.maxX);
      requestFrame();
    }

    function setPosition(x) {
      state.currentX = clamp(x, 0, state.maxX);
      requestFrame();
    }

    function requestFrame() {
      state.needsRender = true;
      if (state.rafId) return;
      state.rafId = window.requestAnimationFrame(renderFrame);
    }

    function renderFrame() {
      state.rafId = 0;
      if (!state.needsRender) return;
      state.needsRender = false;
      handle.style.transform = `translateX(${state.currentX}px)`;
      fill.style.width = `${state.currentX + 26}px`;
    }

    function isComplete() {
      if (state.maxX <= 0) return false;
      return state.currentX >= state.maxX * 0.98;
    }

    function animateBack() {
      handle.classList.add('sc-handle--animate');
      fill.classList.add('sc-fill--animate');
      setPosition(0);
      window.setTimeout(() => {
        handle.classList.remove('sc-handle--animate');
        fill.classList.remove('sc-fill--animate');
      }, 220);
    }

    function beginDrag(evt) {
      if (state.verified) return;

      if (session.interaction_started_at_ms === null) {
        session.interaction_started_at_ms = nowMs();
      }
      recordEvent('down', evt);

      state.dragging = true;
      handle.classList.add('sc-handle--active');
      track.classList.add('sc-track--active');

      const handleRect = handle.getBoundingClientRect();
      state.startX = getClientX(evt) - handleRect.left;
      setStatus('');
    }

    function moveDrag(evt) {
      if (!state.dragging || state.verified) return;
      recordEvent('move', evt);
      const trackRect = track.getBoundingClientRect();
      const x = getClientX(evt) - trackRect.left - state.startX;
      setPosition(x);
    }

    function endDrag(evt) {
      if (!state.dragging) return;

      session.interaction_ended_at_ms = nowMs();
      if (evt) {
        recordEvent('up', evt);
      }

      state.dragging = false;
      handle.classList.remove('sc-handle--active');
      track.classList.remove('sc-track--active');

      if (isComplete()) {
        state.verified = true;
        handle.disabled = true;
        widget.classList.add('sc-widget--verified');
        setPosition(state.maxX);
        const features = computeFeatures(session);
        window.__smartcaptcha_features = features;
        if (!features) {
          setStatus('Slider completed, but not enough movement data to compute features.');
          return;
        }

        setStatus('Verifying...');
        verifyWithBackend(features)
          .then((result) => {
            window.__smartcaptcha_verify_result = result;
            const confidencePct = Math.round((result.confidence ?? 0) * 100);
            if (result.status === 'human') {
              setStatus(`Verified: HUMAN (${confidencePct}%). Token issued.`);
              if (validateBtn) validateBtn.disabled = !result.token;
            } else if (result.status === 'suspicious') {
              setStatus(`Result: SUSPICIOUS (${confidencePct}%).`);
              if (validateBtn) validateBtn.disabled = true;
            } else {
              setStatus(`Result: BOT (${confidencePct}%).`);
              if (validateBtn) validateBtn.disabled = true;
            }

            setExportEnabled(true);
          })
          .catch((err) => {
            setStatus(`Verification error: ${err.message}`);
            if (validateBtn) validateBtn.disabled = true;
            setExportEnabled(true);
          });
      } else {
        animateBack();
        const features = computeFeatures(session);
        window.__smartcaptcha_features = features;
        setStatus(`Try again. Captured ${session.events.length} events.`);

        setExportEnabled(!!features);
      }
    }

    function onMouseDown(evt) {
      evt.preventDefault();
      beginDrag(evt);
      window.addEventListener('mousemove', onMouseMove, { passive: false });
      window.addEventListener('mouseup', onMouseUp, { passive: true });
    }

    function onMouseMove(evt) {
      evt.preventDefault();
      moveDrag(evt);
    }

    function onMouseUp(evt) {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      endDrag(evt);
    }

    function onTouchStart(evt) {
      beginDrag(evt);
      window.addEventListener('touchmove', onTouchMove, { passive: false });
      window.addEventListener('touchend', onTouchEnd, { passive: true });
      window.addEventListener('touchcancel', onTouchEnd, { passive: true });
    }

    function onTouchMove(evt) {
      evt.preventDefault();
      moveDrag(evt);
    }

    function onTouchEnd(evt) {
      window.removeEventListener('touchmove', onTouchMove);
      window.removeEventListener('touchend', onTouchEnd);
      window.removeEventListener('touchcancel', onTouchEnd);
      endDrag(evt);
    }

    handle.addEventListener('mousedown', onMouseDown);
    handle.addEventListener('touchstart', onTouchStart, { passive: true });

    window.addEventListener('resize', measure);
    measure();
    setStatus('');
  }

  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      window.__smartcaptcha_session = undefined;
      window.__smartcaptcha_features = undefined;
      window.__smartcaptcha_verify_result = undefined;
      setStatus('');
      if (validateBtn) validateBtn.disabled = true;
      setExportEnabled(false);
      render();
    });
  }

  if (exportHumanBtn) {
    exportHumanBtn.addEventListener('click', () => exportCurrentFeatures(1));
  }

  if (exportBotBtn) {
    exportBotBtn.addEventListener('click', () => exportCurrentFeatures(0));
  }

  if (validateBtn) {
    validateBtn.addEventListener('click', () => {
      const result = window.__smartcaptcha_verify_result;
      const token = result && result.token;
      if (!token) {
        setStatus('No token available to validate. Complete verification first.');
        validateBtn.disabled = true;
        return;
      }

      setStatus('Validating token...');
      validateTokenWithBackend(token)
        .then((resp) => {
          if (resp && resp.valid === true) {
            setStatus('Token validation: VALID (consumed).');
          } else {
            setStatus('Token validation: INVALID (expired or already used).');
          }
          validateBtn.disabled = true;
        })
        .catch((err) => {
          setStatus(`Token validation error: ${err.message}`);
          validateBtn.disabled = true;
        });
    });
  }

  render();
  setExportEnabled(false);
})();
