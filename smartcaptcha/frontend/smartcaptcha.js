(function () {
  const root = document.getElementById('smartcaptcha-root');
  const statusEl = document.getElementById('smartcaptcha-status');
  const resetBtn = document.getElementById('smartcaptcha-reset');

  const PAGE_LOADED_AT_MS = nowMs();

  const FEATURE_COLUMNS = [
    'avg_mouse_speed',
    'mouse_path_entropy',
    'click_delay',
    'task_completion_time',
    'idle_time',
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

  function clampFinite(n, min, max) {
    if (!Number.isFinite(n)) return min;
    return clamp(n, min, max);
  }

  function buildVerifyPayload(features) {
    const payload = {};
    for (const key of FEATURE_COLUMNS) {
      const v = features && Object.prototype.hasOwnProperty.call(features, key) ? features[key] : 0;
      payload[key] = Number.isFinite(v) ? v : 0;
    }
    return payload;
  }

  function computeFeatures(session) {
    const events = Array.isArray(session?.events) ? session.events : [];
    const moveEvents = events.filter((e) => e && e.type === 'move' && Number.isFinite(e.t_ms));
    if (moveEvents.length < 3) {
      return null;
    }

    const click_delay = safeDivide(
      (session.interaction_started_at_ms ?? 0) - (session.page_loaded_at_ms ?? 0),
      1000
    );

    const task_completion_time = safeDivide(
      (session.interaction_completed_at_ms ?? session.interaction_ended_at_ms ?? 0) - (session.interaction_started_at_ms ?? 0),
      1000
    );

    let totalDistance = 0;
    let activeMoveTimeMs = 0;
    let idle_time_ms = 0;

    const idleMicroPauseIgnoreMs = 50;
    const movementNoisePx = 0.5;

    const angles = [];

    for (let i = 1; i < moveEvents.length; i++) {
      const a = moveEvents[i - 1];
      const b = moveEvents[i];
      const dtMs = b.t_ms - a.t_ms;
      if (!Number.isFinite(dtMs) || dtMs <= 0) continue;

      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.hypot(dx, dy);

      if (dist > movementNoisePx) {
        totalDistance += dist;
        activeMoveTimeMs += dtMs;
        angles.push(Math.atan2(dy, dx));
      } else {
        if (dtMs >= idleMicroPauseIgnoreMs) idle_time_ms += dtMs;
      }
    }

    if (activeMoveTimeMs <= 0 || totalDistance <= 0 || angles.length < 2) {
      return null;
    }

    const avg_mouse_speed = safeDivide(totalDistance, activeMoveTimeMs / 1000);
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

    const featureVector = {
      avg_mouse_speed: clampFinite(avg_mouse_speed, 0, 10000),
      mouse_path_entropy: clampFinite(mouse_path_entropy, 0, 1),
      click_delay: clampFinite(click_delay, 0.001, 60),
      task_completion_time: clampFinite(task_completion_time, 0.001, 60),
      idle_time: clampFinite(idle_time, 0, 60),
    };

    return featureVector;
  }

  async function verifyWithBackend(features) {
    const payload = buildVerifyPayload(features);
    console.debug('[SmartCAPTCHA] verify payload', payload);
    const res = await fetch("https://captcha-2-fix9.onrender.com/verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Verify failed (${res.status}): ${text}`);
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
      page_loaded_at_ms: PAGE_LOADED_AT_MS,
      widget_shown_at_ms: nowMs(),
      interaction_started_at_ms: null,
      interaction_ended_at_ms: null,
      interaction_completed_at_ms: null,
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
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return;
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

      const endTs = nowMs();
      session.interaction_ended_at_ms = endTs;
      if (evt) {
        recordEvent('up', evt);
      }

      state.dragging = false;
      handle.classList.remove('sc-handle--active');
      track.classList.remove('sc-track--active');

      if (isComplete()) {
        session.interaction_completed_at_ms = endTs;
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
            if (result && result.decision === 'human') {
              setStatus('Verified: Human');
              window.location.assign('https://example.com');
            } else {
              setStatus('You are not a human');
            }
          })
          .catch((err) => {
            setStatus(`Verification error: ${err.message}`);
          });
      } else {
        animateBack();
        const features = computeFeatures(session);
        window.__smartcaptcha_features = features;
        setStatus('Try again.');
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
      setStatus('');
      render();
    });
  }

  render();
})();
