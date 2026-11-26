/* Piano Typer front end — plain JS, no dependencies. */
'use strict';
(function () {

  // ------------------------------------------------------------------ basics
  const $ = (sel) => document.querySelector(sel);
  const LO = 36, HI = 96;
  const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const isBlack = (m) => [1, 3, 6, 8, 10].indexOf(((m % 12) + 12) % 12) !== -1;
  const noteName = (m) => NOTE_NAMES[m % 12] + (Math.floor(m / 12) - 1);
  const pcName = (m) => NOTE_NAMES[((m % 12) + 12) % 12];
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const now = () => performance.now();
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  const COLORS = { letter: '#5ee7ff', chord: '#b57bff', control: '#ffb454', char: '#ff7aa8', none: '#9aa4b8' };
  const CONTROL_LABEL = {
    space: '␣', backspace: '⌫', enter: '↵', delete_word: '⌫w', shift: '⇧',
    suggest1: 'S1', suggest2: 'S2', suggest3: 'S3', complete: '✓', double: '×2', undo: '↶'
  };
  const CONTROL_TITLE = {
    space: 'space', backspace: 'backspace (rejects a waiting word)', enter: 'enter', delete_word: 'delete word', shift: 'shift (next letter capital)',
    suggest1: 'option / suggestion 1', suggest2: 'option / suggestion 2', suggest3: 'option / suggestion 3',
    complete: 'complete: accept the waiting word, finish a half word, or space',
    double: 'double the last letter', undo: 'undo'
  };

  // Default layout (controls fixed by the spec) so the keyboard renders before init arrives.
  function defaultLayout() {
    const keys = {};
    const ctrl = { 36: 'enter', 48: 'complete', 50: 'double', 52: 'undo', 53: 'shift', 54: 'space', 55: 'delete_word', 49: 'suggest1', 51: 'suggest2', 46: 'suggest3', 56: 'backspace' };
    const chars = { 38: '.', 40: ',', 41: '?', 43: '!', 45: "'", 47: '-' };
    Object.keys(ctrl).forEach((m) => { keys[m] = { kind: 'control', value: ctrl[m] }; });
    Object.keys(chars).forEach((m) => { keys[m] = { kind: 'char', value: chars[m] }; });
    [83, 84, 86, 88, 89, 91, 93, 95, 96].forEach((m, i) => { keys[m] = { kind: 'char', value: String(i + 1) }; });
    keys[94] = { kind: 'char', value: '0' };
    return { version: 1, lowest_midi: LO, highest_midi: HI, keys: keys };
  }

  // ------------------------------------------------------------------ state
  const S = {
    layout: null,
    keys: {},            // midi -> {kind, value}
    letterToMidi: {},
    charToMidi: {},
    controlToMidi: {},   // first midi per control value
    settings: { input_mode: 'piano' },
    devices: [],
    state: null,
    calib: { phase: 'idle', target: null, done: 0, total: 0, message: '' },
    prevText: null,
    lastLevelAt: 0,
    active: new Set(),
    hoverIdx: -1,
  };

  // ------------------------------------------------------------------ DOM refs
  const el = {
    kb: $('#keyboard'), zones: $('#zones'), trail: $('#trail'),
    textOut: $('#textOut'), textScroll: $('#textScroll'), lastLine: $('#lastLine'), shiftBadge: $('#shiftBadge'),
    suggestRow: $('#suggestRow'),
    aiPill: $('#aiPill'), aiText: $('#aiText'), meterFill: $('#meterFill'), meterPeak: $('#meterPeak'),
    conn: $('#conn'), connText: $('#connText'),
    modeSeg: $('#modeSeg'), extInput: $('#extInput'),
    drawer: $('#drawer'), scrim: $('#scrim'),
    devSel: $('#devSel'), audioState: $('#audioState'), sensRange: $('#sensRange'), sensOut: $('#sensOut'),
    cwRange: $('#cwRange'), cwOut: $('#cwOut'), shiftNum: $('#shiftNum'), aiChk: $('#aiChk'), suggChk: $('#suggChk'), modelSel: $('#modelSel'),
    calib: $('#calib'), calibTitle: $('#calibTitle'), calibMsg: $('#calibMsg'), calibFill: $('#calibFill'),
    calibCount: $('#calibCount'), calibSkip: $('#calibSkip'),
    help: $('#help'), countdown: $('#countdown'), countdownNum: $('#countdownNum'), toasts: $('#toasts'),
  };

  // ------------------------------------------------------------------ layout
  function applyLayout(layout) {
    S.layout = layout && layout.keys ? layout : defaultLayout();
    S.keys = {}; S.letterToMidi = {}; S.charToMidi = {}; S.controlToMidi = {};
    Object.keys(S.layout.keys).forEach((k) => {
      const m = parseInt(k, 10);
      const v = S.layout.keys[k];
      if (!v || isNaN(m)) return;
      S.keys[m] = v;
      if (v.kind === 'letter') S.letterToMidi[String(v.value).toLowerCase()] = m;
      else if (v.kind === 'char') { if (S.charToMidi[v.value] == null) S.charToMidi[v.value] = m; }
      else if (v.kind === 'control') { if (S.controlToMidi[v.value] == null || m < S.controlToMidi[v.value]) S.controlToMidi[v.value] = m; }
    });
    buildKeyboard();
  }

  function keyKind(m) { const k = S.keys[m]; return k ? k.kind : 'none'; }
  function keyLabel(k) {
    if (!k) return '';
    if (k.kind === 'letter') return String(k.value).toUpperCase();
    if (k.kind === 'char') return k.value;
    return CONTROL_LABEL[k.value] || k.value;
  }
  function keyDescr(m) {
    const k = S.keys[m];
    if (!k) return noteName(m);
    if (k.kind === 'control') return noteName(m) + ' · ' + (CONTROL_TITLE[k.value] || k.value);
    return noteName(m) + ' · ' + (k.kind === 'letter' ? 'letter ' : '') + '“' + k.value + '”';
  }

  // ------------------------------------------------------------------ keyboard
  const keyEls = {};
  const geom = {};   // midi -> {left, width, center} in fractions (0..1)
  const BLACK_OFF = { 1: -0.10, 3: 0.10, 6: -0.12, 8: 0, 10: 0.12 };

  function buildKeyboard() {
    el.kb.innerHTML = '';
    let whites = 0;
    for (let m = LO; m <= HI; m++) if (!isBlack(m)) whites++;
    const ww = 1 / whites;
    let wi = 0;
    const blackFrag = document.createDocumentFragment();
    const whiteFrag = document.createDocumentFragment();
    for (let m = LO; m <= HI; m++) {
      const black = isBlack(m);
      let left, width;
      if (!black) { left = wi * ww; width = ww; wi++; }
      else { width = ww * 0.6; const c = wi * ww + BLACK_OFF[m % 12] * ww; left = c - width / 2; }
      geom[m] = { left: left, width: width, center: left + width / 2 };
      const k = S.keys[m];
      const kind = k ? k.kind : 'none';
      const d = document.createElement('div');
      d.className = 'key ' + (black ? 'black' : 'white') + ' ' + (kind === 'control' ? 'ctrl' : kind) +
        (kind === 'control' ? ' c-' + k.value : '');
      d.style.left = (left * 100) + '%';
      d.style.width = (width * 100) + '%';
      d.dataset.midi = m;
      d.title = keyDescr(m);
      d.innerHTML = '<span class="glow"></span><span class="lbl">' + esc(keyLabel(k)) + '</span>' +
        (m % 12 === 0 ? '<span class="note">' + noteName(m) + '</span>' : '');
      keyEls[m] = d;
      (black ? blackFrag : whiteFrag).appendChild(d);
    }
    el.kb.appendChild(whiteFrag);
    el.kb.appendChild(blackFrag);
    buildZones();
    if (S.state) renderState(S.state, true);
    renderCalib();
  }

  function buildZones() {
    el.zones.innerHTML = '';
    const zones = (S.layout && S.layout.zones) || [
      { a: 36, b: 56, cls: 'z-ctrl', name: 'Controls' },
      { a: 57, b: 82, cls: 'z-letter', name: 'Letters' },
      { a: 83, b: 96, cls: 'z-digit', name: 'Digits' },
    ];
    zones.forEach((z) => {
      let any = false;
      for (let m = z.a; m <= z.b; m++) if (S.keys[m]) { any = true; break; }
      if (!any) return;
      const left = geom[z.a].left;
      const right = z.b + 1 <= HI ? geom[z.b + 1].left : 1;
      const d = document.createElement('div');
      d.className = 'zone ' + z.cls;
      d.style.left = 'calc(' + (left * 100) + '% + 6px)';
      d.style.width = 'calc(' + ((right - left) * 100) + '% - 12px)';
      d.innerHTML = '<span>' + z.name + '</span>';
      el.zones.appendChild(d);
    });
  }

  const litTimers = {};
  function flashKey(m, color, ms) {
    const d = keyEls[m]; if (!d) return;
    d.style.setProperty('--hc', color);
    d.classList.remove('hit'); void d.offsetWidth; d.classList.add('hit');
    d.classList.add('lit');
    clearTimeout(litTimers[m]);
    litTimers[m] = setTimeout(() => { d.classList.remove('lit'); d.classList.remove('hit'); }, ms || 280);
  }
  function setHeld(m, on, color) {
    const d = keyEls[m]; if (!d) return;
    if (color) d.style.setProperty('--hc', color);
    d.classList.toggle('lit', !!on);
    if (on) { clearTimeout(litTimers[m]); d.classList.remove('hit'); void d.offsetWidth; d.classList.add('hit'); }
  }

  // ------------------------------------------------------------------ trail + particles (canvas)
  const ctx = el.trail.getContext('2d');
  let cw = 0, ch = 0, dpr = 1;
  const notes = [];      // {m, color, t0, t1, group}
  const particles = [];  // {x, y, vx, vy, life, max, color, r}
  const flashes = [];    // {x, color, t0}
  const SPEED = 150;     // px per second
  let groupSeq = 0;

  function resizeCanvas() {
    const r = el.trail.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    cw = Math.max(1, r.width); ch = Math.max(1, r.height);
    el.trail.width = Math.round(cw * dpr); el.trail.height = Math.round(ch * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  if (window.ResizeObserver) new ResizeObserver(resizeCanvas).observe(el.trail);
  window.addEventListener('resize', resizeCanvas);

  function addTrailNote(m, color, holdMs, group) {
    const t = now();
    const n = { m: m, color: color, t0: t, t1: holdMs == null ? null : t + holdMs, group: group || 0 };
    notes.push(n);
    return n;
  }
  function burst(m, color, strength) {
    const g = geom[m]; if (!g) return;
    const x = g.center * cw, y = ch - 1;
    const count = Math.round(14 * (strength || 1));
    for (let i = 0; i < count; i++) {
      const a = -Math.PI / 2 + (Math.random() - 0.5) * 1.5;
      const sp = 70 + Math.random() * 230 * (strength || 1);
      particles.push({ x: x + (Math.random() - 0.5) * g.width * cw * 0.6, y: y, vx: Math.cos(a) * sp * 0.6, vy: Math.sin(a) * sp,
        life: 0, max: 0.45 + Math.random() * 0.5, color: Math.random() < 0.25 ? '#ffffff' : color, r: 1 + Math.random() * 2.2 });
    }
    flashes.push({ x: x, color: color, t0: now() });
    if (particles.length > 900) particles.splice(0, particles.length - 900);
  }

  let lastFrame = now();
  function frame() {
    const t = now();
    const dt = Math.min(0.05, (t - lastFrame) / 1000);
    lastFrame = t;
    ctx.clearRect(0, 0, cw, ch);

    // faint octave lanes
    ctx.lineWidth = 1;
    for (let m = LO; m <= HI; m += 12) {
      const x = Math.round(geom[m] ? geom[m].left * cw : 0) + 0.5;
      ctx.strokeStyle = 'rgba(120,140,190,0.07)';
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ch); ctx.stroke();
    }
    // horizon glow
    const hg = ctx.createLinearGradient(0, ch - 40, 0, ch);
    hg.addColorStop(0, 'rgba(94,231,255,0)'); hg.addColorStop(1, 'rgba(94,231,255,0.10)');
    ctx.fillStyle = hg; ctx.fillRect(0, ch - 40, cw, 40);

    // release held audio notes that stopped sounding
    const levelStale = t - S.lastLevelAt > 350;
    for (let i = 0; i < notes.length; i++) {
      const n = notes[i];
      if (n.t1 == null && n.audio) {
        const age = t - n.t0;
        if ((age > 140 && (levelStale || !S.active.has(n.m))) || age > 2500) n.t1 = t;
      }
    }

    // notes
    const groupsTop = {};
    ctx.save();
    for (let i = notes.length - 1; i >= 0; i--) {
      const n = notes[i];
      const g = geom[n.m]; if (!g) { notes.splice(i, 1); continue; }
      const end = n.t1 == null ? t : Math.min(n.t1, t);
      const yTop = ch - (t - n.t0) / 1000 * SPEED;
      const yBot = ch - (t - end) / 1000 * SPEED;
      if (yBot < -20) { notes.splice(i, 1); continue; }
      const h = Math.max(4, yBot - yTop);
      const w = Math.max(5, g.width * cw * (isBlack(n.m) ? 0.95 : 0.72));
      const x = g.center * cw - w / 2;
      const alpha = clamp(0.18 + 0.82 * (yBot / ch), 0, 1);
      ctx.globalAlpha = alpha;
      ctx.shadowColor = n.color; ctx.shadowBlur = 14;
      const grad = ctx.createLinearGradient(0, yTop, 0, yTop + h);
      grad.addColorStop(0, n.color); grad.addColorStop(1, shade(n.color, 0.55));
      ctx.fillStyle = grad;
      roundRect(ctx, x, yBot - h, w, h, Math.min(6, w / 2));
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.globalAlpha = alpha * 0.9;
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.fillRect(x + 2, yBot - h + 2, w - 4, 2);
      if (n.group) {
        const gt = groupsTop[n.group] || (groupsTop[n.group] = { y: yTop, x0: 1e9, x1: -1e9, color: n.color, a: alpha });
        gt.x0 = Math.min(gt.x0, g.center * cw); gt.x1 = Math.max(gt.x1, g.center * cw); gt.y = Math.min(gt.y, yBot - h);
      }
    }
    // chord connectors
    Object.keys(groupsTop).forEach((k) => {
      const gt = groupsTop[k]; if (gt.x1 <= gt.x0) return;
      ctx.globalAlpha = gt.a * 0.45;
      ctx.strokeStyle = gt.color; ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(gt.x0, gt.y + 6); ctx.lineTo(gt.x1, gt.y + 6); ctx.stroke();
      ctx.setLineDash([]);
    });
    ctx.restore();

    // strike flashes
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    for (let i = flashes.length - 1; i >= 0; i--) {
      const f = flashes[i];
      const a = (t - f.t0) / 380;
      if (a >= 1) { flashes.splice(i, 1); continue; }
      const r = 18 + a * 46;
      const rg = ctx.createRadialGradient(f.x, ch, 0, f.x, ch, r);
      rg.addColorStop(0, withAlpha(f.color, 0.75 * (1 - a)));
      rg.addColorStop(1, withAlpha(f.color, 0));
      ctx.fillStyle = rg;
      ctx.fillRect(f.x - r, ch - r, r * 2, r);
    }
    // particles
    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.life += dt;
      if (p.life >= p.max) { particles.splice(i, 1); continue; }
      p.vy += 260 * dt; p.vx *= 0.985;
      p.x += p.vx * dt; p.y += p.vy * dt;
      if (p.y > ch) { p.y = ch; p.vy *= -0.3; }
      const k = 1 - p.life / p.max;
      ctx.globalAlpha = k;
      ctx.fillStyle = p.color;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r * (0.6 + k * 0.6), 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
    requestAnimationFrame(frame);
  }

  function roundRect(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y); c.lineTo(x + w - r, y); c.quadraticCurveTo(x + w, y, x + w, y + r);
    c.lineTo(x + w, y + h - r); c.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    c.lineTo(x + r, y + h); c.quadraticCurveTo(x, y + h, x, y + h - r);
    c.lineTo(x, y + r); c.quadraticCurveTo(x, y, x + r, y); c.closePath();
  }
  function hexToRgb(h) { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function withAlpha(h, a) { const c = hexToRgb(h); return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')'; }
  function shade(h, k) { const c = hexToRgb(h); return 'rgb(' + Math.round(c[0] * k) + ',' + Math.round(c[1] * k) + ',' + Math.round(c[2] * k) + ')'; }

  function gestureColor(midis) {
    if (midis.length > 1) return COLORS.chord;
    const kind = keyKind(midis[0]);
    return kind === 'control' ? COLORS.control : (COLORS[kind] || COLORS.none);
  }

  // Visuals for a complete gesture (from server or from a mouse click).
  function showPress(midis, opts) {
    opts = opts || {};
    if (!midis || !midis.length) return;
    const color = gestureColor(midis);
    const group = midis.length > 1 ? ++groupSeq : 0;
    midis.forEach((m) => {
      flashKey(m, color, opts.audio ? 360 : 280);
      burst(m, color, midis.length > 1 ? 0.8 : 1);
      const n = addTrailNote(m, color, opts.audio ? null : 170, group);
      if (opts.audio) n.audio = true;
    });
  }

  // ------------------------------------------------------------------ synth (Simulate mode only)
  let actx = null, master = null, wet = null;
  function ensureAudio() {
    if (!actx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      actx = new AC();
      const comp = actx.createDynamicsCompressor();
      comp.threshold.value = -14; comp.ratio.value = 4;
      master = actx.createGain(); master.gain.value = 0.55;
      master.connect(comp); comp.connect(actx.destination);
      // small synthetic room reverb
      const len = Math.floor(actx.sampleRate * 1.6);
      const ir = actx.createBuffer(2, len, actx.sampleRate);
      for (let c = 0; c < 2; c++) {
        const d = ir.getChannelData(c);
        for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3);
      }
      const conv = actx.createConvolver(); conv.buffer = ir;
      wet = actx.createGain(); wet.gain.value = 0.22;
      wet.connect(conv); conv.connect(comp);
    }
    if (actx.state === 'suspended') actx.resume();
    return actx;
  }
  function synthOn() { return S.settings && S.settings.input_mode === 'sim'; }
  function playNote(m, vel) {
    if (!synthOn() || !ensureAudio()) return null;
    vel = vel == null ? 0.8 : vel;
    const t = actx.currentTime;
    const f = 440 * Math.pow(2, (m - 69) / 12);
    const out = actx.createGain();
    const lp = actx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.setValueAtTime(Math.min(16000, f * 10), t);
    lp.frequency.exponentialRampToValueAtTime(Math.max(300, f * 2.2), t + 1.2);
    lp.Q.value = 0.4;
    const peak = 0.20 * vel * (m < 50 ? 1.2 : 1);
    out.gain.setValueAtTime(0.0001, t);
    out.gain.exponentialRampToValueAtTime(peak, t + 0.006);
    out.gain.exponentialRampToValueAtTime(peak * 0.35, t + 0.35);
    out.gain.exponentialRampToValueAtTime(0.0001, t + 2.6);
    lp.connect(out); out.connect(master); out.connect(wet);
    const partials = [[1, 1, 0], [2, 0.42, 3], [3, 0.16, -4], [4, 0.08, 5], [5, 0.035, 0]];
    const oscs = partials.map((p) => {
      const o = actx.createOscillator();
      o.type = p[0] === 1 ? 'triangle' : 'sine';
      o.frequency.value = f * p[0];
      o.detune.value = p[2];
      const g = actx.createGain(); g.gain.value = p[1];
      o.connect(g); g.connect(lp);
      o.start(t); o.stop(t + 2.8);
      return o;
    });
    return {
      release: function () {
        const tt = actx.currentTime;
        try {
          out.gain.cancelScheduledValues(tt);
          out.gain.setValueAtTime(Math.max(0.0001, out.gain.value), tt);
          out.gain.exponentialRampToValueAtTime(0.0001, tt + 0.35);
          oscs.forEach((o) => { try { o.stop(tt + 0.4); } catch (e) { /* already stopped */ } });
        } catch (e) { /* ignore */ }
      }
    };
  }

  // ------------------------------------------------------------------ websocket
  let ws = null, wsAttempts = 0, wsTimer = null;
  function wsUrl() {
    const q = new URLSearchParams(location.search).get('ws');
    if (q) return q;
    if (location.protocol === 'file:') return 'ws://localhost:8765/ws';
    return (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
  }
  function setConn(state, text) { el.conn.dataset.state = state; el.connText.textContent = text; el.conn.title = 'Server: ' + text; }
  function connect() {
    clearTimeout(wsTimer);
    setConn('connecting', wsAttempts ? 'Reconnecting…' : 'Connecting…');
    try { ws = new WebSocket(wsUrl()); } catch (e) { scheduleReconnect(); return; }
    ws.onopen = () => { wsAttempts = 0; setConn('open', 'Connected'); };
    ws.onmessage = (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch (e) { return; }
      try { handle(msg); } catch (e) { console.warn('Piano Typer: error handling', msg && msg.type, e); }
    };
    ws.onclose = () => { ws = null; scheduleReconnect(); };
    ws.onerror = () => { /* close follows */ };
  }
  function scheduleReconnect() {
    wsAttempts++;
    const delay = Math.min(8000, 400 * Math.pow(2, wsAttempts - 1)) * (0.8 + Math.random() * 0.4);
    setConn('closed', 'Offline · retry in ' + Math.ceil(delay / 1000) + 's');
    wsTimer = setTimeout(connect, delay);
  }
  let offlineToastAt = 0;
  function send(obj) {
    if (ws && ws.readyState === 1) { ws.send(JSON.stringify(obj)); return true; }
    if (now() - offlineToastAt > 4000) { offlineToastAt = now(); toast('Not connected to the Piano Typer server'); }
    return false;
  }

  // Remember gestures we sent so we don't animate the server's echo twice.
  const recentSent = [];
  function sendSim(midis, via) {
    const ms = Array.from(new Set(midis)).filter((m) => m >= LO && m <= HI).sort((a, b) => a - b);
    if (!ms.length) return;
    recentSent.push({ k: ms.join(','), t: now() });
    if (recentSent.length > 30) recentSent.shift();
    send({ type: 'sim_press', midis: ms, via: via || 'mouse' });
  }
  function isOwnEcho(midis) {
    const k = midis.slice().sort((a, b) => a - b).join(',');
    const t = now();
    for (let i = 0; i < recentSent.length; i++) {
      if (recentSent[i].k === k && t - recentSent[i].t < 2000) { recentSent.splice(i, 1); return true; }
    }
    return false;
  }

  // ------------------------------------------------------------------ message handling
  function handle(msg) {
    switch (msg.type) {
      case 'init':
        applyLayout(msg.layout);
        if (msg.devices) S.devices = msg.devices;
        if (msg.settings) applySettings(msg.settings);
        if (msg.calib) { S.calib = msg.calib; renderCalib(); }
        if (msg.state) { S.prevText = null; renderState(msg.state, true); }
        break;
      case 'state': {
        const st = Object.assign({}, msg); delete st.type;
        renderState(st, false);
        break;
      }
      case 'press': {
        const midis = (msg.midis || []).filter((m) => m >= LO && m <= HI);
        if (msg.source === 'sim' && isOwnEcho(midis)) break;
        showPress(midis, { audio: msg.source === 'audio' });
        break;
      }
      case 'level': onLevel(msg); break;
      case 'settings': if (msg.settings) applySettings(msg.settings); break;
      case 'calib': {
        const c = Object.assign({}, msg); delete c.type;
        S.calib = c; renderCalib();
        break;
      }
      case 'toast': toast(msg.message); break;
      case 'lowest_found': toast('Lowest key found: ' + noteName(msg.midi)); break;
      case 'error': toast(msg.message || 'Error'); break;
      default: break;
    }
  }

  // ------------------------------------------------------------------ level meter / active keys
  let peak = 0, peakAt = 0;
  function onLevel(msg) {
    S.lastLevelAt = now();
    const rms = clamp(+msg.rms || 0, 0, 1);
    const v = clamp(Math.sqrt(rms) * 1.15, 0, 1);
    el.meterFill.style.width = (v * 100).toFixed(1) + '%';
    const t = now();
    if (v >= peak || t - peakAt > 900) { peak = v; peakAt = t; }
    else peak = Math.max(v, peak - 0.01);
    el.meterPeak.style.left = (peak * 98).toFixed(1) + '%';
    const next = new Set((msg.active || []).filter((m) => m >= LO && m <= HI));
    S.active.forEach((m) => { if (!next.has(m) && keyEls[m]) keyEls[m].classList.remove('sounding'); });
    next.forEach((m) => { if (!S.active.has(m) && keyEls[m]) keyEls[m].classList.add('sounding'); });
    S.active = next;
  }

  // ------------------------------------------------------------------ state rendering
  function renderState(st, initial) {
    S.state = st;
    renderText(st.text || '', initial, st.pending, st.ghost);
    renderLast(st.last, initial);
    renderSuggestions(st.suggestions || []);
    renderGlow(st.next_letters || {});
    renderAI(st.ai);
    el.shiftBadge.classList.toggle('on', !!st.shift);
    const sk = S.controlToMidi.shift;
    if (sk != null && keyEls[sk]) keyEls[sk].classList.toggle('armed', !!st.shift);
  }

  function renderText(text, initial, pending, ghost) {
    const old = S.prevText;
    const pkey = (pending ? JSON.stringify(pending) : '') + '|' + (ghost ? JSON.stringify(ghost) : '');
    if (old === text && S.prevPending === pkey && !initial) return;
    S.prevText = text;
    S.prevPending = pkey;
    let html;
    if (text && pending) {
      // the chord's word waits, underlined, until Complete / an option / the next key
      const a = pending.start, b = pending.start + pending.len;
      const cls = 'pending' + (pending.unsure ? ' unsure' : '') + (pending.ai ? ' ai' : '');
      const title = pending.unsure ? 'Not sure: check the options below' : (pending.ai ? 'The AI picked this from context' : 'Waiting: ✓ Complete to accept');
      html = esc(text.slice(0, a)) + '<span class="' + cls + '" title="' + esc(title) + '">' + esc(text.slice(a, b)) + '</span>' +
        esc(text.slice(b)) + '<span class="caret"></span>';
    } else if (!text) {
      html = '<span class="caret"></span><span class="placeholder"> Play a key to start typing…</span>';
    } else {
      let i = 0;
      if (old != null && !initial) {
        const n = Math.min(old.length, text.length);
        while (i < n && old.charCodeAt(i) === text.charCodeAt(i)) i++;
      } else i = text.length;
      // animate from the start of the word that changed
      if (i < text.length) { while (i > 0 && /\S/.test(text[i - 1])) i--; }
      const head = text.slice(0, i), fresh = text.slice(i);
      if (fresh) {
        const isWord = fresh.trim().length > 1;
        const blk = fresh.length < 40 && !/\n/.test(fresh);
        html = esc(head) + '<span class="fresh' + (isWord ? ' word' : '') + (blk ? ' blk' : '') + '">' + esc(fresh) + '</span><span class="caret"></span>';
      } else {
        html = esc(head) + '<span class="caret"></span>';
      }
    }
    if (ghost && ghost.rest && !pending && text) {
      // grey preview of the rest of the word: typed only on ✓ Complete
      html += '<span class="ghost' + (ghost.ai ? ' ai' : '') + '" title="' +
        esc((ghost.ai ? 'AI guess' : 'Most common word') + ': press ✓ Complete to accept') + '">' + esc(ghost.rest) +
        '<span class="ghost-key">✓</span></span>';
    }
    el.textOut.innerHTML = html;
    el.textScroll.scrollTop = el.textScroll.scrollHeight;
  }

  function renderLast(last, initial) {
    let html = '';
    if (last) {
      const midis = last.midis || [];
      const ai = last.ai_corrected ? '<span class="ai-mark" title="The AI picked this word from context">✨</span>' : '';
      const alts = (last.alternatives || []).filter((w) => w !== last.value);
      if (last.kind === 'word') {
        const c = chordName(midis);
        html = '<span class="tag">chord</span>' + (c.name ? '<span class="cname">' + esc(c.name) + '</span><span class="notes">' + esc(c.notes) + '</span>' : '<span class="notes">' + esc(c.notes) + '</span>') +
          ' → <span class="word">“' + esc(last.value) + '”</span>' + ai +
          (alts.length ? '<span class="alts">(also: ' + alts.slice(0, 5).map(esc).join(', ') + ')</span>' : '');
      } else if (last.kind === 'letter') {
        html = '<span class="tag">key</span>' + esc(midis.map(noteName).join(' ')) + ' → <span class="k-letter">' + esc(last.value) + '</span>' + ai;
      } else if (last.kind === 'control') {
        const lbl = CONTROL_LABEL[last.value] || '';
        html = '<span class="tag">control</span>' + esc(midis.map(noteName).join(' ')) + ' → <span class="k-ctrl">' + esc(lbl + ' ' + (CONTROL_TITLE[last.value] || last.value)) + '</span>';
      } else if (last.kind === 'char') {
        html = '<span class="tag">key</span>' + esc(midis.map(noteName).join(' ')) + ' → <span class="k-char">' + esc(last.value) + '</span>';
      } else if (last.kind === 'unknown') {
        const c = chordName(midis);
        html = '<span class="tag">chord</span>' + (c.name ? '<span class="cname">' + esc(c.name) + '</span>' : '') + '<span class="notes">' + esc(c.notes) + '</span> → <span class="k-unknown">no matching word</span>';
      }
    }
    if (el.lastLine.innerHTML !== html) {
      el.lastLine.innerHTML = html;
      if (!initial && html) { el.lastLine.classList.remove('pop'); void el.lastLine.offsetWidth; el.lastLine.classList.add('pop'); }
    }
  }

  function suggestionKeyName(s, i) {
    const k = s.key != null ? s.key : S.controlToMidi['suggest' + (i + 1)];
    return 'S' + (i + 1) + (k != null ? ' · ' + noteName(k) : '');
  }

  let lastSuggestKey = '';
  function renderSuggestions(list) {
    list = list.slice(0, 3);
    const key = JSON.stringify(list);
    if (key !== lastSuggestKey) {
      lastSuggestKey = key;
      S.hoverIdx = -1;
      if (!list.length) {
        el.suggestRow.innerHTML = '<div class="suggest-empty">Play a word as a chord — it waits underlined: ✓ Complete accepts it, S1–S3 pick another option, ⌫ rejects it.</div>';
      } else {
        el.suggestRow.innerHTML = (list[0].option ? '<div class="opt-label">Other words:</div>' : '') + list.map((s, i) =>
          '<button type="button" class="chip' + (i === 0 ? ' first' : '') + '" data-i="' + i + '" title="Chord: ' + esc((s.midis || []).map(noteName).join(' ')) + '">' +
          '<span class="w">' + esc(s.word) + '</span>' + miniKeys(s.midis || []) +
          '<span class="meta"><span class="skey">' + esc(suggestionKeyName(s, i)) + '</span>' + (i === 0 ? '<span class="kbd">⇧Tab</span>' : '') + '</span></button>'
        ).join('');
      }
    }
    // dotted outline for the top suggestion's chord
    const hint = new Set(list[0] ? list[0].midis || [] : []);
    for (let m = LO; m <= HI; m++) if (keyEls[m]) keyEls[m].classList.toggle('hint', hint.has(m));
    setHoverChord(S.hoverIdx);
  }

  function setHoverChord(i) {
    S.hoverIdx = i;
    const list = (S.state && S.state.suggestions) || [];
    const set = new Set(i >= 0 && list[i] ? list[i].midis || [] : []);
    for (let m = LO; m <= HI; m++) if (keyEls[m]) keyEls[m].classList.toggle('hover-chord', set.has(m));
  }

  // Tiny SVG keyboard (letter range) with dots on the chord keys.
  function miniKeys(midis) {
    let a = 57, b = 83;
    midis.forEach((m) => { if (m < a) a = m; if (m > b) b = m; });
    while (isBlack(a)) a--; while (isBlack(b)) b++;
    const wk = 6, h = 24;
    const whites = []; for (let m = a; m <= b; m++) if (!isBlack(m)) whites.push(m);
    const set = new Set(midis);
    let x = 0; const pos = {};
    let s = '';
    whites.forEach((m) => { pos[m] = x; s += '<rect x="' + x + '" y="0" width="' + (wk - 0.8) + '" height="' + h + '" rx="1" fill="' + (set.has(m) ? '#e4d2ff' : '#c9ccd6') + '"/>'; x += wk; });
    const W = x;
    let dots = '';
    for (let m = a; m <= b; m++) {
      if (isBlack(m)) {
        const bx = pos[m - 1] + wk - 2;
        s += '<rect x="' + bx + '" y="0" width="3.6" height="14" rx=".8" fill="' + (set.has(m) ? '#8b5cf6' : '#232733') + '"/>';
        if (set.has(m)) dots += '<circle cx="' + (bx + 1.8) + '" cy="10.5" r="1.8" fill="#f3e8ff"/>';
      } else if (set.has(m)) {
        dots += '<circle cx="' + (pos[m] + (wk - 0.8) / 2) + '" cy="19.5" r="2.2" fill="#9b5cff"/>';
      }
    }
    return '<svg class="mini" viewBox="0 0 ' + W + ' ' + h + '" width="' + Math.round(W * 1.15) + '" height="' + Math.round(h * 1.15) + '" aria-hidden="true">' + s + dots + '</svg>';
  }

  function renderGlow(nl) {
    const letters = Object.keys(S.letterToMidi);
    letters.forEach((L) => {
      const m = S.letterToMidi[L];
      const d = keyEls[m]; if (!d) return;
      const p = +nl[L] || 0;
      const g = p > 0 ? clamp(Math.sqrt(p) * 1.05, 0.08, 1) : 0;
      d.style.setProperty('--g', g.toFixed(3));
    });
  }

  function renderAI(ai) {
    ai = ai || {};
    const status = ai.status || 'unavailable';
    el.aiPill.dataset.status = status;
    let txt;
    if (status === 'off') txt = 'AI off';
    else if (status === 'unavailable') txt = 'AI unavailable';
    else if (status === 'error') txt = 'AI error';
    else {
      txt = 'AI' + (ai.model ? ' · ' + ai.model : '');
      if (status === 'thinking') txt += ' · thinking…';
      else if (ai.latency_ms != null) txt += ' · ' + Math.round(ai.latency_ms) + ' ms';
    }
    el.aiText.textContent = txt;
  }

  // ------------------------------------------------------------------ chord namer
  const CHORD_TYPES = [
    [[0, 4, 7], ' major'], [[0, 3, 7], ' minor'], [[0, 3, 6], 'dim'], [[0, 4, 8], 'aug'],
    [[0, 2, 7], 'sus2'], [[0, 5, 7], 'sus4'],
    [[0, 4, 7, 10], '7'], [[0, 4, 7, 11], 'maj7'], [[0, 3, 7, 10], 'm7'], [[0, 3, 7, 11], 'm(maj7)'],
    [[0, 3, 6, 10], 'm7♭5'], [[0, 3, 6, 9], 'dim7'], [[0, 4, 8, 10], 'aug7'],
    [[0, 4, 7, 9], '6'], [[0, 3, 7, 9], 'm6'], [[0, 2, 4, 7], 'add9'], [[0, 2, 3, 7], 'm(add9)'],
    [[0, 4, 5, 7], 'add11'], [[0, 5, 7, 10], '7sus4'], [[0, 2, 7, 10], '7sus2'],
    [[0, 2, 4, 7, 10], '9'], [[0, 2, 4, 7, 11], 'maj9'], [[0, 2, 3, 7, 10], 'm9'], [[0, 2, 4, 7, 9], '6/9'],
    [[0, 4, 10], '7(no5)'], [[0, 4, 11], 'maj7(no5)'], [[0, 3, 10], 'm7(no5)'],
    [[0, 7], '5'],
  ];
  function chordName(midis) {
    const sorted = midis.slice().sort((a, b) => a - b);
    const pcs = [];
    sorted.forEach((m) => { const p = ((m % 12) + 12) % 12; if (pcs.indexOf(p) === -1) pcs.push(p); });
    const notesStr = pcs.map((p) => NOTE_NAMES[p]).join(' ');
    if (pcs.length < 2) return { name: null, notes: notesStr };
    const bass = pcs[0];
    let best = null;
    pcs.forEach((root) => {
      const ints = pcs.map((p) => (p - root + 12) % 12).sort((a, b) => a - b);
      const key = ints.join(',');
      for (let i = 0; i < CHORD_TYPES.length; i++) {
        if (CHORD_TYPES[i][0].join(',') === key) {
          if (root !== bass && /no5|^5$/.test(CHORD_TYPES[i][1])) break;   // only name sparse shapes in root position
          const score = i + (root === bass ? 0 : 100);
          if (!best || score < best.score) best = { root: root, suffix: CHORD_TYPES[i][1], score: score };
          break;
        }
      }
    });
    if (!best) return { name: null, notes: notesStr };
    let name;
    if (best.root !== bass) {
      const suf = best.suffix === ' major' ? '' : best.suffix === ' minor' ? 'm' : best.suffix;
      name = NOTE_NAMES[best.root] + suf + '/' + NOTE_NAMES[bass];
    } else name = NOTE_NAMES[best.root] + best.suffix;
    return { name: name, notes: notesStr };
  }

  // ------------------------------------------------------------------ settings
  function applySettings(s) {
    S.settings = Object.assign({}, S.settings, s);
    if (s.devices) S.devices = s.devices;
    renderSettings();
  }
  function isFocused(node) { return document.activeElement === node; }
  function renderSettings() {
    const s = S.settings;
    el.modeSeg.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.mode === (s.input_mode || 'piano')));
    el.extInput.checked = !!s.type_external;

    // devices
    if (!isFocused(el.devSel)) {
      const devs = S.devices || [];
      const opts = devs.map((d) => '<option value="' + esc(d.index) + '">' + esc(d.name) + (d.samplerate ? ' (' + Math.round(d.samplerate / 1000) + ' kHz)' : '') + '</option>');
      if (!devs.length) opts.push('<option value="">No input devices found</option>');
      const html = opts.join('');
      if (el.devSel.dataset.html !== html) { el.devSel.innerHTML = html; el.devSel.dataset.html = html; }
      el.devSel.value = s.device != null ? String(s.device) : '';
    }
    el.audioState.textContent = s.audio_running ? '● listening' : (s.input_mode === 'sim' ? 'simulate mode' : '○ not listening');
    el.audioState.style.color = s.audio_running ? 'var(--ok)' : '';

    if (!isFocused(el.sensRange) && s.sensitivity != null) el.sensRange.value = s.sensitivity;
    el.sensOut.textContent = (+el.sensRange.value).toFixed(2);
    if (!isFocused(el.cwRange) && s.chord_window_ms != null) el.cwRange.value = s.chord_window_ms;
    el.cwOut.textContent = el.cwRange.value + ' ms';
    if (!isFocused(el.shiftNum) && s.note_shift != null) el.shiftNum.value = s.note_shift;
    el.aiChk.checked = s.ai_enabled !== false;
    el.suggChk.checked = s.suggestions !== false;

    if (!isFocused(el.modelSel)) {
      const models = (s.models && s.models.length) ? s.models : (s.model ? [s.model] : []);
      const html = models.map((m) => '<option value="' + esc(m) + '">' + esc(m) + '</option>').join('');
      if (el.modelSel.dataset.html !== html) { el.modelSel.innerHTML = html; el.modelSel.dataset.html = html; }
      if (s.model) el.modelSel.value = s.model;
    }
    el.modelSel.disabled = s.ai_enabled === false;
  }
  function setSetting(key, value) {
    S.settings[key] = value;
    send({ type: 'set', key: key, value: value });
  }
  function throttle(fn, ms) {
    let last = 0, timer = null, pending = null;
    return function (v) {
      pending = v;
      const t = now();
      if (t - last >= ms) { last = t; fn(v); }
      else { clearTimeout(timer); timer = setTimeout(() => { last = now(); fn(pending); }, ms - (t - last)); }
    };
  }

  el.modeSeg.addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    setSetting('input_mode', b.dataset.mode);
    renderSettings();
    if (b.dataset.mode === 'sim') ensureAudio();
    toast(b.dataset.mode === 'sim' ? 'Simulate mode: on-screen and laptop keys play a synth sound' : 'Piano mode: listening to your keyboard');
  });

  let countdownTimer = null;
  el.extInput.addEventListener('change', () => {
    const on = el.extInput.checked;
    setSetting('type_external', on);
    if (on) startCountdown(); else { stopCountdown(); toast('Typing stays inside Piano Typer'); }
  });
  function startCountdown() {
    stopCountdown();
    let n = 3;
    el.countdown.hidden = false;
    const tick = () => {
      el.countdownNum.textContent = n;
      el.countdownNum.classList.remove('tick'); void el.countdownNum.offsetWidth; el.countdownNum.classList.add('tick');
      n--;
      if (n < 0) { stopCountdown(); toast('Typing into the app in front. Turn it off here when done.'); return; }
      countdownTimer = setTimeout(tick, 1000);
    };
    tick();
  }
  function stopCountdown() { clearTimeout(countdownTimer); el.countdown.hidden = true; }
  el.countdown.addEventListener('click', stopCountdown);

  el.devSel.addEventListener('change', () => { const v = el.devSel.value; if (v !== '') setSetting('device', parseInt(v, 10)); });
  const sendSens = throttle((v) => setSetting('sensitivity', v), 120);
  el.sensRange.addEventListener('input', () => { el.sensOut.textContent = (+el.sensRange.value).toFixed(2); sendSens(+el.sensRange.value); });
  const sendCw = throttle((v) => setSetting('chord_window_ms', v), 120);
  el.cwRange.addEventListener('input', () => { el.cwOut.textContent = el.cwRange.value + ' ms'; sendCw(parseInt(el.cwRange.value, 10)); });
  el.shiftNum.addEventListener('change', () => {
    const v = clamp(parseInt(el.shiftNum.value, 10) || 0, -24, 24);
    el.shiftNum.value = v; setSetting('note_shift', v);
  });
  el.aiChk.addEventListener('change', () => { setSetting('ai_enabled', el.aiChk.checked); renderSettings(); });
  el.suggChk.addEventListener('change', () => setSetting('suggestions', el.suggChk.checked));
  el.modelSel.addEventListener('change', () => setSetting('model', el.modelSel.value));

  function openDrawer(open) {
    el.drawer.classList.toggle('open', open);
    el.drawer.setAttribute('aria-hidden', open ? 'false' : 'true');
    el.scrim.hidden = !open;
    if (!open && el.drawer.contains(document.activeElement)) document.activeElement.blur();
  }
  $('#gearBtn').addEventListener('click', () => openDrawer(!el.drawer.classList.contains('open')));
  $('#drawerClose').addEventListener('click', () => openDrawer(false));
  el.scrim.addEventListener('click', () => openDrawer(false));

  $('#findLowestBtn').addEventListener('click', () => { if (send({ type: 'calib_start', mode: 'lowest' })) openDrawer(false); });
  $('#calibAllBtn').addEventListener('click', () => { if (send({ type: 'calib_start', mode: 'keys' })) openDrawer(false); });
  $('#calibSkip').addEventListener('click', () => send({ type: 'calib_skip' }));
  $('#calibCancel').addEventListener('click', () => send({ type: 'calib_cancel' }));
  const clearText = () => send({ type: 'clear' });
  $('#clearBtn').addEventListener('click', clearText);
  $('#clearBtn2').addEventListener('click', clearText);

  // ------------------------------------------------------------------ calibration overlay
  let calibTargetPrev = null;
  function renderCalib() {
    const c = S.calib || {};
    const active = c.phase && c.phase !== 'idle';
    el.calib.hidden = !active;
    if (calibTargetPrev != null && keyEls[calibTargetPrev]) keyEls[calibTargetPrev].classList.remove('calib-target');
    calibTargetPrev = null;
    if (!active) return;
    el.calibTitle.textContent = c.phase === 'lowest' ? 'Setup · find the lowest key' : 'Setup · calibrate keys';
    let msg = c.message || (c.phase === 'lowest' ? 'Press the lowest key on your keyboard' : 'Press the highlighted key');
    if (c.target != null && !c.message) msg = 'Press the key marked ' + keyLabel(S.keys[c.target]) + ' (' + noteName(c.target) + ')';
    el.calibMsg.textContent = msg;
    const total = c.total || 0, done = c.done || 0;
    el.calibFill.style.width = total ? (100 * done / total).toFixed(1) + '%' : (c.phase === 'lowest' ? '50%' : '0%');
    el.calibCount.textContent = total ? done + ' / ' + total + ' keys' : (c.target != null ? noteName(c.target) : 'Waiting for a key…');
    el.calibSkip.hidden = c.phase !== 'keys';
    if (c.target != null && keyEls[c.target]) { keyEls[c.target].classList.add('calib-target'); calibTargetPrev = c.target; }
  }

  // ------------------------------------------------------------------ suggestion chips
  el.suggestRow.addEventListener('click', (e) => {
    const b = e.target.closest('.chip'); if (!b) return;
    pickSuggestion(parseInt(b.dataset.i, 10));
    b.blur();
  });
  el.suggestRow.addEventListener('mouseover', (e) => {
    const b = e.target.closest('.chip');
    setHoverChord(b ? parseInt(b.dataset.i, 10) : -1);
  });
  el.suggestRow.addEventListener('mouseleave', () => setHoverChord(-1));
  function pickSuggestion(i) {
    const list = (S.state && S.state.suggestions) || [];
    if (!list[i]) return;
    const chip = el.suggestRow.querySelector('.chip[data-i="' + i + '"]');
    if (chip) { chip.classList.remove('picked'); void chip.offsetWidth; chip.classList.add('picked'); }
    const k = list[i].key != null ? list[i].key : S.controlToMidi['suggest' + (i + 1)];
    if (k != null) showPress([k]);
    send({ type: 'pick_suggestion', index: i });
  }

  // ------------------------------------------------------------------ mouse input on keys
  const pendingMouse = new Set();
  el.kb.addEventListener('pointerdown', (e) => {
    const d = e.target.closest('.key'); if (!d) return;
    e.preventDefault();
    const m = parseInt(d.dataset.midi, 10);
    if (synthOn()) ensureAudio();
    if (e.shiftKey) {
      if (pendingMouse.has(m)) { pendingMouse.delete(m); d.classList.remove('pending'); }
      else { pendingMouse.add(m); d.classList.add('pending'); playNote(m, 0.6); }
      return;
    }
    const v = playNote(m);
    if (v) setTimeout(v.release, 600);
    showPress([m]);
    sendSim([m]);
  });
  function flushMouseChord() {
    if (!pendingMouse.size) return;
    const ms = Array.from(pendingMouse);
    pendingMouse.clear();
    ms.forEach((m) => keyEls[m] && keyEls[m].classList.remove('pending'));
    ms.forEach((m) => { const v = playNote(m, 0.7); if (v) setTimeout(v.release, 700); });
    showPress(ms);
    sendSim(ms);
  }

  // ------------------------------------------------------------------ laptop keyboard input
  // Each held key is a record; keys whose down-times overlap meaningfully form one chord.
  const records = [];          // {code, midi, down, up, voice, note}
  const heldByCode = {};
  const OVERLAP_MIN_MS = 35;   // overlap needed to count as "together"
  const OVERLAP_FRAC = 0.45;   // ...and at least this fraction of the shorter key's hold

  function keyToMidi(e) {
    const c = S.controlToMidi;
    if (e.key === ' ' || e.code === 'Space') return c.space;
    if (e.key === 'Backspace') return (e.ctrlKey || e.altKey) && c.delete_word != null ? c.delete_word : c.backspace;
    if (e.key === 'Enter') return c.enter;
    if (e.key && e.key.length === 1) {
      const l = e.key.toLowerCase();
      if (S.letterToMidi[l] != null) return S.letterToMidi[l];
      if (S.charToMidi[e.key] != null) return S.charToMidi[e.key];
    }
    return null;
  }
  function hijackAllowed() {
    // While "Type into other apps" is on, the app itself sends keystrokes; if this
    // page has focus they would come straight back here as presses -> endless loop.
    if (S.settings && S.settings.type_external) return false;
    const a = document.activeElement;
    if (!a || a === document.body) return true;
    if (el.drawer.contains(a)) return false;
    if (a.matches('select, textarea, [contenteditable="true"], input[type="text"], input[type="number"], input[type="range"]')) return false;
    if (a.matches('button, input')) a.blur();   // don't let Space/Enter re-click a focused button
    return true;
  }
  function related(a, b, t) {
    const aUp = a.up == null ? t : a.up, bUp = b.up == null ? t : b.up;
    const ov = Math.min(aUp, bUp) - Math.max(a.down, b.down);
    if (ov <= 0) return false;
    const minDur = Math.min(aUp - a.down, bUp - b.down);
    return ov >= OVERLAP_MIN_MS && ov >= OVERLAP_FRAC * minDur;
  }
  function overlapsAny(a, b) {
    const aUp = a.up == null ? Infinity : a.up, bUp = b.up == null ? Infinity : b.up;
    return Math.min(aUp, bUp) > Math.max(a.down, b.down);
  }
  // Emit finished gestures in order. A cluster is emitted once all its keys are up and
  // no still-held key overlaps it (a held key could still turn out to belong to it).
  function emitReady() {
    const t = now();
    while (records.length && records[0].up != null) {
      const released = records.filter((r) => r.up != null);
      const held = records.filter((r) => r.up == null);
      const cluster = [records[0]];
      const inC = new Set(cluster);
      for (let i = 0; i < cluster.length; i++) {
        released.forEach((r) => { if (!inC.has(r) && related(cluster[i], r, t)) { inC.add(r); cluster.push(r); } });
      }
      if (cluster.some((r) => held.some((h) => overlapsAny(r, h)))) break;
      cluster.forEach((r) => records.splice(records.indexOf(r), 1));
      cluster.sort((a, b) => a.down - b.down);
      const midis = Array.from(new Set(cluster.map((r) => r.midi)));
      if (midis.length > 1) {
        const group = ++groupSeq;
        cluster.forEach((r) => { if (r.note) { r.note.color = COLORS.chord; r.note.group = group; } flashKey(r.midi, COLORS.chord, 300); burst(r.midi, COLORS.chord, 0.6); });
      }
      sendSim(midis, "kbd");
    }
  }
  function onKeyDown(e) {
    if (e.key === 'Escape') {
      if (!el.help.hidden) { closeHelp(); return; }
      if (el.drawer.classList.contains('open')) { openDrawer(false); return; }
      if (!el.countdown.hidden) { stopCountdown(); return; }
    }
    if (!hijackAllowed()) return;
    if (!el.help.hidden) return;
    if (e.key === 'Tab' && !e.ctrlKey && !e.altKey && !e.metaKey) {
      // Tab = ✓ Complete (Shift+Tab = option 1)
      e.preventDefault();
      if (!e.repeat) {
        const m = e.shiftKey ? S.controlToMidi.suggest1 : S.controlToMidi.complete;
        if (m != null) { showPress([m]); sendSim([m], "kbd"); } else pickSuggestion(0);
      }
      return;
    }
    if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z') && S.controlToMidi.undo != null) {
      e.preventDefault();
      if (!e.repeat) { showPress([S.controlToMidi.undo]); sendSim([S.controlToMidi.undo], "kbd"); }
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key !== 'Backspace') return;
    if (e.altKey) return;
    const m = keyToMidi(e);
    if (m == null) return;
    e.preventDefault();
    if (e.repeat || heldByCode[e.code]) return;
    const color = gestureColor([m]);
    const rec = { code: e.code, midi: m, down: now(), up: null, voice: playNote(m), note: null };
    rec.note = addTrailNote(m, color, null, 0);
    setHeld(m, true, color);
    burst(m, color, 1);
    heldByCode[e.code] = rec;
    records.push(rec);
  }
  function releaseRec(rec) {
    rec.up = now();
    if (rec.note) rec.note.t1 = Math.max(rec.up, rec.note.t0 + 90);
    if (rec.voice) rec.voice.release();
    const stillHeld = Object.keys(heldByCode).some((c) => heldByCode[c] !== rec && heldByCode[c].midi === rec.midi);
    if (!stillHeld) { const d = keyEls[rec.midi]; if (d) { clearTimeout(litTimers[rec.midi]); litTimers[rec.midi] = setTimeout(() => d.classList.remove('lit', 'hit'), 60); } }
  }
  function onKeyUp(e) {
    if (e.key === 'Shift') flushMouseChord();
    const rec = heldByCode[e.code];
    if (!rec) return;
    delete heldByCode[e.code];
    releaseRec(rec);
    emitReady();
  }
  function releaseAll() {
    Object.keys(heldByCode).forEach((c) => { releaseRec(heldByCode[c]); delete heldByCode[c]; });
    emitReady();
    flushMouseChord();
  }
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);
  window.addEventListener('blur', releaseAll);
  document.addEventListener('visibilitychange', () => { if (document.hidden) releaseAll(); });
  // Safety net: a keyup that never arrives (OS shortcut, IME…) must not block output forever.
  setInterval(() => {
    const t = now();
    let changed = false;
    Object.keys(heldByCode).forEach((c) => {
      if (t - heldByCode[c].down > 4000) { releaseRec(heldByCode[c]); delete heldByCode[c]; changed = true; }
    });
    if (changed) emitReady();
  }, 500);

  // ------------------------------------------------------------------ help card
  const HELP_KEY = 'pianotyper.help.dismissed.v1';
  function closeHelp() {
    el.help.hidden = true;
    try { localStorage.setItem(HELP_KEY, '1'); } catch (e) { /* storage unavailable */ }
  }
  $('#helpOk').addEventListener('click', closeHelp);
  el.help.addEventListener('click', (e) => { if (e.target === el.help) closeHelp(); });
  $('#helpBtn').addEventListener('click', () => { el.help.hidden = false; });
  let seen = false;
  try { seen = localStorage.getItem(HELP_KEY) === '1'; } catch (e) { seen = false; }
  el.help.hidden = seen;

  // ------------------------------------------------------------------ toasts
  function toast(message) {
    if (!message) return;
    const d = document.createElement('div');
    d.className = 'toast';
    d.textContent = message;
    el.toasts.appendChild(d);
    while (el.toasts.children.length > 4) el.toasts.removeChild(el.toasts.firstChild);
    setTimeout(() => { d.classList.add('out'); setTimeout(() => d.remove(), 350); }, 3600);
  }

  // ------------------------------------------------------------------ boot
  applyLayout(null);
  renderState({ text: '', suggestions: [], next_letters: {}, last: null, ai: { status: 'unavailable' }, shift: false }, true);
  renderSettings();
  resizeCanvas();
  requestAnimationFrame(frame);
  connect();

  // tiny debug hook
  window.PianoTyper = { S: S, chordName: chordName, handle: handle };
})();
