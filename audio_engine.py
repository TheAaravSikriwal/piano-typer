"""
audio_engine.py - Piano Typer audio engine.

Listens to the keyboard's headphone output (USB sound card), detects which keys
were pressed - single notes and chords - and reports one "press" event per
gesture.  See SPEC.md for the public API.

Pipeline (per hop of ~10.7 ms):
  1. Onset detection: log-magnitude spectral flux on a 2048-sample frame, with an
     adaptive noise floor, a silence gate and re-arm hysteresis (so re-striking a
     key that is still ringing is caught, but one attack fires only once).
  2. The first onset opens a gesture.  Further onsets within `chord_window_ms`
     belong to the same gesture.  When the window closes, the "new energy"
     spectrum  d = max(P_after - B_before, 0)  is computed (P: 4096-sample Hann
     window ending at window close; B: element-wise minimum of spectra taken just
     before the onset, so notes that were already ringing cancel out).
  3. d is decomposed with non-negative least squares against a dictionary of
     per-key templates (synthetic harmonic combs, or measured spectra after
     calibration).  Candidates are thresholded relative to the strongest note,
     refitted, and a weaker note in a harmonic relation (12, 19, 24, 28...
     semitones) to a stronger one must reach a fraction of its strength, which
     kills octave / twelfth ghost notes (stricter for uncalibrated keys).
  4. One press event with the sorted nominal MIDI numbers is emitted.

Thread model: the sounddevice callback only enqueues blocks; a worker thread runs
the pipeline and calls on_event / on_level.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time

import numpy as np
from scipy.optimize import nnls

LOWEST_MIDI = 36
HIGHEST_MIDI = 96
N_KEYS = HIGHEST_MIDI - LOWEST_MIDI + 1

FMIN = 50.0           # lowest frequency used by the note analysis (Hz)
FMAX = 5000.0         # highest frequency used (Hz)
COMPRESS = 0.5        # magnitude compression exponent for the NNLS fit
SYN_SHAPES = (0.8, 1.8)    # spectral decay exponents of the synthetic templates
SYN_B = 1.5e-4        # default inharmonicity coefficient of synthetic templates
MAX_PARTIALS = 16
CAL_TILTS = (0.0, 0.5, -0.5)  # spectral tilt variants added to calibrated templates
MAX_NOTES = 8
RISE_RATIO = 0.5
ENV_FALL = 0.85
MIN_NEW_FRAC = 0.015
FUND_RATIO = 0.12
HARM_INTERVALS = (12, 19, 24, 28, 31, 34, 36)
RETRIGGER_S = 0.25    # a "new" press this soon that repeats a note of the last press is the same note
CLIP_LEVEL = 0.97     # input peak treated as clipping


# --------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------

def midi_to_hz(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


def hz_to_midi(f):
    return 69.0 + 12.0 * np.log2(f / 440.0)


def yin_pitch(frame, sr, fmin=50.0, fmax=2200.0, threshold=0.15):
    """YIN pitch detector. Returns (freq Hz, confidence 0..1) or (None, 0)."""
    x = np.asarray(frame, dtype=np.float64)
    x = x - x.mean()
    w = len(x) // 2
    tau_min = max(2, int(sr / fmax))
    tau_max = min(int(sr / fmin), w)
    if tau_max <= tau_min + 2:
        return None, 0.0
    cs = np.concatenate(([0.0], np.cumsum(x * x)))
    taus = np.arange(tau_max + 1)
    e0 = cs[w]
    etau = cs[w + taus] - cs[taus]
    a = x[:w]
    b = x[: w + tau_max]
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    corr = np.fft.irfft(np.conj(np.fft.rfft(a, n)) * np.fft.rfft(b, n), n)[: tau_max + 1]
    d = e0 + etau - 2 * corr
    d[0] = 0.0
    cmnd = np.ones_like(d)
    running = np.cumsum(d[1:])
    running[running <= 0] = 1e-12
    cmnd[1:] = d[1:] * np.arange(1, tau_max + 1) / running
    below = np.where(cmnd[tau_min:] < threshold)[0]
    if len(below) == 0:
        return None, 0.0
    tau = below[0] + tau_min
    while tau + 1 <= tau_max and cmnd[tau + 1] < cmnd[tau]:
        tau += 1
    if 1 <= tau < tau_max:
        s0, s1, s2 = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        den = s0 - 2 * s1 + s2
        shift = 0.5 * (s0 - s2) / den if den != 0 else 0.0
    else:
        shift = 0.0
    return float(sr / (tau + shift)), float(max(0.0, 1.0 - cmnd[tau]))


def _hann_kernel(delta):
    """|DTFT| of a Hann window, normalised to 1 at 0; delta in resolution bins."""
    d = np.abs(delta)
    out = np.zeros_like(d)
    m = d < 3.0
    dd = d[m]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.sinc(dd) / (1.0 - dd * dd)
    v[np.isclose(dd, 1.0)] = 0.5
    out[m] = np.abs(v)
    return out


# --------------------------------------------------------------------------------
# model: settings + templates shared between the live and offline detectors
# --------------------------------------------------------------------------------

class _Model:
    def __init__(self):
        self.lock = threading.RLock()
        self.sensitivity = 0.5
        self.note_shift = 0
        self.chord_window_ms = 70.0
        self.calib = {}          # nominal midi -> (freqs ndarray, mags ndarray)
        self.version = 0
        self.cancel_gen = 0
        self.dict_cache = {}     # sr -> (version, A, col_keys, key_cols, cal_keys)
        # pending one-shot tasks: ("lowest", None) or ("calib", midi)
        self.task = None

    def bump(self):
        with self.lock:
            self.version += 1

    # --- sensitivity-derived thresholds ---
    def thresholds(self):
        s = float(np.clip(self.sensitivity, 0.0, 1.0))
        return {
            "rel_cal": 0.40 - 0.20 * s,        # min note strength relative to strongest
            "rel_def": 0.50 - 0.20 * s,        # same, for keys with synthetic templates
            "harm_cal": 0.55 - 0.30 * s,       # harmonic-relative ghost limit (calibrated)
            "harm_def": 0.75 - 0.30 * s,       # same, synthetic templates
            "flux": 0.035 * (1.7 - 1.4 * s),   # onset flux threshold (log10 units/bin)
            "gate": 4.0 * (1.6 - 1.2 * s),     # onset gate, multiples of noise rms
            "abs": 1.5e-4 * (1.6 - 1.2 * s),   # absolute min note magnitude
        }


class _Detector:
    """Streaming analysis state for one sample rate. Not thread-safe by itself."""

    def __init__(self, model: _Model, sr: int, emit, want_level=None, offline=False):
        self.m = model
        self.sr = int(sr)
        self.emit = emit
        self.want_level = want_level
        self.offline = offline
        sc = self.sr / 48000.0
        self.hop = max(64, int(round(512 * sc)))
        self.win = int(round(4096 * sc))
        self.owin = int(round(2048 * sc))
        self.swin = int(round(1024 * sc))
        self.nfft = 1 << int(np.ceil(np.log2(self.win * 2)))
        self.offt = 1 << int(np.ceil(np.log2(self.owin)))
        freqs = np.fft.rfftfreq(self.nfft, 1.0 / self.sr)
        self.lo = int(np.searchsorted(freqs, FMIN))
        self.hi = int(np.searchsorted(freqs, min(FMAX, self.sr * 0.45)))
        self.freqs = freqs[self.lo:self.hi]
        ofreqs = np.fft.rfftfreq(self.offt, 1.0 / self.sr)
        self.ohi = int(np.searchsorted(ofreqs, min(FMAX, self.sr * 0.45)))
        self.olo = int(np.searchsorted(ofreqs, 40.0))
        self._windows = {}
        # ring buffer
        self.ring_len = int(self.sr * 3)
        self.ring = np.zeros(self.ring_len, dtype=np.float32)
        self.n_abs = 0                 # total samples received
        self.pending = np.zeros(0, dtype=np.float32)
        # onset state
        self.noise_rms = None
        self.noise_med = None
        self.prev_logs = []            # last few log spectra (onset frame)
        self.prev_mags = []
        self.rms_hist = []
        self.debug = []
        self.recent = {}               # key -> sample index of its last press
        self.last_press = None         # (onset sample, set of notes) of the last emitted press
        self.last_clip_warn = -10 ** 12
        self.flux_hist = []
        self.armed = True
        self.last_onset = -10 ** 9
        self.prev_gesture_onset = -10 ** 9
        self.gesture = None            # dict(e0, due)
        self.jobs = []                 # deferred yin/calib jobs
        # level / active
        self.hop_count = 0
        self.active = set()
        self.active_strength = {}
        self.dict_version = -1
        self.proc_time = 0.0
        self.proc_hops = 0
        self._rebuild_dict()
        rng = np.random.default_rng(0)
        nz = rng.standard_normal(self.win * 4)
        self.noise_spec_unit = float(np.median([
            np.linalg.norm(self._spec(nz[i * self.win:(i + 1) * self.win]))
            for i in range(4)]))

    # ---------------------------------------------------------------- spectra
    def _window(self, n):
        w = self._windows.get(n)
        if w is None:
            w = np.hanning(n + 2)[1:-1].astype(np.float64)
            w = w / (w.sum() / 2.0)       # sinusoid of amplitude A -> peak A
            self._windows[n] = w
        return w

    def _get(self, end, n):
        """Samples [end-n, end) by absolute index (zeros if not available)."""
        out = np.zeros(n, dtype=np.float64)
        start = end - n
        avail_start = max(start, self.n_abs - self.ring_len, 0)
        if end <= avail_start:
            return out
        idx = np.arange(avail_start, end) % self.ring_len
        out[avail_start - start:] = self.ring[idx]
        return out

    def _spec(self, x):
        """Magnitude spectrum on the analysis grid for an arbitrary-length frame."""
        w = self._window(len(x))
        X = np.abs(np.fft.rfft(x * w, self.nfft))
        return X[self.lo:self.hi]

    def _spec_at(self, end, n=None):
        n = n or self.win
        return self._spec(self._get(end, n))

    # ------------------------------------------------------------ dictionary
    def _rebuild_dict(self):
        with self.m.lock:
            shift = self.m.note_shift
            calib = dict(self.m.calib)
            self.dict_version = self.m.version
            cached = self.m.dict_cache.get(self.sr)
        if cached is not None and cached[0] == self.dict_version:
            _, self.A, self.col_keys, self.key_cols, self.cal_keys = cached
            return
        res = self.sr / self.win      # Hz per resolution bin of the analysis window
        cols, keys = [], []
        raw_cols = []
        f = self.freqs
        for m in range(LOWEST_MIDI, HIGHEST_MIDI + 1):
            f0 = float(midi_to_hz(m + shift))
            if m in calib:
                cf, cm = calib[m]
                base = np.interp(f, cf, cm, left=0.0, right=0.0)
                if base.max() <= 0:
                    continue
                for tilt in CAL_TILTS:
                    t = base * np.clip(f / f0, 0.25, 40.0) ** tilt
                    raw_cols.append(t)
                    keys.append(m)
                continue
            for p in SYN_SHAPES:
                t = np.zeros_like(f)
                k = np.arange(1, MAX_PARTIALS + 1)
                fk = k * f0 * np.sqrt(1.0 + SYN_B * k * k)
                for kk, fp in zip(k, fk):
                    if fp > f[-1] + 3 * res:
                        break
                    amp = kk ** (-p)
                    eps = fp * (2 ** ((12 + 2.0 * kk) / 1200.0) - 1)  # tolerance Hz
                    lob = np.maximum.reduce([
                        _hann_kernel((f - (fp + s * eps)) / res) for s in (-1, 0, 1)])
                    t = np.maximum(t, amp * lob)
                if t.max() <= 0:
                    continue
                raw_cols.append(t)
                keys.append(m)
        A = np.array(raw_cols).T ** COMPRESS
        A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-12
        self.A = A
        self.col_keys = np.array(keys)
        self.key_cols = {}
        for i, k in enumerate(keys):
            self.key_cols.setdefault(int(k), []).append(i)
        self.cal_keys = frozenset(calib)
        with self.m.lock:
            self.m.dict_cache[self.sr] = (self.dict_version, self.A, self.col_keys,
                                          self.key_cols, self.cal_keys)

    def _check_dict(self):
        if self.dict_version != self.m.version:
            self._rebuild_dict()

    # --------------------------------------------------------------- feeding
    def feed(self, block):
        block = np.asarray(block, dtype=np.float32).ravel()
        if len(self.pending):
            block = np.concatenate((self.pending, block))
        n = len(block)
        pos = 0
        while n - pos >= self.hop:
            chunk = block[pos:pos + self.hop]
            pos += self.hop
            idx = np.arange(self.n_abs, self.n_abs + self.hop) % self.ring_len
            self.ring[idx] = chunk
            self.n_abs += self.hop
            t0 = time.perf_counter()
            try:
                self._process_hop(chunk)
            finally:
                self.proc_time += time.perf_counter() - t0
                self.proc_hops += 1
        self.pending = block[pos:].copy()

    def flush(self):
        """Offline: pad with silence so pending gestures complete."""
        extra = int(self.sr * 0.6)
        self.feed(np.zeros(extra, dtype=np.float32))

    # ------------------------------------------------------------ per hop
    def _process_hop(self, chunk):
        self._check_dict()
        th = self.m.thresholds()
        now = self.n_abs
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        if not self.offline and now - self.last_clip_warn > 8 * self.sr \
                and float(np.max(np.abs(chunk))) >= CLIP_LEVEL:
            self.last_clip_warn = now
            self.emit({"type": "error", "message": "Too loud: the input is clipping. "
                       "Turn the piano volume down (then run Calibrate all keys again)."})
        if self.noise_rms is None:
            self.noise_rms = max(rms, 1e-6)
        elif rms < self.noise_rms:
            self.noise_rms = 0.7 * self.noise_rms + 0.3 * max(rms, 1e-7)
        else:
            self.noise_rms = min(self.noise_rms * 1.0003, max(rms, 1e-7))

        # onset detection function
        x = self._get(now, self.owin)
        mag = np.abs(np.fft.rfft(x * self._window(self.owin), self.offt))[self.olo:self.ohi]
        med = float(np.median(mag))
        if self.noise_med is None:
            self.noise_med = max(med, 1e-8)
        elif med < self.noise_med:
            self.noise_med = 0.7 * self.noise_med + 0.3 * max(med, 1e-9)
        else:
            self.noise_med = min(self.noise_med * 1.0003, max(med, 1e-9))
        floor = max(self.noise_med * 6.0, 2e-6)
        L = np.log10(np.maximum(mag, floor))
        flux = 0.0
        rising = True
        if len(self.prev_logs) >= 2:
            ref = self.prev_logs[-2]
            ref = np.maximum(ref, np.maximum(np.roll(ref, 1), np.roll(ref, -1)))
            flux = float(np.mean(np.maximum(L - ref, 0.0)))
            # net magnitude change: a note release (abrupt cut -> spectral splatter)
            # raises the log flux too, but the linear magnitude goes down overall
            dm = mag * mag - self.prev_mags[-2] ** 2
            rising = float(np.sum(dm[dm > 0])) > RISE_RATIO * float(-np.sum(dm[dm < 0]))
            # and the raw envelope must not be falling (release of a held note)
            if len(self.rms_hist) >= 2 and rms < ENV_FALL * self.rms_hist[-2]:
                rising = False
        self.prev_logs.append(L)
        self.prev_mags.append(mag)
        if len(self.prev_logs) > 3:
            self.prev_logs.pop(0)
            self.prev_mags.pop(0)
        self.flux_hist.append(flux)
        self.rms_hist.append(rms)
        if len(self.rms_hist) > 4:
            self.rms_hist.pop(0)
        if len(self.flux_hist) > 40:
            self.flux_hist.pop(0)

        loud = rms > max(th["gate"] * self.noise_rms, 2e-4)
        thr = max(th["flux"], 2.5 * float(np.median(self.flux_hist)))
        if flux < 0.5 * thr:
            self.armed = True
        if self.armed and flux > thr and loud and rising:
            self.armed = False
            self._on_onset(now)

        # deferred jobs
        if self.jobs:
            for job in list(self.jobs):
                if now >= job["due"]:
                    self.jobs.remove(job)
                    self._run_job(job)

        # gesture completion
        g = self.gesture
        if g is not None and now >= g["due"]:
            self.gesture = None
            self._analyze_gesture(g)

        # level / active
        self.hop_count += 1
        if self.want_level is not None and self.hop_count % 3 == 0:
            self._level(now, th)

    def _on_onset(self, now):
        self.last_onset = now
        if self.gesture is not None:
            self.gesture["n_onsets"] += 1
            return
        # a new onset interrupts deferred jobs waiting on the previous note
        for job in list(self.jobs):
            job["cut"] = now - self.hop
        cw = max(float(self.m.chord_window_ms), 45.0) / 1000.0
        self.gesture = {"e0": now, "due": now + int(cw * self.sr),
                        "prev": self.prev_gesture_onset, "n_onsets": 1}
        self.prev_gesture_onset = now

    # ----------------------------------------------------------- analysis
    def _baseline(self, e0, prev):
        """Spectrum of what was sounding just before onset e0."""
        hop = self.hop
        end = e0 - 2 * hop
        n_long = self.win
        if prev > end - self.win:          # previous attack inside the window
            n_long = max(self.swin, end - (prev + hop))
        specs = [self._spec_at(end, min(n_long, self.win)),
                 self._spec_at(end, self.owin),
                 self._spec_at(end, self.swin),
                 self._spec_at(e0 - 3 * hop, self.swin)]
        return np.minimum.reduce(specs)

    def _feature(self, e0, prev, end):
        P = self._spec_at(end)
        B = self._baseline(e0, prev)
        self._last_P = P
        return np.maximum(P - B, 0.0)

    def _decompose(self, d, th, cand_keys=None):
        """Return dict key -> magnitude (in un-compressed amplitude-ish units)."""
        b = d ** COMPRESS
        bn = float(np.linalg.norm(b))
        if bn <= 0:
            return {}, 0.0
        if cand_keys is None:
            cols = np.arange(self.A.shape[1])
        else:
            cols = np.array([c for k in cand_keys for c in self.key_cols.get(k, [])], dtype=int)
            if len(cols) == 0:
                return {}, bn * bn
        A = self.A[:, cols]
        x, rn = nnls(A, b, maxiter=20 * A.shape[1])
        act = {}
        for ci, xv in zip(cols, x):
            if xv > 0:
                k = int(self.col_keys[ci])
                act.setdefault(k, []).append((ci, xv))
        out = {}
        for k, lst in act.items():
            v = sum(self.A[:, ci] * xv for ci, xv in lst)
            out[k] = float(np.linalg.norm(v))
        return out, float(rn * rn)

    def _has_fundamental(self, d, key):
        """Sub-harmonic ghosts (a low note whose upper partials coincide with real
        notes) have no energy at their own fundamental."""
        with self.m.lock:
            shift = self.m.note_shift
        f0 = float(midi_to_hz(key + shift))
        tol = 2 ** (40 / 1200.0)
        peaks = []
        for k in (1, 2, 3, 4):
            lo = np.searchsorted(self.freqs, k * f0 / tol)
            hi = np.searchsorted(self.freqs, k * f0 * tol * (1 + 0.002 * k))
            peaks.append(float(d[lo:hi + 1].max()) if hi + 1 > lo and lo < len(d) else 0.0)
        return peaks[0] >= FUND_RATIO * max(peaks) if max(peaks) > 0 else False

    def _transcribe(self, d, th):
        """Notes (nominal midis) whose onset energy is present in new-energy spectrum d."""
        act, _ = self._decompose(d, th)
        if not act:
            return [], act
        abs_thr = (max(th["abs"], 6.0 * self.noise_rms * self.noise_spec_unit
                       / np.sqrt(len(d)))) ** COMPRESS
        amax = max(act.values())
        if amax < abs_thr * 3:
            return [], act
        cal = self.cal_keys

        def rel(k):
            return th["rel_cal"] if k in cal else th["rel_def"]

        cand = [k for k, a in act.items() if a >= rel(k) * amax and a >= abs_thr
                and self._has_fundamental(d, k)]
        cand = sorted(cand, key=lambda k: -act[k])[:MAX_NOTES + 4]
        a2 = act
        for _ in range(16):
            if not cand:
                break
            a2, _ = self._decompose(d, th, cand)
            if not a2:
                return [], act
            amax2 = max(a2.values())
            keep = [k for k in cand if a2.get(k, 0.0) >= rel(k) * amax2]
            if len(keep) != len(cand):
                cand = keep
                continue
            # ghost notes: a weaker note in a harmonic relation to a stronger one
            drop = None
            for k in sorted(cand, key=lambda k: a2[k]):
                for j in cand:
                    if j == k or a2[j] <= a2[k] or abs(k - j) not in HARM_INTERVALS:
                        continue
                    lim = th["harm_cal"] if (k in cal and j in cal) else th["harm_def"]
                    if a2[k] < lim * a2[j]:
                        drop = k
                        break
                if drop is not None:
                    break
            if drop is None:
                break
            cand = [c for c in cand if c != drop]
        cand = sorted(cand, key=lambda k: -a2.get(k, 0))[:MAX_NOTES]
        return sorted(int(k) for k in cand), act

    def _analyze_gesture(self, g):
        e0, end = g["e0"], g["due"]
        th = self.m.thresholds()
        d = self._feature(e0, g["prev"], end)
        with self.m.lock:
            task = self.m.task
        if task is not None:
            self._start_task(task, g, d, th)
            return
        P = self._last_P
        frac = float(np.sum(d * d) / (np.sum(P * P) + 1e-20))
        notes, _ = self._transcribe(d, th) if frac >= MIN_NEW_FRAC else ([], None)
        notes = [n for n in notes if LOWEST_MIDI <= n <= HIGHEST_MIDI]
        if self.offline:
            self.debug.append((round(e0 / self.sr, 3), round(frac, 3), notes))
        if not notes:
            return
        # re-trigger: a ringing (often clipped) note can fire a second onset shortly
        # after the real one; that second "press" repeats the note plus overtone ghosts
        lp = self.last_press
        if lp is not None and e0 - lp[0] < RETRIGGER_S * self.sr and set(notes) & lp[1]:
            if self.offline:
                self.debug.append((round(e0 / self.sr, 3), "retrigger dropped", notes))
            return
        self.last_press = (e0, set(notes))
        for n in notes:
            self.recent[n] = end
        ev = {"type": "press", "midis": notes}
        if self.offline:
            ev["t"] = round(max(0, e0 - self.hop // 2) / self.sr, 4)
        else:
            ev["t"] = time.time()
        self.emit(ev)

    # ------------------------------------------------- calibration / lowest
    def _start_task(self, task, g, d, th):
        kind, midi = task
        e0 = g["e0"]
        job = {"kind": kind, "midi": midi, "e0": e0, "d": [d], "prev": g["prev"],
               "cancel_gen": self.m.cancel_gen,
               "due": e0 + int(0.27 * self.sr), "cut": None, "th": th}
        with self.m.lock:
            self.m.task = None
            self.m.task_busy = True
        self.jobs.append(job)

    def _run_job(self, job):
        if job["cancel_gen"] != self.m.cancel_gen:      # cancel_calibration() called
            return
        e0 = job["e0"]
        last = job["cut"] if job["cut"] is not None else job["due"]
        pitches = []
        for off in (0.10, 0.15, 0.20, 0.25):
            end = e0 + int(off * self.sr)
            if end > last:
                break
            f, conf = yin_pitch(self._get(end, self.win), self.sr, fmin=50.0,
                                fmax=4500.0, threshold=0.2)
            if f is not None and conf > 0.75:
                pitches.append(float(hz_to_midi(f)))
        if job["kind"] == "lowest":
            ok = len(pitches) >= 2 and (max(pitches) - min(pitches)) < 0.6
            if ok:
                midi = int(round(float(np.median(pitches))))
                with self.m.lock:
                    self.m.task_busy = False
                self.emit({"type": "lowest_found", "midi": midi})
            else:
                with self.m.lock:       # not a clear note: wait for the next one
                    if self.m.task is None:
                        self.m.task = ("lowest", None)
            return
        # calibration of one key
        target = int(job["midi"])
        with self.m.lock:
            shift = self.m.note_shift
        real = target + shift
        d = job["d"][0]
        energy_ok = float(d.max()) > max(job["th"]["abs"], 20 * self.noise_rms * 0.01)
        pitch_ok = False
        if pitches:
            p = float(np.median(pitches))
            pitch_ok = abs(p - real) < 0.7 or abs(abs(p - real) - 12) < 0.7
        if not pitch_ok:
            # fall back to the template fit with the current dictionary
            act, _ = self._decompose(d, job["th"])
            if act:
                top = max(act, key=act.get)
                pitch_ok = top == target
        ok = bool(energy_ok and pitch_ok)
        if ok:
            # average the new-energy spectrum over the analysis frame and 2 later ones
            ds = [d]
            for k in (1, 2):
                end = e0 + int(max(self.m.chord_window_ms, 45.0) / 1000 * self.sr) + k * self.hop
                if end <= last:
                    ds.append(self._feature(e0, job["prev"], end))
            avg = np.mean(ds, axis=0)
            with self.m.lock:
                self.m.calib[target] = (self.freqs.copy(), avg.astype(np.float64))
                self.m.version += 1
        with self.m.lock:
            self.m.task_busy = False
        self.emit({"type": "calib_done", "midi": target, "ok": ok, "_save": ok})

    # ---------------------------------------------------------------- level
    def _level(self, now, th):
        x = self._get(now, self.hop * 3)
        rms = float(np.sqrt(np.mean(x * x)))
        active = []
        if rms > max(th["gate"] * self.noise_rms, 2e-4):
            horizon = now - 4 * self.sr
            self.recent = {k: t for k, t in self.recent.items() if t > horizon}
            P = self._spec_at(now)
            act, _ = self._decompose(P, th, list(self.recent)) if self.recent else ({}, 0)
            if act:
                amax = max(act.values())
                abs_thr = max(th["abs"], 6.0 * self.noise_rms * self.noise_spec_unit
                              / np.sqrt(len(P))) ** COMPRESS
                new = set()
                for k, a in act.items():
                    on = a >= max(0.35 * amax, 2 * abs_thr)
                    hold = k in self.active and a >= max(0.18 * amax, abs_thr)
                    if on or hold:
                        new.add(k)
                self.active = new
        else:
            self.active = set()
        active = sorted(k for k in self.active if LOWEST_MIDI <= k <= HIGHEST_MIDI)
        self.want_level({"rms": float(min(1.0, rms)), "active": active})


# --------------------------------------------------------------------------------
# public engine
# --------------------------------------------------------------------------------

class AudioEngine:
    def __init__(self, on_event, on_level, calib_path="calibration.json"):
        self.on_event = on_event or (lambda e: None)
        self.on_level = on_level
        self.calib_path = calib_path
        self.model = _Model()
        self.model.task_busy = False
        self._stream = None
        self._worker = None
        self._q = None
        self._stop_flag = threading.Event()
        self._running = False
        self._starting = False
        self._gen = 0
        self._ctl_thread = None
        self._ctl_q = None
        self._lock = threading.RLock()
        self.device = None
        self.samplerate = None
        self.stats = {"dropped_blocks": 0, "hops": 0, "proc_time": 0.0}
        try:
            self._load_calibration()
        except Exception as e:  # never crash on a bad file
            self._safe_event({"type": "error", "message": f"calibration load failed: {e}"})

    # ------------------------------------------------------------ devices
    @staticmethod
    def rescan_devices():
        """PortAudio only scans hardware once; re-init so a newly plugged USB card appears.
        Only call while no stream is open."""
        import sounddevice as sd
        try:
            sd._terminate()
            sd._initialize()
        except Exception:
            pass

    @staticmethod
    def _query():
        import sounddevice as sd
        devs = sd.query_devices()
        apis = sd.query_hostapis()
        return devs, apis

    @staticmethod
    def list_devices() -> list:
        """Input devices, duplicates across host APIs removed (WASAPI preferred)."""
        try:
            devs, apis = AudioEngine._query()
        except Exception:
            return []
        pref = {"Windows WASAPI": 0, "MME": 1, "Windows DirectSound": 2, "Windows WDM-KS": 3}
        best = {}
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) <= 0:
                continue
            api = apis[d["hostapi"]]["name"]
            name = d["name"]
            low = name.lower()
            if "sound mapper" in low or "primary sound capture" in low:
                continue
            key = low[:31].strip()
            rank = pref.get(api, 5)
            cur = best.get(key)
            if cur is None or rank < cur[0]:
                best[key] = (rank, {"index": i, "name": name,
                                    "samplerate": int(d["default_samplerate"])})
        out = [v[1] for v in best.values()]
        out.sort(key=lambda d: d["index"])
        return out

    @staticmethod
    def _candidates():
        """All USB input devices ordered by preference (WASAPI / 48 kHz first)."""
        try:
            devs, apis = AudioEngine._query()
        except Exception:
            return []
        pref = {"Windows WASAPI": 0, "MME": 1, "Windows DirectSound": 2, "Windows WDM-KS": 3}
        c = []
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0 and "usb" in d["name"].lower():
                api = apis[d["hostapi"]]["name"]
                c.append((pref.get(api, 5), 0 if int(d["default_samplerate"]) == 48000 else 1, i))
        c.sort()
        return [i for _, _, i in c]

    @staticmethod
    def default_device():
        c = AudioEngine._candidates()
        return c[0] if c else None

    # ------------------------------------------------------------- control
    @property
    def running(self) -> bool:
        """True while the stream is open or still being opened by start()
        (a failed open turns it False and emits an error event)."""
        return self._running or self._starting

    def _safe_event(self, ev):
        try:
            self.on_event(ev)
        except Exception:
            pass

    def _emit(self, ev):
        if ev.pop("_save", False):
            try:
                self.save_calibration()
            except Exception as e:
                self._safe_event({"type": "error", "message": f"calibration save failed: {e}"})
        self._safe_event(ev)

    def start(self, device=None) -> None:
        """Open the input stream in a background thread and return immediately.

        `running` becomes True once the stream is open (normally < 1 s); failures
        are reported as {"type": "error"} events."""
        with self._lock:
            self._gen += 1
            gen = self._gen
            self._starting = True
        self._control(("start", device, gen))

    def _control(self, cmd):
        """All stream open/close calls run on one long-lived thread: WASAPI streams
        break if opened and closed on different (or finished) threads."""
        with self._lock:
            if self._ctl_thread is None or not self._ctl_thread.is_alive():
                self._ctl_q = queue.Queue()
                self._ctl_thread = threading.Thread(target=self._ctl_loop, daemon=True,
                                                    name="AudioEngineControl")
                self._ctl_thread.start()
            self._ctl_q.put(cmd)

    def _ctl_loop(self):
        # WASAPI needs COM set up on the thread that opens the stream, or start()
        # fails with "Unanticipated host error -9999".
        try:
            import ctypes
            ctypes.windll.ole32.CoInitializeEx(None, 0)
        except Exception:
            pass
        while True:
            cmd = self._ctl_q.get()
            try:
                if cmd[0] == "start":
                    self._start_sync(cmd[1], cmd[2])
                elif cmd[0] == "stop":
                    with self._lock:
                        self._stop_locked()
            except Exception as e:
                self._safe_event({"type": "error", "message": f"audio control error: {e}"})
            finally:
                if cmd[0] == "stop":
                    cmd[1].set()

    @property
    def starting(self) -> bool:
        return self._starting

    def _start_sync(self, device, gen):
        try:
            self._start_locked(device, gen)
        finally:
            with self._lock:
                if gen == self._gen:
                    self._starting = False

    def _start_locked(self, device, gen):
        with self._lock:
            if gen != self._gen:        # superseded by a newer start()/stop()
                return
            if self._running or self._stream is not None:
                self._stop_locked()
            try:
                import sounddevice as sd
            except Exception as e:
                self._safe_event({"type": "error", "message": f"sounddevice unavailable: {e}"})
                return
            # the chosen device first, then the same card through other drivers
            tries = ([device] if device is not None else []) + \
                    [c for c in self._candidates() if c != device]
            if not tries:
                self._safe_event({"type": "error",
                                  "message": "No USB audio input found (is the sound card plugged in?)"})
                return
            last_err = None
            for dev in tries:
                for attempt in range(3):      # a device that was just closed may need a moment
                    if gen != self._gen:
                        return
                    try:
                        info = sd.query_devices(dev, "input")
                        sr = int(info["default_samplerate"])
                        self._open(sd, dev, sr)
                        self.device, self.samplerate = dev, sr
                        return
                    except Exception as e:
                        last_err = e
                        self._close_stream()
                        if "querying device" in str(e):
                            break
                        time.sleep(0.3)
            self._safe_event({"type": "error", "message": f"Could not open audio input {tries[0]}: {last_err}"})

    def _open(self, sd, dev, sr):
        q = queue.Queue(maxsize=400)
        self._q = q
        stats = self.stats

        def callback(indata, frames, t, status):
            try:
                q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                stats["dropped_blocks"] += 1

        def finished():
            if self._running and not self._stop_flag.is_set():
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass

        hop = max(64, int(round(512 * sr / 48000.0)))
        stream = sd.InputStream(device=dev, channels=1, samplerate=sr, blocksize=hop,
                                dtype="float32", latency="low", callback=callback,
                                finished_callback=finished)
        stream.start()
        self._stream = stream
        self._stop_flag.clear()
        self._running = True
        det = _Detector(self.model, sr, self._emit, want_level=self._level_cb)
        self._worker = threading.Thread(target=self._work, args=(det, q), daemon=True,
                                        name="AudioEngineWorker")
        self._worker.start()

    def _level_cb(self, lv):
        if self.on_level is not None:
            try:
                self.on_level(lv)
            except Exception:
                pass

    def _work(self, det, q):
        while not self._stop_flag.is_set():
            try:
                blk = q.get(timeout=0.5)
            except queue.Empty:
                st = self._stream
                if st is not None and not st.active and not self._stop_flag.is_set():
                    self._running = False
                    self._safe_event({"type": "error", "message": "Audio input stopped (device unplugged?)"})
                    return
                continue
            if blk is None:
                self._running = False
                self._safe_event({"type": "error", "message": "Audio input stream ended (device unplugged?)"})
                return
            try:
                det.feed(blk)
                self.stats["hops"] = det.proc_hops
                self.stats["proc_time"] = det.proc_time
            except Exception as e:
                self._safe_event({"type": "error", "message": f"audio processing error: {e}"})

    def _close_stream(self):
        st, self._stream = self._stream, None
        if st is not None:
            try:
                st.stop()
            except Exception:
                pass
            try:
                st.close()
            except Exception:
                pass

    def _stop_locked(self):
        self._stop_flag.set()
        self._running = False
        self._close_stream()
        w, self._worker = self._worker, None
        if w is not None and w is not threading.current_thread():
            w.join(timeout=2.0)

    def stop(self) -> None:
        """Close the stream (waits up to ~3 s for the control thread)."""
        with self._lock:
            self._gen += 1
            self._starting = False
        done = threading.Event()
        self._control(("stop", done))
        if threading.current_thread() is not self._ctl_thread:
            done.wait(3.0)

    # ------------------------------------------------------------ settings
    def set_sensitivity(self, value: float) -> None:
        with self.model.lock:
            self.model.sensitivity = float(np.clip(value, 0.0, 1.0))

    def set_note_shift(self, semitones: int) -> None:
        with self.model.lock:
            self.model.note_shift = int(semitones)
            self.model.version += 1

    def set_chord_window(self, ms: float) -> None:
        with self.model.lock:
            self.model.chord_window_ms = float(np.clip(ms, 20.0, 300.0))

    def begin_find_lowest(self) -> None:
        with self.model.lock:
            self.model.task = ("lowest", None)

    def begin_calibrate_key(self, nominal_midi: int) -> None:
        with self.model.lock:
            self.model.task = ("calib", int(nominal_midi))

    def cancel_calibration(self) -> None:
        """Abort a pending begin_find_lowest / begin_calibrate_key."""
        with self.model.lock:
            self.model.task = None
            self.model.cancel_gen += 1

    def clear_calibration(self) -> None:
        with self.model.lock:
            self.model.calib = {}
            self.model.version += 1

    @property
    def calibrated_keys(self) -> list:
        with self.model.lock:
            return sorted(self.model.calib)

    # --------------------------------------------------------- calibration io
    def save_calibration(self) -> None:
        with self.model.lock:
            items = dict(self.model.calib)
        data = {"version": 1, "keys": {}}
        for m, (f, mag) in items.items():
            sel = mag > mag.max() * 1e-3
            data["keys"][str(m)] = {
                "f0_hz": float(f[0]), "df_hz": float(f[1] - f[0]), "n": int(len(f)),
                "mag": [round(float(v), 7) for v in mag],
            }
        tmp = self.calib_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, self.calib_path)

    def _load_calibration(self):
        if not self.calib_path or not os.path.exists(self.calib_path):
            return
        with open(self.calib_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        calib = {}
        for k, v in data.get("keys", {}).items():
            n = int(v["n"])
            f = float(v["f0_hz"]) + float(v["df_hz"]) * np.arange(n)
            mag = np.asarray(v["mag"], dtype=np.float64)
            if len(mag) == n and LOWEST_MIDI <= int(k) <= HIGHEST_MIDI:
                calib[int(k)] = (f, mag)
        with self.model.lock:
            self.model.calib = calib
            self.model.version += 1

    # --------------------------------------------------------------- offline
    def process_buffer(self, samples: np.ndarray, sr: int) -> list:
        """Run the full pipeline over an array; return the press events (t = seconds).

        Non-press events (calib_done, lowest_found) are delivered to on_event too."""
        presses = []

        def emit(ev):
            if ev.get("type") == "press":
                presses.append(ev)
            else:
                self._emit(ev)

        x = np.asarray(samples, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        det = _Detector(self.model, sr, emit, want_level=None, offline=True)
        step = det.hop * 8
        for i in range(0, len(x), step):
            det.feed(x[i:i + step])
        det.flush()
        self.stats["last_offline_hops"] = det.proc_hops
        self.stats["last_offline_proc_time"] = det.proc_time
        return presses
