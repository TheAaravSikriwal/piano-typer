"""
Offline test for audio_engine.py on synthetic keyboard audio.

    python test_audio_engine.py            # full report
    python test_audio_engine.py --quick    # smaller sets
    python test_audio_engine.py --seed 3

Synthesises a "sampled piano" (fixed per-key timbre: inharmonic partials, random
partial strengths, hammer-position nulls, per-partial exponential decay, hammer
noise, damper release, velocity-dependent loudness and brightness, slight
detuning) and an "organ" (drawbar harmonics, sustained while held, fast release),
plays sequences through AudioEngine.process_buffer and scores gesture matches.
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audio_engine as ae  # noqa: E402

SR = 48000
LETTERS = list(range(57, 83))


# ------------------------------------------------------------------ instruments

class Piano:
    name = "piano"

    def __init__(self, seed=0, shift=0):
        r = np.random.default_rng(1000 + seed)
        self.shift = shift
        self.keys = {}
        for m in range(30, 110):
            nparts = 24
            k = np.arange(1, nparts + 1)
            p = r.uniform(0.7, 1.5)
            amps = k ** (-p) * r.uniform(0.6, 1.4, nparts)
            beta = r.uniform(1 / 9, 1 / 6)                 # hammer position
            amps *= 0.15 + 0.85 * np.abs(np.sin(np.pi * k * beta))
            if m < 48:
                amps[0] *= r.uniform(0.3, 0.8)              # weak fundamental in bass
            B = np.clip(2.5e-4 * 2 ** ((m - 60) / 18), 3e-5, 3e-3) * r.uniform(0.6, 1.4)
            self.keys[m] = dict(amps=amps, B=B, detune=r.uniform(-4, 4),
                                tau0=1.6 * 2 ** (-(m - 60) / 20) * r.uniform(0.8, 1.2))

    def note(self, nominal, vel, hold, rng):
        m = nominal + self.shift
        kd = self.keys[m]
        f0 = 440 * 2 ** ((m - 69 + kd["detune"] / 100) / 12)
        dur = hold + 0.5
        n = int(dur * SR)
        t = np.arange(n) / SR
        y = np.zeros(n)
        bright = 0.6 * (1 - vel)
        for i, a in enumerate(kd["amps"]):
            k = i + 1
            fk = k * f0 * np.sqrt(1 + kd["B"] * k * k)
            if fk > 0.45 * SR:
                break
            tau = kd["tau0"] / (1 + 0.35 * (k - 1))
            amp = a * k ** (-bright)
            y += amp * np.exp(-t / tau) * np.sin(2 * np.pi * fk * t + rng.uniform(0, 2 * np.pi))
        att = int(0.003 * SR)
        y[:att] *= np.linspace(0, 1, att)
        # damper after release
        rel = int(hold * SR)
        y[rel:] *= np.exp(-(t[rel:] - hold) / 0.09)
        # hammer noise
        nn = int(0.012 * SR)
        noise = rng.standard_normal(nn) * np.exp(-np.arange(nn) / (0.003 * SR))
        noise = np.convolve(noise, np.ones(6) / 6, "same")
        y[:nn] += 0.08 * noise * np.max(np.abs(y[:2000]))
        y /= np.max(np.abs(y)) + 1e-9
        return 0.22 * vel * y


class Organ:
    name = "organ"
    DRAW = np.array([1.0, 0.75, 0.55, 0.45, 0.0, 0.35, 0.0, 0.3, 0.0, 0.15])

    def __init__(self, seed=0, shift=0):
        r = np.random.default_rng(2000 + seed)
        self.shift = shift
        self.keys = {m: dict(amps=self.DRAW * r.uniform(0.8, 1.2, len(self.DRAW)),
                             detune=r.uniform(-3, 3)) for m in range(30, 110)}

    def note(self, nominal, vel, hold, rng):
        m = nominal + self.shift
        kd = self.keys[m]
        f0 = 440 * 2 ** ((m - 69 + kd["detune"] / 100) / 12)
        dur = hold + 0.1
        n = int(dur * SR)
        t = np.arange(n) / SR
        y = np.zeros(n)
        for i, a in enumerate(kd["amps"]):
            fk = (i + 1) * f0
            if a <= 0 or fk > 0.45 * SR:
                continue
            y += a * np.sin(2 * np.pi * fk * t + rng.uniform(0, 2 * np.pi))
        env = np.ones(n)
        att = int(0.006 * SR)
        env[:att] = np.linspace(0, 1, att)
        rel = int(hold * SR)
        env[rel:] = np.exp(-(t[rel:] - hold) / 0.015)
        y *= env
        y /= np.max(np.abs(y)) + 1e-9
        return 0.18 * (0.6 + 0.4 * vel) * y  # organ: velocity barely matters


def render(events, inst, rng, total=None, noise_db=-66):
    """events: list of (time, midi, vel, hold)."""
    end = max(t + h for t, _, _, h in events) + 1.0 if total is None else total
    out = np.zeros(int(end * SR) + SR)
    for t, m, v, h in events:
        y = inst.note(m, v, h, rng)
        i = int(t * SR)
        out[i:i + len(y)] += y[:len(out) - i]
    tt = np.arange(len(out)) / SR
    out += 10 ** (noise_db / 20) * rng.standard_normal(len(out))
    out += 10 ** ((noise_db + 6) / 20) * np.sin(2 * np.pi * 50 * tt)
    return np.clip(out, -1, 1).astype(np.float32)


# ------------------------------------------------------------------ sequences

def _release_before(ev, m, t, gap=0.04):
    """A key must be released (>= gap s) before it can be pressed again."""
    for i in range(len(ev) - 1, -1, -1):
        t0, m0, v0, h0 = ev[i]
        if m0 == m:
            if t0 + h0 > t - gap:
                ev[i] = (t0, m0, v0, max(0.03, t - gap - t0))
            break


def seq_singles(rng, n, low=36, high=96, letter_frac=0.7, ioi=(0.12, 0.17)):
    t, ev, gestures = 0.5, [], []
    for _ in range(n):
        m = int(rng.integers(57, 83)) if rng.random() < letter_frac else int(rng.integers(low, high + 1))
        v = rng.uniform(0.35, 1.0)
        h = rng.uniform(0.08, 0.25)
        _release_before(ev, m, t)
        ev.append((t, m, v, h))
        gestures.append((t, [m]))
        t += rng.uniform(*ioi)
    return ev, gestures


def seq_chords(rng, n, sizes=(2, 6), spread=0.030, gap=(0.35, 0.6)):
    t, ev, gestures = 0.5, [], []
    for _ in range(n):
        k = int(rng.integers(sizes[0], sizes[1] + 1))
        notes = sorted(int(x) for x in rng.choice(LETTERS, k, replace=False))
        base_v = rng.uniform(0.45, 1.0)
        h = rng.uniform(0.15, 0.35)
        offs = rng.uniform(0, spread, k)
        offs -= offs.min()
        for m, o in zip(notes, offs):
            _release_before(ev, m, t + o)
            ev.append((t + o, m, base_v * rng.uniform(0.75, 1.0), h))
        gestures.append((t, notes))
        t += rng.uniform(*gap)
    return ev, gestures


def seq_repeats(rng, n_groups, reps=(3, 6), ioi=(0.13, 0.2)):
    t, ev, gestures = 0.5, [], []
    for _ in range(n_groups):
        m = int(rng.integers(57, 83)) if rng.random() < 0.8 else int(rng.integers(36, 97))
        for _ in range(int(rng.integers(reps[0], reps[1] + 1))):
            h = rng.uniform(0.06, 0.12)
            _release_before(ev, m, t)
            ev.append((t, m, rng.uniform(0.4, 1.0), h))
            gestures.append((t, [m]))
            t += rng.uniform(*ioi)
        t += 0.3
    return ev, gestures


# ------------------------------------------------------------------ scoring

def score(gestures, presses, tol=(-0.04, 0.10)):
    out = sorted(presses, key=lambda e: e["t"])
    used = [False] * len(out)
    exact = 0
    missing, extra = [], []
    unmatched_truth = 0
    j0 = 0
    for t, notes in gestures:
        best = None
        for j in range(j0, len(out)):
            dt = out[j]["t"] - t
            if dt > tol[1]:
                break
            if dt >= tol[0] and not used[j]:
                best = j
                break
        if best is None:
            unmatched_truth += 1
            missing.extend(notes)
            continue
        used[best] = True
        got = set(out[best]["midis"])
        want = set(notes)
        if got == want:
            exact += 1
        for m in want - got:
            missing.append(m)
        for m in got - want:
            extra.append(min((m - w for w in want), key=abs))
    false_events = used.count(False)
    n = len(gestures)
    return {
        "n": n, "events": len(out), "exact": exact,
        "recall": exact / max(1, n), "precision": exact / max(1, len(out)),
        "missed_gestures": unmatched_truth, "false_events": false_events,
        "missing_notes": len(missing), "extra_notes": len(extra),
        "extra_intervals": extra,
    }


def fmt(name, s):
    ints = s["extra_intervals"]
    hist = {}
    for d in ints:
        key = "oct" if abs(d) in (12, 24) else "12th" if abs(d) == 19 else "semi" if abs(d) == 1 else "other"
        hist[key] = hist.get(key, 0) + 1
    return (f"  {name:<34} n={s['n']:4d}  exact={s['exact']:4d}  "
            f"recall={100 * s['recall']:5.1f}%  precision={100 * s['precision']:5.1f}%  "
            f"missedG={s['missed_gestures']} falseEv={s['false_events']} "
            f"missN={s['missing_notes']} extraN={s['extra_notes']} {hist if hist else ''}")


def per_size(gestures, presses):
    res = {}
    for k in range(1, 8):
        g = [x for x in gestures if len(x[1]) == k]
        if not g:
            continue
        s = score(g, [p for p in presses])  # false events not meaningful per size
        res[k] = s["exact"] / len(g)
    return res


# ------------------------------------------------------------------ calibration

def calibrate(engine, inst, rng, keys):
    got = []
    engine.on_event = lambda e: got.append(e)
    for m in keys:
        engine.begin_calibrate_key(m)
        audio = render([(0.3, m, 0.7, 0.4)], inst, rng, total=1.2)
        engine.process_buffer(audio, SR)
    ok = sum(1 for e in got if e.get("type") == "calib_done" and e["ok"])
    return ok


# ------------------------------------------------------------------ main

def run_suite(engine, inst, rng, quick, label):
    nS = 150 if quick else 400
    nC = 80 if quick else 250
    nR = 15 if quick else 40
    results = {}
    t_proc = 0.0
    hops = 0

    def go(ev, gestures):
        nonlocal t_proc, hops
        audio = render(ev, inst, rng)
        p = engine.process_buffer(audio, SR)
        t_proc += engine.stats["last_offline_proc_time"]
        hops += engine.stats["last_offline_hops"]
        return p

    ev, g = seq_singles(rng, nS, letter_frac=1.0)
    results["singles 57-82 @6-8/s"] = (g, go(ev, g))
    ev, g = seq_singles(rng, nS // 2, letter_frac=0.0)
    results["singles 36-96 @6-8/s"] = (g, go(ev, g))
    ev, g = seq_chords(rng, nC, sizes=(2, 4))
    results["chords 2-4 (57-82, <=30ms)"] = (g, go(ev, g))
    ev, g = seq_chords(rng, nC // 2, sizes=(5, 6))
    results["chords 5-6 (57-82, <=30ms)"] = (g, go(ev, g))
    ev, g = seq_repeats(rng, nR)
    results["repeated same note"] = (g, go(ev, g))

    print(f"\n[{label}]")
    summary = {}
    for name, (g, p) in results.items():
        s = score(g, p)
        summary[name] = s
        print(fmt(name, s))
        if name.startswith("chords"):
            ps = per_size(g, p)
            print("      exact by size: " + ", ".join(f"{k}:{100 * v:.0f}%" for k, v in ps.items()))
    print(f"  mean processing time per hop: {1000 * t_proc / max(1, hops):.3f} ms "
          f"(hop = {512 / SR * 1000:.1f} ms)")
    return summary


def test_lowest_and_shift(seed):
    rng = np.random.default_rng(seed + 77)
    ok_all = True
    for inst_cls in (Piano, Organ):
        for shift in (0, 3, -5):
            inst = inst_cls(seed, shift=shift)
            events = []
            eng = ae.AudioEngine(lambda e: events.append(e), None,
                                 calib_path=os.path.join(tempfile.gettempdir(), "pt_none.json"))
            eng.begin_find_lowest()
            audio = render([(0.3, 36, 0.8, 0.4)], inst, rng, total=1.3)
            eng.process_buffer(audio, SR)
            lf = [e for e in events if e["type"] == "lowest_found"]
            found = lf[0]["midi"] if lf else None
            ok = found == 36 + shift
            # then apply the shift as the server would and check detection
            if found is not None:
                eng.set_note_shift(found - 36)
            ev, g = seq_singles(rng, 40, letter_frac=1.0)
            s = score(g, eng.process_buffer(render(ev, inst, rng), SR))
            ok_all &= ok and s["recall"] > 0.9
            print(f"  {inst.name:<6} shift {shift:+d}: lowest_found={found} "
                  f"({'ok' if ok else 'WRONG'}), singles after shift recall={100 * s['recall']:.0f}%")
    return ok_all


def test_calib_wrong_key(seed):
    rng = np.random.default_rng(seed + 9)
    res = []
    for inst in (Piano(seed), Organ(seed)):
        events = []
        eng = ae.AudioEngine(events.append, None,
                             calib_path=os.path.join(tempfile.gettempdir(), "pt_wrong.json"))
        eng.begin_calibrate_key(60)
        eng.process_buffer(render([(0.3, 64, 0.7, 0.4)], inst, rng, total=1.2), SR)
        wrong = [e for e in events if e["type"] == "calib_done"]
        eng.begin_calibrate_key(60)
        eng.process_buffer(render([(0.3, 60, 0.7, 0.4)], inst, rng, total=1.2), SR)
        right = [e for e in events if e["type"] == "calib_done"][len(wrong):]
        ok = wrong and not wrong[0]["ok"] and right and right[0]["ok"]
        res.append(ok)
        print(f"  {inst.name:<6} calibrate 60, pressed 64 -> ok={wrong[0]['ok'] if wrong else None}; "
              f"pressed 60 -> ok={right[0]['ok'] if right else None}  {'ok' if ok else 'WRONG'}")
    p = os.path.join(tempfile.gettempdir(), "pt_wrong.json")
    if os.path.exists(p):
        os.remove(p)
    return all(res)


def test_live_like_timing(seed):
    """Feed audio in 512-sample blocks like the live callback, with on_level on."""
    rng = np.random.default_rng(seed + 5)
    inst = Piano(seed)
    levels = []
    eng = ae.AudioEngine(lambda e: None, levels.append,
                         calib_path=os.path.join(tempfile.gettempdir(), "pt_none.json"))
    ev, g = seq_chords(rng, 30, sizes=(1, 4))
    audio = render(ev, inst, rng)
    det = ae._Detector(eng.model, SR, lambda e: None, want_level=eng._level_cb)
    eng.on_level = levels.append
    t0 = time.perf_counter()
    worst = 0.0
    for i in range(0, len(audio) - 512, 512):
        a = time.perf_counter()
        det.feed(audio[i:i + 512])
        worst = max(worst, time.perf_counter() - a)
    dt = time.perf_counter() - t0
    hops = len(audio) // 512
    print(f"  live-like feed with on_level: mean {1000 * dt / hops:.3f} ms/hop, "
          f"worst {1000 * worst:.1f} ms, {len(levels)} level callbacks "
          f"({len(levels) / (len(audio) / SR):.1f} Hz)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tmp = os.path.join(tempfile.gettempdir(), f"pt_calib_test_{os.getpid()}.json")
    if os.path.exists(tmp):
        os.remove(tmp)

    t0 = time.time()
    for inst_cls in (Piano, Organ):
        if args.only and args.only != inst_cls.name:
            continue
        inst = inst_cls(args.seed)
        eng = ae.AudioEngine(lambda e: None, None, calib_path=tmp)
        run_suite(eng, inst, rng, args.quick, f"{inst.name}, default templates")
        n_ok = calibrate(eng, inst, rng, range(36, 97))
        print(f"\n  calibrated {n_ok}/61 keys; reloading from {os.path.basename(tmp)}")
        eng2 = ae.AudioEngine(lambda e: None, None, calib_path=tmp)
        assert len(eng2.calibrated_keys) == n_ok, "calibration did not round-trip"
        run_suite(eng2, inst, rng, args.quick, f"{inst.name}, calibrated")
        os.remove(tmp)

    if not args.only:
        print("\n[find_lowest + note_shift]")
        test_lowest_and_shift(args.seed)
        print("\n[calibration sanity]")
        test_calib_wrong_key(args.seed)
        print("\n[timing]")
        test_live_like_timing(args.seed)
    print(f"\ntotal test time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
