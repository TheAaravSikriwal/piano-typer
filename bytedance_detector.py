"""
Optional note detector for Piano Typer using ByteDance's high-resolution piano
transcription model (Kong et al., 2020).

  Code:  piano_transcription_inference (MIT, Qiuqiang Kong)
         https://github.com/qiuqiangkong/piano_transcription_inference
         https://github.com/bytedance/piano_transcription (Apache-2.0)
  Model: CRNN_note_F1=0.9677_pedal_F1=0.9186.pth, Zenodo record 4034264,
         licensed CC BY 4.0 (attribution: Qiuqiang Kong et al.,
         "High-resolution Piano Transcription with Pedals by Regressing
         Onsets and Offsets Times", 2020).

This file needs torch + piano_transcription_inference, which live in a separate
venv (%USERPROFILE%\\.pianotyper\\venv-bt) so the app venv stays light. Two ways
to use it:

  1. In-process (only from venv-bt):
        det = BytedanceDetector()
        det.notes_started(audio_48k, onset_s=0.2)     -> [60, 64, 67]

  2. From the app venv (numpy only) via a worker subprocess speaking a binary
     protocol on stdin/stdout:
        cli = BytedanceWorkerClient()                 # spawns venv-bt python
        cli.notes_started(audio_48k, onset_s=0.2)     -> [60, 64, 67]
        cli.transcribe_segment(audio_48k)             -> [(midi, t, vel), ...]
        cli.close()

     Worker:  venv-bt\\Scripts\\python.exe bytedance_detector.py --serve
     Request:  b"BTDR" | uint32 n_samples | float32 onset_s | float32 window_s
               | n_samples * float32 (48 kHz mono)
               onset_s < 0 means "transcribe the whole segment".
     Reply:    uint32 length | UTF-8 JSON
               {"ok": true, "notes": [[midi, onset_s, velocity], ...],
                "started": [midi, ...] (only if onset_s >= 0), "ms": float}
     The worker writes b"READY\\n" to stdout once the model is loaded.

Other CLI:  python bytedance_detector.py --file keys.npy     (full transcription)
"""

import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

APP_SR = 48000
MODEL_SR = 16000
FPS = 100                       # model frame rate
BEGIN_NOTE = 21                 # model class 0 = A0
HOME = os.path.join(os.path.expanduser("~"), ".pianotyper")
DEFAULT_CKPT = os.path.join(HOME, "models", "note_F1=0.9677_pedal_F1=0.9186.pth")
WORKER_PY = os.path.join(HOME, "venv-bt", "Scripts", "python.exe")
MAGIC = b"BTDR"


def _resample_48k_to_16k(x):
    from scipy.signal import resample_poly
    return resample_poly(np.asarray(x, dtype=np.float64), 1, 3).astype(np.float32)


def decode_onsets(reg_onset, velocity, threshold=0.3, neighbour=2, frame=None,
                  min_frame=0.0, frame_look=10):
    """Peak-pick the regressed onset curves (same rule as the package's
    RegressionPostProcessor: value > threshold and monotonic over `neighbour`
    frames on each side, plus the paper's sub-frame shift).

    reg_onset, velocity: (frames, 88).  Returns list of (midi, time_s, vel 0-127).
    Unlike the package's note detector this does not need an offset, so notes
    still sounding at the end of a short segment are kept.
    """
    x = reg_onset
    n_frames = x.shape[0]
    out = []
    if n_frames < 2 * neighbour + 1:
        return out
    # loop over candidate frames only (few per segment: cheap)
    cand = np.argwhere(x > threshold)
    for n, k in cand:
        if n < neighbour or n >= n_frames - neighbour:
            continue
        mono = True
        for i in range(neighbour):
            if x[n - i, k] < x[n - i - 1, k] or x[n + i, k] < x[n + i + 1, k]:
                mono = False
                break
        if not mono:
            continue
        if min_frame > 0 and frame is not None and \
                frame[n:n + frame_look, k].max() < min_frame:
            continue        # onset spike without a sustained note: usually an overtone ghost
        a, b, c = x[n - 1, k], x[n, k], x[n + 1, k]
        if a > c:
            shift = (c - a) / max(b - c, 1e-9) / 2
        else:
            shift = (c - a) / max(b - a, 1e-9) / 2
        out.append((int(k) + BEGIN_NOTE, float((n + shift) / FPS),
                    int(round(float(velocity[n, k]) * 128))))
    out.sort(key=lambda e: (e[1], e[0]))
    return out


class BytedanceDetector:
    """Loads the note model once (pedal sub-model skipped) and runs it on CPU."""

    def __init__(self, checkpoint=DEFAULT_CKPT, threads=2, onset_threshold=0.3,
                 pad_to=None, pre_s=0.2, quantize=False, min_frame=0.5, gain=1.0):
        """threads: torch CPU threads (2 was fastest on this laptop; the GRUs
             are latency-bound and more threads made it slower and jittery).
           min_frame: keep an onset only if the frame head reaches this within
             100 ms (0 = package behaviour; 0.5 removed most overtone ghosts).
           gain: input scale before the model (0.3 helped the hot, near-clipping
             real recording; 1.0 is neutral).
           quantize: int8 dynamic quantisation of GRU/Linear (similar speed here)."""
        import torch
        from piano_transcription_inference.models import \
            Regress_onset_offset_frame_velocity_CRNN
        self.torch = torch
        if threads:
            torch.set_num_threads(int(threads))
        t0 = time.perf_counter()
        model = Regress_onset_offset_frame_velocity_CRNN(frames_per_second=FPS,
                                                         classes_num=88)
        try:
            ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except Exception:                       # older pickles with extra objects
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        if "note_model" in sd and isinstance(sd["note_model"], dict):
            note_sd = sd["note_model"]          # Note_pedal checkpoint layout
        else:
            note_sd = {k[len("note_model."):]: v for k, v in sd.items()
                       if k.startswith("note_model.")} or sd
        missing, unexpected = model.load_state_dict(note_sd, strict=False)
        real_missing = [m for m in missing if "spectrogram_extractor" not in m
                        and "logmel_extractor" not in m]
        if real_missing:
            raise RuntimeError(f"checkpoint missing weights: {real_missing[:5]}")
        model.eval()
        if quantize:
            # int8 dynamic quantisation of the GRU/Linear layers (the CPU hot spot)
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.GRU, torch.nn.Linear}, dtype=torch.qint8)
        self.model = model
        self.onset_threshold = onset_threshold
        self.min_frame = min_frame      # >0: also require frame activation (ghost filter)
        self.gain = gain
        self.pad_to = pad_to            # seconds of zero padding (None = run as-is)
        self.pre_s = pre_s              # audio kept before onset in notes_started
        self.load_time = time.perf_counter() - t0
        self.last_ms = 0.0

    # ------------------------------------------------------------ core
    def _forward16k(self, x16):
        torch = self.torch
        n = len(x16)
        if self.pad_to and n < int(self.pad_to * MODEL_SR):
            x16 = np.concatenate([x16, np.zeros(int(self.pad_to * MODEL_SR) - n,
                                                np.float32)])
        with torch.inference_mode():
            out = self.model(torch.from_numpy(np.ascontiguousarray(x16))[None])
        frames = n // (MODEL_SR // FPS) + 1
        return {k: v[0, :frames].numpy() for k, v in out.items()}

    def transcribe_segment(self, audio_48k):
        """audio_48k: float mono at 48 kHz. Returns [(midi, onset_s, velocity)]
        with onset times relative to the start of the segment."""
        t0 = time.perf_counter()
        x16 = _resample_48k_to_16k(audio_48k)
        if self.gain != 1.0:
            x16 = x16 * np.float32(self.gain)
        od = self._forward16k(x16)
        notes = decode_onsets(od["reg_onset_output"], od["velocity_output"],
                              threshold=self.onset_threshold,
                              frame=od["frame_output"], min_frame=self.min_frame)
        self.last_ms = 1000 * (time.perf_counter() - t0)
        return notes

    def notes_started(self, audio, onset_s, window_s=0.4, tol_s=0.06):
        """audio: 48 kHz buffer, onset_s: time of the press within it.
        Runs the model on [onset_s - pre_s, onset_s + window_s] and returns the
        sorted MIDI notes whose onset is within +-tol_s of onset_s."""
        i0 = max(0, int(round((onset_s - self.pre_s) * APP_SR)))
        i1 = min(len(audio), int(round((onset_s + window_s) * APP_SR)))
        seg = audio[i0:i1]
        rel = onset_s - i0 / APP_SR
        notes = self.transcribe_segment(seg)
        return sorted({m for m, t, _ in notes if abs(t - rel) <= tol_s})


# ------------------------------------------------------------------ worker
def _read_exact(f, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def serve(threads=2, gain=1.0):
    """stdin/stdout worker loop (see module docstring for the protocol)."""
    inp = sys.stdin.buffer
    outp = sys.stdout.buffer
    sys.stdout = sys.stderr             # stray library prints must not corrupt the protocol
    det =BytedanceDetector(threads=threads, gain=gain)
    det.transcribe_segment(np.zeros(APP_SR // 2, np.float32))     # warm-up
    outp.write(b"READY\n")
    outp.flush()
    while True:
        hdr = _read_exact(inp, 16)
        if hdr is None:
            return
        magic, n, onset_s, window_s = struct.unpack("<4sIff", hdr)
        if magic != MAGIC:
            sys.stderr.write("bad magic, exiting\n")
            return
        raw = _read_exact(inp, 4 * n)
        if raw is None:
            return
        audio = np.frombuffer(raw, dtype="<f4")
        t0 = time.perf_counter()
        try:
            if onset_s >= 0:
                started = det.notes_started(audio, onset_s, window_s)
                rep = {"ok": True, "started": started}
            else:
                rep = {"ok": True, "notes": det.transcribe_segment(audio)}
        except Exception as e:                 # keep the worker alive
            rep = {"ok": False, "error": repr(e)}
        rep["ms"] = 1000 * (time.perf_counter() - t0)
        body = json.dumps(rep).encode()
        outp.write(struct.pack("<I", len(body)) + body)
        outp.flush()


class BytedanceWorkerClient:
    """Numpy-only client usable from the app venv. Spawns the worker once."""

    def __init__(self, python=WORKER_PY, threads=2, gain=1.0):
        args = [python, os.path.abspath(__file__), "--serve",
                "--threads", str(threads), "--gain", str(gain)]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  creationflags=flags)
        line = self.p.stdout.readline()
        if line.strip() != b"READY":
            raise RuntimeError(f"worker failed to start: {line!r}")

    def _call(self, audio, onset_s, window_s):
        a = np.ascontiguousarray(audio, dtype="<f4")
        self.p.stdin.write(struct.pack("<4sIff", MAGIC, len(a), onset_s, window_s)
                           + a.tobytes())
        self.p.stdin.flush()
        (n,) = struct.unpack("<I", _read_exact(self.p.stdout, 4))
        rep = json.loads(_read_exact(self.p.stdout, n))
        if not rep.get("ok"):
            raise RuntimeError(rep.get("error"))
        self.last_ms = rep["ms"]
        return rep

    def transcribe_segment(self, audio_48k):
        return [tuple(e) for e in self._call(audio_48k, -1.0, 0.0)["notes"]]

    def notes_started(self, audio, onset_s, window_s=0.4):
        return self._call(audio, float(onset_s), float(window_s))["started"]

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(5)
        except Exception:
            self.p.kill()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--file", help=".npy float32 48 kHz mono to transcribe")
    a = ap.parse_args()
    if a.serve:
        serve(a.threads, a.gain)
    elif a.file:
        det = BytedanceDetector(threads=a.threads, gain=a.gain)
        audio = np.load(a.file).astype(np.float32)
        for m, t, v in det.transcribe_segment(audio):
            print(f"{t:8.3f}s  midi {m:3d}  vel {v:3d}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
