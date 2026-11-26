"""
Step 1: live note detector for the piano-typer project.

Listens to the keyboard through the USB sound card and prints which note
you're playing, how confident the detector is, and how long it took from
the moment the sound started to the note being recognised.

Usage:
    python note_detector.py              # auto-picks the USB sound card
    python note_detector.py --list       # list input devices
    python note_detector.py --device 3   # pick a device by number
    python note_detector.py --selftest   # check the detector on fake tones
"""

import argparse
import queue
import sys
import time

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Detector settings
FMIN = 60.0          # lowest pitch we look for (Hz) - a bit below C2
FMAX = 2200.0        # highest pitch (Hz) - a bit above C7
WINDOW = 2048        # samples analysed per frame (~43 ms at 48 kHz)
HOP = 512            # new samples between frames (~11 ms at 48 kHz)
YIN_THRESHOLD = 0.15 # lower = stricter pitch detection
SILENCE_RMS = 0.01   # below this level we treat the input as silent
STABLE_FRAMES = 2    # same note this many frames in a row = a real note


def freq_to_note(freq):
    """Return (note name like 'C4', midi number, cents off) for a frequency."""
    midi_float = 69 + 12 * np.log2(freq / 440.0)
    midi = int(round(midi_float))
    cents = int(round((midi_float - midi) * 100))
    name = f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"
    return name, midi, cents


def yin_pitch(frame, sr):
    """
    YIN pitch detector (de Cheveigne & Kawahara, 2002).
    Returns (frequency in Hz, confidence 0-1), or (None, 0) if no pitch.
    """
    x = frame.astype(np.float64)
    x = x - x.mean()
    w = len(x) // 2
    tau_min = max(2, int(sr / FMAX))
    tau_max = min(int(sr / FMIN), w)

    # Difference function d(tau) = sum (x[j] - x[j+tau])^2, computed with FFTs
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

    # Cumulative mean normalised difference
    cmnd = np.ones_like(d)
    running = np.cumsum(d[1:])
    running[running == 0] = 1e-12
    cmnd[1:] = d[1:] * np.arange(1, tau_max + 1) / running

    # First dip below the threshold, then walk down to its local minimum
    below = np.where(cmnd[tau_min:] < YIN_THRESHOLD)[0]
    if len(below) == 0:
        return None, 0.0
    tau = below[0] + tau_min
    while tau + 1 <= tau_max and cmnd[tau + 1] < cmnd[tau]:
        tau += 1

    # Parabolic interpolation for sub-sample accuracy
    if 1 <= tau < tau_max:
        s0, s1, s2 = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        denom = s0 - 2 * s1 + s2
        shift = 0.5 * (s0 - s2) / denom if denom != 0 else 0.0
    else:
        shift = 0.0

    freq = sr / (tau + shift)
    confidence = float(max(0.0, 1.0 - cmnd[tau]))
    return freq, confidence


def find_usb_input(sd):
    """Pick the USB sound card's microphone if there is one."""
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and "usb" in dev["name"].lower():
            return i
    return None


def meter(rms, width=30):
    level = min(1.0, rms / 0.3)
    filled = int(level * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def run_live(device):
    import sounddevice as sd

    if device is None:
        device = find_usb_input(sd)
        if device is None:
            print("Couldn't find a USB input. Run with --list and pick one with --device N.")
            sys.exit(1)

    info = sd.query_devices(device)
    sr = int(info["default_samplerate"])
    print(f"Listening on: [{device}] {info['name']}  ({sr} Hz)")
    print("Play single notes on the keyboard. Ctrl+C to stop.\n")

    q = queue.Queue()

    def callback(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    buf = np.zeros(WINDOW, dtype=np.float32)
    pending = np.zeros(0, dtype=np.float32)

    candidate = None      # note we're currently seeing
    candidate_count = 0
    current = None        # note we've announced
    onset_time = None     # when the sound got loud
    was_silent = True

    with sd.InputStream(device=device, channels=1, samplerate=sr,
                        blocksize=HOP, callback=callback):
        try:
            while True:
                pending = np.concatenate((pending, q.get()))
                while len(pending) >= HOP:
                    buf = np.concatenate((buf[HOP:], pending[:HOP]))
                    pending = pending[HOP:]

                    rms = float(np.sqrt(np.mean(buf[-HOP:] ** 2)))
                    silent = rms < SILENCE_RMS

                    if silent:
                        if not was_silent:
                            current, candidate, candidate_count = None, None, 0
                        was_silent = True
                        sys.stdout.write(f"\r{meter(rms)}  (silence)              ")
                        sys.stdout.flush()
                        continue

                    if was_silent:
                        onset_time = time.perf_counter()
                        was_silent = False

                    freq, conf = yin_pitch(buf, sr)
                    if freq is None:
                        sys.stdout.write(f"\r{meter(rms)}  (no clear pitch)       ")
                        sys.stdout.flush()
                        continue

                    name, midi, cents = freq_to_note(freq)
                    if midi == candidate:
                        candidate_count += 1
                    else:
                        candidate, candidate_count = midi, 1

                    if candidate_count >= STABLE_FRAMES and midi != current:
                        current = midi
                        lag_ms = (time.perf_counter() - onset_time) * 1000 if onset_time else 0
                        onset_time = None
                        sys.stdout.write("\r" + " " * 60 + "\r")
                        print(f"  {name:<4}  {freq:7.1f} Hz  {cents:+3d} cents  "
                              f"conf {conf:.2f}  ~{lag_ms:4.0f} ms after sound started")

                    sys.stdout.write(f"\r{meter(rms)}  hearing {name} ({conf:.2f})     ")
                    sys.stdout.flush()
        except KeyboardInterrupt:
            print("\nStopped.")


def selftest():
    """Feed the detector synthetic organ-like tones and check it names them right."""
    sr = 48000
    t = np.arange(WINDOW) / sr
    ok = True
    for midi in [36, 48, 57, 60, 64, 67, 72, 84, 96]:
        f = 440.0 * 2 ** ((midi - 69) / 12)
        tone = sum((0.6 / k) * np.sin(2 * np.pi * f * k * t) for k in range(1, 6))
        tone += 0.01 * np.random.randn(len(t))
        freq, conf = yin_pitch(tone, sr)
        expected = freq_to_note(f)[0]
        got = freq_to_note(freq)[0] if freq else "none"
        status = "ok" if got == expected else "WRONG"
        ok &= status == "ok"
        print(f"  {expected:<4} -> {got:<4} conf {conf:.2f}  {status}")
    print("Self-test passed." if ok else "Self-test FAILED.")
    return ok


def main():
    p = argparse.ArgumentParser(description="Live note detector for the piano keyboard.")
    p.add_argument("--list", action="store_true", help="list audio input devices")
    p.add_argument("--device", type=int, help="input device number")
    p.add_argument("--selftest", action="store_true", help="test detector on fake tones")
    args = p.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.list:
        import sounddevice as sd
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                print(f"  [{i}] {dev['name']}  ({int(dev['default_samplerate'])} Hz)")
        return
    run_live(args.device)


if __name__ == "__main__":
    main()
