"""
End to end without the real keyboard: synthesise piano audio of someone typing a
phrase (letters one by one + word chords + space keys), run it through the audio
engine, feed the detected presses into the server's typing logic, compare the text.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import numpy as np

import audio_engine
import server
from test_audio_engine import SR, Piano, calibrate, render

PHRASE = [("letters", "hi"), ("space",), ("chord", "there"), ("chord", "how"), ("chord", "are"),
          ("chord", "you"), ("char", "?"), ("letters", "i"), ("space",), ("chord", "love"),
          ("chord", "this"), ("chord", "piano")]


def main(calibrated=True):
    app = server.App()
    app.settings.update(ai_enabled=False, type_external=False)
    app.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(app.loop)
    space = next(m for m, e in app.keys.items() if e["kind"] == "control" and e["value"] == "space")
    qmark = next(m for m, e in app.keys.items() if e["kind"] == "char" and e["value"] == "?")

    rng = np.random.default_rng(7)
    inst = Piano(seed=3)
    events, t = [], 0.5
    for step in PHRASE:
        if step[0] == "letters":
            for c in step[1]:
                events.append((t, app.letter_midi[c], rng.uniform(0.5, 0.9), 0.15))
                t += 0.22
        elif step[0] == "space":
            events.append((t, space, 0.7, 0.15)); t += 0.3
        elif step[0] == "char":
            events.append((t, qmark, 0.7, 0.15)); t += 0.35
        else:
            notes = app.chord_for(step[1])
            offs = rng.uniform(0, 0.025, len(notes))
            for m, o in zip(notes, offs - offs.min()):
                events.append((t + o, m, rng.uniform(0.55, 0.9), 0.25))
            t += 0.6
    audio = render(events, inst, rng)

    calib = Path(tempfile.mkdtemp()) / "calibration.json"
    eng = audio_engine.AudioEngine(lambda e: None, lambda lv: None, calib_path=str(calib))
    if calibrated:
        calibrate(eng, inst, np.random.default_rng(1), list(range(36, 97)))
    presses = [e for e in eng.process_buffer(audio, SR) if e.get("type") == "press"]
    for e in presses:
        app.handle_press(e["midis"], "audio")

    want = "Hi there how are you? I love this piano "
    print(f"{'calibrated' if calibrated else 'uncalibrated'}: {len(presses)} presses detected")
    print("  typed :", repr(app.text))
    print("  wanted:", repr(want))
    ok = app.text == want
    print("  ", "MATCH" if ok else "DIFFERENT")
    return ok


if __name__ == "__main__":
    a = main(True)
    b = main(False)
    sys.exit(0 if a else 1)
