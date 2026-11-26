# Piano Typer — shared spec

Turn a cheap 61-key keyboard (BCP brand, **no MIDI out**) into a typing keyboard.
The laptop *listens* to the keyboard's headphone output through a USB sound card
("Microphone (USB PnP Audio Device)", 48 kHz) and works out which keys were pressed.

- **One key = one letter** (fixed sound, fixed letter; the piano makes its own sound).
- **Press all of a word's letters together as a chord = type the whole word.**
  Letter order is lost in a chord, so the word list + a local LLM (Ollama) pick the
  right word from context ("stop" vs "pots" vs "spot"). Repeated letters collapse
  ("letter" = L+E+T+R).
- **On-screen guide:** likely next letters glow on a picture of the keyboard;
  suggested next words are shown with the chord that types them.
- Letters are laid out so common words make pleasant chords and most letters sit on
  white keys (C major), i.e. it mostly stays "in key".

Python: `%USERPROFILE%\.pianotyper\venv\Scripts\python.exe` (numpy, scipy, sounddevice,
aiohttp, wordfreq installed). Do not create other venvs. Windows 11, CPU only
(Intel Core Ultra 7 155U, 16 GB RAM). Ollama running at http://localhost:11434 with
`qwen2.5:7b-instruct` and `phi3.5`.

## Files

| File | Owner | What |
|---|---|---|
| `audio_engine.py` | audio agent | capture + polyphonic note detection + press grouping + calibration |
| `test_audio_engine.py` | audio agent | offline test on synthetic piano audio |
| `web/index.html`, `web/style.css`, `web/app.js` | UI agent | the whole front end (no build step, no frameworks, no CDN needed) |
| `layout.py` → `layout.json` | main | letter layout optimiser |
| `words.py` | main | vocabulary, chord→word index, next-letter probabilities |
| `ai.py` | main | Ollama client |
| `typer.py` | main | types into other Windows apps (SendInput) |
| `server.py` | main | ties it together; aiohttp serves `web/` and a websocket at `/ws` |

## Key numbering

Everything uses **nominal MIDI numbers 36–96** (C2…C7), one per physical key, left to
right. Physical key index = midi − 36. (If the keyboard's real pitch is shifted, the
audio engine applies `note_shift` internally; nobody else cares.)

## layout.json

```json
{
  "version": 1,
  "lowest_midi": 36,
  "highest_midi": 96,
  "keys": {
    "57": {"kind": "letter",  "value": "e"},
    "48": {"kind": "control", "value": "space"},
    "38": {"kind": "char",    "value": "."}
  }
}
```

- `letter`: lowercase a–z. Letters occupy MIDI **57–82** (A3…A#5, 26 keys, chromatic).
- `control` values: `space`, `backspace`, `enter`, `delete_word`, `shift`,
  `suggest1`, `suggest2`, `suggest3`.
- `char`: a literal character to type (punctuation, digits).
- Keys not listed do nothing.

Control area (left hand), fixed:

| MIDI | note | action |
|---|---|---|
| 36 C2 | enter |
| 38 D2 | `.` |
| 40 E2 | `,` |
| 41 F2 | `?` |
| 43 G2 | `!` |
| 45 A2 | `'` |
| 47 B2 | `-` |
| 48 C3, 50 D3, 52 E3 | space |
| 53 F3 | backspace |
| 55 G3 | delete_word |
| 49 C#3 | suggest1 |
| 51 D#3 | suggest2 |
| 54 F#3 | suggest3 |
| 56 G#3 | shift (next letter capital) |

Right side: digits `1`–`9` on 83, 84, 86, 88, 89, 91, 93, 95, 96 and `0` on 94.

## audio_engine.py API

```python
class AudioEngine:
    def __init__(self, on_event, on_level, calib_path="calibration.json"): ...
    @staticmethod
    def list_devices() -> list[dict]   # [{"index": int, "name": str, "samplerate": int}] input devices only,
                                       # prefer WASAPI/MME duplicates removed if easy
    @staticmethod
    def default_device() -> int | None # the USB sound card mic if present
    def start(self, device: int | None = None) -> None   # opens the stream (non-blocking)
    def stop(self) -> None
    @property
    def running(self) -> bool
    def set_sensitivity(self, value: float) -> None   # 0..1, 0.5 default
    def set_note_shift(self, semitones: int) -> None  # real pitch = nominal + shift
    def begin_find_lowest(self) -> None   # next clear single note -> {"type":"lowest_found","midi":<real midi>}
    def begin_calibrate_key(self, nominal_midi: int) -> None
        # records that key's spectrum from the next onset -> {"type":"calib_done","midi":n,"ok":bool}
    def cancel_calibration(self) -> None
    def save_calibration(self) -> None / load happens in __init__
    def process_buffer(self, samples: np.ndarray, sr: int) -> list[dict]  # offline, returns press events
```

Callbacks are called **from the audio thread**; they must be cheap (the server forwards
them into asyncio with `call_soon_threadsafe`).

- `on_event(dict)`:
  - `{"type": "press", "midis": [57, 60, 64], "t": <time.time()>}` — one event per
    *gesture*: a single key, or all keys whose onsets fall within ~70 ms of the first
    (a chord). midis are **nominal**, sorted, deduplicated, restricted to 36–96.
  - `{"type": "lowest_found", "midi": int}`
  - `{"type": "calib_done", "midi": int, "ok": bool}`
  - `{"type": "error", "message": str}`
- `on_level(dict)` ~30 Hz: `{"rms": float 0..1, "active": [nominal midis currently sounding]}`

## Websocket protocol (`/ws`, JSON messages)

### server → client

- `{"type": "init", "layout": <layout.json>, "settings": <settings>, "devices": [...], "state": <state>, "calib": <calib>}`
- `{"type": "state", ...<state>}` whenever text/suggestions change
- `{"type": "press", "midis": [...], "source": "audio" | "sim"}` (for key flash/trail animation)
- `{"type": "level", "rms": float, "active": [...]}` ~30 Hz
- `{"type": "settings", "settings": <settings>}`
- `{"type": "calib", ...<calib>}`
- `{"type": "toast", "message": str}`

`<state>`:
```json
{
  "text": "Hello there",
  "prefix": "the",
  "suggestions": [{"word": "there", "midis": [57, 60, 70], "key": 49}],
  "next_letters": {"r": 0.41, "y": 0.22, "m": 0.10},
  "last": {"kind": "word", "value": "stop", "midis": [57, 63, 66, 71],
           "alternatives": ["pots", "spot", "tops"], "ai_corrected": false},
  "ai": {"status": "ready", "model": "phi3.5", "latency_ms": 900},
  "shift": false
}
```
- `last.kind` ∈ `letter | word | control | char | unknown`; `last` may be null.
- `suggestions`: up to 3; `key` is the suggestN control key that accepts it (49, 51, 54).
- `next_letters`: probabilities for the next letter (only top ~8 sent).
- `ai.status` ∈ `off | ready | thinking | error | unavailable`.

`<settings>`:
```json
{"device": 29, "input_mode": "piano", "type_external": false, "ai_enabled": true,
 "model": "phi3.5", "models": ["phi3.5", "qwen2.5:7b-instruct"],
 "sensitivity": 0.5, "note_shift": 0, "chord_window_ms": 70, "audio_running": true}
```
`input_mode`: `piano` (listen to the keyboard) or `sim` (on-screen / laptop keys only,
browser plays a synth sound).

`<calib>`: `{"phase": "idle" | "lowest" | "keys", "target": 57 | null, "done": 3, "total": 61, "message": "Press the key marked E"}`

### client → server

- `{"type": "sim_press", "midis": [...]}` — simulated gesture (single key or chord)
- `{"type": "set", "key": "<settings key>", "value": ...}`
- `{"type": "pick_suggestion", "index": 0}`
- `{"type": "clear"}`
- `{"type": "calib_start", "mode": "lowest" | "keys"}`, `{"type": "calib_skip"}`, `{"type": "calib_cancel"}`
