"""
Piano Typer server.

    python server.py            -> opens http://localhost:8765 in your browser
    python server.py --no-browser --port 8765

Listens to the keyboard (audio_engine), turns key presses into letters / words
(words + ai), keeps the text, optionally types it into other apps (typer), and
talks to the web UI over a websocket (see SPEC.md).
"""

import argparse
import asyncio
import json
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aiohttp import WSMsgType, web

import ai as ai_mod
import typer
from words import Vocab

HERE = Path(__file__).parent
WEB = HERE / "web"
SETTINGS_PATH = HERE / "settings.json"
SUGGEST_KEYS = {"suggest1": 0, "suggest2": 1, "suggest3": 2}
# keys that act on a pending chord word instead of accepting it first
PENDING_KEYS = {"complete", "backspace", "undo", "suggest1", "suggest2", "suggest3", "shift"}
SENTENCE_END = ".?!\n"

def layout_note(m):
    return ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][m % 12] + str(m // 12 - 1)


DEFAULT_SETTINGS = {
    "device": None, "input_mode": "piano", "type_external": False, "ai_enabled": True,
    "model": None, "sensitivity": 0.5, "note_shift": 0, "chord_window_ms": 70,
    "suggestions": True,   # off = no word chips, no next-letter glow; AI only picks chord words
}


class App:
    def __init__(self):
        self.loop = None
        self.clients = set()
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.layout = json.loads((HERE / "layout.json").read_text())
        self.keys = {int(k): v for k, v in self.layout["keys"].items()}
        self.letter_midi = {v["value"]: k for k, v in self.keys.items() if v["kind"] == "letter"}
        self.suggest_midi = {v["value"]: k for k, v in self.keys.items()
                             if v["kind"] == "control" and v["value"] in SUGGEST_KEYS}
        print("Loading vocabulary...")
        self.vocab = Vocab()

        self.settings = dict(DEFAULT_SETTINGS)
        try:
            self.settings.update(json.loads(SETTINGS_PATH.read_text()))
        except Exception:
            pass

        # text state
        self.text = ""
        self.shift = False
        self.last = None
        self.ai_words = []        # AI's next-word ideas for the current position
        self.ai_words_for = None  # (text) they were computed for
        self.suggestions = []
        self.version = 0          # bumps on every text change; stale AI answers are dropped
        self.pending_choice = None
        self.pending = None       # chord word waiting to be accepted (see do_complete)
        self.history = []         # (text, pending, shift) before each press, for undo
        self.ghost_ai = []        # AI's words for the half-typed word (grey preview)

        # AI
        self.models = ai_mod.list_models()
        if self.settings["model"] not in self.models:
            self.settings["model"] = ai_mod.pick_default(self.models)
        self.ai = ai_mod.AI(self.settings["model"], self.vocab) if self.settings["model"] else None
        self.ai_status = "unavailable" if not self.ai else ("ready" if self.settings["ai_enabled"] else "off")
        self.ai_busy = 0
        self.ai_pending = {}

        # audio
        self.engine = None
        self.devices = []
        self.calib = {"phase": "idle", "target": None, "done": 0, "total": 0, "message": ""}
        self.calib_queue = []
        self.last_level_sent = 0.0

    # ------------------------------------------------------------ helpers ---

    def save_settings(self):
        try:
            SETTINGS_PATH.write_text(json.dumps(self.settings, indent=1))
        except Exception as e:
            print("could not save settings:", e)

    def prefix(self):
        i = len(self.text)
        while i > 0 and self.text[i - 1].isalpha():
            i -= 1
        return self.text[i:].lower()

    def at_sentence_start(self):
        t = self.text.rstrip(" ")
        return t == "" or t[-1] in SENTENCE_END

    def chord_for(self, word):
        return sorted({self.letter_midi[c] for c in word if c in self.letter_midi})

    def state(self):
        p = self.prefix()
        nl = dict(self.vocab.next_letters(p))
        if not p and self.ai_words:
            mix = {}
            for i, w in enumerate(self.ai_words):
                mix[w[0]] = mix.get(w[0], 0) + 0.6 / (i + 1.5)
            tot = sum(mix.values())
            nl = {k: 0.45 * nl.get(k, 0) + 0.55 * mix.get(k, 0) / tot for k in set(nl) | set(mix)}
        nl.pop(" ", None)
        if not self.settings.get("suggestions", True):
            nl = {}
        top = dict(sorted(nl.items(), key=lambda kv: -kv[1])[:8])
        return {
            "type": "state",
            "text": self.text,
            "prefix": p,
            "suggestions": self.suggestions,
            "next_letters": {k: round(v, 3) for k, v in top.items()},
            "last": self.last,
            "ghost": self._ghost_state(),
            "pending": ({"start": self.pending["start"], "len": len(self.pending["word"]),
                         "ai": self.pending["ai"], "unsure": self.pending["unsure"]}
                        if self.pending else None),
            "ai": {"status": self.ai_status if self.settings["ai_enabled"] or not self.ai else "off",
                   "model": self.settings["model"], "latency_ms": self.ai.last_latency_ms if self.ai else None},
            "shift": self.shift,
        }

    def _ghost_state(self):
        w = self.ghost_word()
        if not w:
            return None
        p = self.prefix()
        return {"rest": w[len(p):], "word": w,
                "ai": any(x == w for x in self.ghost_ai)}

    def settings_msg(self):
        s = dict(self.settings)
        s["models"] = self.models
        s["audio_running"] = bool(self.engine and self.engine.running)
        s["devices"] = self.devices
        return {"type": "settings", "settings": s}

    def refresh_devices(self):
        """Re-scan inputs (e.g. the USB sound card was plugged in after starting)."""
        try:
            import audio_engine
            if not (self.engine and self.engine.running):
                audio_engine.AudioEngine.rescan_devices()
            self.devices = audio_engine.AudioEngine.list_devices()
        except Exception as e:
            print("device scan failed:", e)

    async def broadcast(self, msg):
        data = json.dumps(msg)
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except Exception:
                self.clients.discard(ws)

    def broadcast_soon(self, msg):
        asyncio.ensure_future(self.broadcast(msg))

    # ---------------------------------------------------------- text edits ---

    def set_text(self, new):
        old = self.text
        self.text = new
        self.version += 1
        if self.settings["type_external"] and old != new:
            try:
                typer.apply_diff(old, new)
            except Exception as e:
                print("typing failed:", e)

    def _sentence_start_of(self, s):
        s = s.rstrip(" ")
        return s == "" or s[-1] in SENTENCE_END

    def apply_word(self, word, replace_prefix=False):
        """Insert a whole word (from a chord or a suggestion), with spacing and capitals."""
        t = self.text
        if replace_prefix:
            p = self.prefix()
            base = t[: len(t) - len(p)]
        else:
            base = t + (" " if t and t[-1].isalpha() else "")
        cap = self.shift or self._sentence_start_of(base) or (replace_prefix and self.prefix()[:1].isupper())
        w = word[0].upper() + word[1:] if cap else word
        if word == "i":
            w = "I"
        self.set_text(base + w + " ")
        self.shift = False
        return len(base), w

    def do_letter(self, ch):
        if self.shift or (self.at_sentence_start() and not self.prefix()):
            ch = ch.upper()
        self.set_text(self.text + ch)
        self.shift = False

    def do_char(self, ch):
        t = self.text
        if ch in ".,?!" and t.endswith(" "):
            t = t.rstrip(" ")
        self.set_text(t + ch + (" " if ch in ".,?!" else ""))

    # ---- pending word: a chord's word waits (underlined) until you accept it ----

    def _set_pending_word(self, w):
        """Swap the pending word in place for w (keeps its spacing)."""
        p = self.pending
        cap = p["word"][:1].isupper()
        w = "I" if w == "i" else (w[0].upper() + w[1:] if cap else w)
        s = p["start"]
        self.set_text(self.text[:s] + w + self.text[s + len(p["word"]):])
        p["word"] = w
        return w

    def commit_pending(self):
        self.pending = None

    def do_complete(self):
        """One key for 'finish': accept the pending word, else complete the
        half-typed word, else a space."""
        if self.pending:
            self.commit_pending()
            return
        p = self.prefix()
        if p:
            guess = self.ghost_word()
            if guess:
                _, w = self.apply_word(guess, replace_prefix=True)
                self.last = {"kind": "word", "value": w, "midis": self.chord_for(guess),
                             "alternatives": [], "ai_corrected": False, "via": "complete"}
                return
        self.set_text(self.text + " ")

    def ghost_word(self):
        """The word the half-typed letters are probably heading for, shown greyed
        after the text and typed only when you press ✓ Complete. The AI's guess for
        this spot comes first, the most common word second."""
        p = self.prefix()
        if not p or self.pending:
            return None
        # "hi" is already a real word: don't offer "his". Rare junk like "hel"
        # (in the word list but almost never used) still counts as half a word.
        if self.vocab.freq.get(p, 0.0) >= 1.8e-5:
            return None
        for w in self.ghost_ai:
            if w.startswith(p) and w != p and w in self.vocab:
                return w
        return next((w for w in self.vocab.completions(p, 3) if w != p), None)

    def do_double(self):
        t = self.text
        if t and t[-1].isalpha():
            self.set_text(t + t[-1])

    def do_undo(self):
        if not self.history:
            return
        text, pending, shift = self.history.pop()
        self.set_text(text)
        self.pending = dict(pending) if pending else None
        self.shift = shift

    def do_control(self, value):
        t = self.text
        if value == "space":
            self.set_text(t + " ")
        elif value == "complete":
            self.do_complete()
        elif value == "double":
            self.do_double()
        elif value == "undo":
            self.do_undo()
        elif value == "enter":
            self.set_text(t.rstrip(" ") + "\n")
        elif value == "backspace":
            if self.pending:            # reject the pending word
                before = self.pending["before"]
                self.pending = None
                self.set_text(before)
            else:
                self.set_text(t[:-1])
        elif value == "delete_word":
            s = t.rstrip(" ")
            i = len(s)
            while i > 0 and not s[i - 1].isspace():
                i -= 1
            self.set_text(s[:i])
        elif value == "shift":
            self.shift = not self.shift
        elif value in SUGGEST_KEYS:
            return self.pick_suggestion(SUGGEST_KEYS[value])
        return True

    def pick_suggestion(self, index):
        if index >= len(self.suggestions):
            return False
        if self.pending:                # choose another option for the pending word
            w = self._set_pending_word(self.suggestions[index]["word"])
            self.last = {**(self.last or {}), "kind": "word", "value": w, "ai_corrected": False,
                         "via": "option"}
            self.commit_pending()
            return True
        word = self.suggestions[index]["word"]
        start, w = self.apply_word(word, replace_prefix=bool(self.prefix()))
        self.last = {"kind": "word", "value": w, "midis": self.chord_for(word), "alternatives": [],
                     "ai_corrected": False, "via": "suggestion"}
        return True

    # ---------------------------------------------------------- key presses ---

    def handle_press(self, midis, source):
        if self.calib["phase"] != "idle" and source == "audio":
            return
        midis = sorted(set(m for m in midis if m in self.keys))
        if source == "audio":
            heard = ", ".join(f"{layout_note(m)}={self.keys[m]['value']}" for m in midis)
            print(f"heard: {heard}", flush=True)
        self.broadcast_soon({"type": "press", "midis": midis, "source": source})
        if not midis:
            return
        entries = [self.keys[m] for m in midis]
        letters = sorted({e["value"] for e in entries if e["kind"] == "letter"})
        single_ctl = entries[0]["value"] if len(entries) == 1 and entries[0]["kind"] == "control" else None
        if single_ctl != "undo":
            self.history.append((self.text, dict(self.pending) if self.pending else None, self.shift))
            del self.history[:-50]
        # playing anything else accepts the pending word as shown
        if self.pending and single_ctl not in PENDING_KEYS:
            self.commit_pending()

        if len(midis) == 1 or len(letters) <= 1 and len(entries) == 1:
            e = entries[0]
            if e["kind"] == "letter":
                self.do_letter(e["value"])
                self.last = {"kind": "letter", "value": e["value"], "midis": midis}
            elif e["kind"] == "char":
                self.do_char(e["value"])
                self.last = {"kind": "char", "value": e["value"], "midis": midis}
            else:
                self.do_control(e["value"])
                if e["value"] not in SUGGEST_KEYS:
                    self.last = {"kind": "control", "value": e["value"], "midis": midis}
        elif len(letters) == 1:
            # a chord whose only letter key is one letter (plus stray controls): treat as that letter
            self.do_letter(letters[0])
            self.last = {"kind": "letter", "value": letters[0], "midis": midis}
        else:
            self.chord_word(letters, midis)
        self.after_change()

    def chord_word(self, letters, midis):
        cands = self.vocab.candidates(letters)
        if not cands:
            self.last = {"kind": "unknown", "value": "".join(letters), "midis": midis}
            self.broadcast_soon({"type": "toast", "message": f"No word uses exactly the letters {' '.join(letters).upper()}"})
            return
        words = [w for w, _ in cands]
        # context-aware pick without waiting: prefer a word the AI already predicted here
        best = words[0]
        for w in self.ai_words:
            if w in words:
                best = w
                break
        context_before = self.text
        start, inserted = self.apply_word(best)
        self.pending = {"before": context_before, "start": start, "word": inserted,
                        "cands": words[:6], "ai": False, "unsure": len(words) > 1}
        self.last = {"kind": "word", "value": inserted, "midis": midis,
                     "alternatives": [w for w in words if w != best][:5], "ai_corrected": False}
        if len(words) > 1 and self.ai_enabled():
            top = words[:6]
            v = self.version
            self.run_ai(self._choose_job, context_before, top, v, start, inserted, kind="choose")

    # ------------------------------------------------------------------ AI ---

    def ai_enabled(self):
        return bool(self.ai and self.settings["ai_enabled"])

    def run_ai(self, fn, *args, kind="other"):
        """One model call at a time; while busy, keep only the newest request of each
        kind (typing fast must not build a backlog). Word choices go first."""
        self.ai_pending[kind] = (fn, args)
        self._ai_pump()

    def _ai_pump(self):
        if self.ai_busy or not self.ai_pending:
            return
        kind = "choose" if "choose" in self.ai_pending else next(iter(self.ai_pending))
        fn, args = self.ai_pending.pop(kind)
        self.ai_busy = 1
        self.ai_status = "thinking"
        fut = self.loop.run_in_executor(self.pool, fn, *args)

        def done(f):
            self.ai_busy = 0
            try:
                result = f.result()
                self.ai_status = "ready"
                if result:
                    result()
            except Exception as e:
                print("AI error:", e)
                self.ai_status = "error"
            self.broadcast_soon(self.state())
            self._ai_pump()

        fut.add_done_callback(lambda f: self.loop.call_soon_threadsafe(done, f))

    def _choose_job(self, context, cands, version, start, inserted):
        choice = self.ai.choose(context, cands)

        def apply():
            p = self.pending
            # only while the word is still waiting: once accepted, the AI never changes it
            if not p or p["start"] != start or p["word"] != inserted:
                return
            p["ai"] = bool(choice)
            p["unsure"] = not choice
            if not choice or choice == inserted.lower():
                self.after_change()
                return
            w = self._set_pending_word(choice)
            if self.last and self.last.get("value") == inserted:
                self.last = {**self.last, "value": w,
                             "alternatives": [c for c in cands if c != choice][:5], "ai_corrected": True}
            self.after_change()
        return apply

    def _next_job(self, context, prefix, version):
        words = self.ai.next_words(context, prefix)
        words = [w for w in words if w in self.vocab or w in ("i",)]

        def apply():
            # the preview stays useful while you keep spelling the same word
            if prefix and self.prefix().startswith(prefix):
                self.ghost_ai = words
            if self.version != version:
                return
            self.ai_words = words if not prefix else self.ai_words
            self.compute_suggestions(ai_list=words)
        return apply

    # ------------------------------------------------------------ suggestions ---

    def pending_options(self):
        """Other words for the pending chord, for keys S1-S3."""
        keys = [self.suggest_midi.get("suggest1"), self.suggest_midi.get("suggest2"), self.suggest_midi.get("suggest3")]
        cur = self.pending["word"].lower()
        alts = [w for w in self.pending["cands"] if w != cur][:3]
        return [{"word": w, "midis": self.chord_for(w), "key": keys[i], "option": True}
                for i, w in enumerate(alts)]

    def compute_suggestions(self, ai_list=None):
        if self.pending:
            self.suggestions = self.pending_options()
            return
        if not self.settings.get("suggestions", True):
            self.suggestions = []
            return
        p = self.prefix()
        out = []
        if ai_list:
            out += [w for w in ai_list if w.startswith(p) and w != p]
        if p:
            out += list(self.vocab.completions(p))
        else:
            out += [w for w in self.ai_words]
            out += ["the", "i", "and", "to", "a", "you"] if self.at_sentence_start() else ["the", "and", "to", "a", "of", "is"]
        seen, sugg = set(), []
        keys = [self.suggest_midi.get("suggest1"), self.suggest_midi.get("suggest2"), self.suggest_midi.get("suggest3")]
        for w in out:
            if w in seen:
                continue
            seen.add(w)
            sugg.append({"word": w, "midis": self.chord_for(w), "key": keys[len(sugg)]})
            if len(sugg) == 3:
                break
        self.suggestions = sugg

    def after_change(self):
        sugg_on = self.settings.get("suggestions", True)
        p = self.prefix()
        if not sugg_on:
            self.ai_words = []
        if not p:
            self.ghost_ai = []
            # new word position: forget old AI ideas unless they were for this exact text
            if self.ai_words_for != self.text:
                self.ai_words = []
        self.compute_suggestions()
        self.broadcast_soon(self.state())
        # ask the AI for next words (suggestion chips) and, mid-word, for the grey
        # completion preview - even with chips off, since the preview types nothing
        if self.ai_enabled() and not self.pending and (sugg_on or p):
            v = self.version
            ctx = self.text[: len(self.text) - len(p)]
            self.ai_words_for = self.text if not p else self.ai_words_for
            self.loop.call_later(0.15, self._maybe_next, ctx, p, v)

    def _maybe_next(self, ctx, p, v):
        if self.version == v and self.ai_enabled():
            self.run_ai(self._next_job, ctx, p, v, kind="next")

    # ----------------------------------------------------------- audio side ---

    def start_audio(self):
        try:
            import audio_engine
        except Exception as e:
            print("audio engine not available:", e)
            self.broadcast_soon({"type": "toast", "message": f"Audio engine failed to load: {e}"})
            return
        if self.engine is None:
            self.engine = audio_engine.AudioEngine(self._on_engine_event, self._on_engine_level,
                                                   calib_path=str(HERE / "calibration.json"))
        if not self.engine.running:
            self.refresh_devices()
        self.engine.set_sensitivity(self.settings["sensitivity"])
        self.engine.set_note_shift(self.settings["note_shift"])
        if hasattr(self.engine, "set_chord_window"):
            self.engine.set_chord_window(self.settings["chord_window_ms"])
        if self.settings["input_mode"] != "piano":
            if self.engine.running:
                self.engine.stop()
            return
        dev = self.settings["device"]
        if dev is None or dev not in [d["index"] for d in self.devices]:
            dev = audio_engine.AudioEngine.default_device()
            self.settings["device"] = dev
        if self.engine.running:
            self.engine.stop()
        try:
            self.engine.start(dev)
        except Exception as e:
            self.broadcast_soon({"type": "toast", "message": f"Could not open audio input: {e}"})

    def _on_engine_event(self, ev):
        self.loop.call_soon_threadsafe(self._engine_event, ev)

    def _on_engine_level(self, lv):
        now = time.monotonic()
        if now - self.last_level_sent < 1 / 30:
            return
        self.last_level_sent = now
        self.loop.call_soon_threadsafe(self.broadcast_soon, {"type": "level", **lv})

    def _engine_event(self, ev):
        t = ev.get("type")
        if t == "press":
            self.handle_press(ev.get("midis", []), "audio")
        elif t == "lowest_found":
            shift = int(ev["midi"]) - 36
            self.settings["note_shift"] = shift
            self.engine.set_note_shift(shift)
            self.save_settings()
            self.calib = {"phase": "idle", "target": None, "done": 1, "total": 1,
                          "message": f"Lowest key found. Pitch offset {shift:+d}."}
            self.broadcast_soon({"type": "calib", **self.calib})
            self.broadcast_soon(self.settings_msg())
            self.broadcast_soon({"type": "toast", "message": f"Lowest key detected (offset {shift:+d} semitones)"})
        elif t == "calib_done":
            if not ev.get("ok"):
                self.broadcast_soon({"type": "toast", "message": "Didn't catch that one - press it again"})
                self.engine.begin_calibrate_key(self.calib["target"])
                return
            self.broadcast_soon({"type": "press", "midis": [ev["midi"]], "source": "audio"})
            self.next_calib_key()
        elif t == "error":
            self.broadcast_soon({"type": "toast", "message": ev.get("message", "audio error")})

    def key_label(self, m):
        e = self.keys.get(m)
        if not e:
            return "the lit key"
        v = e["value"]
        return {"space": "␣ space", "backspace": "⌫", "enter": "↵ enter", "shift": "⇧ shift",
                "delete_word": "delete word", "suggest1": "S1", "suggest2": "S2", "suggest3": "S3"}.get(v, v.upper())

    def next_calib_key(self):
        if self.calib["phase"] == "keys" and self.calib["target"] is not None:
            self.calib["done"] += 1
        if not self.calib_queue:
            if self.engine and hasattr(self.engine, "save_calibration"):
                self.engine.save_calibration()
            self.calib = {"phase": "idle", "target": None, "done": self.calib.get("done", 0),
                          "total": self.calib.get("total", 0), "message": "Calibration saved."}
            self.broadcast_soon({"type": "calib", **self.calib})
            self.broadcast_soon({"type": "toast", "message": "Calibration finished and saved"})
            return
        m = self.calib_queue.pop(0)
        self.calib.update({"phase": "keys", "target": m, "message": f"Press the glowing key ({self.key_label(m)}) once, firmly"})
        self.engine.begin_calibrate_key(m)
        self.broadcast_soon({"type": "calib", **self.calib})

    def calib_start(self, mode):
        if self.settings["input_mode"] != "piano" or not (self.engine and self.engine.running):
            self.broadcast_soon({"type": "toast", "message": "Switch to Piano mode with the keyboard plugged in first"})
            return
        if mode == "lowest":
            self.calib = {"phase": "lowest", "target": 36, "done": 0, "total": 1,
                          "message": "Press the LOWEST key on the keyboard (far left) once"}
            self.engine.begin_find_lowest()
            self.broadcast_soon({"type": "calib", **self.calib})
        else:
            # letters first (most important), then the controls, then the rest
            letters = sorted(self.letter_midi.values())
            rest = [m for m in range(36, 97) if m not in letters]
            self.calib_queue = letters + rest
            self.calib = {"phase": "keys", "target": None, "done": 0, "total": len(self.calib_queue), "message": ""}
            self.next_calib_key()

    def calib_cancel(self):
        if self.engine:
            self.engine.cancel_calibration()
        self.calib_queue = []
        self.calib = {"phase": "idle", "target": None, "done": 0, "total": 0, "message": "Cancelled"}
        self.broadcast_soon({"type": "calib", **self.calib})

    # ------------------------------------------------------------- settings ---

    def set_setting(self, key, value):
        if key not in DEFAULT_SETTINGS:
            return
        self.settings[key] = value
        if key == "model" and value:
            self.ai = ai_mod.AI(value, self.vocab)
            self.ai_status = "ready"
            self.run_ai(lambda: (self.ai.warmup(), None)[1])
        if key == "ai_enabled":
            self.ai_status = ("ready" if value else "off") if self.ai else "unavailable"
        if key in ("input_mode", "device"):
            self.start_audio()
        if key == "suggestions":
            self.after_change()
        if self.engine:
            if key == "sensitivity":
                self.engine.set_sensitivity(float(value))
            elif key == "note_shift":
                self.engine.set_note_shift(int(value))
            elif key == "chord_window_ms" and hasattr(self.engine, "set_chord_window"):
                self.engine.set_chord_window(int(value))
        self.save_settings()
        self.broadcast_soon(self.settings_msg())
        self.broadcast_soon(self.state())

    # ------------------------------------------------------------ websocket ---

    async def ws_handler(self, request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self.clients.add(ws)
        if not (self.engine and self.engine.running):
            # the sound card may have been plugged in since start: pick it up on page load
            if self.settings["input_mode"] == "piano":
                self.start_audio()
            else:
                self.refresh_devices()
        init ={"type": "init", "layout": self.layout, "settings": self.settings_msg()["settings"],
                "devices": self.devices, "state": self.state(), "calib": self.calib}
        await ws.send_str(json.dumps(init))
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    self.on_client(json.loads(msg.data))
                except Exception as e:
                    print("bad client message:", e)
        finally:
            self.clients.discard(ws)
        return ws

    def on_client(self, m):
        t = m.get("type")
        if t == "sim_press":
            if m.get("via") == "kbd" and self.settings["type_external"]:
                return  # keystrokes we typed ourselves would loop back as presses
            self.handle_press([int(x) for x in m.get("midis", [])], "sim")
        elif t == "set":
            self.set_setting(m.get("key"), m.get("value"))
        elif t == "pick_suggestion":
            if self.pick_suggestion(int(m.get("index", 0))):
                self.after_change()
        elif t == "clear":
            self.text = ""  # don't wipe the other app's text
            self.version += 1
            self.pending = None
            self.history = []
            self.last = None
            self.shift = False
            self.after_change()
        elif t == "calib_start":
            self.calib_start(m.get("mode"))
        elif t == "calib_skip":
            if self.calib["phase"] == "keys":
                self.next_calib_key()
            elif self.calib["phase"] == "lowest":
                self.calib_cancel()
        elif t == "calib_cancel":
            self.calib_cancel()

    async def on_startup(self, app):
        self.loop = asyncio.get_running_loop()
        self.start_audio()
        self.compute_suggestions()
        if self.ai_enabled():
            self.run_ai(lambda: (self.ai.warmup(), None)[1])
            self.after_change()

    async def on_shutdown(self, app):
        if self.engine and self.engine.running:
            self.engine.stop()
        self.save_settings()


async def index(request):
    return web.FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    app_state = App()

    @web.middleware
    async def no_cache(request, handler):
        # the UI changes often; never let the browser run an old app.js
        resp = await handler(request)
        if request.path != "/ws":
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp

    app = web.Application(middlewares=[no_cache])
    app.router.add_get("/", index)
    app.router.add_get("/ws", app_state.ws_handler)
    app.router.add_static("/", WEB, show_index=False)
    app.on_startup.append(app_state.on_startup)
    app.on_shutdown.append(app_state.on_shutdown)

    url = f"http://localhost:{args.port}"
    print(f"Piano Typer running at {url}  (Ctrl+C to quit)")
    if not args.no_browser:
        asyncio.get_event_loop_policy()
        webbrowser.open(url)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
