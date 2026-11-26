"""Tiny mock of the Piano Typer server, for front-end testing without audio/AI.

Run:  %USERPROFILE%\\.pianotyper\\venv\\Scripts\\python.exe web\\mock_server.py [--port 8770]
Then open http://127.0.0.1:8770/
"""
import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

from aiohttp import web

WEB = Path(__file__).resolve().parent
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def build_layout():
    keys = {}
    for i, ch in enumerate("etaoinsrhldcumfpgwybvkxjqz"):
        keys[str(57 + i)] = {"kind": "letter", "value": ch}
    ctrl = {36: "enter", 48: "space", 50: "space", 52: "space", 53: "backspace", 55: "delete_word",
            49: "suggest1", 51: "suggest2", 54: "suggest3", 56: "shift"}
    for m, v in ctrl.items():
        keys[str(m)] = {"kind": "control", "value": v}
    for m, v in {38: ".", 40: ",", 41: "?", 43: "!", 45: "'", 47: "-"}.items():
        keys[str(m)] = {"kind": "char", "value": v}
    for i, m in enumerate([83, 84, 86, 88, 89, 91, 93, 95, 96]):
        keys[str(m)] = {"kind": "char", "value": str(i + 1)}
    keys["94"] = {"kind": "char", "value": "0"}
    return {"version": 1, "lowest_midi": 36, "highest_midi": 96, "keys": keys}


LAYOUT = build_layout()
LETTER_MIDI = {v["value"]: int(k) for k, v in LAYOUT["keys"].items() if v["kind"] == "letter"}

WORDS = ("the of and to a in is it you that he was for on are with as i his they be at one have this from "
         "or had by hot word but what some we can out other were all there when up use your how said an each "
         "she which do their time if will way about many then them write would like so these her long make "
         "thing see him two has look more day could go come did number sound no most people my over know "
         "water than call first who may down side been now find any new work part take get place made live "
         "where after back little only round man year came show every good me give our under name very "
         "through just form sentence great think say help low line differ turn cause much mean before move "
         "right boy old too same tell does set three want air well also play small end put home read hand "
         "port large spell add even land here must big high such follow act why ask men change went light "
         "kind off need house picture try us again animal point mother world near build self earth father "
         "stop pots spot tops post opts piano music chord key note song happy hello there their three").split()
FREQ = {w: 1.0 / (i + 1) ** 0.7 for i, w in enumerate(dict.fromkeys(WORDS))}


def chord_of(word):
    return sorted({LETTER_MIDI[c] for c in word if c in LETTER_MIDI})


class Mock:
    def __init__(self):
        self.clients = set()
        self.text = ""
        self.shift = False
        self.last = None
        self.settings = {"device": 1, "input_mode": "sim", "type_external": False, "ai_enabled": True,
                         "model": "phi3.5", "models": ["phi3.5", "qwen2.5:7b-instruct"],
                         "sensitivity": 0.5, "note_shift": 0, "chord_window_ms": 70, "audio_running": False}
        self.devices = [{"index": 1, "name": "Microphone (USB PnP Audio Device)", "samplerate": 48000},
                        {"index": 2, "name": "Microphone Array (Intel Smart Sound)", "samplerate": 48000}]
        self.calib = {"phase": "idle", "target": None, "done": 0, "total": 61, "message": ""}
        self.ai = {"status": "ready", "model": "phi3.5", "latency_ms": 640}
        self.level_boost = 0.0
        self.active_until = {}

    # ---- derived state
    def prefix(self):
        t = self.text
        i = len(t)
        while i > 0 and t[i - 1].isalpha():
            i -= 1
        return t[i:].lower()

    def suggestions(self):
        p = self.prefix()
        cands = [w for w in FREQ if w.startswith(p) and w != p] if p else list(FREQ)
        cands.sort(key=lambda w: -FREQ[w])
        keys = [49, 51, 54]
        return [{"word": w, "midis": chord_of(w), "key": keys[i]} for i, w in enumerate(cands[:3])]

    def next_letters(self):
        p = self.prefix()
        scores = {}
        for w, f in FREQ.items():
            if w.startswith(p) and len(w) > len(p):
                c = w[len(p)]
                scores[c] = scores.get(c, 0) + f
        tot = sum(scores.values()) or 1
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:8]
        return {c: round(v / tot, 3) for c, v in top}

    def state(self):
        return {"text": self.text, "prefix": self.prefix(), "suggestions": self.suggestions(),
                "next_letters": self.next_letters(), "last": self.last, "ai": self.ai, "shift": self.shift}

    # ---- messaging
    async def send(self, ws, obj):
        try:
            await ws.send_str(json.dumps(obj))
        except Exception:
            pass

    async def broadcast(self, obj):
        for ws in list(self.clients):
            await self.send(ws, obj)

    async def push_state(self):
        await self.broadcast({"type": "state", **self.state()})

    def append_word(self, word):
        if self.text and not self.text[-1].isspace() and self.text[-1] not in "\n":
            # replace partial prefix
            p = self.prefix()
            if p:
                self.text = self.text[: len(self.text) - len(p)]
            elif not self.text.endswith(" "):
                self.text += " "
        if self.shift:
            word = word[:1].upper() + word[1:]
            self.shift = False
        self.text += word + " "

    async def on_press(self, midis):
        self.level_boost = 0.7
        t = time.time()
        for m in midis:
            self.active_until[m] = t + 0.25
        await self.broadcast({"type": "press", "midis": midis, "source": "sim"})

        if self.calib["phase"] == "lowest":
            self.calib = {"phase": "idle", "target": None, "done": 0, "total": 61, "message": ""}
            await self.broadcast({"type": "calib", **self.calib})
            await self.broadcast({"type": "toast", "message": f"Lowest key found: {note_name(midis[0])} (no shift needed)"})
            return
        if self.calib["phase"] == "keys":
            await self.advance_calib()
            return

        keys = [LAYOUT["keys"].get(str(m)) for m in midis]
        if len(midis) == 1 and keys[0]:
            k = keys[0]
            v = k["value"]
            if k["kind"] == "letter":
                ch = v.upper() if self.shift else v
                self.shift = False
                self.text += ch
                self.last = {"kind": "letter", "value": v, "midis": midis, "alternatives": [], "ai_corrected": False}
            elif k["kind"] == "char":
                if v in ".,?!" and self.text.endswith(" "):
                    self.text = self.text[:-1]
                self.text += v + (" " if v in ".,?!" else "")
                if v in ".?!":
                    self.shift = True
                self.last = {"kind": "char", "value": v, "midis": midis, "alternatives": [], "ai_corrected": False}
            else:
                if v == "space":
                    self.text += " "
                elif v == "backspace":
                    self.text = self.text[:-1]
                elif v == "enter":
                    self.text += "\n"
                elif v == "delete_word":
                    self.text = self.text.rstrip()
                    while self.text and not self.text[-1].isspace():
                        self.text = self.text[:-1]
                elif v == "shift":
                    self.shift = not self.shift
                elif v.startswith("suggest"):
                    await self.pick(int(v[-1]) - 1)
                    return
                self.last = {"kind": "control", "value": v, "midis": midis, "alternatives": [], "ai_corrected": False}
        else:
            letters = {k["value"] for k in keys if k and k["kind"] == "letter"}
            matches = [w for w in FREQ if set(w) == letters]
            matches.sort(key=lambda w: -FREQ[w])
            if matches:
                corrected = len(matches) > 1 and random.random() < 0.4
                if corrected:
                    matches[0], matches[1] = matches[1], matches[0]
                    self.ai = {**self.ai, "status": "thinking"}
                    await self.push_state()
                    await asyncio.sleep(0.4)
                    self.ai = {**self.ai, "status": "ready", "latency_ms": random.randint(380, 1100)}
                self.append_word(matches[0])
                self.last = {"kind": "word", "value": matches[0], "midis": midis,
                             "alternatives": matches[1:5], "ai_corrected": corrected}
            else:
                self.last = {"kind": "unknown", "value": "", "midis": midis, "alternatives": [], "ai_corrected": False}
        await self.push_state()

    async def pick(self, idx):
        s = self.suggestions()
        if idx < len(s):
            w = s[idx]["word"]
            self.append_word(w)
            self.last = {"kind": "word", "value": w, "midis": s[idx]["midis"], "alternatives": [], "ai_corrected": False}
        await self.push_state()

    async def advance_calib(self, skipped=False):
        c = self.calib
        nxt = (c["target"] or 35) + 1
        done = c["done"] + 1
        if nxt > 96:
            self.calib = {"phase": "idle", "target": None, "done": 61, "total": 61, "message": ""}
            await self.broadcast({"type": "calib", **self.calib})
            await self.broadcast({"type": "toast", "message": "Calibration saved for all keys"})
            return
        k = LAYOUT["keys"].get(str(nxt))
        lbl = (k["value"].upper() if k and k["kind"] == "letter" else (k["value"] if k else ""))
        self.calib = {"phase": "keys", "target": nxt, "done": done, "total": 61,
                      "message": f"Press {note_name(nxt)}" + (f" (marked {lbl})" if lbl else "")}
        await self.broadcast({"type": "calib", **self.calib})

    async def handle(self, ws, msg):
        t = msg.get("type")
        if t == "sim_press":
            midis = sorted({int(m) for m in msg.get("midis", []) if 36 <= int(m) <= 96})
            if midis:
                await self.on_press(midis)
        elif t == "set":
            k, v = msg.get("key"), msg.get("value")
            self.settings[k] = v
            if k == "ai_enabled":
                self.ai = {**self.ai, "status": "ready" if v else "off"}
                await self.push_state()
            if k == "model":
                self.ai = {**self.ai, "model": v}
                await self.push_state()
            if k == "input_mode":
                self.settings["audio_running"] = v == "piano"
            await self.broadcast({"type": "settings", "settings": self.settings})
        elif t == "pick_suggestion":
            await self.pick(int(msg.get("index", 0)))
        elif t == "clear":
            self.text, self.last, self.shift = "", None, False
            await self.push_state()
        elif t == "calib_start":
            if msg.get("mode") == "lowest":
                self.calib = {"phase": "lowest", "target": None, "done": 0, "total": 0,
                              "message": "Press the lowest key on your keyboard"}
                await self.broadcast({"type": "calib", **self.calib})
            else:
                self.calib = {"phase": "keys", "target": 35, "done": -1, "total": 61, "message": ""}
                await self.advance_calib()
        elif t == "calib_skip":
            if self.calib["phase"] == "keys":
                await self.advance_calib(skipped=True)
        elif t == "calib_cancel":
            self.calib = {"phase": "idle", "target": None, "done": 0, "total": 61, "message": ""}
            await self.broadcast({"type": "calib", **self.calib})
            await self.broadcast({"type": "toast", "message": "Calibration cancelled"})

    async def ws_handler(self, request):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        self.clients.add(ws)
        await self.send(ws, {"type": "init", "layout": LAYOUT, "settings": self.settings, "devices": self.devices,
                             "state": self.state(), "calib": self.calib})
        try:
            async for m in ws:
                if m.type == web.WSMsgType.TEXT:
                    try:
                        await self.handle(ws, json.loads(m.data))
                    except Exception as e:  # keep the mock alive
                        print("mock error:", e)
        finally:
            self.clients.discard(ws)
        return ws

    async def level_loop(self, app):
        try:
            while True:
                await asyncio.sleep(1 / 30)
                if not self.clients:
                    continue
                now = time.time()
                self.level_boost *= 0.9
                rms = min(1.0, 0.015 + 0.01 * abs(math.sin(now * 3)) + random.random() * 0.01 + self.level_boost * 0.5)
                active = sorted(m for m, u in self.active_until.items() if u > now)
                self.active_until = {m: u for m, u in self.active_until.items() if u > now}
                await self.broadcast({"type": "level", "rms": round(rms, 4), "active": active})
        except asyncio.CancelledError:
            pass


async def index(request):
    return web.FileResponse(WEB / "index.html", headers={"Cache-Control": "no-store"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    mock = Mock()
    app = web.Application()
    app.router.add_get("/ws", mock.ws_handler)
    app.router.add_get("/", index)
    app.router.add_static("/", WEB, show_index=False)

    async def start_bg(app):
        app["lvl"] = asyncio.create_task(mock.level_loop(app))

    async def stop_bg(app):
        app["lvl"].cancel()

    app.on_startup.append(start_bg)
    app.on_cleanup.append(stop_bg)
    print(f"Piano Typer mock on http://127.0.0.1:{args.port}/")
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
