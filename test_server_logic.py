"""Checks the typing logic in server.App without audio, AI or a browser."""

import asyncio
import sys

import server


def make_app():
    app = server.App()
    app.settings["ai_enabled"] = False
    app.settings["type_external"] = False
    app.settings["suggestions"] = True
    app.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(app.loop)
    return app


def control(app, value):
    return next(m for m, e in sorted(app.keys.items()) if e["kind"] == "control" and e["value"] == value)


def char(app, ch):
    return next(m for m, e in app.keys.items() if e["kind"] == "char" and e["value"] == ch)


def main():
    app = make_app()
    L = app.letter_midi
    fails = 0

    def check(label, want):
        nonlocal fails
        ok = app.text == want
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {app.text!r}" + ("" if ok else f"  (wanted {want!r})"))

    def press(*ms):
        app.handle_press(list(ms), "sim")

    def word(w):
        press(*app.chord_for(w))

    # letter by letter, auto-capital at start
    for c in "hi":
        press(L[c])
    press(control(app, "space"))
    check("letters + space", "Hi ")

    # chord words
    word("there")
    check("chord 'there'", "Hi there ")
    word("stop")
    check("chord 'stop' -> most common anagram", "Hi there stop ")
    print("     alternatives offered:", app.last["alternatives"])

    # punctuation removes the space before it
    press(char(app, "."))
    check("full stop", "Hi there stop. ")
    word("the")
    check("capital after full stop", "Hi there stop. The ")

    # backspace / delete word
    press(control(app, "delete_word"))
    check("delete word", "Hi there stop. ")
    press(control(app, "backspace"))
    check("backspace", "Hi there stop.")

    # shift
    press(control(app, "backspace"))
    press(char(app, ","))
    press(control(app, "shift"))
    press(L["b"])
    check("shift capital", "Hi there stop, B")

    # suggestions complete a prefix
    app.handle_press([control(app, "delete_word")], "sim")
    for c in "pia":
        press(L[c])
    print("     suggestions for 'pia':", [s["word"] for s in app.suggestions])
    press(control(app, "suggest1"))
    check("accept suggestion", "Hi there stop, piano ")

    # 'i' alone is capitalised
    word("am")
    press(L["i"])
    check("letter i", "Hi there stop, piano am i")

    # chord that is not a word
    press(L["q"], L["x"], L["z"])
    check("non-word chord types nothing", "Hi there stop, piano am i")
    print("     last:", app.last)

    # next-letter guide
    app.text = "th"
    print("     next letters after 'th':", app.state()["next_letters"])

    print("ALL PASSED" if not fails else f"{fails} FAILED")
    return fails == 0


def test_new_controls():
    """Pending chord words, Complete, Double letter, Undo, options and reject."""
    app = make_app()
    L = app.letter_midi
    fails = 0

    def check(label, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}" + ("" if ok else f"  (wanted {want!r})"))

    def press(*ms):
        app.handle_press(list(ms), "sim")

    def word(w):
        press(*app.chord_for(w))

    word("stop")
    check("chord word is pending", (app.text, bool(app.pending)), ("Stop ", True))
    check("options shown for pending word", [s["word"] for s in app.suggestions][:1], ["post"])
    press(control(app, "complete"))
    check("Complete accepts it", (app.text, app.pending), ("Stop ", None))

    word("stop")
    press(control(app, "suggest1"))
    check("option 1 replaces the pending word", (app.text, app.pending), ("Stop post ", None))

    word("stop")
    press(control(app, "backspace"))
    check("backspace rejects the pending word", (app.text, app.pending), ("Stop post ", None))

    word("stop")
    press(L["a"])
    check("next letter accepts pending word", (app.text, app.pending), ("Stop post stop a", None))
    press(control(app, "undo"))
    check("undo removes the letter", app.text, "Stop post stop ")
    check("undo brings the word back as pending", bool(app.pending), True)
    press(control(app, "complete"))

    for c in "he":
        press(L[c])
    press(L["l"])
    press(control(app, "double"))
    check("double letter", app.text, "Stop post stop hell")
    press(control(app, "backspace"))
    press(control(app, "complete"))
    check("Complete finishes a half word (hel -> help)", app.text, "Stop post stop help ")
    t = app.text
    press(control(app, "complete"))
    check("Complete with nothing to finish = space", app.text, t + " ")
    press(L["h"]); press(L["i"])
    press(control(app, "complete"))
    check("Complete after a whole word ('hi') = space, not 'his'", app.text.endswith(" Hi ") or app.text.endswith(" hi "), True)
    return fails


if __name__ == "__main__":
    ok = main()
    print()
    print("new controls:")
    f = test_new_controls()
    print("ALL PASSED" if not f else f"{f} FAILED")
    sys.exit(0 if ok and not f else 1)
