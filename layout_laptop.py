"""
"Folded laptop" layout for Piano Typer: a laptop keyboard flattened into one row.

    left edge     left-hand letters        centre        right-hand letters    right edge   digits
    ⇧ ↶ ⌫w S1-3 | q w e r t a s d f g … | ⌫  ␣  ×2  ✓ | y u i o p h j k l n m | . , ? ! ' - ↵ | 1-9 0
                  rare  ───────► common    (thumbs)      common ───────► rare

- Each hand keeps its laptop letters (QWERTY split), so "which hand" still feels right.
- Space / backspace / complete sit in the middle around middle C, like a space bar.
- The most used letters (vowels first) sit next to the centre; rare letters go outward
  and onto black keys.
- Inside those rules the search still prefers word chords that sound good and letters
  the laptop can hear apart (no octave / octave+5th clashes between letters of a word).

Run:  python layout_laptop.py     (writes layout.json; the old one is layout_musical_backup.json)
"""

import json
import math
import random
from pathlib import Path

import numpy as np

from layout import LETTERS, ROUGH, OVERTONE_CLASH, BLACK_PC, build_weights, melodic_cost, note_name

HERE = Path(__file__).parent

LEFT_HAND = set("qwertasdfgzxcvb")
RIGHT_HAND = set("yuiophjklnm")

# --- fixed keys -------------------------------------------------------------
BACKSPACE, SPACE, DOUBLE, COMPLETE = 59, 60, 61, 62        # B3  C4  C#4  D4
LEFT_SLOTS = list(range(44, 59))                           # G#2..A#3  (15 keys)
RIGHT_SLOTS = list(range(63, 74))                          # D#4..C#5  (11 keys)
LEFT_CENTRE, RIGHT_CENTRE = BACKSPACE, COMPLETE

CONTROLS = {
    # left edge (like a laptop's shift / tab side)
    36: ("control", "undo"), 37: ("control", "suggest1"), 38: ("control", "delete_word"),
    39: ("control", "suggest2"), 42: ("control", "suggest3"), 43: ("control", "shift"),
    # centre (thumbs)
    BACKSPACE: ("control", "backspace"), SPACE: ("control", "space"),
    DOUBLE: ("control", "double"), COMPLETE: ("control", "complete"),
    # right edge (like a laptop's punctuation / enter side)
    74: ("char", "."), 75: ("char", "'"), 76: ("char", ","), 77: ("char", "?"),
    78: ("char", "-"), 79: ("char", "!"), 81: ("control", "enter"),
}
DIGITS = {83: "1", 84: "2", 86: "3", 88: "4", 89: "5", 91: "6", 93: "7", 95: "8", 96: "9", 94: "0"}

W_HARMONY, W_INKEY, W_MELODY, W_CENTER, W_DETECT = 1.0, 6.0, 0.25, 2.0, 30.0


def make_cost(harm, melo, letter):
    def cost(slot):  # slot[letter index] = midi
        s = np.asarray(slot)
        diff = np.abs(s[:, None] - s[None, :])
        c_h = (harm * ROUGH[diff % 12]).sum()
        c_m = (melo * np.vectorize(melodic_cost)(s[:, None] - s[None, :])).sum()
        black = np.array([1.0 if m % 12 in BLACK_PC else 0.0 for m in s])
        c_k = (letter * black).sum()
        dist = np.array([(LEFT_CENTRE - m) if m < SPACE else (m - RIGHT_CENTRE) for m in s]) / 12.0
        c_c = (letter * dist).sum()
        clash = np.vectorize(lambda d: OVERTONE_CLASH.get(int(d), 0.0))(diff)
        c_d = (harm * clash).sum()
        return (W_HARMONY * c_h + W_MELODY * c_m + W_INKEY * c_k + W_CENTER * c_c + W_DETECT * c_d,
                (c_h, c_m, c_k, c_c, c_d))
    return cost


def anneal(cost, iters=40000, seed=0):
    rng = random.Random(seed)
    left = [LETTERS.index(c) for c in sorted(LEFT_HAND)]
    right = [LETTERS.index(c) for c in sorted(RIGHT_HAND)]
    slot = [0] * 26
    ls, rs = LEFT_SLOTS[:], RIGHT_SLOTS[:]
    rng.shuffle(ls); rng.shuffle(rs)
    for li, m in zip(left, ls):
        slot[li] = m
    for li, m in zip(right, rs):
        slot[li] = m
    cur, _ = cost(slot)
    best, best_slot = cur, slot[:]
    t0, t1 = 0.05, 0.0005
    for i in range(iters):
        t = t0 * (t1 / t0) ** (i / iters)
        group = left if rng.random() < len(left) / 26 else right
        a, b = rng.sample(group, 2)
        slot[a], slot[b] = slot[b], slot[a]
        new, _ = cost(slot)
        if new <= cur or rng.random() < math.exp((cur - new) / t):
            cur = new
            if cur < best:
                best, best_slot = cur, slot[:]
        else:
            slot[a], slot[b] = slot[b], slot[a]
    return best, best_slot


def main():
    assert LEFT_HAND | RIGHT_HAND == set(LETTERS) and len(LEFT_SLOTS) == 15 and len(RIGHT_SLOTS) == 11
    harm, melo, letter, _ = build_weights()
    cost = make_cost(harm, melo, letter)
    best, slot = min((anneal(cost, seed=s) for s in range(4)), key=lambda r: r[0])
    _, parts = cost(slot)

    keys = {str(m): {"kind": k, "value": v} for m, (k, v) in CONTROLS.items()}
    keys.update({str(m): {"kind": "char", "value": d} for m, d in DIGITS.items()})
    letter_at = {}
    for li, m in enumerate(slot):
        keys[str(m)] = {"kind": "letter", "value": LETTERS[li]}
        letter_at[LETTERS[li]] = m

    out = {
        "version": 2,
        "name": "folded laptop",
        "lowest_midi": 36,
        "highest_midi": 96,
        "keys": dict(sorted(keys.items(), key=lambda kv: int(kv[0]))),
        "zones": [
            {"a": 36, "b": 43, "cls": "z-ctrl", "name": "Edit"},
            {"a": 44, "b": 58, "cls": "z-letter", "name": "Left hand"},
            {"a": 59, "b": 62, "cls": "z-ctrl", "name": "Space"},
            {"a": 63, "b": 73, "cls": "z-letter", "name": "Right hand"},
            {"a": 74, "b": 82, "cls": "z-ctrl", "name": "Punct ↵"},
            {"a": 83, "b": 96, "cls": "z-digit", "name": "Digits"},
        ],
        "meta": {"score": round(best, 5),
                 "parts": dict(zip(["harmony", "melody", "black_keys", "center", "overtone_clash"],
                                   [round(float(x), 4) for x in parts]))},
    }
    (HERE / "layout.json").write_text(json.dumps(out, indent=1))

    print("score", round(best, 4), "parts", [round(float(x), 4) for x in parts])
    row = []
    for m in range(36, 97):
        e = keys.get(str(m))
        row.append(f"{note_name(m)}{'*' if m % 12 in BLACK_PC else ''}={e['value'] if e else '·'}")
    for i in range(0, len(row), 8):
        print("  " + "  ".join(row[i:i + 8]))
    print("\nword chords:")
    for w in ["the", "and", "you", "for", "that", "with", "have", "love", "like", "piano", "hello"]:
        print(f"  {w:<6} {' '.join(note_name(m) for m in sorted({letter_at[c] for c in w}))}")


if __name__ == "__main__":
    main()
