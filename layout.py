"""
Finds a "musical" letter layout for the 26 letter keys (MIDI 57-82, A3..A#5)
and writes layout.json (see SPEC.md).

What makes a layout good:
  1. Common words, played as chords, sound consonant.
  2. Common letters sit on white keys, so most of what you play is in C major.
  3. Typing letter by letter inside common words makes smooth melodies
     (small, consonant steps instead of wild leaps).
  4. The most common letters sit near the middle, under the hands.

Run:  python layout.py            (about 10-20 seconds)
"""

import json
import math
import random
from pathlib import Path

import numpy as np
from wordfreq import top_n_list, word_frequency

HERE = Path(__file__).parent
LETTERS = "abcdefghijklmnopqrstuvwxyz"
SLOTS = list(range(57, 83))           # 26 chromatic keys
BLACK_PC = {1, 3, 6, 8, 10}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Roughness of a harmonic interval class (0 = sweet, 1 = harsh).
ROUGH = np.array([0.0,   # unison / octave
                  1.00,  # minor 2nd
                  0.55,  # major 2nd
                  0.12,  # minor 3rd
                  0.08,  # major 3rd
                  0.15,  # perfect 4th
                  0.75,  # tritone
                  0.00,  # perfect 5th
                  0.15,  # minor 6th
                  0.10,  # major 6th
                  0.45,  # minor 7th
                  0.90]) # major 7th

W_HARMONY, W_INKEY, W_MELODY, W_CENTER, W_DETECT = 1.0, 1.6, 0.35, 0.15, 30.0

# The laptop *listens* for chords. A note sitting exactly on an overtone of a lower
# note in the same chord (octave, octave+5th, two octaves) is very hard to hear apart,
# so letters that share words should avoid these distances.
OVERTONE_CLASH = {12: 1.0, 19: 0.8, 24: 0.9}


def melodic_cost(d):
    d = abs(d)
    size = 0.0 if d <= 4 else 0.12 * (d - 4) if d <= 12 else 1.0 + 0.1 * (d - 12)
    return size + 0.5 * ROUGH[d % 12]


def build_weights(n_words=6000):
    harm = np.zeros((26, 26))
    melo = np.zeros((26, 26))
    letter = np.zeros(26)
    words = [w for w in top_n_list("en", n_words) if w.isascii() and w.isalpha()]
    for w in words:
        f = math.sqrt(word_frequency(w, "en"))
        u = sorted(set(w))
        idx = [LETTERS.index(c) for c in u]
        if 2 <= len(idx) <= 7:
            npairs = len(idx) * (len(idx) - 1) / 2
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    harm[idx[i], idx[j]] += f / npairs
        for a, b in zip(w, w[1:]):
            if a != b:
                melo[LETTERS.index(a), LETTERS.index(b)] += f
        for c in w:
            letter[LETTERS.index(c)] += f
    harm /= harm.sum()
    melo /= melo.sum()
    letter /= letter.sum()
    return harm, melo, letter, words


def make_cost(harm, melo, letter):
    slots = np.array(SLOTS)
    diff = np.abs(slots[:, None] - slots[None, :])
    rough_ss = ROUGH[diff % 12]                                   # slot x slot
    melo_ss = np.vectorize(melodic_cost)(slots[:, None] - slots[None, :])
    black_s = np.array([1.0 if s % 12 in BLACK_PC else 0.0 for s in slots])
    center_s = np.abs(slots - 69.5) / 13.0
    clash_ss = np.vectorize(lambda d: OVERTONE_CLASH.get(abs(int(d)), 0.0))(slots[:, None] - slots[None, :])

    def cost(perm):  # perm[letter] = slot index
        p = perm
        c_h = (harm * rough_ss[p][:, p]).sum()
        c_m = (melo * melo_ss[p][:, p]).sum()
        c_k = (letter * black_s[p]).sum()
        c_c = (letter * center_s[p]).sum()
        c_d = (harm * clash_ss[p][:, p]).sum()
        return (W_HARMONY * c_h + W_MELODY * c_m + W_INKEY * c_k + W_CENTER * c_c + W_DETECT * c_d,
                (c_h, c_m, c_k, c_c, c_d))

    return cost


def anneal(cost, iters=120000, seed=0):
    rng = random.Random(seed)
    perm = np.arange(26)
    rng.shuffle(perm)
    cur, _ = cost(perm)
    best, best_perm = cur, perm.copy()
    t0, t1 = 0.05, 0.0005
    for i in range(iters):
        t = t0 * (t1 / t0) ** (i / iters)
        a, b = rng.randrange(26), rng.randrange(26)
        if a == b:
            continue
        perm[a], perm[b] = perm[b], perm[a]
        new, _ = cost(perm)
        if new <= cur or rng.random() < math.exp((cur - new) / t):
            cur = new
            if cur < best:
                best, best_perm = cur, perm.copy()
        else:
            perm[a], perm[b] = perm[b], perm[a]
    return best, best_perm


CONTROLS = {
    36: ("control", "enter"), 38: ("char", "."), 40: ("char", ","), 41: ("char", "?"),
    43: ("char", "!"), 45: ("char", "'"), 47: ("char", "-"),
    48: ("control", "complete"), 50: ("control", "double"), 52: ("control", "undo"),
    53: ("control", "shift"), 54: ("control", "space"), 55: ("control", "delete_word"),
    56: ("control", "backspace"),
    49: ("control", "suggest1"), 51: ("control", "suggest2"), 46: ("control", "suggest3"),
}
DIGITS = {83: "1", 84: "2", 86: "3", 88: "4", 89: "5", 91: "6", 93: "7", 95: "8", 96: "9", 94: "0"}


def note_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def main():
    harm, melo, letter, words = build_weights()
    cost = make_cost(harm, melo, letter)

    # Baseline: alphabetical layout, for comparison
    base, base_parts = cost(np.arange(26))
    results = [anneal(cost, seed=s) for s in range(4)]
    best, perm = min(results, key=lambda r: r[0])
    _, parts = cost(perm)

    keys = {}
    for m, (kind, value) in CONTROLS.items():
        keys[str(m)] = {"kind": kind, "value": value}
    for m, d in DIGITS.items():
        keys[str(m)] = {"kind": "char", "value": d}
    letter_at = {}
    for li, si in enumerate(perm):
        m = SLOTS[si]
        keys[str(m)] = {"kind": "letter", "value": LETTERS[li]}
        letter_at[LETTERS[li]] = m

    out = {
        "version": 1,
        "lowest_midi": 36,
        "highest_midi": 96,
        "keys": dict(sorted(keys.items(), key=lambda kv: int(kv[0]))),
        "meta": {
            "score": round(best, 5),
            "alphabetical_score": round(base, 5),
            "parts": dict(zip(["harmony", "melody", "black_keys", "center", "overtone_clash"],
                              [round(float(x), 4) for x in parts])),
        },
    }
    (HERE / "layout.json").write_text(json.dumps(out, indent=1))

    print(f"score {best:.4f}  (alphabetical layout: {base:.4f})")
    print("harmony/melody/black/center/clash:", [round(float(x), 4) for x in parts],
          " vs alpha:", [round(float(x), 4) for x in base_parts])
    print("\nKeys left to right:")
    for m in SLOTS:
        l = next(c for c, mm in letter_at.items() if mm == m)
        print(f"  {note_name(m):<4} {'(black)' if m % 12 in BLACK_PC else '       '} {l}")
    print("\nCommon word chords:")
    for w in ["the", "and", "you", "for", "that", "with", "this", "have", "love", "piano", "music"]:
        ms = sorted({letter_at[c] for c in w})
        print(f"  {w:<6} {' '.join(note_name(m) for m in ms)}")


if __name__ == "__main__":
    main()
