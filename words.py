"""
Vocabulary for Piano Typer.

- chord -> word candidates (a chord gives the *set* of letters, not their order)
- next-letter probabilities for the on-screen guide
- word completions for a typed prefix
"""

import bisect
from collections import defaultdict
from functools import lru_cache

from wordfreq import top_n_list, word_frequency

VOCAB_SIZE = 40000
SINGLE_LETTER_WORDS = {"a", "i"}


class Vocab:
    def __init__(self, size=VOCAB_SIZE):
        words = []
        for w in top_n_list("en", size):
            if not w.isascii() or not w.isalpha():
                continue
            if len(w) == 1 and w not in SINGLE_LETTER_WORDS:
                continue
            words.append(w)
        self.freq = {w: word_frequency(w, "en") for w in words}
        self.by_set = defaultdict(list)
        for w in words:
            self.by_set[frozenset(w)].append(w)
        for k in self.by_set:
            self.by_set[k].sort(key=lambda w: -self.freq[w])
        self.sorted_words = sorted(words)
        self.common = words[:200]  # frequency order

    def __contains__(self, word):
        return word in self.freq

    # ---- chords -------------------------------------------------------------

    def candidates(self, letters, limit=8):
        """
        Words whose set of distinct letters equals `letters`.
        If none, allow one detection error (a missing or an extra letter).
        Returns [(word, score)] best first; score is relative frequency.
        """
        key = frozenset(letters)
        exact = [(w, self.freq[w]) for w in self.by_set.get(key, [])]
        if exact:
            return exact[:limit]
        near = []
        # one extra (ghost) note was heard: drop each letter
        for l in key:
            sub = key - {l}
            near += [(w, self.freq[w] * 0.3) for w in self.by_set.get(sub, [])[:4]]
        # one note was missed: add each letter
        for l in "abcdefghijklmnopqrstuvwxyz":
            if l not in key:
                near += [(w, self.freq[w] * 0.2) for w in self.by_set.get(key | {l}, [])[:3]]
        near.sort(key=lambda x: -x[1])
        seen, out = set(), []
        for w, s in near:
            if w not in seen:
                seen.add(w)
                out.append((w, s))
        return out[:limit]

    # ---- prefixes -----------------------------------------------------------

    def _range(self, prefix):
        lo = bisect.bisect_left(self.sorted_words, prefix)
        hi = bisect.bisect_left(self.sorted_words, prefix + "\x7f")
        return lo, hi

    @lru_cache(maxsize=4096)
    def next_letters(self, prefix):
        """{letter: prob} for the letter after `prefix`; ' ' = word could end here."""
        lo, hi = self._range(prefix)
        counts = defaultdict(float)
        n = len(prefix)
        for w in self.sorted_words[lo:hi]:
            f = self.freq[w]
            counts[w[n] if len(w) > n else " "] += f
        total = sum(counts.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in counts.items()}

    @lru_cache(maxsize=4096)
    def completions(self, prefix, limit=6):
        lo, hi = self._range(prefix)
        ws = [w for w in self.sorted_words[lo:hi] if w != prefix]
        ws.sort(key=lambda w: -self.freq[w])
        return tuple(ws[:limit])


if __name__ == "__main__":
    import time
    t = time.time()
    v = Vocab()
    print(f"loaded {len(v.freq)} words in {time.time() - t:.2f}s")
    for s in ["stop", "the", "and", "letter", "piano", "ot", "eht"]:
        print(s, "->", [w for w, _ in v.candidates(set(s))])
    print("next after 'th':", sorted(v.next_letters("th").items(), key=lambda x: -x[1])[:5])
    print("complete 'pia':", v.completions("pia"))
