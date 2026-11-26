"""
Local AI (Ollama) for Piano Typer.

Instead of chatting with the model (slow: seconds per answer on a laptop CPU), we
feed it the text typed so far as a raw completion and read its probabilities for
the very next token. One forward pass (~0.3-0.6 s, prompt cache makes it cheap)
gives both:
  choose(context, candidates)  -> which anagram the chord meant ("stop" vs "pots")
  next_words(context, prefix)  -> likely next words for the suggestion chips

The app works without this (word frequencies only).
"""

import json
import math
import time
import urllib.request

OLLAMA = "http://127.0.0.1:11434"  # not "localhost": it resolves to IPv6 first here and times out
PREFERRED = ["qwen2.5:7b-instruct", "qwen2.5:3b", "qwen2.5:1.5b", "llama3.2:3b", "phi3.5"]
FRAME = "Here is a casual message a friend typed on their phone:\n\n"
OPTIONS = {"temperature": 0, "num_predict": 1, "num_ctx": 1024}  # keep identical across calls or Ollama reloads the model


def _post(path, body, timeout):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def list_models():
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=8) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
        return [n for n in names if "embed" not in n and "coder" not in n]
    except Exception:
        return []


def pick_default(models):
    for p in PREFERRED:
        for m in models:
            if m == p or m.split(":")[0] == p or m == p + ":latest":
                return m
    return models[0] if models else None


class AI:
    def __init__(self, model, vocab=None):
        self.model = model
        self.vocab = vocab
        self.last_latency_ms = None
        self._cache = {}

    def _next_tokens(self, text, timeout=120):
        """[(token, prob)] for the token right after `text` (top 20)."""
        t = time.perf_counter()
        r = _post("/api/generate", {
            "model": self.model, "prompt": text, "raw": True, "stream": False,
            "logprobs": True, "top_logprobs": 20, "keep_alive": "30m", "options": OPTIONS,
        }, timeout)
        self.last_latency_ms = int((time.perf_counter() - t) * 1000)
        lp = (r.get("logprobs") or [{}])[0].get("top_logprobs", [])
        return [(x["token"], math.exp(x["logprob"])) for x in lp]

    def warmup(self):
        self._next_tokens("Hello", timeout=180)

    @staticmethod
    def _context(text, n=400):
        """Trim to the last n chars (at a word boundary) and drop the trailing space:
        tokenizers attach the space to the next word (" stop")."""
        text = text[-n:]
        if len(text) == n and " " in text:
            text = text[text.index(" ") + 1:]
        return text.rstrip(" ") or ""

    def _word_starts(self, context):
        """{lowercase piece: prob} for pieces that would start the next word (cached per context)."""
        ctx = self._context(context)
        if ctx in self._cache:
            return self._cache[ctx]
        # Framing as a typed message stops instruct models from predicting "assistant" words.
        toks = self._next_tokens(FRAME + ctx)
        starts = {}
        for tok, p in toks:
            if ctx and not ctx.endswith("\n") and not tok.startswith(" "):
                continue  # would glue onto the previous word
            piece = tok.strip().lower()
            if piece.isalpha() or piece.replace("'", "").isalpha():
                starts[piece] = starts.get(piece, 0) + p
        if len(self._cache) > 200:
            self._cache.clear()
        self._cache[ctx] = starts
        return starts

    def choose(self, context, candidates):
        best = self._choose_by_continuation(context, candidates)
        if best is None:
            best = self._choose_by_question(context, candidates)
        return best

    def _choose_by_question(self, context, candidates):
        """Fallback: multiple-choice question; read which option's first token is likeliest."""
        ctx = self._context(context)
        prompt = ("<|im_start|>user\nSomeone is typing a message. Which word did they type next?\n"
                  f"Message: \"{ctx} ___\"\nOptions: {', '.join(candidates)}\n"
                  "Reply with just the word.<|im_end|>\n<|im_start|>assistant\n") if "qwen" in self.model else (
                  f"Message: \"{ctx} ___\"\nWhich of these words fills the blank: {', '.join(candidates)}?\n"
                  "The word is:")
        toks = self._next_tokens(prompt)
        best, best_p = None, 0.0
        for tok, p in toks:
            piece = tok.strip().lower()
            if not piece:
                continue
            for w in candidates:
                if w.startswith(piece) and p > best_p:
                    # several candidates can share a first piece; keep the earlier (more common) one
                    best, best_p = w, p
                    break
        # this fallback is much weaker than reading the continuation: only trust it when sure
        return best if best_p > 0.7 else None

    def _choose_by_continuation(self, context, candidates):
        starts = self._word_starts(context)
        best, best_score = None, 0.0
        for i, w in enumerate(candidates):
            score = 0.0
            for piece, p in starts.items():
                if w == piece:
                    score = max(score, p * 1.5)      # whole word is one token: strong evidence
                elif w.startswith(piece):
                    score = max(score, p)
            score *= 1.0 - 0.02 * i                  # tie-break toward the more common word
            if score > best_score:
                best, best_score = w, score
        return best

    def next_words(self, context, prefix="", n=5):
        """Likely next words. With a prefix ("pia"), filter the same prediction
        (asking the model mid-word gives poor answers because of tokenisation)."""
        starts = self._word_starts(context)
        scored = {}
        for piece, p in starts.items():
            if prefix and not (piece.startswith(prefix) or prefix.startswith(piece)):
                continue
            if self.vocab is None or piece in self.vocab:
                if piece.startswith(prefix) and piece != prefix:
                    scored[piece] = scored.get(piece, 0) + p
            if self.vocab is not None and (prefix or piece not in self.vocab):
                # word-piece like "pian" or a longer word: complete it from the word list
                stem = piece if piece.startswith(prefix) else prefix
                for w in self.vocab.completions(stem, 2):
                    if w != prefix and w not in scored:
                        scored[w] = p * 0.5
        return [w for w, _ in sorted(scored.items(), key=lambda kv: -kv[1])][:n]


if __name__ == "__main__":
    from words import Vocab
    v = Vocab()
    models = list_models()
    print("models:", models, "default:", pick_default(models))
    tests_choose = [
        ("Please ", ["stop", "post", "spot", "tops", "pots"]),
        ("I will put the letter in the ", ["post", "stop", "spot", "pots"]),
        ("We cooked dinner with pots and ", ["pans", "snap", "naps", "span"]),
        ("Turn ", ["on", "no"]),
        ("I want to ", ["go", "ego"]),
    ]
    for m in models[:1]:
        ai = AI(m, v)
        t = time.time()
        ai.warmup()
        print(f"\n{m}: warmup {time.time() - t:.1f}s")
        for ctx, c in tests_choose:
            print(f"  choose {ctx!r:40} -> {ai.choose(ctx, c)!s:6} {ai.last_latency_ms} ms")
        for ctx, p in [("I want to learn how to play the ", ""), ("Thank you so ", "m"),
                       ("Thank you so much for ", ""), ("I love playing the pia", "")]:
            # last one: prefix inside the context, split it like the server does
            if not p and not ctx.endswith(" "):
                ctx, p = ctx.rsplit(" ", 1)[0] + " ", ctx.rsplit(" ", 1)[1]
            print(f"  next   {ctx + p!r:40} -> {ai.next_words(ctx, p)} {ai.last_latency_ms} ms")
