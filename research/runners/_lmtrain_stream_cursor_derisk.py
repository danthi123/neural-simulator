"""De-risk: resumable streaming data cursor. A huge corpus won't fit in RAM, so we stream length-T sequences from a
flat (memmap-able) token array with a PERSISTENT cursor {shuffle-epoch, position}. Across a restart the stream MUST
continue from the exact next sequence (no re-seeing seen data, no skipping) — the data-side analog of resume-correctness.
Tested against an uninterrupted stream."""
import numpy as np


class TokenStream:
    """Deterministic, resumable stream of length-T sequences over a flat token array (per-epoch reshuffle)."""
    def __init__(self, tokens, T, batch, seed=0):
        self.tokens, self.T, self.batch, self.seed = tokens, T, batch, seed
        self.n_seq = (len(tokens) - 1) // T
        self.pos, self.epoch = 0, 0
        self._reshuffle()

    def _reshuffle(self):
        self.order = np.random.default_rng(self.seed + self.epoch).permutation(self.n_seq)

    def next_batch(self):
        idx = []
        for _ in range(self.batch):
            if self.pos >= self.n_seq:
                self.epoch += 1; self.pos = 0; self._reshuffle()   # new epoch: reshuffle deterministically
            idx.append(int(self.order[self.pos])); self.pos += 1
        return np.stack([self.tokens[i * self.T:(i + 1) * self.T] for i in idx])

    def state(self):
        return {"pos": self.pos, "epoch": self.epoch, "seed": self.seed}

    def load_state(self, s):
        self.seed, self.epoch, self.pos = s["seed"], s["epoch"], s["pos"]
        self._reshuffle()


if __name__ == "__main__":
    tokens = np.random.default_rng(0).integers(0, 8000, size=100000)      # stand-in for the tokenized corpus
    T, B = 64, 8
    # uninterrupted 20-batch reference (spanning >1 epoch to test the reshuffle boundary)
    s1 = TokenStream(tokens, T, B, seed=42); un = [s1.next_batch() for _ in range(20)]
    # interrupted: 10 batches -> save cursor -> FRESH stream (restart) -> restore -> 10 more
    s2 = TokenStream(tokens, T, B, seed=42); [s2.next_batch() for _ in range(10)]
    cursor = s2.state()
    s3 = TokenStream(tokens, T, B, seed=42); s3.load_state(cursor)
    resumed = [s3.next_batch() for _ in range(10)]
    ok = [np.array_equal(un[10 + i], resumed[i]) for i in range(10)]
    # also force an epoch rollover to test the reshuffle-on-resume
    small = np.random.default_rng(1).integers(0, 8000, size=(50 * T + 1))  # only 50 seqs -> rolls over fast
    e1 = TokenStream(small, T, 8, seed=7); ref = [e1.next_batch() for _ in range(20)]  # 160 draws over 50 seqs = ~3 epochs
    e2 = TokenStream(small, T, 8, seed=7); [e2.next_batch() for _ in range(9)]; c = e2.state()
    e3 = TokenStream(small, T, 8, seed=7); e3.load_state(c); res = [e3.next_batch() for _ in range(11)]
    ok2 = [np.array_equal(ref[9 + i], res[i]) for i in range(11)]
    print(f"resume across restart:        {sum(ok)}/10 batches match uninterrupted")
    print(f"resume across EPOCH rollover:  {sum(ok2)}/11 batches match (reshuffle-on-resume correct)")
    print("CURSOR-RESUMABLE (data stream continues exactly across restarts + epoch boundaries)"
          if all(ok) and all(ok2) else "CURSOR-BROKEN (stream diverges on resume)")
