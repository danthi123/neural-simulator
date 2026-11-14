"""Hippocampus: one-shot episodic memory stored as the change an experience made to cortical fast memory.

Recall is pattern completion from the current cue (the cue's own cortical trace), centred against the average stored
trace (feedforward inhibition), and the winner is reinstated INTO the cortex state; stored text is kept only for
inspection and is never put back into the conversation. Strength grows with use and salience and decays with time
(power law), so often-used memories become near-permanent while rarely used detail fades.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch


@dataclass
class Episode:
    text: str                      # for inspection only; never fed back to the cortex
    delta: list                    # per-layer matrix-memory deltas (fp16), None for token-shift slots
    key: torch.Tensor              # pattern-completion key (layer-0 associative delta)
    strength: float = 1.0
    created: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    uses: int = 0
    permanent: bool = False        # identity-level memories (own name, friends' names)
    source: str = "other"          # source tag: "self" = the episode holds the brain's own words (lived reply)


class Hippocampus:
    # Decay: effective strength = strength * (1 + age_days / TAU) ** -DECAY; recall adds USE_BOOST (spaced use counts).
    TAU_DAYS, DECAY, USE_BOOST = 1.0, 0.5, 0.5
    FAMILIARITY = 0.12             # minimum centred similarity for a cued recall (a question) to fire
    FAMILIARITY_SPONTANEOUS = 0.45  # a plain statement must be much more familiar to pull a memory back unasked
    MARGIN = 0.02                  # winner must beat the runner-up by this much
    GAIN = 4.0  # reinstatement gain (grid 11-02: 4.5 recalls 8/8 on fresh facts; identity Q/A traces loop above ~4.5)

    def __init__(self):
        self.episodes: list[Episode] = []
        self.enabled = True

    def store(self, cortex, base, text: str, speaker: str = "User", salience: float = 1.0, permanent: bool = False,
              key_text: str | None = None):
        """key_text (default None = text): what the pattern-completion key is read from. The reinstated content (delta)
        is always read from text; a lived episode is indexed by what was heard and holds the whole exchange."""
        _, st = cortex.read(f"{speaker}: {text}\n\n", base)
        delta = cortex.matrix_delta(st, base)
        if key_text is not None and key_text != text:
            _, kst = cortex.read(f"{speaker}: {key_text}\n\n", base)
            key = cortex.key_of(cortex.matrix_delta(kst, base)).cpu()
        else:
            key = cortex.key_of(delta).cpu()
        # Episodes are kept on the CPU (host RAM / disk), not on the cortex's device: 6.5 MiB each at 1.5B
        # (adopt-1.5b plan step 5). Reinstatement moves the delta to the state's device (Cortex.add_delta).
        ep = Episode(text=f"{speaker}: {text}", delta=[d.half().cpu() if d is not None else None for d in delta],
                     key=key, strength=salience, permanent=permanent)
        self.episodes.append(ep)
        return ep

    def effective_strength(self, ep: Episode, now: float | None = None) -> float:
        if ep.permanent:
            return max(ep.strength, 3.0)
        age_days = ((now or time.time()) - ep.last_used) / 86400.0
        return ep.strength * (1.0 + age_days / self.TAU_DAYS) ** (-self.DECAY)

    def familiarity(self, cortex, base, cue_text: str, speaker: str = "User", exclude=()) -> float:
        """How familiar the cue is against every stored episode (the best recall score), with no side effects: the
        "have I been told about this" read-out, separate from recollection itself. 0.0 when nothing is stored.
        `exclude`: episode indices left out of the read-out (e.g. a taught self-belief, which is not about a topic)."""
        _, score = self.best(cortex, base, cue_text, speaker, exclude)
        return score

    def best(self, cortex, base, cue_text: str, speaker: str = "User", exclude=()):
        """The best-scoring episode for a cue and its score, with no bar, no margin and no side effects (forced top-1).
        Returns (None, 0.0) when nothing is stored or every episode is excluded."""
        keep = [i for i in range(len(self.episodes)) if i not in set(exclude)]
        if not self.enabled or not keep:
            return None, 0.0
        score = self.scores(cortex, base, cue_text, speaker)[keep]
        j = int(score.argmax())
        return self.episodes[keep[j]], float(score[j])

    def recall(self, cortex, base, cue_text: str, speaker: str = "User", question: bool = True):
        """Pattern completion from a cue. Returns (episode, similarity) or (None, best_similarity)."""
        if not self.enabled or not self.episodes:
            return None, 0.0
        now = time.time()
        score = self.scores(cortex, base, cue_text, speaker, now)
        return self._compete(score, question, now)

    def similarities(self, cortex, base, cue_text: str, speaker: str = "User"):
        """Centred cosine of the cue's key against every stored key (one value per episode)."""
        _, st = cortex.read(f"{speaker}: {cue_text}\n\n", base)
        keys = torch.stack([e.key for e in self.episodes])
        q = cortex.key_of(cortex.matrix_delta(st, base)).to(keys.device)
        if len(self.episodes) > 1:
            mu = keys.mean(0)
            keys, q = keys - mu, q - mu
        keys = keys / keys.norm(dim=1, keepdim=True).clamp_min(1e-8)
        q = q / q.norm().clamp_min(1e-8)
        return keys @ q

    def scores(self, cortex, base, cue_text: str, speaker: str = "User", now: float | None = None):
        """Strength-weighted similarities: what recall competes on. Does not change any episode."""
        sims = self.similarities(cortex, base, cue_text, speaker)
        # weaker memories are harder to bring back: weight similarity by a saturating function of strength
        now = now or time.time()
        w = torch.tensor([math.tanh(self.effective_strength(e, now)) for e in self.episodes])
        return sims * w

    def _compete(self, score, question: bool, now: float):
        order = torch.argsort(score, descending=True)
        best = int(order[0])
        runner = float(score[order[1]]) if len(order) > 1 else -1.0
        bar = self.FAMILIARITY if question else self.FAMILIARITY_SPONTANEOUS
        if float(score[best]) < bar or float(score[best]) - runner < self.MARGIN:
            return None, float(score[best])
        ep = self.episodes[best]
        ep.uses += 1
        ep.strength += self.USE_BOOST
        ep.last_used = now
        return ep, float(score[best])

    def reinstate(self, cortex, state, ep: Episode):
        # Strength sets how easily a memory comes back (recall), not how hard it is reinstated.
        return cortex.add_delta(state, ep.delta, self.GAIN)

    def nbytes(self) -> dict:
        """Memory accounting: bytes the stored episodes hold, by device ({"cpu": ..., "cuda:0": ...})."""
        out: dict = {}
        for ep in self.episodes:
            for t in (*ep.delta, ep.key, *(getattr(ep, "activity", None) or [])):
                if isinstance(t, torch.Tensor):
                    out[str(t.device)] = out.get(str(t.device), 0) + t.numel() * t.element_size()
        return out


def episode_bytes_for(n_layers: int, n_head: int, head_size: int, key_layers: int = 1) -> int:
    """Bytes one default episode holds for a cortex of this shape: fp16 matrix-memory deltas of every layer plus the
    fp32 key (the layer-0 associative delta). 0.1B 1.3 MiB, 0.4B 3.3 MiB, 1.5B 6.5 MiB (adopt-1.5b plan table)."""
    per_layer = n_head * head_size * head_size
    return n_layers * per_layer * 2 + key_layers * per_layer * 4
