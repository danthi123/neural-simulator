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

    def store(self, cortex, base, text: str, speaker: str = "User", salience: float = 1.0, permanent: bool = False):
        _, st = cortex.read(f"{speaker}: {text}\n\n", base)
        delta = cortex.matrix_delta(st, base)
        ep = Episode(text=f"{speaker}: {text}", delta=[d.half() if d is not None else None for d in delta],
                     key=cortex.key_of(delta), strength=salience, permanent=permanent)
        self.episodes.append(ep)
        return ep

    def effective_strength(self, ep: Episode, now: float | None = None) -> float:
        if ep.permanent:
            return max(ep.strength, 3.0)
        age_days = ((now or time.time()) - ep.last_used) / 86400.0
        return ep.strength * (1.0 + age_days / self.TAU_DAYS) ** (-self.DECAY)

    def recall(self, cortex, base, cue_text: str, speaker: str = "User", question: bool = True):
        """Pattern completion from a cue. Returns (episode, similarity) or (None, best_similarity)."""
        if not self.enabled or not self.episodes:
            return None, 0.0
        _, st = cortex.read(f"{speaker}: {cue_text}\n\n", base)
        q = cortex.key_of(cortex.matrix_delta(st, base))
        keys = torch.stack([e.key for e in self.episodes])
        if len(self.episodes) > 1:
            mu = keys.mean(0)
            keys, q = keys - mu, q - mu
        keys = keys / keys.norm(dim=1, keepdim=True).clamp_min(1e-8)
        q = q / q.norm().clamp_min(1e-8)
        sims = keys @ q
        # weaker memories are harder to bring back: weight similarity by a saturating function of strength
        now = time.time()
        w = torch.tensor([math.tanh(self.effective_strength(e, now)) for e in self.episodes])
        score = sims * w
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
