# Capability: Self and identity
# Record: brain/research/2026-11-27-source-monitor-port-prereg-DRAFT.md; research/findings/2026-08-01-agency-authorship-tag-corollary-discharge-source-monitor-6seed-GO.md; brain/experiments/results/2026-11-26-source-tag-read.md
# Expected-fail: the self/other tag is host-marked by provenance (what the cortex emitted), not a comparator the cortex learned; a GO is about 10-15% (prereg section 0); the fact may be stored conditional on the tag (knows falls)
"""Source monitor: an efference-copy tag on what the brain itself produced, absent on what it heard.

Pre-pivot (research/findings/2026-08-01-agency-authorship-...-GO.md, 2026-08-03 co-residency calibration): self-production
emits a corollary discharge that heard content lacks; a comparator reads "was an efference copy present?" and the
content does not decide the source (tag independent of content). Here the carrier is one learned vector added to the
residual after block `tag_layer` on every position the brain itself produced:
  - every token a registered own text holds (the keep bank: the frozen cortex's own samples, plus the host-typed turn
    label that asked it to continue);
  - every token fed back after the cortex sampled it (a one-token forward call during speaking);
  - every "Assistant:" turn of a text read in one call (the turn-open label at the moment of speaking, a committed
    reply, a seeded self-statement), from the label to the blank line or the end.
Heard turns ("User: ...", told episodes) carry nothing. The mask is a physical provenance record, not pronoun logic and
not a rewrite; the vector is learned in sleep (the optimizer trains it with the adapters) or fixed (SHAM).

What is new against the killed tag (source_tag.py, 11-27 NO-ENGAGEMENT): there the tag sat on the heard side, which is
the side the replay teaches and the keep KL also covers, so no data separated the two sources. Here the keep texts and the
brain's own turns carry the tag and every told telling does not, with the same surface form ("User: ..." text) on both
sides, so only provenance differs. Off by default: no tag attached (PlasticCortex.source_tag None) is byte-identical."""
from __future__ import annotations

import contextlib
import re

import torch

from brain import source_tag as ST

_ASSISTANT = re.compile(rb"Assistant:.*?(?:\n\n|\Z)", re.S)
KINDS = ("own", "all")


class Provenance:
    """Which token sequences the brain itself produced, and the mask function PlasticCortex reads.

    kind "own": the efference mask above. kind "all": every position (the no-contrast control: a global bias vector).
    `stats` counts tagged and total tokens over every masked call (the engagement record)."""

    def __init__(self, kind: str = "own"):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
        self.kind = kind
        self.own: set = set()
        self.stats = {"calls": 0, "tokens": 0, "tagged": 0, "registered_hits": 0}

    def register(self, pipe, texts) -> int:
        """Texts the brain produced (a keep bank): every token of them is own. Returns how many were added."""
        n0 = len(self.own)
        for t in texts:
            self.own.add(tuple(pipe.encode(t if isinstance(t, str) else t[0])))
        return len(self.own) - n0

    def mask(self, pipe, ids) -> list:
        ids = [int(t) for t in ids]
        if self.kind == "all":
            out = [True] * len(ids)
        elif tuple(ids) in self.own:
            out = [True] * len(ids)
            self.stats["registered_hits"] += 1
        elif len(ids) == 1:   # a token fed back after it was sampled: its own efference copy
            out = [True]
        else:
            out = _assistant_mask(pipe, ids)
        self.stats["calls"] += 1
        self.stats["tokens"] += len(ids)
        self.stats["tagged"] += sum(out)
        return out


def _assistant_mask(pipe, ids) -> list:
    pieces = [ST._bytes(pipe, t) for t in ids]
    starts, pos = [], 0
    for p in pieces:
        starts.append(pos)
        pos += len(p)
    spans = [(m.start(), m.end()) for m in _ASSISTANT.finditer(b"".join(pieces))]
    return [any(a <= s < b for a, b in spans) for s in starts]


def attach(plastic, vec: torch.Tensor, layer: int, learn: bool, prov: Provenance):
    """Put the efference tag on a PlasticCortex: learned (a Parameter) or fixed (sham). Returns the vector."""
    v = ST.attach(plastic, vec, layer, learn)
    plastic.tag_mask_fn = prov.mask
    return v


def detach(plastic):
    """Remove the tag and its mask (the LESION arm). Returns the vector it held."""
    v = ST.detach(plastic)
    plastic.tag_mask_fn = None
    return v


class _Channel:
    """Stands in for brain.source_tag inside brain.experiments.voice_tag so its sleep/battery/onset machinery attaches the
    efference tag instead of the user-turn tag, with no edit to that harness."""

    def __init__(self, prov: Provenance):
        self.prov = prov

    def attach(self, plastic, vec, layer, learn):
        return attach(plastic, vec, layer, learn, self.prov)

    @staticmethod
    def detach(plastic):
        return detach(plastic)


@contextlib.contextmanager
def channel(voice_tag_module, prov: Provenance):
    """While active, voice_tag.ST is the efference channel; restored on exit (also on error)."""
    was = voice_tag_module.ST
    voice_tag_module.ST = _Channel(prov)
    try:
        yield prov
    finally:
        voice_tag_module.ST = was
