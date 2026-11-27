# Capability: Self and identity
# Record: brain/research/2026-11-26-source-tag-prereg-DRAFT.md
# Expected-fail: the self/other source tag is missing (PREREQUISITES Self and identity, NOT-READY); this is a minimal one
"""Source tag: which tokens the brain heard from the user.

The host's chat channel already marks a heard turn with "User:" (perception, declared in SCAFFOLDS.md). A token is
tagged when it lies inside such a turn: from "User:" up to and including the blank line that closes it, or the end of
the text. This is not pronoun logic and no text is rewritten; only the learned or sham vector in
PlasticCortex.source_tag reads the mask."""
from __future__ import annotations

import re

import torch

_SPAN = re.compile(rb"User:.*?(?:\n\n|\Z)", re.S)


def _bytes(pipe, t: int) -> bytes:
    tok = getattr(pipe, "tokenizer", None)
    table = getattr(tok, "idx2token", None)
    if table is not None:
        b = table[t]
        return b if isinstance(b, bytes) else str(b).encode()
    return pipe.decode([t]).encode()


def user_mask(pipe, ids) -> list:
    """One bool per token: inside a "User:" turn of this call's text (spans do not carry across calls)."""
    pieces = [_bytes(pipe, int(t)) for t in ids]
    starts, pos = [], 0
    for p in pieces:
        starts.append(pos)
        pos += len(p)
    spans = [(m.start(), m.end()) for m in _SPAN.finditer(b"".join(pieces))]
    return [any(a <= s < b for a, b in spans) for s in starts]


def attach(plastic, vec: torch.Tensor, layer: int, learn: bool):
    """Put a source tag on a PlasticCortex: learned (a Parameter, so sleep's optimizer trains it) or fixed (sham)."""
    dev, dt = plastic.z["emb.weight"].device, plastic.z["emb.weight"].dtype
    v = vec.detach().to(device=dev, dtype=dt).clone()
    plastic.source_tag = torch.nn.Parameter(v) if learn else v
    plastic.tag_layer = int(layer)
    return plastic.source_tag


def detach(plastic):
    """Remove the tag (the LESION arm): returns the vector it held."""
    v = plastic.source_tag
    plastic.source_tag = None
    return None if v is None else v.detach().clone()
