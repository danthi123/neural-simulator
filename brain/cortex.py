"""Cortex: the pretrained recurrent language network (RWKV-7 G1) whose state is the brain's moment-to-moment activity.

State layout (rwkv package, v7): per layer l, state[3l] = attention token-shift (C,), state[3l+1] = matrix memory
(H, N, N) written by the delta rule (fast synaptic memory), state[3l+2] = channel-mix token-shift (C,).
"""
from __future__ import annotations

import copy
import os
import re

import torch

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "0")

from rwkv.model import RWKV  # noqa: E402
from rwkv.utils import PIPELINE  # noqa: E402

MODEL_DIR = os.environ.get("BRAIN_MODEL_DIR", "/home/dant123/models/rwkv7")
# The cortex size is configuration (brain/research/2026-11-03-adopt-1.5b-plan.md step 3). The default stays 0.1B until
# the six-seed 1.5B sleep result is in (that plan's decision rule).
DEFAULT_MODEL = os.environ.get("BRAIN_CORTEX_MODEL", os.path.join(MODEL_DIR, "rwkv7-g1d-0.1b-20260129-ctx8192.pth"))
END_OF_TEXT = 0
_SIZE = re.compile(r"-(\d+(?:\.\d+)?b)-", re.I)


def model_size(model_path: str | None) -> str | None:
    """The size tag of a checkpoint ("0.1b", "0.4b", "1.5b") read from its file name; None if it carries none."""
    m = _SIZE.search(os.path.basename(model_path or ""))
    return m.group(1).lower() if m else None


def resolve_model(model: str | None) -> str:
    """A --model value -> a checkpoint path. Accepts a path (with or without .pth) or a size tag ("1.5b", "0.4B")
    looked up in MODEL_DIR. None/"" -> DEFAULT_MODEL. A tag matching no file, or several, is an error."""
    if not model:
        return DEFAULT_MODEL
    if os.sep in model or model.endswith(".pth"):
        return model if model.endswith(".pth") else model + ".pth"
    tag = model.lower()
    files = os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else []
    hits = sorted(f for f in files if f.endswith(".pth") and model_size(f) == tag)
    if len(hits) != 1:
        raise ValueError(f"--model {model!r}: {len(hits)} checkpoints of that size in {MODEL_DIR}: {hits}")
    return os.path.join(MODEL_DIR, hits[0])


def model_name(model_path: str) -> str:
    """The name a brain home records for its cortex: the checkpoint's file name without .pth."""
    base = os.path.basename(model_path)
    return base[:-4] if base.endswith(".pth") else base


class Cortex:
    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "cpu"):
        model_path = resolve_model(model_path)
        self.model_path = model_path
        self.model_name = model_name(model_path)   # recorded in a brain home: Brain.meta["cortex_model"]
        strategy = "cuda fp16" if device == "cuda" else "cpu fp32"
        self.model = RWKV(model=model_path[:-4] if model_path.endswith(".pth") else model_path, strategy=strategy)
        self.pipe = PIPELINE(self.model, "rwkv_vocab_v20230424")
        self.n_layers = self.model.args.n_layer

    def read(self, text: str, state=None):
        """Process text; returns (logits for the next token, new state). Never mutates the given state."""
        st = copy.deepcopy(state) if state is not None else None
        return self.forward(self.pipe.encode(text), st)

    def forward(self, tokens, state):
        """One pass of the network. The brain's live cortex (brain/brain.py LiveCortex) overrides this to run through
        the consolidated slow synapses; here it is the pretrained package forward."""
        return self.model.forward(tokens, state)

    def speak(self, logits, state, max_tokens: int = 120, temperature: float = 1.0, top_p: float = 0.3,
              presence: float = 0.4, frequency: float = 0.4):
        """Generate one turn from (logits, state). Stops at a blank line or end of text. Returns (text, state)."""
        st = copy.deepcopy(state)
        out, counts = [], {}
        logits = logits.clone()
        for _ in range(max_tokens):
            for t, c in counts.items():
                logits[t] -= presence + c * frequency
            tok = int(torch.argmax(logits)) if temperature <= 0 else \
                self.pipe.sample_logits(logits, temperature=temperature, top_p=top_p)
            if tok == END_OF_TEXT:
                break
            out.append(tok)
            counts[tok] = counts.get(tok, 0) + 1
            text = self.pipe.decode(out)
            if "\n\n" in text or "\nUser:" in text:
                break
            logits, st = self.forward([tok], st)
        text = self.pipe.decode(out).split("\n\n")[0].split("\nUser:")[0].strip()
        return text, st

    def matrix_delta(self, state, base):
        """The change an experience made to fast synaptic memory: per-layer matrix-memory deltas (None elsewhere)."""
        return [(state[i] - base[i]).clone() if i % 3 == 1 else None for i in range(len(base))]

    @staticmethod
    def key_of(delta, layer: int = 0):
        """Content key used for pattern completion: the first layer's associative-matrix delta, flattened."""
        return delta[layer * 3 + 1].flatten().float()

    @staticmethod
    def add_delta(state, delta, alpha: float):
        st = [x.clone() for x in state]
        for i, d in enumerate(delta):
            if d is not None:
                st[i] = st[i] + alpha * d.to(st[i].device, st[i].dtype)   # stored episodes live on CPU
        return st
