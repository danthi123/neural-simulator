"""Cortex: the pretrained recurrent language network (RWKV-7 G1) whose state is the brain's moment-to-moment activity.

State layout (rwkv package, v7): per layer l, state[3l] = attention token-shift (C,), state[3l+1] = matrix memory
(H, N, N) written by the delta rule (fast synaptic memory), state[3l+2] = channel-mix token-shift (C,).
"""
from __future__ import annotations

import copy
import os

import torch

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "0")

from rwkv.model import RWKV  # noqa: E402
from rwkv.utils import PIPELINE  # noqa: E402

DEFAULT_MODEL = os.environ.get("BRAIN_CORTEX_MODEL", "/home/dant123/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth")
END_OF_TEXT = 0


class Cortex:
    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "cpu"):
        strategy = "cuda fp16" if device == "cuda" else "cpu fp32"
        self.model = RWKV(model=model_path[:-4] if model_path.endswith(".pth") else model_path, strategy=strategy)
        self.pipe = PIPELINE(self.model, "rwkv_vocab_v20230424")
        self.n_layers = self.model.args.n_layer

    def read(self, text: str, state=None):
        """Process text; returns (logits for the next token, new state). Never mutates the given state."""
        st = copy.deepcopy(state) if state is not None else None
        return self.model.forward(self.pipe.encode(text), st)

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
            logits, st = self.model.forward([tok], st)
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
                st[i] = st[i] + alpha * d.to(st[i].dtype)
        return st
