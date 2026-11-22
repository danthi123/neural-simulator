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
    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "cpu", cuda_precision: str = "fp16"):
        model_path = resolve_model(model_path)
        self.model_path = model_path
        self.model_name = model_name(model_path)   # recorded in a brain home: Brain.meta["cortex_model"]
        # cuda_precision (switch, default "fp16" = unchanged): "fp32" runs the speaking path in full precision on the GPU,
        # matching the CPU's fp32 numerics (the sleep's PlasticCortex already trains in fp32 on every device).
        if cuda_precision not in ("fp16", "fp32"):
            raise ValueError(f"cuda_precision must be fp16 or fp32, not {cuda_precision!r}")
        strategy = f"cuda {cuda_precision}" if device == "cuda" else "cpu fp32"
        self.strategy = strategy
        self.model = RWKV(model=model_path[:-4] if model_path.endswith(".pth") else model_path, strategy=strategy)
        self.pipe = PIPELINE(self.model, "rwkv_vocab_v20230424")
        self.n_layers = self.model.args.n_layer

    def read(self, text: str, state=None):
        """Process text; returns (logits for the next token, new state). Never mutates the given state."""
        st = copy.deepcopy(state) if state is not None else None
        return self.forward(self.pipe.encode(text), st)

    def read_nll(self, text: str, state=None):
        """Process text and keep the cortex's own prediction error: returns (last logits, new state, per-token NLL in
        nats as a 1-D tensor of len(tokens)-1, entry t = -log p(token t+1 | tokens <= t)). One forward pass, the same
        call as brain/experiments/novelty_stage0.item_nll with full output, so the state and last logits are those of
        read() (checked at <= 1e-5 by arousal_gain smoke S1). Never mutates the given state."""
        ids = self.pipe.encode(text)
        st = copy.deepcopy(state) if state is not None else None
        with torch.no_grad():
            logits, new = self.model.forward(ids, st, full_output=True)
        lp = torch.log_softmax(logits.float(), -1)
        nll = -lp[:-1].gather(1, torch.tensor(ids[1:], device=lp.device)[:, None])[:, 0]
        return logits[-1], new, nll

    def forward(self, tokens, state):
        """One pass of the network. The brain's live cortex (brain/brain.py LiveCortex) overrides this to run through
        the consolidated slow synapses; here it is the pretrained package forward."""
        return self.model.forward(tokens, state)

    def speak(self, logits, state, max_tokens: int = 120, temperature: float = 1.0, top_p: float = 0.3,
              presence: float = 0.4, frequency: float = 0.4, block: dict | None = None, block_n: int = 4,
              carry: dict | None = None, decay: float = 0.996):
        """Generate one turn from (logits, state). Stops at a blank line or end of text. Returns (text, state).
        block (default None = off): {(n-1 token prefix): {next tokens}} that may not follow that prefix in this turn
        (no-repeat n-gram across turns; Brain(habituation=...)).
        carry (default None = off): a token-occurrence dict kept across turns by the caller and updated in place, each
        count decaying by `decay` per generated token (the RWKV chat practice, penalty_decay; Brain(carry_penalty))."""
        st = copy.deepcopy(state)
        out = []
        counts = carry if carry is not None else {}
        logits = logits.clone()
        for _ in range(max_tokens):
            for t, c in counts.items():
                logits[t] -= presence + c * frequency
            if block and len(out) >= block_n - 1:
                for t in block.get(tuple(out[-(block_n - 1):]), ()):
                    logits[t] = float("-inf")
            tok = int(torch.argmax(logits)) if temperature <= 0 else \
                self.pipe.sample_logits(logits, temperature=temperature, top_p=top_p)
            if tok == END_OF_TEXT:
                break
            out.append(tok)
            if carry is not None:
                for t in counts:
                    counts[t] *= decay
            counts[tok] = counts.get(tok, 0) + 1
            text = self.pipe.decode(out)
            if "\n\n" in text or "\nUser:" in text:
                break
            logits, st = self.forward([tok], st)
        text = self.pipe.decode(out).split("\n\n")[0].split("\nUser:")[0].strip()
        return text, st

    def ngram_block(self, texts, n: int = 4) -> dict:
        """{(n-1 prefix): {next}} for every token n-gram in texts (for speak(block=...))."""
        out = {}
        for t in texts:
            ids = self.pipe.encode(t)
            for i in range(len(ids) - n + 1):
                out.setdefault(tuple(ids[i:i + n - 1]), set()).add(ids[i + n - 1])
        return out

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
