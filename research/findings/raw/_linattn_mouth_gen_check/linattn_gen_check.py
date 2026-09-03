"""CPU-only, GPU-avoiding generation check for the linattn spiking mouth (board task, 2026-09-03).

MUST be launched with SIM_BACKEND=numpy already set in the environment BEFORE this process starts (get_backend()
caches its choice on first call within the process) -- see sim/backend.py's own module docstring. This script
does not touch cupy or the running 6-seed GPU training job; it only np.load()s already-saved checkpoint .npz
files and drives FewSpikeWordRead's tiny (topk*pop neuron) Izhikevich bank, which honors SIM_BACKEND=numpy.
"""
import os
import sys
import time

assert os.environ.get("SIM_BACKEND") == "numpy", "SIM_BACKEND=numpy must be set before this process starts"

REPO_ROOT = "/home/dant123/Projects/sim"          # main checkout -- bridges/wkv_ckpt/ lives here, not the worktree
sys.path.insert(0, "/home/dant123/Projects/sim/.claude/worktrees/agent-aed59f0b381164491")

import numpy as np  # noqa: E402

from webapp import wkv_mouth_generator as wmg  # noqa: E402
from research.runners._wkv_fewspike_read_derisk import WKVReadout, LinAttnReadout  # noqa: E402

os.environ["BRAIN_WKV_MOUTH_TOKENIZER"] = "bpe"

LINATTN_TMPL = REPO_ROOT + "/bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{seed}.npz"
SSM_BPE_TMPL = REPO_ROOT + "/bridges/wkv_ckpt/wkv_ssm_bpe8k_d192_simplewiki_depth2_contiguous_seed{seed}.npz"

PROMPTS = [
    "The sun is",
    "A dog is an animal that",
    "Water is made of",
    "Yesterday I went to the",
    "The city of London is",
    "In the beginning of the story",
    "The most important thing about science is",
    "My favorite food is",
    "The weather today",
    "Once upon a time there was",
]

GEN_KW = dict(max_new_tokens=60, topk=64, read_window=40, pop=8, gen_temp=0.8,
              repetition_penalty=1.0, no_repeat_ngram_size=0)


def run_family(label, ckpt_tmpl, recur, seeds):
    os.environ["BRAIN_WKV_MOUTH_RECURRENCE"] = recur
    wmg._CKPT_TEMPLATE = ckpt_tmpl
    wmg._CKPT_CACHE.clear()
    wmg._HEAD_INFO.clear()
    results = {}
    for seed in seeds:
        t0 = time.time()
        rows = []
        for p in PROMPTS:
            text, secs = wmg.generate(p, seed=seed, **GEN_KW)
            rows.append({"prompt": p, "text": text, "secs": secs})
            print(f"[{label} seed{seed}] {p!r} ({secs}s)\n    -> {text}\n", flush=True)
        results[seed] = rows
        print(f"=== {label} seed{seed} done in {time.time()-t0:.1f}s ===\n", flush=True)
    return results


if __name__ == "__main__":
    print("### LINATTN (new mouth, --recurrence linattn --n-layers 2, crossed trigram +0.049/+0.053) ###\n")
    linattn_results = run_family("LINATTN", LINATTN_TMPL, "linattn", [42, 43])

    print("\n### SSM/DUAL-NONNEG (same-arc BPE control, same corpus/d_model/depth2_contiguous, NO-GO -0.125) ###\n")
    ssm_results = run_family("SSM-BPE", SSM_BPE_TMPL, "ssm", [42, 43])

    import json
    out = {"linattn": linattn_results, "ssm_bpe_control": ssm_results, "gen_kwargs": GEN_KW, "prompts": PROMPTS}
    outpath = "/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad/linattn_gen_results.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {outpath}")
