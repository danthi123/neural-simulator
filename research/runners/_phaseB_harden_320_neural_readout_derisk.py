"""HARDEN the 320-concept stream cortex, piece 2 (the READ-OUT NORMALIZATION) — cheap-first (CPU): does the FULL
who-Q&A conversation (recall + the learned no-confab moat) SURVIVE swapping the host double-centring read-out for
the fully-on-brain NEURAL normalization (per-hub spike-frequency adaptation + per-concept feedforward inhibition,
with rate-coded-pool noise on the means)?

CONTEXT. The 320 stream cortex's per-concept code = double_center(log1p(M·100)). double_center is host arithmetic.
The brain-based read-out (`_phaseB_biologize_readout_norm_derisk.py`, GO) replaces it with two real cortical
gain-control ops (adaptation + feedforward inhibition), de-risked at 96% of host on the STRUCTURE (corr vs S_true,
6 seeds). The open question this closes BEFORE the ~96-min/seed GPU re-stream: 96% of the structure is not 100% —
does the slightly-different code still carry the CONVERSATION (who-Q&A recall 1.00 + the learned moat abstaining)?

This is CPU/instant because it uses the corpus co-occurrence PROXY (build_real_corpus, the SAME proxy the read-out
de-risk used, corr ~0.9 to the on-bridge stream-learned M) — so it de-risks the *integration* before committing
the real-stream GPU run. The definitive test (real stream M) is the single-seed 320 re-stream with
`--readout-norm neural`; this says whether that run is worth launching.

GATE (6 seeds): with the NEURAL read-out, who-Q&A recall stays >= the host recall (no binding loss) AND the learned
moat abstains with 0 false-accepts on every seed (the no-fabrication guarantee survives the on-brain read-out).
ANTI-CHEAT: same facts/queries for host vs neural; the moat is the learned gate (a-priori threshold, lesionable);
a no-normalization control is reported (must be far worse).

Reuse-by-import (build_real_corpus + neural_norm + double_center + the production run_conversation + the learned
moat). CPU/numpy, NO GPU.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_harden_320_neural_readout_derisk
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.learned_graded_cortex_fair_test import build_real_corpus  # noqa: E402
from research.runners._phaseB_biologize_readout_norm_derisk import neural_norm, double_center, N_HUB  # noqa: E402
from research.runners._phaseB_onbridge_stream_conversation_derisk import run_conversation  # noqa: E402


def _unit(code):
    return code / (np.linalg.norm(code, axis=1, keepdims=True) + 1e-12)


def run_seed(seed):
    C, labels, _S = build_real_corpus(seed, N_HUB)
    L = np.log1p(C * 100.0)
    rng = np.random.RandomState(seed * 911 + 7)
    host_codes = _unit(double_center(L))
    neural_codes = _unit(neural_norm(L, rng))
    nonorm_codes = _unit(L)                                  # the no-normalization control
    labels = np.asarray(labels)
    rh = run_conversation(host_codes, labels, seed, moat="learned")
    rn = run_conversation(neural_codes, labels, seed, moat="learned")
    rz = run_conversation(nonorm_codes, labels, seed, moat="learned")
    print(f"  [neural read-out seed {seed}] {C.shape[0]}c | HOST recall {rh['recall']:.2f} abstain {rh['abstain']:.2f}"
          f"(fa {rh['false_accept']}) | NEURAL recall {rn['recall']:.2f} abstain {rn['abstain']:.2f}"
          f"(fa {rn['false_accept']}) | no-norm recall {rz['recall']:.2f}", flush=True)
    return {"seed": seed, "host": rh, "neural": rn, "nonorm": rz}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    print(f"[harden 320 read-out] does the conversation survive the fully-on-brain NEURAL read-out normalization "
          f"(adaptation + feedforward inhibition) vs host double-centring? (CPU proxy, the integration de-risk)",
          flush=True)
    rows = [run_seed(s) for s in (42, 43, 44, 45, 46, 47)]

    h_recall = float(np.mean([r["host"]["recall"] for r in rows]))
    n_recall = float(np.mean([r["neural"]["recall"] for r in rows]))
    z_recall = float(np.mean([r["nonorm"]["recall"] for r in rows]))
    n_fa = sum(r["neural"]["false_accept"] for r in rows)
    h_fa = sum(r["host"]["false_accept"] for r in rows)
    all_recall_ok = all(r["neural"]["recall"] >= r["host"]["recall"] - 1e-9 for r in rows)
    all_moat_ok = all(r["neural"]["false_accept"] == 0 for r in rows)
    helps = n_recall > z_recall + 0.05                       # the normalization is load-bearing (vs no-norm)

    go = bool(all_recall_ok and all_moat_ok and helps)
    verdict = "GO" if go else "NEGATIVE"
    print(f"\n{'='*100}", flush=True)
    print(f"  MEAN (6 seeds): recall HOST {h_recall:.2f} | NEURAL {n_recall:.2f} | no-norm {z_recall:.2f} || "
          f"moat false-accepts NEURAL {n_fa} (host {h_fa})", flush=True)
    print(f"  ==> {verdict}\n{'='*100}", flush=True)
    if go:
        print(f"  GO: the fully-on-brain NEURAL read-out normalization carries the conversation — recall {n_recall:.2f}"
              f" (>= host {h_recall:.2f} every seed), the learned moat abstains with {n_fa} false-accepts, and the "
              f"normalization is load-bearing (no-norm recall {z_recall:.2f}). ==> the last host scaffold in the "
              f"read-out is removable; the single-seed 320 re-stream with --readout-norm neural is worth launching "
              f"as the production-scale confirmation on the REAL stream-learned M.", flush=True)
    else:
        why = []
        if not all_recall_ok: why.append(f"neural recall < host on a seed (the 96% read-out costs binding)")
        if not all_moat_ok: why.append(f"neural-read-out moat false-accepts ({n_fa})")
        if not helps: why.append("normalization not load-bearing vs no-norm")
        print(f"  NEGATIVE: {'; '.join(why)}. Honest — the on-brain read-out's pool noise costs the conversation; "
              f"keep host double-centring as the read-out (piece 1's learned moat stands), or raise the pool sizes "
              f"(lower SEM) first. Do NOT launch the GPU re-stream.", flush=True)

    out = {"verdict": verdict, "host_recall": h_recall, "neural_recall": n_recall, "nonorm_recall": z_recall,
           "neural_false_accepts": n_fa, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_harden_320_neural_readout.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}\n  Total elapsed: {time.time()-t0:.1f}s", flush=True)
    raise SystemExit(0 if go else 1)


if __name__ == "__main__":
    main()
