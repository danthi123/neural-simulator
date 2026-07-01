"""Phase-0 DROP-IN de-risk: the ~21M TinyStories generator IN the VALIDATED grounded-lang constrain->verify harness.

The core "minimize the transformer" thesis test. It swaps the external Qwen-0.5B (and the smaller Generator-F) for the
Phase-0 ~21.3M TinyStories generator (2026-07-01-fluid-conversation-phase0-minimal-generator.md, held-out ppl 5.66)
inside the EXISTING, byte-UNMODIFIED grounded-lang constrained-decode harness
(research/runners/constrained_decode_gate: `_GroundedConstrainedLM` + `_run_rung`; sim/grounded_decode.grounded_decode;
the FROZEN _CDC_* verdict via constrained_decode_core.cdc_verdict / cdc_scale_confidence). It reuses the harness WHOLE
-- same `_GROUNDED` (24 children's-story facts, in-domain for a TinyStories model) + `_UNGROUNDED` nonsense tokens,
same 3-way constrained/unconstrained/shuffled decode, same frozen bars, same (6,12,24) scale ladder -- so the ONLY
thing that changes is which fluent generator sits behind the veto.

The discriminating question (per the harness design): does the 21M generator, behind the per-token grounded veto,
stay NON-VACUOUS (>= 2 distinct on-proposition content words, answer-rate >= 0.5) while the UNCONSTRAINED and
SHUFFLED controls DRIFT above the faithful bar (UER > 0.20)? And does the no-confab MOAT hold (abstain on the
ungrounded nonsense entities)? The per-token veto making constrained-UER ~0 is MECHANICAL by construction, NOT the
result -- the result is non-vacuity-behind-the-veto surviving with a much smaller, transformer-minimized generator,
scale-confident across the ladder.

Reuse-by-import (NO sim/ edit; the validated gate runner is byte-UNCHANGED -- `_GroundedConstrainedLM.__init__` was
already parameterized additively for d_model/n_layer/n_head/bpe_path, defaults = the original arch).

Run (decisive, GPU):  python -m research.runners._fluidconv_phase0_dropin_derisk --seeds 42 43 44 --out research/findings/raw/_fluidconv_phase0_dropin.json
Run (fast wiring smoke): ... --tiny --seeds 42 43 44 --out <...>
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
from research.runners.constrained_decode_gate import (  # noqa: E402
    _GroundedConstrainedLM, _run_rung, _params)
from research.runners.constrained_decode_core import (  # noqa: E402
    cdc_scale_confidence, _CDC_SCALE_LADDER)

# The Phase-0 ~21.3M TinyStories generator (d512/L6/H8/V2049/blk512). The gate loads `<CKPT>.pt` + `bpe_path`.
CKPT = "research/findings/raw/fluidconv/gen_tinystories_20M.ckpt"
BPE = "research/findings/raw/fluidconv/gen_tinystories.bpe.json"
ARCH = dict(d_model=512, n_layer=6, n_head=8, block_size=512, bpe_path=BPE)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--tiny", action="store_true",
                    help="single-rung fast wiring smoke (K=6, max_new=12) -- NOT propagated")
    ap.add_argument("--out", default="research/findings/raw/_fluidconv_phase0_dropin.json")
    a = ap.parse_args(argv)
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds (the frozen _CDC_MIN_SEEDS)"); return 2
    if not (os.path.exists(CKPT + ".pt") and os.path.exists(BPE)):
        print(f"NOT-RUNNABLE: 21M generator artifact absent ({CKPT}.pt / {BPE})"); return 2

    P = _params(a.tiny)
    t0 = time.time()
    print(f"[phase0-dropin] loading the ~21M TinyStories generator into the validated constrain harness "
          f"(constrained/unconstrained/shuffled)...", flush=True)
    lm_c = _GroundedConstrainedLM(CKPT, mode="constrained", **ARCH)
    lm_u = _GroundedConstrainedLM(CKPT, mode="unconstrained", **ARCH)
    lm_s = _GroundedConstrainedLM(CKPT, mode="shuffled", **ARCH)
    npar = sum(p.numel() for p in lm_c.model.parameters()) / 1e6
    print(f"[phase0-dropin] loaded ~{npar:.1f}M params (V={lm_c.tok.vocab_size}, dev={lm_c.device}) -- "
          f"decisive run MUST be cuda; ladder={list(P['ladder'])} seeds={a.seeds}\n", flush=True)

    rungs = []
    for K in P["ladder"]:
        rg = _run_rung(K, a.seeds, lm_c, lm_u, lm_s, P["max_new"], P["n_ungrounded"])
        rungs.append(rg)
        v = rg["verdict"]
        # Surface the per-rung discriminators (mean across seeds) for the log.
        ps = v.get("per_seed", {})
        cu = float(np.mean([d["constrained_uer"] for d in ps.values()])) if ps else float("nan")
        uu = float(np.mean([d["unconstrained_uer"] for d in ps.values()])) if ps else float("nan")
        su = float(np.mean([d["shuffled_uer"] for d in ps.values()])) if ps else float("nan")
        ab = float(np.mean([d["abstain_on_ungrounded_rate"] for d in ps.values()])) if ps else float("nan")
        print(f"  [K={K:2d}] GATE={v.get('GATE'):5s} instrument_valid={v.get('instrument_valid')} | "
              f"constrained: UER {cu:.3f} nonvac {rg['constrained_nonvac_rate_mean']:.2f} | "
              f"drift: unconstrained-UER {uu:.3f} shuffled-UER {su:.3f} | moat abstain-on-ungrounded {ab:.2f}",
              flush=True)
        if v.get("GATE") == "VOID":
            print(f"       VOID reason: {v.get('reason')}", flush=True)

    sc = (cdc_scale_confidence(rungs) if not a.tiny else
          {"scale_confident": False, "classification": "TINY (wiring smoke; NOT propagated)",
           "reason": "single-rung tiny smoke"})
    out = {"generator": f"phase0-tinystories-{npar:.1f}M", "ckpt": CKPT, "arch": ARCH,
           "ladder": rungs, "scale_confident": sc["scale_confident"],
           "scale_classification": sc["classification"], "scale_reason": sc.get("reason", ""),
           "nonvac_by_rung": sc.get("nonvac_by_rung"), "device": lm_c.device, "tiny": bool(a.tiny),
           "seeds": a.seeds, "elapsed_seconds": round(time.time() - t0, 1),
           "HONEST_CEILING": ("the ~21M TinyStories generator -- ~15-25x SMALLER than the external Qwen-0.5B -- "
               "supplies fluency behind the validated per-token grounded veto; constrained decoding TRADES open-ended "
               "fluency for faithfulness BY DESIGN (NOT free composition, NOT an LLM, NOT conversation-solved). The "
               "thesis it tests: the transformer can be MINIMIZED (and made spiking-on-substrate, 88.6M-forward "
               "validated) without losing grounded non-vacuity or the no-confab moat.")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    go = bool(sc["scale_confident"])
    print(f"\n{'='*108}", flush=True)
    print(f"  {'SCALE-CONFIDENT-PASS' if go else sc['classification']}: 21M generator behind the grounded veto | "
          f"device={lm_c.device} | nonvac_by_rung={sc.get('nonvac_by_rung')}", flush=True)
    print(f"  scale_reason: {sc.get('reason','')}", flush=True)
    print("  => the transformer-minimized fluent+grounded path holds, scale-confident across the ladder, moat intact."
          if go else "  => localize per-rung GATE (VOID=instrument / FAIL=science); an honest partial IS a finding.",
          flush=True)
    print(f"  [saved] {a.out}\n{'='*108}", flush=True)
    return 0 if (go or a.tiny) else 1


if __name__ == "__main__":
    sys.exit(main())
