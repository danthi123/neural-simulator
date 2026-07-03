"""EMERGE-75b -- SURPASS the EMERGE-75 boundary: make the multi-bridge A->W read-out HISTORY-INDEPENDENT so the
EMERGE-72 broadened constructions render EVERY word ON SPIKES with ZERO regression (all 6 seeds), by generalizing
EMERGE-61's inter-utterance substrate WASH-OUT to the A->W decode.

THE BOUNDARY THIS SURPASSES (EMERGE-75, honest negative). EMERGE-75 added a 3rd A->W bridge (BRIDGE-C: 13 object nouns
+ to/on/is) so the broadened constructions spike-spell every word. Its de-risk returned BOUNDARY: 3 full-render surfaces
regressed vs the token spell (GO bar 0). The runner GUESSED the cause was "the 16 overflow words don't all separate on
one bridge (closed-class prepositions co-trained with content nouns)". THE EVIDENCE CONTRADICTS THAT GUESS:
  * the ISOLATED per-word decode is **16/16 perfect** (`overflow_wordwise_rate 1.000`) -- the words DO separate;
  * **5 of 6 seeds regress ZERO**; only seed 102 (the LAST seed in the shared-engine loop) regresses 3;
  * the 3 "mismatches" render every CORRECT word (`all_acc 1.000`, `overflow_acc 1.000`).
So the boundary is NOT a vocab-separation wall. It is the EMERGE-61 ACCUMULATION effect leaking into the A->W read-out:
the A->W engines are a SHARED object across the 6-seed de-risk loop; by seed 102 the concept-pool bridge has run
~1000+ sequential pool-drives, and the Izhikevich slow-adaptation recovery current `cp_recovery_variable_u` ACCUMULATES
across them (the exact mechanism EMERGE-61 root-caused for the ORDER path). `drive_pool_and_read_lang_output`'s 50-step
soft reset recovers from a SHALLOW history but not from a DEEP one -> on the deepest seed, 3 borderline decodes flip.

THE ROOT-CAUSE FIX (a proven mechanism, reuse-by-import, NO `sim/` edit). EMERGE-61 fixed the ORDER path with a
`ResetFrameSlotCQ` that SNAPSHOTS the substrate's exact post-init dynamic state (`cp_membrane_potential_v`,
`cp_recovery_variable_u`, conductances, STP, firing) and HARD-RESTORES it before every production. EMERGE-75b applies the
SAME wash-out to the A->W read-out: snapshot each A->W concept-pool bridge's post-load state, and restore it before EVERY
`_decode`. Then every decode starts from IDENTICAL substrate state regardless of how many renders preceded it -> the
isolated-16/16 property holds at ANY render depth -> the seed-102 full-render regression closes. This is the biologically
grounded move: the same inter-utterance wash-out real Broca needs between utterances (EMERGE-61), now on the A->W path.

THE LOAD-BEARING CONTROL (causal proof the wash-out is what fixes it). ONE `UnifiedHistIndepSpell75` engine, toggled:
  * hi-OFF (the EMERGE-75 baseline reproduced): run the full 6-seed de-risk -> regress > 0 (reproduces the seed-102
    boundary; the accumulation is un-washed).
  * hi-ON (the fix): re-run the full 6-seed de-risk on the SAME engine -> regress == 0 on ALL 6 seeds.
A GO requires BOTH: hi-ON regress == 0 across all seeds (surpasses the boundary) AND hi-OFF regress > 0 (the wash-out is
causally load-bearing, not a no-op). Plus every EMERGE-75 gate preserved: all-word spike-spell acc >= 0.90, overflow acc
>= 0.90, genuinely spiking (BRIDGE-C pool->language_output lesion collapses the overflow decode), gate-first moat 0.

HONEST SCOPE. This is a READ-OUT robustness fix (make the A->W decode history-independent), NOT a new capability: the
words, bridges, and moat are EMERGE-75's; only the substrate wash-out between decodes is added (EMERGE-61's mechanism).
It closes the EMERGE-75 boundary at the SAME 16-overflow-word scale (no vocab split needed -- the evidence showed the
split the EMERGE-75 verdict proposed would NOT have fixed a sequential-accumulation problem). Reuse-by-import; NO `sim/`
edit; the gate-first no-confab moat is untouched. Renders the BOUNDED EMERGE-72 inventory, NOT open prose (R4).

Run:
  SIM_BACKEND=cupy python -m research.runners._emerge75b_history_independent_aw_derisk --demo
  SIM_BACKEND=cupy python -m research.runners._emerge75b_history_independent_aw_derisk --derisk
  SIM_BACKEND=cupy python -m research.runners._emerge75b_history_independent_aw_derisk --derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # numpy for the CPU logic; the A->W engines force cupy when built
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse-by-import ONLY -- NO sim/ edit. EMERGE-75 multi-bridge A->W dispatch + de-risk; EMERGE-61 substrate wash-out.
from research.runners._emerge75_aw_vocab_scaling_derisk import (  # noqa: E402
    UnifiedNeuralSpell75, OverflowNeuralSpell, _facts, _render_and_score_all, _derisk_one, _overflow_wordwise_accuracy,
    _sample_transcript, _SCOPE_CONSTRUCTIONS, _OVF_VOCAB16, _OVF_FUNC, _OVF_OBJ,
)
from research.runners._emerge61_spiking_broca_order_robustness_derisk import (  # noqa: E402
    _snapshot_state, _restore_state,
)
import research.runners._emerge67_neural_spell_wirein_derisk as m67  # noqa: E402
import research.runners._emerge68_function_word_spell_derisk as m68  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_emerge75b_history_independent_aw.json"


# ---------------------------------------------------------------------------------------------------------------------
# THE FIX: install the EMERGE-61 substrate WASH-OUT onto an A->W engine's `_decode`. Snapshot the post-load dynamic
# state; restore it before every decode. Toggleable (`_hi_enabled`) so ONE engine yields both the un-washed EMERGE-75
# baseline (hi OFF) and the fix (hi ON) -- the causal load-bearing control. ADDITIVE; the engine object is otherwise
# unchanged (same bridge, same weights, same plasticity-freeze); only the dynamic state is reset before each read.
# ---------------------------------------------------------------------------------------------------------------------
def _install_history_independence(engine):
    """Wrap `engine._decode` so it restores the post-load substrate state before decoding (when `_hi_enabled`)."""
    engine._hi_snapshot = _snapshot_state(engine.bridge)   # post-load, post-plasticity-freeze -> the clean read state
    engine._hi_enabled = True
    _orig_decode = engine._decode

    def _decode_hi(word):
        if getattr(engine, "_hi_enabled", False):
            _restore_state(engine.bridge, engine._hi_snapshot)   # EMERGE-61 wash-out on the A->W read path
        return _orig_decode(word)

    engine._decode = _decode_hi
    return engine


class UnifiedHistIndepSpell75(UnifiedNeuralSpell75):
    """EMERGE-75's 3-bridge A->W dispatch + EMERGE-61's inter-decode substrate wash-out on each engine. `set_hi(False)`
    reproduces the un-washed EMERGE-75 baseline; `set_hi(True)` (default) is the fix. The `.spell` dispatch, the moat,
    the lesion path, and the words/bridges are all EMERGE-75's -- only the per-decode wash-out is added."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)   # builds content(BRIDGE-A)/func(BRIDGE-F)/overflow(BRIDGE-C) engines
        self._hi_installed = False
        if self._backend_gpu:
            for eng in (self.content, self.func, self.overflow):
                _install_history_independence(eng)
            self._hi_installed = True

    def set_hi(self, on: bool):
        """Toggle the inter-decode wash-out on ALL three engines (the load-bearing control)."""
        for eng in (self.content, self.func, self.overflow):
            if hasattr(eng, "_hi_enabled"):
                eng._hi_enabled = bool(on)


def _run_six_seed(unified, unified_ovf_lesion, seeds):
    """Run EMERGE-75's per-seed de-risk over `seeds` on a shared engine (so adaptation accumulates across seeds exactly
    as in EMERGE-75). Returns (per_seed_list, total_regress)."""
    per = []
    for s in seeds:
        d = _derisk_one(s, unified, unified_ovf_lesion=unified_ovf_lesion)
        per.append(d)
    total_regress = int(sum(d["regress_mismatch"] for d in per))
    return per, total_regress


def _demo(seed=42):
    print("\n=== EMERGE-75b -- HISTORY-INDEPENDENT A->W: EMERGE-61's substrate wash-out on the A->W read-out closes the "
          "EMERGE-75 deep-accumulation regression (the boundary was NOT vocab separation -- isolated decode was 16/16; "
          "it was the Izhikevich slow-adaptation accumulating across the shared 6-seed render loop) ===\n", flush=True)
    unified = UnifiedHistIndepSpell75(load=True)
    if not unified._backend_gpu:
        print("  [skip] the A->W engines need a GPU (SIM_BACKEND=cupy); numpy fallback cannot run the read-out.\n")
        return
    orate, oper = _overflow_wordwise_accuracy(unified.overflow)
    print(f"  BRIDGE-C isolated: {int(orate*len(oper))}/{len(oper)} overflow words spike-decoded (history-independent)\n")
    lines, pc = _sample_transcript(unified, seed)
    print("  render the EMERGE-72 broadened inventory with EVERY word SPIKE-SPELLED (history-independent 3-bridge dispatch):")
    for tag, q, surface, inv in lines:
        print(f"    you> {q}\n      broca> {surface}   [{tag}; {inv}]")
    print(f"\n  producer-invocation count after {len(lines)} probes: {pc} (the abstain never invoked the producer -- moat)\n")


def _derisk(seeds, train_events=m67._TRAIN_EVENTS):
    print(f"EMERGE-75b de-risk: HISTORY-INDEPENDENT A->W read-out (EMERGE-61 wash-out on the A->W decode) closes the "
          f"EMERGE-75 deep-accumulation regression; hi-OFF reproduces the boundary (load-bearing), hi-ON == 0 regress "
          f"all seeds; {len(seeds)}-seed", flush=True)
    t0 = time.time()
    err = None
    gpu = True
    baseline_per = hi_per = None
    baseline_regress = hi_regress = None
    orate = oper = None
    all_acc = overflow_acc = lesion_ovf_acc = None
    lesion_ovf_engine = None
    spell_calls_abstain = producer_calls_abstain = None
    answer_ok = None
    try:
        unified = UnifiedHistIndepSpell75(load=True, train_events=train_events)
        gpu = unified._backend_gpu
        if not gpu:
            raise RuntimeError("A->W engines require SIM_BACKEND=cupy (GPU); numpy cannot run the spiking read-out")
        if not unified._hi_installed:
            raise RuntimeError("history-independence wash-out failed to install (no snapshot)")
        # the overflow-lesioned engine for the genuinely-spiking control (hi-independent; lesion is a pathway edit)
        unified_ovf_lesion = UnifiedHistIndepSpell75(load=True, train_events=train_events, overflow_lesion=True)

        # (1) LOAD-BEARING BASELINE (hi OFF): reproduce the EMERGE-75 un-washed 6-seed run -> expect regress > 0.
        unified.set_hi(False)
        baseline_per, baseline_regress = _run_six_seed(unified, unified_ovf_lesion, seeds)
        print(f"  [hi-OFF baseline == EMERGE-75] total regress over {len(seeds)} seeds: {baseline_regress} "
              f"(per-seed: {[d['regress_mismatch'] for d in baseline_per]})", flush=True)

        # (2) THE FIX (hi ON): the SAME engine, wash-out restored before every decode -> expect regress == 0 all seeds.
        unified.set_hi(True)
        hi_per, hi_regress = _run_six_seed(unified, unified_ovf_lesion, seeds)
        print(f"  [hi-ON  fix]                 total regress over {len(seeds)} seeds: {hi_regress} "
              f"(per-seed: {[d['regress_mismatch'] for d in hi_per]})", flush=True)

        # aggregate the fix run's gates
        def m(k):
            return float(np.mean([d[k] for d in hi_per]))
        all_acc = m("all_acc")
        overflow_acc = m("overflow_acc")
        lesion_vals = [d["lesion_overflow_acc"] for d in hi_per if d["lesion_overflow_acc"] is not None]
        lesion_ovf_acc = float(np.mean(lesion_vals)) if lesion_vals else None
        spell_calls_abstain = int(sum(d["spell_calls_on_abstain"] for d in hi_per))
        producer_calls_abstain = int(sum(d["producer_calls_on_abstain"] for d in hi_per))
        answer_ok = all(d["answer_produced"] for d in hi_per)
        # isolated overflow rate + engine-lesion, under the fix
        unified.set_hi(True)
        orate, oper = _overflow_wordwise_accuracy(unified.overflow)
        lesion_ovf_engine, _ = _overflow_wordwise_accuracy(unified_ovf_lesion.overflow)
    except Exception as e:
        err = repr(e)
        traceback.print_exc()

    if err is None and gpu:
        BAR = 0.90
        all_ok = all_acc >= BAR
        overflow_ok = overflow_acc >= BAR
        spiking_ok = (orate is not None and orate >= BAR
                      and lesion_ovf_acc is not None and (overflow_acc - lesion_ovf_acc) >= 0.40
                      and lesion_ovf_engine is not None and (orate - lesion_ovf_engine) >= 0.40)
        fix_ok = (hi_regress == 0)                     # the fix: ZERO regression across all seeds
        loadbearing_ok = (baseline_regress > 0)        # the wash-out is causally load-bearing (baseline regresses)
        moat_ok = (spell_calls_abstain == 0) and (producer_calls_abstain == 0) and answer_ok

        go = bool(all_ok and overflow_ok and spiking_ok and fix_ok and loadbearing_ok and moat_ok)
        if go:
            verdict = (
                f"GO -- the EMERGE-75 boundary is SURPASSED by making the multi-bridge A->W read-out HISTORY-INDEPENDENT "
                f"(EMERGE-61's substrate wash-out generalized to the A->W decode). The EMERGE-75 boundary was NOT a "
                f"vocab-separation wall (its isolated decode was 16/16; 5 of 6 seeds regressed 0; only the LAST seed in "
                f"the shared-engine loop regressed 3) -- it was the Izhikevich slow-adaptation current "
                f"`cp_recovery_variable_u` ACCUMULATING across the ~1000+ sequential pool-drives of the 6-seed render "
                f"loop, exactly the EMERGE-61 mechanism. EMERGE-75b snapshots each A->W concept-pool bridge's post-load "
                f"dynamic state and HARD-RESTORES it before EVERY `_decode`, so every decode starts from identical "
                f"substrate state regardless of render depth. RESULT: hi-ON regress {hi_regress} across all "
                f"{len(seeds)} seeds (was 3 on seed 102), all-word spike-spell accuracy {all_acc:.3f} (>= {BAR}), "
                f"overflow-word slot accuracy {overflow_acc:.3f}, BRIDGE-C isolated {int((orate or 0)*len(oper))}/"
                f"{len(oper)} (rate {orate:.3f}). LOAD-BEARING: the SAME engine with the wash-out DISABLED (hi-OFF) "
                f"regresses {baseline_regress} over the {len(seeds)} seeds (per-seed "
                f"{[d['regress_mismatch'] for d in baseline_per]}) -- so the wash-out is causally what closes it, not a "
                f"no-op. GENUINELY SPIKING: the overflow LESION (zeroing BRIDGE-C's pool->language_output pathway) "
                f"collapses the overflow decode to {lesion_ovf_acc:.3f} (engine-lesion {lesion_ovf_engine:.3f}). The "
                f"gate-first no-confab MOAT holds by construction: {spell_calls_abstain} spell + "
                f"{producer_calls_abstain} producer invocations on abstains. ==> the EMERGE-72 broadened constructions "
                f"render EVERY word ON SPIKES with ZERO regression, all seeds, at the SAME 16-overflow-word scale (no "
                f"vocab split needed -- the evidence showed the split the EMERGE-75 verdict proposed would NOT have "
                f"fixed a sequential-accumulation problem). HONEST SCOPE: a READ-OUT robustness fix (history-independent "
                f"A->W decode), NOT a new capability; the words/bridges/moat are EMERGE-75's; only EMERGE-61's wash-out "
                f"is added. Reuse-by-import; NO sim/ edit. Renders the BOUNDED EMERGE-72 inventory, NOT open prose (R4).")
        else:
            miss = []
            if not fix_ok:
                miss.append(f"hi-ON still regresses ({hi_regress} mismatches -- the wash-out did not fully close it; "
                            f"per-seed {[d['regress_mismatch'] for d in hi_per]})")
            if not loadbearing_ok:
                miss.append(f"hi-OFF baseline did NOT regress ({baseline_regress}) -- cannot prove the wash-out is "
                            f"load-bearing on these seeds (the EMERGE-75 boundary may not reproduce in this run order; "
                            f"the fix may still be correct but the causal control is inconclusive)")
            if not all_ok:
                miss.append(f"all-word spike-spell accuracy {all_acc:.3f} < {BAR}")
            if not overflow_ok:
                miss.append(f"overflow-word slot accuracy {overflow_acc:.3f} < {BAR}")
            if not spiking_ok:
                miss.append(f"overflow read-out not clearly spiking (rate {orate}, overflow-lesion {lesion_ovf_acc}, "
                            f"engine-lesion {lesion_ovf_engine})")
            if not moat_ok:
                miss.append(f"MOAT: {spell_calls_abstain} spell + {producer_calls_abstain} producer on abstains / "
                            f"answer {answer_ok} -- BLOCKING if the producer/spell ran on abstain")
            verdict = ("BOUNDARY -- " + "; ".join(miss) + ". The history-independent wash-out (EMERGE-61 mechanism on "
                       "the A->W decode) is the evidence-driven root-cause fix (the EMERGE-75 boundary was deep-"
                       "accumulation, not vocab separation); if hi-ON still regresses, the accumulation is deeper than "
                       "the snapshot restores (next: also snapshot/restore any per-synapse dynamic state, or reload the "
                       "bridge between seeds). Do NOT weaken the moat.")
    elif not gpu:
        go = False
        verdict = ("SKIP/BOUNDARY -- the spiking A->W read-out requires SIM_BACKEND=cupy (GPU); this run had only the "
                   "numpy backend. Re-run on GPU. The wash-out install + dispatch + moat logic are CPU-testable "
                   "(tests/test_emerge75b_history_independent_aw.py); the on-spikes A->W is GPU-only.")
    else:
        go = False
        verdict = f"ERROR -- {err}"

    transcript = []
    try:
        if err is None and gpu:
            unified.set_hi(True)
            lines, _ = _sample_transcript(unified, seeds[0])
            transcript = [{"question": q, "surface": s, "invocation": i} for (_t, q, s, i) in lines]
    except Exception:
        pass

    summary = {
        "probe": "emerge75b_history_independent_aw", "verdict": verdict,
        "go": bool(go) if (err is None and gpu) else False,
        "mechanism": ("SURPASS the EMERGE-75 boundary by making the multi-bridge A->W read-out HISTORY-INDEPENDENT. The "
                      "EMERGE-75 boundary (3 full-render regressions, seed 102 only) was NOT vocab separation (isolated "
                      "decode 16/16) -- it was the Izhikevich slow-adaptation current cp_recovery_variable_u "
                      "accumulating across the shared 6-seed render loop (the EMERGE-61 mechanism). EMERGE-75b installs "
                      "EMERGE-61's substrate wash-out (snapshot cp_membrane_potential_v/cp_recovery_variable_u/"
                      "conductances/STP/firing; hard-restore before every decode) on each A->W concept-pool bridge, so "
                      "every decode is history-independent. Toggleable (_hi_enabled) so one engine yields both the "
                      "un-washed EMERGE-75 baseline (hi OFF, the load-bearing control) and the fix (hi ON). Reuse-by-"
                      "import; NO sim/ edit; the gate-first no-confab moat is untouched."),
        "task": ("close the EMERGE-75 deep-accumulation regression by generalizing EMERGE-61's inter-utterance "
                 "substrate wash-out to the A->W read-out; hi-ON regress == 0 all seeds (fix) AND hi-OFF regress > 0 "
                 "(load-bearing) AND all-word/overflow accuracy >= 0.90 AND genuinely spiking (overflow lesion collapse) "
                 "AND gate-first moat 0; >= 6 seeds; GPU"),
        "overflow_words": _OVF_VOCAB16, "overflow_func": _OVF_FUNC, "overflow_obj": _OVF_OBJ,
        "scope_constructions": _SCOPE_CONSTRUCTIONS,
        "seeds": list(seeds), "gpu": bool(gpu), "elapsed_seconds": round(time.time() - t0, 1),
        "aggregate": None if (err is not None or not gpu) else {
            "hi_on_regress_total": hi_regress, "hi_off_baseline_regress_total": baseline_regress,
            "hi_on_per_seed_regress": [d["regress_mismatch"] for d in hi_per] if hi_per else None,
            "hi_off_per_seed_regress": [d["regress_mismatch"] for d in baseline_per] if baseline_per else None,
            "all_acc": all_acc, "overflow_acc": overflow_acc, "overflow_wordwise_rate": orate,
            "lesion_overflow_acc": lesion_ovf_acc, "engine_lesion_overflow_acc": lesion_ovf_engine,
            "spell_calls_on_abstain_total": spell_calls_abstain,
            "producer_calls_on_abstain_total": producer_calls_abstain,
        },
        "overflow_wordwise": oper,
        "sample_transcript": transcript,
        "hi_on_per_seed": hi_per,
        "hi_off_per_seed": baseline_per,
        "HONEST_NOTE": ("A READ-OUT robustness fix, NOT a new capability. The EMERGE-75 boundary was mis-diagnosed by "
                        "its own verdict as a vocab-separation wall; the raw data showed isolated decode 16/16 and only "
                        "the deepest-history seed regressing -> the true cause is EMERGE-61 slow-adaptation "
                        "accumulation in the A->W read path. EMERGE-75b applies EMERGE-61's proven substrate wash-out "
                        "(snapshot + hard-restore the dynamic per-neuron state before each decode) so the A->W decode "
                        "is history-independent. The words, bridges, plasticity-freeze, and gate-first no-confab moat "
                        "are all EMERGE-75's -- unchanged. Closes the boundary at the SAME 16-overflow-word scale (no "
                        "vocab split). Reuse-by-import; NO sim/ edit. NOT open prose (R4, the deferred wall)."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 118, flush=True)
    print(f"[emerge75b] VERDICT: {verdict}", flush=True)
    print(f"[emerge75b] wrote {OUT}\n" + "=" * 118, flush=True)
    return 0 if (err is None and gpu and go) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--train-events", type=int, default=m67._TRAIN_EVENTS)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--derisk", action="store_true")
    a = ap.parse_args()
    if a.derisk:
        return _derisk(a.seeds, train_events=a.train_events)
    _demo(a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
