"""SOAK / no-regression gate for the OFFLINE SLEEP-REPLAY consumer flip (#64, BRAIN_SLEEP_REPLAY).

The whole scenario is run twice on the SAME session state — flag OFF vs flag ON — through the REAL production wiring
(EpisodicRecallOrgan.recall + recall_disclosure + continuous_engine.consolidate_sleep_replay). The bars:

  ORDINARY-TURN NO-REGRESSION (the flip gate): sleep-replay fires ONLY on a genuine sleep-depth idle tick and mutates
  ONLY on an actual pass, so a plain recall turn with the flag ON — before any sleep event — is BYTE-IDENTICAL to the
  flag OFF turn (reply strings + in_memory verdicts + surfaced completion all match; the moat topic abstains identically).

  DEEP-IDLE LOAD-BEARING: on a sleep event, flag ON replays the BATCH of episodes stored since the last sleep so every
  batch topic's later recall reads STRONGER (its graded apical depth_hold rises) and the reply surfaces the batch
  retention + host store-order WHEN position; flag OFF is a no-op (consolidate returns None, store hash unchanged, recall
  flat). The strengthened batch survives the validated AdEx sleep phase-switch byte-identical.

  CRASH-ROLLBACK: a simulated mid-pass failure (reactivate raises on the 2nd batch topic, after the 1st already mutated
  the persistent store) MUST roll the store back byte-identically to its pre-sleep hash AND return the bridge to the wake
  neuron-model (a mid-sleep AdEx strand would corrupt every future recall), then re-raise so the caller logs it.

Scenario (topics cat/dog/bird): note dog+bird (cat never discussed). ORDINARY recall dog/bird/cat (no sleep) — the
no-regression bar. DEEP-IDLE sleep pass replays the [dog, bird] batch. POST-SLEEP recall dog/bird (stronger ON, flat OFF)
+ cat (abstains). All reads snapshot-isolated for determinism. HONEST BOUND: this is the DIRECT retain/re-order payoff on
the episodic store only — NO replay-driven hippocampus->cortex compositional transfer is claimed (still NO-GO,
2026-08-03-replay-cortical-consolidation-v2-calibration-NO-GO). The WHEN-order is the declared host store-order residual.

  Run: SIM_BACKEND=cupy python -m research.runners._sleep_replay_flip_soak --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
# Import the changed modules from _REPO BEFORE consolidate_sleep_replay lazily imports the _gap5 phase-switch runners
# (which hardcode sys.path.insert(0, main-repo)); harmless when _REPO IS the main repo (the parent's run context).
import sim.bridge  # noqa: F401,E402
from sim.backend import get_backend  # noqa: E402
from research.runners.d5_episodic_production_organ import (  # noqa: E402
    EpisodicRecallOrgan, recall_disclosure, SURFACED_GRADED_READ)
from research.runners._gap5_dendritic_dap_readout_completion_derisk import _reset_apical_latch  # noqa: E402
from research.runners._gap5_d5_latch_self_termination_derisk import snapshot_state, restore_state  # noqa: E402
from webapp import continuous_engine as CE  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_sleep_replay_flip" / "soak_seed42.json"


def _whash(cp, W):
    h = np.asarray(cp.asnumpy(W) if hasattr(cp, "asnumpy") else W, dtype=np.float32)
    return hashlib.sha1(h.tobytes()).hexdigest()[:16]


def _run_scenario(org, mem, snap, cp, cache_key, W0, *, flag_on):
    """Ordinary recall turns (no sleep) then ONE deep-idle sleep event, on store-weights W0 (snapshot-isolated reads).
    `flag_on` sets BRAIN_SLEEP_REPLAY. Returns per-turn reply records + the post-sleep store hash."""
    os.environ["BRAIN_SLEEP_REPLAY"] = "1" if flag_on else "0"
    CE.forget_session(cache_key)

    def read(topic, W):
        restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W)
        rec = org.recall(topic)
        return {"topic": topic, "in_memory": bool(rec["in_memory"]),
                "apical_cue": round(float(rec["apical_cue"]), 5),
                "depth_hold": round(float((rec.get("graded_cue") or {}).get(SURFACED_GRADED_READ, 0.0)), 5),
                "reply": recall_disclosure(rec, cache_key=cache_key)}

    # ORDINARY TURNS (no sleep event has fired) -> flag ON must be byte-identical to flag OFF here
    ord_dog = read("dog", W0)
    ord_bird = read("bird", W0)
    ord_cat = read("cat", W0)
    # DEEP-IDLE SLEEP EVENT: the pass replays the [dog, bird] batch (OFF -> None + store unchanged)
    restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W0)
    mem.R.C.data[:] = cp.asarray(W0)   # the persistent store starts at the known baseline
    rec = CE.consolidate_sleep_replay(cache_key, org)
    W1 = mem.R.C.data.copy()
    # POST-SLEEP TURNS
    post_dog = read("dog", W1)
    post_bird = read("bird", W1)
    post_cat = read("cat", W1)
    return {"consolidate_ret": (None if rec is None else "record"),
            "sleep_record": (None if rec is None else {k: rec[k] for k in
                             ("batch", "batch_size", "w_within_before", "w_within_after_replay",
                              "w_within_after", "store_survived_sleep_bracket")}),
            "store_hash": _whash(cp, W1),
            "ord_dog": ord_dog, "ord_bird": ord_bird, "ord_cat": ord_cat,
            "post_dog": post_dog, "post_bird": post_bird, "post_cat": post_cat}


def _crash_rollback_check(org, mem, snap, cp, cache_key, W0):
    """Simulate a mid-pass crash (reactivate raises on the 2nd batch topic) and verify the persistent store rolls back
    byte-identically, the bridge is returned to the wake neuron-model, and the exception propagates."""
    import research.runners._gap5_d5_learn_through_use_derisk as _LTU
    real_reactivate = _LTU.reactivate
    state = {"n": 0}

    def flaky(*a, **k):
        state["n"] += 1
        if state["n"] >= 2:   # succeed once (mutates the store on dog), then fail mid-batch (bird)
            raise RuntimeError("simulated GPU fall-off mid-sleep-replay")
        return real_reactivate(*a, **k)

    os.environ["BRAIN_SLEEP_REPLAY"] = "1"
    CE.forget_session(cache_key)
    restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W0)
    mem.R.C.data[:] = cp.asarray(W0)
    hash_pre = _whash(cp, mem.R.C.data)
    wake_model_pre = str(getattr(mem.bridge.core_config, "neuron_model_type", ""))
    _LTU.reactivate = flaky
    raised = False
    try:
        CE.consolidate_sleep_replay(cache_key, org)
    except RuntimeError:
        raised = True
    finally:
        _LTU.reactivate = real_reactivate
    hash_post = _whash(cp, mem.R.C.data)
    wake_model_post = str(getattr(mem.bridge.core_config, "neuron_model_type", ""))
    os.environ["BRAIN_SLEEP_REPLAY"] = "0"
    return {"raised": raised, "rolled_back": bool(hash_post == hash_pre),
            "wake_model_restored": bool(wake_model_post == wake_model_pre),
            "hash_pre": hash_pre, "hash_post": hash_post, "reactivate_calls": state["n"],
            "wake_model_pre": wake_model_pre, "wake_model_post": wake_model_post}


def run_one(seed, a, backend):
    t0 = time.time()
    print("\n" + "=" * 118)
    print(f"[sleep-soak] seed={seed} backend={backend} — ordinary turns OFF==ON (no-regression); deep-idle sleep pass "
          f"strengthens the batch ON / flat OFF; crash-rollback intact.", flush=True)
    result = {"seed": seed, "backend": backend, "params": vars(a)}
    cache_key = ("sleep-soak", seed)
    try:
        cp, _ = get_backend()
        org = EpisodicRecallOrgan(seed, ["cat", "dog", "bird"], verbose=False, sep_bias=a.sep_bias)
        org._ensure_built()
        mem = org.mem
        assert org.note_topic("dog") and org.note_topic("bird"), "note_topic failed"
        mem.recall("dog"); mem.R.hard_silence(); _reset_apical_latch(mem.bridge)
        snap = snapshot_state(mem.bridge)
        W0 = mem.R.C.data.copy()

        off = _run_scenario(org, mem, snap, cp, cache_key, W0, flag_on=False)
        on = _run_scenario(org, mem, snap, cp, cache_key, W0, flag_on=True)
        crash = _crash_rollback_check(org, mem, snap, cp, cache_key, W0)

        # ── ORDINARY-TURN NO-REGRESSION: before any sleep event, ON == OFF byte-identical ──
        ord_identical = (off["ord_dog"] == on["ord_dog"] and off["ord_bird"] == on["ord_bird"]
                         and off["ord_cat"] == on["ord_cat"])
        # ── OFF path: sleep pass is a no-op — store byte-identical + recall flat ──
        off_store_flat = (off["consolidate_ret"] is None and off["store_hash"] == _whash(cp, W0)
                          and off["post_dog"] == off["ord_dog"] and off["post_bird"] == off["ord_bird"])
        # ── ON path: the batch's later recall rose (retention) + reply changed; the batch survived the sleep bracket ──
        sr = on["sleep_record"]
        on_batch_rose = (sr is not None and sr["batch"] == ["dog", "bird"]
                         and on["post_dog"]["depth_hold"] > on["ord_dog"]["depth_hold"]
                         and on["post_bird"]["depth_hold"] > on["ord_bird"]["depth_hold"]
                         and on["post_dog"]["reply"] != on["ord_dog"]["reply"]
                         and "replayed it offline" in on["post_dog"]["reply"]
                         and sr["store_survived_sleep_bracket"] is True)
        # ── MOAT: cat (never discussed) abstains identically OFF and ON, before and after sleep ──
        moat_intact = (not off["ord_cat"]["in_memory"] and not on["ord_cat"]["in_memory"]
                       and not off["post_cat"]["in_memory"] and not on["post_cat"]["in_memory"]
                       and off["ord_cat"]["reply"] == on["ord_cat"]["reply"] == on["post_cat"]["reply"])
        crash_ok = crash["raised"] and crash["rolled_back"] and crash["wake_model_restored"]

        no_regression = bool(ord_identical and off_store_flat and moat_intact)
        GO = bool(no_regression and on_batch_rose and crash_ok)

        result.update(dict(
            GO=GO, no_regression=no_regression, ord_identical=ord_identical, off_store_flat=off_store_flat,
            on_batch_rose=on_batch_rose, moat_intact=moat_intact, crash_ok=crash_ok,
            off=off, on=on, crash=crash, assembly_sizes=mem.assembly_sizes))
        print(f"[sleep-soak] ordinary ON==OFF (no-regression): {ord_identical}", flush=True)
        print(f"[sleep-soak] OFF sleep no-op (store flat + recall flat): {off_store_flat}", flush=True)
        if sr is not None:
            print(f"[sleep-soak] ON batch {sr['batch']} w_within {sr['w_within_before']} -> "
                  f"{sr['w_within_after_replay']} (survived bracket: {sr['store_survived_sleep_bracket']})", flush=True)
        print(f"[sleep-soak] ON dog depth_hold ord={on['ord_dog']['depth_hold']} post={on['post_dog']['depth_hold']} "
              f"| bird ord={on['ord_bird']['depth_hold']} post={on['post_bird']['depth_hold']} "
              f"(rose={on_batch_rose})", flush=True)
        print(f"[sleep-soak] ON post-sleep dog reply: {on['post_dog']['reply']}", flush=True)
        print(f"[sleep-soak] moat cat abstains identically: {moat_intact}", flush=True)
        print(f"[sleep-soak] crash-rollback: raised={crash['raised']} rolled_back={crash['rolled_back']} "
              f"wake_restored={crash['wake_model_restored']} (pre={crash['hash_pre']} post={crash['hash_post']})",
              flush=True)
        print(f"[sleep-soak] seed={seed} NO_REGRESSION={no_regression} ON_BATCH_ROSE={on_batch_rose} "
              f"CRASH_OK={crash_ok} => {'GO' if GO else 'NO-GO'}", flush=True)
        CE.forget_session(cache_key)
        del mem, org
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e); result["GO"] = False; traceback.print_exc()
    finally:
        os.environ["BRAIN_SLEEP_REPLAY"] = "0"
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--sep-bias", dest="sep_bias", type=float, default=0.0,
                    help="D5 pattern-separation set-point (board #73). Default 0 = the PRODUCTION default (separator "
                         "NOT armed, unmodified emergent assemblies).")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else [a.seed]
    _, backend = get_backend()
    results = {}; go = []
    for seed in seeds:
        r = run_one(seed, a, backend)
        results[seed] = r; go.append(bool(r.get("GO")))
    out_path = Path(a.out)
    if len(seeds) > 1:
        out_path = out_path.parent / f"soak_summary_{len(seeds)}seed.json"
        print("\n" + "#" * 118)
        print(f"[sleep-soak] {len(seeds)}-SEED SOAK: {int(sum(go))}/{len(seeds)} GO seeds={seeds}")
        print("#" * 118)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"seeds": seeds, "n_go": int(sum(go)), "go": go, "backend": backend,
                                    "results": {str(s): results[s] for s in seeds}}, indent=2, default=str))
    print(f"[sleep-soak] wrote {out_path}")
    return 0 if (go and all(go)) else 1


if __name__ == "__main__":
    sys.exit(main())
