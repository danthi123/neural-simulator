"""#138 -- DA-encoding substrate homeostasis: fire the Turrigiano consolidation pass on a SLEEP/NREM event, not only
the light between-turn idle tick (additive/default-off).

THE QUESTION. `webapp.continuous_engine.consolidate_substrate_homeostasis` runs the on-substrate Turrigiano
synaptic-scaling consolidation pass (already 6-seed GO, production-default:
research/findings/2026-08-25-da-encoding-substrate-turrigiano-scaling-FLIP.md). Today `tick_idle_sessions` fires it on
EVERY idle tick (>= IDLE_SEC ~20s) whenever the store grew -- a stopgap, since Turrigiano scaling is canonically a
slow OFFLINE/SLEEP process, not a between-utterance one. This runner does NOT re-validate the scaling MATH (that
question is already closed); it validates the NEW #138 TRIGGER WIRING added to `continuous_engine.py`:
`BRAIN_DA_ENCODING_SLEEP_TRIGGER` retargets the pass from "every idle tick" to "genuine sleep-depth idle only"
(>= SLEEP_IDLE_SEC, minutes), via an if/else on the SAME call site (not an addition beside it).

WHAT'S CHECKED (per seed, through the REAL production call path -- `continuous_engine.tick_idle_sessions` ->
`consolidate_substrate_homeostasis` -> `da_encoding_drives_chat.apply_substrate_homeostasis` ->
`OneBrainComposer.apply_homeostatic_scaling`, never a re-implementation):
  1. OFF (unset flag): a light-idle tick (>= IDLE_SEC, < SLEEP_IDLE_SEC) FIRES the pass (byte-identical to HEAD),
     tagged trigger="idle_tick".
  2. ON + light idle (< SLEEP_IDLE_SEC): the pass does NOT fire (no-op) -- proves the original idle-tick call is
     genuinely SKIPPED while the flag is armed, not merely joined by a second path.
  3. ON + sleep-depth idle (>= SLEEP_IDLE_SEC): the pass FIRES, tagged trigger="nrem_sleep", and the store's
     synaptic weights actually change (a real rescale, not a metadata-only event).
  4. MECHANISM IDENTITY (the anti-cheat): the per-engram scale vector the sleep-triggered pass computes is BYTE-EQUAL
     to what the (already-validated) idle-tick pass computes on an identically-built, identically-taught store --
     #138 changes WHEN the pass fires, never WHAT it computes.
  5. COMPOUNDING GUARD: immediately re-ticking a session that already consolidated this batch (still sleep-depth
     idle, no new facts taught) is a no-op -- the shared `_LAST_HOMEO_KB` new-writes-since-last-pass guard, which now
     also has to hold ACROSS the two #138 trigger paths, not just within one.
  6. LESION DELEGATION: with `BRAIN_DA_ENCODING_LESION=1`, the sleep-triggered path ALSO no-ops -- proves the new
     branch calls into the SAME self-gating faculty check (`apply_substrate_homeostasis`) rather than bypassing it.

INSTRUMENT: a LEAN OneBrainComposer (research/runners/one_brain_composer.py), the same 12-word vocab / 9-fact / Latin-
square DA battery `_da_encoding_leansoak.py` uses (reused import, not re-derived) -- ~5.7k neurons, seconds per build.
Each of the 6 scenarios above gets its OWN freshly-built composer + cache_key per seed (no cross-scenario state
leakage through `continuous_engine`'s module-level dicts).

SEEDS: 42 43 44 100 101 102 (varies the substrate build -- heterogeneity/thresholds ride `cfg.seed` inside
OneBrainComposer -- so the wiring is shown to hold across distinct substrate realizations, not one lucky net).

BACKEND: works on SIM_BACKEND=numpy (default here) or cupy. The claim under test is CONTROL FLOW (Python `if`/`else`
branching in continuous_engine.py, zero cupy/numpy array ops) -- backend-independent by construction; the underlying
Turrigiano MATH already has a separate cupy 6-seed GO. `SIM_BACKEND=cupy` through gpu_queue.sh is queued separately
for production-backend parity (non-blocking due-diligence, not required to decide this trigger-wiring question).

Run:
  .venv/bin/python -u -m research.runners._da_encoding_138_sleep_trigger_derisk --out research/findings/raw/_da_encoding_138_sleep_trigger/numpy_6seed.json
  SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._da_encoding_138_sleep_trigger_derisk --out research/findings/raw/_da_encoding_138_sleep_trigger/cupy_6seed.json   # via gpu_queue.sh add
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging
logging.getLogger().setLevel(logging.ERROR)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.one_brain_composer import OneBrainComposer  # noqa: E402
from research.runners._burndown_I7_dopamine_encoding_deploy_derisk import da_to_encoding_gain  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

SEEDS = [42, 43, 44, 100, 101, 102]
D = 64
K_DA = 2.0
DA_BASELINE = 0.5

# Reused verbatim from _da_encoding_leansoak.py (the validated lean battery) -- NOT re-derived.
VOCAB = ["apple", "bird", "cat", "chase", "dog", "eat", "grass", "home", "leaf", "river", "see", "seed"]
FACTS = [
    ("dog", "eat", "grass"), ("cat", "eat", "apple"), ("bird", "eat", "river"),
    ("dog", "chase", "home"), ("cat", "chase", "seed"), ("bird", "chase", "leaf"),
    ("dog", "see", "cat"), ("cat", "see", "bird"), ("bird", "see", "dog"),
]
DA_CLASS = ["high", "low", "tonic", "tonic", "high", "low", "low", "tonic", "high"]
_DA = {"high": 1.24, "tonic": 0.5, "low": 0.05}
K_MAX = 13
HOMEO_KW = dict(homeostatic_scaling=True, homeo_beta_down=0.25, homeo_s_min=0.34, homeo_s_max=4.0)


def _gains(seed_das):
    return [da_to_encoding_gain(da, DA_BASELINE, K_DA) for da in seed_das]


def _build(seed):
    """One production OneBrainComposer, homeostatic_scaling armed, all 9 facts taught at their DA-schedule gain
    (the real per-fact encoding_gain_fn build -- not the derived-block shortcut, since this runner is not measuring
    the write-magnitude flip, and a real build is the more direct instrument for a trigger-wiring question)."""
    das = [_DA[c] for c in DA_CLASS]
    gains = _gains(das)
    holder = {"g": 1.0}
    c = OneBrainComposer(seed=seed, D=D, vocab=VOCAB, k_max=K_MAX, enable_batched=False,
                         enable_rf_cudagraph=False, enable_csr_cache=False, enable_spiking_cleanup=False,
                         encoding_gain_fn=lambda: holder["g"], **HOMEO_KW)
    for (a, act, p), g in zip(FACTS, gains):
        holder["g"] = float(g)
        c.store(a, act, p)
    return c


def _weight_fingerprint(comp):
    """A cheap scalar sensitive to ANY per-engram rescale: sum of |w| over every stored synapse. Two composers built
    from the identical seed/facts/gains with identical store_conns have identical fingerprints; any homeostatic
    rescale (up OR down) moves it."""
    return float(sum(abs(complex(w)) for (_p, _q, w) in comp.store_conns))


class _StubChat:
    """The minimal object `da_encoding_drives_chat.apply_substrate_homeostasis` / `consolidate_substrate_homeostasis`
    need: `chat.inner.composer`. No DA-level read is required by the substrate-homeostasis path (only the per-write
    gain path reads `chat._last_da_drives`), so nothing else is stubbed."""
    class _Inner:
        def __init__(self, composer):
            self.composer = composer

    def __init__(self, composer):
        self.inner = self._Inner(composer)


def _run_seed(seed, out):
    import webapp.continuous_engine as CE
    import webapp.da_encoding_drives_chat as DAE

    v = Verdict("da-encoding #138 sleep/NREM trigger wiring, seed=%d" % seed)
    seed_result = {"seed": seed}

    # Fresh module-level state per seed run (these dicts are process-global; this runner is the only writer).
    def _reset_ce_state():
        CE._LAST_REQUEST.clear(); CE._LAST_HOMEO_KB.clear(); CE._INNER_LIFE.clear()
        CE._WANDER_BUDGET.clear(); CE._D5_BUDGET.clear(); CE._RECALLED_TOPIC.clear()

    def _tick(cache_key, chat, idle_for_sec):
        """One tick_idle_sessions() pass over exactly this session, idle for `idle_for_sec`. Returns the inner-life
        record the DA-homeostasis pass appended this call (or None if it appended nothing == a no-op)."""
        now = 10_000.0  # a fixed arbitrary clock; _LAST_REQUEST is set relative to it below
        CE._LAST_REQUEST[cache_key] = now - float(idle_for_sec)
        session_mood = {cache_key: {"valence": 0.0, "arousal": 0.0}}
        before = list(CE._INNER_LIFE.get(cache_key, []))
        CE.tick_idle_sessions(session_mood, affect_organ_getter=lambda: None, now=now,
                              selfinit_getter=None, episodic_getter=None, chat_getter=lambda ck: chat)
        after = CE._INNER_LIFE.get(cache_key, [])
        new = after[len(before):]
        for rec in new:
            if rec.get("substrate_homeostasis"):
                return rec
        return None

    # ---- Scenario 1: OFF, light idle -> fires (byte-identical to HEAD) -----------------------------------------
    _reset_ce_state()
    os.environ.pop("BRAIN_DA_ENCODING_SLEEP_TRIGGER", None)
    os.environ.pop("BRAIN_DA_ENCODING_LESION", None)
    v.require("BRAIN_DA_ENCODING_SLEEP_TRIGGER unset -> sleep-trigger reports disabled",
              CE.substrate_homeostasis_sleep_trigger_enabled(), expect=False)
    comp_off = _build(seed)
    fp_off_before = _weight_fingerprint(comp_off)
    chat_off = _StubChat(comp_off)
    rec_off = _tick("s%d-off" % seed, chat_off, idle_for_sec=CE.IDLE_SEC + 1.0)  # light idle, well under SLEEP_IDLE_SEC
    fp_off_after = _weight_fingerprint(comp_off)
    off_fired = rec_off is not None
    v.require("OFF + light idle: the pass FIRES (unconditional idle-tick path, matches pre-#138 HEAD)",
              off_fired, expect=True)
    v.require("OFF: fired record tagged trigger=idle_tick", (rec_off or {}).get("trigger"), expect="idle_tick")
    v.reaches("OFF: store weights actually changed (a real rescale, not a metadata event)",
              before=fp_off_before, after=fp_off_after)
    scales_off = list(getattr(comp_off, "_homeo_scales", []) or [])

    # ---- Scenario 2: ON, light idle -> does NOT fire (the original call is genuinely skipped) --------------------
    _reset_ce_state()
    os.environ["BRAIN_DA_ENCODING_SLEEP_TRIGGER"] = "1"
    v.require("BRAIN_DA_ENCODING_SLEEP_TRIGGER=1 -> sleep-trigger reports enabled",
              CE.substrate_homeostasis_sleep_trigger_enabled(), expect=True)
    comp_on_light = _build(seed)
    fp_light_before = _weight_fingerprint(comp_on_light)
    chat_on_light = _StubChat(comp_on_light)
    rec_on_light = _tick("s%d-on-light" % seed, chat_on_light, idle_for_sec=CE.IDLE_SEC + 1.0)
    fp_light_after = _weight_fingerprint(comp_on_light)
    v.require("ON + light idle (< SLEEP_IDLE_SEC): the pass does NOT fire (no-op)",
              rec_on_light is None, expect=True)
    v.require("ON + light idle: weight fingerprint literally unchanged", fp_light_after == fp_light_before, expect=True)

    # ---- Scenario 3: ON, sleep-depth idle -> FIRES, tagged nrem_sleep, real rescale -------------------------------
    _reset_ce_state()
    comp_on_sleep = _build(seed)
    fp_sleep_before = _weight_fingerprint(comp_on_sleep)
    chat_on_sleep = _StubChat(comp_on_sleep)
    ck_sleep = "s%d-on-sleep" % seed
    rec_on_sleep = _tick(ck_sleep, chat_on_sleep, idle_for_sec=CE.SLEEP_IDLE_SEC + 1.0)
    fp_sleep_after = _weight_fingerprint(comp_on_sleep)
    sleep_fired = rec_on_sleep is not None
    v.require("ON + sleep-depth idle (>= SLEEP_IDLE_SEC): the pass FIRES", sleep_fired, expect=True)
    v.require("ON + sleep-depth: fired record tagged trigger=nrem_sleep",
              (rec_on_sleep or {}).get("trigger"), expect="nrem_sleep")
    v.reaches("ON + sleep-depth: store weights actually changed (a real store-synapse rescale)",
              before=fp_sleep_before, after=fp_sleep_after)
    scales_on_sleep = list(getattr(comp_on_sleep, "_homeo_scales", []) or [])

    # ---- Check 4: MECHANISM IDENTITY -- the sleep trigger computes the SAME scales as the idle-tick trigger -------
    same_len = len(scales_off) == len(scales_on_sleep) and len(scales_off) > 0
    ident = same_len and all(abs(a - b) < 1e-9 for a, b in zip(scales_off, scales_on_sleep))
    v.require("MECHANISM IDENTITY: sleep-triggered scale vector byte-equal to idle-tick-triggered "
              "(same seed/facts/gains) -- #138 changes WHEN, never WHAT",
              ident, expect=True)
    seed_result["scales_off"] = scales_off
    seed_result["scales_on_sleep"] = scales_on_sleep

    # ---- Check 5: COMPOUNDING GUARD -- re-tick the SAME still-asleep session, no new facts -> no-op --------------
    fp_recheck_before = _weight_fingerprint(comp_on_sleep)
    rec_on_sleep_2 = _tick(ck_sleep, chat_on_sleep, idle_for_sec=CE.SLEEP_IDLE_SEC + 5.0)  # still asleep, later tick
    fp_recheck_after = _weight_fingerprint(comp_on_sleep)
    v.require("COMPOUNDING GUARD: re-ticking an already-consolidated sleeping session is a no-op "
              "(no new facts since the last pass)", rec_on_sleep_2 is None, expect=True)
    v.require("COMPOUNDING GUARD: weights identical across the re-tick (no double rescale)",
              fp_recheck_after == fp_recheck_before, expect=True)

    # ---- Check 6: LESION DELEGATION -- the sleep path still no-ops under BRAIN_DA_ENCODING_LESION=1 ---------------
    _reset_ce_state()
    os.environ["BRAIN_DA_ENCODING_LESION"] = "1"
    comp_on_lesion = _build(seed)
    fp_lesion_before = _weight_fingerprint(comp_on_lesion)
    chat_on_lesion = _StubChat(comp_on_lesion)
    rec_on_lesion = _tick("s%d-on-lesion" % seed, chat_on_lesion, idle_for_sec=CE.SLEEP_IDLE_SEC + 1.0)
    fp_lesion_after = _weight_fingerprint(comp_on_lesion)
    v.require("ON + sleep-depth + LESION: still no-ops (delegates through the SAME faculty self-gate, "
              "does not bypass it)", rec_on_lesion is None, expect=True)
    v.require("LESION: weights unchanged", fp_lesion_after == fp_lesion_before, expect=True)
    os.environ.pop("BRAIN_DA_ENCODING_LESION", None)
    os.environ.pop("BRAIN_DA_ENCODING_SLEEP_TRIGGER", None)
    _reset_ce_state()

    decided = v.decide(go=(not v.unmet and not v.unmeasured))
    seed_result["verdict"] = decided
    return seed_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = []
    n_go = 0
    for seed in args.seeds:
        print("=" * 100)
        print("SEED %d" % seed)
        r = _run_seed(seed, args.out)
        results.append(r)
        if r["verdict"]["status"] == "GO":
            n_go += 1

    agg = {
        "mechanism": "da-encoding-138-sleep-nrem-trigger-wiring",
        "backend": os.environ.get("SIM_BACKEND"),
        "seeds": args.seeds,
        "n_go": n_go,
        "n_seeds": len(args.seeds),
        "aggregate_status": "GO" if n_go == len(args.seeds) else (
            "UNDEFINED" if any(r["verdict"]["status"] == "UNDEFINED" for r in results) else "NO-GO"),
        "per_seed": results,
    }
    print("=" * 100)
    print("AGGREGATE: %d/%d seeds GO -> %s" % (n_go, len(args.seeds), agg["aggregate_status"]))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(agg, f, indent=2, default=str)
        print("wrote", args.out)
    return 0 if agg["aggregate_status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
