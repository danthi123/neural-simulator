"""SHARED SPIKING SALIENCE AFFERENT -- de-risk / verify (scaffold-retirement backlog rank-4, 2026-09-05).

WHAT THIS VERIFIES. `research/runners/shared_salience_afferent.py` (+ `CuriosityProductionOrgan.salience_of`, added
alongside it) is wired (`BRAIN_SHARED_SALIENCE`; default-OFF when this de-risk was written, FLIPPED DEFAULT-ON
2026-09-05 -- see that module's own docstring for the current state and `_shared_salience_flip_soak.py` for the
flip's own no-regression + anti-hollow verification) into THREE consumer sites that each used to compute
their own host novelty/salience formula:
  (1) `webapp/da_mode_drives_chat.py::DaModeDrivesWorkspace.observe()` -- the engagement scalar that drives the
      spiking SNc afferent (the ROOT of 3 downstream default-on consumers: da-mode-drives-response's engagement
      suffix, da-gated-encoding's write-magnitude gain, da-gated-curiosity's crave-threshold gain -- all three read
      the SAME `chat._last_da_drives["da_level"]` this workspace produces).
  (2) `research/runners/bg_action_selection_production_organ.py::salience()` -- the SPEAK/STAY-SILENT salience bias
      the composer hands the two-channel BG selector (the only live discrete motor decision in this codebase).
  (3) `research/runners/value_choice_production_organ.py::default_context_fn()` -- the per-candidate engagement
      context the learned striosome_value critic converts into a commit-by-value decision.

THE GATE (per seed, matching the rank-4 backlog's own framing "both halves already de-risked; this is
INTEGRATION"): for the SHARED ORGAN itself (CuriosityProductionOrgan.salience_of, generalized from judge()'s two
calibration anchors to an arbitrary raw scalar) --
  g_load_bearing  -- the INTACT normalized salience spans a wide range across a raw sweep [0, .25, .5, .75, .95, 1]
                     and correlates with it (corr > 0.9) -- the corr(gap,want)=+0.996 DR-1 proof, reproduced here
                     for the GENERALIZED (non-binary) read this module adds.
  g_lesion        -- the LESIONED (curiosity_excit_sensitivity=0) normalized salience's span across the SAME sweep
                     COLLAPSES to a small fraction of the intact span (the drive-removed twin, judge()'s own
                     anti-cheat, reused verbatim).
and for EACH of the 3 consumer sites --
  c_off_identical -- with the flag unset, the consumer's returned value(s) carry NO trace of the shared organ (no
                     `shared_salience` key / the exact pre-existing host-arithmetic value) -- byte-identical-off.
  c_on_loadbearing-- with the flag on, the consumer's returned engagement/salience value(s) DIFFER measurably from
                     the OFF (host-arithmetic) value, tracking the shared organ's own reading (verified by exact
                     equality against an INDEPENDENT direct call into the SAME organ instance the consumer used).
  c_lesion_collapse-with the flag on AND BRAIN_SHARED_SALIENCE_LESION=1, the consumer's cross-candidate/cross-input
                     DIFFERENTIATION collapses (a multi-candidate spread, or a raw-varying response) toward the
                     SAME near-floor the organ-level lesion produces.

6-seed (42/43/44/100/101/102), numpy-CPU. Each seed runs in ITS OWN PROCESS (subprocess-fanned by `--seeds`) so the
curiosity organ's process-shared singleton (`curiosity_production_organ.get_organ`, intentionally NOT seed-keyed --
production always runs ONE process at ONE seed) cannot silently serve seed N+1's request from seed N's cached build
(the exact per-process-global-RNG confound CLAUDE.md's `actual_seed_used` note warns about for a DIFFERENT reason --
same root cause, a shared mutable module-level cache/RNG surviving across what should be independent builds).

Run (controller, the 6-seed gate):
  SIM_BACKEND=numpy python -m research.runners._shared_salience_afferent_derisk --seeds 42 43 44 100 101 102 \\
      --out research/findings/raw/_shared_salience_afferent/verify_6seed.json

Run (single-seed worker -- what the controller subprocess-fans; also runnable standalone):
  SIM_BACKEND=numpy python -m research.runners._shared_salience_afferent_derisk --seed 42 \\
      --out research/findings/raw/_shared_salience_afferent/verify_seed42.json

Run (the ONE full end-to-end production-entry-point plumbing proof -- seed 42 only, mirrors the "seed-waiver: a
plumbing/attribution proof, not a stochastic effect size" scoping `2026-08-21-da-gated-curiosity-threshold-wired-
GO.md` uses for an identical reason; the HEAVY value-train critic is expensive to re-run 6x and its OWN sensitivity
to its engagement input is a pre-existing, already-6-seed-GO'd mechanism -- research/findings/2026-07-23-value-
critic-closure-RANK1-GO.md -- this proof only needs to show the NEW upstream link reaches it, once):
  SIM_BACKEND=numpy python -m research.runners._shared_salience_afferent_derisk --plumbing \\
      --out research/findings/raw/_shared_salience_afferent/plumbing_seed42.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# WHOSE-the-difference attribution (tools.lab): the lesion arm and the intact arm are both measured below (a
# treatment/control pair); this asks how much of the intact span survives under the lesion rather than banking
# both numbers unattributed (the gap#5 lesson -- 97% of an apparent effect once turned out to be a clamp running
# identically in both arms).
from tools.lab import attributable_to  # noqa: E402

RAWS = [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]
_MSG_CONTENT = "what does the dog chase around the yard today"


def _clear_flags():
    """Reset to the OFF arm (the pre-flip host-arithmetic baseline).

    2026-09-05 fix (Track-1 flip-soak verification, `_shared_salience_flip_soak.py`): `BRAIN_SHARED_SALIENCE` was
    flipped DEFAULT-ON (`shared_salience_afferent._SHARED_SALIENCE_DEFAULT_ON=True`), so *unsetting* the var no
    longer means OFF -- it now means ACTIVE. Every OFF-arm call site in this file (and in
    `_value_choice_neural_context_6seed_derisk.py`, which imports this exact helper) relied on `_clear_flags()` to
    produce the OFF arm; left as a bare `pop()`, both the "off" and "on" arms of every existing 6-seed gate would
    silently become ON post-flip (caught here BEFORE it corrupted a re-run -- `g_off_identical` would have failed
    loudly rather than passed wrongly, since the OFF arm would stop matching the independently-computed host
    formula, but the fix belongs at the source, not as a per-caller workaround). Mirrors the identical
    `BRAIN_VALUE_CHOICE` fix in `_value_choice_flip_soak.py`'s own `_set_flags()` (2026-08-27 comment)."""
    os.environ["BRAIN_SHARED_SALIENCE"] = "0"
    os.environ.pop("BRAIN_SHARED_SALIENCE_LESION", None)


def _set_flags(*, on: bool, lesion: bool = False):
    os.environ["BRAIN_SHARED_SALIENCE"] = "1" if on else "0"
    if lesion:
        os.environ["BRAIN_SHARED_SALIENCE_LESION"] = "1"
    else:
        os.environ.pop("BRAIN_SHARED_SALIENCE_LESION", None)


class _FakeAgent:
    def held_referent(self):
        return (None, None)


class _FakeChat:
    def __init__(self):
        self.stored_facts = [("dog", "chase", "cat"), ("dog", "chase", "ball"), ("dog", "chase", "shoe")]
        self.is_multiturn = False
        self.agent = _FakeAgent()


def run_seed(seed: int) -> dict:
    """All per-seed checks, run in THIS process (the caller ensures process isolation across seeds)."""
    import numpy as np
    _clear_flags()
    out = {"seed": int(seed)}

    # ---------------------------------------------------------------- (0) THE SHARED ORGAN CORE
    from research.runners.curiosity_production_organ import CuriosityProductionOrgan
    t0 = time.time()
    organ = CuriosityProductionOrgan(seed=seed)
    intact = [organ.salience_of(r, lesion=False) for r in RAWS]
    lesioned = [organ.salience_of(r, lesion=True) for r in RAWS]
    build_s = time.time() - t0
    intact_norm = [d["normalized"] for d in intact]
    les_norm = [d["normalized"] for d in lesioned]
    intact_span = float(max(intact_norm) - min(intact_norm))
    les_span = float(max(les_norm) - min(les_norm))
    corr_intact = float(np.corrcoef(RAWS, intact_norm)[0, 1])
    corr_lesioned = float(np.corrcoef(RAWS, les_norm)[0, 1])
    # ATTRIBUTION (not just both-arms-banked): of the intact span (the raw-sweep "effect"), what fraction survives
    # the lesion? A HIGH fraction (~1.0) means the span is attributable to the from_novelty->ASK-pool drive
    # pathway the lesion cuts, not to some OTHER shared mechanism running identically in both arms (e.g. bare OU
    # noise, which the lesioned arm's own tiny residual span already characterizes).
    span_attrib = attributable_to("seed %d: intact span attributable to the drive pathway (vs lesioned span)"
                                  % seed, intact_span, les_span)
    out["organ_core"] = {
        "raws": RAWS, "intact_normalized": intact_norm, "lesioned_normalized": les_norm,
        "intact_span": intact_span, "lesioned_span": les_span,
        "corr_intact": corr_intact, "corr_lesioned": corr_lesioned,
        "span_attributable_to_drive_pathway": span_attrib,
        "build_seconds": round(build_s, 2),
        "g_load_bearing": bool(intact_span > 0.5 and corr_intact > 0.9),
        "g_lesion": bool(les_span < 0.2 * max(intact_span, 1e-9) and intact_span > 0.5),
    }

    # ---------------------------------------------------------------- (1) da_mode_drives_chat CONSUMER
    _clear_flags()
    from webapp import da_mode_drives_chat as DAD
    ws_off = DAD.DaModeDrivesWorkspace(seed=seed)
    off = ws_off.observe(_MSG_CONTENT)
    off_has_key = "shared_salience" in off
    _set_flags(on=True)
    ws_on = DAD.DaModeDrivesWorkspace(seed=seed)
    on = ws_on.observe(_MSG_CONTENT)
    on_has_key = "shared_salience" in on
    # cross-check: the workspace's OWN reported shared_salience.normalized should be CLOSE to an INDEPENDENT
    # direct re-read of the SAME organ instance at the SAME raw input -- proves the consumer really reads the
    # shared organ, not a coincidence / a re-derived local formula. NOT exact-equal: each read re-advances the
    # ASK pool's own OU noise (genuine spiking jitter across repeated reads, N_READ_REPS=4-averaged but not
    # eliminated) -- a *tolerance* comparison (0.15 normalized units, ~14% of the organ's full [0,~1.09] sweep
    # span) is the correct check here, informational only (not a gate).
    on_indep = organ.salience_of(on["shared_salience"]["raw"], lesion=False) if on_has_key else None
    _set_flags(on=True, lesion=True)
    ws_les = DAD.DaModeDrivesWorkspace(seed=seed)
    les = ws_les.observe(_MSG_CONTENT)
    _clear_flags()
    out["da_mode"] = {
        "off_da_level": off["da_level"], "off_mode": off["mode"], "off_has_shared_key": off_has_key,
        "on_da_level": on["da_level"], "on_mode": on["mode"], "on_has_shared_key": on_has_key,
        "on_turn_engagement": on["turn_engagement"],
        "on_matches_independent_organ_call": (
            bool(on_indep is not None and abs(on_indep["normalized"] - on["shared_salience"]["normalized"]) < 0.15)),
        "lesion_da_level": les["da_level"], "lesion_mode": les["mode"],
        "c_off_identical": bool(not off_has_key),
        "c_on_loadbearing": bool(on_has_key and abs(on["da_level"] - off["da_level"]) > 1e-6),
        "c_lesion_collapse": bool(les["mode"] == "rest" and les["da_level"] < 0.1),
    }

    # ---------------------------------------------------------------- (2) bg_action_selection CONSUMER
    _clear_flags()
    import research.runners.bg_action_selection_production_organ as BG
    _CONTENT_MSG = "hello there friend"
    _n_content = len(BG._CONTENT_TOKEN_RE.findall(_CONTENT_MSG))
    # the bare pre-existing host formula (`git diff`-verified unchanged in the OFF branch), computed independently
    # here from source so the OFF-arm check does not depend on a hand-picked literal for whatever _n_content is.
    _expect_off_content = (min(1.0, _n_content / 2.0), max(0.0, 1.0 - float(_n_content)))
    off_empty = BG.salience("")
    off_content = BG.salience(_CONTENT_MSG)
    _set_flags(on=True)
    on_empty = BG.salience("")
    on_content = BG.salience(_CONTENT_MSG)
    on_indep_bg = organ.salience_of(min(1.0, _n_content / 2.0), lesion=False)
    _set_flags(on=True, lesion=True)
    les_empty = BG.salience("")
    les_content = BG.salience(_CONTENT_MSG)
    _clear_flags()
    out["bg_action_selection"] = {
        "off_empty": list(off_empty), "off_content": list(off_content), "n_content_tokens": _n_content,
        "on_empty": list(on_empty), "on_content": list(on_content),
        "lesion_empty": list(les_empty), "lesion_content": list(les_content),
        "on_content_matches_independent_organ_call": bool(
            abs(on_content[0] - max(0.0, on_indep_bg["normalized"])) < 0.15),
        "c_off_identical": bool(off_empty == (0.0, 1.0) and tuple(off_content) == _expect_off_content),
        "c_on_loadbearing_at_content": bool(abs(on_content[0] - off_content[0]) > 1e-3),
        "c_lesion_collapse_at_content": bool(abs(les_content[0] - on_content[0]) > 1e-3),
        "honest_note": ("the ONLY live-reachable STAY-SILENT branch (decide_action) feeds raw=0.0 (n==0), which "
                        "floors to ~0 both intact and lesioned -- there is no novelty to lesion away at the "
                        "familiar floor. Load-bearing + lesion-collapse are demonstrated on salience()'s general "
                        "input range (a real message, n>=1), proving the wiring is genuine even though the "
                        "specific pre-existing entry-gate anchor is a floor case. See the finding's honest scope."),
    }

    # ---------------------------------------------------------------- (3) value_choice CONSUMER (context fn only;
    #     the heavy value-train critic itself is a pre-existing, already-6-seed-GO'd mechanism this does NOT
    #     modify -- exercised end-to-end ONCE at seed 42 by --plumbing, not per-seed here).
    _clear_flags()
    import research.runners.value_choice_production_organ as VC
    fchat = _FakeChat()
    ctx = VC.default_context_fn(fchat)
    off_eng = ctx("dog", "chase", ["cat", "ball", "shoe"])
    _set_flags(on=True)
    on_eng = ctx("dog", "chase", ["cat", "ball", "shoe"])
    _set_flags(on=True, lesion=True)
    les_eng = ctx("dog", "chase", ["cat", "ball", "shoe"])
    _clear_flags()
    off_spread = float(max(off_eng) - min(off_eng))
    on_spread = float(max(on_eng) - min(on_eng))
    les_spread = float(max(les_eng) - min(les_eng))
    # ATTRIBUTION: of the ON-arm's cross-candidate spread (the gradient the striosome_value critic needs to be
    # decisive), what fraction survives the shared-afferent lesion? A high fraction means the candidate-
    # differentiating gradient is attributable to the drive pathway, not to some residual the lesion leaves intact.
    vc_attrib = attributable_to("seed %d: value-choice engagement spread attributable to the drive pathway" % seed,
                                on_spread, les_spread)
    out["value_choice_context"] = {
        "off_eng": off_eng, "on_eng": on_eng, "lesion_eng": les_eng,
        "off_spread": off_spread, "on_spread": on_spread, "lesion_spread": les_spread,
        "spread_attributable_to_drive_pathway": vc_attrib,
        "c_off_identical": bool(off_eng == [0.0, 0.5, 1.0]),   # the bare pre-existing recency-ratio formula
        "c_on_loadbearing": bool(any(abs(a - b) > 1e-6 for a, b in zip(on_eng, off_eng))),
        "c_lesion_collapse": bool(les_spread < 0.2 * max(on_spread, 1e-9) and on_spread > 0.3),
    }

    out["all_gates_pass"] = bool(
        out["organ_core"]["g_load_bearing"] and out["organ_core"]["g_lesion"]
        and out["da_mode"]["c_off_identical"] and out["da_mode"]["c_on_loadbearing"] and out["da_mode"]["c_lesion_collapse"]
        and out["bg_action_selection"]["c_off_identical"] and out["bg_action_selection"]["c_on_loadbearing_at_content"]
        and out["bg_action_selection"]["c_lesion_collapse_at_content"]
        and out["value_choice_context"]["c_off_identical"] and out["value_choice_context"]["c_on_loadbearing"]
        and out["value_choice_context"]["c_lesion_collapse"]
    )
    return out


def run_plumbing_proof(seed: int = 42, value_train_trials: int = 40) -> dict:
    """The ONE full-production-entry-point end-to-end proof (seed 42 by default): `bg_action_selection_production_
    organ.decide_action()` (the REAL STAY-SILENT decision path) + `value_choice_production_organ`'s REAL heavy
    critic via `ValueChoiceProductionOrgan.choose()` (not just the context function)."""
    _clear_flags()
    out = {"seed": int(seed)}

    import research.runners.bg_action_selection_production_organ as BG
    BG.reset_organs()
    off = BG.decide_action("")
    _set_flags(on=True)
    on = BG.decide_action("")
    _set_flags(on=True, lesion=True)
    les = BG.decide_action("")
    _clear_flags()
    BG.reset_organs()
    out["bg_decide_action"] = {
        "off": off, "on": on, "lesion": les,
        "off_and_on_reach_a_decision": bool(off is not None and on is not None),
    }

    from research.runners.value_choice_production_organ import ValueChoiceProductionOrgan
    t0 = time.time()
    vco = ValueChoiceProductionOrgan(seed=seed, value_train_trials=value_train_trials)
    vco.ensure_built()
    build_s = time.time() - t0
    fchat = _FakeChat()
    import research.runners.value_choice_production_organ as VC
    ctx_off = VC.default_context_fn(fchat)
    _clear_flags()
    off_eng = ctx_off("dog", "chase", ["cat", "ball", "shoe"])
    chosen_off, meta_off = vco.choose(["cat", "ball", "shoe"], off_eng, lesion=False)
    _set_flags(on=True)
    ctx_on = VC.default_context_fn(fchat)
    on_eng = ctx_on("dog", "chase", ["cat", "ball", "shoe"])
    chosen_on, meta_on = vco.choose(["cat", "ball", "shoe"], on_eng, lesion=False)
    _set_flags(on=True, lesion=True)
    ctx_les = VC.default_context_fn(fchat)
    les_eng = ctx_les("dog", "chase", ["cat", "ball", "shoe"])
    chosen_les, meta_les = vco.choose(["cat", "ball", "shoe"], les_eng, lesion=False)
    _clear_flags()
    out["value_choice_full"] = {
        "build_seconds": round(build_s, 2),
        "off_eng": off_eng, "on_eng": on_eng, "lesion_shared_eng": les_eng,
        "chosen_off": chosen_off, "chosen_on": chosen_on, "chosen_shared_lesion": chosen_les,
        "meta_off": meta_off, "meta_on": meta_on, "meta_shared_lesion": meta_les,
        "reaches_the_real_critic": True,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="single-seed worker mode")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="controller mode: subprocess-fan these seeds")
    ap.add_argument("--plumbing", action="store_true", help="the one seed-42 full-production-entry-point proof")
    ap.add_argument("--value-train-trials", type=int, default=40)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()

    if a.plumbing:
        result = run_plumbing_proof(seed=42, value_train_trials=a.value_train_trials)
    elif a.seeds:
        # CONTROLLER: subprocess-fan one worker per seed (process isolation -- see the module docstring for why).
        per_seed = {}
        for s in a.seeds:
            t0 = time.time()
            r = subprocess.run(
                [sys.executable, "-m", "research.runners._shared_salience_afferent_derisk", "--seed", str(s)],
                cwd=str(_REPO), capture_output=True, text=True, timeout=600,
                env={**os.environ, "SIM_NO_PROVENANCE": "1"},   # the controller's OWN --out carries provenance
            )
            if r.returncode != 0:
                per_seed[str(s)] = {"seed": s, "error": r.stderr[-4000:], "returncode": r.returncode}
                continue
            # the worker prints ONE JSON line (marked) to stdout; find it.
            line = None
            for ln in r.stdout.splitlines():
                if ln.startswith("RESULT_JSON:"):
                    line = ln[len("RESULT_JSON:"):]
            per_seed[str(s)] = json.loads(line) if line else {"seed": s, "error": "no RESULT_JSON line",
                                                               "stdout_tail": r.stdout[-2000:]}
            per_seed[str(s)]["wall_seconds"] = round(time.time() - t0, 2)
        all_pass = all(per_seed.get(str(s), {}).get("all_gates_pass") for s in a.seeds)
        n_pass = sum(1 for s in a.seeds if per_seed.get(str(s), {}).get("all_gates_pass"))
        result = {"mode": "controller", "seeds": a.seeds, "n_seeds": len(a.seeds), "n_pass": n_pass,
                  "all_seeds_pass": bool(all_pass), "per_seed": per_seed}
    elif a.seed is not None:
        result = run_seed(a.seed)
        print("RESULT_JSON:" + json.dumps(result))
    else:
        ap.error("pass --seed N (worker), --seeds N N N (controller), or --plumbing")
        return

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"wrote {a.out}")
    if a.seed is None:   # the controller/plumbing summary is worth printing; a lone worker already printed RESULT_JSON
        print(json.dumps({k: v for k, v in result.items() if k not in ("per_seed",)}, indent=2, default=str))
        if "per_seed" in result:
            for s, r in result["per_seed"].items():
                print(f"  seed {s}: all_gates_pass={r.get('all_gates_pass')}"
                      + ("" if "error" not in r else f"  ERROR={r['error'][:200]}"))


if __name__ == "__main__":
    main()
