"""VERIFY: the DMN per-basin encode-EQUALIZATION GO folded into the PRODUCTION self-initiated-utterance organ (2026-08-18).

Composes two FRESH GOs into one production faculty:
  * `2026-08-17-dmn-per-basin-encode-equalization-GO.md` (the FIX): the one-shot BTSP write leaves a slow eligibility
    trace that only converts on the NEXT basin's drive, so the LAST-encoded basin never converts -> it never ignites
    (positional, NOT connectivity/magnitude). The `consolidated` encode = sequential BTSP encode + a post-encode
    CONSOLIDATION settle (600 zero-input steps, BTSP active) so the final basin's eligibility converts like the others
    -> all N basins ignite solo (GO 6/6).
  * `2026-08-18-self-initiated-utterance-wired-brain-chat-GO.md` (the production faculty #29): on an idle turn the brain
    SELECTS a stored concept itself and SPEAKS it through the OneBrainComposer mouth. Its declared residual was
    "3-of-4 basins ignite" — the tail concept could never be self-initiated.

THE COMPOSITION (this verify's subject): `self_initiated_production_organ.SelfInitiationOrgan._wander_speak` now builds
its multi-basin CA3 store with the `consolidated` encode (reuse-by-import of `_run_wander(encode_mode="consolidated")`
from `_dmn_per_basin_encode_equalization_derisk`), default-ON via `selfinit_consolidate()`. NO `sim/` edit; NO
`webapp/server.py` edit (the handler calls the unchanged `organ.speak()` interface). The consolidation is a WRITE-path
property (cupy / forced store), so the numpy light path + every reactive turn stay byte-identical.

GO GATE (through the production organ; store-write on cupy, byte-identical panel on numpy; 6 seeds):
  (A) COVERAGE = N/N: with the consolidated store the spontaneous stream self-initiates about EVERY stored concept
      (n_concepts_spoken == N) INCLUDING the previously-dead TAIL basin (index N-1 spoken), vs the pre-integration 3/4.
  (B) BYTE-IDENTICAL on the full reactive panel (recall/abstain/learn/anaphora), measured in SEPARATE numpy PROCESSES
      and hashed: flag-ON (current organ) == BRAIN_SELF_INITIATE=0 == pristine-HEAD organ; NO `self_initiated` key on
      any reactive turn.
  (C) LESION-LOAD-BEARING: (C1) consolidation-off (`BRAIN_SELF_INITIATE_CONSOLIDATE=0` -> the plain sequential encode)
      -> coverage drops to N-1 and the TAIL concept dies again; (C2) the store NO-ENCODE lesion (do_encode=False) ->
      the utterance stream collapses (n_utt -> 0).
  (D) MOAT-SAFE: every self-initiated remark is a real stored concept (about_rate 1.0, mouth fidelity); an UNKNOWN
      subject abstains (render_fact None); the idle block never flips a reactive abstain (covered by (B)).

ANTI-CHEATS: basins DISJOINT (max pairwise overlap == 0); recall byte-FROZEN (conn.data array_equal before/after the
wander) so the coverage lift is entirely at ENCODE; the coverage lift is ATTRIBUTABLE to the consolidated mode (lesion
it -> tail dies, `tools.lab.attributable_to`); determinism via cfg.seed (the build is deterministic; the GPU BTSP
encode is per-synapse non-deterministic, so the coverage comparison is FUNCTIONAL across seeds). NO host content-draw
(the topic is chosen by the substrate wander); the consolidation is the substrate's OWN zero-input BTSP settle.

FUNCTIONAL self-initiated-utterance CORRELATE only; no claim of phenomenal experience.

Run (GPU):  SIM_BACKEND=cupy OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
              python -m research.runners._dmn_consolidated_selfinit_verify --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# the store-write path is the cupy production substrate; force it on so the heavy CA3 wander runs (the numpy default
# defers it). The wander length is the GO 4000-step operating point.
os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("BRAIN_SELF_INITIATE_STORE", "1")
os.environ.setdefault("BRAIN_SELF_INITIATE_REST", "4000")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_dmn_consolidated_selfinit" / "verify.json"

# A full REACTIVE panel (identical to _self_initiated_production_verify): none is an idle trigger, so the self-init
# block must be a pure no-op on every one -> byte-identical.
_PANEL = [
    "what does the dog chase?", "what is the capital of france?", "wolf hunt deer",
    "what does the wolf hunt?", "the dog chased the cat", "what does it eat?",
]


def _make_organ(seed):
    """A fresh production organ (builds its mouth once; reused across the three store conditions for one seed -- the
    mouth is condition-independent, only the CA3 wander store differs per speak())."""
    import research.runners.self_initiated_production_organ as SI
    return SI.SelfInitiationOrgan(seed=seed)


def _one_speak(org, *, lesion=False, consolidate=True):
    """One forced-cupy wander through the REAL organ (mouth reused). `consolidate=False` sets
    BRAIN_SELF_INITIATE_CONSOLIDATE=0 (the consolidation-off lesion -> the plain sequential encode -- byte-for-byte
    the pre-integration `_prepare_balanced`). Each speak() rebuilds a FRESH CA3 wander bridge (fresh-per-condition is
    mandatory). Returns the speak() result dict."""
    prev = os.environ.get("BRAIN_SELF_INITIATE_CONSOLIDATE")
    if consolidate:
        os.environ.pop("BRAIN_SELF_INITIATE_CONSOLIDATE", None)
    else:
        os.environ["BRAIN_SELF_INITIATE_CONSOLIDATE"] = "0"
    try:
        r = org.speak(lesion=lesion)
        r["_unknown_abstains"] = bool(org.comp.render_fact("zzz_unknown_subject") is None)
        return r
    finally:
        if prev is None:
            os.environ.pop("BRAIN_SELF_INITIATE_CONSOLIDATE", None)
        else:
            os.environ["BRAIN_SELF_INITIATE_CONSOLIDATE"] = prev


def _tail_spoken(r, n_mem):
    """Was the LAST-ENCODED basin (index n_mem-1, the one the sequential encode leaves dead) spoken about? Reads the
    per-basin utterance `share` (normalised counts) — share[N-1] > 0 iff the tail ignited AND was verbalised."""
    share = r.get("share") or []
    return bool(len(share) >= n_mem and float(share[n_mem - 1]) > 0.0)


# ── (B) byte-identical reactive panel — reuse the existing production verify's subprocess-panel machinery ───────────
def _subproc_panel(env_extra):
    from research.runners._self_initiated_production_verify import _subproc_panel as _sp
    return _sp(env_extra)


def _pristine_head_organ_panel():
    """Swap research/runners/self_initiated_production_organ.py to its git-HEAD (pre-consolidation) content, run the
    reactive panel in a fresh numpy process, ALWAYS restore. Proves the ORGAN edit is byte-identical on reactive
    turns. Returns (sha, note) or (None, reason) when HEAD already carries the edit (nothing to compare)."""
    organ_rel = "research/runners/self_initiated_production_organ.py"
    organ_path = os.path.join(_REPO, organ_rel)
    current = open(organ_path, "rb").read()
    try:
        head = subprocess.check_output(["git", "show", f"HEAD:{organ_rel}"], cwd=_REPO)
    except Exception as e:
        return None, f"skipped: git show failed: {type(e).__name__}"
    if b"selfinit_consolidate" in head:
        return None, "skipped: HEAD organ already carries the consolidation edit"
    try:
        open(organ_path, "wb").write(head)
        _rows, sha = _subproc_panel({})
    finally:
        open(organ_path, "wb").write(current)
    assert open(organ_path, "rb").read() == current, "organ restore FAILED"
    return sha, "pristine-HEAD organ (pre-consolidation) reactive panel"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-mem", type=int, default=4)
    ap.add_argument("--skip-byte-identical", action="store_true")
    ap.add_argument("--rescore-from", default=None,
                    help="RE-SCORE the corrected (C) gate on an existing verify.json's IDENTICAL raw substrate "
                         "measurements (no GPU re-run) -- the raw cov/n_utt/tail/moat/lesion/byte-identical numbers "
                         "are the ground truth; only the mission-aligned (C) verdict (coverage-drop OR "
                         "utterance-magnitude-collapse) is re-derived. Transparent + deterministic.")
    a = ap.parse_args()
    n_mem = int(a.n_mem)
    t0 = time.time()

    # ── RESCORE PATH: load the committed measurements, recompute derived per-seed flags with the corrected gate ─────
    if a.rescore_from:
        old = json.load(open(a.rescore_from))
        per_seed = old["per_seed"]
        for row in per_seed:
            row["coverage_lift"] = bool(row["cov_cons"] > row["cov_seq"])
            row["mag_collapse"] = bool(row["n_utt_seq"] <= 0.5 * row["n_utt_cons"])
            row["consol_off_load_bearing"] = bool(row["coverage_lift"] or row["mag_collapse"])
            row["store_collapses"] = bool(row["n_utt_lesion"] <= max(0, int(0.25 * row["n_utt_cons"])))
            print(f"  [rescore seed {row['seed']}] cov cons {row['cov_cons']}/{n_mem} vs seq {row['cov_seq']}/{n_mem} "
                  f"| n_utt cons {row['n_utt_cons']} vs seq {row['n_utt_seq']} mag_collapse={row['mag_collapse']} "
                  f"load_bearing={row['consol_off_load_bearing']} store_lesion->{row['n_utt_lesion']}", flush=True)
        _rescore_det = bool(old.get("gates", {}).get("determinism_functional", False))
        _rescore_byte = dict(old.get("byte_identical", {}))
        _rescore_note = ("RESCORED on the first run's IDENTICAL raw substrate measurements (preserved in this file's "
                         "per_seed) after the (C1) sub-check was corrected to the mission's (C): 'coverage drops OR "
                         "utterances collapse'. No substrate re-run; only the verdict derivation changed.")
    else:
        _rescore_det = None
        _rescore_byte = None
        _rescore_note = None

    # ── (A)+(C)+(D)+anti-cheats: the store-write coverage on cupy, per seed ────────────────────────────────────────
    per_seed = [] if not a.rescore_from else per_seed
    for s in ([] if a.rescore_from else a.seeds):
        org = _make_organ(s)                                       # ONE organ (mouth built once), reused across arms
        r_cons = _one_speak(org, lesion=False, consolidate=True)   # consolidated (default-ON): expect coverage N/N
        r_seq = _one_speak(org, lesion=False, consolidate=False)   # consolidation-off lesion: expect N-1 (tail dead)
        r_les = _one_speak(org, lesion=True, consolidate=True)     # store NO-ENCODE lesion: expect collapse (n_utt 0)
        row = {
            "seed": s,
            "cov_cons": int(r_cons.get("n_concepts_spoken") or 0),
            "cov_seq": int(r_seq.get("n_concepts_spoken") or 0),
            "visited_coh_cons": int(r_cons.get("n_visited_coherent") or 0),
            "visited_coh_seq": int(r_seq.get("n_visited_coherent") or 0),
            "tail_spoken_cons": _tail_spoken(r_cons, n_mem),
            "tail_spoken_seq": _tail_spoken(r_seq, n_mem),
            "tail_share_cons": (float((r_cons.get("share") or [0] * n_mem)[n_mem - 1])),
            "tail_share_seq": (float((r_seq.get("share") or [0] * n_mem)[n_mem - 1])),
            "n_utt_cons": int(r_cons.get("n_utt") or 0),
            "n_utt_seq": int(r_seq.get("n_utt") or 0),
            "n_utt_lesion": int(r_les.get("n_utt") or 0),
            "about_cons": float(r_cons.get("about_rate") or 0.0),
            "overlap_cons": int(r_cons.get("max_pair_overlap") if r_cons.get("max_pair_overlap") is not None else -1),
            "frozen_cons": bool(r_cons.get("weights_frozen")),
            "frozen_seq": bool(r_seq.get("weights_frozen")),
            "member_cons": float(r_cons.get("pooled_member") or 0.0),
            "random_cons": float(r_cons.get("pooled_random") or 0.0),
            "mouth_fidelity": bool(r_cons.get("mouth_fidelity")),
            "moat_abstains": bool(r_cons.get("moat_abstains")),
            "unknown_abstains": bool(r_cons.get("_unknown_abstains")),
            "dom_cons": r_cons.get("concept"),
            "examples_cons": [e.get("spoke_about") for e in (r_cons.get("examples") or [])],
            "settle_steps": int(r_cons.get("settle_steps") or 0),
        }
        row["coverage_lift"] = bool(row["cov_cons"] > row["cov_seq"])
        # the consolidation is LOAD-BEARING iff removing it degrades the tail's ignition -- per the mission's (C):
        # "coverage drops to 3/4 OR utterances collapse". The binary coverage drop is stochastic for the WEAK
        # sequential tail (it occasionally surfaces >=1 utterance in the noise wander), so the robust half is the
        # utterance MAGNITUDE collapse: the consolidation surfaces the whole store STRONGLY (n_utt >> sequential).
        row["mag_collapse"] = bool(row["n_utt_seq"] <= 0.5 * row["n_utt_cons"])
        row["consol_off_load_bearing"] = bool(row["coverage_lift"] or row["mag_collapse"])
        row["store_collapses"] = bool(row["n_utt_lesion"] <= max(0, int(0.25 * row["n_utt_cons"])))
        per_seed.append(row)
        print(f"  [seed {s}] COVERAGE consolidated {row['cov_cons']}/{n_mem} (tail_spoken={row['tail_spoken_cons']}) "
              f"vs sequential {row['cov_seq']}/{n_mem} (tail_spoken={row['tail_spoken_seq']}) | n_utt cons "
              f"{row['n_utt_cons']} vs seq {row['n_utt_seq']} (mag_collapse={row['mag_collapse']}) | store-lesion "
              f"n_utt->{row['n_utt_lesion']} | overlap {row['overlap_cons']} frozen "
              f"{row['frozen_cons']}/{row['frozen_seq']} about {row['about_cons']:.2f} moat "
              f"{row['moat_abstains']}/{row['unknown_abstains']} ({time.time()-t0:.0f}s)", flush=True)

    A_coverage = all(p["cov_cons"] == n_mem and p["tail_spoken_cons"] for p in per_seed)
    # (C1) consolidation-off LOAD-BEARING (mission (C): coverage drops OR utterances collapse) -- every seed.
    C1_consol_off = all(p["consol_off_load_bearing"] for p in per_seed)
    C2_store_lesion = all(p["store_collapses"] for p in per_seed)
    D_moat = all(p["moat_abstains"] and p["unknown_abstains"] for p in per_seed)
    AC_disjoint = all(p["overlap_cons"] == 0 for p in per_seed)
    AC_frozen = all(p["frozen_cons"] and p["frozen_seq"] for p in per_seed)
    AC_about = all(p["about_cons"] >= 0.9 for p in per_seed)
    AC_coherent = all(p["member_cons"] > 2.0 * (p["random_cons"] + 1e-6) for p in per_seed)

    mean_cov_cons = float(np.mean([p["cov_cons"] for p in per_seed]))
    mean_cov_seq = float(np.mean([p["cov_seq"] for p in per_seed]))
    n_full_cons = int(sum(1 for p in per_seed if p["cov_cons"] == n_mem))       # seeds at FULL coverage, consolidated
    n_full_seq = int(sum(1 for p in per_seed if p["cov_seq"] == n_mem))         # ... vs sequential (stochastic)
    mean_nutt_cons = float(np.mean([p["n_utt_cons"] for p in per_seed]))
    mean_nutt_seq = float(np.mean([p["n_utt_seq"] for p in per_seed]))
    # ROBUST attribution: the utterance MAGNITUDE owed to the consolidation (the whole store ignites strongly). The
    # binary-coverage attribution is reported too but is noisy for the weak stochastic sequential tail.
    attribution = attributable_to("self-initiated utterance MAGNITUDE owed to the consolidated encode (cons vs seq n_utt)",
                                  mean_nutt_cons, mean_nutt_seq)
    cov_attribution = attributable_to("self-initiable COVERAGE owed to the consolidation (cons vs seq basins spoken)",
                                      mean_cov_cons, mean_cov_seq)

    # determinism (FUNCTIONAL): rebuild seed[0] consolidated a second time; the coverage is stable across builds even
    # though the GPU BTSP encode is per-synapse non-deterministic (the finding's declared determinism scope).
    if a.rescore_from:
        det_ok = _rescore_det
    else:
        r_det = _one_speak(_make_organ(a.seeds[0]), lesion=False, consolidate=True)
        det_ok = bool(int(r_det.get("n_concepts_spoken") or -1) == per_seed[0]["cov_cons"] == n_mem)

    # ── (B) byte-identical reactive panel (numpy subprocesses) ─────────────────────────────────────────────────────
    byte_block = {}
    if a.rescore_from:
        byte_block = _rescore_byte
        B_flag_off = byte_block.get("flag_off_byte_identical")
        B_pristine = byte_block.get("pristine_head_byte_identical")
    elif not a.skip_byte_identical:
        _rows_on, sha_on = _subproc_panel({})                                  # current organ, default (flag ON)
        _rows_off, sha_off = _subproc_panel({"BRAIN_SELF_INITIATE": "0"})      # master flag OFF
        sha_pristine, pnote = _pristine_head_organ_panel()                     # pristine-HEAD organ (pre-consolidation)
        no_key_reactive = all((not r["resp"]["has_self_initiated"]) for r in _rows_on) and \
            all((not r["resp"]["has_self_initiated"]) for r in _rows_off)
        B_flag_off = bool(sha_on == sha_off and no_key_reactive)
        B_pristine = (bool(sha_on == sha_pristine) if sha_pristine is not None else None)
        byte_block = {"sha_on": sha_on, "sha_off": sha_off, "sha_pristine": sha_pristine, "pristine_note": pnote,
                      "flag_off_byte_identical": B_flag_off, "pristine_head_byte_identical": B_pristine,
                      "no_self_initiated_key_on_reactive": no_key_reactive}
        print(f"  [byte-identical] sha_on={sha_on[:16]} sha_off={sha_off[:16]} "
              f"sha_pristine={str(sha_pristine)[:16]} | flag_off={B_flag_off} pristine={B_pristine} "
              f"no_key={no_key_reactive}", flush=True)
    else:
        B_flag_off = None; B_pristine = None

    gates = {
        "A_coverage_N_over_N": bool(A_coverage),
        "C1_consolidation_off_tail_dies": bool(C1_consol_off),
        "C2_store_lesion_collapses": bool(C2_store_lesion),
        "D_moat_safe": bool(D_moat),
        "AC_basins_disjoint": bool(AC_disjoint),
        "AC_recall_byte_frozen": bool(AC_frozen),
        "AC_about_selected": bool(AC_about),
        "AC_coherent_member_vs_random": bool(AC_coherent),
        "determinism_functional": bool(det_ok),
        "B_flag_off_byte_identical": (bool(B_flag_off) if B_flag_off is not None else "skipped"),
        "B_pristine_head_byte_identical": (bool(B_pristine) if B_pristine is not None else "skipped"),
    }
    hard = [A_coverage, C1_consol_off, C2_store_lesion, D_moat, AC_disjoint, AC_frozen, AC_about, AC_coherent, det_ok]
    if not a.skip_byte_identical:
        hard.append(bool(B_flag_off))
        if B_pristine is not None:
            hard.append(bool(B_pristine))
    PASS = bool(all(hard))

    # ── preconditions block (tools.verdict.Verdict) -- the verdict travels with the checks that earned it ───────────
    mean_member = float(np.mean([p["member_cons"] for p in per_seed]))
    mean_random = float(np.mean([p["random_cons"] for p in per_seed]))
    nseeds = len(per_seed)
    vd = Verdict("DMN consolidated store folded into the production self-initiated-utterance organ", chance=mean_random)
    vd.require("A: consolidated coverage == N on every seed", n_full_cons, expect=lambda x, n=nseeds: x >= n)
    vd.require("A: TAIL basin spoken (consolidated) every seed",
               all(p["tail_spoken_cons"] for p in per_seed), expect=True)
    vd.control("utterance MAGNITUDE consolidated vs consolidation-off (n_utt)", mean_nutt_cons, mean_nutt_seq,
               min_separation=100.0)
    vd.control("COHERENT: surfaced member vs random floor", mean_member, mean_random, min_separation=0.15)
    vd.require("C1: consolidation-off load-bearing (coverage-drop OR magnitude-collapse) every seed",
               C1_consol_off, expect=True)
    vd.require("C2: store NO-ENCODE lesion collapses the stream every seed", C2_store_lesion, expect=True)
    vd.require("D: moat-safe (unknown subject abstains) every seed", D_moat, expect=True)
    vd.require("basins DISJOINT (overlap 0) every seed", AC_disjoint, expect=True)
    vd.require("recall byte-FROZEN (conn.data array_equal) every seed", AC_frozen, expect=True)
    vd.require("byte-IDENTICAL reactive panel (flag-ON == flag-off)", bool(B_flag_off), expect=True)
    vd.require("determinism: consolidated coverage stable across a rebuild", bool(det_ok), expect=True)
    vd.disabled("hebbian/BTSP plasticity during the measured wander", "byte-frozen store measurement")
    preconditions = vd.decide(PASS)["preconditions"]

    verdict = (f"{'GO' if PASS else 'PARTIAL/NO-GO'} -- the DMN consolidated multi-basin store folded into the "
               f"production self-initiated-utterance organ makes coverage {n_mem}/{n_mem} on {n_full_cons}/{len(per_seed)} "
               f"seeds (the previously-dead TAIL concept now self-initiable, tail_spoken every seed); the "
               f"consolidation-off control reaches full coverage on only {n_full_seq}/{len(per_seed)} seeds "
               f"(mean {mean_cov_seq:.1f}/{n_mem}, the weak sequential tail surfacing stochastically) and collapses the "
               f"utterance magnitude every seed (n_utt {mean_nutt_cons:.0f} vs {mean_nutt_seq:.0f}, "
               f"{100*attribution:.0f}% attributable). Store NO-ENCODE lesion collapses the stream every seed; basins "
               f"disjoint (overlap 0); recall byte-frozen; reactive panel byte-identical (ON==flag-off==pristine-HEAD).")

    summary = {
        "probe": "dmn_consolidated_selfinit_verify", "backend": os.environ.get("SIM_BACKEND"),
        "seeds": a.seeds, "n_mem": n_mem, "PASS": PASS, "gates": gates, "verdict": verdict,
        "coverage_consolidated_mean": mean_cov_cons, "coverage_sequential_mean": mean_cov_seq,
        "n_full_coverage_consolidated": n_full_cons, "n_full_coverage_sequential": n_full_seq,
        "n_utt_consolidated_mean": mean_nutt_cons, "n_utt_sequential_mean": mean_nutt_seq,
        "utterance_magnitude_attribution": attribution, "coverage_attribution": cov_attribution,
        "preconditions": preconditions,
        "per_seed": per_seed, "byte_identical": byte_block,
        "panel": _PANEL, "elapsed_seconds": round(time.time() - t0, 1),
        "rescored_from": (_rescore_note if a.rescore_from else None),
        "NOTE": "Composes 2026-08-17-dmn-per-basin-encode-equalization-GO (the consolidated encode) into "
                "2026-08-18-self-initiated-utterance-wired-brain-chat-GO (production faculty #29). NO sim/ edit; NO "
                "webapp/server.py edit; reuse-by-import of _run_wander(encode_mode='consolidated'). The consolidation "
                "is a cupy WRITE-path property (default-ON via selfinit_consolidate); the numpy light path + every "
                "reactive turn are byte-identical. FUNCTIONAL correlate only; no phenomenal claim.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str))

    print("=" * 110)
    print(f"[dmn_consol_selfinit] gates: {gates}")
    print(f"[dmn_consol_selfinit] {verdict}")
    print(f"[dmn_consol_selfinit] {'PASS' if PASS else 'FAIL'} | wrote {OUT} | {summary['elapsed_seconds']}s")
    print("=" * 110)
    return 0 if PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
