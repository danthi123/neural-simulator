"""GNW CONGRUENCE spiking read — PRODUCTION-HOOK verify (byte-identical off / dispatch parity on / lesion reverts).

WHY A SEPARATE FILE (the rank-12 precedent — TERMS.md `wired` needs a call-path + a byte-identical assertion IN THE
DATA, not inferred from reading the diff). `_gnw_congruence_spiking_read_derisk.py`'s own 6/6-seed GO gate proves
the CIRCUIT (`SpikingCongruenceReader`) reproduces the host `==` congruence verdict on a real organ-B/organ-C
battery. It does NOT exercise the ACTUAL production entry point (`webapp.gnw_bus_shadow._organ_reads` /
`bus_combine`) with the flag threaded through exactly as `webapp/server.py::brain_reply` would run it. This module
does that, on the SAME `CHAINS` fixture, calling `_organ_reads`/`bus_combine` DIRECTLY (no `sim/` edit; no fake
`ChatBrain` needed — both functions take a bare `composer`).

THREE CLAIMS, EACH CHECKED SEPARATELY:
  1. FLAG-OFF BYTE-IDENTICAL. `BRAIN_GNW_CONGRUENCE_SPIKING` unset -> `_organ_reads`'s output is compared, per
     query, against a FROZEN literal copy of the PRE-EDIT host `==` logic (`_frozen_original_organ_reads`, embedded
     in this file's data, never imported by production) — a tuple compare, not an inferred diff-read.
  2. FLAG-ON REAL-MATCH PARITY. On the fixture's genuine (agent, action) queries (organ B/C's real reads DO match
     organ A's — this composer's unpermuted facts never disagree naturally), the flag-ON (spiking) triple must
     equal the flag-OFF (host) triple: the retirement changes the MECHANISM, not genuine-match behaviour.
  3. LESION-VIA-FLAG REVERTS, on a MANUFACTURED mismatch. This fixture's clean facts never disagree naturally, so a
     wrapper composer (`_ForceWrongSecondRead`) forces organ B's re-read / organ C's reverse-binding to a WRONG
     value on one probe each — a genuine mismatch. Flag ON (no lesion): the spiking read correctly withholds
     (`cand_B`/`cand_C` -> None), matching the host `==`'s own correct withhold. Flag ON + `BRAIN_GNW_CONGRUENCE_
     LESION=1`: the withhold INCORRECTLY reverts to a false corroboration (`cand_A`) — proving the correct-withhold
     behaviour depends on the mm circuit's actual firing, not the flag/addressing alone. `bus_combine`'s own
     COMMITTED decision is also checked on the organ-C manufactured mismatch: intact -> the 3-way unanimity fails
     on organ C's correct withhold (matching the host); lesioned -> organ C's false corroboration lets ALL THREE
     organs agree -> the substrate WRONGLY commits — the same class of "collapsed to the wrong answer" proof
     rank-12's hook-verify used for its own lesion-via-flag lever.

Run (CPU cheap-first):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_congruence_spiking_hook_verify \
      --seeds 42 43 44 100 101 102 --json research/findings/raw/_gnw_congruence_spiking_hook_verify.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._phaseB_multihop_query_chain_derisk import CHAINS, EAT, build_vocab, store_facts
from research.runners.rf_phasor_composer import RFPhasorComposer
from webapp.gnw_bus_shadow import _organ_reads, bus_combine
from tools.verdict import Verdict
from tools.lab import attributable_to

D_COMPOSER = 64


def _frozen_original_organ_reads(composer, agent, action):
    """A FROZEN, literal copy of the PRE-rank-8 `_organ_reads` (host `==` congruence, no flag branch at all) — the
    byte-identical-off reference. Never imported by production; exists only in this verify script's data."""
    try:
        cand_A = composer.query_patient(agent, action)
    except Exception:
        cand_A = None
    if cand_A is None:
        return None, [None, None, None]
    try:
        cand_B = cand_A if composer.query_patient(agent, action) == cand_A else None
    except Exception:
        cand_B = None
    try:
        cand_C = cand_A if composer.query_agent(action, cand_A) == agent else None
    except Exception:
        cand_C = None
    return cand_A, [cand_A, cand_B, cand_C]


class _ForceWrongSecondRead:
    """Wraps a REAL composer; passes every call through UNCHANGED except one deliberately-forced wrong return (the
    Nth `query_patient`/`query_agent` call), manufacturing the genuine mismatch this fixture's clean, unpermuted
    facts never produce naturally. `__setattr__`/`__getattr__` forward everything else (incl. `last_trace`
    writes) to the real composer so `bus_combine`'s trace save/restore still works unmodified."""

    def __init__(self, real, *, wrong_patient_on_call: int = 0, wrong_patient=None,
                wrong_agent_on_call: int = 0, wrong_agent=None):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_qp_calls", 0)
        object.__setattr__(self, "_qa_calls", 0)
        object.__setattr__(self, "_wrong_patient_on_call", int(wrong_patient_on_call))
        object.__setattr__(self, "_wrong_patient", wrong_patient)
        object.__setattr__(self, "_wrong_agent_on_call", int(wrong_agent_on_call))
        object.__setattr__(self, "_wrong_agent", wrong_agent)

    def query_patient(self, agent, action):
        object.__setattr__(self, "_qp_calls", self._qp_calls + 1)
        if self._qp_calls == self._wrong_patient_on_call and self._wrong_patient is not None:
            return self._wrong_patient
        return self._real.query_patient(agent, action)

    def query_agent(self, action, patient):
        object.__setattr__(self, "_qa_calls", self._qa_calls + 1)
        if self._qa_calls == self._wrong_agent_on_call and self._wrong_agent is not None:
            return self._wrong_agent
        return self._real.query_agent(action, patient)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        setattr(self._real, name, value)


def _build_composer(seed):
    vocab = build_vocab()
    composer = RFPhasorComposer(seed=seed, D=D_COMPOSER, vocab=vocab)
    store_facts(composer, CHAINS, distractor_rng=np.random.default_rng(seed * 53 + 1))
    return composer


def _all_concepts(composer):
    return sorted({c for ch in CHAINS for c in ch})


def evaluate_seed(seed: int, verbose: bool = True):
    os.environ.pop("BRAIN_GNW_CONGRUENCE_SPIKING", None)
    os.environ.pop("BRAIN_GNW_CONGRUENCE_LESION", None)
    composer = _build_composer(seed)
    all_concepts = _all_concepts(composer)
    queries = [(ch[0], EAT) for ch in CHAINS]

    # ── CLAIM 1: flag-off byte-identical vs the FROZEN pre-edit reference, on every real query ──────────────────
    flag_off_rows = []
    for agent, action in queries:
        cand_A_new, triple_new, _tr = _organ_reads(composer, agent, action, seed=seed)
        cand_A_ref, triple_ref = _frozen_original_organ_reads(composer, agent, action)
        identical = bool(cand_A_new == cand_A_ref and triple_new == triple_ref)
        flag_off_rows.append({"agent": agent, "action": action, "new": triple_new, "frozen_ref": triple_ref,
                              "identical": identical})
    flag_off_byte_identical = bool(flag_off_rows and all(r["identical"] for r in flag_off_rows))

    # ── CLAIM 2: flag-on real-match parity (no lesion) — genuine matches must read the SAME as flag-off ──────────
    os.environ["BRAIN_GNW_CONGRUENCE_SPIKING"] = "1"
    parity_rows = []
    for agent, action in queries:
        _cA, triple_on, _tr = _organ_reads(composer, agent, action, seed=seed)
        ref_row = next(r for r in flag_off_rows if r["agent"] == agent and r["action"] == action)
        same = bool(triple_on == ref_row["new"])
        parity_rows.append({"agent": agent, "action": action, "flag_on": triple_on,
                            "flag_off": ref_row["new"], "same": same})
    real_match_parity = bool(parity_rows and all(r["same"] for r in parity_rows))
    os.environ.pop("BRAIN_GNW_CONGRUENCE_SPIKING", None)

    # ── CLAIM 3: lesion-via-flag reverts, on a MANUFACTURED mismatch (this fixture never disagrees naturally) ────
    agent0, action0 = queries[0]
    cand_A0 = composer.query_patient(agent0, action0)
    other_chain = CHAINS[1] if CHAINS[0][0] == agent0 else CHAINS[0]
    wrong_patient = other_chain[1]                 # a REAL patient from a DIFFERENT chain (genuine mismatch content)
    wrong_agent = other_chain[0]                   # a REAL agent from a DIFFERENT chain

    # organ B's manufactured mismatch: force the SECOND query_patient call (organ B's re-read) wrong.
    wrap_b = _ForceWrongSecondRead(composer, wrong_patient_on_call=2, wrong_patient=wrong_patient)
    os.environ["BRAIN_GNW_CONGRUENCE_SPIKING"] = "1"
    os.environ.pop("BRAIN_GNW_CONGRUENCE_LESION", None)
    _cA, triple_b_intact, _tr = _organ_reads(wrap_b, agent0, action0, seed=seed)
    wrap_b2 = _ForceWrongSecondRead(composer, wrong_patient_on_call=2, wrong_patient=wrong_patient)
    os.environ["BRAIN_GNW_CONGRUENCE_LESION"] = "1"
    _cA, triple_b_lesioned, _tr = _organ_reads(wrap_b2, agent0, action0, seed=seed)
    os.environ.pop("BRAIN_GNW_CONGRUENCE_LESION", None)
    organB_withholds_intact = bool(triple_b_intact[1] is None)
    organB_false_corroborates_lesioned = bool(triple_b_lesioned[1] == cand_A0)

    # organ C's manufactured mismatch: force the FIRST query_agent call (organ C's reverse-binding) wrong.
    wrap_c = _ForceWrongSecondRead(composer, wrong_agent_on_call=1, wrong_agent=wrong_agent)
    os.environ.pop("BRAIN_GNW_CONGRUENCE_LESION", None)
    _cA, triple_c_intact, _tr = _organ_reads(wrap_c, agent0, action0, seed=seed)
    wrap_c2 = _ForceWrongSecondRead(composer, wrong_agent_on_call=1, wrong_agent=wrong_agent)
    os.environ["BRAIN_GNW_CONGRUENCE_LESION"] = "1"
    _cA, triple_c_lesioned, _tr = _organ_reads(wrap_c2, agent0, action0, seed=seed)
    organC_withholds_intact = bool(triple_c_intact[2] is None)
    organC_false_corroborates_lesioned = bool(triple_c_lesioned[2] == cand_A0)

    # the FULL bus_combine COMMITTED decision on organ C's manufactured mismatch: intact -> withholds/no-unanimity
    # (matches the host); lesioned -> the false corroboration lets all 3 organs agree -> the substrate WRONGLY
    # commits. (organ A/B still genuinely agree on cand_A0 in this probe -- only organ C is forced wrong.)
    wrap_c3 = _ForceWrongSecondRead(composer, wrong_agent_on_call=1, wrong_agent=wrong_agent)
    os.environ.pop("BRAIN_GNW_CONGRUENCE_LESION", None)
    info_intact = bus_combine(wrap_c3, agent0, action0, all_concepts, seed=seed, lesion=False)
    wrap_c4 = _ForceWrongSecondRead(composer, wrong_agent_on_call=1, wrong_agent=wrong_agent)
    os.environ["BRAIN_GNW_CONGRUENCE_LESION"] = "1"
    info_lesioned = bus_combine(wrap_c4, agent0, action0, all_concepts, seed=seed, lesion=False)
    os.environ.pop("BRAIN_GNW_CONGRUENCE_SPIKING", None)
    os.environ.pop("BRAIN_GNW_CONGRUENCE_LESION", None)

    # the host (flag-off) reference verdict on the SAME manufactured mismatch, for comparison.
    wrap_c_host = _ForceWrongSecondRead(composer, wrong_agent_on_call=1, wrong_agent=wrong_agent)
    info_host = bus_combine(wrap_c_host, agent0, action0, all_concepts, seed=seed, lesion=False)

    bus_intact_matches_host = bool(info_intact.get("committed") == info_host.get("committed"))
    bus_lesion_wrongly_commits = bool(info_lesioned.get("committed") == cand_A0
                                      and info_host.get("committed") != cand_A0)

    lesion_reverts = bool(organB_withholds_intact and organB_false_corroborates_lesioned
                          and organC_withholds_intact and organC_false_corroborates_lesioned
                          and bus_intact_matches_host and bus_lesion_wrongly_commits)

    seed_go = bool(flag_off_byte_identical and real_match_parity and lesion_reverts)

    # ATTRIBUTION (tools.lab.attributable_to): measuring the intact AND lesioned arms is not the same as asking
    # whose the difference was — credit the correct-withhold behaviour to the lesion lever specifically (it zeroes
    # ONLY mm's proposal drive; the addressing/wiring and the host `==` path are untouched by it).
    attr_organB = attributable_to("organ B correct-withhold (intact vs lesion-via-flag) at the production hook",
                                  float(organB_withholds_intact), float(not organB_false_corroborates_lesioned),
                                  warn_below=0.5)
    attr_organC = attributable_to("organ C correct-withhold (intact vs lesion-via-flag) at the production hook",
                                  float(organC_withholds_intact), float(not organC_false_corroborates_lesioned),
                                  warn_below=0.5)
    attr_bus = attributable_to("bus_combine correct verdict (intact vs lesion-via-flag) at the production hook",
                               float(bus_intact_matches_host), float(not bus_lesion_wrongly_commits),
                               warn_below=0.5)

    v = Verdict("GNW congruence spiking read: production-hook verify (seed %d)" % seed)
    v.require("flag-off byte-identical vs the FROZEN pre-edit reference (every real query)",
              flag_off_byte_identical, expect=True, note="%d/%d" % (
                  sum(1 for r in flag_off_rows if r["identical"]), len(flag_off_rows)))
    v.require("flag-on real-match parity vs flag-off (genuine matches unaffected)",
              real_match_parity, expect=True, note="%d/%d" % (
                  sum(1 for r in parity_rows if r["same"]), len(parity_rows)))
    v.require("organ B: intact withholds on a manufactured mismatch", organB_withholds_intact, expect=True)
    v.require("organ B: lesion falsely corroborates the SAME mismatch", organB_false_corroborates_lesioned,
              expect=True)
    v.require("organ C: intact withholds on a manufactured mismatch", organC_withholds_intact, expect=True)
    v.require("organ C: lesion falsely corroborates the SAME mismatch", organC_false_corroborates_lesioned,
              expect=True)
    v.require("bus_combine intact matches the host verdict on the manufactured mismatch",
              bus_intact_matches_host, expect=True)
    v.require("bus_combine lesioned WRONGLY commits (the load-bearing collapse)", bus_lesion_wrongly_commits,
              expect=True)
    vd = v.decide(go=seed_go, verbose=verbose)

    result = {
        "seed": int(seed), "verdict": vd["status"], "seed_go": bool(seed_go and vd["status"] == "GO"),
        "flag_off_byte_identical": flag_off_byte_identical, "real_match_parity": real_match_parity,
        "lesion_reverts": lesion_reverts,
        "n_queries": len(queries), "n_flag_off_identical": sum(1 for r in flag_off_rows if r["identical"]),
        "n_parity_matches": sum(1 for r in parity_rows if r["same"]),
        "organ_b": {"withholds_intact": organB_withholds_intact,
                   "false_corroborates_lesioned": organB_false_corroborates_lesioned,
                   "attributable_fraction": attr_organB},
        "organ_c": {"withholds_intact": organC_withholds_intact,
                   "false_corroborates_lesioned": organC_false_corroborates_lesioned,
                   "attributable_fraction": attr_organC},
        "bus_combine": {"host_committed": info_host.get("committed"), "intact_committed": info_intact.get("committed"),
                        "lesioned_committed": info_lesioned.get("committed"),
                        "intact_matches_host": bus_intact_matches_host,
                        "lesion_wrongly_commits": bus_lesion_wrongly_commits,
                        "attributable_fraction": attr_bus},
        "flag_off_rows": flag_off_rows, "parity_rows": parity_rows,
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
    }
    if verbose:
        print(f"[gnw-congruence-hook seed={seed}] verdict={vd['status']} seed_go={result['seed_go']} "
              f"flag_off_identical={flag_off_byte_identical} real_match_parity={real_match_parity} "
              f"lesion_reverts={lesion_reverts} bus(host={info_host.get('committed')} "
              f"intact={info_intact.get('committed')} lesioned={info_lesioned.get('committed')})", flush=True)
    return result


def run_six_seed(args):
    seeds = [42, 43, 44, 100, 101, 102]
    per_seed = [evaluate_seed(s, verbose=True) for s in seeds]
    n_go = sum(1 for r in per_seed if r["seed_go"])
    n_off = sum(1 for r in per_seed if r["flag_off_byte_identical"])
    n_parity = sum(1 for r in per_seed if r["real_match_parity"])
    n_lesion = sum(1 for r in per_seed if r["lesion_reverts"])
    pooled_go = bool(n_go == len(seeds))
    verdict = "GO" if pooled_go else ("PARTIAL" if n_off == len(seeds) else "NO-GO")

    v = Verdict("GNW congruence spiking read: production-hook 6-seed aggregate")
    v.require("flag-off byte-identical on 6/6 seeds", bool(n_off == len(seeds)), expect=True)
    v.require("flag-on real-match parity on 6/6 seeds", bool(n_parity == len(seeds)), expect=True)
    v.require("lesion-via-flag reverts on 6/6 seeds", bool(n_lesion == len(seeds)), expect=True)
    vd = v.decide(go=pooled_go)

    summary = {"runner": "_gnw_congruence_spiking_hook_verify", "mode": "six_seed", "verdict": verdict,
               "pooled_go": pooled_go, "seeds": seeds, "verdict_status": vd["status"],
               "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
               "counts": {"seed_go": n_go, "flag_off_byte_identical": n_off, "real_match_parity": n_parity,
                          "lesion_reverts": n_lesion, "n_seeds": len(seeds)},
               "per_seed": per_seed}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[gnw-congruence-hook six-seed] verdict={verdict} seed_go {n_go}/6 flag_off_identical {n_off}/6 "
          f"real_match_parity {n_parity}/6 lesion_reverts {n_lesion}/6", flush=True)
    print(f"[gnw-congruence-hook six-seed] wrote {args.json}", flush=True)
    return 0 if pooled_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW congruence spiking read: production-hook verify.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--single", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_congruence_spiking_hook_verify.json")
    args = ap.parse_args()
    if args.single:
        r = evaluate_seed(args.seed, verbose=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"runner": "_gnw_congruence_spiking_hook_verify", "mode": "single", "result": r}, f,
                      indent=2, default=str)
        return 0 if r["seed_go"] else 1
    return run_six_seed(args)


if __name__ == "__main__":
    raise SystemExit(main())
