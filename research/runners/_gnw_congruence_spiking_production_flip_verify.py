"""PRODUCTION-FLIP verification for the rank-8 GNW congruence spiking read — is it SAFE + GENUINELY LOAD-BEARING
to run `BRAIN_GNW_CONGRUENCE_SPIKING` DEFAULT-ON (Lane C · Self/Workspace, mirrors the rank-12 GNW STOP-trigger
production-flip precedent, `_gnw_stop_trigger_production_flip_verify.py`)?

This is NOT a re-derivation of the circuit's own GO gate (`_gnw_congruence_spiking_read_derisk.py`, 6/6 seeds GO) or
the production-dispatch hook-verify (`_gnw_congruence_spiking_hook_verify.py`, 6/6 seeds GO, both already committed
2026-09-05 and reused-by-import below). It answers the FLIP-SPECIFIC questions those two did not ask: is flipping
the module's DEFAULT (not merely setting the env var to "1") safe for the rest of a live turn, and does the
load-bearing proof survive at the ACTUAL shipped default (bare-unset), not just an explicit `="1"` override?

ARM 1 — THE FLIP IS REAL + NO REGRESSION (per seed, 42/43/44/100/101/102):
  (a) with the flag genuinely POPPED (unset — the actual shipped default after this commit),
      `webapp.gnw_bus_shadow._congruence_spiking_enabled()` resolves True — asserted in the data, no monkeypatch.
  (b) the explicit opt-out escape hatch (`BRAIN_GNW_CONGRUENCE_SPIKING="0"`) still reproduces the FROZEN pre-edit
      host `==` logic byte-for-byte on every real query — never rely on unset==off once a default is flipped
      (`gates/flip_offarm_staleness`'s own lesson; this runner's OWN off arm below is written with the fix baked
      in, and the sibling hook-verify file was updated in the SAME commit to stop relying on `os.environ.pop`).
  (c) with the flag genuinely UNSET, `_organ_reads` on real queries is byte-identical to explicit `="1"` — the new
      default really is the audited spiking-read path, not a different one.

ARM 2 — LOAD-BEARING, NOT HOLLOW, AT THE SHIPPED DEFAULT (the crux; per seed). The hook-verify's own lesion lever
  (`BRAIN_GNW_CONGRUENCE_LESION`) is re-exercised on a manufactured organ-C mismatch, but this time with
  `BRAIN_GNW_CONGRUENCE_SPIKING` genuinely UNSET (the actual shipped default) rather than explicitly `="1"`:
  intact -> `bus_combine`'s committed decision matches the explicit-off host reference (correctly withholds on the
  mismatch); lesioned -> the false corroboration lets all three organs agree and the substrate WRONGLY commits.
  Proving this at bare-unset (not merely at an explicit override) is the flip-specific claim the circuit's own GO
  gate and the hook-verify did not make.

ARM 3 — CROSS-FACULTY REGRESSION (one-shot, not per-seed). `onebrain_regression_battery.run_regression_battery`
  (reused verbatim, no re-derivation) drives ~38 OTHER default-ON faculties through the REAL
  `webapp.server.brain_chat` handler (`brain="tiny-demo"`, GPU-free) and asserts every one still DECIDES
  identically with `BRAIN_GNW_CONGRUENCE_SPIKING` explicit-ON vs explicit-OFF — "every other faculty stays alive"
  on the actual `/api/brain-chat` path, not an isolated stub.

GO iff all arms hold on all 6 seeds (+ the one-shot battery).

Run (CPU-only; the fixture is a small synthetic RFPhasorComposer + tiny-demo — no bundles, no GPU):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_congruence_spiking_production_flip_verify \
      --seeds 42 43 44 100 101 102 \
      --json research/findings/raw/_gnw_congruence_spiking_production_flip_verify.json
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from webapp import gnw_bus_shadow as BUS
from webapp.gnw_congruence_spiking import spiking_congruent  # noqa: F401  (import-path sanity; not called directly)
from research.runners._gnw_congruence_spiking_hook_verify import (
    _build_composer, _all_concepts, _frozen_original_organ_reads, _ForceWrongSecondRead,
)
from research.runners._phaseB_multihop_query_chain_derisk import CHAINS, EAT
from tools.verdict import Verdict
from tools.lab import attributable_to

DEFAULT_SEEDS = [42, 43, 44, 100, 101, 102]

_FLAG = "BRAIN_GNW_CONGRUENCE_SPIKING"
_LESION_FLAG = "BRAIN_GNW_CONGRUENCE_LESION"


def _reset():
    os.environ.pop(_FLAG, None)
    os.environ.pop(_LESION_FLAG, None)


# ── ARM 1: the flip is real, the escape hatch works, and the default is the audited ON path ────────────────────────
def arm1_flip_and_no_regression(seed: int, *, verbose: bool = True) -> dict:
    composer = _build_composer(seed)
    queries = [(ch[0], EAT) for ch in CHAINS]

    # (a) the flip is REAL: bare-unset resolves True, asserted in the data, no monkeypatch.
    _reset()
    flip_is_on_when_unset = bool(BUS._congruence_spiking_enabled() is True)

    # (b) explicit OPT-OUT still reproduces the FROZEN original host `==` byte-for-byte (never rely on unset==off
    #     again -- flip_offarm_staleness's own lesson).
    os.environ[_FLAG] = "0"
    off_rows = []
    for agent, action in queries:
        cand_A_new, triple_new, _tr = BUS._organ_reads(composer, agent, action, seed=seed)
        cand_A_ref, triple_ref = _frozen_original_organ_reads(composer, agent, action)
        off_rows.append(bool(cand_A_new == cand_A_ref and triple_new == triple_ref))
    off_arm_byte_identical = bool(off_rows and all(off_rows))

    # (c) bare-unset (the actual shipped default) reproduces explicit `="1"` exactly -- the new default really is
    #     the audited spiking-read path.
    _reset()
    unset_rows = {(a, act): BUS._organ_reads(composer, a, act, seed=seed)[1] for a, act in queries}
    os.environ[_FLAG] = "1"
    on_rows = {(a, act): BUS._organ_reads(composer, a, act, seed=seed)[1] for a, act in queries}
    default_matches_explicit_on = bool(all(unset_rows[k] == on_rows[k] for k in unset_rows))
    _reset()

    ok = bool(flip_is_on_when_unset and off_arm_byte_identical and default_matches_explicit_on)
    result = {
        "seed": int(seed), "arm1_ok": ok,
        "flip_is_on_when_unset": flip_is_on_when_unset,
        "off_arm_byte_identical": off_arm_byte_identical,
        "default_matches_explicit_on": default_matches_explicit_on,
    }
    if verbose:
        print(f"[flip-verify seed={seed}] ARM1 ok={ok} flip_on_unset={flip_is_on_when_unset} "
              f"off_byte_id={off_arm_byte_identical} default==explicit_on={default_matches_explicit_on}", flush=True)
    return result


# ── ARM 2: load-bearing, not hollow -- at the ACTUAL shipped default (bare-unset), not an explicit override ────────
def arm2_load_bearing_at_shipped_default(seed: int, *, verbose: bool = True) -> dict:
    composer = _build_composer(seed)
    all_concepts = _all_concepts(composer)
    queries = [(ch[0], EAT) for ch in CHAINS]
    agent0, action0 = queries[0]
    cand_A0 = composer.query_patient(agent0, action0)
    other_chain = CHAINS[1] if CHAINS[0][0] == agent0 else CHAINS[0]
    wrong_agent = other_chain[0]   # a REAL agent from a DIFFERENT chain -> a genuine manufactured mismatch

    # the host EXPLICIT-off reference verdict on the manufactured mismatch, for comparison.
    os.environ[_FLAG] = "0"
    wrap_host = _ForceWrongSecondRead(composer, wrong_agent_on_call=1, wrong_agent=wrong_agent)
    info_host = BUS.bus_combine(wrap_host, agent0, action0, all_concepts, seed=seed, lesion=False)

    # INTACT at the SHIPPED DEFAULT (bare-unset, not explicit "1"): must match the host's correct withhold.
    _reset()
    wrap_intact = _ForceWrongSecondRead(composer, wrong_agent_on_call=1, wrong_agent=wrong_agent)
    info_intact = BUS.bus_combine(wrap_intact, agent0, action0, all_concepts, seed=seed, lesion=False)

    # LESIONED at the SHIPPED DEFAULT: the false corroboration must let all 3 organs wrongly agree.
    os.environ[_LESION_FLAG] = "1"
    wrap_lesioned = _ForceWrongSecondRead(composer, wrong_agent_on_call=1, wrong_agent=wrong_agent)
    info_lesioned = BUS.bus_combine(wrap_lesioned, agent0, action0, all_concepts, seed=seed, lesion=False)
    _reset()

    intact_matches_host = bool(info_intact.get("committed") == info_host.get("committed"))
    lesion_wrongly_commits = bool(info_lesioned.get("committed") == cand_A0
                                  and info_host.get("committed") != cand_A0)

    attrib = attributable_to("bus_combine correct verdict (shipped-default intact vs lesion) seed %d" % seed,
                             float(intact_matches_host), float(not lesion_wrongly_commits), warn_below=0.5)

    seed_ok = bool(intact_matches_host and lesion_wrongly_commits)
    result = {
        "seed": int(seed), "arm2_ok": seed_ok,
        "host_committed": info_host.get("committed"), "intact_committed": info_intact.get("committed"),
        "lesioned_committed": info_lesioned.get("committed"),
        "intact_matches_host": intact_matches_host, "lesion_wrongly_commits": lesion_wrongly_commits,
        "attribution": (None if attrib is None else float(attrib)),
    }
    if verbose:
        print(f"[flip-verify seed={seed}] ARM2 ok={seed_ok} intact_matches_host={intact_matches_host} "
              f"lesion_wrongly_commits={lesion_wrongly_commits} "
              f"bus(host={info_host.get('committed')} intact={info_intact.get('committed')} "
              f"lesioned={info_lesioned.get('committed')}) attrib={attrib}", flush=True)
    return result


def run_battery_once(out_dir: str) -> dict:
    from research.runners.onebrain_regression_battery import run_regression_battery
    return run_regression_battery(flag=_FLAG, out_dir=out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Production-flip verification for BRAIN_GNW_CONGRUENCE_SPIKING (rank-8).")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--no-battery", action="store_true",
                    help="skip the one-shot cross-faculty regression battery (mechanics-only smoke)")
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_gnw_congruence_spiking_production_flip_verify.json")
    args = ap.parse_args()

    print(f"[flip-verify] seeds={args.seeds} backend={os.environ.get('SIM_BACKEND')} flag={_FLAG}\n", flush=True)
    arm1 = [arm1_flip_and_no_regression(s) for s in args.seeds]
    arm2 = [arm2_load_bearing_at_shipped_default(s) for s in args.seeds]

    battery = None
    if not args.no_battery:
        raw_dir = os.path.join(os.path.dirname(os.path.abspath(args.json)), "_gnw_congruence_flip_battery")
        battery = run_battery_once(raw_dir)

    all_arm1 = all(r["arm1_ok"] for r in arm1)
    all_arm2 = all(r["arm2_ok"] for r in arm2)
    battery_ok = bool(battery is None or battery.get("all_pass"))

    flip_go = bool(all_arm1 and all_arm2 and battery_ok)

    v = Verdict("GNW congruence spiking read PRODUCTION-FLIP verify (%d seeds)" % len(args.seeds))
    v.require("the flag's default genuinely resolves ON when unset (asserted in the data), all seeds",
              all(r["flip_is_on_when_unset"] for r in arm1), expect=True)
    v.require("explicit opt-out (=0) still reproduces the frozen original host `==` byte-for-byte, all seeds",
              all(r["off_arm_byte_identical"] for r in arm1), expect=True)
    v.require("bare-unset default byte-identical to explicit ON on real queries, all seeds",
              all(r["default_matches_explicit_on"] for r in arm1), expect=True)
    v.require("bus_combine intact matches the host verdict on a manufactured mismatch AT THE SHIPPED DEFAULT, "
              "all seeds", all(r["intact_matches_host"] for r in arm2), expect=True)
    v.require("bus_combine lesion-via-flag WRONGLY commits on the SAME mismatch AT THE SHIPPED DEFAULT (the "
              "load-bearing collapse), all seeds", all(r["lesion_wrongly_commits"] for r in arm2), expect=True)
    if battery is not None:
        v.require("the cross-faculty regression battery (~38 other default-ON faculties) reports all_pass",
                  battery_ok, expect=True)
    else:
        v.disabled("regression_battery", why="--no-battery smoke run: the one-shot cross-faculty check was skipped")
    vd = v.decide(go=flip_go, verbose=True)

    summary = {
        "runner": "_gnw_congruence_spiking_production_flip_verify", "verdict": vd["status"], "flip_go": flip_go,
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
        "undefined_reasons": vd["undefined_reasons"],
        "seeds": list(args.seeds), "flag": _FLAG, "lesion_flag": _LESION_FLAG,
        "arm1_per_seed": arm1, "arm2_per_seed": arm2,
        "regression_battery": battery,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 100}", flush=True)
    print(f"  PRODUCTION-FLIP VERDICT ({_FLAG}): {vd['status']}  (arm1={all_arm1} arm2={all_arm2} "
          f"battery_all_pass={battery_ok})", flush=True)
    print(f"    [saved] {args.json}\n{'=' * 100}", flush=True)
    return 0 if flip_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
