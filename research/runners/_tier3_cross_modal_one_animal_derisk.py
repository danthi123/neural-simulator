"""Tier-3 Option 3 'cross-modal one animal' de-risk: the SHARED hunger drive tightens the CONVERSATIONAL moat.

Per the controller-verified scoping (research/findings/2026-07-01-tier3-option3-cross-modal-one-animal-scoping.md):
the "one brain" property that the SAME limbic drive touches BOTH halves -- a HUNGRY brain is measurably more
conservative in conversation (higher abstention on uncertain reads) than a SATED one -- with the no-confab moat
ASSERTED byte-unchanged (the gate can ONLY tighten, never loosen). Three of the four pieces already exist + are
validated on the merged one brain: the shared spiking `dopamine` modulator (nav-critic default-ON), the 6/6-GO
spiking hunger drive (`drive_agrp`/`drive_pomc`), and the moat-safe `_da_confidence_gate` (DA -> abstention, 6/6 GO
2026-06-18-DA-composer-precision-derisk). The genuine residual was ONE missing arrow -- hunger did not raise DA --
now closed by the additive default-off `drive_to_da` link (a `from_region_firing` rule reading `drive_agrp` appended
to the shared `dopamine` modulator; NO `sim/` edit).

THE DE-RISK (two composed, each-validated links):
  1. MEASURE the NEW link on the REAL merged bridge: inject the body deficit as interoceptive current into the
     co-resident `drive_agrp`/`drive_pomc` pools, run the bridge, and READ the shared `dopamine` concentration
     (off the neuromodulator manager, driven by `drive_agrp` firing via the `from_region_firing` rule). Hungry
     (deficit high) MUST raise DA above the sated baseline; a DRIVE-LESION (interoceptive current OFF) MUST leave DA
     at baseline; DA rises monotonically with the deficit.
  2. FEED those real-bridge-measured DA levels into the VALIDATED DA-gate machinery (reuse-by-import: `da_to_gate` +
     `FHRRCleanupComposer` + `run_condition` from `_da_composer_salience_cleanup_derisk`, the 6/6-GO precision-gate
     de-risk) under matched cleanup NOISE: the HUNGRY DA yields a HIGHER gate `g_eff` -> MORE abstention on the
     uncertain reads + a LOWER error-rate among the non-abstained (salience-gated precision), while the no-confab
     MOAT stays 0 false-accepts at BOTH hunger levels (structurally guaranteed -- `da_to_gate` clamps at the `g0`
     floor, so DA can only RAISE the gate).

GATES / ANTI-CHEATS (ALL must hold)
-----------------------------------
  (1) HUNGER->DA LINK: DA(hungry) > DA(sated) + margin, on the real bridge, read from the shared modulator.
  (2) DRIVE-LESION: DA(deficit high, current OFF) ~ DA(sated) -- the DA rise is the drive's doing (validate-by-function).
  (3) MONOTONE: DA rises with the body deficit (a controlled sweep).
  (4) GATE TIGHTENS: da_to_gate(DA_hungry) > da_to_gate(DA_sated) -- the measured DA sharpens the conversational gate.
  (5) CROSS-MODAL BEHAVIOR: hungry abstains MORE than sated on uncertain reads AND error-rate(hungry) <= error-rate(sated).
  (6) NO-CONFAB MOAT (HARD): 0 false-accepts at BOTH hunger levels (the gate can only tighten -- moat-safe by
      construction; asserted).

HONEST SCOPE: this is a PROPERTY demonstration (one drive touches both halves), not a new life -- the scoping's cheap
"Phase-3.1" follow-on to the two closed slices (live-and-remember, develop-with-a-body). The DA->abstention half is
Option-1-era-validated (2026-06-18); this de-risk adds the NEW hunger->DA link on the real bridge + composes them.
NO `sim/` edit (the link is a runner-layer additive default-off `from_region_firing` rule).

Run (GPU -- the merged bridge is GPU-only):
  python -m research.runners._tier3_cross_modal_one_animal_derisk --smoke
  python -m research.runners._tier3_cross_modal_one_animal_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# reuse-by-import the VALIDATED DA-gate machinery (the 6/6-GO precision-gate de-risk) VERBATIM.
from research.runners._da_composer_salience_cleanup_derisk import (
    FHRRCleanupComposer, run_condition, da_to_gate,
)

G0, K, NOISE_SIGMA, N_QUERY_REPS, N_FACTS, D = 0.06, 2.0, 2.0, 20, 8, 64


def _measure_hunger_da(agent, deficit, lesion=False, n_settle=300, i_scale=300.0):
    """Inject the body deficit as interoceptive current into the co-resident drive pools (drive_agrp ∝ deficit,
    drive_pomc ∝ surplus), run the merged bridge `n_settle` steps so the `from_region_firing` rule drives the shared
    `dopamine` up from `drive_agrp` firing, and READ the settled DA concentration. lesion=True zeros the interoceptive
    current (drive_agrp silent) -> DA stays at baseline. NOTE the DA is read off the neuromodulator manager (the
    shared modulator), driven by drive_agrp SPIKES -- brain-based (not a host deficit value)."""
    import sim.backend as B
    xp, _ = B.get_backend()
    br = agent._merged_bridge
    rm = br.region_manager
    agrp = xp.asarray(np.asarray(rm.indices("drive_agrp"), dtype=np.int64))
    pomc = xp.asarray(np.asarray(rm.indices("drive_pomc"), dtype=np.int64))
    i_agrp = 0.0 if lesion else i_scale * max(0.0, float(deficit))
    i_pomc = i_scale * max(0.0, 1.0 - float(deficit))
    for _ in range(int(n_settle)):
        br.cp_external_input_current[:] = 0.0
        br.cp_external_input_current[agrp] = i_agrp
        br.cp_external_input_current[pomc] = i_pomc
        br._run_one_simulation_step()
    return float(br.neuromodulator_manager.get_concentration("dopamine"))


def _build_agent(seed):
    """The merged one brain with the co-resident SPIKING drive + the hunger->DA link (drive_to_da) + the moat-safe
    DA-confidence gate (default-ON) + the shared dopamine modulator (co_resident_nav_critic default-ON)."""
    from research.runners.nav_conv_merged_bridge import MergedNavConvAgent
    return MergedNavConvAgent(seed=seed, co_resident_composer=True, co_resident_composer_kind="rf",
                              co_resident_drive=True, drive_to_da=True)


def _store_facts(comp, n_facts, rng):
    """Mirror the DA-gate de-risk's fixed fact store (distinct (agent, action) cues)."""
    used = set()
    while len(comp.kb) < n_facts:
        a = comp.words[rng.integers(len(comp.words))]
        v = comp.words[rng.integers(len(comp.words))]
        p = comp.words[rng.integers(len(comp.words))]
        if (a, v) in used:
            continue
        used.add((a, v)); comp.store(a, v, p)


def run_seed(seed, *, n_settle=300, verbose=True):
    """One seed: measure the hunger->DA link on the REAL merged bridge, then feed the measured DA levels into the
    validated noisy-cue DA-gate test (hungry abstains more + moat 0-FA both)."""
    agent = _build_agent(seed)
    nm = agent._merged_bridge.neuromodulator_manager
    da_baseline = float(nm._config_by_name("dopamine").baseline)

    # ── (1)-(3) the NEW link, on the real bridge ──
    da_sated = _measure_hunger_da(agent, 0.0, n_settle=n_settle)
    da_hungry = _measure_hunger_da(agent, 1.0, n_settle=n_settle)
    da_lesion = _measure_hunger_da(agent, 1.0, lesion=True, n_settle=n_settle)
    sweep = [(d, _measure_hunger_da(agent, d, n_settle=n_settle)) for d in (0.0, 0.5, 1.0)]
    sweep_da = [v for _, v in sweep]
    monotone = all(b >= a - 1e-6 for a, b in zip(sweep_da, sweep_da[1:])) and (sweep_da[-1] > sweep_da[0] + 0.02)

    # ── (4) the gate composition (validated da_to_gate on the measured DAs) ──
    g_sated = da_to_gate(da_sated, da_baseline, G0, K)
    g_hungry = da_to_gate(da_hungry, da_baseline, G0, K)

    # ── (5)-(6) the behavioral cross-modal effect: feed the measured DAs into the VALIDATED noisy-cue gate test ──
    comp = FHRRCleanupComposer(seed=seed, D=D)
    _store_facts(comp, N_FACTS, np.random.default_rng(seed + 12345))
    sated = run_condition(comp, da_sated, da_baseline, G0, K, NOISE_SIGMA, N_QUERY_REPS, seed)
    hungry = run_condition(comp, da_hungry, da_baseline, G0, K, NOISE_SIGMA, N_QUERY_REPS, seed)

    link = bool(da_hungry > da_sated + 0.02)
    lesion_ok = bool(da_lesion <= da_sated + 0.02)
    gate_tightens = bool(g_hungry > g_sated + 1e-6)
    behavioral = bool(hungry["abstain_rate"] > sated["abstain_rate"] + 0.05
                      and hungry["error_rate"] <= sated["error_rate"] + 1e-9)
    moat = bool(hungry["moat_false_accepts"] == 0 and sated["moat_false_accepts"] == 0)   # HARD
    go = bool(link and lesion_ok and monotone and gate_tightens and behavioral and moat)

    out = {"seed": seed, "go": go,
           "da": {"baseline": da_baseline, "sated": da_sated, "hungry": da_hungry, "lesion": da_lesion,
                  "sweep": sweep_da},
           "gate": {"g_sated": g_sated, "g_hungry": g_hungry},
           "sated_cond": sated, "hungry_cond": hungry,
           "checks": {"link": link, "lesion_ok": lesion_ok, "monotone": monotone,
                      "gate_tightens": gate_tightens, "behavioral": behavioral, "moat": moat}}
    if verbose:
        print(f"  [seed {seed}] DA sated {da_sated:.3f} hungry {da_hungry:.3f} lesion {da_lesion:.3f} (base "
              f"{da_baseline:.2f}) | g {g_sated:.3f}->{g_hungry:.3f} | abstain {sated['abstain_rate']:.2f}->"
              f"{hungry['abstain_rate']:.2f} err {sated['error_rate']:.2f}->{hungry['error_rate']:.2f} | moat "
              f"{sated['moat_false_accepts']}/{hungry['moat_false_accepts']} FA || {'GO' if go else 'NO'} "
              f"{out['checks']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-settle", type=int, default=300)
    ap.add_argument("--out", default="research/findings/raw/_tier3_cross_modal_one_animal.json")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    print("[Tier-3 cross-modal one-animal] does the SHARED hunger drive tighten the CONVERSATIONAL moat -- one drive "
          "touching BOTH halves?\n  GATES: (1) hunger->DA link on the real bridge  (2) drive-lesion -> no DA rise  "
          "(3) monotone with deficit  (4) gate tightens  (5) hungry abstains more + lower error  (6) no-confab MOAT "
          "0-FA both (can ONLY tighten).\n", flush=True)

    seeds = a.seeds[:1] if a.smoke else a.seeds
    results = [run_seed(s, n_settle=(150 if a.smoke else a.n_settle)) for s in seeds]
    n_go = sum(r["go"] for r in results)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results, "n_go": n_go, "n_seeds": len(results)}, fh, indent=2, default=str)

    print(f"\n{'='*110}", flush=True)
    if a.smoke:
        r = results[0]
        ok = bool(r["checks"]["link"] and r["checks"]["gate_tightens"] and r["checks"]["moat"])
        print(f"  [smoke] {'LINK+GATE+MOAT OK' if ok else 'CHECK'} -- {r['checks']}", flush=True)
    elif n_go == len(results) and results:
        print(f"  GO ({n_go}/{len(results)} seeds): the SHARED hunger drive touches BOTH halves of the one brain. A "
              "HUNGRY brain's interoceptive drive raises the shared spiking dopamine (measured on the real bridge; a "
              "drive-lesion leaves DA at baseline; DA rises monotonically with the deficit), which sharpens the "
              "moat-safe conversational gate -> the hungry brain ABSTAINS MORE on uncertain reads with a LOWER "
              "error-rate, while the no-confab MOAT holds 0 false-accepts at BOTH hunger levels (the gate can only "
              "TIGHTEN). ⇒ one limbic drive demonstrably modulates BOTH the acting and the conversing halves. NO "
              "`sim/` edit (the hunger->DA link is an additive default-off from_region_firing rule). HONEST SCOPE: a "
              "one-brain PROPERTY demonstration; the DA->abstention half was validated 2026-06-18, this adds the new "
              "hunger->DA link + composes them.", flush=True)
    else:
        print(f"  PARTIAL/NEGATIVE ({n_go}/{len(results)} seeds): localize (link / lesion / monotone / gate / "
              "behavioral / moat). An honest negative that pins the wall is a valid deliverable.", flush=True)
    print(f"  [saved] {a.out}\n{'='*110}", flush=True)
    return 0 if (a.smoke or (results and n_go == len(results))) else 1


if __name__ == "__main__":
    sys.exit(main())
