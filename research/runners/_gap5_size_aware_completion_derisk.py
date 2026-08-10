"""gap#5 SIZE-AWARE COMPLETION — close the end-to-end composition SEAM named in
`2026-08-10-gap5-e2e-episodic-loop-...-SEAM-NEGATIVE.md`.

THE SEAM (verified): the three gap#5 pieces are individually GO but do NOT naively compose. Emergent-DG SELECTION
composes (6/6, membership genuinely DG-selected) and BTSP FORMATION is genuine + weight-specific, but the slow-NMDA +
FS-basket COMPLETION readout (fixed feedBACK inhibition ca3_fb_inhib=60 + fixed recurrent density) is NON-SPECIFIC on
the SMALL (~23-cell), VARIABLE emergently-selected assemblies: perm ~ nocue ~ cue, and it fails even HAND-INSTALLED (so
the seam is the completion OPERATING POINT, not BTSP). A recurrent-density sweep UP {0.12,0.35,0.5} made it WORSE. The
completion operating point's fixed inhibition gain is implicitly tuned for LARGER, UNIFORM ~72-cell assemblies.

THE NAMED FIX (this runner): make the completion inhibition assembly-SIZE-AWARE. Two biological levers, both from the
SEAM finding:
  (1) FEEDFORWARD divisive-normalization inhibition that SCALES with the active-population size (de Almeida-Idiart-
      Lisman 2009 E%-max / Pouille-Scanziani 2001 FF inhibition) -- the ca3_ff_basket already wired in `_build` for
      SELECTION, now applied to the completion READ. During completion DG is silent, so the ca3->ca3_ff_basket arm
      makes it a disynaptic FEEDFORWARD inhibition driven by the CUE VOLLEY (active cells -> basket -> held cells),
      whose gain scales with the number of active cells -> a size-invariant cue-ignitable bistable window. This is the
      SAME divisive-normalization companion process that made emergent-DG SELECTION robust across a >10x input range.
  (2) SPARSER recurrence (the density sweep UP HURT -> go DOWN toward Guzman-Jonas CA3 ~1-2%): fewer within-assembly
      synapses per held cell -> lower total recurrent drive -> the rest state stays silent (nocue low), the cue can
      still complete. NB a ~23-cell assembly at 2% density has almost no internal recurrence -> the honest tension the
      finding named (sparse is the point, but a tiny assembly needs SOME recurrence); the sweep bounds it.

DECISIVE DECOMPOSITION (as the SEAM finding did):
  - CONTROL arm = the OLD fixed readout (ca3_fb_inhib=60, ca3_ff_inhib=None, density 0.12). It MUST STILL FAIL
    (perm ~ nocue ~ cue) -- else the size-aware inhibition is not load-bearing.
  - SIZE-AWARE arms = sparser density x ca3_ff_inhib>0 (+ optionally lower ca3_fb_inhib).
  - HAND-INSTALL sub-arm (perfect within-assembly W, zero plasticity) isolates the READOUT operating point from BTSP
    formation: if hand-install + size-aware gives cue-specificity, the readout fix works independent of formation.
  - BTSP sub-arm confirms the FULL emergent composition (emergent-DG select -> BTSP form -> size-aware complete).

ANTI-CHEATS:
  #1 EMERGENT membership: assemblies are the DG-selected ~14-33 (NOT the readout's 0.18*N pre-assigned mask); low
     Jaccard vs the random-permutation set; mossy-LESION (mossy_weight=0) collapses every assembly to ~0 (DG->CA3
     detonation load-bearing). (reused from the e2e loop runner.)
  #2 SIZE-AWARE inhibition is the fix: the CONTROL (fixed fb=60, ff=None) MUST still fail on the same assemblies.
  #3 completion cue-specific: perm=0, silent-rest nocue<=0.10 (specificity + silent rest), cue>=3x each.
  #4 BTSP genuine: within grew from fused_btsp_update, cross/non-member did not (inherited genuine_formation).
  #5 plasticity FROZEN at recall; OU OFF (deterministic) -- the seam is deterministic.
  + build-twice threshold-hash determinism check; cfg.seed set explicitly.

GO (6-seed 42/43/44/100/101/102): an emergently-selected ~23-cell assembly completes CUE-SPECIFICALLY with the
size-aware completion (held_cue>=0.20 AND cue>=3*perm AND cue>=3*nocue AND nocue<=0.10, BTSP-formed + genuine) on >=5/6
seeds at SOME size-aware working point, AND the fixed-inhib CONTROL still fails. A PARTIAL / quantified NEGATIVE (the
assembly is fundamentally too small for a stable bistable attractor at any inhibition -> the residual is a
detonator-gain DG-larger-assembly lever or intrinsic-dendritic bistability) is a first-class honest deliverable.
SIM_BACKEND=cupy.
  Run: SIM_BACKEND=cupy python -m research.runners._gap5_size_aware_completion_derisk \
         --seeds 42 43 44 100 101 102 \
         --out research/findings/raw/_gap5_e2e/size_aware_completion_6seed.json
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402
from research.runners._gap5_emergent_end_to_end_episodic_loop_derisk import (  # noqa: E402
    emergent_assemblies, _readout_ca3_range, _jacc, R1, READ)
from research.runners._gap5_btsp_forms_nmda_slow_reverberatory_derisk import (  # noqa: E402
    run_seed as btsp_run_seed, _build_bridge as _readout_build_bridge)

cp, _ = get_backend()
OUT = _REPO / "research" / "findings" / "raw" / "_gap5_e2e" / "size_aware_completion.json"


def _threshold_hash(bridge):
    arr = getattr(bridge, "cp_neuron_firing_thresholds", None)
    if arr is None:
        return "none"
    return hashlib.sha1(np.asarray(to_host(arr)).tobytes()).hexdigest()[:12]


def _emergent_anticheat(seed, assemblies, r1_range, n_patterns, check_lesion=True):
    """Reproduce anti-cheat #1 from the e2e loop runner: the membership is DG-selected, not a hand-set mask."""
    from tools.lab import attributable_to
    t = {}
    sizes = [len(a) for a in assemblies]
    n_ca3 = r1_range[2]
    preassigned_size = max(6, int(READ["assembly_frac"] * n_ca3))
    t["emergent_not_preassigned_size"] = bool(max(sizes) < 0.5 * preassigned_size)
    rng = np.random.default_rng(seed)
    perm_idx = rng.permutation(n_ca3)
    lo = r1_range[0]
    default_asm = [set(int(lo + perm_idx[a * preassigned_size:(a + 1) * preassigned_size][k])
                       for k in range(preassigned_size)) for a in range(n_patterns)]
    t["jaccard_vs_preassigned"] = [round(_jacc(assemblies[i], default_asm[i]), 4) for i in range(n_patterns)]
    if check_lesion:
        les, _ = emergent_assemblies(seed, n_patterns=n_patterns, mossy_weight=0.0)
        t["lesion_sizes"] = [len(a) for a in les]
        t["mossy_lesion_collapses"] = bool(sum(len(a) for a in les) <= max(1, 0.2 * sum(sizes)))
        t["membership_attributable_to_mossy"] = attributable_to(
            f"[s{seed}] emergent-DG assembly SIZE: intact vs mossy-LESION (membership is DG-derived, not hand-set)",
            float(sum(sizes)), float(sum(len(a) for a in les)))
    t["anticheat1_emergent_membership"] = bool(
        t["emergent_not_preassigned_size"] and all(j <= 0.34 for j in t["jaccard_vs_preassigned"])
        and t.get("mossy_lesion_collapses", True))
    return t


def _go_row(r):
    return bool((r.get("held_cue") or 0) >= 0.20
                and (r.get("held_cue") or 0) >= 3 * ((r.get("held_perm") or 0) + 1e-6)
                and (r.get("held_cue") or 0) >= 3 * ((r.get("held_nocue") or 0) + 1e-6)
                and (r.get("held_nocue") or 0) <= 0.10)


def run_one_seed(seed, *, conditions, wmax, n_patterns, check_lesion, verbose=True):
    t = {"seed": seed}
    assemblies, r1_range = emergent_assemblies(seed, n_patterns=n_patterns)
    sizes = [len(a) for a in assemblies]
    t["assembly_sizes"] = sizes
    t["r1_ca3_range"] = r1_range
    if min(sizes) == 0:
        t["error"] = f"EMERGENT SELECTION produced an EMPTY assembly (sizes={sizes})"; return t
    t.update(_emergent_anticheat(seed, assemblies, r1_range, n_patterns, check_lesion=check_lesion))
    n_ca3 = r1_range[2]

    # index-space verify (ca3 global indices are basket-independent -> match holds with/without ff_basket)
    rr_range = _readout_ca3_range(seed, n_ca3, conditions[0]["density"])
    t["index_space_match"] = bool(rr_range == r1_range)

    rows = []
    for c in conditions:
        rr = btsp_run_seed(seed, n_ca3=n_ca3, ca3_density=c["density"], assembly_frac=READ["assembly_frac"],
                           cue_frac=READ["cue_frac"], ca3_fb_inhib=c["fb"], ca3_ff_inhib=c["ff"],
                           nmda_tau=READ["nmda_tau"], drive_pA=READ["drive_pA"], warm_steps=READ["warm_steps"],
                           read_steps=READ["read_steps"], enable_ou=False, element=READ["element"],
                           btsp_w_max_grid=[wmax], btsp_lr=READ["btsp_lr"], encode_drive=READ["encode_drive"],
                           encode_plateau_pA=READ["encode_plateau_pA"], train_events=READ["train_events"],
                           handinstall_W=[wmax], assemblies_ext=assemblies, verbose=verbose)
        for r in rr:
            r["cond"] = c["name"]; r["density"] = c["density"]; r["ca3_fb_inhib"] = c["fb"]; r["ca3_ff_inhib"] = c["ff"]
        rows.extend(rr)
        bt = [r for r in rr if r.get("arm") == "btsp"]
        hi = [r for r in rr if r.get("arm") == "handinstall"]
        b = bt[0] if bt else {}
        h = hi[0] if hi else {}
        if verbose:
            ff_lab = "--" if c["ff"] is None else f"{c['ff']:.0f}"
            b_go = "GO" if (_go_row(b) and b.get("genuine_formation")) else "--"
            h_go = "GO" if _go_row(h) else "--"
            print(f"  [s{seed} {c['name']:>14} d{c['density']:.2f} fb{c['fb']:.0f} ff{ff_lab}] "
                  f"BTSP cue={b.get('held_cue',0):.3f} perm={b.get('held_perm',0):.3f} nocue={b.get('held_nocue',0):.3f} "
                  f"w_in={b.get('w_within',0):.0f} genuine={b.get('genuine_formation')} {b_go} || "
                  f"HANDINST cue={h.get('held_cue',0):.3f} perm={h.get('held_perm',0):.3f} "
                  f"nocue={h.get('held_nocue',0):.3f} {h_go}", flush=True)
    t["rows"] = rows
    # per-seed verdict flags
    size_aware = [r for r in rows if r.get("arm") == "btsp" and r.get("ca3_ff_inhib") is not None]
    control = [r for r in rows if r.get("arm") == "btsp" and r.get("ca3_ff_inhib") is None]
    t["seed_size_aware_go"] = bool(any(_go_row(r) and r.get("genuine_formation") for r in size_aware))
    t["seed_control_go"] = bool(any(_go_row(r) and r.get("genuine_formation") for r in control))
    # decisive readout isolation: hand-install + size-aware
    hi_sa = [r for r in rows if r.get("arm") == "handinstall" and r.get("ca3_ff_inhib") is not None]
    t["seed_size_aware_handinstall_go"] = bool(any(_go_row(r) for r in hi_sa))
    return t


def build_summary(per, seeds, conditions, wmax, elapsed, err=None):
    from tools.verdict import Verdict
    valid = [p for p in per if not p.get("error")]
    n = len(valid)
    n_emergent = sum(1 for p in valid if p.get("anticheat1_emergent_membership"))
    n_index = sum(1 for p in valid if p.get("index_space_match"))
    n_sa = sum(1 for p in valid if p.get("seed_size_aware_go"))
    n_ctrl = sum(1 for p in valid if p.get("seed_control_go"))
    n_sa_hi = sum(1 for p in valid if p.get("seed_size_aware_handinstall_go"))
    all_btsp = [r for p in valid for r in p.get("rows", []) if r.get("arm") == "btsp"]
    all_genuine = bool(all_btsp) and all(r.get("genuine_formation") for r in all_btsp)

    v = Verdict("gap5 size-aware completion (emergent-DG select -> BTSP form -> size-aware FF-basket complete)")
    v.require("emergent membership anti-cheat holds (all valid seeds)", (n_emergent == n and n > 0), expect=True,
              note="the assembly membership is DG-selected, not a hand-set/pre-assigned mask")
    v.require("index-space match (all valid seeds)", (n_index == n and n > 0), expect=True)
    v.require("BTSP formation genuine (all btsp rows) -> a completion failure is a READOUT seam, not dead formation",
              all_genuine, expect=True)
    v.require("LOAD-BEARING: the fixed-inhib CONTROL (fb=60, ff=None) still FAILS (else size-aware is not the fix)",
              (n_ctrl == 0), expect=True, note=f"control GO {n_ctrl}/{n} must be 0")
    v.disabled("plasticity at recall (hebbian/stdp/btsp/bdsp)", why="isolation: the frozen attractor is the read var")
    v.disabled("OU membrane noise (OU-on NOT run)",
               why="the seam is DETERMINISTIC (present at OU-off, even HAND-INSTALLED); OU can only add self-ignition")

    go = bool(n_sa >= max(1, int(np.ceil(5 / 6 * len(seeds)))) and n_ctrl == 0 and n_emergent == n and n > 0)
    decided = v.decide(go=go)
    status = decided["status"]
    if go and status == "GO":
        verdict = (f"SEAM-CLOSED size-aware-GO {n_sa}/{n} | control-fails {n_ctrl}/{n} | hand-install-size-aware "
                   f"{n_sa_hi}/{n} | emergent {n_emergent}/{n}: the emergently-selected ~23-cell assembly completes "
                   f"CUE-SPECIFICALLY with the size-aware FF-basket completion, and the fixed-inhib control still fails")
    else:
        residual = ("size-aware inhibition did NOT recover specificity on the small emergent assemblies -> the residual "
                    "is DG producing LARGER assemblies (detonator-gain) or intrinsic-dendritic bistability"
                    if n_sa == 0 else
                    f"size-aware recovers specificity on {n_sa}/{n} seeds (PARTIAL); "
                    + ("but the control ALSO passes -> NOT load-bearing" if n_ctrl > 0 else "seed-robustness residual"))
        verdict = (f"{'PARTIAL' if n_sa > 0 else 'SEAM-STILL-NEGATIVE'} size-aware-GO {n_sa}/{n} | "
                   f"control-GO {n_ctrl}/{n} | hand-install-size-aware {n_sa_hi}/{n} | emergent {n_emergent}/{n}: "
                   + residual)
    if err is not None:
        verdict = f"ERROR -- {err}"; go = False
    return {"probe": "gap5_size_aware_completion", "GO": go, "status": status, "verdict": verdict, "seeds": seeds,
            "n_size_aware_go": n_sa, "n_control_go": n_ctrl, "n_size_aware_handinstall_go": n_sa_hi,
            "n_emergent_membership": n_emergent, "n_index_space_match": n_index,
            "conditions": conditions, "wmax": wmax, "elapsed_seconds": elapsed,
            "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
            "undefined_reasons": decided["undefined_reasons"], "per_seed": per}


def _parse_conditions(spec):
    """spec entries 'name:density:fb:ff' (ff '-' => None). Default = control + sparse size-aware sweep."""
    if not spec:
        spec = [
            "control_d012:0.12:60:-",
            "sa_d05_ff150:0.05:60:150",
            "sa_d05_ff400:0.05:60:400",
            "sa_d08_ff150:0.08:60:150",
            "sa_d08_ff400:0.08:60:400",
            "sa_d02_ff400:0.02:60:400",
        ]
    conds = []
    for s in spec:
        name, d, fb, ff = s.split(":")
        conds.append(dict(name=name, density=float(d), fb=float(fb), ff=(None if ff in ("-", "none", "None") else float(ff))))
    return conds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-patterns", type=int, default=3)
    ap.add_argument("--wmax", type=float, default=5000.0, help="BTSP saturation ceiling / hand-install W (single value)")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="'name:density:fb:ff' (ff '-'=None=control). Default=control + sparse size-aware sweep.")
    ap.add_argument("--no-lesion", action="store_true", help="skip the mossy-lesion anti-cheat (faster smoke)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    conditions = _parse_conditions(a.conditions)
    t0 = time.time(); err = None; per = []

    # build-twice threshold-hash determinism check (cfg.seed actually seeds the substrate)
    b1 = _readout_build_bridge(a.seeds[0], n_ca3=R1["n_ca3"], ca3_density=conditions[0]["density"],
                               ca3_fb_inhib=conditions[0]["fb"], nmda_tau=READ["nmda_tau"], nmda_ratio=READ["nmda_ratio"],
                               enable_ou=False, element=READ["element"], ca3_ff_inhib=conditions[0]["ff"])
    b2 = _readout_build_bridge(a.seeds[0], n_ca3=R1["n_ca3"], ca3_density=conditions[0]["density"],
                               ca3_fb_inhib=conditions[0]["fb"], nmda_tau=READ["nmda_tau"], nmda_ratio=READ["nmda_ratio"],
                               enable_ou=False, element=READ["element"], ca3_ff_inhib=conditions[0]["ff"])
    h1, h2 = _threshold_hash(b1), _threshold_hash(b2)
    print(f"[determinism] threshold-hash build1={h1} build2={h2} -> {'SEEDED' if h1 == h2 else 'UNSEEDED-BUG'}", flush=True)
    del b1, b2

    print(f"[gap5-SIZE-AWARE] emergent-DG select -> BTSP form -> size-aware FF-basket complete | seeds={a.seeds} "
          f"wmax={a.wmax} conditions={[c['name'] for c in conditions]}", flush=True)
    print("  GO: size-aware btsp cue>=0.20 & cue>=3x perm & cue>=3x nocue & nocue<=0.10 (>=5/6) AND control still fails",
          flush=True)
    try:
        for s in a.seeds:
            r = run_one_seed(s, conditions=conditions, wmax=a.wmax, n_patterns=a.n_patterns,
                             check_lesion=(not a.no_lesion), verbose=True)
            per.append(r)
            if r.get("error"):
                print(f"  [seed {s}] ERROR {r['error']}", flush=True); continue
            print(f"  [seed {s}] sizes {r['assembly_sizes']} emergent-OK={r['anticheat1_emergent_membership']} "
                  f"|| size-aware-GO={r['seed_size_aware_go']} control-GO={r['seed_control_go']} "
                  f"(hand-install-SA={r['seed_size_aware_handinstall_go']}) ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = build_summary(per, a.seeds, conditions, a.wmax, round(time.time() - t0, 1),
                            err=(err if (err is not None or not [p for p in per if not p.get("error")]) else None))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100 + f"\n[gap5-SIZE-AWARE] VERDICT: {summary['verdict']}\n[gap5-SIZE-AWARE] wrote {a.out}\n"
          + "=" * 100, flush=True)
    return 0 if summary["GO"] else 1


if __name__ == "__main__":
    sys.exit(main())
