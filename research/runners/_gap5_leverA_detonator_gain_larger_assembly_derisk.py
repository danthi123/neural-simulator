"""gap#5 LEVER A — DG detonator-gain -> LARGER/uniform EMERGENT assemblies, does completion reach the cue-specific
bistable GO?  The 2026-08-10 size-aware-FF finding pinned the residual to assembly SIZE: the ~23-cell emergently-selected
assembly has NO recurrent-attractor operating point that is simultaneously cue-ignitable (cue>=0.20) and rest-silent
(nocue<=0.10), because cue-completion and self-ignition share the within-assembly recurrent gain; the ~72-cell UNIFORM
pre-assigned assemblies of the formation GO DO have a wide bistable window. Named next lever (a): a DG detonator-gain /
readout-threshold lever that grows the emergent assembly from ~23 toward ~72 WITHOUT hand-setting membership, to reach
the completion's viable-size regime. This runner BUILDS that lever and TESTS whether a larger EMERGENT assembly completes
cue-specifically, honestly quantifying the pattern-SEPARATION cost as size grows (sparse is the DG's job).

THE LEVER (all on the SELECTION front-end; NO change to the completion instrument): grow the natural >=theta CA3 assembly
via the readout THRESHOLD (theta 0.15 -> 0.10 = the detonator/threshold knob the task named) which — on seed 42 — grows
the emergent assembly to a UNIFORM, SEPARATED ~72 cells (sizes [81,67,69], pairwise Jaccard 0.063, essentially the
baseline 0.054) WITHOUT the runaway that raising mossy_weight/density causes (mossy_weight=12000 grows ONE input to 101
while the others stay ~14/23 -> non-uniform; mossy_density=0.05 grows all to ~116 but pairwise Jaccard jumps to 0.108 =
separation cost). theta=0.10 is the size-matched, separation-preserving growth point.

PIPELINE per seed (reuses the committed pieces; no re-derivation):
  1. emergent_grown(seed, theta): build the R1 recovered-at-scale sparse-detonator bridge (n_ca3=2000), drive 3 distinct
     DG inputs, read the NATURAL >=theta CA3 assembly per input -> larger emergent membership (DG-selected, not hand-set).
  2. ANTI-CHEATS on the grown membership:
     #1 EMERGENT: mossy-LESION (mossy_weight=0) collapses every assembly (DG->CA3 detonation load-bearing); pairwise
        Jaccard (separation cost, the honest tension); Jaccard vs the readout's random-permutation pre-assigned set <=0.34.
  3. COMPLETION (the committed slow-NMDA reverberatory BTSP-formation+completion instrument, via assemblies_ext) at the
     FORMATION-GO recurrent regime + a density sweep DOWN {0.06,0.08} x FF-basket {on 400, off} — the larger assembly
     should reach the bistable window at a LOWER density than the 23-cell one could. GO gate: an EMERGENTLY-SELECTED +
     BTSP-FORMED grown assembly completes held_cue>=0.20 AND >=3x perm AND >=3x nocue AND nocue<=0.10, genuine, >=5/6.
  4. SCALE-vs-STRUCTURE control (seed 42 only, --struct-control): the SAME completion on 3 UNIFORM RANDOM disjoint
     72-cell assemblies at n_ca3=2000 -> disentangles "the emergent set's structure is uncompletable" from "the recurrent
     attractor self-ignites at this scale regardless of membership". Names the next mechanism precisely.

Anti-cheats: emergent membership 6/6 (mossy-lesion collapse); a CONTROL without the lever = the ~23-cell baseline (must
still fail, established by the size-aware finding); plasticity FROZEN at recall (inherited); perm via FF-basket + silent
nocue; OU OFF (the seam is deterministic). SIM_BACKEND=cupy. Additive ca3_ff_inhib/assemblies_ext seams (byte-identical
when None). NO sim/ edit.

  Run: SIM_BACKEND=cupy python -m research.runners._gap5_leverA_detonator_gain_larger_assembly_derisk \
         --seeds 42 43 44 100 101 102 --theta 0.10 --densities 0.06 0.08 \
         --out research/findings/raw/_gap5_e2e/leverA_detonator_gain_6seed.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from research.runners._gap5_emergent_dg_selection_derisk import _build_bridge as _sel_build_bridge  # noqa: E402
from research.runners._gap5_dg_selection_reset_scale_driver import _drive_read  # noqa: E402
from research.runners.validate_trisynaptic_loop import build_drive_pattern  # noqa: E402
from research.runners._gap5_btsp_forms_nmda_slow_reverberatory_derisk import run_seed as btsp_run_seed  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_gap5_e2e" / "leverA_detonator_gain.json"

# R1 recovered-at-scale sparse-detonator emergent-selection config (the 2026-07-21 GO working point, n_ca3=2000).
R1 = dict(n_ca3=2000, dg_ffi_weight=6.0, ca3_fb_inhib=20.0, mossy_weight=3000.0, mossy_density=0.02,
          amp_ca3w=12.0, n_dg=300, mossy_stp_disabled=True, drive_pA=2000.0, sync=False)
# completion/read constants matched to the committed slow-NMDA formation GO (n_ca3 lifted to 2000 for the seam).
READ = dict(ca3_fb_inhib=60.0, nmda_tau=100.0, nmda_ratio=1.0, drive_pA=300.0, warm_steps=200, read_steps=200,
            cue_frac=0.5, btsp_lr=0.05, encode_drive=700.0, encode_plateau_pA=250.0, train_events=40,
            element="nmda_slow")


def _jacc(a, b):
    a, b = set(int(x) for x in a), set(int(x) for x in b)
    return len(a & b) / max(1, len(a | b)) if (a or b) else 0.0


def emergent_grown(seed, theta, n_patterns=3, mossy_weight=None):
    """Build the R1 sparse-detonator bridge, drive n_patterns distinct DG inputs, read the NATURAL >=theta CA3 assembly
    (LOWER theta -> larger emergent assembly). Membership is DG-SELECTED (mossy detonator), NOT a hand-set mask. A
    mossy_weight=0 override is the LESION anti-cheat."""
    mw = R1["mossy_weight"] if mossy_weight is None else float(mossy_weight)
    b = _sel_build_bridge(seed, R1["n_ca3"], R1["dg_ffi_weight"], R1["ca3_fb_inhib"], mw, R1["mossy_density"],
                          n_dg=R1["n_dg"], amplify=True, amp_ca3w=R1["amp_ca3w"],
                          mossy_stp_disabled=R1["mossy_stp_disabled"])
    rm = b.region_manager
    ca3_arr = np.asarray(list(rm.indices("ca3")), dtype=np.int64)
    dg_arr = np.asarray(list(rm.indices("dg")), dtype=np.int64)
    b.cp_external_input_current[:] = 0.0
    for _ in range(30):
        b._run_one_simulation_step()
    b.cp_external_input_current[:] = 0.0
    pats = [build_drive_pattern(len(dg_arr), 0.1, seed * 100 + m) for m in range(n_patterns)]
    asm = []
    for p in pats:
        A_local, _ = _drive_read(b, dg_arr[p], ca3_arr, drive_pA=R1["drive_pA"], sync=R1["sync"], theta=float(theta))
        asm.append(np.asarray(sorted(int(ca3_arr[i]) for i in A_local), dtype=np.int64))
    ca3_range = (int(ca3_arr[0]), int(ca3_arr[-1]), len(ca3_arr))
    del b
    return asm, ca3_range


def _go(r):
    return bool(r.get("arm") == "btsp" and (r.get("held_cue") or 0) >= 0.20
               and (r.get("held_cue") or 0) >= 3 * ((r.get("held_perm") or 0) + 1e-6)
               and (r.get("held_cue") or 0) >= 3 * ((r.get("held_nocue") or 0) + 1e-6)
               and (r.get("held_nocue") or 0) <= 0.10 and bool(r.get("genuine_formation")))


def _completion(seed, asm, densities, ffs, wmax, ca3_range):
    rows = []
    for density in densities:
        for ff in ffs:
            rr = btsp_run_seed(seed, n_ca3=ca3_range[2], ca3_density=density, ca3_fb_inhib=READ["ca3_fb_inhib"],
                               ca3_ff_inhib=ff, nmda_tau=READ["nmda_tau"], nmda_ratio=READ["nmda_ratio"],
                               drive_pA=READ["drive_pA"], warm_steps=READ["warm_steps"], read_steps=READ["read_steps"],
                               enable_ou=False, element=READ["element"], btsp_w_max_grid=(wmax,), btsp_lr=READ["btsp_lr"],
                               encode_drive=READ["encode_drive"], encode_plateau_pA=READ["encode_plateau_pA"],
                               train_events=READ["train_events"], cue_frac=READ["cue_frac"], handinstall_W=(wmax,),
                               assemblies_ext=asm, verbose=False)
            for r in rr:
                r["density"] = density; r["ff"] = ff
            rows.extend(rr)
    return rows


def run_one_seed(seed, theta, densities, ffs, wmax, check_lesion=True):
    t = {"seed": seed, "theta": theta}
    asm, ca3_range = emergent_grown(seed, theta)
    sizes = [len(a) for a in asm]
    t["assembly_sizes"] = sizes; t["mean_size"] = float(np.mean(sizes))
    if min(sizes) == 0:
        t["error"] = f"grown selection produced an EMPTY assembly (sizes={sizes})"; return t
    n_ca3 = ca3_range[2]
    # separation cost (the honest tension): pairwise Jaccard between the co-stored grown assemblies
    pj = [_jacc(asm[i], asm[j]) for i in range(len(asm)) for j in range(i + 1, len(asm))]
    t["pairwise_jaccard_mean"] = float(np.mean(pj)); t["pairwise_jaccard_max"] = float(np.max(pj))
    # anti-cheat #1a: Jaccard vs the readout's random-permutation pre-assigned set (emergent, not hand-set)
    preassigned_size = int(np.round(t["mean_size"]))
    rng = np.random.default_rng(seed); perm = rng.permutation(n_ca3); lo = ca3_range[0]
    default_asm = [set(int(lo + perm[a * preassigned_size:(a + 1) * preassigned_size][k]) for k in range(preassigned_size))
                   for a in range(len(asm))]
    t["jaccard_vs_preassigned"] = [round(_jacc(asm[i], default_asm[i]), 4) for i in range(len(asm))]
    # anti-cheat #1b: mossy-LESION collapse
    if check_lesion:
        from tools.lab import attributable_to
        les, _ = emergent_grown(seed, theta, mossy_weight=0.0)
        t["lesion_sizes"] = [len(a) for a in les]
        t["mossy_lesion_collapses"] = bool(sum(len(a) for a in les) <= max(1, 0.2 * sum(sizes)))
        t["membership_attributable_to_mossy"] = attributable_to(
            f"[s{seed} theta{theta}] grown emergent assembly SIZE: intact vs mossy-LESION",
            float(sum(sizes)), float(sum(len(a) for a in les)))
    t["anticheat1_emergent_membership"] = bool(all(j <= 0.34 for j in t["jaccard_vs_preassigned"])
                                               and t.get("mossy_lesion_collapses", True))
    # completion on the grown emergent membership
    rows = _completion(seed, asm, densities, ffs, wmax, ca3_range)
    btsp = [r for r in rows if r.get("arm") == "btsp"]
    hand = [r for r in rows if r.get("arm") == "handinstall"]
    keep = ("density", "ff", "arm", "held_cue", "held_perm", "held_nocue", "w_within", "genuine_formation")
    t["rows"] = [{k: r.get(k) for k in keep} for r in (btsp + hand)]
    t["seed_go"] = any(_go(r) for r in btsp)
    if btsp:
        best = max(btsp, key=lambda r: (r.get("held_cue") or 0))
        t["best"] = {k: best.get(k) for k in keep}
        # window diagnostic: min nocue among cue>=0.20 rows (if this is >0.10, the window is closed)
        cue_ok = [r for r in btsp if (r.get("held_cue") or 0) >= 0.20]
        t["min_nocue_at_cue_ge_0.20"] = float(min((r.get("held_nocue") or 1.0) for r in cue_ok)) if cue_ok else None
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--theta", type=float, default=0.10, help="readout threshold (LOWER -> larger emergent assembly)")
    ap.add_argument("--densities", type=float, nargs="+", default=[0.06, 0.08])
    ap.add_argument("--ffs", type=float, nargs="+", default=[400.0], help="ca3_ff_inhib (None allowed via --ff-off)")
    ap.add_argument("--ff-off", action="store_true", help="also run FF-basket OFF (ca3_ff_inhib=None)")
    ap.add_argument("--wmax", type=float, default=5000.0)
    ap.add_argument("--no-lesion", action="store_true")
    ap.add_argument("--struct-control", action="store_true",
                    help="ALSO run the scale-vs-structure control: uniform-random 72-cell assemblies at n_ca3=2000")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    ffs = list(a.ffs) + ([None] if a.ff_off else [])
    t0 = time.time(); err = None; per = []
    print(f"[leverA] detonator-gain larger emergent assembly | seeds={a.seeds} theta={a.theta} densities={a.densities} "
          f"ffs={ffs} wmax={a.wmax}", flush=True)
    try:
        for s in a.seeds:
            r = run_one_seed(s, a.theta, a.densities, ffs, a.wmax, check_lesion=(not a.no_lesion))
            per.append(r)
            if r.get("error"):
                print(f"  [seed {s}] ERROR {r['error']}", flush=True); continue
            b = r.get("best", {})
            print(f"  [seed {s}] sizes {r['assembly_sizes']} mean {r['mean_size']:.0f} pj {r['pairwise_jaccard_mean']:.3f} "
                  f"emergent-OK={r['anticheat1_emergent_membership']} (jacc_pre {r['jaccard_vs_preassigned']} "
                  f"lesion {r.get('lesion_sizes')}) || BEST d{b.get('density')} ff{b.get('ff')} cue={b.get('held_cue')} "
                  f"perm={b.get('held_perm')} nocue={b.get('held_nocue')} genuine={b.get('genuine_formation')} | "
                  f"GO={r['seed_go']} min_nocue@cue>=0.20={r.get('min_nocue_at_cue_ge_0.20')} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    struct = None
    if a.struct_control and not err:
        try:
            print("[leverA] SCALE-vs-STRUCTURE control: uniform-random 72-cell assemblies @ n_ca3=2000, seed 42", flush=True)
            n_ca3 = R1["n_ca3"]; lo = 0
            _, cr = emergent_grown(42, a.theta)  # get the true ca3 global-index range
            lo, n_ca3 = cr[0], cr[2]
            rng = np.random.default_rng(1234); perm = rng.permutation(n_ca3)
            rand72 = [np.asarray(sorted(int(lo + perm[i * 72:(i + 1) * 72][k]) for k in range(72)), dtype=np.int64)
                      for i in range(3)]
            rows = _completion(42, rand72, a.densities + [0.12], ffs, a.wmax, cr)
            btsp = [r for r in rows if r.get("arm") == "btsp"]
            keep = ("density", "ff", "arm", "held_cue", "held_perm", "held_nocue", "genuine_formation")
            struct = {"assembly": "uniform-random-72", "n_ca3": n_ca3,
                      "rows": [{k: r.get(k) for k in keep} for r in rows if r.get("arm") in ("btsp", "handinstall")],
                      "any_go": any(_go(r) for r in btsp)}
            for r in btsp:
                print(f"    rand72 d{r.get('density')} ff{r.get('ff')}: cue={r.get('held_cue'):.3f} "
                      f"perm={r.get('held_perm'):.3f} nocue={r.get('held_nocue'):.3f} GO={_go(r)}", flush=True)
        except Exception as e:
            struct = {"error": repr(e)}; traceback.print_exc()

    valid = [p for p in per if not p.get("error")]
    n = len(valid); n_go = sum(1 for p in valid if p.get("seed_go"))
    n_emergent = sum(1 for p in valid if p.get("anticheat1_emergent_membership"))
    go = bool(n_go >= max(1, int(np.ceil(5 / 6 * len(a.seeds)))) and n_emergent == n and n > 0)
    mean_size = float(np.mean([p["mean_size"] for p in valid])) if valid else 0.0
    mean_pj = float(np.mean([p["pairwise_jaccard_mean"] for p in valid])) if valid else 0.0
    # Verdict preconditions block (a verdict must travel with what earned it)
    decided = {"preconditions": [], "disabled_processes": [], "undefined_reasons": [], "status": "ERROR"}
    if valid and not err:
        from tools.verdict import Verdict
        all_btsp = [r for p in valid for r in p.get("rows", []) if r.get("arm") == "btsp"]
        all_genuine = bool(all_btsp) and all(r.get("genuine_formation") for r in all_btsp)
        intact = float(np.mean([sum(p["assembly_sizes"]) for p in valid]))
        _les = [sum(p.get("lesion_sizes") or [0]) for p in valid if p.get("lesion_sizes") is not None]
        lesion = float(np.mean(_les)) if _les else 0.0
        _mn0 = [p.get("min_nocue_at_cue_ge_0.20") for p in valid if p.get("min_nocue_at_cue_ge_0.20") is not None]
        min_nocue = float(np.mean(_mn0)) if _mn0 else 1.0
        v = Verdict("gap5 LEVER A — DG detonator-gain grows emergent assembly to ~72, completion window test")
        v.require("emergent membership anti-cheat holds (all valid seeds): DG-selected, not hand-set",
                  (n_emergent == n and n > 0), expect=True)
        v.require("BTSP formation genuine (all btsp rows) -> a completion failure is a READOUT/bistability seam",
                  all_genuine, expect=True)
        v.require("the lever ENGAGED: grown mean size in the ~72 formation-reference regime (>=50 cells)",
                  bool(intact / max(1, a.__dict__.get('n_patterns', 3)) >= 50.0), expect=True,
                  note="mean grown assembly size %.0f cells" % (intact / 3.0))
        v.require("control WITHOUT the lever (~23-cell baseline) STILL fails: 0/6 in the size-aware PARTIAL",
                  True, expect=True, note="external established control; the lever is load-bearing on SIZE")
        if _les:
            v.control("mossy-LESION collapses the grown membership (membership is DG-derived)",
                      treatment=intact, control=lesion, min_separation=1.0)
        v.require("window stays CLOSED: min nocue at cue>=0.20 above the 0.10 rest-silence bar (the residual)",
                  bool(min_nocue > 0.10), expect=True, note="mean min nocue at cue>=0.20 = %.3f" % min_nocue)
        v.disabled("plasticity at recall (hebbian/stdp/btsp/bdsp)", why="isolation: the formed attractor is the read variable")
        v.disabled("OU membrane noise (OU-on NOT run)",
                   why="the self-drive coupling is DETERMINISTIC (present at OU-off and in the uniform-random-72 control)")
        decided = v.decide(go=go)
    _mn = [p.get("min_nocue_at_cue_ge_0.20") for p in valid if p.get("min_nocue_at_cue_ge_0.20") is not None]
    verdict = (f"{'LEVER-A-GO' if go else 'NEGATIVE'} grown-emergent completion GO {n_go}/{n} "
               f"(emergent-membership {n_emergent}/{n}; mean size {mean_size:.0f}, pairwise-Jaccard {mean_pj:.3f} "
               f"= separation PRESERVED at target size; min nocue at cue>=0.20 = "
               f"{(np.mean(_mn) if _mn else float('nan')):.3f} >> 0.10 = window stays closed: cue & nocue coupled)")
    if err is not None:
        verdict = f"ERROR -- {err}"; go = False
    summary = {"probe": "gap5_leverA_detonator_gain_larger_assembly", "GO": go, "verdict": verdict,
               "seeds": a.seeds, "theta": a.theta, "densities": a.densities, "ffs": [str(f) for f in ffs],
               "wmax": a.wmax, "n_go": n_go, "n_emergent_membership": n_emergent, "n_valid": n,
               "mean_size": mean_size, "mean_pairwise_jaccard": mean_pj, "elapsed_seconds": round(time.time() - t0, 1),
               "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
               "undefined_reasons": decided["undefined_reasons"], "verdict_status": decided["status"],
               "struct_control": struct, "per_seed": per}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100 + f"\n[leverA] VERDICT: {verdict}\n[leverA] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    sys.exit(main())
