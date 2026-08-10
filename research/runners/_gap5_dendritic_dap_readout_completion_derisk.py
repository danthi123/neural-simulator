"""gap#5 LEVER B — intrinsic per-cell DENDRITIC dAP READOUT bistability (SIZE-INDEPENDENT) completes the EMERGENT
~23-cell BTSP-formed assembly where the RECURRENT slow-NMDA reverberatory attractor could not.

THE SEAM (just characterized, 544c0b742 / cff6a8e2f): the emergently-selected assemblies are SMALL (~23 cells) and the
RECURRENT completion is non-specific on them (perm~=nocue~=cue), because cue-completion + self-ignition share the
WITHIN-assembly recurrent gain -- a ~23-cell set is too small for a RECURRENT bistable attractor at any inhibition.

LEVER B replaces the RECURRENT-attractor completion READ with an INTRINSIC per-cell dendritic dAP READOUT bistability.
Each CA3 cell's APICAL dendrite (enable_two_compartment_dap machinery: fused_coincidence_plateau regenerating on
cp_v_apical + self-regen + KIR down-state) holds its OWN bistable UP/DOWN state. A partial cue's within-assembly
recurrent volley drives the held-out cell's apical coincident drive c_drive; if it crosses coincidence_k_threshold the
apical NMDA plateau IGNITES and LATCHES (self_regen + KIR) -- per-cell, so cue-ignition is DECOUPLED from a large
recurrent population. This is the 2026-07-08 R-iii dAP completion (GO 0.571 vs LINEAR 0.007 on HAND-INSTALLED
attractors, research/findings/2026-07-08-riii-onsubstrate-dendritic-dAP-completion-SURPASS-6seed.md) applied to the
EMERGENT small assembly.

Two READS (the 2026-07-18 magnitude-capped payoff named the second as the cheapest UNTESTED next-mechanism):
  SOMA read   -- count held-cell SOMA firing (the payoff finding's read; capped cue 0.156 on PRE-ASSIGNED assemblies
                 because the assembly's OWN within-member recurrent loop re-closes through the soma read;
                 research/findings/2026-07-18-gap5-CA3-bistable-dendrite-payoff-bistability+specificity-solved-magnitude-capped.md).
  APICAL read -- the DECOUPLED read (that finding's next-mechanism #1, never tested): fraction of held cells whose
                 cp_v_apical is in the UP state, with WEAK apical<->soma back-coupling so the plateau HOLDS (completion)
                 without the soma firing hard enough to re-drive the recurrent loop. Biologically the apical UP state IS
                 the held memory; the soma spike is the output.
WHY the EMERGENT small assembly could FREE this where PRE-ASSIGNED capped it: the payoff cap came from the assembly's
within-member recurrent loop; a ~23-cell emergent assembly has a MUCH weaker within-member loop than the 0.18*N
pre-assigned one -- exactly the smallness that killed the recurrent attractor may free the per-cell read.

EXPLICITLY DISTINCT from the tested-NEGATIVE dendritic deep-CREDIT / BDSP rule
(research/findings/2026-05-17-dendritic-credit-assignment-NEGATIVE.md): this is a READOUT-bistability lever (a dendritic
plateau READING a learned recurrent weight), NOT a hidden-credit learning rule. BDSP learning is OFF throughout.

REUSE (no re-derive): emergent membership + mossy-lesion anti-cheat from _gap5_emergent_end_to_end_episodic_loop_derisk;
BTSP one-shot formation (form_btsp_multi) + masks/eval (make_readout) from _gap5_btsp_forms_nmda_slow_reverberatory_derisk
VERBATIM -- ONLY the READOUT bridge changes (coincidence+two_comp dAP instead of nmda_slow reverberatory). NO sim/ edit.

ANTI-CHEATS: emergent membership 6/6 (mossy-LESION collapses); LEVER LOAD-BEARING (LINEAR control = coincidence OFF,
point-neuron read, SAME formed weights -> must STILL fail); genuine BTSP formation (within grew from the rule, cross~0);
plasticity FROZEN at recall; permuted cue -> ~0; silent-rest nocue<=0.10; recurrence-zero collapses; OU off.

GO (6-seed 42/43/44/100/101/102): an EMERGENTLY-SELECTED + BTSP-FORMED assembly completes cue-specifically via the dAP
read -- held_cue>=0.20 AND held_cue>=3*held_perm AND held_cue>=3*held_nocue AND held_nocue<=0.10 -- on >=5/6 seeds, at
some (density,wmax,k_thresh) working point, where the LINEAR control still fails. Honest negative otherwise (quantify
the residual + name the next mechanism). SIM_BACKEND=cupy.
  Run: SIM_BACKEND=cupy python -m research.runners._gap5_dendritic_dap_readout_completion_derisk \
         --seeds 42 43 44 100 101 102 --densities 0.5 --wmax 100 --kthresh 15 30 \
         --out research/findings/raw/_gap5_dapB/dapB_6seed.json
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from sim.backend import get_backend  # noqa: E402
from research.runners._riii_ca3_coincidence_completion_derisk import _build  # noqa: E402
from research.runners._gap5_btsp_forms_nmda_slow_reverberatory_derisk import (  # noqa: E402
    make_readout, form_btsp_multi, _build_bridge as _formation_build_bridge)
from research.runners._gap5_emergent_end_to_end_episodic_loop_derisk import (  # noqa: E402
    emergent_assemblies, _jacc, R1 as _R1)

OUT = _REPO / "research" / "findings" / "raw" / "_gap5_dapB" / "dapB.json"


def _build_dap_readout(seed, *, n_ca3, ca3_density, ca3_fb_inhib, k_thresh, plateau_strength, apical_R,
                       self_regen, v_hold, apical_kir_g, apical_gc, apical_gc_read, coincidence=True):
    """Build the READOUT bridge with the per-cell dendritic dAP readout on the ca3->ca3 recurrent (coincidence + two-
    compartment). coincidence=False => the LINEAR point-neuron control (byte-identical connectivity, SAME formed
    weights read at the point soma -> the lever-load-bearing control). Plasticity FROZEN. NO nmda_recurrent (this is the
    dAP read, not the reverberatory attractor)."""
    b = _build(seed, n_ca3=n_ca3, ca3_density=ca3_density, coincidence=coincidence, k_thresh=k_thresh,
               plateau_strength=plateau_strength, weighted=True, two_comp=coincidence, train=False,
               nmda_recurrent=False, ca3_fb_inhib=ca3_fb_inhib, enable_ou=False,
               plateau_self_regen=(self_regen if coincidence else 0.0), plateau_v_hold=v_hold,
               apical_kir_g=(apical_kir_g if coincidence else 0.0), apical_R=(apical_R if coincidence else None),
               apical_gc=(apical_gc if coincidence else None),
               apical_gc_read=(apical_gc_read if coincidence else None))
    cfg = b.core_config
    cfg.enable_hebbian_learning = False; cfg.enable_stdp = False; cfg.enable_structural_plasticity = False
    cfg.enable_btsp = False; cfg.enable_bdsp = False
    return b


def _reset_apical_latch(bridge):
    """Clear the intrinsic apical latch (cp_v_apical) + the plateau conductance to REST before a read. CRITICAL: the
    committed hard_silence resets the SOMA (v/u/firing/g_e/g_i/g_nmda) but NOT the apical compartment or the coincidence
    plateau conductance -- so without this, a prior read's self_regen UP-latch carries over and every subsequent read
    (even nocue) reads UP (the specificity-destroying carryover bug found 2026-08-10). Each recall episode starts from
    the apical DOWN state at rest, which is the correct biology."""
    cp = get_backend()[0]
    if getattr(bridge, "cp_v_apical", None) is not None:
        bridge.cp_v_apical[:] = cp.float32(getattr(bridge.core_config, "apical_E_rest", -65.0))
    for _a in ("cp_conductance_g_coincidence", "cp_conductance_g_coincidence_rise"):
        _arr = getattr(bridge, _a, None)
        if _arr is not None:
            _arr[:] = 0.0


def _apical_up_read(bridge, R, held_pos_by_asm, cue_by_asm, up_thresh):
    """Drive each assembly's cue (via R.drive_read machinery: hard-silence -> RESET the apical latch -> drive cue ->
    warm+read), then read the fraction of held-out cells whose cp_v_apical is in the UP state (> up_thresh). Returns
    mean over assemblies. cp_v_apical is the intrinsically-bistable apical latch (the DECOUPLED memory-holding read)."""
    cp = R.cp; to_host = R.to_host
    ca3_idx = R.ca3_idx
    fracs = []
    for A_i, (held_pos, cue_g) in enumerate(zip(held_pos_by_asm, cue_by_asm)):
        R.hard_silence(); _reset_apical_latch(bridge)
        if cue_g is not None and len(cue_g) > 0:
            darr = cp.asarray(np.asarray(cue_g, dtype=np.int64), dtype=cp.int64)
            bridge.cp_external_input_current[darr] = cp.float32(R._drive_pA)
        else:
            darr = None
        for _ in range(R._warm + R._read):
            bridge._run_one_simulation_step(); bridge.runtime_state.current_time_step += 1
        if getattr(bridge, "cp_v_apical", None) is None:
            fracs.append(0.0)
        else:
            va = to_host(bridge.cp_v_apical)
            held_global = [int(ca3_idx[p]) for p in held_pos]
            up = np.mean([1.0 if va[g] > up_thresh else 0.0 for g in held_global]) if held_global else 0.0
            fracs.append(float(up))
        if darr is not None:
            bridge.cp_external_input_current[darr] = 0.0
    return float(np.mean(fracs)) if fracs else 0.0


def _held_cue_perm(R, seed):
    """Return (soma held_cue, soma held_perm, held_pos_by_asm, cue_by_asm, perm_by_asm) reusing make_readout's
    eval_assembly geometry so the SOMA read is identical to the committed instrument."""
    held_pos_by_asm, cue_by_asm, perm_by_asm = [], [], []
    cues, perms = [], []
    for A in R.assemblies:
        se = np.asarray(A, dtype=np.int64)
        r = np.random.default_rng(seed * 131 + int(se[0]))
        se = se[r.permutation(len(se))]
        n_cue = max(2, int(R._cue_frac * len(se)))
        cue, held = se[:n_cue], se[n_cue:]
        held_pos = [R.ca3_pos[int(g)] for g in held]
        member = set(int(g) for g in se)
        nonA = np.asarray([g for g in R.ca3_idx if int(g) not in member], dtype=np.int64)
        perm_cue = r.choice(nonA, size=len(cue), replace=False)
        held_pos_by_asm.append(held_pos); cue_by_asm.append(cue); perm_by_asm.append(perm_cue)
        cues.append((cue, held_pos)); perms.append((perm_cue, held_pos))
    return held_pos_by_asm, cue_by_asm, perm_by_asm


def run_one_seed(seed, *, densities, wmax_grid, kthresh_grid, plateau_strength, apical_R, self_regen, v_hold,
                 apical_kir_g, apical_gc, apical_gc_read, up_thresh, ca3_fb_inhib, btsp_lr, encode_drive,
                 encode_plateau_pA, train_events, drive_steps, reset_steps, assembly_frac, cue_frac, drive_pA,
                 warm_steps, read_steps, silence_steps, n_patterns, check_lesion=True, verbose=True):
    from sim.backend import get_backend, to_host
    from tools.lab import attributable_to
    cp, _ = get_backend()
    t = {"seed": seed}

    # ---- STEP 1: EMERGENT SELECTION (membership anti-cheat) ---------------------------------------------------------
    assemblies, r1_range = emergent_assemblies(seed, n_patterns=n_patterns)
    sizes = [len(a) for a in assemblies]
    t["assembly_sizes"] = sizes; t["r1_ca3_range"] = r1_range
    if min(sizes) == 0:
        t["error"] = f"EMERGENT SELECTION produced an EMPTY assembly (sizes={sizes})"; return t
    n_ca3 = r1_range[2]
    preassigned_size = max(6, int(assembly_frac * n_ca3))
    t["emergent_not_preassigned_size"] = bool(max(sizes) < 0.5 * preassigned_size)
    rng = np.random.default_rng(seed)
    perm_idx = rng.permutation(n_ca3); lo = r1_range[0]
    default_asm = [set(int(lo + perm_idx[a * preassigned_size:(a + 1) * preassigned_size][k])
                       for k in range(preassigned_size)) for a in range(n_patterns)]
    t["jaccard_vs_preassigned"] = [round(_jacc(assemblies[i], default_asm[i]), 4) for i in range(n_patterns)]
    if check_lesion:
        les, _ = emergent_assemblies(seed, n_patterns=n_patterns, mossy_weight=0.0)
        t["lesion_sizes"] = [len(a) for a in les]
        t["mossy_lesion_collapses"] = bool(sum(len(a) for a in les) <= max(1, 0.2 * sum(sizes)))
        t["membership_attributable_to_mossy"] = attributable_to(
            f"[s{seed}] emergent-DG assembly SIZE: intact vs mossy-LESION", float(sum(sizes)),
            float(sum(len(a) for a in les)))
    emergent_ok = (t["emergent_not_preassigned_size"] and all(j <= 0.34 for j in t["jaccard_vs_preassigned"])
                   and t.get("mossy_lesion_collapses", True))
    t["anticheat1_emergent_membership"] = bool(emergent_ok)

    read_kwargs = dict(assembly_frac=assembly_frac, cue_frac=cue_frac, drive_pA=drive_pA, warm_steps=warm_steps,
                       read_steps=read_steps, silence_steps=silence_steps, assemblies_ext=assemblies)
    # formation episodes are built by the committed btsp instrument (coincidence OFF, nmda_slow); only WEIGHTS transfer.
    form_build_kwargs = dict(n_ca3=n_ca3, ca3_density=None, ca3_fb_inhib=ca3_fb_inhib, ca3_ff_inhib=None,
                             nmda_tau=100.0, nmda_ratio=1.0, enable_ou=False, element="nmda_slow")

    rows = []
    for density in densities:
        form_build_kwargs["ca3_density"] = density
        for wmax in wmax_grid:
            for k_thresh in kthresh_grid:
                k_thresh = float(k_thresh)
                # ---- build the dAP readout bridge + form the BTSP weights on it -------------------------------------
                bridge = _build_dap_readout(seed, n_ca3=n_ca3, ca3_density=density, ca3_fb_inhib=ca3_fb_inhib,
                                            k_thresh=k_thresh, plateau_strength=plateau_strength, apical_R=apical_R,
                                            self_regen=self_regen, v_hold=v_hold, apical_kir_g=apical_kir_g,
                                            apical_gc=apical_gc, apical_gc_read=apical_gc_read, coincidence=True)
                R = make_readout(bridge, seed, **read_kwargs)
                # NO-ENCODING baseline (weak recurrent) -- both reads must be ~0
                held_pos_by_asm, cue_by_asm, perm_by_asm = _held_cue_perm(R, seed)
                base_soma = float(np.mean([R.eval_assembly(A)["held_cue"] for A in R.assemblies]))
                base_apical = _apical_up_read(bridge, R, held_pos_by_asm, cue_by_asm, up_thresh)
                # FORM the emergent within-assembly attractor by BTSP one-shot (weights = the rule's output)
                diag = form_btsp_multi(seed, form_build_kwargs, R, btsp_w_max=wmax, btsp_lr=btsp_lr,
                                       encode_drive=encode_drive, encode_plateau_pA=encode_plateau_pA,
                                       train_events=train_events, drive_steps=drive_steps, reset_steps=reset_steps,
                                       plateau=True, assemblies_ext=assemblies)
                w1_within, cross_dw, nonmem_dw = diag["w_within"], diag["cross_dw"], diag["nonmem_dw"]
                # baseline recurrent weight is 1.5 -> genuine = grew WELL above baseline (small-weight dAP regime uses
                # wmax~100 -> w_within~85, so the old >100 reverberatory threshold is wrong here) + cross/non ~0.
                genuine = bool(w1_within > 20.0 and w1_within <= wmax * 1.01
                               and abs(cross_dw) < 0.02 * w1_within and abs(nonmem_dw) < 0.02 * w1_within)
                # ---- READ 1: SOMA firing (the payoff finding's read) ------------------------------------------------
                soma_cue = float(np.mean([R.eval_assembly(A)["held_cue"] for A in R.assemblies]))
                soma_perm = float(np.mean([R.eval_assembly(A)["held_perm"] for A in R.assemblies]))
                soma_nocue = R.no_cue_rest()
                # ---- READ 2: APICAL UP-state (the DECOUPLED read, untested next-mechanism) ---------------------------
                ap_cue = _apical_up_read(bridge, R, held_pos_by_asm, cue_by_asm, up_thresh)
                ap_perm = _apical_up_read(bridge, R, held_pos_by_asm, perm_by_asm, up_thresh)
                ap_nocue = _apical_up_read(bridge, R, held_pos_by_asm, [None] * len(R.assemblies), up_thresh)
                # RECURRENCE-ZERO teeth (on the formed matrix) -- completion must collapse
                _saved = R.C.data.copy(); R.C.data[R.rec_mask] = cp.float32(0.0)
                zr_soma = float(np.mean([R.eval_assembly(A)["held_cue"] for A in R.assemblies]))
                zr_ap = _apical_up_read(bridge, R, held_pos_by_asm, cue_by_asm, up_thresh)
                R.C.data[:] = _saved

                def _go(cue, perm, nocue):
                    return bool(cue >= 0.20 and cue >= 3 * (perm + 1e-6) and cue >= 3 * (nocue + 1e-6) and nocue <= 0.10)
                r = dict(seed=seed, arm="dap", density=density, btsp_w_max=float(wmax),
                         k_thresh=k_thresh, w_within=w1_within, cross_dw=cross_dw, nonmem_dw=nonmem_dw,
                         genuine_formation=genuine, up_thresh=up_thresh,
                         soma_held_cue=soma_cue, soma_held_perm=soma_perm, soma_held_nocue=soma_nocue,
                         soma_no_encoding=base_soma, soma_recurrence_zero=zr_soma, soma_GO=_go(soma_cue, soma_perm, soma_nocue),
                         apical_held_cue=ap_cue, apical_held_perm=ap_perm, apical_held_nocue=ap_nocue,
                         apical_no_encoding=base_apical, apical_recurrence_zero=zr_ap,
                         apical_GO=(_go(ap_cue, ap_perm, ap_nocue) and genuine and base_apical <= 0.10))
                rows.append(r)
                if verbose:
                    print(f"  [s{seed} d{density} wmax{wmax:.0f} kt{k_thresh:.0f}] SOMA cue={soma_cue:.3f} perm={soma_perm:.3f} "
                          f"nocue={soma_nocue:.3f}{' GO' if r['soma_GO'] else ''} | APICAL cue={ap_cue:.3f} "
                          f"perm={ap_perm:.3f} nocue={ap_nocue:.3f} noenc={base_apical:.3f} reczero={zr_ap:.3f}"
                          f"{' GO' if r['apical_GO'] else ''} | w_in={w1_within:.0f} genuine={genuine}", flush=True)
                del bridge, R

    # ---- LEVER-LOAD-BEARING control: LINEAR (coincidence OFF) at the best dAP working point -> must STILL fail -------
    best = None
    dap_go_rows = [r for r in rows if r.get("apical_GO")]
    pick = max(rows, key=lambda r: r["apical_held_cue"]) if rows else None
    if pick is not None:
        density, wmax = pick["density"], pick["btsp_w_max"]
        form_build_kwargs["ca3_density"] = density
        lin = _build_dap_readout(seed, n_ca3=n_ca3, ca3_density=density, ca3_fb_inhib=ca3_fb_inhib,
                                 k_thresh=pick["k_thresh"], plateau_strength=plateau_strength, apical_R=apical_R,
                                 self_regen=self_regen, v_hold=v_hold, apical_kir_g=apical_kir_g, apical_gc=apical_gc,
                                 apical_gc_read=apical_gc_read, coincidence=False)   # LINEAR point neuron
        Rl = make_readout(lin, seed, **read_kwargs)
        form_btsp_multi(seed, form_build_kwargs, Rl, btsp_w_max=wmax, btsp_lr=btsp_lr, encode_drive=encode_drive,
                        encode_plateau_pA=encode_plateau_pA, train_events=train_events, drive_steps=drive_steps,
                        reset_steps=reset_steps, plateau=True, assemblies_ext=assemblies)
        lin_cue = float(np.mean([Rl.eval_assembly(A)["held_cue"] for A in Rl.assemblies]))
        lin_perm = float(np.mean([Rl.eval_assembly(A)["held_perm"] for A in Rl.assemblies]))
        t["linear_control_held_cue"] = lin_cue; t["linear_control_held_perm"] = lin_perm
        t["linear_control_fails"] = bool(not (lin_cue >= 0.20 and lin_cue >= 3 * (lin_perm + 1e-6)))
        if verbose:
            print(f"  [s{seed} LINEAR-CONTROL d{density} wmax{wmax:.0f}] cue={lin_cue:.3f} perm={lin_perm:.3f} "
                  f"-> {'FAILS (lever load-bearing)' if t['linear_control_fails'] else 'COMPLETES (lever NOT load-bearing!)'}",
                  flush=True)
        del lin, Rl
        best = {k: pick.get(k) for k in ("density", "btsp_w_max", "k_thresh", "apical_held_cue", "apical_held_perm",
                                         "apical_held_nocue", "apical_no_encoding", "apical_recurrence_zero",
                                         "soma_held_cue", "soma_held_perm", "soma_held_nocue", "w_within",
                                         "genuine_formation", "apical_GO", "soma_GO")}
    t["rows"] = rows; t["best"] = best
    # per-seed GO: an apical (or soma) working point that completes cue-specifically WITH lever load-bearing
    lb = t.get("linear_control_fails", True)
    t["seed_go_apical"] = bool(dap_go_rows and lb)
    t["seed_go_soma"] = bool(any(r.get("soma_GO") and r.get("genuine_formation") for r in rows) and lb)
    return t


def build_summary(per, seeds, densities, wmax_grid, kthresh_grid, elapsed, err=None):
    from tools.verdict import Verdict
    valid = [p for p in per if not p.get("error")]
    n = len(valid)
    n_ap = sum(1 for p in valid if p.get("seed_go_apical"))
    n_soma = sum(1 for p in valid if p.get("seed_go_soma"))
    n_emergent = sum(1 for p in valid if p.get("anticheat1_emergent_membership"))
    n_lever = sum(1 for p in valid if p.get("linear_control_fails"))
    all_rows = [r for p in valid for r in p.get("rows", [])]
    all_genuine = bool(all_rows) and all(r.get("genuine_formation") for r in all_rows if r.get("btsp_w_max"))
    intact = float(np.mean([sum(p["assembly_sizes"]) for p in valid])) if valid else 0.0
    _les = [sum(p.get("lesion_sizes") or [0]) for p in valid if p.get("lesion_sizes") is not None]
    lesion = float(np.mean(_les)) if _les else None
    ap_cue = float(np.mean([p["best"]["apical_held_cue"] for p in valid if p.get("best")])) if any(p.get("best") for p in valid) else None

    v = Verdict("gap5 lever-B: per-cell dendritic dAP readout completes the EMERGENT small BTSP assembly")
    v.require("emergent membership anti-cheat holds (all valid seeds)", (n_emergent == n and n > 0), expect=True,
              note="membership DG-selected (mossy-LESION collapses), not hand-set")
    v.require("BTSP formation genuine (all rows) -> a completion failure is a READOUT seam, not dead formation",
              all_genuine, expect=True)
    v.require("LEVER LOAD-BEARING: LINEAR (coincidence-off) control still FAILS (all valid seeds)",
              (n_lever == n and n > 0), expect=True, note="the dendritic dAP readout is what completes, not the weights")
    if lesion is not None:
        v.control("mossy-LESION collapses the selected membership (membership is DG-derived)",
                  treatment=intact, control=lesion, min_separation=1.0)
    v.disabled("plasticity at recall (hebbian/stdp/btsp/bdsp)", why="the frozen attractor is the read variable")
    v.disabled("dendritic deep-CREDIT / BDSP learning (2026-05-17-dendritic-credit-assignment-NEGATIVE)",
               why="this is a READOUT-bistability lever, not the tested-NEGATIVE hidden-credit rule; BDSP learning OFF")
    v.disabled("OU membrane noise", why="isolate the DETERMINISTIC per-cell bistability; OU-on can only add self-ignition")
    go = bool(n_ap >= max(1, int(np.ceil(5 / 6 * len(seeds)))) and n_emergent == n and n_lever == n and n > 0)
    decided = v.decide(go=go)
    status = decided["status"]
    seam = (status == "NO-GO")
    verdict = (f"{'LEVER-B-GO' if (go and status == 'GO') else ('READOUT-SEAM' if seam else status)} "
               f"apical-read GO {n_ap}/{n} | soma-read GO {n_soma}/{n} | emergent {n_emergent}/{n} | "
               f"lever-load-bearing {n_lever}/{n} | best apical_cue~{ap_cue}")
    if err is not None:
        verdict = f"ERROR -- {err}"; go = False
    return {"probe": "gap5_dendritic_dap_readout_completion", "lever": "B", "GO": go, "status": status,
            "verdict": verdict, "seeds": seeds, "n_go_apical": n_ap, "n_go_soma": n_soma,
            "n_emergent_membership": n_emergent, "n_lever_load_bearing": n_lever, "densities": densities,
            "wmax": wmax_grid, "kthresh": kthresh_grid, "elapsed_seconds": elapsed, "preconditions": decided["preconditions"],
            "disabled_processes": decided["disabled_processes"], "undefined_reasons": decided["undefined_reasons"],
            "per_seed": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-patterns", type=int, default=3)
    ap.add_argument("--densities", type=float, nargs="+", default=[0.5])
    ap.add_argument("--wmax", type=float, nargs="+", default=[100.0],
                    help="BTSP saturation ceiling -- SMALL-weight dAP regime (the plateau diverges at the reverberatory ~9000 scale)")
    ap.add_argument("--kthresh", type=float, nargs="+", default=[15.0, 30.0],
                    help="ABSOLUTE per-step within-assembly weighted-drive trigger (weight units); calibrated between "
                         "the baseline synapse (1.5) and a within-assembly synapse (~wmax) so cue volleys cross, perm don't")
    ap.add_argument("--plateau-strength", type=float, default=30.0)
    ap.add_argument("--apical-R", type=float, default=0.15,
                    help="apical input resistance (SMALL: the plateau ODE diverges under forward-Euler at aR~50 with these weights)")
    ap.add_argument("--self-regen", type=float, default=2.0)
    ap.add_argument("--v-hold", type=float, default=-35.0)
    ap.add_argument("--apical-kir-g", type=float, default=1.0)
    ap.add_argument("--apical-gc", type=float, default=0.3, help="soma->apical back-coupling (WEAK: keep soma out of the latch)")
    ap.add_argument("--apical-gc-read", type=float, default=0.3, help="apical->soma read coupling (WEAK for the decoupled read)")
    ap.add_argument("--up-thresh", type=float, default=-20.0, help="cp_v_apical UP-state threshold (mV) for the apical read")
    ap.add_argument("--ca3-fb-inhib", type=float, default=60.0)
    ap.add_argument("--btsp-lr", type=float, default=0.05)
    ap.add_argument("--encode-drive", type=float, default=700.0)
    ap.add_argument("--encode-plateau", type=float, default=250.0)
    ap.add_argument("--train-events", type=int, default=40)
    ap.add_argument("--drive-steps", type=int, default=48)
    ap.add_argument("--reset-steps", type=int, default=15)
    ap.add_argument("--assembly-frac", type=float, default=0.18)
    ap.add_argument("--cue-frac", type=float, default=0.5)
    ap.add_argument("--drive-pa", type=float, default=300.0)
    ap.add_argument("--warm-steps", type=int, default=100)
    ap.add_argument("--read-steps", type=int, default=100)
    ap.add_argument("--silence-steps", type=int, default=50)
    ap.add_argument("--no-lesion", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time(); err = None; per = []
    print(f"[gap5-dapB] emergent membership -> BTSP formation -> DENDRITIC dAP READOUT completion | seeds={a.seeds} "
          f"densities={a.densities} wmax={a.wmax} kthresh={a.kthresh} plateau_strength={a.plateau_strength} "
          f"apical_R={a.apical_R} self_regen={a.self_regen} kir={a.apical_kir_g} gc={a.apical_gc} "
          f"gc_read={a.apical_gc_read} up_thresh={a.up_thresh}", flush=True)
    try:
        for s in a.seeds:
            r = run_one_seed(s, densities=a.densities, wmax_grid=a.wmax, kthresh_grid=a.kthresh,
                             plateau_strength=a.plateau_strength, apical_R=a.apical_R, self_regen=a.self_regen,
                             v_hold=a.v_hold, apical_kir_g=a.apical_kir_g, apical_gc=a.apical_gc,
                             apical_gc_read=a.apical_gc_read, up_thresh=a.up_thresh, ca3_fb_inhib=a.ca3_fb_inhib,
                             btsp_lr=a.btsp_lr, encode_drive=a.encode_drive, encode_plateau_pA=a.encode_plateau,
                             train_events=a.train_events, drive_steps=a.drive_steps, reset_steps=a.reset_steps,
                             assembly_frac=a.assembly_frac, cue_frac=a.cue_frac, drive_pA=a.drive_pa,
                             warm_steps=a.warm_steps, read_steps=a.read_steps, silence_steps=a.silence_steps,
                             n_patterns=a.n_patterns, check_lesion=(not a.no_lesion), verbose=True)
            per.append(r)
            if r.get("error"):
                print(f"  [seed {s}] ERROR {r['error']}", flush=True); continue
            b = r.get("best") or {}
            print(f"  [seed {s}] sizes {r['assembly_sizes']} emergent-OK={r['anticheat1_emergent_membership']} "
                  f"lever-load-bearing={r.get('linear_control_fails')} || BEST d{b.get('density')} wmax{b.get('btsp_w_max')} "
                  f"kt{b.get('k_thresh')} APICAL cue={b.get('apical_held_cue')} perm={b.get('apical_held_perm')} "
                  f"nocue={b.get('apical_held_nocue')} | GO apical={r.get('seed_go_apical')} soma={r.get('seed_go_soma')} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()
    summary = build_summary(per, a.seeds, a.densities, a.wmax, a.kthresh, round(time.time() - t0, 1),
                            err=(err if (err is not None or not [p for p in per if not p.get("error")]) else None))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100 + f"\n[gap5-dapB] VERDICT: {summary['verdict']}\n[gap5-dapB] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if summary["GO"] else 1


if __name__ == "__main__":
    sys.exit(main())
