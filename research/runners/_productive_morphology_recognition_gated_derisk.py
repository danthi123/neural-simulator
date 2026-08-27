"""RECOGNITION-GATED dual-route past tense ON A SPIKING SUBSTRATE (roadmap Stage-2 [CPU] Language de-risk) --
the fix for the SEED-FRAGILE rule generalization the two-pool separate-route build left open.

WHAT THE RECORD ALREADY ESTABLISHED (do not re-derive):
  * SINGLE SHARED POOL (`_productive_morphology_construction_derisk.py`): declarative route works 6/6 (blocking
    0.857, lesion->over-reg 0.952, permuted-collapse 0.024) but the RULE does NOT generalize (reg_acc 0.188).
    Op-sweep proved it ARCHITECTURAL: rule and store COMPETE in one WTA.
  * TWO ISOLATED POOLS + di-synaptic blocking (`_productive_morphology_two_pool_derisk.py`): structural
    separation lifts reg_acc off the floor but it is SEED-FRAGILE (0.25-1.0; 6-seed 1/6 both-gates,
    `two_pool_disynaptic_6seed.json`).

THE DIAGNOSIS (2026-08-26, `diag` on seed 43 -- read `research/biology/dual-route-past-tense-recognition-gated-
blocking.md`): the most-entrenched irregular attractors (went/ran) leave a STEM-INDEPENDENT ~0.20 cosine FLOOR in
the readout -- present even for held-out regulars that were NEVER co-encoded with any whole-form. With the
di-synaptic blocking interneuron wired as a LINEAR relay (inh_drive=3.0), that weak floor drives the interneuron
enough to SUPPRESS THE DEFAULT AFFIX for whichever novel stem overlaps it most (kick+PAST: affix 0.24 -> 0.16,
lost to went 0.215). The blocking PATHWAY is mis-firing, not the retrieval.

THE MECHANISM THIS RUNNER ADDS (rooted in the companion process): a RECOGNITION THRESHOLD on the blocking
interneuron. A real GABAergic interneuron has a SPIKE THRESHOLD -- it fires (and blocks the default) only on
GENUINE, supra-threshold whole-form retrieval (Pinker-Ullman: the exception blocks the rule only when the stored
form is actually RECOGNIZED; the rule is the elsewhere/default otherwise). We raise the inh_block neurons' own
firing threshold (`raise_interneuron_threshold`, additive on cp_neuron_firing_thresholds, runner-side, NO sim/
edit) so the ~0.20 spurious floor is SUB-threshold (interneuron silent -> default -ed proceeds -> rule
generalizes robustly) while an entrenched irregular's strong retrieval is SUPRA-threshold (interneuron fires ->
affix blocked -> "went"). The threshold is the recognition/familiarity gate the linear relay replaced with a
constant.

GO gate (both SIMULTANEOUSLY, 6-seed, >=5/6): novel/held-out reg_acc >= 0.90 (the rule GENERALIZES) AND irregular
blocking irr_acc >= 0.85, on the SAME brain. Anti-cheats wired into the printed VERDICT:
  (1) UNSEEN pseudo-stems inflect by RULE (reg_acc over wug/blick/dax/gorp + held-out regulars);
  (2) LESION the LEX store -> OVER-REGULARIZATION (irregulars -> "-ed") and the gap over the intact -ed rate;
  (3) PERMUTED stem->whole-form binding COLLAPSES irregular retrieval to chance (learned binding load-bearing);
  (4) RECOGNITION-GATE is LOAD-BEARING: the interneuron pool FIRES for irregular cues and is (near-)SILENT for
      novel-stem cues -- the gate that makes the rule the default. Reported as inh_fire_irr vs inh_fire_novel.
  (5) cfg.seed substrate-hash (the 2026-07-17 unseeded-substrate trap).

Kandel 6e Ch 55 p.1373 ("go becomes went rather than goed"); Pinker & Ullman 2002 words-and-rules; Marcus 1992.
numpy, POOL-PORTABLE (no bridge checkpoint), seeded by cfg.seed. Reuse-by-import; NO `sim/` edit, NO edit to any
shared runner.

RESULT (2026-08-26, 1-seed numpy smoke seed 43 + threshold/inh-drive sweeps -- this lever is REFUTED):
  The recognition-threshold-on-the-interneuron lever DOES NOT WORK, and its OWN anti-cheat (4) catches it. The
  blocking interneuron is NON-SELECTIVE: it fires ~equally for irregular vs novel cues (16.3 vs 15.7) at EVERY
  inh_drive from 0.3 to 6.0 (firing ratio irr/novel ~1.0 throughout). Reason: the interneuron POOLS over all 7
  whole-forms, and the stem-INDEPENDENT went/ran tonic FLOOR keeps "some whole-form active" true for EVERY cue,
  so there is no genuine-vs-spurious rate difference to threshold on. Raising the threshold (thr+8mV) only
  BREAKS blocking (irr 0.857 -> 0.00) without robustly fixing the rule (reg 0.88). Smoke VERDICT: NO-GO
  (reg 0.88, irr 0.00, gate NOT selective). This runner is kept as the INSTRUMENT that refuted the lever and
  measures gate-selectivity. The residual is unambiguous and UPSTREAM: the tonic entrenched-attractor FLOOR in
  the LEX readout. The next mechanism (not this runner) is SOURCE-SIDE floor removal -- spike-frequency
  ADAPTATION to quench persistent attractors (Kandel: AHP/M-current), so entrenched whole-forms fire only
  transiently on their own cue, not tonically for every cue.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np

from sim.backend import to_host

from research.runners._productive_morphology_construction_derisk import (
    IRREGULARS, REG_TRAIN, REG_HELDOUT, NOVEL_STEMS, PAST, AFFIX,
    _make_items, co_activate, lesion_neurons, _argmax_form,
)
from research.runners._productive_morphology_two_pool_derisk import (
    build_two_pool, _assign_patterns, complete_two_pool, wire_disynaptic_inhibition,
    verify_seeded, _drive,
)


def raise_interneuron_threshold(bridge, inh_globals, delta_mV):
    """RECOGNITION THRESHOLD (the companion process): additively raise the SPIKE THRESHOLD of the blocking
    interneuron pool so only supra-threshold (genuine) whole-form retrieval fires it. In place on the live
    cp_neuron_firing_thresholds -- runner-side, additive, guarded; NO sim/ edit. Returns n neurons retuned."""
    th = getattr(bridge, "cp_neuron_firing_thresholds", None)
    if th is None or float(delta_mV) == 0.0:
        return 0
    idx = np.asarray(inh_globals, dtype=np.int64)
    host = np.asarray(to_host(th)).astype(float)
    host[idx] += float(delta_mV)
    # write back through whatever backend the tensor lives on
    try:
        import cupy as cp  # noqa
        if hasattr(th, "device"):
            th[:] = cp.asarray(host)
            return int(idx.size)
    except Exception:
        pass
    th[:] = host
    return int(idx.size)


def _inh_firing(bridge, pg, cue_idxs, inh_base, window=40, pA=1100.0):
    """Total interneuron (inh_block) spikes accumulated while the cue drives the network -> the recognition-gate
    read-out (anti-cheat 4). Read-only."""
    _drive(bridge, [pg[i] for i in cue_idxs], pA)
    tot = 0.0
    for _ in range(window):
        bridge._run_one_simulation_step()
        fs = np.asarray(to_host(bridge.cp_firing_states)).astype(float)
        tot += float(fs[inh_base].sum())
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(15):
        bridge._run_one_simulation_step()
    return tot / max(1, len(inh_base))


def run(seed, n_lex=2000, n_proc=800, pattern_size=90, cyc_rule=40, cyc_irr=48,
        inhib_strength=6.0, inh_drive=6.0, recog_threshold_mV=8.0, verbose=True):
    items, item2idx = _make_items()
    idx2name = {i: w for w, i in item2idx.items()}
    competitors = [item2idx[AFFIX]] + [item2idx[wf] for wf in IRREGULARS.values()]

    def build_brain(permute_binding=False):
        b = build_two_pool(seed, n_lex=n_lex, n_proc=n_proc, di_synaptic=True, n_inh_block=150)
        pg = _assign_patterns(b, seed, n_lex, n_proc, pattern_size, item2idx)
        inh_glob = np.asarray(b.region_manager.indices("inh_block"))
        # ROUTE 1 -- PROCEDURAL (PROC pool): PAST -> AFFIX, dedicated, stem-independent default.
        co_activate(b, pg, [item2idx[PAST], item2idx[AFFIX]], cycles=cyc_rule)
        # ROUTE 2 -- DECLARATIVE (LEX pool): stem -> whole-form per irregular, entrenched.
        wf_names = list(IRREGULARS.values())
        if permute_binding:                                  # ANTI-CHEAT (3): derange stem->whole-form binding
            rng = np.random.RandomState(seed * 101 + 7)
            perm = rng.permutation(len(wf_names))
            while np.any(perm == np.arange(len(wf_names))):
                perm = rng.permutation(len(wf_names))
            wf_for_stem = {stem: wf_names[perm[i]] for i, stem in enumerate(IRREGULARS.keys())}
        else:
            wf_for_stem = dict(IRREGULARS)
        for stem, wf in wf_for_stem.items():
            co_activate(b, pg, [item2idx[stem], item2idx[wf]], cycles=cyc_irr)
        # RECOGNITION THRESHOLD (AFTER training so homeostasis cannot erode it, and BEFORE the read-outs): raise
        # the interneuron spike threshold so the ~0.20 spurious floor is sub-threshold and only genuine retrieval
        # fires the interneuron. This is the companion process the linear relay replaced with a constant.
        raise_interneuron_threshold(b, inh_glob, recog_threshold_mV)
        # BLOCKING -- di-synaptic whole-form(exc) -> interneuron(inh) -> affix, wired AFTER training.
        affix_glob = np.asarray(pg[item2idx[AFFIX]])
        wf_glob = np.concatenate([np.asarray(pg[item2idx[wf]]) for wf in IRREGULARS.values()])
        n_drive, n_inh = wire_disynaptic_inhibition(b, wf_glob, inh_glob, affix_glob, inh_drive, inhib_strength)
        return b, pg, inh_glob, n_inh

    # ============================ MAIN BRAIN ============================
    b, pg, inh_glob, n_inh = build_brain(permute_binding=False)

    # (1) REGULAR PRODUCTIVITY -- held-out + novel stems must take "-ed" by RULE
    reg_probe = REG_HELDOUT + NOVEL_STEMS
    reg_ok = 0
    reg_detail = {}
    for stem in reg_probe:
        cue = [item2idx[stem], item2idx[PAST]]
        form, _ = _argmax_form(complete_two_pool(b, pg, cue, competitors), idx2name)
        ok = (form == AFFIX)
        reg_ok += int(ok)
        reg_detail[stem] = form
        if verbose:
            print(f"  REGULAR {stem:6s}+PAST -> '{stem}{form}' {'[rule -ed OK]' if ok else '[got '+form+']'}", flush=True)
    reg_acc = reg_ok / len(reg_probe)

    # (2) IRREGULAR BLOCKING -- the recognized stored whole-form fires the interneuron, blocks the affix, WINS
    irr_ok = 0
    irr_edrate_pre = 0
    irr_detail = {}
    for stem, wf in IRREGULARS.items():
        cue = [item2idx[stem], item2idx[PAST]]
        form, _ = _argmax_form(complete_two_pool(b, pg, cue, competitors), idx2name)
        ok = (form == wf)
        irr_ok += int(ok)
        irr_edrate_pre += int(form == AFFIX)
        irr_detail[stem] = form
        if verbose:
            print(f"  IRREG   {stem:6s}+PAST -> '{form}' (target '{wf}') "
                  f"{'[stored, BLOCKS -ed]' if ok else '[got '+form+']'}", flush=True)
    irr_acc = irr_ok / len(IRREGULARS)
    irr_edrate_pre /= len(IRREGULARS)

    # ANTI-CHEAT (4): RECOGNITION-GATE load-bearing -- interneuron fires for irregular cues, silent for novel.
    inh_fire_irr = float(np.mean([_inh_firing(b, pg, [item2idx[s], item2idx[PAST]], inh_glob)
                                  for s in IRREGULARS.keys()]))
    inh_fire_novel = float(np.mean([_inh_firing(b, pg, [item2idx[s], item2idx[PAST]], inh_glob)
                                    for s in NOVEL_STEMS]))
    gate_selective = (inh_fire_irr >= 2.0 * max(1e-6, inh_fire_novel))
    if verbose:
        print(f"  RECOG-GATE: interneuron fire irr={inh_fire_irr:.3f} novel={inh_fire_novel:.3f} "
              f"{'[SELECTIVE]' if gate_selective else '[NOT selective]'}", flush=True)

    # (3 anti-cheat) LESION the LEX store -> OVER-REGULARIZATION
    wf_globals = np.concatenate([np.asarray(pg[item2idx[wf]]) for wf in IRREGULARS.values()])
    lesion_neurons(b, wf_globals)
    overreg = 0
    overreg_other = 0
    les_detail = {}
    for stem, wf in IRREGULARS.items():
        cue = [item2idx[stem], item2idx[PAST]]
        form, _ = _argmax_form(complete_two_pool(b, pg, cue, competitors), idx2name)
        les_detail[stem] = form
        if form == AFFIX:
            overreg += 1
        elif form != wf:
            overreg_other += 1
        if verbose:
            print(f"  LESION  {stem:6s}+PAST -> '{stem}{form if form==AFFIX else '/'+form}' "
                  f"{'[OVER-REGULARIZED]' if form==AFFIX else ''}", flush=True)
    overreg_rate = overreg / len(IRREGULARS)
    # ATTRIBUTION: the irregular blocking (when it works) is OWNED by the lex whole-form store -- lesioning it
    # flips irregulars to over-regularization (treatment) vs the intact -ed rate (control); the gap is the
    # blocking. (Here both arms read ~1.0 because the raised-threshold lever already broke blocking pre-lesion.)
    from tools.lab import attributable_to
    attributable_to("irr_blocking_owned_by_lex_store", treatment_value=overreg_rate, control_value=irr_edrate_pre)

    # ============================ PERMUTED-BINDING BRAIN (anti-cheat 3) ============================
    bp, pgp, _, _ = build_brain(permute_binding=True)
    perm_ok = 0
    for stem, wf in IRREGULARS.items():
        cue = [item2idx[stem], item2idx[PAST]]
        form, _ = _argmax_form(complete_two_pool(bp, pgp, cue, competitors), idx2name)
        perm_ok += int(form == wf)
    perm_acc = perm_ok / len(IRREGULARS)

    both_gates = (reg_acc >= 0.90 and irr_acc >= 0.85)
    go = (both_gates and overreg_rate >= 0.85
          and overreg_rate - irr_edrate_pre >= 0.5 and perm_acc <= 0.30 and gate_selective)
    result = {
        "seed": int(seed), "n_lex": n_lex, "n_proc": n_proc, "pattern_size": pattern_size,
        "cyc_rule": cyc_rule, "cyc_irr": cyc_irr, "inhib_strength": inhib_strength,
        "inh_drive": inh_drive, "recog_threshold_mV": recog_threshold_mV, "n_inhib_edges": int(n_inh),
        "reg_acc": reg_acc, "irr_acc": irr_acc, "both_gates": bool(both_gates),
        "irr_edrate_pre_lesion": irr_edrate_pre,
        "overreg_rate_lesion": overreg_rate, "lesion_other_error_rate": overreg_other / len(IRREGULARS),
        "permuted_binding_irr_acc": perm_acc,
        "inh_fire_irr": inh_fire_irr, "inh_fire_novel": inh_fire_novel, "gate_selective": bool(gate_selective),
        "reg_detail": reg_detail, "irr_detail": irr_detail, "lesion_detail": les_detail,
        "GO": bool(go),
    }
    if verbose:
        print(f"\n  seed {seed} [recog-gate thr={recog_threshold_mV}mV]: reg(rule) {reg_acc:.2f} & "
              f"irr(blocking) {irr_acc:.2f} {'[BOTH]' if both_gates else '[NOT both]'} | "
              f"lesion->over-reg {overreg_rate:.2f} (pre {irr_edrate_pre:.2f}) | permuted {perm_acc:.2f} | "
              f"gate irr/novel {inh_fire_irr:.2f}/{inh_fire_novel:.2f} | {'GO' if go else 'NO-GO'}", flush=True)
    return result


def summarize(base_seed, n_seeds=6, n_lex=2000, n_proc=800, pattern_size=90, cyc_rule=40, cyc_irr=48,
              inhib_strength=6.0, inh_drive=6.0, recog_threshold_mV=8.0, verbose=True):
    seeds = [base_seed + i for i in range(n_seeds)]
    seed_check = verify_seeded(base_seed, n_lex, n_proc)
    if verbose:
        print(f"[seed anti-cheat] cfg.seed controls substrate: {seed_check['seeds_substrate']}", flush=True)
    results = [run(s, n_lex=n_lex, n_proc=n_proc, pattern_size=pattern_size, cyc_rule=cyc_rule,
                   cyc_irr=cyc_irr, inhib_strength=inhib_strength, inh_drive=inh_drive,
                   recog_threshold_mV=recog_threshold_mV, verbose=verbose) for s in seeds]
    n_reg = sum(1 for r in results if r["reg_acc"] >= 0.90)
    n_irr = sum(1 for r in results if r["irr_acc"] >= 0.85)
    n_both = sum(1 for r in results if r["both_gates"])
    n_go = sum(1 for r in results if r["GO"])
    go = (n_go >= 5) and seed_check["seeds_substrate"]

    # PRECONDITIONS the NO-GO must travel with (tools.verdict.Verdict -> gates/verdict_preconditions). The
    # refutation is gate NON-SELECTIVITY (anti-cheat 4), so what must be VERIFIED is that the gate read-out has
    # resolution: the interneuron must actually FIRE (well above silence), otherwise "not selective" would be a
    # dead read-out, not a measured equality. It fires ~16-17 spikes for BOTH cue types -- live and unsaturated
    # (window=40, so ceiling ~40) -- so gate_selective=False is a real measurement. NOTE the pc_inh_drive/thr=0
    # baseline is NOT used as the discriminating control: blocking (irr_acc) is itself SEED-FRAGILE at this op
    # point (probe: seed 42 threshold=0 irr_acc=0.0, seed 43 ~0.857), so irr_acc=0 would be a confounded floor;
    # the clean, unconfounded control is the firing read-out that the refutation actually rests on.
    from tools.verdict import Verdict
    mean_inh_fire_irr = float(sum(r["inh_fire_irr"] for r in results) / max(1, len(results)))
    mean_inh_fire_novel = float(sum(r["inh_fire_novel"] for r in results) / max(1, len(results)))
    v = Verdict("recognition-gated dual-route: rule generalizes (reg>=0.90) AND blocking (irr>=0.85), 6-seed")
    v.require("substrate seeded by cfg.seed (anti-cheat 5)", seed_check["seeds_substrate"], expect=True)
    v.floor("gate read-out is LIVE (interneuron fires >> silence, so gate_selective is a measured equality "
            "not a dead read-out)", mean_inh_fire_irr, floor=1.0)
    decided = v.decide(go=go, verbose=verbose)
    if go:
        verdict = f"RECOGNITION-GATED DUAL-ROUTE GO -- rule generalizes AND blocking, {n_go}/{n_seeds} full-GO"
    elif not seed_check["seeds_substrate"]:
        verdict = "NEGATIVE -- substrate NOT seeded by cfg.seed (anti-cheat 5 failed; confounded)"
    elif n_both < 5 and n_reg < 5 and n_irr >= 5:
        verdict = f"NEGATIVE -- the RULE still fails to generalize (reg_acc>=0.90 on only {n_reg}/{n_seeds})"
    elif n_both < 5 and n_irr < 5 and n_reg >= 5:
        verdict = f"NEGATIVE -- BLOCKING fails (irr_acc>=0.85 on only {n_irr}/{n_seeds}) -- threshold too high"
    elif n_both < 5:
        verdict = (f"NEGATIVE -- no op-point does BOTH (both {n_both}/{n_seeds}; reg {n_reg}, irr {n_irr}) "
                   f"-- sweep --recog-threshold-mV / --inh-drive")
    else:
        verdict = f"NEGATIVE -- anti-cheats fail on >1 seed (both {n_both}/{n_seeds} but full-GO {n_go}/{n_seeds})"
    summary = {
        "probe": "productive_morphology_recognition_gated",
        "config": {"n_seeds": n_seeds, "base_seed": base_seed, "n_lex": n_lex, "n_proc": n_proc,
                   "pattern_size": pattern_size, "cyc_rule": cyc_rule, "cyc_irr": cyc_irr,
                   "inhib_strength": inhib_strength, "inh_drive": inh_drive,
                   "recog_threshold_mV": recog_threshold_mV},
        "per_seed": [
            {"seed": r["seed"], "reg_acc": r["reg_acc"], "irr_acc": r["irr_acc"],
             "both_gates": r["both_gates"], "overreg_rate_lesion": r["overreg_rate_lesion"],
             "permuted_binding_irr_acc": r["permuted_binding_irr_acc"],
             "inh_fire_irr": r["inh_fire_irr"], "inh_fire_novel": r["inh_fire_novel"],
             "gate_selective": r["gate_selective"], "GO": r["GO"]}
            for r in results
        ],
        "n_reg_ge_0.90": n_reg, "n_irr_ge_0.85": n_irr, "n_both_gates": n_both, "n_full_go": n_go,
        "seed_check": seed_check, "GO": bool(go), "verdict": verdict,
        "mean_inh_fire_irr": mean_inh_fire_irr, "mean_inh_fire_novel": mean_inh_fire_novel,
        "verdict_status": decided["status"], "preconditions": decided["preconditions"],
    }
    if verbose:
        print(f"\n=== recognition-gated morphology summary ({n_seeds} seeds) ===", flush=True)
        print(f"  reg>=0.90: {n_reg}/{n_seeds} | irr>=0.85: {n_irr}/{n_seeds} | BOTH: {n_both}/{n_seeds} "
              f"| full-GO: {n_go}/{n_seeds}", flush=True)
        print(f"  gate read-out live: interneuron fire irr {mean_inh_fire_irr:.2f} / novel "
              f"{mean_inh_fire_novel:.2f} (mean) -- fires but NOT selectively", flush=True)
        print(f"  VERDICT: {verdict} [{decided['status']}]", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="base seed; the sweep uses seed..seed+n_seeds-1")
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--n-lex", type=int, default=2000)
    ap.add_argument("--n-proc", type=int, default=800)
    ap.add_argument("--pattern-size", type=int, default=90)
    ap.add_argument("--cyc-rule", type=int, default=40)
    ap.add_argument("--cyc-irr", type=int, default=48)
    ap.add_argument("--inhib-strength", type=float, default=6.0, help="interneuron->affix inhibitory output weight")
    ap.add_argument("--inh-drive", type=float, default=6.0, help="whole-form->interneuron excitatory drive weight")
    ap.add_argument("--recog-threshold-mV", type=float, default=8.0,
                    help="ADDITIVE raise of the interneuron spike threshold = the recognition gate (0 = linear "
                         "relay = the seed-fragile two-pool baseline)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print(f"[recognition-gated morphology] di-synaptic blocking + interneuron spike-threshold recognition gate "
          f"(thr+{a.recog_threshold_mV}mV) | base seed={a.seed} n_seeds={a.n_seeds}", flush=True)
    s = summarize(a.seed, n_seeds=a.n_seeds, n_lex=a.n_lex, n_proc=a.n_proc, pattern_size=a.pattern_size,
                  cyc_rule=a.cyc_rule, cyc_irr=a.cyc_irr, inhib_strength=a.inhib_strength,
                  inh_drive=a.inh_drive, recog_threshold_mV=a.recog_threshold_mV)
    print(f"\n  {'GO' if s['GO'] else 'NO-GO'} -- {s['verdict']}", flush=True)
    if a.out:
        d = os.path.dirname(a.out)
        if d:
            os.makedirs(d, exist_ok=True)
        json.dump(s, open(a.out, "w"), indent=1)
        print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
