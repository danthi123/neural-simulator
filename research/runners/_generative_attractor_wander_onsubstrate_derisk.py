"""Board #104 — ON-SUBSTRATE port of the generative attractor-wander de-risk (2026-08-20,
research/runners/_generative_attractor_wander_derisk.py) onto the REAL spiking CA3 the production episodic/D5
machinery uses, so idle-tick wander can GENERATE genuinely novel combinations on the actual substrate, not a
detached numpy stand-in (webapp/continuous_engine.py's `_ideation_blend_settle` today builds a FRESH random numpy
Hopfield net, unrelated to the organ's real stored engrams).

THE NUMPY DE-RISK'S CLAIM: a BLENDED cue of two/three stored sparse patterns, settled through a mean+std DYNAMIC
THRESHOLD (a closed-form stand-in for CA3 feedforward-inhibition / divisive normalization), lands on a STABLE fixed
point that is a genuine RECOMBINATION of the cued sources (balanced overlap with both, far from any single stored
item, far from any OTHER non-cued stored item) — novelty from the DYNAMICS, not the nodes.

THE ON-SUBSTRATE MECHANISM (reuse-by-import, NO sim/ edit, NO re-derive): this project's validated completion
mechanism is NOT a population-level recurrent Hopfield settle — that path is CLOSED-NEGATIVE on the point-neuron
soma (2026-08-10-ca3-point-neuron-attractor-completion-trilemma-NEGATIVE: magnitude and specificity trade off, no
seed-robust operating point). The mechanism that DOES work, and that the production D5/episodic organ actually runs
(research/runners/_episodic_dap_dialogue_memory.py -> `_gap5_dendritic_dap_readout_completion_derisk.py`), is the
per-cell TWO-COMPARTMENT DENDRITIC dAP BISTABLE LATCH: each CA3 cell's apical dendrite independently ignites and
HOLDS an UP state when its coincident within-assembly drive crosses a threshold (`fused_coincidence_plateau`,
`enable_coincidence_detection` + `enable_two_compartment_dap`), decoupling completion from a population-wide
recurrent loop gain. This runner drives that SAME mechanism with a BLENDED cue (a small, calibrated absolute
number of stored assembly A's cue cells + the same number of stored assembly B's -- see the P dict's
`blend_cells_each` comment for why an ABSOLUTE count, not a fraction, is the scale-invariant knob) and reads, for
EVERY stored assembly, what fraction of ITS OWN member cells
enter the dendritic UP state — the on-substrate analogue of "overlap(settled, stored_m)". The CA3 feedback
inhibition (`ca3_fb_inhib`, already wired into the SAME `_build_dap_readout` the production organ uses) plays the
role of the numpy mean+std dynamic threshold (a divisive-normalization companion process, not a fixed-count top-k),
so the "already-GO" dynamic-threshold framing carries over structurally even though the READ is per-cell-bistable
rather than population-iterative.

SCOPE (honestly declared): reuses `_build_dap_readout` / `make_readout` / `_form_one_assembly` / `form_btsp_multi`
VERBATIM (research/runners/_gap5_dendritic_dap_readout_completion_derisk.py,
research/runners/_gap5_btsp_forms_nmda_slow_reverberatory_derisk.py) — the IDENTICAL functions
research/runners/_episodic_dap_dialogue_memory.py (the production D5/episodic organ) composes. The ONE scope
reduction: PRE-ASSIGNED (random-permutation) assembly membership at n_ca3=400 rather than the production organ's
n_ca3=2000 EMERGENT DG-selected membership (n_ca3=2000 emergent selection is a mandatory-scale mechanism per the
gap#5 seam findings, and a single BTSP store there runs ~6 min on numpy — not foreground-feasible for a 6-seed x
3-assembly sweep on CPU). n_ca3=400 / assembly_frac=0.18 (72-cell assemblies) is itself a previously-validated
operating point for this exact BTSP-formation + dendritic-completion mechanism family (the slow-NMDA formation
runner's own reference scale). Everything downstream — BTSP one-shot formation, the dAP two-compartment plateau,
the apical UP-state read, the lesion/no-encoding/permuted-cue teeth — is the PRODUCTION mechanism, unmodified.

2026-08-27 PRODUCTION-SCALE PORT (board #104 rung 2, closes both blockers this finding's "Residual" named): `--emergent`
threads `_gap5_emergent_end_to_end_episodic_loop_derisk.emergent_assemblies` (the IDENTICAL DG-selected sparse-detonator
membership `_episodic_dap_dialogue_memory.EpisodicDapMemory` — the production D5/episodic organ — actually uses) through
`assemblies_ext` into `_build_dap_readout`/`make_readout`/`form_btsp_multi`, exactly as that production organ composes
them (n_ca3 lifts to whatever the emergent selection returns, i.e. 2000 at R1's default config; train_events defaults to
40, the ALREADY-VALIDATED GO_DEFAULTS value for this SAME mechanism at this SAME scale, not re-derived). This closes
BOTH declared blockers at once: (i) reduced n_ca3=400 -> production n_ca3=2000, (ii) PRE-ASSIGNED/direct-index cue
membership -> the emergent, BTSP-formed membership the production organ actually stores concepts into. The `assemblies_
ext` array is computed ONCE per seed and reused verbatim across every fresh per-condition bridge rebuild below (never
re-called) — emergent membership is measured non-deterministic ACROSS separate builds (FMA/summation-order drift, see
_episodic_dap_dialogue_memory.py's kthresh note), so re-deriving it per read would silently decouple the read from what
was actually BTSP-formed; reusing one fixed array is the SAME discipline `EpisodicDapMemory.__init__`/`.store()` already
use. `blend_cells_each` stays at the SAME calibrated absolute count (3) — the finding's own scale-invariance argument
(driven by ABSOLUTE cued-neighbour count, not assembly size) is the reason this is a re-run, not a re-tune. `build_
production_store`/`blend_settle_production` below factor the build+BTSP-form / blend-read steps out of `run_seed` into
standalone reusable functions — the SAME code path `webapp/continuous_engine.py`'s live wiring calls, so there is
exactly ONE implementation of the production-scale mechanism, not a duplicate for the offline verify vs. the live wire.

METRICS (mirrors the numpy de-risk's definitions, mapped onto per-assembly-membership UP-fraction instead of
binary-pattern overlap):
  novelty        = max_m UP-fraction(assembly m | blended cue)      -- low = not fully re-forming any ONE stored item
  blend_balance  = min(UP-fraction(A | blend), UP-fraction(B | blend))   -- both cued sources genuinely represented
  blend_vs_other = UP-fraction(C | blend), C = the third, NON-cued stored assembly
  persistence    = |UP-fraction(A | blend, cue-still-on) - UP-fraction(A | blend, cue-RELEASED + held)|  -- small =
                   the per-cell bistable latch actually HOLDS without ongoing drive (the on-substrate analogue of
                   the numpy "fixed point" check — arguably a STRONGER stability proof, since nothing sustains it).

ANTI-CHEATS:
  (A) SINGLE-cue positive control: cueing ONLY assembly A must recover A specifically (high UP-fraction(A), low on
      B and C) — the completion mechanism itself works, on THIS harness, at THIS scale.
  (B) UNTRAINED (lesion): the SAME blended cue read through the pre-formation baseline (unformed) recurrent weights
      must NOT complete anything (rules out "the dAP latch alone, without the learned BTSP potentiation, fakes it").
  (C) NOISE cue: an equal-sized cue of CA3 cells outside A/B's cue sets must NOT balance on A,B (rules out "any
      drive produces a balanced-looking read").
  (D) PERSISTENCE: the blended UP-state must survive cue release (bistable latch), not merely track ongoing drive.

Run:  SIM_BACKEND=numpy python -m research.runners._generative_attractor_wander_onsubstrate_derisk \
        --seeds 42 43 44 100 101 102 --json research/findings/raw/_generative_attractor_wander_onsubstrate/evidence.json
Run (production scale, emergent DG-selected n_ca3=2000, GPU):
      SIM_BACKEND=cupy python -m research.runners._generative_attractor_wander_onsubstrate_derisk --emergent \
        --seeds 42 43 44 100 101 102 \
        --json research/findings/raw/_generative_attractor_wander_onsubstrate/production_n_ca3_2000_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners._gap5_dendritic_dap_readout_completion_derisk import (  # noqa: E402
    _build_dap_readout, _reset_apical_latch)
from research.runners._gap5_btsp_forms_nmda_slow_reverberatory_derisk import (  # noqa: E402
    make_readout, form_btsp_multi)

# The production organ's calibrated dAP operating point (research/runners/_episodic_dap_dialogue_memory.py
# GO_DEFAULTS) — reused VERBATIM, not re-tuned, except n_ca3/assembly_frac (the declared scope reduction) and
# train_events (bumped from 40->60 to reach genuine within-assembly formation at the larger 72-cell scale; measured
# empirically below via the genuine_formation teeth, not assumed).
P = dict(
    n_ca3=400, ca3_density=0.5, assembly_frac=0.18, cue_frac=0.5,
    kthresh=8.0, plateau_strength=30.0, apical_R=0.15, self_regen=2.0, v_hold=-35.0,
    apical_kir_g=1.0, apical_gc=0.3, apical_gc_read=0.3, up_thresh=-20.0, ca3_fb_inhib=60.0,
    btsp_lr=0.05, wmax=100.0, encode_drive=700.0, encode_plateau_pA=250.0,
    train_events=60, drive_steps=48, reset_steps=15,
    drive_pA=300.0, warm_steps=100, read_steps=100, silence_steps=50, hold_steps=100,
    # THE CALIBRATED BLEND OPERATING POINT (empirically swept, isolated fresh-bridge reads --
    # research/findings/raw/_generative_attractor_wander_onsubstrate/frac_sweep_*.json): driving HALF of each cued
    # source's own cue-eligible cells saturates BOTH assemblies to ~0.97-1.0 simultaneously (a dual-recall, not a
    # novel blend -- the per-cell dAP latch has no population-wide competitive budget forcing a choice between
    # sources, unlike the numpy mean+std threshold it replaces). Sweeping the blend size as a FRACTION of assembly
    # size did NOT transfer across scale (11% of a 36-cell assembly ~= 4 cells landed graded; 11% of a 72-cell
    # assembly ~= 8 cells still saturated to ~0.99) -- the coincident drive a held-out cell receives depends on the
    # ABSOLUTE number of directly-connected cued neighbours (~n_cue * ca3_density), not on assembly size, so the
    # right invariant is an ABSOLUTE cell count. blend_cells_each=3 lands in a genuinely GRADED, balanced regime at
    # BOTH n_ca3=200/assembly=36 ([0.667,0.694,0.0]) and n_ca3=400/assembly=72 ([0.778,0.639,0.0]) for seed 42 --
    # the on-substrate analogue of the numpy blend's balanced-but-not-fully-either fixed point.
    blend_cells_each=3,
)


def _population_up(bridge, R, cue_indices, *, cp, to_host, drive_pA, warm_steps, read_steps, up_thresh,
                   hold_steps=0):
    """Drive `cue_indices` (CA3 GLOBAL ids) for warm+read steps (mirrors _apical_up_read's drive/read loop exactly),
    read the boolean UP array (cp_v_apical > up_thresh) over EVERY CA3 cell (not just one assembly's held set) —
    then, if hold_steps>0, RELEASE the cue (external current -> 0) and step hold_steps further with NO drive,
    re-reading UP — the bistable-PERSISTENCE check (does the latch hold once nothing sustains it?). Returns
    (up_while_driven, up_after_release_or_None)."""
    R.hard_silence()
    _reset_apical_latch(bridge)
    darr = None
    if cue_indices is not None and len(cue_indices) > 0:
        darr = cp.asarray(np.asarray(cue_indices, dtype=np.int64), dtype=cp.int64)
        bridge.cp_external_input_current[darr] = cp.float32(drive_pA)
    for _ in range(warm_steps + read_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    up_driven = None
    if getattr(bridge, "cp_v_apical", None) is not None:
        up_driven = to_host(bridge.cp_v_apical) > up_thresh
    if darr is not None:
        bridge.cp_external_input_current[darr] = 0.0
    up_release = None
    if hold_steps > 0:
        for _ in range(hold_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        if getattr(bridge, "cp_v_apical", None) is not None:
            up_release = to_host(bridge.cp_v_apical) > up_thresh
    return up_driven, up_release


def _frac(up_bool, member_ids):
    if up_bool is None or len(member_ids) == 0:
        return 0.0
    return float(np.mean([1.0 if bool(up_bool[int(g)]) else 0.0 for g in member_ids]))


def _cue_cells(assembly, seed, tag, cue_frac):
    """A deterministic CUE_FRAC subset of an assembly's own member cells (mirrors _held_cue_perm's split)."""
    se = np.asarray(assembly, dtype=np.int64).copy()
    r = np.random.default_rng(seed * 131 + tag)
    r.shuffle(se)
    n_cue = max(2, int(cue_frac * len(se)))
    return se[:n_cue]


def _build_readout(seed, p, n_mem, assemblies_ext=None):
    """The bare readout harness (bridge + R), NO formation -- cheap (~0.5s), used to get a FRESH transient-state
    bridge for every read condition below. assemblies_ext (ADDITIVE, default None => byte-identical): a fixed list
    of CA3 GLOBAL-index arrays (e.g. from `emergent_assemblies`) to use as membership INSTEAD of make_readout's
    internal seed-deterministic random permutation -- forwarded verbatim to `make_readout`."""
    bridge = _build_dap_readout(
        seed, n_ca3=p["n_ca3"], ca3_density=p["ca3_density"], ca3_fb_inhib=p["ca3_fb_inhib"],
        k_thresh=p["kthresh"], plateau_strength=p["plateau_strength"], apical_R=p["apical_R"],
        self_regen=p["self_regen"], v_hold=p["v_hold"], apical_kir_g=p["apical_kir_g"],
        apical_gc=p["apical_gc"], apical_gc_read=p["apical_gc_read"], coincidence=True)
    R = make_readout(bridge, seed, n_assembly=n_mem, assembly_frac=p["assembly_frac"], cue_frac=p["cue_frac"],
                     drive_pA=p["drive_pA"], warm_steps=p["warm_steps"], read_steps=p["read_steps"],
                     silence_steps=p["silence_steps"], assemblies_ext=assemblies_ext)
    return bridge, R


def _build_and_form(seed, n_mem, p, *, emergent=False):
    """ONE build + ONE BTSP formation pass (the expensive part) -> the formed + baseline weight arrays + the
    assembly membership actually formed. Factored out of `run_seed` so BOTH the offline anti-cheat verify
    (`run_seed`) and the live production store (`build_production_store`, reused by webapp/continuous_engine.py)
    share exactly ONE build+BTSP-form code path -- no duplicate mechanism.

    emergent=False (default, byte-identical to pre-2026-08-27): assemblies_ext stays None throughout -> every fresh
    bridge rebuild below reproduces the SAME pre-assigned membership via make_readout's own seed-deterministic
    internal permutation (the existing, already-6/6-GO invariant this instrument already relied on).
    emergent=True: calls `emergent_assemblies` ONCE (membership is measured non-deterministic ACROSS separate
    bridge builds -- FMA/summation-order drift, see _episodic_dap_dialogue_memory.py's kthresh note -- so it must
    NOT be re-called per read), overrides p['n_ca3'] to the emergent selection's own CA3 range, and threads that
    ONE fixed array through every downstream build via assemblies_ext -- identical to how the production
    EpisodicDapMemory organ composes it."""
    p = dict(p)
    assemblies_ext = None
    if emergent:
        from research.runners._gap5_emergent_end_to_end_episodic_loop_derisk import emergent_assemblies
        assemblies_ext, ca3_range = emergent_assemblies(seed, n_patterns=n_mem)
        p["n_ca3"] = int(ca3_range[2])
        if int(p.get("train_events", 0)) == int(P["train_events"]):
            # the reduced-scale (72-cell pre-assigned) operating point bumped train_events 40->60 (see the P dict's
            # comment); at the production emergent scale, reuse GO_DEFAULTS' train_events=40 -- the ALREADY-VALIDATED
            # value _episodic_dap_dialogue_memory.py's own 6-seed GO uses for this SAME mechanism at this SAME scale
            # -- rather than re-deriving. Only applied when the caller left train_events at its non-emergent default.
            p["train_events"] = 40

    bridge, R = _build_readout(seed, p, n_mem, assemblies_ext=assemblies_ext)
    assemblies = R.assemblies
    sizes = [int(len(a)) for a in assemblies]
    baseline_weights = R.C.data.copy()   # UNFORMED recurrent weights -- the anti-cheat (B) lesion target

    form_build_kwargs = dict(n_ca3=p["n_ca3"], ca3_density=p["ca3_density"], ca3_fb_inhib=p["ca3_fb_inhib"],
                             ca3_ff_inhib=None, nmda_tau=100.0, nmda_ratio=1.0, enable_ou=False,
                             element="nmda_slow")
    diag = form_btsp_multi(seed, form_build_kwargs, R, btsp_w_max=p["wmax"], btsp_lr=p["btsp_lr"],
                           encode_drive=p["encode_drive"], encode_plateau_pA=p["encode_plateau_pA"],
                           train_events=p["train_events"], drive_steps=p["drive_steps"],
                           reset_steps=p["reset_steps"], plateau=True, assemblies_ext=assemblies_ext)
    w_within, cross_dw, nonmem_dw = diag["w_within"], diag["cross_dw"], diag["nonmem_dw"]
    genuine = bool(w_within > 20.0 and abs(cross_dw) < 0.05 * max(w_within, 1.0)
                   and abs(nonmem_dw) < 0.05 * max(w_within, 1.0))
    formed_weights = R.C.data.copy()
    del bridge, R   # the ONLY bridge that ever runs a drive/read is a FRESH one, per condition, below
    return dict(p=p, assemblies_ext=assemblies_ext, assemblies=assemblies, sizes=sizes,
               baseline_weights=baseline_weights, formed_weights=formed_weights,
               diag=diag, genuine_formation=genuine)


def build_production_store(seed, n_mem, *, p=None):
    """PRODUCTION-SCALE store builder: emergent n_ca3=2000 DG-selected membership, BTSP-formed. Thin wrapper around
    `_build_and_form(..., emergent=True)` returning exactly the fields `blend_settle_production` needs. This is the
    ONE build+BTSP-form call `webapp/continuous_engine.py`'s live ideation wiring makes (once per session's concept
    list, cached -- see that module's `_SPIKING_IDEATE_STORE`), and the same call `run_seed(emergent=True)` makes
    for the offline 6-seed anti-cheat verify -- one mechanism, two callers."""
    built = _build_and_form(seed, n_mem, dict(P) if p is None else p, emergent=True)
    return dict(seed=int(seed), n_mem=int(n_mem), p=built["p"], assemblies_ext=built["assemblies_ext"],
               assemblies=built["assemblies"], sizes=built["sizes"],
               baseline_weights=built["baseline_weights"], formed_weights=built["formed_weights"],
               diag=built["diag"], genuine_formation=built["genuine_formation"])


def blend_settle_production(store, iA, iB, *, hold_steps=None):
    """Read the blended-cue dAP completion for basins iA,iB of an already-formed `store` (from
    `build_production_store`). Returns the SAME-SHAPED dict `_ideation_blend_settle` (webapp/continuous_engine.py's
    numpy stand-in) returns (novelty_max_overlap, blend_balance, blend_vs_other, fixed_point) so either source can
    feed the identical downstream ideation gate (IDEATE_NOVELTY_MAX etc.) unmodified. Returns None if the store's
    formation was not genuine, or iA/iB are invalid -- an honest 'no idea surfaced' rather than a fake read.
    FRESH bridge per read (the measured instrument-contamination fix this module's docstring documents)."""
    from sim.backend import get_backend, to_host
    p, n_mem = store["p"], store["n_mem"]
    seed, assemblies_ext = store["seed"], store["assemblies_ext"]
    if not store["genuine_formation"] or n_mem < 2 or not (0 <= iA < n_mem) or not (0 <= iB < n_mem) or iA == iB:
        return None
    cp, _ = get_backend()
    hold = p["hold_steps"] if hold_steps is None else hold_steps

    def _read_fresh(cue_ids, *, weights, hold=0):
        b, r = _build_readout(seed, p, n_mem, assemblies_ext=assemblies_ext)
        r.C.data[:] = weights
        up_driven, up_release = _population_up(b, r, cue_ids, cp=cp, to_host=to_host, drive_pA=p["drive_pA"],
                                               warm_steps=p["warm_steps"], read_steps=p["read_steps"],
                                               up_thresh=p["up_thresh"], hold_steps=hold)
        out = ([_frac(up_driven, r.assemblies[m]) for m in range(n_mem)],
               [_frac(up_release, r.assemblies[m]) for m in range(n_mem)] if hold else None)
        del b, r
        return out

    assemblies = store["assemblies"]
    cueA = _cue_cells(assemblies[iA], seed, iA, p["cue_frac"])
    cueB = _cue_cells(assemblies[iB], seed, iB, p["cue_frac"])
    n_blend = max(1, int(p["blend_cells_each"]))
    blend_cue = np.concatenate([cueA[:n_blend], cueB[:n_blend]])
    ov_blend, ov_blend_release = _read_fresh(blend_cue, weights=store["formed_weights"], hold=hold)

    novelty = max(ov_blend)
    balance = min(ov_blend[iA], ov_blend[iB])
    others_list = [ov_blend[m] for m in range(n_mem) if m not in (iA, iB)]
    others = max(others_list) if others_list else 0.0
    persist_gap = None
    if ov_blend_release:
        persist_gap = max(abs(ov_blend[iA] - ov_blend_release[iA]), abs(ov_blend[iB] - ov_blend_release[iB]))
    fixed_point = bool(persist_gap is not None and persist_gap < 0.20)
    return {"novelty_max_overlap": round(float(novelty), 3), "blend_balance": round(float(balance), 3),
            "blend_vs_other": round(float(others), 3), "fixed_point": fixed_point,
            "persistence_gap": round(float(persist_gap), 3) if persist_gap is not None else None}


def run_seed(seed, *, p=None, n_mem=3, verbose=True, emergent=False):
    from sim.backend import get_backend, to_host
    p = dict(P) if p is None else p
    cp, _ = get_backend()

    # ---- ONE build + ONE BTSP formation pass (the expensive part) -> save the formed + baseline weight arrays ----
    built = _build_and_form(seed, n_mem, p, emergent=emergent)
    p = built["p"]; assemblies_ext = built["assemblies_ext"]; assemblies = built["assemblies"]
    sizes = built["sizes"]; baseline_weights = built["baseline_weights"]; formed_weights = built["formed_weights"]
    diag = built["diag"]; genuine = built["genuine_formation"]
    w_within, cross_dw, nonmem_dw = diag["w_within"], diag["cross_dw"], diag["nonmem_dw"]

    # MEASURED CONTAMINATION (research/findings/raw/_generative_attractor_wander_onsubstrate/*): a second read on
    # an ALREADY-DRIVEN bridge collapses (e.g. a blend re-read fell from [1.0, 0.972, 0.0] to [0.056, 0.0, 0.0]) --
    # short-term synaptic depression on the just-driven recurrent synapses (`CoreSimConfig.enable_short_term_
    # plasticity` defaults True; `_build`'s `enable_stp=False` arg only skips its OWN explicit STP tuning, it never
    # turns the flag off) carries over past `hard_silence`/`_reset_apical_latch` (which reset SOMA + apical state,
    # not synaptic resources). So EVERY read below gets its own FRESH bridge, `R.C.data` seeded from the SAME
    # `formed_weights` (or `baseline_weights` for the lesion condition) -- genuinely independent measurements, not
    # sequential reads on one contaminated bridge. Disabling STP outright was tried and REJECTED: it collapsed the
    # positive-control single-cue completion too (STP dynamics are load-bearing for the plateau reaching
    # threshold), so isolation-by-fresh-bridge is the correct fix, not disabling the mechanism.
    def _read_fresh(cue_ids, *, weights, hold=0):
        b, r = _build_readout(seed, p, n_mem, assemblies_ext=assemblies_ext)
        r.C.data[:] = weights
        up_driven, up_release = _population_up(b, r, cue_ids, cp=cp, to_host=to_host, drive_pA=p["drive_pA"],
                                               warm_steps=p["warm_steps"], read_steps=p["read_steps"],
                                               up_thresh=p["up_thresh"], hold_steps=hold)
        out = ([_frac(up_driven, r.assemblies[m]) for m in range(n_mem)],
               [_frac(up_release, r.assemblies[m]) for m in range(n_mem)] if hold else None)
        del b, r
        return out

    cueA = _cue_cells(assemblies[0], seed, 0, p["cue_frac"])
    cueB = _cue_cells(assemblies[1], seed, 1, p["cue_frac"])
    all_members = set(int(x) for m in range(n_mem) for x in assemblies[m])   # EVERY stored assembly's own cells
    rng = np.random.default_rng(seed * 977 + 13)

    # ---- condition under test: BLENDED cue = blend_cells_each of A's cue-eligible cells + same of B's -------------
    # (the calibrated GRADED operating point -- see the P dict's comment: half of the cue-eligible pool SATURATES
    # both sources to ~1.0 simultaneously; a small ABSOLUTE cell count lands in a genuine partial/balanced regime
    # instead, scale-invariant because the coincident drive depends on absolute cued-neighbour count, not fraction.)
    n_blend = max(1, int(p["blend_cells_each"]))
    halfA = cueA[:n_blend]
    halfB = cueB[:n_blend]
    blend_cue = np.concatenate([halfA, halfB])
    ov_blend, ov_blend_release = _read_fresh(blend_cue, weights=formed_weights, hold=p["hold_steps"])

    # ---- anti-cheat (A): SINGLE-cue positive control (assembly A alone) ------------------------------------------
    ov_single, _ = _read_fresh(cueA, weights=formed_weights)

    # ---- anti-cheat (B): UNTRAINED / lesioned -- same blend cue through the UNFORMED baseline weights -------------
    ov_untrained, _ = _read_fresh(blend_cue, weights=baseline_weights)

    # ---- anti-cheat (C): NOISE cue -- same size, cells outside EVERY stored assembly's own membership + A/B's cue --
    excluded = all_members | set(int(x) for x in cueA) | set(int(x) for x in cueB)
    _bx, _rx = _build_readout(seed, p, n_mem, assemblies_ext=assemblies_ext)   # deterministic -- only ca3_idx used
    ca3_ids = list(_rx.ca3_idx)
    del _bx, _rx
    pool = np.asarray([g for g in ca3_ids if int(g) not in excluded], dtype=np.int64)
    n_noise = len(blend_cue)
    noise_cue = rng.choice(pool, size=min(n_noise, len(pool)), replace=False) if len(pool) else np.array([], dtype=np.int64)
    ov_noise, _ = _read_fresh(noise_cue, weights=formed_weights)

    novelty = max(ov_blend)
    balance = min(ov_blend[0], ov_blend[1])
    others = max(ov_blend[m] for m in range(n_mem) if m not in (0, 1)) if n_mem > 2 else 0.0
    persist_gap = max(abs(ov_blend[0] - ov_blend_release[0]), abs(ov_blend[1] - ov_blend_release[1]))
    single_recovered = ov_single[0]
    single_others = max(ov_single[m] for m in range(1, n_mem)) if n_mem > 1 else 0.0
    noise_sorted = sorted(ov_noise, reverse=True)
    noise_best = noise_sorted[0] if noise_sorted else 0.0
    noise_2nd = noise_sorted[1] if len(noise_sorted) > 1 else 0.0
    untrained_best = max(ov_untrained) if ov_untrained else 0.0

    row = dict(
        seed=seed, n_ca3=p["n_ca3"], emergent=bool(emergent), assembly_sizes=sizes,
        w_within=round(w_within, 2), cross_dw=round(cross_dw, 4),
        nonmem_dw=round(nonmem_dw, 4), genuine_formation=genuine,
        novelty_max_overlap=round(novelty, 3), blend_balance_min=round(balance, 3),
        blend_overlap_others=round(others, 3), persistence_gap=round(persist_gap, 3),
        blend_overlaps_driven=[round(x, 3) for x in ov_blend],
        blend_overlaps_released=[round(x, 3) for x in ov_blend_release],
        single_recovered=round(single_recovered, 3), single_overlap_others=round(single_others, 3),
        noise_best_overlap=round(noise_best, 3), noise_2nd_overlap=round(noise_2nd, 3),
        untrained_best_overlap=round(untrained_best, 3),
    )
    if verbose:
        print(f"  [seed {seed}] sizes={sizes} w_within={w_within:.1f} genuine={genuine} | "
              f"BLEND(A,B): novelty={novelty:.3f} balance={balance:.3f} others={others:.3f} "
              f"persist_gap={persist_gap:.3f} driven={row['blend_overlaps_driven']} "
              f"released={row['blend_overlaps_released']} || SINGLE-cue: recovered={single_recovered:.3f} "
              f"others={single_others:.3f} || NOISE: best={noise_best:.3f} 2nd={noise_2nd:.3f} || "
              f"UNTRAINED-blend: best={untrained_best:.3f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--n-mem", type=int, default=3)
    ap.add_argument("--n-ca3", type=int, default=P["n_ca3"])
    ap.add_argument("--train-events", type=int, default=None,
                    help="default: P['train_events'] (60) normally, or 40 (the production GO_DEFAULTS value) "
                         "when --emergent is set -- see _build_and_form's comment.")
    ap.add_argument("--emergent", action="store_true",
                    help="PRODUCTION-SCALE membership: emergent DG-selected assemblies via "
                         "_gap5_emergent_end_to_end_episodic_loop_derisk.emergent_assemblies (n_ca3 lifts to "
                         "whatever that selection returns, i.e. 2000 at its default R1 config) instead of the "
                         "reduced-scale PRE-ASSIGNED n_ca3=400 membership. --n-ca3 is ignored when set.")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    p = dict(P); p["n_ca3"] = a.n_ca3
    p["train_events"] = a.train_events if a.train_events is not None else P["train_events"]
    print(f"[GENERATIVE ATTRACTOR WANDER on-substrate de-risk, board #104] backend={os.environ.get('SIM_BACKEND')} "
          f"emergent={a.emergent} n_ca3={'<emergent-selected>' if a.emergent else p['n_ca3']} "
          f"assembly_frac={p['assembly_frac']} kthresh={p['kthresh']} blend_cells_each="
          f"{p['blend_cells_each']} | BLENDED cue ({p['blend_cells_each']} of stored-A's cue-eligible cells + "
          f"{p['blend_cells_each']} of stored-B's) into the SAME two-compartment dendritic dAP bistable-latch "
          f"completion the D5/episodic production organ uses -> is the settled UP-state population a NOVEL "
          f"recombination (not equal to any single stored assembly) while staying STABLE (persists past cue "
          f"release)?", flush=True)
    t0 = time.time()
    rows = [run_seed(s, p=p, n_mem=a.n_mem, emergent=a.emergent) for s in a.seeds]
    elapsed = time.time() - t0
    if a.emergent and rows:
        p = dict(p); p["n_ca3"] = rows[0].get("n_ca3", p["n_ca3"])   # the emergent-selected n_ca3, for the record
    if a.json and rows:
        os.makedirs(os.path.dirname(a.json), exist_ok=True)
        json.dump({"rows": rows, "params": p, "elapsed_seconds": elapsed}, open(a.json, "w"), indent=1)

    novelty = [r["novelty_max_overlap"] for r in rows]
    balance = [r["blend_balance_min"] for r in rows]
    others = [r["blend_overlap_others"] for r in rows]
    persist = [r["persistence_gap"] for r in rows]
    genuine = [r["genuine_formation"] for r in rows]
    single_rec = [r["single_recovered"] for r in rows]
    single_oth = [r["single_overlap_others"] for r in rows]
    noise_best = [r["noise_best_overlap"] for r in rows]
    noise_2nd = [r["noise_2nd_overlap"] for r in rows]
    untrained_best = [r["untrained_best_overlap"] for r in rows]

    # GO: genuine BTSP formation (the load-bearing rule wrote the weights) + the blended cue settles into a STABLE
    # (persists past cue release) state that is NOT equal to any single stored assembly (novelty) + BALANCED across
    # both cued sources + clearly above the non-cued third assembly + the single-cue positive control recovers
    # SPECIFICALLY + the untrained (lesioned) network does NOT fake a balanced read.
    go = (all(genuine)
          and all(n < 0.85 for n in novelty)
          and all(b > 0.35 for b in balance)
          and all(b - o > 0.10 for b, o in zip(balance, others))
          and all(g < 0.20 for g in persist)
          and all(s > 0.50 for s in single_rec)
          and all(s < 0.20 for s in single_oth)
          and all(ub < 0.20 for ub in untrained_best))
    noise_clean = all(n2 < 0.20 for n2 in noise_2nd)
    print(f"\n  AGGREGATE ({len(rows)} seeds): novelty={np.mean(novelty):.3f} | balance={np.mean(balance):.3f} "
          f"vs other-stored={np.mean(others):.3f} | persistence_gap={np.mean(persist):.3f} | "
          f"genuine_formation={all(genuine)}", flush=True)
    print(f"  controls: SINGLE-cue recovered={np.mean(single_rec):.3f} (others={np.mean(single_oth):.3f}) | "
          f"UNTRAINED-blend best={np.mean(untrained_best):.3f} (both GATED) || REPORTED: NOISE best="
          f"{np.mean(noise_best):.3f} 2nd={np.mean(noise_2nd):.3f} (clean-at-every-seed={noise_clean})", flush=True)
    verdict_detail = ("SETTLES onto a stable, balanced, genuinely novel recombination via the SAME dendritic dAP "
                      "bistable latch the D5/episodic organ uses -> port SUCCEEDS" if go else
                      "does NOT yet clear the on-substrate bar -- see the failing metric(s) above")
    print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} "
          f"({'anti-cheats ALL clean incl. noise' if go and noise_clean else 'GATED anti-cheats clean; noise caveat' if go else 'not yet'}) "
          f"-- on-substrate port of the generative attractor-wander mechanism ({verdict_detail})", flush=True)


if __name__ == "__main__":
    main()
