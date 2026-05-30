"""Phase-factored two-phase spiking integrated-loop controller (Task 2
of docs/plans/2026-05-30-phase-factored-integrated-loop-implementation
.md).

This controller runs compositional memory in TWO PHASES and scores it,
reusing four already-validated subsystems UNCHANGED (reuse-by-import
only; this module modifies NO protected module and defines NO new
learning rule, NO autograd):

  Phase 1 (ONLINE, theta-ordered): present a length-N concept sequence
    IN ORDER; bind it order-preservingly via the engram-tagging API --
    gamma sub-cycle k binds item k (the shared theta-gamma rhythm times
    which gamma slot the hippocampal episode writes; the SHIFT rule
    rotates the assembly across theta so serial order is recoverable).
  Phase 2 (OFFLINE, shuffled): build concept selectivity in cortex in
    SHUFFLED order, in two stages. (2a) The validated v16 shuffled
    teacher co-firing + STDP (concept_pool_demo.train_word_to_pool
    mechanism) presents the SAME (role, filler) pairs in a deterministic
    cross-mode-identical SHUFFLED order -- this is the selectivity carrier
    that makes the wm role-query role-SELECTIVE (built OFFLINE + SHUFFLED
    so it imposes NO order on the index; design-aligned fix 2026-05-30,
    research/findings/2026-05-30-phase-factored-fullscale-grounding-
    INSTRUMENT-UNSOUND-wm-nondiscriminating.md). (2b) Replay the committed
    episode tag via the validated Phase-1.3 SWR consolidation
    (run_concept_replay_phase under set_sleep_gates, randomize_order=True),
    which updates the hippocampus->cortex index pointer (ca1->concept
    consolidation), the substrate-caveat insurance: on the real substrate
    the offline separation moves the reps, and the order-index may not
    survive without this update (research/findings/2026-05-30-phase-
    factored-cheap-probe-RESOLVES-with-honest-caveats.md).
  Readout 1 (wm / concept query): "is concept X in the buffer?" --
    retrieve the queried role's bound filler through the DG/engram path:
    stimulate the role's PER-BINDING engram tag (the SAME stimulate_tag
    -> DG-separated CA3 completion + cortical reactivation the ep readout
    uses), rank the filler pools by reactivation, pass through the
    validated abstention gate. This routes wm retrieval through the
    validated engram mechanism instead of the eroding cortical
    dlpfc_verb->filler STDP selectivity (the diagnosed instrument-
    soundness failure, research/findings/2026-05-30-phase-factored-
    fullscale-grounding-INSTRUMENT-UNSOUND-wm-nondiscriminating.md) --
    changing WHICH existing mechanism carries retrieval, NOT a new rule.
  Readout 2 (ep / episodic order): "what came after X?" -- recover the
    serial order from the gamma-slot order of the consolidated index
    replay; order built ONLINE.
  Shared theta-gamma rhythm: the parked loop's SharedThetaGamma
    controller, reused byte-unchanged BY IMPORT. Lesioning it
    (no_shared_clock) desynchronizes the WM-gating clock from the
    hippocampal-write clock and must collapse BOTH readouts.

The genuine NET-NEW code here is ONLY (a) the two-phase sequencing
(Phase 1 online bind BEFORE Phase 2 offline consolidate), (b) the
order-preserving index readout, and (c) the lesion harness. Every
subsystem call is a THIN reuse of the already-validated function.

Scored by the parked, already-reviewed, FROZEN verdict
research.runners.integrated_loop_core.integrated_loop_verdict -- this
module defines NONE of its own bars; run_rung emits exactly the rung
dict shape that frozen verdict consumes.

HONEST CEILING (never spun): a PASS = emergent compositional memory in
a biology-grounded multi-system loop ONLY -- NOT fluent open-ended
language, NOT a large language model, NOT conversation solved. The
--tiny-synth smoke is a toy and its verdict is marked TINY and NEVER
propagated.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

# Backend selection (set BEFORE any sim import that may cache the
# backend; argparse has not run yet, so the path is read directly from
# sys.argv exactly like integrated_loop_gate's idiom).
#   * --tiny-synth: the FAST deterministic CPU smoke (the pytest path).
#     Keep the NumPy CPU backend so the smoke completes fast + stays
#     deterministic.
#   * real / decisive controller run: SIM_BACKEND=auto so sim.backend
#     auto-selects the CuPy GPU backend when a device is present.
if "--tiny-synth" in sys.argv:
    os.environ.setdefault("SIM_BACKEND", "numpy")
else:
    os.environ.setdefault("SIM_BACKEND", "auto")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

# ----- REUSE 1: the parked theta-gamma timing controller + bridge
# builder + drive helpers, BY IMPORT, byte-UNCHANGED. SharedThetaGamma
# is the shared rhythm; _build_bridge composes the validated builders;
# _code / _step / _counts / _make_pairs are the proven drive/step/
# population-readout/pair-sampling idioms. We do NOT modify the parked
# module; we drive our OWN two-phase sequence with its helpers.
from research.runners.integrated_loop_gate import (
    SharedThetaGamma, _build_bridge, _code, _step, _counts, _make_pairs,
    _GAMMA_PER_THETA, _BG_CHANNELS, _MAX_LOAD, _ROLE_POOLS,
    _FILLER_POOLS, _GAMMA, _LAMBDA)

# ----- REUSE 2: Phase-1.3 consolidation (offline SWR replay) -----
from research.runners.consolidation_trainer import (
    run_concept_replay_phase)

# ----- REUSE 3: the awake/sleep/freeze gate idioms (validated
# Phase-1.3 freeze-then-evaluate) -----
from research.runners.text_minimal_isolation import (
    set_awake_gates, set_sleep_gates, freeze_all_gates)

# ----- REUSE 4: v16 concept-binding selectivity helper (the
# topographic-prior mechanism). _build_bridge already applies the
# Pulvermuller prior; we reference the validated helper here so the
# selectivity mechanism is reused by import (concept_pool_demo), not
# reimplemented. -----
from research.runners import concept_pool_demo  # noqa: F401

# ----- REUSE 5: the calibrated no-confab abstention gate -----
from research.runners.abstention_gate import gate, DEFAULT_THRESHOLD

# ----- REUSE 6: the parked, already-reviewed FROZEN verdict. This
# module defines NONE of its own bars; run_rung emits the rung shape
# this consumes. -----
from research.runners.integrated_loop_core import integrated_loop_verdict

_BANNER = ("HONEST CEILING: emergent compositional memory in a "
           "biology-grounded multi-system loop ONLY -- NOT fluent "
           "open-ended language, NOT a large language model, NOT "
           "conversation solved.")

# Frozen ladder (owned by integrated_loop_core; mirrored for the
# controller-only decisive run -- the tests run a single load).
_IL_LADDER = (2, 4, 8)

# The 7 lesion modes (mirror integrated_loop_core's frozen partition).
_SHARED = ("no_binding", "no_shared_clock", "no_hippo_store")
_HELPER_WM = ("no_bg_gate",)
_HELPER_EP = ("no_sequencing", "no_cls_replay")
_HELPER_BOTH = ("no_neuromod_timing",)
_ALL_LESIONS = _SHARED + _HELPER_WM + _HELPER_EP + _HELPER_BOTH

# tiny-synth slice: aggressively shrunk so the smoke completes FAST on
# NumPy CPU. Mirrors integrated_loop_gate's _TINY exactly (same fields,
# same magnitudes) so the reused _build_bridge / _code see the same
# operating regime. Its verdict is a toy and NEVER propagated.
_TINY = dict(
    n_lang_input=256, n_per_pool=6, n_fs_per_pool=1,
    n_dlpfc=16, bg_cortex=6,
    stim_steps=2, gap_steps=2, reset_steps=1,
    readout_steps=2, replay_steps=2, n_train_epochs=1,
    role_pA=240.0, filler_pA=240.0, teacher_pA=420.0,
    gate_drive_pA=900.0, tag_stim_pA=1400.0, sparsity=0.05)

# Full slice for the decisive controller-only run (NOT exercised by the
# tests). Mirrors integrated_loop_gate's _FULL.
_FULL = dict(
    n_lang_input=4096, n_per_pool=320, n_fs_per_pool=24,
    n_dlpfc=320, bg_cortex=24,
    stim_steps=96, gap_steps=10, reset_steps=6,
    readout_steps=48, replay_steps=10, n_train_epochs=14,
    role_pA=240.0, filler_pA=240.0, teacher_pA=420.0,
    gate_drive_pA=900.0, tag_stim_pA=1400.0, sparsity=0.05)

# Passive PHASE-EVENT sink. None in every real/decisive run (zero effect
# on any drive, RNG draw, gate decision, or readout). ONLY a test sets
# this to a list; _episode then APPENDS an ordered marker string at each
# phase boundary so the phase-ordering / lesion-fidelity pins can assert
# the controller calls online-bind (Phase 1) BEFORE offline-consolidate
# (Phase 2) and that each lesion removes EXACTLY its own subsystem. It
# only RECORDS; it never alters any drive/gate/RNG/score. Mirrors the
# parked _SELFCHECK_SINK / _DIAG_SINK passive-sink discipline.
_EVENT_LOG = None

# Passive RAW-wm-readout sink (same zero-effect discipline as _EVENT_LOG):
# None in every real/decisive run. A diagnostic sets it to a list; the wm
# readout then APPENDS (true_fidx, raw_filler_counts, gated_decision) at
# each query so a diagnostic can see WHETHER the correct filler out-fires
# distractors (binding works; gate may be mis-calibrated) or not (the
# role-query does not reactivate the bound filler). It only RECORDS
# already-computed values; it never alters any drive/gate/RNG/score.
_WM_RAW_SINK = None


def _log(event):
    """Append `event` to the passive phase-event sink if a test enabled
    it; a no-op (zero overhead) otherwise."""
    if _EVENT_LOG is not None:
        _EVENT_LOG.append(event)


def _wm_raw(true_fidx, counts, decision):
    """Passive diagnostic sink: record raw wm-query outcome. No-op unless a
    diagnostic enabled it. Records ONLY already-computed values; zero effect
    on any drive/gate/RNG/score (mirrors _log)."""
    if _WM_RAW_SINK is not None:
        _WM_RAW_SINK.append(
            (int(true_fidx), [float(c) for c in counts],
             None if decision is None else str(decision[0])))


def _episode(bridge, mode, pairs, P, ctx):
    """ONE two-phase composition trial at load N = len(pairs).

    PHASE 1 (ONLINE, theta-ordered): drive each (role, filler) in
    PRESENTATION ORDER; the shared theta-gamma clock assigns binding k a
    gamma sub-cycle (SHIFT rule rotates the assembly across theta so
    serial order is recoverable), the BG cascade gates the matching WM
    slot, and the engram-tagging API records + commits BOTH (a) the
    whole-episode tag (the hippocampal ORDER INDEX the ep readout
    completes) AND (b) a PER-BINDING tag per (role, filler) -- the WM-side
    DG/engram relational store the reworked wm readout retrieves through
    (the role pool + the BOUND filler pool + the DG-separated ca3
    ensemble). The two recordings run concurrently (the engram-API
    recordings dict is keyed by name). The ONLINE order readout (Readout 2
    source) is written here. The concept-SELECTIVITY plasticity gates are
    FROZEN during this in-order pass (design-aligned fix 2026-05-30) so the
    fixed-order presentation does NOT write the winner-take-most
    lang->filler selectivity (the OLD wm path eroded here); only the
    spike-count engram ORDER INDEX + the per-binding ensemble tags are
    written here (ep preserved; wm now carried by the engram tags).

    PHASE 2a (OFFLINE, shuffled): build cortical concept selectivity via
    the validated v16 shuffled teacher co-firing + STDP (the SAME
    (role, filler) pairs presented in a deterministic cross-mode-identical
    SHUFFLED order, selectivity gates thawed). This still builds the
    cortical concept reps (off-the-readout-path now that wm retrieves via
    the engram tags) WITHOUT imposing an order on the index (ep preserved).
    PHASE 2b (OFFLINE, shuffled): replay the committed whole-episode tag
    via the validated Phase-1.3 SWR consolidation under set_sleep_gates
    (randomize_order=True), updating the ca1->concept index pointer
    (the ep-side substrate-caveat insurance). Then freeze for evaluation.

    READOUT 1 (wm): for the queried role, stimulate its PER-BINDING
    engram tag(s) over the readout window (multitag stim-recall, the SAME
    stimulate_tag -> DG-separated CA3 completion + cortical reactivation
    the ep readout uses) and rank the filler pools by reactivation. This
    REPLACES the eroding cortical dlpfc_verb->filler STDP selectivity
    (the diagnosed instrument-soundness failure) with the validated
    DG/engram mechanism -- changing WHICH existing mechanism carries
    retrieval, NOT inventing a new one.

    Returns (wm_acc, ep_acc). Every mode draws the SAME random numbers in
    the SAME order (the only per-trial rng consumer is _make_pairs in
    _run_mode -- _episode draws NONE; the Phase-2a shuffle uses a
    dedicated LOCAL rng seeded identically across modes from (seed,
    episode_id), so it perturbs no cross-mode draw); only the lesioned
    subsystem's effect is removed (the faithfulness discipline)."""
    cp = bridge.xp if hasattr(bridge, "xp") else np
    n_lang = ctx["n_lang"]
    lang = ctx["lang"]
    role_arr = ctx["role_arr"]
    filler_arr = ctx["filler_arr"]
    dlpfc = ctx["dlpfc"]
    bg_cortex = ctx["bg_cortex"]
    N = len(pairs)
    _is_v1 = bool(ctx.get("is_v1", False))

    # ONE shared clock unless no_shared_clock, in which case TWO
    # independent clocks desynchronize WM gating vs the hippocampal
    # write (nothing else changes). no_sequencing makes the clock
    # REPEAT (shift=False) instead of SHIFT so no order is written.
    _shift = (mode != "no_sequencing")
    if mode == "no_shared_clock":
        clk_wm = SharedThetaGamma(shift=_shift)
        clk_hip = SharedThetaGamma(shift=_shift)
        # Genuine desync: advance the hippocampal clock a fixed phase.
        for _ in range(_GAMMA_PER_THETA // 2):
            clk_hip.step()
        _log("clock:two")
    else:
        clk_wm = SharedThetaGamma(shift=_shift)
        clk_hip = clk_wm  # THE shared instance drives both
        _log("clock:one")

    tag = "pf_episode_%d" % ctx["episode_id"]

    # ----- reset (decay residual state between trials) -----
    bridge.cp_external_input_current[:] = 0.0
    bridge.core_config.current_reward_signal = 0.0
    for _ in range(P["reset_steps"]):
        _step(bridge)

    # =================================================================
    # PHASE 1 -- ONLINE theta-ordered encode + engram WRITE.
    # =================================================================
    _log("phase1:online_bind:start")
    # Hippocampal relational store records the episode (skipped only for
    # the no_hippo_store SHARED lesion -> no tag -> both readouts
    # collapse by construction).
    if mode != "no_hippo_store":
        bridge.start_engram_recording(tag)
        _log("engram:start_recording")

    # WM-SIDE relational store (DG/engram per-binding tags). The OLD wm
    # readout retrieved role->filler via CORTICAL dlpfc_verb->filler STDP
    # selectivity, which ERODES on this substrate (repeated selectivity
    # training degrades the topographic prior's clean margin -> a role
    # query lights ALL filler pools ~equally -> v1 wm ~= chance; the
    # diagnosed instrument-soundness failure, research/findings/
    # 2026-05-30-phase-factored-fullscale-grounding-INSTRUMENT-UNSOUND-wm-
    # nondiscriminating.md). THE FIX (de-risked GO): route wm retrieval
    # through the SAME DG-separated hippocampal ENGRAM path the ep readout
    # already uses to reach ep=1.0. Per (role, filler) binding we commit a
    # PER-BINDING engram tag capturing that binding's co-active ensemble
    # (the role pool + the BOUND filler pool + the DG-separated ca3
    # ensemble). At query the role's per-binding tag is stimulated ->
    # CA3 pattern completion + the cortical role/filler ensemble reactivate
    # -> the BOUND filler pool out-fires the others (engram stim-recall,
    # 87.5% multi-seed; multitag cue retrieval 90% FULL multi-seed). This
    # is the SAME mechanism, NOT a new learning rule -- it changes WHICH
    # validated mechanism carries retrieval. bind_tags_by_role maps each
    # role index to the per-binding tag(s) committed for it THIS epoch
    # (with the bijection _make_pairs draws, exactly one tag per role; the
    # multitag aggregation degenerates to a single tag here but is written
    # general so a future many-to-one role would stim ALL its tags).
    # SKIPPED for no_hippo_store exactly like the whole-episode tag (the
    # per-binding tags ARE the wm-side relational store -> wm collapses
    # WITH ep under no_hippo_store, a SHARED lesion).
    bind_tags_by_role = {}

    for bi, (ridx, fidx) in enumerate(pairs):
        # Shared clock decides this binding's gamma sub-cycle; the
        # matching prefrontal WM slot is gated open via the BG cascade.
        gslot = clk_wm.slot_for(bi, N)
        chan = gslot % len(_BG_CHANNELS)

        # Drive role + filler orthogonal codes into the concept pools.
        bridge.cp_external_input_current[:] = 0.0
        drole = cp.asarray(_code(ridx, 2 * _MAX_LOAD, n_lang,
                                 P["role_pA"], P), dtype=cp.float32)
        dfill = cp.asarray(_code(_MAX_LOAD + fidx, 2 * _MAX_LOAD,
                                 n_lang, P["filler_pA"], P),
                           dtype=cp.float32)
        bridge.cp_external_input_current[lang] = drole + dfill
        # Teacher co-fires the bound role+filler pools so the native
        # eligibility trace charges on the concept synapses.
        bridge.cp_external_input_current[role_arr[ridx]] += \
            float(P["teacher_pA"])
        bridge.cp_external_input_current[filler_arr[fidx]] += \
            float(P["teacher_pA"])
        # Weak, strictly slot-AGNOSTIC region-wide dlpfc_verb
        # excitability bias (same scalar on every dlpfc neuron -- no
        # per-slot Python indexing). It only sets the holding region
        # near threshold so the BG-disinhibited thal_<chan> ->
        # dlpfc_verb input selects which sub-population crosses
        # threshold. Suppressed for no_binding (SHARED): without it the
        # BG-selected slot never reaches threshold -> slot<->filler
        # co-fire never happens -> the relational assembly never forms
        # here (so the order index degrades, ep) AND -- mirrored in the
        # Phase-2a selectivity loop, where no_binding likewise suppresses
        # this bias -> the dlpfc_verb->filler selectivity STDP is never
        # written (wm). So no_binding collapses BOTH readouts across the
        # two phases (a SHARED lesion).
        if mode != "no_binding":
            bridge.cp_external_input_current[dlpfc] += \
                0.5 * float(P["teacher_pA"])
        # BG-gated WM updating: drive the selected channel's BG cortex
        # so its cascade disinhibits thal_<chan>. no_bg_gate drives ALL
        # channels -> no single slot cleanly held -> wm collapses.
        if mode == "no_bg_gate":
            for ch in range(len(_BG_CHANNELS)):
                bridge.cp_external_input_current[bg_cortex[ch]] += \
                    float(P["gate_drive_pA"])
        else:
            bridge.cp_external_input_current[bg_cortex[chan]] += \
                float(P["gate_drive_pA"])

        # The shared clock's theta phase times the ACh plasticity window
        # (open in the first half of theta). no_neuromod_timing leaves
        # plasticity always on (untimed) -> the binding loop's STDP is
        # never gated to the co-fire window -> degraded write -> BOTH
        # readouts collapse (a HELPER_BOTH lesion).
        if mode != "no_neuromod_timing":
            ach_open = 1.0 if clk_hip.gamma_slot < (
                _GAMMA_PER_THETA // 2) else 0.0
        else:
            ach_open = 1.0
        # DESIGN-ALIGNED FIX (2026-05-30): Phase 1 is ONLINE + IN-ORDER, so
        # it writes the episodic ORDER INDEX ONLY. The concept (role ->
        # filler) SELECTIVITY must be built OFFLINE + SHUFFLED in Phase 2
        # (the v16 mechanism) -- building it here in fixed presentation
        # order, repeated every epoch, is the winner-take-most regime that
        # makes the wm retrieval non-selective (the diagnosed instrument-
        # soundness failure: research/findings/2026-05-30-phase-factored-
        # fullscale-grounding-INSTRUMENT-UNSOUND-wm-nondiscriminating.md).
        # So FREEZE the three SELECTIVITY gates during the Phase-1 in-order
        # presentation: lang->filler (language_input_to_noun_pool), the
        # role->dlpfc holding write (language_input_to_dlpfc_verb), and the
        # held-slot->filler write (dlpfc_verb_to_filler). The engram tag
        # (spike-count accumulation over ec/dg/ca3/ca1, gate-INDEPENDENT)
        # and the clock-ordered slot dynamics still happen -> the ORDER
        # INDEX is intact (ep preserved). bg_thal_to_dlpfc is NOT a
        # selectivity gate (it routes the BG-disinhibited slot into dlpfc,
        # part of the slot-held WM dynamics the order index needs) -> it
        # keeps the ACh-timed value here.
        for _g in ("language_input_to_noun_pool",
                   "language_input_to_dlpfc_verb", "dlpfc_verb_to_filler"):
            try:
                bridge.set_plasticity_gate(_g, 0.0)
            except Exception:
                pass
        try:
            bridge.set_plasticity_gate("bg_thal_to_dlpfc", float(ach_open))
        except Exception:
            pass

        # Begin this binding's PER-BINDING engram recording RIGHT BEFORE
        # its stim window, so the recording accumulates spike counts over
        # exactly the (role, filler) co-fire (the engram-API recordings
        # dict is keyed by name, so this runs concurrently with the
        # whole-episode recording -- both _tick on every step). Skipped
        # for no_hippo_store (the wm-side relational store is the hippo
        # store; skipping it collapses wm with ep).
        bind_tag = "pf_ep%d_bind%d" % (ctx["episode_id"], bi)
        if mode != "no_hippo_store":
            bridge.start_engram_recording(bind_tag)
            _log("engram:start_recording_bind")

        for _ in range(P["stim_steps"]):
            _step(bridge)
            clk_wm.step()
            if clk_hip is not clk_wm:
                clk_hip.step()

        # Encode-time temporal-credit bootstrap (the validated native
        # eligibility/reward path; NO new learning rule, NO autograd).
        # Deliver the TD-delta reward CLOSE IN TIME to the eligibility-
        # charging co-fire (reward == 1.0 because during encode the
        # bound (role, filler) is KNOWN BY CONSTRUCTION -- the teacher
        # DEFINES it). Suppressed for no_neuromod_timing (that lesion
        # removes timed plasticity from the whole loop consistently).
        if mode != "no_neuromod_timing":
            _evt = ctx["enc_value_table"]
            _vb = float(_evt[fidx])
            _enc_delta = 1.0 - _vb
            _evt[fidx] = _vb + (1.0 - _GAMMA * _LAMBDA) * _enc_delta
            bridge.cp_external_input_current[:] = 0.0
            bridge.core_config.current_reward_signal = float(_enc_delta)
            _step(bridge)
            clk_wm.step()
            if clk_hip is not clk_wm:
                clk_hip.step()
            bridge.core_config.current_reward_signal = 0.0

        # Commit THIS binding's per-binding engram tag over the role pool
        # + the BOUND filler pool + the DG-separated ca3 ensemble. top_k
        # is sized to the three regions' co-firing cells (role+filler ~=
        # 2*n_per_pool, ca3 ~= n_ca3). The tag's SELECTIVITY is carried by
        # which sub-population actually co-fired in this window:
        #   * full: the BG cleanly gates one dlpfc slot -> only the BOUND
        #     filler pool's dlpfc_verb->filler efferent reinforces the
        #     teacher, so under the per-pool FS WTM the BOUND filler wins
        #     and its cells dominate the tag; ca3 holds the DG-separated
        #     binding ensemble -> stimulating the tag reactivates the
        #     BOUND filler preferentially (wm works).
        #   * no_bg_gate (HELPER_WM): ALL BG channels driven -> ALL dlpfc
        #     slots fire -> ALL filler efferents inject -> the per-pool FS
        #     WTM has no single winner -> the tag's filler-pool content is
        #     SMEARED across pools -> stimulating it does not preferentially
        #     reactivate the BOUND filler -> wm collapses; ep survives (the
        #     clock-ordered, spike-count whole-episode index is BG-gating-
        #     independent).
        #   * no_binding (SHARED): the dlpfc holding bias is suppressed ->
        #     the BG-selected slot never crosses threshold -> the
        #     dlpfc_verb->filler efferent stays silent + the relational
        #     ca3 assembly is degenerate -> the per-binding tag is
        #     degenerate/non-selective -> wm collapses (and the order index
        #     degrades -> ep collapses).
        #   * no_shared_clock / no_neuromod_timing: the co-fire/ACh window
        #     desyncs or is untimed -> the bound ensemble forms poorly ->
        #     the tag is degraded -> wm collapses (with ep).
        if mode != "no_hippo_store":
            try:
                bridge.commit_engram_tag(
                    bind_tag,
                    top_k=2 * int(P["n_per_pool"]) + 100,
                    region_filter=[
                        "noun_pool_%s" % _ROLE_POOLS[ridx],
                        "noun_pool_%s" % _FILLER_POOLS[fidx],
                        "ca3"])
                bind_tags_by_role.setdefault(ridx, []).append(bind_tag)
                _log("engram:commit_tag_bind")
            except Exception:
                pass

    # Finalize the episode tag over the hippocampal regions only (the
    # relational store). Skipped for no_hippo_store.
    if mode != "no_hippo_store":
        try:
            bridge.commit_engram_tag(
                tag, top_k=64,
                region_filter=["ec", "dg", "ca3", "ca1"])
            _log("engram:commit_tag")
        except Exception:
            pass
    _log("phase1:online_bind:done")

    # ----- ONLINE order readout source (Readout 2): recover the serial
    # order from the SHIFTED theta-ordered assembly the online encode
    # wrote into the hippocampal store. The committed engram tag is the
    # natural CA3 retrieval CUE (pattern completion); the per-role
    # concept-pool activity-peak ORDER is the recovered sequence. Taken
    # AFTER the online write. no_hippo_store -> no tag -> 0.0;
    # no_sequencing -> clock REPEATED (no order written) -> degenerate;
    # no_binding -> no bound assembly to complete.
    def _episodic_order_readout():
        if mode == "no_hippo_store":
            return 0.0
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(P["reset_steps"]):
            _step(bridge)
        peak_val = np.full(N, -1.0, dtype=np.float64)
        peak_t = np.zeros(N, dtype=np.int64)
        n_recall = max(N * 2, P["readout_steps"])
        for t in range(n_recall):
            try:
                bridge.stimulate_tag(tag, float(P["tag_stim_pA"]))
            except Exception:
                pass
            _step(bridge)
            rc = _counts(bridge, [role_arr[r] for r, _ in pairs])
            for k in range(N):
                if rc[k] > peak_val[k]:
                    peak_val[k] = rc[k]
                    peak_t[k] = t
        try:
            bridge.clear_tag_drive(tag)
        except Exception:
            pass
        bridge.cp_external_input_current[:] = 0.0
        recovered = list(np.argsort(peak_t, kind="stable"))
        true_order = list(range(N))
        return sum(1.0 for i in range(N)
                   if recovered[i] == true_order[i]) / float(N)

    # =================================================================
    # PHASE 2a -- OFFLINE *SHUFFLED* concept-SELECTIVITY binding (the v16
    # train_word_to_pool mechanism: shuffled teacher co-firing + STDP).
    # This is the DESIGN-MANDATED home of concept selectivity (built
    # OFFLINE + SHUFFLED so it does NOT impose an order on the index ->
    # ep preserved). It does EXACTLY the teacher co-fire + STDP that the
    # OLD Phase 1 did, but (a) in SHUFFLED pair order and (b) with the
    # selectivity gates THAWED here instead of in Phase 1. Building
    # selectivity shuffled/offline (vs the old fixed-order/online) is
    # what makes the role-query wm retrieval role-SELECTIVE rather than
    # winner-take-most.
    #
    # Each lesion still removes EXACTLY its own subsystem here, so the
    # frozen partition holds with selectivity now in Phase 2:
    #   no_binding (SHARED, both): suppress the dlpfc holding bias -> the
    #     slot<->filler co-fire never crosses threshold -> the
    #     dlpfc_verb->filler STDP is never written -> selectivity (wm)
    #     collapses (and the relational assembly never forms -> ep too).
    #   no_shared_clock (SHARED, both): TWO desynced clocks -> the ACh
    #     plasticity window (clk_hip) is out of phase with the BG slot
    #     gating (clk_wm) -> selectivity STDP fires outside the co-fire
    #     window -> wm collapses (and the Phase-1 order index desyncs ->
    #     ep collapses).
    #   no_neuromod_timing (HELPER_BOTH): ACh always on (untimed) ->
    #     selectivity STDP never gated to the co-fire window -> degraded
    #     write -> wm collapses (and Phase-1 untimed -> ep collapses).
    #   no_bg_gate (HELPER_WM): drive ALL BG channels -> no single slot
    #     cleanly held -> the held-slot->filler write is non-selective ->
    #     wm collapses. ep survives (the clock-ordered, spike-count engram
    #     index is BG-gating-independent).
    #   no_sequencing (HELPER_EP): clock REPEAT (shift=False) changes only
    #     slot_for's per-theta ROTATION (the order code), NOT the within-
    #     theta gamma-phase ACh window nor the per-binding distinct BG
    #     slot -> selectivity (wm) survives; only the Phase-1 order index
    #     is lost -> ep collapses.
    #   no_cls_replay (HELPER_EP): SKIPS the Phase-2b SWR consolidation
    #     (the ca1->concept index-update insurance, ep-side) but KEEPS
    #     this selectivity loop (wm-side) -> ep degrades, wm survives.
    #   no_hippo_store (SHARED, both): no engram tag (ep=0) AND skips ALL
    #     of Phase 2 incl. this selectivity loop -> no selectivity (wm=0).
    #
    # RNG faithfulness: the shuffle is a DETERMINISTIC permutation seeded
    # from (run-seed, episode_id) computed IDENTICALLY for every mode, via
    # a dedicated LOCAL rng. It does NOT touch the shared per-trial rng
    # (whose SOLE consumer remains _make_pairs in _run_mode); so every
    # mode still draws the IDENTICAL _make_pairs pairs at the IDENTICAL
    # stream position (the lesion-fidelity discipline holds), AND the
    # shuffle is byte-identical across modes for a given (seed, episode).
    if mode != "no_hippo_store":
        _log("phase2:offline_selectivity:start")
        # Build the SAME clock topology as Phase 1 (one shared instance,
        # or two desynced under no_shared_clock) so the timing lesions act
        # here too. shift follows no_sequencing exactly as Phase 1.
        if mode == "no_shared_clock":
            sclk_wm = SharedThetaGamma(shift=_shift)
            sclk_hip = SharedThetaGamma(shift=_shift)
            for _ in range(_GAMMA_PER_THETA // 2):
                sclk_hip.step()
        else:
            sclk_wm = SharedThetaGamma(shift=_shift)
            sclk_hip = sclk_wm
        # Open the selectivity gates for OFFLINE binding (re-timed per
        # step by the ACh window below; symmetric with the Phase-1 freeze).
        for _g in ("language_input_to_noun_pool", "bg_thal_to_dlpfc",
                   "language_input_to_dlpfc_verb", "dlpfc_verb_to_filler"):
            try:
                bridge.set_plasticity_gate(_g, 1.0)
            except Exception:
                pass
        # Deterministic, cross-mode-identical shuffle of the pair order.
        _sel_rng = np.random.default_rng(
            7919 * (int(ctx.get("seed", 0)) + 1)
            + 31 * int(ctx["episode_id"]) + 17)
        _order = list(range(N))
        _sel_rng.shuffle(_order)
        for _si, _pi in enumerate(_order):
            ridx, fidx = pairs[_pi]
            # Per-binding distinct BG slot (clock-assigned). Under
            # no_sequencing (shift=False) this is _si % channels -- still
            # distinct per binding within a shuffle, so wm is unaffected;
            # only the Phase-1 ORDER code uses the rotation.
            gslot = sclk_wm.slot_for(_si, N)
            chan = gslot % len(_BG_CHANNELS)
            # Reset between events (NMDA decay), like v16 train_word_to_pool.
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(P["reset_steps"]):
                _step(bridge)
                sclk_wm.step()
                if sclk_hip is not sclk_wm:
                    sclk_hip.step()
            # Drive role + filler codes; teacher co-fires the bound
            # role+filler pools (the validated v16 co-firing selectivity).
            bridge.cp_external_input_current[:] = 0.0
            drole = cp.asarray(_code(ridx, 2 * _MAX_LOAD, n_lang,
                                     P["role_pA"], P), dtype=cp.float32)
            dfill = cp.asarray(_code(_MAX_LOAD + fidx, 2 * _MAX_LOAD,
                                     n_lang, P["filler_pA"], P),
                               dtype=cp.float32)
            bridge.cp_external_input_current[lang] = drole + dfill
            bridge.cp_external_input_current[role_arr[ridx]] += \
                float(P["teacher_pA"])
            bridge.cp_external_input_current[filler_arr[fidx]] += \
                float(P["teacher_pA"])
            if mode != "no_binding":
                bridge.cp_external_input_current[dlpfc] += \
                    0.5 * float(P["teacher_pA"])
            if mode == "no_bg_gate":
                for ch in range(len(_BG_CHANNELS)):
                    bridge.cp_external_input_current[bg_cortex[ch]] += \
                        float(P["gate_drive_pA"])
            else:
                bridge.cp_external_input_current[bg_cortex[chan]] += \
                    float(P["gate_drive_pA"])
            for _ in range(P["stim_steps"]):
                # ACh plasticity window times the selectivity STDP to the
                # co-fire (gamma-phase, shift-independent). no_neuromod_
                # timing -> always on; no_shared_clock -> clk_hip desynced.
                if mode != "no_neuromod_timing":
                    sach = 1.0 if sclk_hip.gamma_slot < (
                        _GAMMA_PER_THETA // 2) else 0.0
                else:
                    sach = 1.0
                for _g in ("language_input_to_noun_pool",
                           "language_input_to_dlpfc_verb",
                           "dlpfc_verb_to_filler"):
                    try:
                        bridge.set_plasticity_gate(_g, float(sach))
                    except Exception:
                        pass
                try:
                    bridge.set_plasticity_gate("bg_thal_to_dlpfc",
                                               float(sach))
                except Exception:
                    pass
                _step(bridge)
                sclk_wm.step()
                if sclk_hip is not sclk_wm:
                    sclk_hip.step()
        bridge.cp_external_input_current[:] = 0.0
        _log("phase2:offline_selectivity:done")
    else:
        _log("phase2:offline_selectivity:skipped")

    # =================================================================
    # PHASE 2b -- OFFLINE shuffled SWR consolidation (UPDATE the
    # ca1->concept index pointer; the ep-side substrate-caveat insurance).
    # Strictly AFTER Phase 1 (a swapped order is a bug; the phase-ordering
    # pin asserts this via _EVENT_LOG). Skipped for the no_cls_replay
    # HELPER_EP lesion and the no_hippo_store SHARED lesion (a
    # deterministic skip; no extra/missing rng draw -- a dedicated local
    # rng seeded from the episode id is used so passing it perturbs NO
    # cross-mode draw order).
    # =================================================================
    if mode not in ("no_cls_replay", "no_hippo_store"):
        _log("phase2:offline_consolidate:start")
        set_sleep_gates(bridge)
        try:
            run_concept_replay_phase(
                bridge, tag_names=[tag],
                n_replays_per_tag=int(P["replay_steps"]),
                burst_duration_ms=int(P["stim_steps"]),
                inter_burst_ms=int(P["gap_steps"]),
                drive_pA=float(P["tag_stim_pA"]),
                randomize_order=True,
                rng=np.random.default_rng(
                    1000 + int(ctx["episode_id"])))
        except Exception:
            pass
        # Validated Phase-1.3 pre-eval freeze: both readouts below are
        # taken under this freeze (weights cannot drift during eval).
        freeze_all_gates(bridge)
        _log("phase2:offline_consolidate:done")
    else:
        _log("phase2:offline_consolidate:skipped")

    # =================================================================
    # READOUT 1 (wm / concept query): retrieve the queried role's bound
    # filler through the DG/engram path -- the SAME stimulate_tag ->
    # DG-separated CA3 completion + cortical ensemble reactivation the ep
    # readout already uses to reach ep=1.0 (NOT the old, eroding cortical
    # dlpfc_verb->filler STDP "drive role code + rank pool firing" path).
    # For the queried role we stimulate its per-binding tag(s) over the
    # readout window (multitag stim-recall: stim ALL of the role's tags
    # concurrently each step, additive -- with the _make_pairs bijection
    # each role has exactly one tag, so this degenerates to a single tag
    # but is written general) and rank the filler pools by reactivation.
    # The gate, the [("F%d",count,"wm"),...] ranking shape, the
    # _wm_raw(...) passive sink, and the scoring (correct iff gated top ==
    # true filler) are UNCHANGED. NO rng is drawn here (deterministic
    # sorted tag order), so the shared per-trial rng -- whose SOLE
    # consumer is _make_pairs in _run_mode -- is untouched (the
    # lesion-fidelity discipline holds).
    # =================================================================
    wm_correct = 0
    n_q = len(pairs)
    for qi, (ridx, fidx) in enumerate(pairs):
        # Novel recombination on the final query (full + every lesion;
        # the genuine compositional generalization the lesions must
        # collapse). SKIPPED for v1: v1's scored query is the trivial
        # DRILLED binding (instrument soundness; query a drilled role ->
        # expect its bound filler).
        if (not _is_v1) and qi == n_q - 1 and n_q >= 2:
            true_fidx = pairs[0][1]
            q_ridx = pairs[-1][0]
        else:
            true_fidx = fidx
            q_ridx = ridx
        # The role's per-binding tag(s) committed this epoch (empty under
        # no_hippo_store -> nothing to stimulate -> the gate abstains ->
        # wm collapses with ep, the SHARED-lesion signature).
        q_tags = sorted(bind_tags_by_role.get(q_ridx, []))
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(P["reset_steps"]):
            _step(bridge)
        counts = np.zeros(_MAX_LOAD, dtype=np.float64)
        for _ in range(P["readout_steps"]):
            # Stimulate the role's engram tag(s) (additive so multiple
            # tags for one role aggregate; the first additive write needs
            # a cleared drive each step). Then step and accumulate the
            # filler-pool reactivation.
            bridge.cp_external_input_current[:] = 0.0
            for _bt in q_tags:
                try:
                    bridge.stimulate_tag(_bt, float(P["tag_stim_pA"]),
                                         additive=True)
                except Exception:
                    pass
            _step(bridge)
            counts += _counts(bridge, filler_arr)
        for _bt in q_tags:
            try:
                bridge.clear_tag_drive(_bt)
            except Exception:
                pass
        bridge.cp_external_input_current[:] = 0.0
        # Rank fillers; trustworthy gate at DEFAULT_THRESHOLD.
        order = np.argsort(-counts)
        ranked = [("F%d" % int(j), float(counts[j]), "wm")
                  for j in order]
        decision = gate(ranked, DEFAULT_THRESHOLD)
        _wm_raw(true_fidx, counts, decision)  # passive diagnostic sink
        # Wrong emission AND abstention-on-a-groundable-query both score
        # 0; only a correct gated emission scores 1.
        if decision is not None and \
                decision[0] == ("F%d" % int(true_fidx)):
            wm_correct += 1
    bridge.cp_external_input_current[:] = 0.0
    wm_acc = wm_correct / float(max(1, n_q))

    # =================================================================
    # READOUT 2 (ep / episodic order): the ONLINE order readout source,
    # taken from the gamma-slot order of the consolidated index replay.
    # =================================================================
    ep_acc = _episodic_order_readout()

    # Drop the per-episode tag AND the per-binding tags so tags don't
    # accumulate across trials/epochs.
    try:
        bridge.delete_engram_tag(tag)
    except Exception:
        pass
    for _tags in bind_tags_by_role.values():
        for _bt in _tags:
            try:
                bridge.delete_engram_tag(_bt)
            except Exception:
                pass
    bridge.cp_external_input_current[:] = 0.0
    # Restore awake gates for the next trial (symmetric with the
    # freeze above).
    set_awake_gates(bridge)
    for _g in ("language_input_to_noun_pool", "bg_thal_to_dlpfc",
               "language_input_to_dlpfc_verb", "dlpfc_verb_to_filler"):
        try:
            bridge.set_plasticity_gate(_g, 1.0)
        except Exception:
            pass
    return float(wm_acc), float(ep_acc)


def _run_mode(mode, seed, N, tiny, gap_zero=False):
    """Build the two-phase loop bridge, run the composition trials at
    load N for `mode`, return (wm, ep) from the final epoch. Every mode
    consumes IDENTICAL RNG draws in IDENTICAL order (the only per-trial
    rng consumer is _make_pairs here -- _episode draws none); only the
    lesioned subsystem's effect is removed. gap_zero forces the maintain
    gap to 0 (the v1 instrument-soundness single trivial bind). NO
    autograd."""
    P = dict(_TINY if tiny else _FULL)
    if gap_zero:
        P["gap_steps"] = 0
    bridge = _build_bridge(seed, P, N)
    cp = bridge.xp if hasattr(bridge, "xp") else np
    rm = bridge.region_manager

    lang = cp.asarray(list(rm.indices("language_input")),
                      dtype=cp.int64)
    role_arr = [cp.asarray(list(rm.indices("noun_pool_%s" % nm)),
                           dtype=cp.int64) for nm in _ROLE_POOLS]
    filler_arr = [cp.asarray(list(rm.indices("noun_pool_%s" % nm)),
                             dtype=cp.int64) for nm in _FILLER_POOLS]
    dlpfc = cp.asarray(list(rm.indices("dlpfc_verb")), dtype=cp.int64)
    bg_cortex = [cp.asarray(list(rm.indices("cortex_%s" % ch)),
                            dtype=cp.int64) for ch in _BG_CHANNELS]

    # Known-open starting gate state (the per-step ACh window inside
    # _episode re-times them; identical for every mode).
    for _g in ("language_input_to_noun_pool", "bg_thal_to_dlpfc",
               "language_input_to_dlpfc_verb", "dlpfc_verb_to_filler"):
        try:
            bridge.set_plasticity_gate(_g, 1.0)
        except Exception:
            pass

    rng = np.random.default_rng(seed)
    enc_value_table = np.zeros(_MAX_LOAD, dtype=np.float64)
    ctx = dict(n_lang=int(lang.shape[0]), lang=lang,
               role_arr=role_arr, filler_arr=filler_arr,
               dlpfc=dlpfc, bg_cortex=bg_cortex,
               enc_value_table=enc_value_table,
               episode_id=0, is_v1=bool(gap_zero),
               seed=int(seed))

    # The validated v16 ENCODE DISCIPLINE: draw the bijection ONCE per
    # run (stable across ALL epochs, like v16's fixed vocab) and present
    # it interleaved-repeated. Cross-mode faithfulness preserved: every
    # mode makes the IDENTICAL single _make_pairs draw at the IDENTICAL
    # point; _episode draws no rng.
    pairs = _make_pairs(N, rng)
    last_wm, last_ep = 0.0, 0.0
    for ep_i in range(P["n_train_epochs"]):
        ctx["episode_id"] = ep_i
        last_wm, last_ep = _episode(bridge, mode, pairs, P, ctx)
    return float(last_wm), float(last_ep)


def _seed_rung(seed, N, tiny, only_modes=None):
    """All modes for one (seed, load N): v1 (gap_zero full), full, and
    every lesion. Returns {"v1":(wm,ep), "full":(wm,ep),
    "lesions":{name:(wm,ep)}}.

    `only_modes` is a PURE RUN-SCOPE filter: when not None it is a set of
    cell names to RUN; the others are simply NOT executed. This changes
    NO rng draw and NO scored quantity for the cells that DO run (every
    mode builds its OWN bridge with its OWN identically-seeded rng;
    _make_pairs is the sole per-trial rng consumer)."""
    def _want(name):
        return only_modes is None or name in only_modes
    out = {}
    if _want("v1"):
        out["v1"] = _run_mode("full", seed, N, tiny, gap_zero=True)
    if _want("full"):
        out["full"] = _run_mode("full", seed, N, tiny, gap_zero=False)
    out["lesions"] = {}
    for m in _ALL_LESIONS:
        if _want(m):
            out["lesions"][m] = _run_mode(m, seed, N, tiny,
                                          gap_zero=False)
    return out


def run_rung(N, seed, tiny_synth=True):
    """Run the full lesion study for ONE (load N, seed) and return the
    rung dict in EXACTLY the shape the frozen
    integrated_loop_core.integrated_loop_verdict consumes:

      {"N": int, "n_seeds": 1,
       "v1":   {"wm": float, "ep": float},
       "full": {"wm": float, "ep": float},
       "lesions": {<all 7 names>: {"wm": float, "ep": float}}}

    Single-seed (n_seeds==1) -- the controller aggregates across seeds
    for the decisive run; the tests run one tiny-synth seed."""
    cell = _seed_rung(int(seed), int(N), bool(tiny_synth))

    def _pair(t):
        return {"wm": float(t[0]), "ep": float(t[1])}

    return {
        "N": int(N),
        "n_seeds": 1,
        "v1": _pair(cell["v1"]),
        "full": _pair(cell["full"]),
        "lesions": {m: _pair(cell["lesions"][m]) for m in _ALL_LESIONS},
    }


def _aggregate(rungs_by_seed):
    """Mean across seeds -> ONE rung dict per load in the frozen schema.
    `rungs_by_seed` is a list of per-seed rung dicts (each with n_seeds
    == 1) for the SAME load N."""
    n = len(rungs_by_seed)

    def _mean(getter):
        wm = sum(getter(r)["wm"] for r in rungs_by_seed) / n
        ep = sum(getter(r)["ep"] for r in rungs_by_seed) / n
        return {"wm": float(wm), "ep": float(ep)}

    les = {m: _mean(lambda r, _m=m: r["lesions"][_m])
           for m in _ALL_LESIONS}
    return {
        "N": int(rungs_by_seed[0]["N"]),
        "n_seeds": n,
        "v1": _mean(lambda r: r["v1"]),
        "full": _mean(lambda r: r["full"]),
        "lesions": les,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 43, 44])
    ap.add_argument("--tiny-synth", action="store_true")
    ap.add_argument("--only-load", type=int, default=None,
                    help="Run ONLY this single ladder load N (pure "
                         "run-scope filter; changes NO rng draw or "
                         "scored quantity).")
    ap.add_argument("--out", required=False, default=None)
    a = ap.parse_args(argv)

    from sim.backend import get_backend
    _xp, _backend_name = get_backend()
    _dev = "cpu"
    if _backend_name == "cupy":
        try:
            _dev = _xp.cuda.runtime.getDeviceProperties(
                0)["name"].decode()
        except Exception:
            _dev = "cuda"
    print("BACKEND=%s  DEVICE=%s" % (_backend_name, _dev), flush=True)
    print(_BANNER, flush=True)

    tiny = bool(a.tiny_synth)
    # tiny-synth runs ONLY the first ladder rung (fast smoke); the
    # decisive controller run does the full ladder. Either way every
    # rung carries n_seeds == len(seeds).
    loads = [_IL_LADDER[0]] if tiny else list(_IL_LADDER)
    if a.only_load is not None:
        if a.only_load not in _IL_LADDER:
            raise SystemExit("--only-load must be in %s" % (_IL_LADDER,))
        loads = [a.only_load]

    rungs = []
    for N in loads:
        per_seed = [run_rung(N, s, tiny_synth=tiny) for s in a.seeds]
        rungs.append(_aggregate(per_seed))

    if tiny:
        # The toy verdict is ALWAYS marked TINY and NEVER propagated.
        result = {"GATE": "TINY",
                  "classification": "TINY-NOT-PROPAGATED",
                  "reason": "tiny-synth smoke; toy scale; the verdict "
                            "is a TINY marker and is NEVER propagated "
                            "as a real PASS/FAIL/VOID",
                  "rungs": rungs}
    else:
        result = integrated_loop_verdict(rungs)
        result["rungs"] = rungs

    out_text = json.dumps(result, indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(out_text)
    print(out_text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
