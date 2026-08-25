"""Continuous-state engine (2026-08-19 reframe: "make the brain continuous") — v1: the mood keeps evolving BETWEEN turns.

THE POINT (the LLM-surpassing differentiator): a brain is ALIVE between questions — it keeps feeling / thinking /
changing when nothing is asked; an LLM (and a plastic RAG) is only alive while being queried. Today the sim brain
mostly wakes per-turn. This engine adds an always-on background TICK so the substrate runs between requests. It is
LESS blocked than fluent speech: it needs recurrent activity + a spiking read + a homeostatic relaxation — all
on-substrate today, no deep credit.

v1 scope (one seed of the four; the smallest genuinely-continuous property): the FELT MOOD keeps evolving while a
session is idle. Each tick, for an idle session: (a) the appraisal EMA RELAXES toward baseline (the felt state
decays with no new input — a homeostatic process; the appraisal drive is the body/appraisal boundary), then
(b) the spiking affect ladder is RE-READ at the relaxed appraisal — so the mood the brain *feels* now is a genuine
spiking read, evolving between turns. Each tick is recorded to a per-session INNER-LIFE log the next turn's
monologue surfaces ("while you were away, my mood drifted from X toward neutral"). The mood read IS spiking; the
relaxation is a host homeostat (declared). Next rungs (own tasks): self-initiated WANDER on the tick (surface a
concept), idle BTSP CONSOLIDATION, generative attractor-wandering.

Default-OFF behind `BRAIN_CONTINUOUS`; when off the tick loop is inert (byte-identical to today). Host code here is
the CLOCK + the relaxation formula + the log — legitimate world/body-timer infrastructure (it computes no
cognition; every mood read reuses the existing spiking affect ladder). No sim/ edit.
"""
from __future__ import annotations

import os
import time

IDLE_SEC = 20.0            # a session with no request for this long is "idle" -> the tick may run on it
RELAX = 0.85              # per-tick appraisal relaxation toward the neutral setpoint (felt state decays with no input)
NEUTRAL = 0.0            # the appraisal setpoint the mood relaxes toward
_INNER_LIFE_MAX = 24     # keep the last N tick records per session (a short autobiographical trace)


def _wander_budget_per_turn() -> int:
    """How many heavy self-init WANDERs a session may run per idle period (refilled on each real turn).

    PRODUCTION-SAFETY (2026-08-20): the wander is a ~55s CA3 run on cupy. Without a bound, an idle session
    would fire one every IDLE_SEC forever -> an abandoned server pegs the GPU indefinitely, and N idle sessions
    serialize into N*55s tick batches. Bounding it to a small budget per idle period keeps the ALIVE-between-turns
    behaviour a returning user sees (the wander surfaces on the next turn) while the mind SETTLES when a
    conversation is truly abandoned. The cheap mood relaxation still runs every tick regardless. Biologically:
    mind-wandering during rest is ongoing but not maximal-intensity forever; an abandoned rest state settles."""
    try:
        return max(0, int(os.environ.get("BRAIN_WANDER_BUDGET", "1")))
    except Exception:
        return 1


# per-session inner-life: cache_key -> list of {t, valence, arousal, differential, note}
_INNER_LIFE: dict = {}
# per-session last-request wall-clock (set by the handler); "idle" = now - last >= IDLE_SEC
_LAST_REQUEST: dict = {}
# per-session remaining heavy-wander budget for this idle period (refilled to _wander_budget_per_turn() each turn)
_WANDER_BUDGET: dict = {}
# per-session topic the last live turn RECALLED (set by the handler on an in_memory referential recall); the next
# idle tick CONSOLIDATES it (D5 learn-through-use). None/absent until a genuine spiking recall happens.
_RECALLED_TOPIC: dict = {}
# per-session remaining D5-consolidation budget for this idle period (refilled by mark_recall on each real recall)
_D5_BUDGET: dict = {}
# per-session SET of topics that have actually been D5-CONSOLIDATED (learn-through-use) this conversation. The recall
# reply surfaces the graded recall STRENGTH only for a topic in this set (see recall_disclosure) — so consolidating one
# memory can only change ITS OWN reply, never a neighbour's (the no-regression property). Populated on a successful
# consolidate_used_memory; cleared on forget_session. Empty whenever BRAIN_D5_CONSOLIDATE is off (nothing consolidates),
# so the OFF reply stays byte-identical to HEAD.
_CONSOLIDATED_TOPICS: dict = {}
# per-session composer store-size (len(kb)) at the LAST DA-encoding substrate-homeostasis consolidation pass. The idle
# tick runs the Turrigiano synaptic-scaling pass (webapp/da_encoding_drives_chat.apply_substrate_homeostasis) only when
# the store has GROWN since the last pass (new facts were taught) — Turrigiano scaling is slow/offline and re-running it
# on an already-scaled store with no new writes would compound the strong-engram down-regulation toward unit and erase
# the DA-salience ordering. Cleared on forget_session. Empty whenever the DA-encoding faculty is off (nothing scales),
# so the =0 escape stays byte-identical to HEAD.
_LAST_HOMEO_KB: dict = {}


# 2026-08-21 FLIP: the between-turn CONTINUOUS LIFE is DEFAULT-ON (the mission-defining flip — the brain keeps a
# thought wandering + its mood relaxing between turns). Gated by the soak's no-regression (ON byte-identical to OFF on
# ordinary turns with no pending wander — round-1 0/21 diverged, VRAM stable) + byte-identical-by-construction (the
# drive only prepends a lead when recent_wander() is non-None, which needs a prior idle tick). `BRAIN_CONTINUOUS=0` is
# the byte-identical escape (reverts to the pre-flip inert loop). Mirrors the affect-drives / gnw-multistep flips.
_CONTINUOUS_DEFAULT_ON = True


def continuous_enabled() -> bool:
    """Default-ON anchor (flipped 2026-08-21). `BRAIN_CONTINUOUS=0` disarms the background tick (byte-identical
    escape to the pre-flip behaviour); unset -> the default; any of {1,true,on,yes} also arms it."""
    v = os.environ.get("BRAIN_CONTINUOUS")
    if v is None:
        return _CONTINUOUS_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "on", "yes")


def mark_request(cache_key) -> None:
    """The handler calls this each turn so the tick knows a session is active (and not to tick mid-conversation).
    Also REFILLS this session's heavy-wander budget: a real turn re-opens the mind to wander again during the next
    idle period (so a returning user sees a fresh wandered thought), then the budget drains as it wanders."""
    _LAST_REQUEST[cache_key] = time.time()
    _WANDER_BUDGET[cache_key] = _wander_budget_per_turn()


def forget_session(cache_key) -> None:
    """On a session reset, drop its continuous state (mirrors _SESSION_MOOD/_SESSION_SELFINIT cleanup)."""
    _LAST_REQUEST.pop(cache_key, None)
    _INNER_LIFE.pop(cache_key, None)
    _WANDER_BUDGET.pop(cache_key, None)
    _WANDER_ADAPT.pop(cache_key, None)
    _RECALLED_TOPIC.pop(cache_key, None)
    _D5_BUDGET.pop(cache_key, None)
    _CONSOLIDATED_TOPICS.pop(cache_key, None)
    _LAST_HOMEO_KB.pop(cache_key, None)
    _IDEATE_TICK.pop(cache_key, None)


# ── D5 LEARN-THROUGH-USE: consolidate a memory the brain USED, between turns ─────────────────────────────────────
# Rung 3 of the continuous-substrate frontier (arc-1 step-4, board: the production-integration rung). A memory the
# brain RECALLED during a live turn is CONSOLIDATED during the following idle period: the arc-1 recall → step-2
# self-terminating apical-plateau window → the substrate's OWN plateau-gated BTSP loop runs on the organ's real
# store, so a used memory becomes measurably more robust for a LATER turn (learn-through-use). Default-OFF behind
# `BRAIN_D5_CONSOLIDATE`; byte-identical to HEAD when off. The arc: steps 1-3 are 6/6-GO
# (research/findings/2026-08-20-d5-learn-through-use-*-arc1-closed.md); this wires them under the idle tick.
_D5_EPISODES = 1          # recall→strengthen episodes per consolidation call. CORRECTED 3→1 (2026-08-20, the graded-read
                          # flip): the surfaced GRADED recall strength (depth_hold) rises SMOOTHLY + monotonically at
                          # n_episodes=1 (the conversation-visible learn-through-use signal) but SATURATES on the first
                          # tick at n_episodes=3 (step-6 de-risk: depth_hold GO 5/6 @ n_ep=1 vs 2/6 @ n_ep=3). The
                          # robustness gain is front-loaded (max-lesion-survived moves by episode 1-2), so one episode
                          # per tick still strengthens the used memory (it accumulates across turns) while keeping the
                          # graded read climbing gradually. GPU cost ~580 steps/episode. Finding
                          # 2026-08-20-d5-learn-through-use-DEFAULT-ON-graded-apical-read.
# The step-3 GO knobs (the recall→self-terminating-window→BTSP loop). up_thresh / v_hold are read from the organ.
_D5_RK = dict(tau_w=150.0, tau_apical=15.0, cue_pa=300.0, ignite_steps=80, window_steps=500,
              btsp_lr=0.02, btsp_w_max=100.0, btsp_elig_tau_ms=1000.0, b_adapt=0.8)


# 2026-08-21 FLIP: D5 learn-through-use consolidation is DEFAULT-ON. The idle tick re-activates the memory the last turn
# RECALLED and the substrate's OWN plateau-gated BTSP strengthens it, so a used memory recalls VISIBLY STRONGER next
# turn. GATED by the 6-seed no-regression soak (research/runners/_d5_graded_flip_soak.py, sep_bias=0): 5/6 GO + 1
# moat-abstaining self-ignition build (s102), OFF byte-identical to HEAD, crash-rollback 6/6. The default-ON flip is
# safe BECAUSE the surfaced recall strength is gated PER CONSOLIDATED TOPIC (recall_disclosure + _CONSOLIDATED_TOPICS):
# consolidating one memory changes ONLY its own reply. `BRAIN_D5_CONSOLIDATE=0` is the byte-identical escape to HEAD.
# finding 2026-08-21-d5-learn-through-use-flip-GO-per-topic-strength-surfacing-the-prior-NO-GO-was-a-surfacing-artifact.
_D5_CONSOLIDATE_DEFAULT_ON = True


def d5_consolidate_enabled() -> bool:
    """Default-ON anchor (flipped 2026-08-21). Unset -> the default (on); `BRAIN_D5_CONSOLIDATE=0` (or false/no/off) is
    the byte-identical escape: the tick's consolidation step is inert AND no recall surfaces a strength -> a later recall
    is byte-identical to HEAD. Any of {1,true,on,yes} also arms it."""
    v = os.environ.get("BRAIN_D5_CONSOLIDATE")
    if v is None:
        return _D5_CONSOLIDATE_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "on", "yes")


def topic_consolidated(cache_key, topic) -> bool:
    """True iff `topic` has actually been D5-CONSOLIDATED (learn-through-use) this conversation. Gates whether the recall
    reply surfaces the risen graded strength for it (recall_disclosure). Because a memory is added to the set only on a
    SUCCESSFUL consolidation (which requires BRAIN_D5_CONSOLIDATE on), a topic never consolidated — including EVERY topic
    when the flag is off — reads False, so the reply is byte-identical to HEAD. Pure bookkeeping (a set lookup)."""
    if not topic:
        return False
    return str(topic).lower() in _CONSOLIDATED_TOPICS.get(cache_key, ())


def _d5_budget_per_recall() -> int:
    """How many heavy consolidation calls a session may run per idle period (refilled by mark_recall on each real
    recall). PRODUCTION-SAFETY (mirrors the wander budget): each call is a ~few-second cupy run; bounding it means an
    idle server does not peg the GPU re-consolidating the same memory forever, while a returning user still gets the
    used memory strengthened on the next turn. Biologically: post-encoding replay is repeated but not unbounded."""
    try:
        return max(0, int(os.environ.get("BRAIN_D5_CONSOLIDATE_BUDGET", "1")))
    except Exception:
        return 1


def mark_recall(cache_key, topic) -> None:
    """The live handler calls this when a referential turn RECALLED a topic via a genuine spiking completion
    (in_memory=True). The next idle tick CONSOLIDATES that memory (learn-through-use). Refills this session's
    consolidation budget (a real recall re-opens between-turn consolidation), which then drains as it consolidates.
    Pure bookkeeping (a dict write) — no cognition, no spiking here; the strengthening runs in the tick."""
    if topic:
        _RECALLED_TOPIC[cache_key] = str(topic).lower()
        _D5_BUDGET[cache_key] = _d5_budget_per_recall()


def consolidate_used_memory(cache_key, episodic_organ, *, n_episodes: int | None = None) -> dict | None:
    """BETWEEN-TURN D5 CONSOLIDATION (learn-through-use). For the topic the last live turn RECALLED, run the arc-1
    step-3 loop on the organ's REAL store: re-activate the assembly (a sustained cue → the dendritic apical latch
    COMPLETES it), the step-2 self-terminating apical-plateau window opens, and the substrate's OWN plateau-gated
    BTSP (`sim/bridge.py` `fused_btsp_update`, gated by `IS_post = max(cp_v_apical − v_hold, 0)`) potentiates the
    co-active within-assembly recurrence — written back to `mem.R.C.data` BY OBJECT IDENTITY (the same array
    `recall()` reads). So a memory the brain USED becomes more robust for the NEXT turn.

    BRAIN-BASED: the weight change is the substrate's own plasticity kernel (no host `dw` formula); host code here
    is only the clock, the episode budget, and the snapshot/restore determinism guard (the step-2/3 runners' guard).
    BYTE-IDENTITY: every `cfg` field and BTSP transient the reactivate loop mutates is saved and restored, so the
    ONLY lasting change to the organ is the strengthened within-assembly weights — a later recall differs from HEAD
    in nothing else. Returns a record, or None (disabled / no recalled topic / no store yet / topic not formed)."""
    if not d5_consolidate_enabled():
        return None
    topic = _RECALLED_TOPIC.get(cache_key)
    if not topic or episodic_organ is None:
        return None
    mem = getattr(episodic_organ, "mem", None)
    if mem is None:
        return None
    slot = mem.topic_slot.get(topic)
    if slot is None or slot not in getattr(mem, "formed", set()):
        return None  # a completion-failure recall (or an unformed topic) leaves no store to consolidate

    # Heavy imports kept lazy so the OFF path stays byte-identical + import-light (reuse-by-import; NO sim/ edit).
    import numpy as _np
    from research.runners._gap5_dendritic_dap_readout_completion_derisk import _reset_apical_latch
    from research.runners._gap5_d5_latch_self_termination_derisk import snapshot_state, restore_state
    from research.runners._gap5_d5_learn_through_use_derisk import reactivate

    bridge = mem.bridge
    cp = mem.R.cp
    cfg = bridge.core_config
    n_ep = int(_D5_EPISODES if n_episodes is None else n_episodes)
    rk = dict(_D5_RK)
    rk["up_thresh"] = mem.p["up_thresh"]
    rk["v_hold"] = mem.p["v_hold"]
    cue_full = _np.asarray(mem.cue_by_asm[slot], dtype=_np.int64)

    # SAVE the cfg fields + BTSP transients that `reactivate` mutates, so nothing but the weights persists.
    _cfg_keys = (
        "enable_hebbian_learning", "enable_stdp", "enable_structural_plasticity", "enable_bdsp", "enable_btsp",
        "btsp_learning_rate", "btsp_w_max", "btsp_w_min", "btsp_elig_tau_ms", "btsp_hetero_dep",
        "btsp_milstein_k_dep", "btsp_mean_subtract", "btsp_dog_a_dep", "btsp_elig_tau_slow_ms",
        "btsp_win_gate_theta", "btsp_elig_exponent", "btsp_elig_hard_thresh", "coincidence_plateau_v_hold")
    cfg_saved = {k: getattr(cfg, k, None) for k in _cfg_keys}
    btsp_saved = {a: getattr(bridge, a, None)
                  for a in ("cp_btsp_pre_elig", "cp_btsp_pre_elig_slow", "cp_btsp_win_count", "cp_btsp_wmax")}

    w0 = float(cp.mean(bridge.cp_connections.data[mem.R.withinA_masks[slot]]))
    # ROLLBACK anchor: a full copy of the pre-consolidation weights. `reactivate` mutates the organ's PERSISTENT store
    # (bridge.cp_connections.data == mem.R.C.data) IN PLACE across episodes, so a crash mid-loop -- e.g. the RTX 3090
    # falling off the bus mid-load -- would otherwise leave the store corrupted with no rollback (and the armed topic
    # un-drained, so the next idle tick re-runs from the half-mutated weights, compounding drift).
    W_pre = bridge.cp_connections.data.copy()
    try:
        # WARM the apical latch if it was never allocated (a fresh bridge has cp_v_apical=None; the reactivate loop's
        # plateau-gated BTSP then never fires -> zero potentiation). In the live flow the recall that ARMED this
        # consolidation already warmed it, so this is a cheap no-op there; it self-heals the cold-start case.
        if getattr(bridge, "cp_v_apical", None) is None:
            try:
                mem.recall(topic)
            except Exception:
                pass
        # a clean transient rest holding the CURRENT weights, snapshotted (the reactivate loop restores it each ep)
        mem.R.hard_silence()
        _reset_apical_latch(bridge)
        snap = snapshot_state(bridge)
        W = bridge.cp_connections.data.copy()
        for _i in range(n_ep):
            W = reactivate(mem, slot, snap, W, cue_indices=cue_full, strengthen=True,
                           clamp_apical=False, adapt_on=True, **rk)["W_out"]
        # leave the bridge at CLEAN REST with only the strengthened weights persisting
        restore_state(bridge, snap)
        bridge.cp_connections.data[:] = W
        wN = float(cp.mean(bridge.cp_connections.data[mem.R.withinA_masks[slot]]))
    except Exception:
        # ON-PATH SAFETY: on ANY failure mid-consolidation, roll the PERSISTENT store back to its pre-consolidation
        # state and DRAIN the armed topic, then re-raise so the caller LOGS it (never silently corrupts + retries).
        try:
            bridge.cp_connections.data[:] = W_pre
        except Exception:
            pass
        _RECALLED_TOPIC.pop(cache_key, None)
        raise
    finally:
        for k, v in cfg_saved.items():
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
        for a, v in btsp_saved.items():
            setattr(bridge, a, v)

    _RECALLED_TOPIC.pop(cache_key, None)  # consolidate a given recall once (drained; a new recall re-arms it)
    # LEARN-THROUGH-USE SURFACING GATE: record that THIS topic has been consolidated this conversation, so its later
    # recall reply surfaces the risen graded strength — and ONLY its reply (a neighbour that was never consolidated is
    # not in this set, so its reply is untouched). Reached only on the SUCCESS path (a crash re-raises above, before
    # this line, so a rolled-back consolidation never surfaces a strength).
    _CONSOLIDATED_TOPICS.setdefault(cache_key, set()).add(topic)
    rec = {"t": time.time(), "consolidated": topic, "slot": int(slot), "n_episodes": n_ep,
           "w_within_before": round(w0, 3), "w_within_after": round(wN, 3),
           "note": "idle: consolidated the used memory ‘%s’ (within-assembly weight %.1f → %.1f)" % (topic, w0, wN)}
    lst = _INNER_LIFE.setdefault(cache_key, [])
    lst.append(rec)
    if len(lst) > _INNER_LIFE_MAX:
        del lst[:-_INNER_LIFE_MAX]
    return rec


# ── DA-ENCODING SUBSTRATE HOMEOSTASIS: the Turrigiano synaptic-scaling consolidation pass, between turns ─────────────
# The DA-gated encoding faculty writes a taught fact at a per-write RECALL-SAFE FLOOR (g >= set-point) so the LIVE store
# is always safe. The POPULATION regulation — multiplicatively down-scaling over-strong (high-DA) engrams toward the
# activity set-point while preserving their DA-salience ORDER (Turrigiano 2008 homeostatic synaptic scaling) — needs the
# WHOLE stored population and is biologically SLOW/OFFLINE (hours-days, during sleep). So it runs here, on the idle tick,
# as a CONSOLIDATION pass — alongside D5 learn-through-use — NOT per write. The pass SENSES each engram's readout
# activity on the substrate and rescales its store synapses (webapp/da_encoding_drives_chat.apply_substrate_homeostasis
# -> OneBrainComposer.apply_homeostatic_scaling), a real synaptic-weight change. The finding: research/findings
# 2026-08-25-da-encoding-substrate-turrigiano-scaling-FLIP (6-seed GO, byte-equal to a real production build).
def consolidate_substrate_homeostasis(cache_key, chat) -> dict | None:
    """Run ONE DA-encoding substrate-homeostasis (Turrigiano synaptic-scaling) consolidation pass on this idle session's
    live composer store, IF the store has grown since the last pass. Returns a record (or None when disabled / no
    composer / no new writes / the pass was a no-op).

    NEW-WRITES-SINCE-LAST-PASS TRIGGER (load-bearing): `apply_homeostatic_scaling` is idempotent input-wise but NOT
    idempotent on repeated calls — a second pass on an already-scaled store re-senses the (now-regulated) activity and
    keeps pulling strong engrams toward unit, eroding the DA-salience ordering. So the pass fires only when new engrams
    were stored since the last consolidation (len(kb) grew), matching the biology: scaling consolidates a BATCH of
    freshly-encoded facts once, offline, not continuously.

    BRAIN-BASED + GATED: the actual scaling self-gates inside `apply_substrate_homeostasis` (no-op unless
    da_encoding_enabled() AND da_encoding_substrate_enabled() AND not lesioned), so with `BRAIN_DA_ENCODING=0` this is a
    pure no-op -> byte-identical to HEAD. Never raises (an idle-tick helper must not crash the loop)."""
    comp = getattr(getattr(chat, "inner", None), "composer", None)
    if comp is None:
        return None
    try:
        cur_len = len(getattr(comp, "kb", []) or [])
    except Exception:
        return None
    if cur_len == 0:
        return None
    last_len = _LAST_HOMEO_KB.get(cache_key, 0)
    if cur_len <= last_len:
        return None  # no new facts taught since the last scaling pass -> don't re-scale (would compound toward unit)
    try:
        from webapp import da_encoding_drives_chat as _DAE
        scales = _DAE.apply_substrate_homeostasis(chat)  # self-gates on faculty/substrate/lesion; None == disabled/no-op
    except Exception:
        import logging as _lg
        _lg.getLogger(__name__).warning(
            "DA-encoding substrate homeostasis failed for %s", cache_key, exc_info=True)
        return None
    if scales is None:
        return None  # faculty off / substrate homeostat off / lesioned / composer lacks the rule -> byte-identical
    _LAST_HOMEO_KB[cache_key] = cur_len  # mark this batch consolidated (only advanced on an ACTUAL scaling pass)
    n = len(scales)
    rec = {"t": time.time(), "substrate_homeostasis": True, "n_engrams": n,
           "note": "idle: Turrigiano synaptic-scaling pass over %d DA-encoded engram(s)" % n}
    lst = _INNER_LIFE.setdefault(cache_key, [])
    lst.append(rec)
    if len(lst) > _INNER_LIFE_MAX:
        del lst[:-_INNER_LIFE_MAX]
    return rec


def inner_life(cache_key) -> list:
    """The recent between-turn tick records for a session (for the monologue read-out)."""
    return list(_INNER_LIFE.get(cache_key, []))


def recent_wander(cache_key) -> str | None:
    """Rung 2 (board #86, 2026-08-20): make the idle-wandered THOUGHT load-bearing on the NEXT real turn (drive, not
    just observe -- mirrors the #84 affect-lead / #85 swap-lead pattern). Returns the most recent concept an idle
    tick's self-initiation organ wandered to for this session, or None if continuous is off, no idle tick has run
    yet, or no tick surfaced a concept. CONSUMES the record on read (sets it back to None) so the same wandered
    concept is brought up exactly once -- on the next live turn after the tick that produced it -- not repeated on
    every subsequent turn. Pure bookkeeping over the existing inner-life log; no new spiking read (the concept was
    already produced by the selfinit organ's spiking selection at tick time, see tick_session)."""
    if not continuous_enabled():
        return None
    lst = _INNER_LIFE.get(cache_key)
    if not lst:
        return None
    for rec in reversed(lst):
        w = rec.get("wandered")
        if w:
            rec["wandered"] = None  # consume -> surfaces on exactly the next turn, not every turn after
            return w
    return None


IOR_STRENGTH = 0.15      # multiply the just-wandered basin's curiosity gain by this (inhibition-of-return fatigue)
IOR_RECOVERY = 0.3       # fraction of each basin's adaptation deficit recovered per tick (slower = longer memory).
                         # 0.3 verified best in a cupy coverage sweep (wander_ior_r0.30.json): reaches 3 of the 4
                         # stored concepts vs 2/4 at the old 0.5.
IOR_GAIN_FLOOR = 1.6     # clamp each basin's base curiosity STEERING gain UP to this floor before IOR fatigue.
                         # The 4th concept never surfaced at 3/4 because its steering gain was too LOW to win even when
                         # the top 3 are IOR-fatigued (the winner is steering-dominated, 2026-08-20-per-neuron-SFA-
                         # wrong-locus). A floor raises the tail's drive -> full 4/4 coverage, verified cupy GO
                         # (wander_gainfloor.json: IOR-only 3/4 'cat,dog,bird' -> IOR+floor 4/4 'cat,dog,bird,fish').
# per-session wander inhibition-of-return: cache_key -> {"base": [gains], "adapt": [multipliers]}
_WANDER_ADAPT: dict = {}


def _wander_ior_enabled() -> bool:
    """Default-ON anti-fixation for the between-turn wander. Without it the wander is content-DEGENERATE (6/6 'cat',
    finding 2026-08-20-continuous-wander-content-degenerate) — a load-bearing coupling to a constant. IOR (fatigue the
    just-ignited basin so the next wander explores elsewhere; GO in 2026-08-20-inhibition-of-return-breaks-the-
    degenerate-wander) makes trains-of-thought actually move. `BRAIN_WANDER_IOR=0` restores the pre-IOR behaviour."""
    return os.environ.get("BRAIN_WANDER_IOR", "1").strip().lower() in ("1", "true", "on", "yes")


def _pre_wander_ior(cache_key, organ) -> None:
    """Before the wander: bias this session's curiosity gains AWAY from recently-visited basins (apply the adaptation).
    SCAFFOLD NOTE: this modulates the neuromod curiosity DRIVE host-side — the de-risked stand-in for a per-neuron CA3
    spike-frequency-adaptation current (the faithful burn-down, reusing the 2026-08-14 SFA machinery). No-op on the
    first tick (no adaptation captured yet) and whenever gains are unavailable."""
    st = _WANDER_ADAPT.get(cache_key)
    g = getattr(organ, "gains_on", None)
    if st is not None and g:
        organ.gains_on = [float(st["base"][j] * st["adapt"][j]) for j in range(len(st["base"]))]


def _post_wander_ior(cache_key, organ, concept) -> None:
    """After the wander: FATIGUE the basin that just won, then RECOVER all basins toward rest. Captures the pristine
    base gains on the first call (before any pre-adaptation has run)."""
    g = getattr(organ, "gains_on", None)
    agents = getattr(organ, "agents", None)
    if not g or not agents or concept not in agents:
        return
    st = _WANDER_ADAPT.get(cache_key)
    if st is None:  # first post: gains_on is the pristine base (this tick's pre was a no-op)
        # clamp the base steering gains up to the floor so the weakly-driven tail concept can win under IOR (4/4).
        base = [max(float(x), IOR_GAIN_FLOOR) for x in g]
        st = {"base": base, "adapt": [1.0] * len(g)}
        _WANDER_ADAPT[cache_key] = st
    i = list(agents).index(concept)
    st["adapt"][i] *= IOR_STRENGTH                                              # fatigue the just-won basin
    st["adapt"] = [1.0 - (1.0 - a) * (1.0 - IOR_RECOVERY) for a in st["adapt"]]  # all basins recover toward rest


# ── IDEATION: an OCCASIONAL between-turn wander that GENERATES a NOVEL blended concept (creativity rung) ──────────
# Rung of the continuous-substrate frontier: today the idle wander SELECTS one stored basin and speaks its concept
# (recall). IDEATION makes the wander OCCASIONALLY *generate* instead — it drives a BLENDED cue of the TWO most
# curiosity-active basins into a sparse associative-attractor, which settles into a NOVEL recombination that was
# NEVER stored (novelty from the DYNAMICS, not the nodes). Reuse-by-import of the GO de-risk
# (research/runners/_generative_attractor_wander_derisk: Tsodyks-Feigelman sparse-Hopfield + the ca3_ff_inhib
# MEAN+std dynamic-threshold settle — a fixed feedforward-inhibition threshold, not a forced top-k, is what lets a
# blend stay balanced instead of collapsing onto one source; finding 2026-08-20-generative-attractor-wander-derisk).
#
# STRICTLY ADDITIVE + DEFAULT-OFF behind `BRAIN_CONTINUOUS_IDEATE`. Unset -> the wander is EXACTLY today's
# single-basin recall selection (byte-identical to the live default-on continuous wander); the ideation branch is
# never entered and no `ideation` key is recorded. The live continuous default (BRAIN_CONTINUOUS) is untouched.
#
# HONESTY BOUNDARY (load-bearing): a novel-ideation wander is TAGGED as a novel idea/association (kind=
# "novel-association", is_fact=False), surfaced via `recent_ideation()` on a DISTINCT channel from `recent_wander()`
# (recalled concepts). The next turn frames it as "a thought that occurred to me", NEVER as a stored fact, and it
# NEVER enters the recall/abstain moat as an assertion (it only decorates an already-matched surface, like the
# other between-turn leads). FUNCTIONAL creativity correlate, NOT a phenomenal claim.
#
# DECLARED SCAFFOLDS (named, not hidden): (1) the every-Nth-tick cadence is a host-timed scheduler (WHEN to ideate,
# like the idle-tick clock); (2) the fast standalone numpy attractor is the de-risked stand-in for the on-substrate
# CA3 blend (the SAME latency residual the self-init organ declares — cupy CA3 is ~seconds, numpy@scale is minutes);
# (3) the SELECTION of the two source basins rides the organ's spiking curiosity gains (one-brain merge #1).

IDEATE_NOVELTY_MAX = 0.85    # the settled state's max-overlap with ANY single stored basin must be BELOW this (it is
                             #  not any single stored item; a single recall reads 1.0). A 2-source blend sits ~0.72.
IDEATE_BALANCE_MIN = 0.50    # both cued sources genuinely represented: min(overlap A, overlap B) ABOVE this.
IDEATE_BLEND_MARGIN = 0.15   # the balanced blend must exceed any OTHER (non-cued) stored basin by this margin (the
                             #  novelty is a RECOMBINATION of the two cued sources, not arbitrary drift).
_IDEATE_N, _IDEATE_K, _IDEATE_THRESH_C = 1200, 60, 0.7   # de-risk operating point (clean 6-seed x 2-scale)

# per-session ideation-tick counter (advanced ONLY while ideation is enabled -> untouched, byte-identical, when off)
_IDEATE_TICK: dict = {}


def ideation_enabled() -> bool:
    """Default-OFF anchor. `BRAIN_CONTINUOUS_IDEATE` in {1,true,on,yes} arms the between-turn IDEATION mode. Unset/0
    -> the wander is EXACTLY today's single-basin recall selection (byte-identical to the live continuous wander)."""
    # 2026-08-21 FLIPPED default-ON (owner-approved; composed no-regression GO — byte-identical on out-of-scope
    # turns since ideation only fires on an IDLE tick, never during a live recall/self turn). BRAIN_CONTINUOUS_IDEATE=0
    # is the byte-identical escape.
    return os.environ.get("BRAIN_CONTINUOUS_IDEATE", "1").strip().lower() in ("1", "true", "on", "yes")


def _ideate_every_n() -> int:
    """How often an idle wander IDEATES instead of recalling (a host-timed scaffold, like the idle-tick clock): every
    Nth wander drives the novel blend. Default 3 (occasional). `BRAIN_CONTINUOUS_IDEATE_EVERY` overrides (1 = every)."""
    try:
        return max(1, int(os.environ.get("BRAIN_CONTINUOUS_IDEATE_EVERY", "3")))
    except Exception:
        return 3


def _is_ideation_tick(cache_key) -> bool:
    """True on every Nth wander for this session (the host-timed ideation cadence). Advances a per-session counter;
    called ONLY when ideation_enabled() (short-circuited), so the OFF path never touches this state."""
    c = _IDEATE_TICK.get(cache_key, 0) + 1
    _IDEATE_TICK[cache_key] = c
    return (c % _ideate_every_n()) == 0


def _ideation_blend_settle(seed: int, n_mem: int, iA: int, iB: int) -> dict | None:
    """Drive a sparse associative-attractor with a BLENDED cue of two stored basins -> settle into a NOVEL
    recombination. Reuse-by-import of the GO de-risk's mechanism (Tsodyks-Feigelman sparse-Hopfield + the
    ca3_ff_inhib MEAN+std dynamic-threshold settle). DECLARED SCAFFOLD: a fast standalone numpy attractor is the
    de-risked stand-in for the on-substrate CA3 blend. Returns the settled-state metrics, or None on any failure."""
    try:
        import numpy as _np
        from research.runners._generative_attractor_wander_derisk import (
            _sparse_pattern, _train_weights, _threshold_settle, _overlap)
    except Exception:
        return None
    if n_mem < 2 or not (0 <= iA < n_mem) or not (0 <= iB < n_mem) or iA == iB:
        return None
    n, k, c = _IDEATE_N, _IDEATE_K, _IDEATE_THRESH_C
    # a deterministic ideation stream disjoint from the wander RNG (the patterns REPRESENT the fixed basins; WHICH
    # two are blended shifts with the curiosity gains, so trains-of-thought move while the substrate stays fixed).
    rng = _np.random.default_rng(int(seed) * 100003 + 17)
    pats = [_sparse_pattern(rng, n, k) for _ in range(n_mem)]
    W, a = _train_weights(pats, n)
    idxA = _np.flatnonzero(pats[iA]).copy(); idxB = _np.flatnonzero(pats[iB]).copy()
    rng.shuffle(idxA); rng.shuffle(idxB)
    hk = k // 2
    cue = _np.zeros(n); cue[idxA[:hk]] = 1.0; cue[idxB[:hk]] = 1.0
    settled, fixed, _ = _threshold_settle(W, a, cue, n_iters=12, c=c)
    ov = [_overlap(settled, p) for p in pats]
    novelty = max(ov)
    balance = min(ov[iA], ov[iB])
    others = max([ov[m] for m in range(n_mem) if m not in (iA, iB)], default=0.0)
    return {"novelty_max_overlap": round(float(novelty), 3),
            "blend_balance": round(float(balance), 3),
            "blend_vs_other": round(float(others), 3),
            "fixed_point": bool(fixed)}


def _ideation_wander(cache_key, organ) -> dict | None:
    """OCCASIONAL between-turn IDEATION: drive a BLENDED cue of the TWO most curiosity-active basins into the
    attractor -> it settles into a NOVEL recombination that was never stored (the creativity/novelty rung). Returns
    a record TAGGED as a novel idea/association (never a recalled fact), or None if the two-source blend did NOT
    settle into a genuine novel balanced state (honest: no novel idea surfaced this tick). SELECTION of the two
    source basins rides the organ's spiking curiosity gains; the novelty rides the attractor DYNAMICS."""
    agents = getattr(organ, "agents", None)
    if not agents or len(agents) < 2:
        return None
    n_mem = len(agents)
    gains = getattr(organ, "gains_on", None)
    if gains and len(gains) == n_mem:
        order = sorted(range(n_mem), key=lambda i: (-float(gains[i]), i))  # the two most curiosity-active basins
    else:
        order = list(range(n_mem))
    iA, iB = order[0], order[1]
    res = _ideation_blend_settle(getattr(organ, "seed", 42), n_mem, iA, iB)
    if res is None:
        return None
    novel = bool(res["fixed_point"]
                 and res["novelty_max_overlap"] < IDEATE_NOVELTY_MAX
                 and res["blend_balance"] > IDEATE_BALANCE_MIN
                 and (res["blend_balance"] - res["blend_vs_other"]) > IDEATE_BLEND_MARGIN)
    if not novel:
        return None
    return {
        "kind": "novel-association",   # the HONESTY TAG — an IDEA, not a recalled fact
        "is_fact": False,
        "sources": [agents[iA], agents[iB]],
        "novelty_max_overlap": res["novelty_max_overlap"],
        "blend_balance": res["blend_balance"],
        "blend_vs_other": res["blend_vs_other"],
        "fixed_point": res["fixed_point"],
    }


def recent_ideation(cache_key) -> dict | None:
    """The most recent between-turn IDEATION (a NOVEL blended association the attractor settled into while idle),
    TAGGED as an idea/association — NOT a recalled fact. CONSUMES on read (surfaces exactly once, on the next live
    turn), a DISTINCT channel from recent_wander() (recalled concepts). Returns None if continuous or ideation is
    off, or none is pending. The caller frames it as 'a thought that occurred to me'; it NEVER enters the
    recall/abstain moat as an assertion. Off (BRAIN_CONTINUOUS_IDEATE unset) -> returns None without touching state."""
    if not (continuous_enabled() and ideation_enabled()):
        return None
    lst = _INNER_LIFE.get(cache_key)
    if not lst:
        return None
    for rec in reversed(lst):
        idea = rec.get("ideation")
        if idea and not rec.get("_ideation_consumed"):
            rec["_ideation_consumed"] = True   # consume -> surfaces on exactly the next turn, not every turn after
            return dict(idea)
    return None


def tick_session(cache_key, session_mood: dict, affect_organ, now: float | None = None,
                 selfinit_organ=None) -> dict | None:
    """One idle tick for ONE session: (a) FEELING keeps evolving — relax the appraisal + RE-READ the spiking affect
    ladder; (b) a THOUGHT wanders — if a self-initiation organ is given, its curiosity-biased spiking selection
    surfaces a concept (the mind drifting to something while idle). Both are recorded to the inner-life.

    Returns the tick record (or None if the session has no mood yet). Pure w.r.t. the reply path — it only mutates
    this session's mood EMA + the inner-life log; a live turn re-appraises from the message anyway."""
    now = time.time() if now is None else now
    m = session_mood.get(cache_key)
    if m is None:
        return None
    v0, a0 = float(m.get("valence", 0.0)), float(m.get("arousal", 0.0))
    # (a) FEELING: homeostatic relaxation of the felt state toward baseline, then the spiking read at the new point
    v1 = NEUTRAL + (v0 - NEUTRAL) * RELAX
    a1 = a0 * RELAX
    m["valence"], m["arousal"] = v1, a1
    session_mood[cache_key] = m
    try:
        diff = float(affect_organ.read_differential(v1, lesion=False)["differential"])
    except Exception:
        diff = None
    trend = "toward neutral" if abs(v1) < abs(v0) else "steady"
    note = "idle: felt state relaxing %s (was %+.2f, now %+.2f)" % (trend, v0, v1)
    # (b) THOUGHT: a curiosity-biased spiking selection surfaces a wandered concept (a thought drifting while idle).
    # OCCASIONALLY (default-OFF `BRAIN_CONTINUOUS_IDEATE`, every Nth wander) the wander instead GENERATES a NOVEL
    # blended concept — the ideation/creativity rung — surfaced on a distinct, honestly-flagged channel. When
    # ideation is off, or this is not an ideation tick, or the blend did not settle novel, the DEFAULT recall wander
    # runs EXACTLY as today (byte-identical): the `ideation` key is then absent from `rec`.
    wandered = None
    ideation = None
    if selfinit_organ is not None:
        try:
            if ideation_enabled() and _is_ideation_tick(cache_key):
                try:
                    selfinit_organ._ensure_mouth()   # need agents + curiosity gains (idempotent, cheap)
                except Exception:
                    pass
                ideation = _ideation_wander(cache_key, selfinit_organ)
            if ideation is not None:
                sA, sB = ideation["sources"][0], ideation["sources"][1]
                note += ("; an idea occurred — a novel association linked ‘%s’ and ‘%s’ (a thought, not a "
                         "recalled fact)" % (sA, sB))
            else:
                # DEFAULT WANDER PATH — UNCHANGED (byte-identical when ideation is off / not an ideation tick /
                # the blend didn't surface): today's single-basin curiosity-biased recall selection.
                ior = _wander_ior_enabled()
                if ior:
                    _pre_wander_ior(cache_key, selfinit_organ)   # bias away from recently-visited basins
                out = selfinit_organ.speak(lesion=False)
                wandered = out.get("concept")
                if ior and wandered:
                    _post_wander_ior(cache_key, selfinit_organ, wandered)  # fatigue the just-won basin
                if wandered:
                    note += "; a thought wandered to ‘%s’" % wandered
        except Exception:
            wandered = None
    rec = {"t": now, "valence": v1, "arousal": a1, "differential": diff, "wandered": wandered, "note": note}
    if ideation is not None:   # additive: absent when ideation is off -> rec is byte-identical to today
        rec["ideation"] = ideation
    lst = _INNER_LIFE.setdefault(cache_key, [])
    lst.append(rec)
    if len(lst) > _INNER_LIFE_MAX:
        del lst[:-_INNER_LIFE_MAX]
    return rec


def tick_idle_sessions(session_mood: dict, affect_organ_getter, now: float | None = None,
                       selfinit_getter=None, episodic_getter=None, chat_getter=None) -> int:
    """Run one tick over every session that is IDLE (no request for >= IDLE_SEC). Returns #sessions ticked.

    Called by the server's background loop. Skips sessions mid-conversation (raced writes) and any with no mood yet.
    `selfinit_getter(cache_key)` (optional) supplies that session's self-initiation organ for the thought-wander.
    `episodic_getter(cache_key)` (optional) supplies that session's ALREADY-BUILT episodic organ for the D5
    learn-through-use consolidation (default-OFF behind `BRAIN_D5_CONSOLIDATE`; never builds an organ just to tick).
    `chat_getter(cache_key)` (optional) supplies that session's ALREADY-BUILT chat for the DA-encoding substrate
    homeostasis (Turrigiano synaptic-scaling) consolidation pass; None-returning getter -> that step is skipped."""
    if not continuous_enabled():
        return 0
    now = time.time() if now is None else now
    n = 0
    _d5_on = d5_consolidate_enabled()
    for cache_key in list(session_mood.keys()):
        last = _LAST_REQUEST.get(cache_key)
        if last is not None and (now - last) < IDLE_SEC:
            continue  # still active -> don't tick mid-turn
        try:
            organ = affect_organ_getter()
            # The cheap mood-relax runs every tick; the EXPENSIVE self-init wander only while this session has
            # budget left for this idle period (refilled by mark_request each turn). Once drained, the mind keeps
            # relaxing but stops wandering -> a truly idle server does not peg the GPU with endless ~55s wanders.
            siorg = None
            if selfinit_getter is not None and _WANDER_BUDGET.get(cache_key, 0) > 0:
                try:
                    siorg = selfinit_getter(cache_key)
                except Exception:
                    siorg = None
            rec = tick_session(cache_key, session_mood, organ, now, selfinit_organ=siorg)
            if rec is not None:
                n += 1
                if rec.get("wandered"):  # a heavy wander actually fired -> spend one unit of budget
                    _WANDER_BUDGET[cache_key] = max(0, _WANDER_BUDGET.get(cache_key, 0) - 1)
            # D5 LEARN-THROUGH-USE: consolidate the memory this session USED, bounded by the per-recall budget.
            # Runs independently of the mood tick (a recalled-but-mood-less session still consolidates), and only
            # while a genuine recall has armed the budget for this idle period -> a truly idle server does not
            # re-consolidate forever. Never builds an organ (episodic_getter returns None if none exists yet).
            if _d5_on and episodic_getter is not None and _D5_BUDGET.get(cache_key, 0) > 0 \
                    and _RECALLED_TOPIC.get(cache_key):
                try:
                    eorg = episodic_getter(cache_key)
                    if eorg is not None and consolidate_used_memory(cache_key, eorg) is not None:
                        _D5_BUDGET[cache_key] = max(0, _D5_BUDGET.get(cache_key, 0) - 1)
                except Exception:
                    # consolidate_used_memory already rolled the store back + drained the topic on failure; LOG so a
                    # mid-consolidation crash is VISIBLE (never silently swallowed), then move on to the next session.
                    import logging as _lg
                    _lg.getLogger(__name__).warning(
                        "D5 consolidation failed for %s (persistent store rolled back)", cache_key, exc_info=True)
            # DA-ENCODING SUBSTRATE HOMEOSTASIS: run the Turrigiano synaptic-scaling consolidation pass on this idle
            # session's live composer store when new facts were taught since the last pass. Self-gates on the DA-encoding
            # faculty (no-op under BRAIN_DA_ENCODING=0), so this is byte-identical to HEAD when the faculty is off. It
            # consolidates a batch of freshly-encoded facts, offline — the between-turn cadence Turrigiano scaling wants.
            if chat_getter is not None:
                try:
                    _chat = chat_getter(cache_key)
                    if _chat is not None:
                        consolidate_substrate_homeostasis(cache_key, _chat)
                except Exception:
                    import logging as _lg
                    _lg.getLogger(__name__).warning(
                        "DA-encoding substrate homeostasis tick failed for %s", cache_key, exc_info=True)
        except Exception:
            continue
    return n
