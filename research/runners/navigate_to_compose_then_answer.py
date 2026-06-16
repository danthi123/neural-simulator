"""NAVIGATE-TO-COMPOSE-THEN-ANSWER — the COMPOSITIONAL successor of the (B) navigate-to-see milestone, ONE brain.

This upgrades `research/runners/navigate_to_see_then_answer.py` from RECALL ("I saw the apple") to COMPOSE ("dog
chase <perceived cat>") on ONE `SimulationBridge`. The scoping that gated this build:
`research/findings/2026-06-16-step3-integration-build-scoping.md` (§2 the wiring + reuse points, §4 the four
anti-cheats, §5 the honest scope, §6 the ranked task list — this is T3, the behavioral runner + single-seed GPU
smoke). Every piece already exists and is separately validated; this JOINS them in-episode.

────────────────────────────────────────────────────────────────────────────────────────────────────────────
TERMS (defined once, owner standing requirement — no undefined acronyms)
────────────────────────────────────────────────────────────────────────────────────────────────────────────
- merged bridge   : the single `SimulationBridge` from `nav_conv_merged_bridge.build_merged_nav_conv_bridge` holding
                    the navigation basal-ganglia cascade (the BODY) + the conversational parser + the dlPFC dialogue
                    planner + the co-resident resonate-and-fire (RF) composer `rf` slice + (STEP-3, new) the bare
                    `cortex_it` perception region.
- cortex_it       : the navigation perception region (the ventral "what"-stream object-identity ensembles). "The
                    agent sees object X" = X's distinct cortex_it sub-ensemble fires. The environment RENDERS the
                    object by driving X's orthogonal band of cortex_it (a legitimate sensory render — the body/world).
- grounded code   : a composer concept code that is a deterministic function of an object's LIVE cortex_it spiking
                    rate (vs a free random code). `composer.concepts[o] = angle(M @ rate)` where M is the de-risked
                    fixed complex projection. The percept becomes a phasor the composer algebra can bind.
- rate-vs-phasor wall : the perceived object is a RATE ensemble (Izhikevich cortex_it firing); the composer consumes
                    PHASOR codes (phases in [0,1)^D on RF neurons). The grounded-code map M dissolves the wall by
                    making the percept a phasor (the de-risk `2026-06-16-step3-live-cortex-grounded-compose-cheap-first.md`).
- compose / bind / unbind : the composer's fixed FHRR (Fourier Holographic Reduced Representation) vector-symbolic
                    algebra on the `rf` slice: bind=(role,filler)->composite (complex product), bundle=sum bound
                    pairs into a fact, unbind=recover a filler (multiply by the conjugate), cleanup=nearest concept.
- held-out fact   : a (perceived-object, role) combination NEVER composed in any setup step. COMPOSE generalizes to
                    it (>> a memorization-floor recall baseline); RECALL cannot. This is the control that separates
                    COMPOSE from the (B) milestone's RECALL.
- no-confab moat  : the composer ABSTAINS (returns None) on an unstored (agent, action) / (action, patient) query —
                    the no-confabulation guarantee. NEVER weakened to make a number look better (a breach = HARD STOP).

────────────────────────────────────────────────────────────────────────────────────────────────────────────
THE TASK (scoping §2.1, §5): navigate -> perceive+ground in-episode -> compose a novel perceived-object fact -> answer.
────────────────────────────────────────────────────────────────────────────────────────────────────────────
The agent is on a grid with OBJECTS at >=2 cells. It NAVIGATES a route (the BG cascade selects each move; the body
steps; OU noise runs — a LIVE episode). As the agent ARRIVES at an object's cell, the environment renders that
object's identity into cortex_it, the agent reads the LIVE cortex_it spiking rate OFF THE MERGED BRIDGE and GROUNDS
it into the co-resident composer's codebook (`composer.concepts[o] = M @ rate`) — IN-EPISODE. AFTER the episode, the
agent COMPOSES novel (held-out) facts over the perceived objects on the `rf` slice (`_encode`/`store`), and answers
`what_does`/`who_does` about them + ABSTAINS on unstored queries.

THE ANTI-CHEATS (scoping §4 — all four):
  1. LESION the grounded-code map  : restore the perceived objects' codes to their random codes -> compose collapses
                                     (the unbind no longer recovers the perceived object). Proves the compose rides
                                     the live-percept grounding, not a structural bias.
  2. HELD-OUT novel fact (compose != recall) : a never-composed (perceived-object, role) pairing unbinds correctly
                                     >> chance AND >> a memorization-floor recall baseline (by >= 0.30). THE control
                                     that separates COMPOSE from RECALL.
  3. PROVENANCE + co-residence     : the filler code in any composed fact is the percept-derived code read from the
                                     merged bridge's cortex_it firing (never a host-set "apple" phasor); the compose
                                     actually ran on the merged `rf` slice (composer._merged is the merged bridge,
                                     cp_rf_w_re is not None after a store).
  4. The no-confab MOAT stays intact: every unstored query returns None (abstain). NEVER weakened (a breach = HARD STOP).

────────────────────────────────────────────────────────────────────────────────────────────────────────────
BRAIN-BASED-ONLY (owner standing bar): everything between sensation and action is neurons/synapses. Host code is
legitimate ONLY for (1) the environment — the grid, the agent position, object placement, and RENDERING an object's
identity into cortex_it on arrival (a sensory render) — and (2) the body — moving the agent based on which sel/motor
pool fires. The grounded code is a LIVE spiking rate read (cp_firing_states over the read window) — the substrate's
own response to the rendered percept, not a host stand-in. The bind/unbind/bundle is the validated fixed FHRR
primitive on the co-resident `rf` slice. The grounded-code map M is host arithmetic on the substrate's own live rate
(the scoping's recommended Option-a shared-grounded-code projection). The compose + the abstention are the brain's
(the composer's) job, on its neurons.

HONEST SCOPE (scoping §5): this COMPOSES flat-distinct object facts via shared grounded codes; it does NOT transfer
knowledge from "dog" to "cat" because their codes are similar (the dendritic/PPMI generalization frontier is the
deferred step-3 fork). It uses the FIXED composer algebra (not a learned cortical bind). The grounding is for OBJECTS
only (the perceived fillers); abstract relata (verbs like "chase") use the composer's own concept codes (the composer's
own honest limit — verbs are not perceived). It is the genuine "the agent composes what it perceived," in-episode, on
one bridge — a consolidation of EXISTING separately-validated capabilities, not a new one.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim.backend import get_backend, to_host

# the merged bridge + the co-resident RF composer (STEP 2b) + the additive STEP-3 perception + WTA kwargs.
from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge, MergedRFComposer
# the perceived-object vocabulary + the grounding constants (reuse-by-import, VERBATIM with the cheap-first smoke).
from research.runners.funcint_perception_to_memory_probe import OBJECT_WORDS, N_OBJECTS
# the de-risked grounded-code map (live rate read -> fixed complex projection -> phasor), VERBATIM.
from research.runners._step3_grounded_codes_production_composer_derisk import (
    read_cortex_it_rate, _projection, grounded_phases,
)
# the navsee LIVE episode loop primitives (reuse-by-import): the neural move selection, the greedy steer, the grid
# delta, and the deterministic object layout. The arrival handler is REPLACED (ground instead of engram-tag), but the
# navigation traversal + selection are byte-for-byte the validated (B) milestone's.
from research.runners.navigate_to_see_then_answer import (
    _cascade_select_move, _steer_toward, default_object_layout, ACTION_DELTA, _settle,
)
from research.runners.g11_bg_runner import ACTION_NAMES


# ── constants ────────────────────────────────────────────────────────────────────────────────────────────
D = 128                     # composer dim (the merged-bridge production tier; rf_D=D so 7*rf_D covers a 3-role bundle)
# the verbs (actions are NOT perceived — they use the composer's own concept codes). Used for the moat store/query.
ACTIONS = ["chase", "near"]
# the no-confab moat: unstored (agent, action) queries that MUST abstain (None). None is a stored (agent, action) pair.
MOAT_ABSENT = [("river", "chase"), ("apple", "near")]


# ── the merged nav-body + perception(cortex_it) + co-resident composer bridge ────────────────────────────────
def build_compose_bridge(seed: int = 42, with_body: bool = True, co_resident_generalization: bool = False):
    """Build the merged nav+conv bridge WITH the bare cortex_it perception region + the co-resident RF composer, and
    construct the navsee-style navigation handles (readout/cortex/tonic) against it + the fixed grounded-code map.

    with_body=True : the full nav cascade selects moves (spiking-WTA sel_X). with_body=False is the ISOLATED-PERCEPTION
                     control: no sel_X/cortex steer -> the cascade cannot move the agent, so it never ARRIVES at an
                     object cell -> nothing is grounded in-episode -> nothing composes.
    co_resident_generalization=True (STAGE 2, additive default-off): ALSO append the Stage-1 generalization stack
                     (structured-perception gen_perception -> NMDA gen_concept -> gen_fact + the trained-then-frozen
                     rate-Hebbian convergence) so the SAME live bridge can generalize a novel similar perceived object
                     to its category. Default False keeps the validated compose-perceived 6-seed path byte-identical
                     (the gen regions are appended LAST, after rf + cortex_it, so all nav/parser/dlPFC/rf/cortex_it
                     index bases are unchanged). handles["gen"] is present only when True.

    Returns (bridge, composer, handles, proj).
    """
    xp, _ = get_backend()
    vocab = list(OBJECT_WORDS) + ACTIONS
    # enable_spiking_wta_readout=with_body: the body gets the validated sel_X selection (Step-1 de-risk: motor_X also
    # selects, but sel_X has the higher SNR margin navsee was validated with). co_resident_rf + co_resident_perception
    # bring the composer `rf` slice + the bare cortex_it perception region onto the one bridge (both appended LAST so
    # the nav/parser/dlPFC index bases are byte-unchanged; cortex_it is the very last region).
    bridge, handles = build_merged_nav_conv_bridge(
        seed=seed, vocab=vocab, n_cortex=100, co_resident_rf=True, rf_D=D,
        co_resident_perception=True, enable_spiking_wta_readout=with_body,
        co_resident_generalization=co_resident_generalization)
    rm = bridge.region_manager
    region_names = set(rm.region_indices_dict())

    # the co-resident composer (STEP 2b): RF binding ops run on the merged bridge's own `rf` slice. Same seed + vocab.
    composer = MergedRFComposer(bridge, handles["rf_base"], handles["rf_size"],
                                seed=seed, D=D, vocab=vocab, period=200)

    # the fixed grounded-code map M (live cortex_it rate -> composer phases), the de-risk's exact construction.
    it_indices = np.asarray(list(rm.indices("cortex_it")), dtype=np.int64)
    proj = _projection(D, int(it_indices.size), seed)

    h = {
        "seed": int(seed),
        "with_body": bool(with_body),
        "it_indices": xp.asarray(it_indices),
        "it_indices_host": it_indices,
        "rf_base": int(handles["rf_base"]),
        "rf_size": int(handles["rf_size"]),
        "n_neurons": int(bridge.core_config.num_neurons),
        "rf_last_index": int(np.asarray(list(rm.indices("rf")), dtype=np.int64)[-1]),
        "grounded_objects": [],          # the objects grounded in-episode (provenance: filled by _perceive_and_ground)
    }

    # STAGE 2: surface the generalization handles (the build returns them in `handles["gen"]` only when
    # co_resident_generalization=True) so the unified-episode runner can run the H5/H6 generalization check on this
    # SAME live bridge. Absent (and h has no "gen" key) for the default compose-perceived path -> byte-preserved.
    if co_resident_generalization and isinstance(handles, dict) and "gen" in handles:
        h["gen"] = handles["gen"]

    # the navigation body's readout + tonic-pacemaker handles (MIRRORS navsee build_navsee_bridge lines 280-305).
    if with_body and all(f"sel_{a}" in region_names for a in ACTION_NAMES):
        h["readout_region"] = "sel"
        h["readout_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"sel_{a}")), dtype=np.int64))
                            for a in ACTION_NAMES}
    elif with_body and all(f"motor_{a}" in region_names for a in ACTION_NAMES):
        h["readout_region"] = "motor"
        h["readout_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"motor_{a}")), dtype=np.int64))
                            for a in ACTION_NAMES}
    if with_body:
        h["cortex_idx"] = {a: xp.asarray(np.asarray(list(rm.indices(f"cortex_{a}")), dtype=np.int64))
                           for a in ACTION_NAMES}

        def _ridx(name):
            return xp.asarray(np.asarray(list(rm.indices(name)), dtype=np.int64)) if name in region_names else None
        h["cascade_tonic"] = []
        for a in ACTION_NAMES:
            for name, pa in ((f"gpe_{a}", 150.0), (f"gpe_arky_{a}", 120.0), (f"gpi_{a}", 110.0),
                             (f"thal_{a}", 300.0)):
                ii = _ridx(name)
                if ii is not None:
                    h["cascade_tonic"].append((ii, float(pa)))
        for name, pa in (("stn", 150.0), ("snc", 150.0)):
            ii = _ridx(name)
            if ii is not None:
                h["cascade_tonic"].append((ii, float(pa)))
    return bridge, composer, h, proj


# ── the arrival handler: render the percept, read the live rate, GROUND it into the composer codebook ─────────
def _perceive_and_ground(bridge, composer, handles, proj, obj_word):
    """LIVE perception+ground (the STEP-3 part, in-episode): the agent has ARRIVED at object `obj_word`'s cell. Read
    the object's LIVE cortex_it spiking rate OFF THE MERGED BRIDGE (the de-risk `read_cortex_it_rate` renders the
    object's orthogonal cortex_it band — the sensory render — and accumulates its firing under OU), map -> phases via
    the fixed projection, and SET `composer.concepts[obj_word] = phases` (the grounded code). The percept becomes a
    phasor the composer algebra binds. OU is enabled for the read (the resting nav config rests OU-off) so the
    grounded code reflects the actual noisy co-resident episode condition (the scoping's stated #1 risk).

    PROVENANCE: the only write into a perceived object's code is `composer.concepts[o] = M @ (the live cortex_it
    rate)`; no host code copies a labeled "apple" phasor in. Returns the per-neuron rate vector (for the provenance
    assert that the grounded code derives from the live read)."""
    obj_idx = OBJECT_WORDS.index(obj_word)
    cc = bridge.core_config
    prev_ou, prev_std = cc.enable_ou_process, cc.ou_std_current_pA
    cc.enable_ou_process, cc.ou_std_current_pA = True, 20.0
    try:
        rate = read_cortex_it_rate(bridge, handles["it_indices"], obj_idx)   # LIVE rate read OFF THE MERGED BRIDGE
        phases = grounded_phases(rate, proj)                                 # phases in [0,1)^D (the grounded code)
        composer.concepts[obj_word] = phases
    finally:
        cc.enable_ou_process, cc.ou_std_current_pA = prev_ou, prev_std
    if obj_word not in handles["grounded_objects"]:
        handles["grounded_objects"].append(obj_word)
    # capture the FIRST grounded object's (rate, phases) for the provenance assert (the grounded code == the live
    # read), so provenance never needs a second live read that would consume different OU noise.
    if "provenance_sample" not in handles:
        handles["provenance_sample"] = {"obj": obj_word, "rate": np.asarray(rate).copy(), "phases": phases.copy()}
    return rate, phases


# ── the live navigation episode: traverse a route, perceiving + grounding objects on the path ────────────────
def run_compose_episode(bridge, composer, handles, proj, object_layout, start_pos, route_waypoints, perceive=True):
    """Run ONE live navigation episode: the agent starts at `start_pos` and navigates toward each waypoint; each step
    the BG cascade selects the move (neural) and the body steps. When the agent ARRIVES at an object's cell AND
    `perceive`, the object is rendered into cortex_it and GROUNDED into the composer codebook IN-EPISODE.

    Returns the episode trace (the grounded objects + per-step moves). ISOLATED-PERCEPTION (with_body=False): no
    cascade move -> the agent never traverses -> never arrives -> nothing grounded."""
    pos = tuple(int(c) for c in start_pos)
    grounded = []
    moves = []
    visited_cells = {pos}

    # ISOLATED-PERCEPTION: no body -> the agent cannot move, so it never reaches any object cell.
    if not handles.get("with_body", True) or handles.get("readout_region") is None:
        _settle(bridge, 30)
        return {"grounded": [], "moves": [], "path": [pos], "reached_all_waypoints": False}

    # ground at the START cell if an object sits there.
    if perceive and pos in object_layout:
        obj = object_layout[pos]
        _perceive_and_ground(bridge, composer, handles, proj, obj)
        grounded.append(obj)

    path = [pos]
    max_steps = 64
    step = 0
    for target in route_waypoints:
        while pos != tuple(target) and step < max_steps:
            steer = _steer_toward(pos, tuple(target))
            chosen, _counts = _cascade_select_move(bridge, handles, steer)
            step += 1
            move = chosen if chosen is not None else None     # a silent/tied cascade decision does not move the agent
            moves.append({"steer": steer, "chosen": chosen})
            if move is not None:
                dx, dy = ACTION_DELTA[move]
                pos = (pos[0] + dx, pos[1] + dy)
            path.append(pos)
            # on ARRIVAL at a fresh object cell, perceive + ground from the live episode.
            if perceive and pos in object_layout and pos not in visited_cells:
                obj = object_layout[pos]
                _perceive_and_ground(bridge, composer, handles, proj, obj)
                if obj not in grounded:
                    grounded.append(obj)
            visited_cells.add(pos)
    return {
        "grounded": grounded, "moves": moves, "path": path,
        "reached_all_waypoints": all(tuple(t) in path for t in route_waypoints),
    }


# ── held-out compose vs memorization-floor (anti-cheat 2: compose != recall) ──────────────────────────────────
def _held_out_split(grounded_objs, seed):
    """Split the ordered distinct-object (agent, patient) pairs over the IN-EPISODE GROUNDED objects into MEMORIZED
    vs HELD-OUT (leakage-free: held-out pairs are NEVER composed in the memorization-floor store)."""
    names = list(grounded_objs)
    n = len(names)
    if n < 2:
        return names, [], []
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    rng = np.random.default_rng(seed * 99 + 7)
    rng.shuffle(pairs)
    return names, pairs[:len(pairs) // 2], pairs[len(pairs) // 2:]


def _held_out_compose_score(composer, grounded_objs, seed):
    """The compose-vs-recall anti-cheat: for each HELD-OUT (never-composed) fact, COMPOSE `_encode({agent,patient})`
    on the merged `rf` slice and `unbind` each role -> the perceived object (clean compose). Score a
    memorization-floor recall baseline on the SAME held-out facts (nearest stored composite -> its remembered filler).
    COMPOSE generalizes to never-composed pairings (>> the floor); a recall-only system scores at the floor.

    Returns (clean, floor, n_held_out, held_composites) where held_composites = [(comp_phases, ai, bi)] are the
    held-out composites COMPOSED UNDER THE CURRENT (grounded) codebook — captured so the lesion can re-cleanup them
    against a lesioned codebook (the navsee 'cut-after-encode, re-query' lesion)."""
    names, memorized, held_out = _held_out_split(grounded_objs, seed)
    if not held_out:
        return 0.0, 0.0, 0, []

    # the memorization floor (a recall-only baseline): nearest stored composite -> its remembered filler.
    mem_store = [(composer._encode({"agent": names[ai], "patient": names[bi]}), ai, bi) for (ai, bi) in memorized]

    def _mem_recall(comp, role):
        best, bk = -1.0, 0
        for k, (f, a, b) in enumerate(mem_store):
            c = float(np.mean(np.cos(2.0 * np.pi * (f - comp))))    # the composer's phase-cosine similarity
            if c > best:
                best, bk = c, k
        return mem_store[bk][1] if role == "agent" else mem_store[bk][2]

    clean_ok = mem_ok = 0
    held_composites = []
    for (ai, bi) in held_out:
        comp = composer._encode({"agent": names[ai], "patient": names[bi]})    # COMPOSE on the merged rf slice
        held_composites.append((comp, ai, bi))
        ra = composer.unbind(comp, "agent")
        rb = composer.unbind(comp, "patient")
        clean_ok += int(ra == names[ai]) + int(rb == names[bi])
        mem_ok += int(names[_mem_recall(comp, "agent")] == names[ai]) + int(names[_mem_recall(comp, "patient")] == names[bi])
    clean = clean_ok / (2 * len(held_out))
    floor = mem_ok / (2 * len(held_out))
    return clean, floor, len(held_out), held_composites


def _lesion_recompose_score(composer, grounded_objs, held_composites):
    """Anti-cheat 1 (lesion), the navsee 'cut-after-encode, re-query' form: the held-out composites were COMPOSED
    under the GROUNDED codebook (so their unbind recovers a phasor near the grounded filler). After the codebook is
    LESIONED (the grounded objects' `composer.concepts[o]` restored to random codes), re-cleanup those SAME stored
    composites: the recovered phasor still carries the grounded filler's phase, but the cleanup now compares against
    the RANDOM codebook -> the grounded filler no longer matches its codebook entry -> the cleanup picks the wrong
    object -> compose collapses. This proves the COMPOSE rode the live-percept grounding (the consistency between the
    bound filler and the codebook), not a code-agnostic algebra trick. Returns (lesion_clean, n)."""
    names = list(grounded_objs)
    if not held_composites:
        return 0.0, 0
    ok = 0
    for (comp, ai, bi) in held_composites:
        ra = composer.unbind(comp, "agent")     # cleanup vs the now-LESIONED codebook
        rb = composer.unbind(comp, "patient")
        ok += int(ra == names[ai]) + int(rb == names[bi])
    return ok / (2 * len(held_composites)), len(held_composites)


# ── the no-confab moat (anti-cheat 4) ─────────────────────────────────────────────────────────────────────
def _moat_check(composer, grounded_objs):
    """Store a couple of perceived-object facts, then assert the no-confab moat: an unstored (agent, action) query
    returns None (abstain), AND a stored fact DOES retrieve (so the moat isn't trivially abstaining on everything).
    Returns (moat_ok, moat_tot, pos_recall, stored_facts)."""
    names = list(grounded_objs)
    stored = []
    if len(names) >= 2:
        composer.store(names[0], ACTIONS[0], names[1]); stored.append((names[0], ACTIONS[0], names[1]))
    if len(names) >= 4:
        composer.store(names[2], ACTIONS[1], names[3]); stored.append((names[2], ACTIONS[1], names[3]))
    elif len(names) >= 3:
        composer.store(names[2], ACTIONS[1], names[0]); stored.append((names[2], ACTIONS[1], names[0]))
    # the moat: every absent (agent, action) over the GROUNDED objects must abstain.
    absent = [(a, v) for (a, v) in MOAT_ABSENT if a in names] or [(names[-1], ACTIONS[0])]
    # ensure 'absent' pairs are genuinely unstored.
    stored_av = {(a, v) for (a, v, p) in stored}
    absent = [(a, v) for (a, v) in absent if (a, v) not in stored_av]
    if not absent:                                          # fall back to a guaranteed-unstored pair
        absent = [(names[1], ACTIONS[1])] if (names[1], ACTIONS[1]) not in stored_av else []
    moat_ok = sum(int(composer.query_patient(a, v) is None) for (a, v) in absent)
    pos = int(composer.query_patient(stored[0][0], stored[0][1]) == stored[0][2]) if stored else 0
    return moat_ok, len(absent), pos, stored, absent


# ── provenance + co-residence (anti-cheat 3) ──────────────────────────────────────────────────────────────
def _provenance_check(bridge, composer, handles, ground_rate, ground_phases, obj_word):
    """Anti-cheat 3: the grounded code is the LIVE perception (not a host copy), and the compose ran on the merged
    `rf` slice. Asserts: (i) the grounded code in the composer's codebook == the fixed projection of the live rate
    read (structural provenance: composer.concepts[o] == grounded_phases(rate, proj)); (ii) the composer is the
    co-resident MergedRFComposer bound to the merged bridge (composer._merged is bridge); (iii) after a store,
    cp_rf_w_re is not None (the bind actually ran on the bridge's RF complex synapses). Raises on violation."""
    # (i) the grounded code derives from the live rate read (not a host-set "apple" phasor).
    assert np.allclose(composer.concepts[obj_word], ground_phases), \
        f"FAIL provenance: composer.concepts[{obj_word!r}] is not the live-rate-derived grounded code"
    assert float(np.asarray(ground_rate).sum()) > 0.0, \
        f"FAIL provenance: the cortex_it live rate for {obj_word!r} was all-zero (no perception)"
    # (ii) the co-resident composer is bound to the merged bridge (not a silent standalone fallback).
    assert composer._merged is bridge, "FAIL co-residence: the composer is not bound to the merged bridge"
    # (iii) the bind actually ran on the bridge's RF complex synapses (cp_rf_w_re allocated after a store).
    rf_w_re = getattr(bridge, "cp_rf_w_re", None)
    assert rf_w_re is not None, "FAIL co-residence: cp_rf_w_re is None after a store (the bind did not run on the bridge)"
    return {
        "grounded_code_is_live_rate_projection": True,
        "composer_bound_to_merged_bridge": True,
        "rf_complex_weights_allocated_after_store": True,
        "perception_side_write": "cortex_it <- orthogonal_drive_pattern(object) DURING arrival [sensory render]",
        "ground_write": "composer.concepts[o] = angle(M @ live_cortex_it_rate) [the only write into the percept code]",
    }


# ── one seed: the COUPLED compose episode + the LESION + ISOLATED-PERCEPTION controls ─────────────────────────
def run_seed(seed):
    xp, backend = get_backend()
    print(f"\n[navcompose] ===== seed {seed} (backend={backend}) =====", flush=True)
    chance = 1.0 / N_OBJECTS

    layout = default_object_layout(seed)
    start_pos = (0, 2)
    sorted_cells = sorted(layout.keys(), key=lambda c: c[0])
    route_waypoints = [sorted_cells[1], sorted_cells[2]]      # walk to the 2nd then 3rd object cell (encounter a subset)
    print(f"[navcompose] object layout: {{ {', '.join(f'{c}:{w}' for c, w in sorted(layout.items()))} }}", flush=True)
    print(f"[navcompose] start={start_pos} route_waypoints={route_waypoints}", flush=True)

    # --- COUPLED: navigate + perceive+ground the encountered objects in-episode, then COMPOSE held-out facts. ---
    bridge, composer, h, proj = build_compose_bridge(seed, with_body=True)
    print(f"[navcompose] merged bridge: {int(bridge.core_config.num_neurons)} neurons, readout={h.get('readout_region')}_X; "
          f"rf_base={h['rf_base']} cortex_it_base={int(h['it_indices_host'][0])}", flush=True)

    # T0 BYTE-IDENTITY GATE: cortex_it is appended AFTER rf (its first index is rf_last+1) AND it is the LAST region.
    it0 = int(h["it_indices_host"][0]); itL = int(h["it_indices_host"][-1])
    byte_identity = bool(it0 == h["rf_last_index"] + 1 and itL == h["n_neurons"] - 1)
    print(f"[navcompose] T0 byte-identity (cortex_it after rf, last region): {byte_identity}  "
          f"(it[0]={it0} rf_last={h['rf_last_index']} it[-1]={itL} N-1={h['n_neurons'] - 1})", flush=True)

    ep = run_compose_episode(bridge, composer, h, proj, layout, start_pos, route_waypoints, perceive=True)
    grounded = list(h["grounded_objects"])
    print(f"[navcompose]  COUPLED   grounded {len(grounded)} objects in-episode: {grounded} "
          f"(moves={len(ep['moves'])}, reached_waypoints={ep['reached_all_waypoints']})", flush=True)

    # held-out compose vs memorization-floor (anti-cheat 2). Capture the held-out composites (composed UNDER the
    # grounded codebook) so the lesion can re-cleanup them against a lesioned codebook.
    clean, floor, n_held, held_composites = _held_out_compose_score(composer, grounded, seed)
    print(f"[navcompose]  COMPOSE   held-out clean {clean:.3f} | mem-floor {floor:.3f} (chance {chance:.3f}) | "
          f"n_held_out={n_held}", flush=True)

    # the no-confab moat (anti-cheat 4) + a positive recall control.
    moat_ok, moat_tot, pos, stored, absent = _moat_check(composer, grounded)
    print(f"[navcompose]  MOAT      abstain {moat_ok}/{moat_tot} on {absent}  |  pos-recall {pos}/1 "
          f"(stored {stored[:1]})", flush=True)

    # provenance + co-residence (anti-cheat 3): assert the FIRST grounded object's codebook code == the live cortex_it
    # rate projection (captured in-episode), the composer is the co-resident merged composer, and the bind ran on the
    # bridge's RF complex synapses (cp_rf_w_re allocated after the moat's store). No second live read needed.
    prov = {}
    samp = h.get("provenance_sample")
    if samp is not None:
        prov = _provenance_check(bridge, composer, h, samp["rate"], samp["phases"], samp["obj"])

    # --- LESION the grounded-code map (anti-cheat 1, navsee cut-after-encode form): the held-out composites were
    # composed UNDER the grounded codebook; now restore the grounded objects to RANDOM codes and re-cleanup those
    # SAME composites -> the recovered phasor (carrying the grounded filler) no longer matches the random codebook ->
    # the compose collapses. Proves the COMPOSE rode the live-percept grounding, not a code-agnostic algebra trick. ---
    rng = np.random.default_rng(seed * 7919 + 3)
    for o in grounded:
        composer.concepts[o] = rng.uniform(0.0, 1.0, D)     # sever the live-percept grounding (random code)
    lesion_clean, _ = _lesion_recompose_score(composer, grounded, held_composites)
    print(f"[navcompose]  LESION    grounded->random codes, re-cleanup the SAME composites: held-out compose "
          f"{lesion_clean:.3f} (was {clean:.3f}; should collapse toward chance)", flush=True)

    # --- ISOLATED-PERCEPTION: NO body -> never traverses -> never arrives -> nothing grounded -> nothing composes. ---
    bridge_ip, composer_ip, h_ip, proj_ip = build_compose_bridge(seed, with_body=False)
    ep_ip = run_compose_episode(bridge_ip, composer_ip, h_ip, proj_ip, layout, start_pos, route_waypoints, perceive=True)
    grounded_ip = list(h_ip["grounded_objects"])
    print(f"[navcompose]  ISO-PERC  no body -> grounded {len(grounded_ip)} objects in-episode (expect 0)", flush=True)

    # --- the verdict for this seed ---
    n_grounded = len(grounded)
    compose_go = (n_grounded >= 2 and clean >= 0.90 and clean >= floor + 0.30)
    moat_ok_all = (moat_ok == moat_tot and moat_tot >= 1 and pos == 1)
    moat_breach = (moat_tot >= 1 and moat_ok < moat_tot)
    lesion_ok = (lesion_clean <= floor + 1e-9 or lesion_clean <= chance + 0.10)   # compose collapses without grounding
    iso_ok = (len(grounded_ip) == 0)
    go = bool(compose_go and moat_ok_all and byte_identity and lesion_ok and iso_ok)

    return {
        "seed": int(seed), "backend": backend, "chance": chance,
        "object_layout": {f"{c[0]},{c[1]}": w for c, w in layout.items()},
        "n_grounded": n_grounded, "grounded": grounded,
        "compose_clean": clean, "compose_floor": floor, "n_held_out": n_held,
        "moat_ok": moat_ok, "moat_tot": moat_tot, "pos_recall": pos, "moat_absent": absent, "stored": stored,
        "lesion_compose": lesion_clean, "byte_identity": byte_identity,
        "iso_perception_grounded": len(grounded_ip),
        "provenance": prov,
        "compose_go": compose_go, "moat_ok_all": moat_ok_all, "moat_breach": moat_breach,
        "lesion_ok": lesion_ok, "iso_ok": iso_ok, "go": go,
        "n_moves": len(ep["moves"]), "reached_all_waypoints": ep["reached_all_waypoints"],
    }


def _ser(o):
    if isinstance(o, dict):
        return {k: _ser(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_ser(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser(
        description="Navigate-to-compose-then-answer: the agent NAVIGATES the merged nav+conv bridge, PERCEIVES + "
                    "GROUNDS each encountered object IN-EPISODE (live cortex_it rate -> fixed projection -> composer "
                    "codebook), then COMPOSES a novel perceived-object fact on the co-resident rf slice + answers "
                    "who/what + ABSTAINS on unstored queries. The COMPOSE successor of the (B) navigate-to-see RECALL "
                    "milestone, on ONE bridge.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42], help="single-seed GPU smoke uses [42]")
    ap.add_argument("--out", type=str, default="research/findings/raw/navigate_to_compose_then_answer.json")
    args = ap.parse_args()

    _, backend = get_backend()
    print(f"[navcompose] backend={backend} — does the agent NAVIGATE, GROUND perceived objects IN-EPISODE, and "
          f"COMPOSE novel perceived-object facts on ONE bridge (held-out >> floor, moat intact)?", flush=True)
    results = [run_seed(s) for s in args.seeds]

    any_breach = any(r["moat_breach"] for r in results)
    all_go = all(r["go"] for r in results)
    mean_clean = float(np.mean([r["compose_clean"] for r in results]))
    mean_floor = float(np.mean([r["compose_floor"] for r in results]))
    verdict = "MOAT_BREACH" if any_breach else ("GO" if all_go else "NO-GO")

    print(f"\n{'=' * 100}", flush=True)
    print(f"[navcompose] {len(results)} seed(s): held-out compose clean {mean_clean:.3f} | mem-floor {mean_floor:.3f} "
          f"(chance {results[0]['chance']:.3f})  ==>  [{verdict}]", flush=True)
    for r in results:
        print(f"[navcompose]   seed {r['seed']}: grounded={r['n_grounded']} compose={r['compose_clean']:.3f} "
              f"floor={r['compose_floor']:.3f} lesion={r['lesion_compose']:.3f} moat={r['moat_ok']}/{r['moat_tot']} "
              f"pos={r['pos_recall']} iso-grounded={r['iso_perception_grounded']} byte-id={r['byte_identity']} "
              f"-> {'GO' if r['go'] else ('MOAT_BREACH' if r['moat_breach'] else 'NO-GO')}", flush=True)
    if verdict == "GO":
        print("[navcompose] GO: the agent NAVIGATES the merged nav+conv bridge, GROUNDS >=2 perceived objects "
              "IN-EPISODE (live cortex_it rate -> fixed projection -> the co-resident composer codebook), and "
              "COMPOSES held-out (never-composed) perceived-object facts on the merged rf slice that recover the "
              "perceived object >> the memorization floor; the no-confab moat abstains on every unstored query AND a "
              "stored fact retrieves; lesioning the grounding collapses the compose; the byte-identity holds. ==> the "
              "(B) RECALL milestone is upgraded to COMPOSE on ONE brain. 6-seed + the isolated/lesion battery is next.",
              flush=True)
    elif verdict == "MOAT_BREACH":
        print("[navcompose] MOAT_BREACH (HARD STOP): an unstored query was accepted on the merged bridge — "
              "investigate before any further build; NEVER weaken the moat.", flush=True)
    else:
        print("[navcompose] NO-GO: an honest negative — see the per-seed line (insufficient grounding / held-out "
              "compose ~ floor / lesion did not collapse / iso-perception grounded / byte-identity broke). Localize "
              "before the 6-seed run.", flush=True)
    print(f"{'=' * 100}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(_ser({"verdict": verdict, "backend": backend, "mean_clean": mean_clean, "mean_floor": mean_floor,
                        "results": results}), f, indent=2, default=str)
    print(f"[navcompose] wrote {args.out}", flush=True)
    raise SystemExit(0 if verdict == "GO" else (3 if verdict == "MOAT_BREACH" else 1))


if __name__ == "__main__":
    main()
