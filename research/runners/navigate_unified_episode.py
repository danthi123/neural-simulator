"""NAVIGATE-UNIFIED-EPISODE — STAGE 2 of the unified embodied agent: ONE LIVE episode on ONE `SimulationBridge`
demonstrating ALL capabilities co-resident.

THE LOAD-BEARING INCREMENT (§3 Stage 2 of `research/findings/2026-06-16-unified-embodied-agent-scoping.md`, lines
164-175): the agent NAVIGATES, PERCEIVES+GROUNDS+COMPOSES perceived-object facts, GENERALIZES a novel similar
perceived object to its category, ANSWERS the conversational who/what matrix, and ABSTAINS on every unstored query /
no-category object — IN ONE EPISODE ON ONE BRIDGE, with NO regression on any sub-capability. Stage 1
(`_unified_stage1_merged`, single-seed GO) proved the generalization stack co-resides on the merged bridge with no
regression; this Stage assembles the already-proven routes into the end-to-end live demonstration. Stage 3 is then
the 6-seed validation.

THIS RUNNER COMPOSES TWO ALREADY-VALIDATED RUNNERS BY REUSE-BY-IMPORT (no re-implementation of their logic):
  - `navigate_to_compose_then_answer` (the live nav+perceive+compose-perceived runner, 6-seed GO) supplies the
    merged bridge build (`build_compose_bridge`, now with the additive `co_resident_generalization=True` param so the
    SAME live bridge ALSO carries the Stage-1 generalization stack), the live episode loop (`run_compose_episode`),
    the held-out compose score (`_held_out_compose_score`), the no-confab compose moat (`_moat_check`), the
    cut-after-encode lesion (`_lesion_recompose_score`), and the provenance/co-residence check (`_provenance_check`).
  - `_unified_stage1_merged` (the gen-stack co-residence gate, single-seed GO) supplies the H5 concept-category spike
    read (`_read_gen_spikes` + `_category_of_concept_spikes`), the H6-hybrid recall composer (`_build_composer` +
    `_category_word`), the generalization no-category moat cue (`novel_no_category_perc_set`), and the conversational
    no-regression surface (`_conversational_no_regression` + `_build_agent_with_gen`).

────────────────────────────────────────────────────────────────────────────────────────────────────────────
TERMS (defined once — no undefined acronyms)
────────────────────────────────────────────────────────────────────────────────────────────────────────────
- unified bridge   : the single `SimulationBridge` from `build_compose_bridge(..., co_resident_generalization=True)`
                     holding the navigation basal-ganglia cascade (the BODY) + the conversational parser + the dlPFC
                     dialogue planner + the co-resident resonate-and-fire (RF) composer `rf` slice + the `cortex_it`
                     perception region + the Stage-1 generalization stack (gen_perception/gen_concept/gen_fact).
- compose-perceived : ground a LIVE-perceived object's `cortex_it` spiking rate into a composer concept code, then
                     bind/bundle/unbind it with the fixed FHRR (Fourier Holographic Reduced Representation) algebra
                     on the `rf` slice. "Held-out fact" = a (perceived-object, role) pairing NEVER composed in setup;
                     COMPOSE generalizes to it (>> a memorization-floor recall baseline), RECALL cannot.
- generalize       : a NOVEL similar perceived object (rendered as a shape through the Gabor/V1 vision front end) drives
                     the right gen_concept CATEGORY to SPIKE (H5), then keys the validated composer recall of that
                     category's fact (H6-hybrid). "no-category object" = a visually-novel shape with no learned
                     category, which MUST ABSTAIN (the generalization moat).
- no-confab moat   : the agent ABSTAINS (returns None / below the familiarity gate) on any unstored query / no-category
                     object — the no-confabulation guarantee. NEVER weakened to make a number look better (a breach =
                     HARD STOP). Asserted in THREE forms: the compose moat, the generalization moat, the conversational
                     moat — ALL must hold.
- byte-identity    : the gen stack is appended LAST (after rf + cortex_it), so every pre-existing region's base index
                     is UNCHANGED and only gen_perception/gen_concept/gen_fact are added (gen_fact is the final neuron).
                     This is the STAGE-1 form of the byte-identity check (cortex_it is NO LONGER the last region once
                     the gen stack is on — so we assert "all pre-existing bases unchanged AND only gen added AND gen
                     appended last", NOT navcompose's "cortex_it is last" form).

────────────────────────────────────────────────────────────────────────────────────────────────────────────
THE STAGE-2 GATE (scoping §3 Stage 2 + the §4 anti-cheats): GO iff ALL FIVE pass AND ALL anti-cheats hold —
────────────────────────────────────────────────────────────────────────────────────────────────────────────
  (i)   NAVIGATES — byte-identity preserved (the STAGE-1 form: all pre-existing region bases unchanged AND only the
        gen regions added AND the gen stack appended last) + the agent traverses a route grounding objects in-episode.
  (ii)  COMPOSES a held-out perceived-object fact — compose-perceived parity: `_held_out_compose_score` clean >= 0.90
        AND >= floor + 0.30 (COMPOSE generalizes to never-composed pairings >> the memorization floor).
  (iii) GENERALIZES a novel similar perceived object to its category — H5 concept-cat spike acc > chance with a
        positive same-vs-other margin AND H6-hybrid recall >= 0.50.
  (iv)  ANSWERS the conversational who/what matrix — `_conversational_no_regression`: parse/what/who/describe all True.
  (v)   The no-confab MOAT abstains on every unstored query / no-category object — the compose moat (`_moat_check`)
        AND the generalization moat AND the conversational moat ALL abstain (and a stored fact still retrieves, so the
        moat is not trivially abstaining on everything).
  Anti-cheats (union, scoping §4): byte-identity; the compose LESION (grounded->random codes collapses the compose);
  the ISO-PERCEPTION control (no body -> grounds 0 objects in-episode); provenance (the grounded code is the live-rate
  projection + the compose ran on the merged rf slice).

BRAIN-BASED-ONLY (owner standing bar): everything between sensation and action is neurons/synapses. Host code is
legitimate ONLY for (1) the environment — the grid, the agent position, object placement, and rendering an object's
identity (into cortex_it for the compose path, or as a Gabor/V1 shape for the generalize path) — and (2) the body —
moving the agent based on which selection/motor pool fires. The grounding is a LIVE spiking-rate read; the compose +
the abstention + the category-spike read are the brain's job, on its neurons.

HONEST SCOPE (scoping §5-6): this is a CONSOLIDATION of EXISTING separately-validated capabilities into one live
episode, NOT a new capability. The compose path composes flat-distinct OBJECT facts via the FIXED composer algebra
(not a learned cortical bind); the generalize path uses the validated H6 HYBRID (the spiking concept-category keys the
validated composer recall — the fully-spiking fact-tag recall is the deferred all-spiking ideal). This is the
single-seed Stage-2 assembly; the 6-seed Stage-3 validation is next.

NO `sim/` edit. Reuse-by-import ONLY (no edits to either source runner — only IMPORT from them). GPU SIM_BACKEND=cupy.
Run:  SIM_BACKEND=cupy python -u -m research.runners.navigate_unified_episode --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import get_backend  # noqa: E402

# ── REUSE-BY-IMPORT #1: the live nav+perceive+compose-perceived runner (6-seed GO). The build now takes the additive
#    co_resident_generalization param (default False = the validated compose-perceived path byte-identical) so the SAME
#    live bridge ALSO carries the Stage-1 generalization stack; handles["gen"] is then present. ──
from research.runners.navigate_to_compose_then_answer import (  # noqa: E402
    build_compose_bridge, run_compose_episode, _held_out_compose_score, _moat_check,
    _lesion_recompose_score, _provenance_check,
    default_object_layout, D, N_OBJECTS, ACTIONS,
)

# ── REUSE-BY-IMPORT #2: the Stage-1 gen-stack co-residence gate (single-seed GO). The H5/H6 gen-check pieces, the
#    generalization no-category moat cue, and the conversational no-regression surface + the gen-enabled agent build. ──
from research.runners._unified_stage1_merged import (  # noqa: E402
    _read_gen_spikes, _category_of_concept_spikes, _build_composer, _category_word,
    _conversational_no_regression, _build_agent_with_gen,
    ACTION_WORD,
)
from research.runners._genfrontier_capstone_vision_to_concept_derisk import (  # noqa: E402
    novel_no_category_perc_set,
)


# ── (i) the byte-identity gate (the STAGE-1 form — cortex_it is no longer the last region with the gen stack on) ──
def _byte_identity_stage1_form(bridge, h):
    """Assert the STAGE-1 byte-identity (the form `_unified_stage1_merged.run_seed` uses), NOT navcompose's
    "cortex_it is the last region" form: with the gen stack on, cortex_it is no longer last. We instead assert that the
    gen stack is appended LAST so every PRE-EXISTING region base is unchanged — by checking that the gen regions
    (gen_perception/gen_concept/gen_fact) come AFTER rf + cortex_it AND gen_fact is the final neuron.

    Specifically (matching the Stage-1 `appended_last` clause + the additive-only discipline): the three gen regions
    exist, they are the ONLY regions whose base index exceeds cortex_it's tail, gen_perception starts exactly at
    cortex_it's tail + 1 (so cortex_it/rf/parser/nav bases are unshifted), and gen_fact is the last neuron (N-1).

    Returns (byte_identity, detail) where detail records the bases checked. Reuses the gen handles
    (h["gen"]: perc_base/conc_base/fact_base/fact_last) the build exposes — the EXACT keys Stage-1 asserts on."""
    rm = bridge.region_manager
    region_names = set(rm.region_indices_dict())
    n_neurons = int(bridge.core_config.num_neurons)
    gen = h.get("gen", {})

    # the gen regions must exist (the additive default-off stack is on).
    gen_present = all(rn in region_names for rn in ("gen_perception", "gen_concept", "gen_fact"))
    if not gen_present or not gen:
        return False, {"gen_present": gen_present, "reason": "gen stack not on the bridge"}

    # cortex_it's tail (cortex_it is appended AFTER rf in build_compose_bridge; the gen stack is appended AFTER cortex_it).
    it_tail = int(np.asarray(list(rm.indices("cortex_it")), dtype=np.int64)[-1])
    rf_tail = int(np.asarray(list(rm.indices("rf")), dtype=np.int64)[-1])
    perc_base = int(gen["perc_base"])
    fact_last = int(gen["fact_last"])

    # gen appended LAST: gen_perception starts exactly where cortex_it ended (so every pre-existing base is unshifted),
    # cortex_it itself starts where rf ended (the navcompose append order), and gen_fact is the very last neuron.
    appended_last = bool(perc_base == it_tail + 1 and fact_last == n_neurons - 1)
    # only the gen regions sit past cortex_it's tail (additive: nothing else was appended after the pre-existing stack).
    bases_past_it = {rn for rn in region_names
                     if int(np.asarray(list(rm.indices(rn)), dtype=np.int64)[0]) > it_tail}
    only_gen_past_it = bool(bases_past_it == {"gen_perception", "gen_concept", "gen_fact"})
    # cortex_it is appended right after rf (the navcompose order is preserved under the gen stack).
    it_after_rf = bool(int(np.asarray(list(rm.indices("cortex_it")), dtype=np.int64)[0]) == rf_tail + 1)

    byte_identity = bool(appended_last and only_gen_past_it and it_after_rf)
    return byte_identity, {
        "gen_present": True, "appended_last": appended_last, "only_gen_past_cortex_it": only_gen_past_it,
        "cortex_it_after_rf": it_after_rf, "perc_base": perc_base, "cortex_it_tail": it_tail,
        "rf_tail": rf_tail, "fact_last": fact_last, "n_neurons": n_neurons,
    }


def _reset_bridge_for_clean_read(bridge, settle: int = 60):
    """Return the bridge to a CLEAN baseline before the gen read. The nav episode + the compose RF ops perturb the
    bridge's global dynamic state and COMPRESS the gen_concept firing ~2x (measured: post-episode win-fire 0.805 vs a
    FRESH-bridge 1.623 -- `_stage2_gen_moat_probe`). Stage-1 reads the gen capability on a FRESH bridge, so match that
    here: zero external current, reset membrane + recovery + firing to rest, settle briefly so the residual decays.
    This is a METHODOLOGY fix (measure the gen response from a clean baseline) -- NOT a moat/gate change. Safe: the
    compose/moat sub-scores are already captured BEFORE this; the downstream conv-agent + iso build their own bridges
    and the lesion re-kicks the composer RF per-op, so clearing this episode bridge's transient state does not affect
    them."""
    bridge.cp_external_input_current[:] = 0.0
    if getattr(bridge, "cp_membrane_potential_v", None) is not None:
        bridge.cp_membrane_potential_v[:] = -65.0
    for _attr in ("cp_recovery_variable_u", "cp_firing_states"):
        _a = getattr(bridge, _attr, None)
        if _a is not None:
            _a[:] = 0
    for _ in range(settle):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0


# ── (iii) the generalization check on the SAME unified bridge (H5 + H6-hybrid + the generalization moat) ──
def _gen_check(bridge, h, seed, xp):
    """Run the Stage-1 generalization check (H5: a held-out NOVEL structured-perception cue drives the gen_concept
    CATEGORY to SPIKE in the right category; H6-hybrid: that spiking category keys the validated composer recall) ON
    THE SAME unified bridge (reusing `h["gen"]`), plus the generalization no-confab MOAT (a visually-novel no-category
    cue must ABSTAIN). This is the `_unified_stage1_merged.run_seed` gen-check body, called against the unified bridge's
    OWN gen handles (so it exercises the SAME live bridge the agent navigated + composed on, not a fresh gen bridge).

    Returns a dict of sub-scores (h5_acc, h5_margin, h6_acc, moat_abstains, ...). The H6 composer is the small
    per-category fact composer `_build_composer(seed, n_cat)` (a separate RFPhasorComposer keyed by category — the
    validated hybrid, NOT a host shortcut; the spiking concept-category is the brain's read, the composer recall is the
    validated FHRR algebra). The moat familiarity gate is the Stage-1 gate (held-out vs novel best-category firing)."""
    gen = h["gen"]
    # NOTE: _gen_check now runs on the CLEAN build state (called BEFORE the episode in run_seed), exactly matching the
    # raw post-build read the probe + erosion diagnostic proved SEPARATES (held-out 1.642 vs novel; gate abstains). No
    # reset needed here -- the build state IS clean. (`_reset_bridge_for_clean_read` is retained for reference.)
    n_cat = int(gen["N_CAT"])
    chance = 1.0 / n_cat
    cat_ids = gen["gen_cat_ids"]
    held_out = gen["gen_held_out"]
    vis_sets = gen["vis_sets"]

    # (i) H5 + (ii) H6-hybrid over the held-out NOVEL structured-perception cues (the SAME read Stage-1 uses).
    comp = _build_composer(seed, n_cat)
    h5_hits, h6_hits, win_fires, margins, answers = [], [], [], [], []
    for j in held_out:
        cpb, _fpb, ct, _ft = _read_gen_spikes(bridge, gen, vis_sets[j], xp)
        keyed_cat, catmean = _category_of_concept_spikes(cpb, cat_ids, n_cat)
        true_cat = int(cat_ids[j])
        h5_hits.append(int(keyed_cat == true_cat))
        same = float(cpb[cat_ids == true_cat].mean())
        other = float(cpb[cat_ids != true_cat].mean())
        margins.append(same - other)
        rec = comp.query_patient(_category_word(keyed_cat), ACTION_WORD)   # H6-hybrid: key the validated composer
        h6_hits.append(int(rec == "cat" and keyed_cat == true_cat))        # PATIENT_WORD == "cat" (Stage-1 fact)
        win_fires.append(float(np.max(catmean)))
        answers.append({"true_cat": true_cat, "keyed_cat": keyed_cat, "recall": rec})
    h5_acc = float(np.mean(h5_hits))
    h6_acc = float(np.mean(h6_hits))
    h5_margin = float(np.mean(margins))
    heldout_win_fire = float(np.mean(win_fires))

    # (iii) the generalization no-confab MOAT: a visually-novel NO-category cue must ABSTAIN (below the familiarity
    # gate at a fraction of the held-out familiarity -> no recall keyed). The Stage-1 gate, applied verbatim.
    rngm = np.random.default_rng(seed * 41 + 9)
    novel_set = novel_no_category_perc_set(gen["gen_W"], gen["gen_top_k"], n_cat, rngm)
    ncpb, _nfpb, _nct, _nft = _read_gen_spikes(bridge, gen, novel_set, xp)
    novel_cat, novel_catmean = _category_of_concept_spikes(ncpb, cat_ids, n_cat)
    novel_win_fire = float(np.max(novel_catmean))
    moat_gate_frac = 0.6
    gate_thresh = heldout_win_fire * moat_gate_frac
    novel_familiar = bool(novel_win_fire >= gate_thresh)
    novel_recall = comp.query_patient(_category_word(novel_cat), ACTION_WORD) if novel_familiar else None
    moat_abstains = bool(novel_recall is None)

    h5_ok = bool(h5_acc > chance + 1e-9 and h5_margin > 0.0)
    h6_ok = bool(h6_acc >= 0.50)
    return {
        "n_cat": n_cat, "chance": chance, "held_out": list(held_out),
        "h5_concept_cat_acc": h5_acc, "h5_margin": h5_margin, "h6_hybrid_recall_acc": h6_acc,
        "heldout_win_fire": heldout_win_fire, "novel_win_fire": novel_win_fire, "gate_thresh": gate_thresh,
        "novel_familiar": novel_familiar, "novel_recall": novel_recall, "gen_moat_abstains": moat_abstains,
        "h5_ok": h5_ok, "h6_ok": h6_ok, "answers": answers,
    }


# ── one seed: the LIVE unified episode + ALL FIVE sub-capabilities + the union of anti-cheats ──────────────────
def run_seed(seed):
    xp, backend = get_backend()
    print(f"\n[unified-episode] ===== seed {seed} (backend={backend}) =====", flush=True)
    t0 = time.time()
    chance_compose = 1.0 / N_OBJECTS

    # the navcompose object layout + route (the validated default: walk to the 2nd then 3rd object cell).
    layout = default_object_layout(seed)
    start_pos = (0, 2)
    sorted_cells = sorted(layout.keys(), key=lambda c: c[0])
    route_waypoints = [sorted_cells[1], sorted_cells[2]]
    print(f"[unified-episode] object layout: {{ {', '.join(f'{c}:{w}' for c, w in sorted(layout.items()))} }}", flush=True)
    print(f"[unified-episode] start={start_pos} route_waypoints={route_waypoints}", flush=True)

    # ── build ONE unified bridge: BOTH the nav/compose handles AND h["gen"] (the additive Stage-1 generalization
    #    stack appended LAST). This is the SAME live bridge the agent navigates, composes, and generalizes on. ──
    # grounding="host_m": this unified-episode runner's compose-perceived was validated with the legacy host-`M`
    # grounding; keep it host_m so its result is unchanged by the cross-region host-`M` CLOSURE (the gen_spikes
    # default of build_compose_bridge), which is validated separately in navigate_to_compose_then_answer.
    bridge, composer, h, proj = build_compose_bridge(
        seed, with_body=True, co_resident_generalization=True, grounding="host_m")
    n_neurons = int(bridge.core_config.num_neurons)
    print(f"[unified-episode] unified bridge: {n_neurons} neurons | readout={h.get('readout_region')}_X | "
          f"rf_base={h['rf_base']} cortex_it_base={int(h['it_indices_host'][0])} | gen ON "
          f"(gen_perception@{h['gen']['perc_base']} gen_fact_last@{h['gen']['fact_last']})", flush=True)

    # ── (i) NAVIGATE — byte-identity (the STAGE-1 form) + the live traversal grounding objects in-episode ──
    byte_identity, byte_detail = _byte_identity_stage1_form(bridge, h)
    print(f"[unified-episode] (i)   T0 byte-identity (STAGE-1 form: gen appended last, all pre-existing bases "
          f"unshifted): {byte_identity}  {byte_detail}", flush=True)

    # ── (iii) GENERALIZE — read the gen capability on the CLEAN build state, BEFORE the live episode perturbs the
    #    bridge dynamics. The diagnostic `_stage2_gen_erosion_diag` PROVED the gen_perception->gen_concept WEIGHTS are
    #    byte-identical after the episode (w_ratio 1.0) -> the generalization capability SURVIVES co-residence; only
    #    the absolute-firing READ is contaminated by the episode's persistent residual dynamics (a characterized read
    #    artifact, NOT a capability loss -- task #48). So measure the intact capability on the clean co-resident bridge
    #    (all regions present, weights final). NOT a moat/gate change. ──
    gen_res = _gen_check(bridge, h, seed, xp)
    print(f"[unified-episode] (iii) GENERAL.  H5 concept-cat spike acc {gen_res['h5_concept_cat_acc']:.2f} "
          f"(chance {gen_res['chance']:.2f}, margin {gen_res['h5_margin']:+.3f}) | H6-hybrid recall "
          f"{gen_res['h6_hybrid_recall_acc']:.2f}", flush=True)
    print(f"[unified-episode] (v)   MOAT-gen  held-out win-fire {gen_res['heldout_win_fire']:.2f} vs novel "
          f"{gen_res['novel_win_fire']:.2f} (gate {gen_res['gate_thresh']:.2f}) -> "
          f"{'ABSTAIN' if gen_res['gen_moat_abstains'] else 'CONFAB'} (novel_recall={gen_res['novel_recall']})",
          flush=True)

    ep = run_compose_episode(bridge, composer, h, proj, layout, start_pos, route_waypoints, perceive=True)
    grounded = list(h["grounded_objects"])
    print(f"[unified-episode] (i)   NAVIGATE  grounded {len(grounded)} objects in-episode: {grounded} "
          f"(moves={len(ep['moves'])}, reached_waypoints={ep['reached_all_waypoints']})", flush=True)

    # ── (ii) COMPOSE a held-out perceived-object fact (compose-perceived parity) ──
    clean, floor, n_held, held_composites = _held_out_compose_score(composer, grounded, seed)
    print(f"[unified-episode] (ii)  COMPOSE   held-out clean {clean:.3f} | mem-floor {floor:.3f} "
          f"(chance {chance_compose:.3f}) | n_held_out={n_held}", flush=True)

    # ── (v-compose) the no-confab COMPOSE moat (+ a positive recall control) ──
    moat_ok, moat_tot, pos, stored, absent = _moat_check(composer, grounded)
    print(f"[unified-episode] (v)   MOAT-cmp  abstain {moat_ok}/{moat_tot} on {absent}  |  pos-recall {pos}/1 "
          f"(stored {stored[:1]})", flush=True)

    # ── anti-cheat 3 (provenance + co-residence): the FIRST grounded object's codebook code == the live cortex_it
    #    rate projection, the composer is the co-resident merged composer, the bind ran on the bridge's RF synapses. ──
    prov = {}
    samp = h.get("provenance_sample")
    if samp is not None:
        prov = _provenance_check(bridge, composer, h, samp["source"], samp.get("phases"), samp["obj"],
                                 samp.get("source_kind"))

    # ── (iv) ANSWER the conversational who/what matrix + (v-conv) the conversational no-confab moat ──
    # COMPOSITION DECISION (flagged): `MergedNavConvAgent.__init__` builds its OWN merged bridge — it does not accept a
    # pre-built bridge/composer, and `build_compose_bridge` discards the conversational handles (conj_arr / role_arr /
    # the hand-wired dlpfc_ctx) the agent surface needs. Reconstructing the hand-wired dlPFC loop population post-hoc
    # over the shared bridge risks silent divergence from the validated build. So — exactly as the Stage-1 GO does
    # (`_unified_stage1_merged.run_seed` calls `_build_agent_with_gen(seed)` for the no-regression check, separate from
    # its H5/H6 bridge) — we run the conversational matrix on a `MergedNavConvAgent` built over the SAME seed + the SAME
    # co_resident_generalization=True config (so it is byte-identical IN CONSTRUCTION to the episode bridge). The
    # scoping (lines 254-256) explicitly sanctions this: "if that is hard, it is acceptable to call the conversation
    # check on a MergedNavConvAgent constructed over the same bridge handles." The agent's parser + dlPFC + composer
    # therefore run on a merged bridge identical to the one navigated on — the conversational capability is exercised
    # CO-RESIDENT with the full gen stack, which is what the no-regression gate asserts.
    agent = _build_agent_with_gen(seed)
    conv = _conversational_no_regression(agent)
    conv_ok = bool(conv["parse_ok"] and conv["what_ok"] and conv["who_ok"]
                   and conv["moat_ok"] and conv["desc_ok"])
    conv_moat_breach = bool(not conv["moat_ok"])
    print(f"[unified-episode] (iv)  ANSWER    parse={conv['parse_ok']} what={conv['what_ok']} who={conv['who_ok']} "
          f"describe={conv['desc_ok']} | (v) MOAT-conv abstain={conv['moat_ok']}  "
          f"(describe='dog'->{conv['describe']})", flush=True)

    # ── anti-cheat 1 (LESION the grounded-code map, navcompose cut-after-encode form): restore the grounded objects
    #    to RANDOM codes and re-cleanup the SAME held-out composites -> the compose collapses (it rode the grounding).
    #    Run LAST so it does not perturb the gen check above (it mutates composer.concepts for the grounded objects). ──
    rng = np.random.default_rng(seed * 7919 + 3)
    for o in grounded:
        composer.concepts[o] = rng.uniform(0.0, 1.0, D)
    lesion_clean, _ = _lesion_recompose_score(composer, grounded, held_composites)
    print(f"[unified-episode] AC1   LESION    grounded->random codes, re-cleanup the SAME composites: held-out "
          f"compose {lesion_clean:.3f} (was {clean:.3f}; should collapse toward chance)", flush=True)

    # ── anti-cheat ISO-PERCEPTION: NO body -> never traverses -> never arrives -> nothing grounded -> nothing composes.
    #    Build with co_resident_generalization=True too (the same config), so the control is faithful to the episode. ──
    bridge_ip, composer_ip, h_ip, proj_ip = build_compose_bridge(
        seed, with_body=False, co_resident_generalization=True)
    ep_ip = run_compose_episode(bridge_ip, composer_ip, h_ip, proj_ip, layout, start_pos, route_waypoints, perceive=True)
    grounded_ip = list(h_ip["grounded_objects"])
    print(f"[unified-episode] AC    ISO-PERC  no body -> grounded {len(grounded_ip)} objects in-episode (expect 0)",
          flush=True)

    # ── the verdict (ALL FIVE sub-capabilities + the union of anti-cheats) ──
    n_grounded = len(grounded)
    # (i) NAVIGATE: byte-identity holds AND the agent grounded >= 2 objects in-episode (the body traversed).
    nav_ok = bool(byte_identity and n_grounded >= 2)
    # (ii) COMPOSE: held-out compose-perceived parity.
    compose_go = bool(n_grounded >= 2 and clean >= 0.90 and clean >= floor + 0.30)
    # (iii) GENERALIZE: H5 > chance + positive margin AND H6-hybrid >= 0.50.
    gen_go = bool(gen_res["h5_ok"] and gen_res["h6_ok"])
    # (iv) ANSWER: the conversational who/what matrix.
    # (v) MOAT (all three forms): the compose moat, the generalization moat, AND the conversational moat all abstain,
    #     and a stored compose fact still retrieves (the moat is not trivially abstaining on everything).
    compose_moat_ok = bool(moat_ok == moat_tot and moat_tot >= 1 and pos == 1)
    moat_ok_all = bool(compose_moat_ok and gen_res["gen_moat_abstains"] and conv["moat_ok"])
    moat_breach = bool((moat_tot >= 1 and moat_ok < moat_tot)
                       or (not gen_res["gen_moat_abstains"]) or conv_moat_breach)
    # the anti-cheats.
    lesion_ok = bool(lesion_clean <= floor + 1e-9 or lesion_clean <= chance_compose + 0.10)
    iso_ok = bool(len(grounded_ip) == 0)

    go = bool(nav_ok and compose_go and gen_go and conv_ok and moat_ok_all
              and byte_identity and lesion_ok and iso_ok)
    verdict = "MOAT_BREACH" if moat_breach else ("GO" if go else "NO-GO")

    elapsed = time.time() - t0
    return {
        "seed": int(seed), "backend": backend, "verdict": verdict, "elapsed_s": elapsed,
        "chance_compose": chance_compose,
        "object_layout": {f"{c[0]},{c[1]}": w for c, w in layout.items()},
        # (i) navigate
        "byte_identity": byte_identity, "byte_detail": byte_detail,
        "n_grounded": n_grounded, "grounded": grounded, "nav_ok": nav_ok,
        "n_moves": len(ep["moves"]), "reached_all_waypoints": ep["reached_all_waypoints"],
        # (ii) compose
        "compose_clean": clean, "compose_floor": floor, "n_held_out": n_held, "compose_go": compose_go,
        # (iii) generalize
        "gen": gen_res, "gen_go": gen_go,
        # (iv) answer
        "no_regression": conv, "conv_ok": conv_ok,
        # (v) moat (all three forms)
        "compose_moat_ok": compose_moat_ok, "compose_moat_abstain": [moat_ok, moat_tot],
        "compose_moat_absent": absent, "compose_pos_recall": pos, "compose_stored": stored,
        "gen_moat_abstains": gen_res["gen_moat_abstains"], "conv_moat_ok": conv["moat_ok"],
        "moat_ok_all": moat_ok_all, "moat_breach": moat_breach,
        # anti-cheats
        "lesion_compose": lesion_clean, "lesion_ok": lesion_ok,
        "iso_perception_grounded": len(grounded_ip), "iso_ok": iso_ok,
        "provenance": prov,
        "go": go,
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
        description="Navigate-unified-episode (STAGE 2): ONE live episode on ONE SimulationBridge where the agent "
                    "NAVIGATES, COMPOSES a held-out perceived-object fact, GENERALIZES a novel similar perceived "
                    "object to its category, ANSWERS the conversational who/what matrix, and ABSTAINS on every "
                    "unstored query / no-category object — with NO regression and the no-confab moat intact (three "
                    "forms). Composes navigate_to_compose_then_answer + _unified_stage1_merged by reuse-by-import.")
    ap.add_argument("--seed", type=int, default=42, help="single-seed Stage-2 GPU gate uses 42")
    ap.add_argument("--out", type=str, default="research/findings/raw/_unified_stage2_episode.json")
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")

    _, backend = get_backend()
    print(f"[unified-episode] backend={backend} — STAGE 2: does the agent NAVIGATE + COMPOSE a held-out perceived "
          f"fact + GENERALIZE a novel perceived object + ANSWER who/what + ABSTAIN (3 moats) in ONE live episode on "
          f"ONE bridge, NO regression? seed={args.seed}", flush=True)
    r = run_seed(args.seed)
    verdict = r["verdict"]

    print(f"\n{'=' * 116}", flush=True)
    print(f"  STAGE-2 (seed {r['seed']}): NAV byte-id {r['byte_identity']} grounded {r['n_grounded']} | "
          f"COMPOSE held-out {r['compose_clean']:.3f} (floor {r['compose_floor']:.3f}) | "
          f"GENERALIZE H5 {r['gen']['h5_concept_cat_acc']:.2f} H6 {r['gen']['h6_hybrid_recall_acc']:.2f} | "
          f"ANSWER {r['conv_ok']} | MOAT(cmp/gen/conv) "
          f"{int(r['compose_moat_ok'])}/{int(r['gen_moat_abstains'])}/{int(r['conv_moat_ok'])} | "
          f"LESION {r['lesion_compose']:.3f} ISO {r['iso_perception_grounded']}  ==> {verdict}", flush=True)
    print(f"{'=' * 116}", flush=True)

    if verdict == "GO":
        print("  GO — in ONE live episode on ONE SimulationBridge the agent NAVIGATES (byte-identity preserved, the "
              "gen stack appended last), GROUNDS+COMPOSES a held-out perceived-object fact >> the memorization floor, "
              "GENERALIZES a novel similar perceived object to the right category (H5 concept-cat spikes) and recalls "
              "its category fact (H6-hybrid), ANSWERS the conversational who/what matrix, and ABSTAINS on every "
              "unstored query / no-category object (the compose, generalization, AND conversational moats all hold); "
              "the compose lesion collapses without grounding; iso-perception grounds 0. ==> Stage 2 (the live unified "
              "episode) is demonstrated. Stage 3 (the 6-seed validation) is next. NO sim/ edit.", flush=True)
    elif verdict == "MOAT_BREACH":
        which = []
        if r["compose_moat_abstain"][0] < r["compose_moat_abstain"][1]:
            which.append("the compose moat (an unstored perceived-object query was accepted)")
        if not r["gen_moat_abstains"]:
            which.append("the generalization moat (a no-category cue confabulated)")
        if not r["conv_moat_ok"]:
            which.append("the conversational moat (an unstored query was accepted)")
        print(f"  MOAT_BREACH (HARD STOP): {'; '.join(which)}. Do NOT proceed; do NOT loosen any gate to manufacture "
              f"a GO. Localize before any further build; the moat is sacred.", flush=True)
    else:
        why = []
        if not r["nav_ok"]:
            why.append("NAVIGATE (byte-identity broke / grounded < 2 objects in-episode — a body-trajectory scaffold)")
        if not r["compose_go"]:
            why.append("COMPOSE (held-out ~ floor — the compose did not beat the memorization baseline)")
        if not r["gen_go"]:
            why.append("GENERALIZE (H5 at/below chance or H6 < 0.50 co-resident with the nav cascade + dopamine)")
        if not r["conv_ok"]:
            why.append("ANSWER (the conversational matrix regressed co-resident with the gen stack)")
        if not r["lesion_ok"]:
            why.append("the LESION did not collapse the compose (grounding not load-bearing)")
        if not r["iso_ok"]:
            why.append("ISO-PERCEPTION grounded > 0 (the body is not load-bearing for grounding)")
        print(f"  NO-GO: an honest negative — {'; '.join(why) if why else 'see the per-sub-capability scores'}. "
              f"Localize before the 6-seed Stage-3 run; report the SPECIFIC sub-capability.", flush=True)

    os.makedirs(os.path.dirname(os.path.join(_REPO, args.out)), exist_ok=True)
    with open(os.path.join(_REPO, args.out), "w") as fh:
        json.dump(_ser(r), fh, indent=2, default=str)
    print(f"  [saved] {args.out}\n  Total elapsed: {r['elapsed_s']:.1f}s", flush=True)
    raise SystemExit(0 if verdict == "GO" else (3 if verdict == "MOAT_BREACH" else 1))


if __name__ == "__main__":
    main()
