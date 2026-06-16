"""Unified embodied agent — STAGE 1: the generalization stack CO-RESIDENT on the merged nav+conv bridge.

THE LOAD-BEARING INCREMENT (§3 Stage 1 of `research/findings/2026-06-16-unified-embodied-agent-scoping.md`):
place the validated generalization stack (H5: structured perception → NMDA concept assembly SPIKES for the right
category; H6-hybrid: the spiking concept-category keys the validated RFPhasorComposer recall) onto the ONE merged
`SimulationBridge` that already navigates + parses + plans (dlPFC) + composes (RF) — with NO regression and the
no-confab moat intact. This is the only genuinely-new integration; Stages 2–3 are then assembly + validation.

HOW THE STACK IS ADDED (additive, default-off, byte-identity-preserving):
  `build_merged_nav_conv_bridge(..., co_resident_generalization=True)` appends THREE framework regions LAST
  (after rf + cortex_it, so the nav/parser/dlPFC/rf/cortex_it index bases are BYTE-UNCHANGED — the exact pattern
  rf/cortex_it already use):
    gen_perception (N_V1_COMPLEX=2048, the Gabor/V1 top-K drive target),
    gen_concept    (F × gen_n_concept_per, enable_nmda=True),
    gen_fact       (N_CAT × gen_n_fact_per, enable_nmda=True),
  + the plastic rate-Hebbian gen_perception→gen_concept convergence (tagged GEN_CONV_GATE)
  + the FIXED convergent gen_concept→gen_fact pathway.
  The convergence is trained-then-frozen via the cp_plasticity_rate_gain INDEX MASK (the finalize_conv_for_nav_gate
  discipline: only the perception→concept edges plastic during the train pass; nav/parser/dlPFC frozen), so it is
  ISOLATED from the navigation reward-STDP + the global dopamine scope="all" + the parser. hebbian_max_weight=400
  (the merged config) serves both stacks (Stage-0 finding: at the convergence default 20 the parser collapses; the
  convergence's category-MEAN-over-spikes read is robust to the higher cap). The Gabor/V1 vision pipeline + the
  leakage-free split are the de-risk's exact machinery, reused-by-import.

THE H6-HYBRID RECALL (the one cross-code handoff the agent newly requires end-to-end — validated at 0.92 by the
capstone): for a HELD-OUT novel structured-perception cue, read which gen_concept-CATEGORY SPIKES (a population
read of cp_firing_states), then key the validated RFPhasorComposer.query_patient by that category's stored fact.
A no-category cue must ABSTAIN (the familiarity gate on the concept spikes returns None → no recall keyed).

THE GATE (single-seed GPU first; §5):
  GO if, ON THE MERGED BRIDGE:
    (i)   a held-out NOVEL structured-perception cue drives the concept-category to SPIKE in the right category
          > chance with a positive same-vs-other margin;
    (ii)  the H6-hybrid recalls the matched category's fact (≥ 0.50 single-seed);
    (iii) the no-confab moat ABSTAINS on a no-category cue (ZERO breaches — HARD);
    AND — the load-bearing no-regression check —
    (iv)  the EXISTING merged-bridge capabilities DON'T regress: the conversational parse + who/what recall + the
          moat still pass (the merged-agent assertions), AND the nav/parser/rf index bases are BYTE-UNCHANGED vs
          co_resident_generalization=False (the gen regions appended last).
  NEGATIVE/PARTIAL (acceptable, honest) if H5 is too noisy co-resident with the nav cascade + global dopamine, or a
  co-residence regression appears — report the SPECIFIC localized issue. MOAT BREACH = HARD STOP.

NO sim/ edit. Reuse-by-import. GPU SIM_BACKEND=cupy.
Run:  SIM_BACKEND=cupy python -u -m research.runners._unified_stage1_merged --seed 42
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

from sim.backend import get_backend, to_host  # noqa: E402

from research.runners.nav_conv_merged_bridge import (  # noqa: E402
    build_merged_nav_conv_bridge, MergedRFComposer, GEN_PERCEPTION, GEN_CONCEPT, GEN_FACT,
)
# the H5 spike read + the no-category moat cue, reuse-by-import (the validated de-risk machinery).
from research.runners._genfrontier_graded_propagation_derisk import read_heldout_spikes  # noqa: E402
from research.runners._genfrontier_capstone_vision_to_concept_derisk import (  # noqa: E402
    novel_no_category_perc_set,
)
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402


# the per-category fact "<category> chase cat" the H6-hybrid recalls for a matched-category cue.
ACTION_WORD = "chase"
PATIENT_WORD = "cat"
def _category_word(c):
    return f"category{c}"


# ── H5 read on the merged bridge: drive ONLY a gen_perception cue, accumulate gen_concept + gen_fact spikes ──
class _ReadArgs:
    perc_scale = 300.0
    read_steps = 80


def _read_gen_spikes(bridge, gen, perc_idx, xp):
    """Drive ONLY the gen_perception cue on the MERGED bridge; accumulate gen_concept (per concept block) AND
    gen_fact (per CATEGORY block) SPIKES over the read window (cp_firing_states). Returns
    (conc_per_block[F], fact_per_block[N_CAT], conc_total, fact_total)."""
    a = _ReadArgs()
    # read_heldout_spikes rebases perception indices by `- perc_region[0]` (expects GLOBAL); perc_idx (vis_sets /
    # novel_set) are LOCAL 0-based V1-complex indices → globalize so the rebase recovers the correct local indices.
    perc_idx_g = np.asarray(perc_idx, dtype=np.int64) + int(np.asarray(gen["perc_region"])[0])
    return read_heldout_spikes(
        bridge, xp, gen["perc_region"], gen["conc_region"], gen["fact_region"],
        gen["conc_blocks"], gen["fact_blocks"], perc_idx_g, a.perc_scale, a.read_steps)


def _category_of_concept_spikes(conc_per_block, cat_ids, n_cat):
    """The H5 read: the category whose concept blocks SPIKE most (category-mean over the concept-block spikes)."""
    catmean = [float(conc_per_block[cat_ids == c].mean()) for c in range(n_cat)]
    return int(np.argmax(catmean)), catmean


def _build_composer(seed, n_cat):
    """The H6-hybrid composer: a per-category fact "<category> chase cat" stored in the validated RFPhasorComposer."""
    vocab = [_category_word(c) for c in range(n_cat)] + [ACTION_WORD, PATIENT_WORD]
    comp = RFPhasorComposer(seed=seed, D=64, vocab=vocab)
    for c in range(n_cat):
        comp.store(_category_word(c), ACTION_WORD, PATIENT_WORD)
    return comp


# ── the no-regression conversational check (the existing merged-agent capabilities must still pass) ──────────
def _conversational_no_regression(agent):
    """Reuse the merged-agent surface: comprehend (parser) + store + who/what recall + the no-confab moat. Returns
    a dict of booleans. The agent is the MergedNavConvAgent on the SAME co_resident_generalization=True bridge."""
    # comprehension + store (the parser runs on the merged framework slices). Use the SHIPPED in-vocab sentence
    # ("dog go north") that tests/test_nav_conv_step2b_coresident.py validates -- the canonical no-regression
    # definition. (The earlier "dog chase cat" used "chase", which is the generalization-test ACTION_WORD and is
    # NOT in the composer's DEFAULT_VOCAB, so composer.store raised KeyError once the parser was fixed.)
    roles = agent.hear("dog go north")
    parse_ok = bool(roles.get("agent") == "dog" and roles.get("action") == "go"
                    and roles.get("patient") == "north")
    # who/what recall over the stored fact.
    what_ok = bool(agent.what_does("dog", "go") == "north")
    who_ok = bool(agent.who_does("go", "north") == "dog")
    # the no-confab moat: an unstored (agent, action) query must abstain (None) -- the shipped moat words.
    moat_what = agent.what_does("river", "look")           # river never stored as a looker
    moat_describe = agent.describe("river")                # river has no known fact
    moat_ok = bool(moat_what is None and moat_describe is None)
    # a positive describe (the stored fact renders, so the moat is not trivially abstaining on everything).
    desc = agent.describe("dog")
    desc_ok = bool(desc is not None and "dog" in str(desc))
    return {"parse_ok": parse_ok, "what_ok": what_ok, "who_ok": who_ok, "moat_ok": moat_ok,
            "desc_ok": desc_ok, "describe": desc, "moat_what": moat_what, "moat_describe": moat_describe}


def run_seed(seed):
    xp, backend = get_backend()
    print(f"\n[unified STAGE-1] ===== seed {seed} (backend={backend}) =====", flush=True)
    t0 = time.time()

    # ── BASELINE bridge (co_resident_generalization=False) for the byte-identity check ──
    base_bridge, base_h = build_merged_nav_conv_bridge(
        seed=seed, co_resident_rf=True, co_resident_perception=True, enable_spiking_wta_readout=True)
    base_rm = base_bridge.region_manager
    base_regions = list(base_rm.region_indices_dict())
    base_n = int(base_bridge.core_config.num_neurons)
    # capture EVERY baseline region's base index so the byte-identity check can assert all are unchanged (additive:
    # only the gen_* regions may be added; every pre-existing base must match).
    base_region_bases = {rn: int(base_rm.indices(rn)[0]) for rn in base_regions}
    base_rf0 = base_region_bases["rf"]
    base_parse_conj0 = base_region_bases["parse_conj"]
    base_cortex_it0 = base_region_bases["cortex_it"]
    base_cortex_N0 = base_region_bases["cortex_N"]
    print(f"[stage1] BASELINE (gen OFF): {len(base_regions)} regions, {base_n} neurons | "
          f"cortex_N@{base_cortex_N0} parse_conj@{base_parse_conj0} rf@{base_rf0} cortex_it@{base_cortex_it0}",
          flush=True)
    del base_bridge

    # ── the STAGE-1 merged bridge WITH the generalization stack ──
    bridge, h = build_merged_nav_conv_bridge(
        seed=seed, co_resident_rf=True, co_resident_perception=True, enable_spiking_wta_readout=True,
        co_resident_generalization=True)
    rm = bridge.region_manager
    gen = h["gen"]
    n_cat = gen["N_CAT"]
    chance = 1.0 / n_cat
    region_names = list(rm.region_indices_dict())
    n_neurons = int(bridge.core_config.num_neurons)
    print(f"[stage1] STAGE-1 (gen ON): {len(region_names)} regions, {n_neurons} neurons | "
          f"gen_perception@{gen['perc_base']} gen_concept@{gen['conc_base']} gen_fact@{gen['fact_base']} "
          f"(fact_last@{gen['fact_last']}, N-1={n_neurons - 1})", flush=True)
    print(f"[stage1] vision structure margin {gen['gen_set_margin']:+.3f} "
          f"[{'PRESERVED' if gen['gen_structure_preserved'] else 'LOST'}] | "
          f"train firing diag {gen['gen_train_diag']}", flush=True)

    # ── (iv-byte) BYTE-IDENTITY: the nav/parser/rf/cortex_it index bases are UNCHANGED vs gen OFF, AND the gen
    #    regions are appended LAST (gen_perception base == rf/cortex_it region tail + 1; gen_fact is the last). ──
    g_rf0 = int(rm.indices("rf")[0])
    g_parse_conj0 = int(rm.indices("parse_conj")[0])
    g_cortex_it0 = int(rm.indices("cortex_it")[0])
    g_cortex_N0 = int(rm.indices("cortex_N")[0])
    # EVERY pre-existing region must keep its base (additive: only gen_* may be added). This is the strict
    # byte-identity assertion across all regions, not just the load-bearing handful.
    same_region_bases = all(rn in region_names and int(rm.indices(rn)[0]) == base_region_bases[rn]
                            for rn in base_regions)
    n_regions_added = len(region_names) - len(base_regions)
    only_gen_added = bool(set(region_names) - set(base_regions) == {GEN_PERCEPTION, GEN_CONCEPT, GEN_FACT})
    bases_unchanged = bool(same_region_bases and only_gen_added
                           and g_rf0 == base_rf0 and g_parse_conj0 == base_parse_conj0
                           and g_cortex_it0 == base_cortex_it0 and g_cortex_N0 == base_cortex_N0)
    # the gen stack is appended LAST: gen_perception starts exactly where the baseline ended (= base_n), and gen_fact
    # is the very last neuron.
    appended_last = bool(gen["perc_base"] == base_n and gen["fact_last"] == n_neurons - 1)
    byte_identity = bool(bases_unchanged and appended_last)
    print(f"[stage1] (iv-byte) byte-identity: same_all_bases={same_region_bases} only_gen_added={only_gen_added} "
          f"(+{n_regions_added} regions) appended_last={appended_last} "
          f"(rf {base_rf0}->{g_rf0}, parse_conj {base_parse_conj0}->{g_parse_conj0}, "
          f"cortex_it {base_cortex_it0}->{g_cortex_it0}) => {byte_identity}", flush=True)

    cat_ids = gen["gen_cat_ids"]
    held_out = gen["gen_held_out"]
    vis_sets = gen["vis_sets"]

    # ── (i) H5: held-out NOVEL structured-perception cue → concept-category SPIKES on the merged bridge ──
    comp = _build_composer(seed, n_cat)
    h5_hits, h6_hits, conc_s, fact_s, win_fires, margins, answers = [], [], [], [], [], [], []
    for j in held_out:
        cpb, fpb, ct, ft = _read_gen_spikes(bridge, gen, vis_sets[j], xp)
        keyed_cat, catmean = _category_of_concept_spikes(cpb, cat_ids, n_cat)
        true_cat = int(cat_ids[j])
        h5_hits.append(int(keyed_cat == true_cat))
        same = float(cpb[cat_ids == true_cat].mean())
        other = float(cpb[cat_ids != true_cat].mean())
        margins.append(same - other)
        # (ii) H6-hybrid: key the validated composer recall by the spiking concept-category.
        rec = comp.query_patient(_category_word(keyed_cat), ACTION_WORD)
        h6_hits.append(int(rec == PATIENT_WORD and keyed_cat == true_cat))
        conc_s.append(ct); fact_s.append(ft); win_fires.append(float(np.max(catmean)))
        answers.append({"true_cat": true_cat, "keyed_cat": keyed_cat, "recall": rec, "cat_means": catmean})
    h5_acc = float(np.mean(h5_hits))
    h6_acc = float(np.mean(h6_hits))
    h5_margin = float(np.mean(margins))
    heldout_win_fire = float(np.mean(win_fires))
    conc_per_cue = float(np.mean(conc_s))
    fact_per_cue = float(np.mean(fact_s))
    print(f"[stage1] (i)  H5 concept-cat spike acc {h5_acc:.2f} (chance {chance:.2f}) margin {h5_margin:+.3f} | "
          f"concept spikes/cue {conc_per_cue:.0f} | fact spikes/cue {fact_per_cue:.0f}", flush=True)
    print(f"[stage1] (ii) H6-hybrid recall acc {h6_acc:.2f}", flush=True)

    # ── (iii) the no-confab MOAT: a visually-novel NO-category cue must ABSTAIN ──
    rngm = np.random.default_rng(seed * 41 + 9)
    novel_set = novel_no_category_perc_set(gen["gen_W"], gen["gen_top_k"], n_cat, rngm)
    ncpb, nfpb, nct, nft = _read_gen_spikes(bridge, gen, novel_set, xp)
    novel_cat, novel_catmean = _category_of_concept_spikes(ncpb, cat_ids, n_cat)
    novel_win_fire = float(np.max(novel_catmean))
    # the familiarity gate: a known held-out cue drives a HIGH best-category response; a no-category cue drives a
    # LOW, diffuse one → below the gate → NO recall keyed (abstain). Gate at a fraction of the held-out familiarity.
    moat_gate_frac = 0.6
    gate_thresh = heldout_win_fire * moat_gate_frac
    novel_familiar = bool(novel_win_fire >= gate_thresh)
    novel_recall = comp.query_patient(_category_word(novel_cat), ACTION_WORD) if novel_familiar else None
    moat_abstains = bool(novel_recall is None)
    fam_contrast_ok = bool(heldout_win_fire > novel_win_fire * 1.2 + 1e-9)
    print(f"[stage1] (iii) MOAT: held-out win-fire {heldout_win_fire:.2f} vs novel {novel_win_fire:.2f} "
          f"(gate {gate_thresh:.2f}) -> {'ABSTAIN' if moat_abstains else 'CONFAB'} (novel_recall={novel_recall})",
          flush=True)

    # ── (iv) NO-REGRESSION: the existing merged-bridge conversational capabilities still pass ──
    # use the MergedNavConvAgent surface on its OWN co_resident_generalization bridge so the assertions exercise the
    # agent surface co-resident with the generalization stack (a fresh agent build with the gen stack on).
    agent = _build_agent_with_gen(seed)
    conv = _conversational_no_regression(agent)
    conv_ok = bool(conv["parse_ok"] and conv["what_ok"] and conv["who_ok"] and conv["moat_ok"] and conv["desc_ok"])
    conv_moat_breach = bool(not conv["moat_ok"])
    print(f"[stage1] (iv) NO-REGRESSION conv: parse={conv['parse_ok']} what={conv['what_ok']} who={conv['who_ok']} "
          f"moat={conv['moat_ok']} describe={conv['desc_ok']}  (describe='dog'->{conv['describe']})", flush=True)

    # ── the verdict ──
    h5_ok = bool(h5_acc > chance + 1e-9 and h5_margin > 0.0 and conc_per_cue > 0.0)
    h6_ok = bool(h6_acc >= 0.50)
    moat_ok = bool(moat_abstains)                  # HARD: a breach FAILS outright
    moat_breach = bool((not moat_abstains) or conv_moat_breach)
    if moat_breach:
        verdict = "MOAT_BREACH"
    elif h5_ok and h6_ok and moat_ok and byte_identity and conv_ok:
        verdict = "GO"
    elif (conc_per_cue > 0.0 and (h5_acc > chance or h6_acc > 0.0) and moat_ok and byte_identity and conv_ok):
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"

    elapsed = time.time() - t0
    return {
        "seed": int(seed), "backend": backend, "verdict": verdict, "chance": chance, "elapsed_s": elapsed,
        "byte_identity": byte_identity, "bases_unchanged": bases_unchanged, "appended_last": appended_last,
        "base_n_neurons": base_n, "stage1_n_neurons": n_neurons,
        "bases": {"rf": [base_rf0, g_rf0], "parse_conj": [base_parse_conj0, g_parse_conj0],
                  "cortex_it": [base_cortex_it0, g_cortex_it0], "cortex_N": [base_cortex_N0, g_cortex_N0]},
        "vision_structure_margin": gen["gen_set_margin"], "vision_structure_preserved": gen["gen_structure_preserved"],
        "gen_train_diag": {k: int(v) for k, v in gen["gen_train_diag"].items()},
        "held_out": held_out,
        "h5_concept_cat_acc": h5_acc, "h5_margin": h5_margin,
        "concept_spikes_per_cue": conc_per_cue, "fact_spikes_per_cue": fact_per_cue,
        "h6_hybrid_recall_acc": h6_acc,
        "moat": {"heldout_win_fire": heldout_win_fire, "novel_win_fire": novel_win_fire, "gate_thresh": gate_thresh,
                 "novel_familiar": novel_familiar, "novel_recall": novel_recall, "moat_abstains": moat_abstains,
                 "fam_contrast_ok": fam_contrast_ok},
        "no_regression": conv, "conv_ok": conv_ok,
        "answers": answers,
    }


def _build_agent_with_gen(seed):
    """A MergedNavConvAgent built on a co_resident_generalization=True bridge (the gen stack co-resident with the
    full conversational agent surface), so the no-regression assertions exercise the agent ON the stage-1 bridge.
    The agent's __init__ builds its own merged bridge; we pass co_resident_composer=True so the RF composer is
    co-resident too. We post-hoc enable the gen stack by reconstructing through build_merged_nav_conv_bridge — but
    the agent owns its bridge build, so instead we monkeypatch the build kwargs via a thin subclass below."""
    from research.runners.nav_conv_merged_bridge import (
        MergedNavConvAgent, _MergedParserAdapter, build_merged_nav_conv_bridge as _bmb, PARSER_R,
    )

    class _GenAgent(MergedNavConvAgent):
        def __init__(self, seed=42, vocab=None):
            self.seed = int(seed)
            self.co_resident_composer = True
            _D = 128
            self._merged_bridge, self._handles = _bmb(
                seed=seed, vocab=vocab, co_resident_rf=True, rf_D=_D,
                co_resident_generalization=True)
            words = self._handles["vocab"]
            self.composer = MergedRFComposer(
                self._merged_bridge, self._handles["rf_base"], self._handles["rf_size"],
                seed=seed, D=_D, vocab=words, period=200)
            self.parser = _MergedParserAdapter(
                self._merged_bridge, self._handles["conj_arr"], self._handles["role_arr"])
            self._dlpfc_ctx = self._handles["dlpfc_ctx"]
            self._dlpfc_controller = None
            self._dlpfc_graph_key = None
            region_names = self._merged_bridge.region_manager.region_indices_dict()
            assert "parse_conj" in region_names and "dlpfc_wm" in region_names and "rf" in region_names
            assert "gen_concept" in region_names, "FAIL: gen stack not on the agent's merged bridge"
            assert self.composer._merged is self._merged_bridge

    return _GenAgent(seed=seed)


def main():
    ap = argparse.ArgumentParser(description="Unified embodied agent STAGE 1: the generalization stack co-resident "
                                             "on the merged nav+conv bridge (H5+H6 hybrid + the moat + no-regression).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="research/findings/raw/_unified_stage1_merged.json")
    args = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "cupy")

    _, backend = get_backend()
    print(f"[unified STAGE-1] backend={backend} — the generalization stack CO-RESIDENT on the merged nav+conv "
          f"bridge: H5 (concept-category spikes) + H6-hybrid recall + the no-confab moat + NO regression of the "
          f"existing capabilities (byte-identity + the conversational matrix). seed={args.seed}", flush=True)
    r = run_seed(args.seed)
    verdict = r["verdict"]

    print(f"\n{'=' * 116}")
    print(f"  STAGE-1 (seed {r['seed']}): H5 concept-spike acc {r['h5_concept_cat_acc']:.2f} "
          f"(chance {r['chance']:.2f}, margin {r['h5_margin']:+.3f}) | H6-hybrid {r['h6_hybrid_recall_acc']:.2f} | "
          f"moat {'INTACT (abstain)' if r['moat']['moat_abstains'] else 'BREACH (CONFAB)'} | "
          f"byte-id {r['byte_identity']} | conv-no-regression {r['conv_ok']}  ==> {verdict}")
    print(f"{'=' * 116}", flush=True)

    if verdict == "GO":
        print(f"  GO — the GENERALIZATION STACK CO-RESIDES on the ONE merged nav+conv bridge with NO regression: a "
              f"held-out NOVEL structured-perception cue drives the concept-category to SPIKE "
              f"({r['concept_spikes_per_cue']:.0f} concept spikes/cue) in the right category ({r['h5_concept_cat_acc']:.0%} "
              f"> chance {r['chance']:.0%}, margin {r['h5_margin']:+.3f}); the H6-hybrid reads that spiking category "
              f"and keys the VALIDATED RFPhasorComposer recall of the category's fact ({r['h6_hybrid_recall_acc']:.0%}); "
              f"a no-category cue ABSTAINS (the no-confab moat survives); the existing conversational matrix "
              f"(parse + who/what + the moat) still passes; AND the nav/parser/rf/cortex_it index bases are "
              f"BYTE-UNCHANGED vs gen-off (the gen regions appended last). ==> promote to Stage 2 (the live unified "
              f"episode). NO sim/ edit.", flush=True)
    elif verdict == "MOAT_BREACH":
        breach = "the generalization moat (a no-category cue confabulated)" if not r["moat"]["moat_abstains"] \
            else "the conversational no-confab moat (an unstored query was accepted)"
        print(f"  MOAT_BREACH — HARD STOP: {breach}. Do NOT proceed; do NOT loosen the gate to manufacture a GO. "
              f"Localize the familiarity contrast (held-out win-fire {r['moat']['heldout_win_fire']:.2f} vs novel "
              f"{r['moat']['novel_win_fire']:.2f}).", flush=True)
    elif verdict == "PARTIAL":
        print(f"  PARTIAL: the route closes + the moat holds + byte-identity + the conversational matrix passes, but "
              f"below the GO bar (H5 {r['h5_concept_cat_acc']:.0%}, H6 {r['h6_hybrid_recall_acc']:.0%}). Bounded knobs "
              f"(gen_n_concept_per / nmda-ratio / read-steps / top-K / epochs), not walls. Single-seed; the de-risk "
              f"GO config is 0.75-0.92.", flush=True)
    else:
        why = []
        if not r["byte_identity"]:
            why.append("BYTE-IDENTITY broke (the gen regions are NOT appended last / a base shifted)")
        if not r["conv_ok"]:
            why.append("the conversational matrix REGRESSED (parse/who-what/describe) co-resident with the gen stack")
        if r["concept_spikes_per_cue"] <= 0.0:
            why.append("the gen_concept assembly does NOT spike on the merged bridge (raise perc-scale / nmda-ratio)")
        elif r["h5_concept_cat_acc"] <= r["chance"]:
            why.append("the concept-category read is at/below chance co-resident with the nav cascade + dopamine")
        print(f"  NEGATIVE: {'; '.join(why) if why else 'the spiking-concept read is too noisy co-resident'}. "
              f"Moat {'INTACT' if r['moat']['moat_abstains'] else 'BREACH'}. Honest negative + the localized next "
              f"step (the co-residence/representation match is the boundary).", flush=True)

    os.makedirs(os.path.dirname(os.path.join(_REPO, args.out)), exist_ok=True)
    with open(os.path.join(_REPO, args.out), "w") as fh:
        json.dump(r, fh, indent=2, default=str)
    print(f"  [saved] {args.out}\n  Total elapsed: {r['elapsed_s']:.1f}s", flush=True)
    raise SystemExit(0 if verdict == "GO" else (2 if verdict == "PARTIAL" else (3 if verdict == "MOAT_BREACH" else 1)))


if __name__ == "__main__":
    main()
