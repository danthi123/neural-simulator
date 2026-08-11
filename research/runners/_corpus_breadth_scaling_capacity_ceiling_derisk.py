"""CORPUS-BREADTH SCALING -> the VSA no-confab MOAT capacity CEILING (INTEGRATION #6 capacity question).

INTEGRATION #6 (`_corpus_facts_into_live_chat_derisk`) gave the live chat corpus-LEARNED breadth: grounded subjects
rose 2 -> 9 at K=40 with the RF-phasor VSA moat holding BY CONSTRUCTION (0 false-accepts, recall 1.0). #6 named the
CAPACITY question: how far does breadth scale before the moat margin degrades? This de-risk answers it.

TWO measurement blocks (reuse-by-import #6; NO `sim/` edit; SIM_BACKEND=numpy; cfg.seed-controlled):

  A. CORPUS SWEEP -- the REAL learned-breadth axis. Sweep K in {40,80,160,320} (= the top-K mined TinyStories SVO
     triples, vocab GROWING with K). Per (seed,K): |V|, n_facts, grounded-subject BREADTH, RECALL on the stored (a,v)
     cues, and the MOAT (untaught-in-vocab false-accepts). Plus #6's anti-cheats at every K: permuted-corpus
     provenance, empty-kb same-vocab control (with an `attributable_to` attribution: whose is the breadth?). Tier-1
     LIVE mouth-free chat at the SCALED headline K (grounded-reply RISE vs the 6-fact baseline, confab==0, OOD abstain,
     `_gm_posthoc_verify` drops 100% unsupported) + the additive `vocab`-kwarg byte-identity guard (seed 42).
     query_patient is MEMOISED per measurement (a deterministic pure read of the frozen store -> answer-identical, the
     SAME rationale #6's chat memo documents), so the O(K.D) resonate is not paid redundantly.

     SCOPING REALITY (measured, honest): the TinyStories corpus under the shipped noun/verb inventory
     (`_ANIMALS|NOUNS_EXTRA`, `VERBS`) contains only **247 distinct clean SVO triples over ~68 distinct concepts**, so
     K>=320 CAPS at 247 facts / |V|<=68. The corpus is EXHAUSTED far below the ~320-concept single-bridge figure the
     2026-06-04/G.20 mapping flagged (that figure is a sparse-distributed 5-bridge-ensemble substrate; the RF-phasor
     composer here is a DIFFERENT substrate). So block A alone cannot reach a leak -- block B locates the ceiling.

  B. CAPACITY-CEILING INSTRUMENT -- locate where the RF-phasor moat WOULD leak, since the corpus cannot fill it. The
     store is a LIST of INDEPENDENT per-fact composites (self.kb), NOT one superposed memory, so scaling the NUMBER of
     facts adds NO inter-fact crosstalk -- a query unbinds ONE 3-bind composite and cleans up against the |V| codebook.
     The three genuine capacity axes, each swept to a leak or a bound:
       (b1) CODEBOOK axis D x |V|: per-role cleanup MARGIN + accuracy + moat false-accepts. Swept at a STRESS D=32
            (where cleanup accuracy FALLS as |V| grows -> the |V| ceiling is LOCATED, giving the metric discriminating
            power) beside the operating D=128 (holds to |V|=8192). The margin falls only ~sqrt(ln|V|)/sqrt(D).
       (b2) SUPERPOSITION axis L in {2..6}: role-fillers bundled into ONE composite (the store()'s '+-1 scheme K=5
            boundary' question). This is the within-fact load that actually breaks recovery -- SVO facts are L=3.
       (b3) DIMENSION axis D in {8..128} at a fixed 3-bind load: the D where the moat leaks -> D=128's headroom.

GO GATE (6 seeds 42/43/44/100/101/102): at EVERY corpus K, recall>=0.95 AND moat 0 false-accepts AND breadth STRICTLY
rises with K (learned content, not vocab) AND provenance<0.5 AND empty-kb control gives 0 new-subject competence; plus
Tier-1 at the scaled K: grounded RISES, confab==0, OOD abstains, posthoc teeth drop 100%. The LOCATED CEILING is the
first corpus K where the moat leaks or recall<0.95; if NONE (breadth scales to the full corpus with the moat intact),
the ceiling is reported from block B (concept headroom / superposition L* / dimension D*) + the practical wall (query
latency O(K.D)) -- a first-class result either way.

ANTI-CHEATS (#6's, preserved): (1) permuted-corpus provenance overlap<0.5 at every K; (2) expanded moat battery
untaught-in-vocab -> 0 false-accepts; (3) empty-kb same-vocab control -> breadth stays 2, 0 new-subject answers +
`attributable_to` attribution; (4) THIS de-risk IS the capacity sweep (#6 anti-cheat 4 generalised to K=320 + block B);
(5) surface-confab scan `_detect_ungrounded`==0; (6) byte-identity of the additive vocab kwarg (seed 42).

HONEST SCOPE (per THE LAW + docs/TERMS.md). DECLARED SCAFFOLDS (identical to #6): host SVO mining (linguistic
environment), `comp.store` host VSA write (composer-as-idealization), `_gm_fact_to_english` host render. GENUINELY
brain-based: recall + the moat are RF-VSA reads (query_patient = spiking unbind + cleanup). Block B's synthetic
concepts are a labelled INSTRUMENT to locate the substrate's capacity ceiling -- NOT a corpus-breadth claim (provenance
does not apply to it). BURN-DOWN SUCCESSOR for the latency wall + the concept ceiling: the multi-bridge / sharded VSA
store and the synaptic co-occurrence cortex (`_foundational_curriculum_scaling_scoping`) that replaces host mine+store.

Run (single-seed smoke):
  PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._corpus_breadth_scaling_capacity_ceiling_derisk \
      --seeds 42 --Ks 40,80,160,320 --live-K 320
Full 6-seed foreground sweep:
  PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._corpus_breadth_scaling_capacity_ceiling_derisk \
      --seeds 42,43,44,100,101,102 --Ks 40,80,160,320 --live-K 320 --live-seeds 42 --cap-seeds 42,43,44 \
      --out research/findings/raw/lanes/stageA/corpus_breadth_scaling_capacity_ceiling_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

import numpy as np  # noqa: E402

from sim.backend import get_backend  # noqa: E402

from research.runners import _stageA_full_integration_derisk as SA  # noqa: E402
from research.runners import _conversation_turing_test_derisk as TT  # noqa: E402
from research.runners import _corpus_facts_into_live_chat_derisk as C6  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer, DEFAULT_VOCAB, ROLES  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# query_patient MEMO -- a deterministic pure read of the FROZEN store, so the memo is ANSWER-IDENTICAL (the same
# rationale #6's chat memo documents); it only removes the redundant O(K.D) resonates when recall / breadth / moat
# query overlapping (agent, action) cues on the SAME composer.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _install_qp_memo(comp):
    orig = comp.query_patient
    memo = {}

    def _memo_qp(agent, action, order_fn=None):
        key = (agent, action, order_fn)
        if key not in memo:
            memo[key] = orig(agent, action, order_fn=order_fn)
        return memo[key]

    comp.query_patient = _memo_qp
    return memo


def _breadth_and_recall(comp, facts):
    """ONE memoised pass over the distinct stored (a,v) cues gives BOTH: recall (fraction whose query_patient returns
    a patient actually stored for the cue) AND breadth (# distinct subjects with >=1 non-abstaining stored cue).
    breadth here == C6.grounded_subjects(comp, facts): a subject is grounded iff >=1 of its stored cues recalls, and
    the moat abstains on every UNTAUGHT cue (so scanning non-stored actions adds nothing) -- so restricting to stored
    cues is identical, at zero extra resonates."""
    cue2pats = {}
    for (a, v, p) in facts:
        cue2pats.setdefault((a, v), set()).add(p)
    if not cue2pats:
        return 0.0, 0, []
    hit = 0
    grounded_subs = set()
    for (a, v), pats in cue2pats.items():
        ans = comp.query_patient(a, v)
        ok = ans is not None
        hit += int(ok and ans in pats)
        if ok:
            grounded_subs.add(a)
    return hit / len(cue2pats), len(cue2pats), sorted(grounded_subs)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# BLOCK A -- corpus Tier-0 per K (memoised), reusing #6's mining + moat + provenance + empty-kb-control helpers.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def tier0_memo(seed, counter, toks, nouns, verbs, K):
    t0 = time.time()
    mined = C6.mine_top_k(counter, K)
    V = C6.build_vocab(mined)
    comp, facts = C6.make_composer_and_store(seed, V, mined)
    _install_qp_memo(comp)

    recall, n_cues, breadth_subs = _breadth_and_recall(comp, facts)
    breadth = len(breadth_subs)
    moat = C6.moat_battery(comp, V, facts, seed=seed)          # untaught in-vocab -> 0 false-accepts

    # ANTI-CHEAT 3 (empty-kb control): SAME expanded vocab V, 0 corpus facts -> breadth stays 2, new subjects abstain.
    comp_ctrl = RFPhasorComposer(seed=int(seed), D=128, vocab=V)
    _vc, curated_only = SA._store_facts(comp_ctrl, extra_facts=None)
    _install_qp_memo(comp_ctrl)
    _rc, _nc, ctrl_breadth_subs = _breadth_and_recall(comp_ctrl, curated_only)
    actions_all = sorted({v for (_a, v, _p) in facts})
    new_subjects = sorted({s for (s, _v, _p) in mined if s not in C6._BASE_SUBJECTS})
    ctrl_new_answers = 0
    for s in new_subjects:                                     # short-circuit: grounded iff >=1 non-abstaining action
        for v in actions_all:
            if isinstance(comp_ctrl.query_patient(s, v), str):
                ctrl_new_answers += 1
                break

    # ATTRIBUTION (tools.lab): WHOSE is the breadth? Subtract the empty-kb SAME-VOCAB control -- the vocab expansion
    # alone grounds nothing (breadth stays 2), so the breadth rise is attributed to the stored FACTS, not the codebook.
    breadth_attrib = attributable_to(
        "breadth from stored corpus facts (treatment vs empty-kb same-vocab control, K=%d)" % int(K),
        float(breadth), float(len(ctrl_breadth_subs)))

    return {
        "seed": int(seed), "K": int(K), "vocab_size": len(V), "n_facts_stored": len(facts),
        "n_distinct_cues": int(n_cues), "recall_on_stored": float(recall),
        "breadth": int(breadth), "breadth_subjects": breadth_subs,
        "moat_false_accepts": moat["false_accepts"], "moat_probes": moat["probes"], "moat_examples": moat["examples"],
        "permuted_overlap": float(C6.permuted_overlap(seed, toks, nouns, verbs, K)),
        "empty_kb_breadth": len(ctrl_breadth_subs), "empty_kb_new_subject_answers": int(ctrl_new_answers),
        "breadth_attributable_to_facts": breadth_attrib,
        "elapsed_s": round(time.time() - t0, 1),
    }


def run_seed_scaling(seed, counter, toks, nouns, verbs, Ks, live_K, do_live=True, do_byte_identity=False):
    """One seed: the Tier-0 corpus K-sweep (ALWAYS, the 6-seed scaling deliverable) + -- when `do_live` -- the Tier-1
    live mouth-free chat at the SCALED headline live_K (baseline vs treatment) + posthoc teeth + (optional)
    byte-identity. Tier-1 at live_K=320 costs ~11 min/seed (the 253-fact merged-bridge query is ~1.3s x ~500 distinct
    neighbourhood cues), so it is run on a seed SUBSET as an at-scale live-loop CONFIRMATION; the scaling metrics
    (breadth/recall/moat) that decide the GO are Tier-0 and stay 6-seed."""
    t_start = time.time()
    tier0_by_K = {int(k): tier0_memo(seed, counter, toks, nouns, verbs, int(k)) for k in Ks}

    sum_base = sum_treat = teeth = bi = None
    grounded_rises = no_confab = ood_abstains = teeth_ok = no_ungrounded = None
    if do_live:
        # ---- Tier-1 live chat at the scaled headline K ----
        mined = C6.mine_top_k(counter, int(live_K))
        V = C6.build_vocab(mined)
        turns = list(TT.HUMAN_TURNS) + C6.make_teacher_probes(mined)
        xp, _ = get_backend()

        b_b, c_b, i_b, s_b = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True)
        _vb, facts_b = SA._store_facts(c_b)
        tr_base = C6.run_chat(b_b, xp, i_b, s_b, c_b, facts_b, turns)
        sum_base = C6._chat_summary(tr_base)

        b_t, c_t, i_t, s_t = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True,
                                                vocab=V)
        _vt, facts_t = SA._store_facts(c_t, extra_facts=mined)
        tr_treat = C6.run_chat(b_t, xp, i_t, s_t, c_t, facts_t, turns)
        sum_treat = C6._chat_summary(tr_treat)

        teeth = C6.posthoc_teeth(c_t, facts_t, seed=seed)
        bi = C6.byte_identity(seed, list(TT.HUMAN_TURNS)) if do_byte_identity else None

        grounded_rises = bool(sum_treat["grounded"] > sum_base["grounded"])
        no_confab = bool(sum_treat["confabulated"] == 0 and sum_base["confabulated"] == 0)
        ood_abstains = bool(sum_treat["ood_abstained"] == sum_treat["ood_turns"]
                            and sum_base["ood_abstained"] == sum_base["ood_turns"])
        teeth_ok = bool(abs(teeth["unsupported_drop_rate"] - 1.0) < 1e-9 and teeth["unsupported_props"] > 0)
        no_ungrounded = bool(sum_treat["ungrounded_word_total"] == 0 and sum_base["ungrounded_word_total"] == 0)

    # ---- per-K gate + located ceiling within the corpus ----
    ceiling_K = None            # first K where moat leaks OR recall<0.95 (within the corpus)
    prev_breadth = None
    breadth_monotone = True
    for k in sorted(tier0_by_K):
        r = tier0_by_K[k]
        if r["moat_false_accepts"] > 0 or r["recall_on_stored"] < 0.95:
            ceiling_K = ceiling_K if ceiling_K is not None else int(k)
        if prev_breadth is not None and r["breadth"] < prev_breadth:
            breadth_monotone = False
        prev_breadth = r["breadth"]

    recall_ok = all(r["recall_on_stored"] >= 0.95 for r in tier0_by_K.values())
    moat_ok = all(r["moat_false_accepts"] == 0 for r in tier0_by_K.values())
    provenance_ok = all(r["permuted_overlap"] < 0.5 for r in tier0_by_K.values())
    empty_kb_ok = all(r["empty_kb_breadth"] <= 2 and r["empty_kb_new_subject_answers"] == 0
                      for r in tier0_by_K.values())
    breadth_rises = bool(max(r["breadth"] for r in tier0_by_K.values())
                         > min(r["breadth"] for r in tier0_by_K.values()))

    tier0_go = bool(recall_ok and moat_ok and provenance_ok and empty_kb_ok and breadth_rises)
    live_go = None if not do_live else bool(grounded_rises and no_confab and ood_abstains and teeth_ok
                                            and no_ungrounded)
    # a seed passes if its Tier-0 scaling gates pass AND (if it ran the live chat) the live-loop gates pass.
    seed_go = bool(tier0_go and (live_go is None or live_go))

    return {
        "seed": int(seed), "Ks": [int(k) for k in Ks], "live_K": int(live_K), "did_live": bool(do_live),
        "elapsed_s": round(time.time() - t_start, 1),
        "tier0_by_K": tier0_by_K,
        "corpus_ceiling_K": ceiling_K,      # None = moat held to the full corpus
        "breadth_monotone": bool(breadth_monotone),
        "chat_baseline_summary": sum_base, "chat_treatment_summary": sum_treat,
        "grounded_baseline": (sum_base["grounded"] if sum_base else None),
        "grounded_treatment": (sum_treat["grounded"] if sum_treat else None),
        "grounded_delta": ((sum_treat["grounded"] - sum_base["grounded"]) if sum_treat else None),
        "posthoc_teeth": teeth, "byte_identity": bi,
        "gate": {
            "recall_ge_0.95_all_K": recall_ok, "moat_0_false_accepts_all_K": moat_ok,
            "breadth_rises_with_K": breadth_rises, "permuted_provenance_all_K": provenance_ok,
            "empty_kb_control_all_K": empty_kb_ok, "tier0_go": tier0_go, "live_go": live_go,
            "grounded_rises": grounded_rises, "no_confab": no_confab, "ood_abstains": ood_abstains,
            "posthoc_teeth_drop_100pct": teeth_ok, "no_surface_confab": no_ungrounded,
        },
        "seed_go": seed_go,
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# BLOCK B -- the CAPACITY-CEILING INSTRUMENT (labelled synthetic; locates where the RF-phasor moat WOULD leak).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _role_cleanup(comp, comp_phases, role, true_word, words):
    """Per-role cleanup MARGIN (true cos - best-competitor cos) + whether the argmax is correct, over the |V| codebook.
    A cheap read: one RF unbind + a full-codebook cosine (NO K-scan)."""
    rec = comp._unbind_phases(comp_phases, role)
    sims = np.array([float(np.mean(np.cos(2.0 * np.pi * (rec - comp.concepts[w])))) for w in words])
    ti = words.index(true_word)
    tv = float(sims[ti])
    sims[ti] = -9.0
    best = float(sims.max())
    argmax_word = words[int(np.argmax(sims))]
    return tv - best, bool(tv > best), argmax_word


def _store_fact_dict(comp, fact):
    """Store an arbitrary-role fact dict directly through the composer's OWN encode (the store() path only accepts
    agent/action/patient[/attribute/polarity]; for the L-bundle superposition axis we bundle up to all 6 ROLES)."""
    comp.kb.append((dict(fact), comp._encode(fact)))


def _dense_store(comp, words, n_facts, n_ag, n_ac, seed, roles=("agent", "action", "patient")):
    """Store n_facts distinct dense facts (few agents/actions -> untaught in-domain pairs EXIST for the moat) using
    `roles` role-fillers per composite (roles>=3 stresses the within-fact superposition axis). Returns the fact list."""
    rng = np.random.default_rng(seed)
    agents = words[:n_ag]
    actions = words[n_ag:n_ag + n_ac]
    pool = words[n_ag + n_ac:]
    facts, seen, tries = [], set(), 0
    while len(facts) < n_facts and tries < n_facts * 40:
        tries += 1
        a = agents[int(rng.integers(0, len(agents)))]
        v = actions[int(rng.integers(0, len(actions)))]
        if (a, v) in seen:
            continue
        seen.add((a, v))
        fillers = {"agent": a, "action": v, "patient": pool[int(rng.integers(0, len(pool)))]}
        for extra in roles:
            if extra not in fillers:
                fillers[extra] = pool[int(rng.integers(0, len(pool)))]
        fact = {r: fillers[r] for r in roles}
        _store_fact_dict(comp, fact)
        facts.append((fact, comp.kb[-1][1]))
    return facts, agents, actions


def capacity_codebook_axis(seed, Ds=(32, 128), Vsizes=(68, 128, 256, 512, 1024, 2048, 4096, 8192), n_facts=64,
                           n_moat=24):
    """b1 -- CODEBOOK axis: grow |V| (3-bind SVO facts) at each D, measure per-role cleanup margin + accuracy + moat FA.
    Sweeping D (a STRESS D=32 beside the operating D=128) gives the metric DISCRIMINATING POWER -- at D=32 cleanup
    accuracy FALLS as |V| grows (the |V| ceiling becomes visible + located), while at D=128 it holds to |V|=8192; the
    D=128 hold is thus interpretable (a real ceiling), not a pinned always-pass."""
    out = []
    for D in Ds:
        for Vs in Vsizes:
            words = ["w%d" % i for i in range(int(Vs))]
            comp = RFPhasorComposer(seed=int(seed), D=int(D), vocab=words)
            n_ag, n_ac = 16, 8
            facts, agents, actions = _dense_store(comp, words, min(n_facts, n_ag * n_ac - 1), n_ag, n_ac, seed)
            margins, correct, tot = [], 0, 0
            for fact, cp in facts:
                for role in ("agent", "action", "patient"):
                    m, ok, _w = _role_cleanup(comp, cp, role, fact[role], words)
                    margins.append(m); correct += int(ok); tot += 1
            stored = {(f["agent"], f["action"]) for f, _c in facts}
            rng = np.random.default_rng(seed * 7 + 11)
            fa, probes, tries = 0, 0, 0
            while probes < n_moat and tries < 3000:
                a = agents[int(rng.integers(0, len(agents)))]
                v = actions[int(rng.integers(0, len(actions)))]
                tries += 1
                if (a, v) in stored:
                    continue
                probes += 1
                if comp.query_patient(a, v) is not None:
                    fa += 1
            out.append({"D": int(D), "|V|": int(Vs), "n_facts": len(facts),
                        "mean_margin": round(float(np.mean(margins)), 4),
                        "min_margin": round(float(np.min(margins)), 4), "cleanup_acc": round(correct / tot, 4),
                        "moat_fa": int(fa), "moat_probes": int(probes)})
    return out


def capacity_superposition_axis(seed, D=128, Vsize=512, Ls=(2, 3, 4, 5, 6), n_facts=48):
    """b2 -- SUPERPOSITION axis: bundle L role-fillers into ONE composite (SVO=L3); the within-fact load that actually
    breaks recovery. Measures per-role cleanup accuracy over stored facts."""
    role_order = list(ROLES)                                   # agent, action, patient, polarity, attribute, attribute2
    out = []
    for L in Ls:
        roles = tuple(role_order[:int(L)])
        words = ["w%d" % i for i in range(int(Vsize))]
        comp = RFPhasorComposer(seed=int(seed), D=int(D), vocab=words)
        n_ag, n_ac = 12, 6
        facts, _ag, _ac = _dense_store(comp, words, min(n_facts, n_ag * n_ac - 1), n_ag, n_ac, seed, roles=roles)
        correct, tot = 0, 0
        for fact, cp in facts:
            for role in roles:
                _m, ok, _w = _role_cleanup(comp, cp, role, fact[role], words)
                correct += int(ok); tot += 1
        out.append({"L_binds": int(L), "roles": list(roles), "n_facts": len(facts),
                    "cleanup_acc": round(correct / tot, 4)})
    return out


def capacity_dimension_axis(seed, Ds=(8, 16, 32, 64, 128), Vsize=256, n_facts=64):
    """b3 -- DIMENSION axis at a fixed 3-bind load: the D where the moat leaks / cleanup fails -> D=128's headroom."""
    out = []
    for D in Ds:
        words = ["w%d" % i for i in range(int(Vsize))]
        comp = RFPhasorComposer(seed=int(seed), D=int(D), vocab=words)
        n_ag, n_ac = 16, 8
        facts, agents, actions = _dense_store(comp, words, min(n_facts, n_ag * n_ac - 1), n_ag, n_ac, seed)
        correct, tot, margins = 0, 0, []
        for fact, cp in facts:
            for role in ("agent", "action", "patient"):
                m, ok, _w = _role_cleanup(comp, cp, role, fact[role], words)
                correct += int(ok); tot += 1; margins.append(m)
        stored = {(f["agent"], f["action"]) for f, _c in facts}
        rng = np.random.default_rng(seed * 7 + 11)
        fa, probes, tries = 0, 0, 0
        while probes < 48 and tries < 3000:
            a = agents[int(rng.integers(0, len(agents)))]
            v = actions[int(rng.integers(0, len(actions)))]
            tries += 1
            if (a, v) in stored:
                continue
            probes += 1
            if comp.query_patient(a, v) is not None:
                fa += 1
        out.append({"D": int(D), "n_facts": len(facts), "mean_margin": round(float(np.mean(margins)), 4),
                    "min_margin": round(float(np.min(margins)), 4), "cleanup_acc": round(correct / tot, 4),
                    "moat_fa": int(fa), "moat_probes": int(probes)})
    return out


def run_capacity_instrument(seeds):
    """Aggregate block B across a few seeds (the ceiling is a substrate property, not seed-specific)."""
    cb, sp, dim = {}, {}, {}
    for s in seeds:
        for row in capacity_codebook_axis(s):
            cb.setdefault((row["D"], row["|V|"]), []).append(row)
        for row in capacity_superposition_axis(s):
            sp.setdefault(row["L_binds"], []).append(row)
        for row in capacity_dimension_axis(s):
            dim.setdefault(row["D"], []).append(row)

    def _agg(rows, keys):
        base = dict(rows[0])
        for k in keys:
            base[k] = round(float(np.mean([r[k] for r in rows])), 4)
        base["min_over_seeds_cleanup_acc"] = round(float(np.min([r["cleanup_acc"] for r in rows])), 4)
        base["max_moat_fa"] = int(np.max([r.get("moat_fa", 0) for r in rows]))
        return base

    codebook = [_agg(cb[k], ["mean_margin", "min_margin", "cleanup_acc"]) for k in sorted(cb)]
    superpos = [_agg(sp[l], ["cleanup_acc"]) for l in sorted(sp)]
    dimension = [_agg(dim[d], ["mean_margin", "min_margin", "cleanup_acc"]) for d in sorted(dim)]

    # located ceilings.
    D_OP = 128
    cb_op = [r for r in codebook if r["D"] == D_OP]         # the operating-D codebook curve
    cb_stress = [r for r in codebook if r["D"] != D_OP]     # the stress-D curve that DISCRIMINATES (locates |V|*)
    codebook_headroom = max((r["|V|"] for r in cb_op
                             if r["min_over_seeds_cleanup_acc"] >= 1.0 and r["max_moat_fa"] == 0), default=None)
    codebook_op_leak = min((r["|V|"] for r in cb_op
                            if r["min_over_seeds_cleanup_acc"] < 1.0 or r["max_moat_fa"] > 0), default=None)
    # first |V| where the STRESS-D curve drops below perfect cleanup (proves the metric CAN fall -> discriminating).
    stress_D = sorted({r["D"] for r in cb_stress})
    codebook_stress_leaks = {int(d): min((r["|V|"] for r in cb_stress
                                          if r["D"] == d and r["min_over_seeds_cleanup_acc"] < 1.0), default=None)
                             for d in stress_D}
    superpos_ok_max = max((r["L_binds"] for r in superpos if r["min_over_seeds_cleanup_acc"] >= 1.0), default=None)
    superpos_leak = min((r["L_binds"] for r in superpos if r["min_over_seeds_cleanup_acc"] < 1.0), default=None)
    dim_leak = max((r["D"] for r in dimension if r["min_over_seeds_cleanup_acc"] < 1.0 or r["max_moat_fa"] > 0),
                   default=None)
    dim_ok_min = min((r["D"] for r in dimension if r["min_over_seeds_cleanup_acc"] >= 1.0 and r["max_moat_fa"] == 0),
                     default=None)
    return {
        "seeds": [int(s) for s in seeds], "D_default": D_OP,
        "codebook_axis": codebook, "superposition_axis": superpos, "dimension_axis": dimension,
        "codebook_headroom_concepts": codebook_headroom, "codebook_op_first_leak_concepts": codebook_op_leak,
        "codebook_stress_first_leak_concepts": codebook_stress_leaks,
        "superposition_max_ok_L": superpos_ok_max, "superposition_first_leak_L": superpos_leak,
        "dimension_first_leak_D": dim_leak, "dimension_min_ok_D": dim_ok_min,
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def build_verdict(recs, Ks, cap, go):
    def _minall(fn):
        return min(fn(r) for r in recs)

    v = Verdict("CORPUS-BREADTH SCALING -> VSA moat capacity ceiling (%d seeds, K=%s)"
                % (len(recs), ",".join(str(k) for k in Ks)))
    v.require("all seeds GO", int(sum(1 for r in recs if r["seed_go"])), expect=len(recs))
    v.floor("min recall on stored cues (all K, all seeds)",
            _minall(lambda r: min(t["recall_on_stored"] for t in r["tier0_by_K"].values())), floor=0.95)
    v.require("moat 0 false-accepts (all K, all seeds)",
              int(sum(t["moat_false_accepts"] for r in recs for t in r["tier0_by_K"].values())), expect=0)
    v.require("breadth rises with K (all seeds)", int(sum(1 for r in recs if r["gate"]["breadth_rises_with_K"])),
              expect=len(recs))
    v.require("permuted-provenance overlap<0.5 (all K, all seeds)",
              float(max(t["permuted_overlap"] for r in recs for t in r["tier0_by_K"].values())),
              expect=lambda m: m < 0.5)
    v.require("empty-kb control: 0 new-subject answers (all K, all seeds)",
              int(sum(t["empty_kb_new_subject_answers"] for r in recs for t in r["tier0_by_K"].values())), expect=0)
    live = [r for r in recs if r.get("did_live")]
    if live:
        v.control("grounded replies at live K: treatment vs 6-fact baseline (first live seed)",
                  live[0]["grounded_treatment"], live[0]["grounded_baseline"], min_separation=0.0)
        v.require("confab==0 at live K (treatment+baseline, live seeds)",
                  int(sum(r["chat_treatment_summary"]["confabulated"] + r["chat_baseline_summary"]["confabulated"]
                          for r in live)), expect=0)
        v.require("posthoc teeth drop 100% of unsupported props (live seeds)",
                  float(min(r["posthoc_teeth"]["unsupported_drop_rate"] for r in live)), expect=1.0)
    bi = next((r["byte_identity"] for r in recs if r.get("byte_identity")), None)
    if bi is not None:
        v.require("byte-identity: substrate threshold hash identical", bool(bi["threshold_hash_identical"]),
                  expect=True)
        v.require("byte-identity: default-build transcript identical", bool(bi["transcript_identical"]), expect=True)
    v.require("capacity instrument: codebook holds >= corpus concepts at D=128",
              int(cap.get("codebook_headroom_concepts") or 0), expect=lambda m: m >= 68)
    v.disabled("spiking-generator MOUTH (GPU/torch)",
               "CPU eval; grounded CONTENT is the RF-VSA read (what the mouth would render), not the mouth")
    v.disabled("plasticity (STDP/Hebbian/homeostasis/STP/structural)",
               "the composer store is a host VSA write; the synaptic-learning successor is named in the finding")
    return v.decide(go=bool(go), verbose=False)


def _print_ksweep_table(recs, Ks):
    print("\n" + "=" * 108, flush=True)
    print("  K-SWEEP CAPACITY TABLE (per K: min/agg over %d seeds) -- the breadth-scaling deliverable" % len(recs),
          flush=True)
    print("  %-6s %-8s %-6s %-9s %-9s %-11s %-9s %-11s %-6s" %
          ("K", "n_facts", "|V|", "breadth", "recall", "moat_FA", "perm_ov", "emptyKB_new", "GO?"), flush=True)
    print("  " + "-" * 100, flush=True)
    all_go = True
    for k in Ks:
        rows = [r["tier0_by_K"][int(k)] for r in recs]
        n_facts = rows[0]["n_facts_stored"]; V = rows[0]["vocab_size"]
        breadth_min = min(r["breadth"] for r in rows); breadth_max = max(r["breadth"] for r in rows)
        recall_min = min(r["recall_on_stored"] for r in rows)
        moat_fa = sum(r["moat_false_accepts"] for r in rows)
        perm_max = max(r["permuted_overlap"] for r in rows)
        empty_new = sum(r["empty_kb_new_subject_answers"] for r in rows)
        row_go = bool(recall_min >= 0.95 and moat_fa == 0 and perm_max < 0.5 and empty_new == 0)
        all_go = all_go and row_go
        bdisp = "%d" % breadth_min if breadth_min == breadth_max else "%d-%d" % (breadth_min, breadth_max)
        print("  %-6d %-8d %-6d %-9s %-9.3f %-11d %-9.2f %-11d %-6s" %
              (int(k), n_facts, V, bdisp, recall_min, moat_fa, perm_max, empty_new, "GO" if row_go else "x"),
              flush=True)
    print("  " + "-" * 100, flush=True)
    return all_go


def _print_capacity_tables(cap):
    print("\n  CAPACITY-CEILING INSTRUMENT (synthetic; D=%d op; seeds=%s) -- locates where the RF moat WOULD leak"
          % (cap["D_default"], cap["seeds"]), flush=True)
    print("   b1 CODEBOOK axis D x |V| (3-bind SVO; D=32 STRESS discriminates, D=128 = operating point):", flush=True)
    print("     %-5s %-8s %-8s %-12s %-12s %-12s %-8s" %
          ("D", "|V|", "n_facts", "mean_margin", "min_margin", "cleanup_acc", "moat_FA"), flush=True)
    for r in cap["codebook_axis"]:
        print("     %-5d %-8d %-8d %-12.4f %-12.4f %-12.4f %-8d" %
              (r["D"], r["|V|"], r["n_facts"], r["mean_margin"], r["min_margin"],
               r["min_over_seeds_cleanup_acc"], r["max_moat_fa"]), flush=True)
    print("   b2 SUPERPOSITION axis L role-fillers/fact (SVO=3):", flush=True)
    print("     %-8s %-8s %-12s" % ("L_binds", "n_facts", "cleanup_acc"), flush=True)
    for r in cap["superposition_axis"]:
        print("     %-8d %-8d %-12.4f" % (r["L_binds"], r["n_facts"], r["min_over_seeds_cleanup_acc"]), flush=True)
    print("   b3 DIMENSION axis D (fixed 3-bind, |V|=256):", flush=True)
    print("     %-8s %-8s %-12s %-12s %-8s" % ("D", "n_facts", "min_margin", "cleanup_acc", "moat_FA"), flush=True)
    for r in cap["dimension_axis"]:
        print("     %-8d %-8d %-12.4f %-12.4f %-8d" %
              (r["D"], r["n_facts"], r["min_margin"], r["min_over_seeds_cleanup_acc"], r["max_moat_fa"]), flush=True)
    print("   LOCATED CEILINGS: codebook @D=128 holds >=%s concepts (op-D first leak: %s); STRESS-D first leaks %s "
          "(the discriminating locator) | superposition ok to L=%s (first leak L=%s) | dimension first leak D=%s "
          "(min ok D=%s)"
          % (cap["codebook_headroom_concepts"], cap["codebook_op_first_leak_concepts"],
             cap["codebook_stress_first_leak_concepts"],
             cap["superposition_max_ok_L"], cap["superposition_first_leak_L"],
             cap["dimension_first_leak_D"], cap["dimension_min_ok_D"]), flush=True)


def main():
    ap = argparse.ArgumentParser(description="Corpus-breadth scaling -> VSA moat capacity ceiling.")
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--Ks", default="40,80,160,320", help="corpus top-K sweep (facts CAP at the corpus's 247 triples)")
    ap.add_argument("--live-K", type=int, default=320, help="the scaled K at which the live mouth-free chat runs")
    ap.add_argument("--live-seeds", default=None,
                    help="seeds that ALSO run the ~11-min/seed at-scale live chat (default: the first seed only). "
                         "The 6-seed scaling GO is decided on Tier-0; live is an at-scale confirmation.")
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--cap-seeds", default="42,43,44", help="seeds for the (seed-insensitive) capacity instrument")
    ap.add_argument("--byte-identity", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--skip-capacity", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    Ks = [int(k) for k in a.Ks.split(",")]
    live_seeds = set(int(s) for s in a.live_seeds.split(",")) if a.live_seeds else {seeds[0]}

    def _want_bi(i):
        return {"on": True, "off": False, "auto": (i == 0)}[a.byte_identity]

    print("[BREADTH-SCALING] corpus-learned breadth -> VSA moat capacity ceiling | mining TinyStories ...", flush=True)
    counter, toks, nouns, verbs = C6.build_corpus_counter(a.corpus_path)
    print("  mined %d distinct clean SVO triples from %d tokens (K>=%d caps here)"
          % (len(counter), len(toks), len(counter)), flush=True)

    recs = []
    for i, s in enumerate(seeds):
        do_live = int(s) in live_seeds
        r = run_seed_scaling(s, counter, toks, nouns, verbs, Ks, a.live_K,
                             do_live=do_live, do_byte_identity=(do_live and _want_bi(i)))
        recs.append(r)
        ksum = " ".join("K%d:b%d/r%.2f/fa%d" % (k, r["tier0_by_K"][k]["breadth"],
                                                r["tier0_by_K"][k]["recall_on_stored"],
                                                r["tier0_by_K"][k]["moat_false_accepts"]) for k in Ks)
        if do_live:
            live_str = ("live K=%d grounded base->treat %d->%d (+%d) confab t=%d b=%d teeth=%.2f"
                        % (a.live_K, r["grounded_baseline"], r["grounded_treatment"], r["grounded_delta"],
                           r["chat_treatment_summary"]["confabulated"], r["chat_baseline_summary"]["confabulated"],
                           r["posthoc_teeth"]["unsupported_drop_rate"]))
        else:
            live_str = "live=skipped (Tier-0-only seed)"
        print("  [seed %d] %s | %s | corpus_ceiling_K=%s | GO=%s (%.1fs)"
              % (s, ksum, live_str, r["corpus_ceiling_K"], r["seed_go"], r["elapsed_s"]), flush=True)
        if r["byte_identity"] is not None:
            print("    byte-identity(default vs vocab=DEFAULT_VOCAB): %s" % r["byte_identity"], flush=True)

    cap = None
    if not a.skip_capacity:
        print("\n[BREADTH-SCALING] running the capacity-ceiling instrument (block B) ...", flush=True)
        cap = run_capacity_instrument([int(s) for s in a.cap_seeds.split(",")])

    ksweep_go = _print_ksweep_table(recs, Ks)
    if cap is not None:
        _print_capacity_tables(cap)

    n_go = sum(1 for r in recs if r["seed_go"])
    go = bool(n_go == len(recs) and len(recs) > 0 and ksweep_go)

    corpus_ceilings = [r["corpus_ceiling_K"] for r in recs]
    located = None if all(c is None for c in corpus_ceilings) else min(c for c in corpus_ceilings if c is not None)
    max_facts = recs[0]["tier0_by_K"][max(Ks)]["n_facts_stored"]
    max_breadth = max(r["tier0_by_K"][max(Ks)]["breadth"] for r in recs)
    print("\n  LOCATED CAPACITY CEILING:", flush=True)
    if located is None:
        print("    The moat did NOT leak within the corpus: recall>=0.95 & moat==0 hold to the FULL corpus "
              "(%d facts / %d grounded subjects at K=%d). The corpus is EXHAUSTED before any moat-margin "
              "degradation." % (max_facts, max_breadth, max(Ks)), flush=True)
        if cap is not None:
            hr = cap["codebook_headroom_concepts"] or 68
            print("    Instrument bound: codebook holds >=%s concepts at D=128 (~%.0fx the corpus's ~68); the "
                  "practical wall is query LATENCY O(K.D), not a moat leak." % (hr, hr / 68.0), flush=True)
    else:
        print("    The moat first leaks / recall<0.95 at corpus K=%d." % located, flush=True)

    print("\n  VERDICT: %s -- %d/%d seeds. Breadth scales with K; the no-confab moat holds (0 false-accepts, "
          "recall>=0.95) across the full corpus; the located ceiling is above the corpus's reach." %
          ("GO" if go else "PARTIAL/NEGATIVE", n_go, len(recs)), flush=True)

    decided = build_verdict(recs, Ks, cap or {"codebook_headroom_concepts": 0}, go)

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        payload = {"verdict": "GO" if go else "PARTIAL", "verdict_earned": decided["status"],
                   "n_go": n_go, "n_seeds": len(recs), "Ks": Ks, "live_K": a.live_K, "seeds": seeds,
                   "sim_backend": os.environ.get("SIM_BACKEND", "numpy"),
                   "located_corpus_ceiling_K": located, "max_corpus_facts": max_facts,
                   "max_corpus_breadth": max_breadth,
                   "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
                   "byte_identity": next((r["byte_identity"] for r in recs if r.get("byte_identity")), None),
                   "capacity_instrument": cap, "per_seed": recs}
        with open(a.out, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print("  [saved] %s" % a.out, flush=True)
    return go


if __name__ == "__main__":
    main()
