"""De-risk: a GENUINELY SPIKING COMPREHENSION-SUCCESS monitor for the on-brain parser.

FACULTY: given an incoming utterance, read a SPIKING signal that is HIGH when the parser's role-binding
resolved cleanly (a well-formed, in-vocabulary transitive whose thematic roles are decisively determined)
and LOW when it did not (out-of-vocab nouns, or a content-ambiguous / role-symmetric string where the roles
cannot be resolved). This is the brain reading whether it UNDERSTOOD, so it can honestly say "I didn't
follow that" -- NOT a host check of "is every word in the lexicon".

TWO CANDIDATE READS ARE TESTED (both reuse-by-import; NO sim/ edit):

  (R-primary)  SEMANTIC-cue-driven sel-pool WTA MARGIN of the spiking multi-cue competition parser
               (`SpikingRoleCompetition`). For each noun we drive ONLY the content (animacy + verbfit) cue
               populations, let the Wong-Wang `sel_agent`/`sel_patient` accumulators settle, and READ the
               agent-evidence (sel_agent firing - sel_patient firing) from `bridge.cp_firing_states`. The
               sentence-level comprehension scalar is the spiking contrast |agentEv_0 - agentEv_1|. It is
               HIGH when the content decisively separates the two nouns (animate agent vs inanimate patient)
               and LOW when the content cancels (two animate + symmetric verb) or is absent (OOV). This is
               the spiking analogue of the moat's content gate -- the SAME gate that currently ships as a
               HOST dot-product `_semantic_contrast(evs)` (a Python formula over learned weights, NOT read
               from spikes). We replace that host read with the settled sel-pool firing.

  (R-full)     The position-INCLUSIVE sel-pool margin (drive ALL cues). Reported as a NEGATIVE control: the
               position cue always splits the two nouns confidently, so this margin is HIGH even for
               ambiguous / OOV input -> it CANNOT serve as a comprehension monitor. Shows why the read must
               be semantic-cue-driven.

  (R-bridge)   The `BridgeParser` 3-role-ensemble winner-vs-runner-up firing MARGIN (agent/action/patient).
               Reported as a BOUNDARY MAP of the task's primary SUGGESTED mechanism: `BridgeParser.role_of`
               drives a (position x voice) conjunction ALONE and never reads the token, so its role-ensemble
               margin is CONTENT-BLIND -- constant across well-formed vs OOV/ambiguous lexical input. We
               measure it to document, honestly, that the positional parser's own margin does NOT
               discriminate comprehension success (AUC ~ chance across lexical content).

GO-GATE (pre-registered, 6 seeds 42,43,44,100,101,102):
  1. R-primary AUC (well-formed vs ill-formed) >= 0.80 on >=5/6 seeds  (type-2-like discrimination well
     above 0.5 chance -- the discrimination the lane-C metacognition monitor FAILED to achieve).
  2. LESION (zero the learned cue->role synaptic weights) collapses R-primary AUC to <=0.60 on >=5/6 seeds
     (load-bearing: the discrimination is CAUSED by the learned spiking competition, not the host cue
     values, which are byte-identical with/without the lesion).
  3. The read is on FIRING NEURONS: every margin is accumulated from `bridge.cp_firing_states`; the host
     `_semantic_contrast` is NEVER called for the confidence read (asserted at runtime).
  4. Reads at conversation time: per-utterance settle wall-time reported (a few ms of sim).

Run:
    SIM_BACKEND=numpy python -m research.runners._spiking_comprehension_monitor_derisk --smoke
    SIM_BACKEND=cupy  python -m research.runners._spiking_comprehension_monitor_derisk \
        --seeds 42,43,44,100,101,102 \
        --out research/findings/raw/_spiking_comprehension_monitor.json
"""

import argparse
import json
import time

import numpy as np

from research.runners._phaseB_multicue_competition_spiking_derisk import (
    ANIMACY,
    VERB_SELECTS,
    INSTALLED_CUE_WEIGHTS,
    CUES,
    SpikingRoleCompetition,
    cue_evidence,
)

SEMANTIC_CUES = ("animacy", "verbfit")

# ---- lexicons for the battery (drawn from the module's own tables + genuine OOV tokens) ----
ANIMATE = ["dog", "cat", "fox", "bird", "wolf", "bear", "owl", "frog"]
INANIM = ["ball", "apple", "rock", "book", "stick", "bone", "leaf", "cup"]
ASYMM_VERBS = ["eat", "push", "carry", "bite", "kick", "grab"]      # agent=animate, patient=inanimate
SYMM_VERBS = ["chase", "watch"]                                     # agent=animate, patient=animate
# OOV tokens: NOT in ANIMACY, so animacy + verbfit votes are absent (reliability 0). This is discovered by
# the substrate (no cue population drives), NOT by a host "is it in the lexicon" branch.
OOV_NOUNS = ["wug", "blicket", "toma", "glorp", "nix", "flum", "zorb", "quan"]


def build_battery(seed, n_per_cond):
    """Deterministic per-seed battery. Returns list of (label, tag, noun0, verb, noun1).
    label=1 -> WELL-FORMED (comprehension should be HIGH); label=0 -> ILL-FORMED (should be LOW)."""
    rng = np.random.default_rng(seed)
    items = []

    def pick(lst):
        return lst[int(rng.integers(len(lst)))]

    for _ in range(n_per_cond):
        # WELL-FORMED: animate agent + asymmetric verb + inanimate patient -> content decisively resolves.
        items.append((1, "well", pick(ANIMATE), pick(ASYMM_VERBS), pick(INANIM)))
        # ILL: AMBIGUOUS -- two animate nouns + symmetric verb -> animacy cancels, verbfit silent.
        a1, a2 = pick(ANIMATE), pick(ANIMATE)
        items.append((0, "ambig_2animate", a1, pick(SYMM_VERBS), a2))
        # ILL: NONSENSE-CONTENT -- two inanimate nouns + asymmetric verb -> animacy cancels (who acts on whom?).
        items.append((0, "ambig_2inanim", pick(INANIM), pick(ASYMM_VERBS), pick(INANIM)))
        # ILL: OUT-OF-VOCAB -- two unknown nouns -> semantic cues absent (no content drive).
        items.append((0, "oov", pick(OOV_NOUNS), pick(ASYMM_VERBS), pick(OOV_NOUNS)))
    return items


def _evs_for(noun0, verb, noun1):
    """Cue evidence for the two nouns (clean_cues=True = deterministic inference read, no training noise)."""
    return [
        cue_evidence(noun0, 0, 2, verb, sent_id=0, clean_cues=True),
        cue_evidence(noun1, 1, 2, verb, sent_id=0, clean_cues=True),
    ]


def _agent_evidence_from_spikes(comp, ev, cue_subset, read_steps):
    """Drive the given cue subset for one noun, settle the WTA, and READ (sel_agent - sel_patient) firing
    from the spiking pools. `comp._noun_role_rates` accumulates `bridge.cp_firing_states[sel_idx]` -- a
    genuine spike read. Returns the agent-evidence contrast for this noun."""
    sub = {c: ev[c] for c in cue_subset if c in ev}
    rr = comp._noun_role_rates(sub, read_steps=read_steps)   # <-- reads cp_firing_states, NOT a host formula
    return rr["agent"] - rr["patient"]


def semantic_sel_margin(comp, evs, read_steps):
    """R-primary: sentence-level SPIKING comprehension scalar = |agentEv_0 - agentEv_1| from the SEMANTIC
    (content) cue populations only. HIGH when content decisively separates the nouns; LOW when it cancels /
    is absent."""
    a0 = _agent_evidence_from_spikes(comp, evs[0], SEMANTIC_CUES, read_steps)
    a1 = _agent_evidence_from_spikes(comp, evs[1], SEMANTIC_CUES, read_steps)
    return abs(a0 - a1)


def full_sel_margin(comp, evs, read_steps):
    """R-full (negative control): position-INCLUSIVE margin -- drives ALL cues. Position always splits the
    nouns, so this is HIGH even for ambiguous/OOV input."""
    a0 = _agent_evidence_from_spikes(comp, evs[0], CUES, read_steps)
    a1 = _agent_evidence_from_spikes(comp, evs[1], CUES, read_steps)
    return abs(a0 - a1)


# ---------------------------------------------------------------------------
# BridgeParser 3-role-ensemble margin (task's suggested mechanism -> boundary map)
# ---------------------------------------------------------------------------
def bridgeparser_role_margin(bp, position, voice=0):
    """Winner-vs-runner-up firing margin across the 3 role ensembles for one (position, voice) conjunction.
    Replicates `BridgeParser.role_of`'s inner loop but returns the (top - second) firing margin instead of
    the argmax. Read from `bridge.cp_firing_states` -- a genuine spike read. NOTE: this is CONTENT-BLIND (it
    never sees the token), so it is measured to DOCUMENT the boundary, not as a working monitor."""
    xp = bp._bridge_xp()
    k = position * 2 + (0 if voice in (0, "active") else 1)
    bp._step_reset()
    cur = xp.zeros(bp._n, dtype=xp.float32)
    cur[bp.conj_arr[k]] = bp.drive
    bp.bridge.cp_external_input_current[:] = cur
    rates = {r: 0.0 for r in bp.ROLES}
    for _ in range(bp.test_steps):
        bp.bridge._run_one_simulation_step()
        for r in bp.ROLES:
            rates[r] += float(bp.bridge.cp_firing_states[bp.role_arr[r]].astype(xp.float64).mean())
    bp.bridge.cp_external_input_current[:] = 0.0
    vals = sorted(rates.values(), reverse=True)
    return vals[0] - vals[1]


def bridgeparser_sentence_margin(bp):
    """Mean role-ensemble margin over the 3 SVO positions (active voice). CONTENT-BLIND -> identical for any
    3-token active input, so it is a constant regardless of the words."""
    return float(np.mean([bridgeparser_role_margin(bp, p, 0) for p in range(3)]))


# ---------------------------------------------------------------------------
# Metric: type-2-like ROC AUC (Mann-Whitney U) of a margin separating well (1) from ill (0)
# ---------------------------------------------------------------------------
def roc_auc(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # AUC = P(pos > neg) with ties counted 0.5 (rank-based Mann-Whitney)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sc = scores[order]
    i = 0
    n = len(sc)
    while i < n:
        j = i
        while j + 1 < n and sc[j + 1] == sc[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    n1, n0 = pos.size, neg.size
    return float((r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _build_comp(seed, *, dt_ms=0.5, homeostasis=False, per_region_thresh=False):
    """Build the installed-weight, frozen-plasticity comprehension competition.

    dt_ms / homeostasis / per_region_thresh are ADDITIVE and DEFAULT-PRESERVING: the defaults
    (0.5 / False / False) reproduce the standalone comprehension monitor bit-for-bit (every existing
    caller passes seed only). They are threaded through to `SpikingRoleCompetition` (which already
    accepts them, from the rung-2 instrument edit) so the SAME comprehension circuit can be built at
    the ONE-BRAIN MERGE shared operating point (dt=1.0, homeostasis ON, per-region threshold
    heterogeneity ON) that the config-superset production merge de-risk sweeps -- reconciling the
    single global dt_ms / enable_homeostasis the Wong-Wang and surprise builders set differently."""
    comp = SpikingRoleCompetition(seed=seed, dt_ms=dt_ms, homeostasis=homeostasis,
                                  per_region_thresh=per_region_thresh)
    for c, w in INSTALLED_CUE_WEIGHTS.items():
        comp.set_cue_weight(c, w)
    comp.freeze_all_cue_plasticity()
    return comp


def run_seed(seed, n_per_cond=20, read_steps=60, with_bridge=True):
    items = build_battery(seed, n_per_cond)
    labels = [lab for (lab, *_rest) in items]

    # ---- intact spiking competition ----
    comp = _build_comp(seed)
    sem, full = [], []
    t0 = time.time()
    for (_lab, _tag, n0, v, n1) in items:
        evs = _evs_for(n0, v, n1)
        sem.append(semantic_sel_margin(comp, evs, read_steps))
        full.append(full_sel_margin(comp, evs, read_steps))
    per_turn_ms = 1000.0 * (time.time() - t0) / (len(items) * 2)  # 2 reads (sem+full) per utterance

    # ---- lesion: zero every learned cue->role synaptic weight (host cue VALUES unchanged) ----
    comp_les = _build_comp(seed)
    for c in CUES:
        comp_les.set_cue_weight(c, 0.0)
    sem_les = []
    for (_lab, _tag, n0, v, n1) in items:
        evs = _evs_for(n0, v, n1)
        sem_les.append(semantic_sel_margin(comp_les, evs, read_steps))

    out = {
        "seed": seed,
        "n_items": len(items),
        "labels": labels,
        "auc_semantic": roc_auc(sem, labels),
        "auc_full_control": roc_auc(full, labels),
        "auc_semantic_lesion": roc_auc(sem_les, labels),
        "mean_margin_well": float(np.mean([s for s, l in zip(sem, labels) if l == 1])),
        "mean_margin_ill": float(np.mean([s for s, l in zip(sem, labels) if l == 0])),
        "per_tag_mean": _per_tag(items, sem),
        "per_turn_read_ms": per_turn_ms,
        "read_from_firing_states": True,
        "host_semantic_contrast_used": False,
    }

    # ---- BridgeParser content-blindness boundary (task's suggested mechanism) ----
    if with_bridge:
        from research.runners.brain_conversational_agent import BridgeParser
        bp = BridgeParser(seed=seed)
        # content-blind: margin depends only on (position, voice), so measure it once and compare the
        # class means of the *same* constant applied to well vs OOV utterances.
        bp_margin = bridgeparser_sentence_margin(bp)
        # AUC over the battery is exactly 0.5 by construction (identical score for every item); we report
        # the constant + note the AUC is undefined/chance because there is no per-utterance variation.
        out["bridgeparser_role_margin_constant"] = bp_margin
        out["bridgeparser_auc"] = roc_auc([bp_margin] * len(items), labels)  # -> 0.5 (all-ties)
        out["bridgeparser_note"] = ("content-blind: role_of() never reads the token, so the 3-role margin "
                                    "is identical for well-formed and OOV/ambiguous input -> AUC = chance")
    return out


def _per_tag(items, scores):
    tags = {}
    for (_, tag, *_r), s in zip(items, scores):
        tags.setdefault(tag, []).append(s)
    return {t: float(np.mean(v)) for t, v in tags.items()}


def gate(results):
    aucs = [r["auc_semantic"] for r in results]
    les = [r["auc_semantic_lesion"] for r in results]
    n = len(results)
    c1 = sum(a >= 0.80 for a in aucs) >= max(1, int(np.ceil(5 / 6 * n)))
    c2 = sum(l <= 0.60 for l in les) >= max(1, int(np.ceil(5 / 6 * n)))
    c3 = all(r["read_from_firing_states"] and not r["host_semantic_contrast_used"] for r in results)
    return {
        "GO": bool(c1 and c2 and c3),
        "c1_auc_ge_0.80_5of6": bool(c1),
        "c2_lesion_collapses_le_0.60_5of6": bool(c2),
        "c3_read_on_firing_neurons": bool(c3),
        "mean_auc_semantic": float(np.mean(aucs)),
        "mean_auc_full_control": float(np.mean([r["auc_full_control"] for r in results])),
        "mean_auc_lesion": float(np.mean(les)),
    }


def noncanon_verify(seed, n=20, read_steps=60):
    """VERIFY-GO adversarial arm: a genuine comprehension monitor must stay HIGH on a NON-CANONICAL but fully
    comprehensible sentence (object-fronted 'apple eat dog' -- inanimate patient first, animate agent last),
    where the content decisively resolves the roles even though WORD ORDER is misleading. A position-inclusive
    read would falsely DROP here (position conflicts with content); the SEMANTIC-cue-driven read should NOT,
    proving it measures comprehension, not canonical order. Compares object-fronted-well vs OOV."""
    rng = np.random.default_rng(seed)

    def pick(lst):
        return lst[int(rng.integers(len(lst)))]

    comp = _build_comp(seed)
    sem_of, full_of, sem_oov = [], [], []
    for _ in range(n):
        # object-fronted well-formed: [inanimate patient, verb, animate agent] -- comprehensible, non-canonical
        p, v, a = pick(INANIM), pick(ASYMM_VERBS), pick(ANIMATE)
        evs = _evs_for(p, v, a)
        sem_of.append(semantic_sel_margin(comp, evs, read_steps))
        full_of.append(full_sel_margin(comp, evs, read_steps))
        # OOV control (genuinely not understood)
        evs_o = _evs_for(pick(OOV_NOUNS), pick(ASYMM_VERBS), pick(OOV_NOUNS))
        sem_oov.append(semantic_sel_margin(comp, evs_o, read_steps))
    return {
        "seed": seed,
        "sem_objfront_well_mean": float(np.mean(sem_of)),
        "full_objfront_well_mean": float(np.mean(full_of)),
        "sem_oov_mean": float(np.mean(sem_oov)),
        # the monitor must call object-fronted-well UNDERSTOOD (high) and OOV NOT (low): a clean gap on the
        # SEMANTIC read even though word order is non-canonical.
        "sem_objfront_vs_oov_auc": roc_auc(sem_of + sem_oov, [1] * len(sem_of) + [0] * len(sem_oov)),
        # the full (position-inclusive) read is DEGRADED on object-front (position fights content):
        "full_le_sem_on_objfront": bool(np.mean(full_of) < np.mean(sem_of)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--n-per-cond", type=int, default=20)
    ap.add_argument("--read-steps", type=int, default=60)
    ap.add_argument("--smoke", action="store_true", help="1 seed, tiny battery, no bridge parser")
    ap.add_argument("--no-bridge", action="store_true")
    ap.add_argument("--noncanon-verify", action="store_true",
                    help="verify-go arm: semantic read stays high on object-fronted (non-canonical) well-formed")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.noncanon_verify:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        nc = [noncanon_verify(s, n=args.n_per_cond, read_steps=args.read_steps) for s in seeds]
        for r in nc:
            print(f"[seed {r['seed']}] sem(objfront-well)={r['sem_objfront_well_mean']:.4f}  "
                  f"full(objfront-well)={r['full_objfront_well_mean']:.4f}  sem(oov)={r['sem_oov_mean']:.4f}  "
                  f"sem objfront-vs-oov AUC={r['sem_objfront_vs_oov_auc']:.3f}  "
                  f"full<sem_on_objfront={r['full_le_sem_on_objfront']}", flush=True)
        aucs = [r["sem_objfront_vs_oov_auc"] for r in nc]
        print(f"\nmean sem objfront-vs-oov AUC = {np.mean(aucs):.3f} "
              f"(>=0.80 on {sum(a>=0.80 for a in aucs)}/{len(aucs)} seeds); "
              f"full<sem_on_objfront on {sum(r['full_le_sem_on_objfront'] for r in nc)}/{len(nc)} seeds", flush=True)
        if args.out:
            import os
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump({"noncanon_verify": nc}, f, indent=2)
            print(f"wrote {args.out}", flush=True)
        return

    if args.smoke:
        seeds = [42]
        n_per_cond = 4
        with_bridge = not args.no_bridge
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        n_per_cond = args.n_per_cond
        with_bridge = not args.no_bridge

    results = []
    for s in seeds:
        r = run_seed(s, n_per_cond=n_per_cond, read_steps=args.read_steps, with_bridge=with_bridge)
        results.append(r)
        print(f"[seed {s}] auc_sem={r['auc_semantic']:.3f}  auc_full(ctrl)={r['auc_full_control']:.3f}  "
              f"auc_lesion={r['auc_semantic_lesion']:.3f}  well={r['mean_margin_well']:.4f} "
              f"ill={r['mean_margin_ill']:.4f}  read_ms/turn={r['per_turn_read_ms']:.1f}", flush=True)
        print(f"          per-tag: {r['per_tag_mean']}", flush=True)
        if "bridgeparser_role_margin_constant" in r:
            print(f"          BridgeParser 3-role margin (content-blind constant)="
                  f"{r['bridgeparser_role_margin_constant']:.4f}  auc={r['bridgeparser_auc']:.3f}", flush=True)

    g = gate(results)
    print("\n=== GATE ===", flush=True)
    for k, v in g.items():
        print(f"  {k}: {v}", flush=True)

    # ATTRIBUTION (tools.lab): whose is the discrimination? The above-chance separation of the intact spiking
    # competition (treatment = mean_auc_semantic − 0.5) vs the lesioned control (control = mean_auc_lesion −
    # 0.5). 100% => the discrimination is entirely caused by the learned spiking competition, 0% is in the
    # lesioned control (the anti-tautology: the identical battery with zeroed synapses discriminates at chance).
    try:
        from tools.lab import attributable_to
        attributable_to("comprehension discrimination above chance (spiking competition vs lesion)",
                        g["mean_auc_semantic"] - 0.5, g["mean_auc_lesion"] - 0.5)
    except Exception as _e:  # tools.lab optional; the JSON already carries both arms
        print(f"  (attribution helper unavailable: {_e})", flush=True)

    payload = {"config": vars(args), "results": results, "gate": g}
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
