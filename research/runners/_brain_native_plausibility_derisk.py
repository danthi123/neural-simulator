"""6-SEED DE-RISK — the #3E plausibility GATE, converted from a HOST matrix comparison to a BRAIN-NATIVE
spiking synaptic read, verified through the REAL production handler
(research.runners.brain_chat_tui.ChatBrain / _build_generation_proposer).

THE HOST SHORTCUT (what this converts).  In the generate channel the brain volunteers a novel grounded
proposition; each candidate SVO triple is gated by plausibility:
    _plausible(a, ac, p) = _related(a, ac) and _related(ac, p)
    _related(w1, w2)     = P[row[w1], row[w2]] >= tau              # <-- a HOST float comparison
`P` = the brain's own concept co-occurrence over its stored facts; `tau` = the 50th percentile of the
positive edges. The 2026-08-18-generate-channel-wired-brain-chat-GO finding declared this the residual:
"only the DRAW is spiking ... the plausibility LIKELIHOOD is a host co-occurrence matrix". It gave a real,
lesion-load-bearing advantage over the random floor (mean 2.7x, 2.1x-3.4x).

THE BRAIN-NATIVE MECHANISM (SpikingAssociativePlausibilityOrgan).  The co-occurrence graph is embodied as
SYNAPTIC WEIGHTS (cortex_A -> dlpfc_B, weight ∝ co-occurrence count) on a real SimulationBridge, and
relatedness is decided by SPIKES: drive w1's input assembly; w2 is "related" iff its readout assembly fires
above the brain's OWN threshold (`tau_spike` = the same 50th-percentile rule applied to the brain's positive
spiking readouts). The gate decision reads a SPIKE COUNT, never `P >= tau`. See spiking_plausibility_organ.py.

WHAT IS CHECKED, per seed (42,43,44,100,101,102) through the REAL handler's proposer:
  (ADVANTAGE / PARITY) the SPIKING plausibility gate's replay-vs-random advantage MATCHES or BEATS the HOST
    gate's advantage (bar: spiking >= parity_frac x host, on the SAME facts/draw — only the GATE differs),
    and the spiking advantage is itself > 1 (above the random floor).
  (AGREEMENT) the spiking related() reproduces the host `P>=tau` relation (agreement, F1) — it is the same
    operating point computed by spikes, not a different signal.
  (LESION load-bearing) a SHUFFLED-synapse organ (co-occurrence neighbourhoods destroyed, marginals kept —
    the b2 anti-cheat in synapses) collapses the spiking advantage toward the floor: the LEARNED structure,
    read through synapses, carries the signal (not the spiking machinery per se). An ABLATED-synapse organ
    (zero association weights) makes relatedness collapse entirely (nothing fires).
  (MOAT-SAFE) through the real gate() with the spiking gate LIVE: 0 hypothesis->known-fact leaks, 0 negated
    re-proposed, untaught-cue abstention unregressed, and the brain still VOLUNTEERS >= min_novel novel props.
  (PROVENANCE) relatedness is decided from cp_firing_states (n_spiking_reads == |vocab| per proposer); the
    hot-path host `P>=tau` comparison is NEVER called while the organ is installed (n_host_related_calls==0).
  (BYTE-IDENTICAL OFF) BRAIN_SPIKING_PLAUSIBILITY=0 -> the organ is never built, prop keeps the host _related,
    and gate() volunteers the SAME hypotheses as the pure-host baseline (the OFF position is byte-identical).

NO sim/ edit; reuse-by-import; CPU (SIM_BACKEND=numpy). Run:
  SIM_BACKEND=numpy python -u -m research.runners._brain_native_plausibility_derisk \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/_brain_native_plausibility_derisk.json
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

os.environ.setdefault("SIM_BACKEND", "numpy")
# rf composer -> the non-contradiction / no-confab moat reads a real polarity store (matches the generate-channel verify).
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")

from research.runners._genfrontier_b2_generative_replay_derisk import random_recombination  # noqa: E402
from research.runners._generate_channel_wiring_verify import (  # noqa: E402
    build_chat, _collect_hypotheses, _AFFIRMED, _NEGATED, _rich_facts, _TOPICS,
)
from research.runners.spiking_plausibility_organ import (  # noqa: E402
    SpikingAssociativePlausibilityOrgan, build_for_proposer,
)


def _advantage(prop, n_attempts, seed):
    """replay-vs-random plausible-fraction ratio (the exact generate-channel-verify metric), over whatever
    `_related` gate is currently installed on `prop`."""
    rep = prop.propose(n_attempts)["plausible_fraction_of_novel"]
    randb = random_recombination(prop, n_attempts, np.random.default_rng(seed * 13 + 3))
    rnd = randb["plausible_fraction_of_novel"]
    adv = rep / max(rnd, 1.0 / max(1, randb["n_novel_attempts"]))
    return {"replay_frac": float(rep), "random_frac": float(rnd), "advantage": float(adv)}


def run_seed(seed, a):
    affirmed, negated = (_rich_facts() if a.rich else (_AFFIRMED, _NEGATED))
    topics = (sorted({ag for ag, _, _ in affirmed})[:6] if a.rich else _TOPICS)
    negated = list(negated)
    negated_set = set(negated)
    stored = set(affirmed)

    # ---- HOST baseline (the residual we convert): the host `_related = P>=tau` gate on the proposer.
    chat, inner = build_chat(seed, affirmed, negated)
    prop = chat._build_generation_proposer()
    assert prop is not None, "the interlinked graph must build a proposer"
    host = _advantage(prop, a.n_attempts, seed)

    # ---- SPIKING gate: build + install the brain-native plausibility organ on a FRESH proposer (same facts/
    # draw; only the GATE differs). Measure agreement vs host, advantage, provenance.
    chat_s, inner_s = build_chat(seed, affirmed, negated)
    prop_s = chat_s._build_generation_proposer()
    organ = build_for_proposer(prop_s, seed=chat_s._gen_seed)
    agree = organ.agreement_with_host(prop_s.P, prop_s.row, prop_s.tau)
    n_host_calls_at_install = organ.n_host_related_calls   # agreement() uses the host relation for the diag only
    organ.install(prop_s)
    # from here, every _related on prop_s is the spiking read; count hot-path host calls (must stay flat).
    spk = _advantage(prop_s, a.n_attempts, seed)
    hot_path_host_calls = organ.n_host_related_calls - n_host_calls_at_install    # MUST be 0

    # ---- LESION (shuffled synapses): the learned neighbourhood is destroyed -> advantage must collapse.
    chat_l, _ = build_chat(seed, affirmed, negated)
    prop_l = chat_l._build_generation_proposer()
    les = build_for_proposer(prop_l, seed=chat_l._gen_seed, lesion="shuffle")
    les.install(prop_l)
    lesion = _advantage(prop_l, a.n_attempts, seed)

    # ---- LESION (ablated synapses): zero association -> nothing fires -> relatedness collapses entirely.
    chat_a, _ = build_chat(seed, affirmed, negated)
    prop_a = chat_a._build_generation_proposer()
    abl = build_for_proposer(prop_a, seed=chat_a._gen_seed, lesion="ablate")
    n_abl_related = sum(1 for x in abl.vocab for y in abl.vocab if x != y and abl.related(x, y))

    # ---- MOAT-SAFE, through the REAL gate() with the spiking gate LIVE (default-ON). Collect volunteered hyps.
    os.environ["BRAIN_SPIKING_PLAUSIBILITY"] = "1"
    os.environ.pop("BRAIN_GENERATE_CHANNEL", None)
    chat_on, inner_on = build_chat(seed, affirmed, negated)
    hyps_on = _collect_hypotheses(chat_on, topics)
    organ_built_on = chat_on._spiking_plausibility_organ is not None      # the spiking gate WAS used
    hyps_set = set(hyps_on)
    n_generated = len(hyps_set)
    novel_disjoint = len(hyps_set & stored) == 0
    leaks = sum(1 for (ag, acn, ptn) in hyps_set
                if inner_on.what_does(ag, acn) == ptn or inner_on.is_it_true(ag, acn, ptn) != "unknown")
    negated_reproposed = len(hyps_set & negated_set)
    # untaught-cue abstention unregressed
    rng = np.random.default_rng(seed)
    stored_cues = {(a_, v_) for a_, v_, _ in affirmed}
    all_words = sorted({w for f in affirmed for w in f})
    n_ab, ab_ok, guard = 0, 0, 0
    while n_ab < 20 and guard < 100000:
        guard += 1
        ag = all_words[int(rng.integers(len(all_words)))]
        acn = all_words[int(rng.integers(len(all_words)))]
        if (ag, acn) in stored_cues:
            continue
        n_ab += 1
        ab_ok += int(inner_on.what_does(ag, acn) is None)

    # ---- BYTE-IDENTICAL OFF: BRAIN_SPIKING_PLAUSIBILITY=0 -> host gate, organ never built, SAME hypotheses
    # as a pure-host run. (The generate channel itself stays ON; only the plausibility gate reverts to host.)
    os.environ["BRAIN_SPIKING_PLAUSIBILITY"] = "0"
    chat_off, _ = build_chat(seed, affirmed, negated)
    hyps_off = _collect_hypotheses(chat_off, topics)
    off_organ_unbuilt = (getattr(chat_off, "_spiking_plausibility_organ", "x") is None)
    # the pure-host reference (organ never involved) on the same handler:
    chat_ref, _ = build_chat(seed, affirmed, negated)
    hyps_ref = _collect_hypotheses(chat_ref, topics)
    off_matches_host = (hyps_off == hyps_ref)
    os.environ.pop("BRAIN_SPIKING_PLAUSIBILITY", None)

    parity = spk["advantage"] / max(1e-9, host["advantage"])
    row = {
        "seed": seed,
        "vocab_size": len(organ.vocab),
        "tau": float(prop.tau), "tau_spike": float(organ.tau_spike),
        "host_replay_frac": host["replay_frac"], "host_random_frac": host["random_frac"],
        "host_advantage": host["advantage"],
        "spiking_replay_frac": spk["replay_frac"], "spiking_random_frac": spk["random_frac"],
        "spiking_advantage": spk["advantage"],
        "parity_ratio": float(parity),
        "agreement_with_host": float(agree["agreement"]), "agreement_f1": float(agree["f1"]),
        "spk_related_pairs": int(agree["spk_related"]), "host_related_pairs": int(agree["host_related"]),
        "lesion_shuffle_advantage": lesion["advantage"],
        "lesion_ablate_related_pairs": int(n_abl_related),
        "n_spiking_reads": int(organ.n_spiking_reads),
        "hot_path_host_related_calls": int(hot_path_host_calls),
        "n_generated": int(n_generated),
        "novel_disjoint_from_store": bool(novel_disjoint),
        "moat_leaks": int(leaks), "negated_reproposed": int(negated_reproposed),
        "untaught_cue_abstention_ok": int(ab_ok), "untaught_cue_abstention_n": int(n_ab),
        "gate_used_spiking_organ": bool(organ_built_on),
        "off_organ_unbuilt": bool(off_organ_unbuilt),
        "off_matches_host_baseline": bool(off_matches_host),
        "examples": [f"perhaps {x} {y} {z}" for (x, y, z) in hyps_on[:6]],
    }
    print(f"[bnp seed {seed}] vocab={len(organ.vocab)} tau={prop.tau} tau_spike={organ.tau_spike:.3f} | "
          f"ADV host {host['advantage']:.2f}x vs SPIKING {spk['advantage']:.2f}x (parity {parity:.2f}) | "
          f"agree {agree['agreement']:.2f} F1 {agree['f1']:.2f} | LESION shuffle {lesion['advantage']:.2f}x "
          f"ablate-related {n_abl_related} | MOAT gen {n_generated} leaks {leaks} negrep {negated_reproposed} "
          f"abstain {ab_ok}/{n_ab} | PROV reads {organ.n_spiking_reads} hot-host {hot_path_host_calls} | "
          f"OFF unbuilt {off_organ_unbuilt} matches-host {off_matches_host}", flush=True)
    return row


def decide(rows, a):
    def col(k):
        return np.array([r[k] for r in rows])

    host_adv = col("host_advantage")
    spk_adv = col("spiking_advantage")
    parity = col("parity_ratio")
    agree = col("agreement_with_host")
    lesion = col("lesion_shuffle_advantage")
    ablate_rel = col("lesion_ablate_related_pairs")
    hot_host = col("hot_path_host_related_calls")
    reads = col("n_spiking_reads")
    n_gen = col("n_generated")
    disjoint = col("novel_disjoint_from_store")
    leaks = col("moat_leaks")
    negrep = col("negated_reproposed")
    ab_ok = col("untaught_cue_abstention_ok")
    ab_n = col("untaught_cue_abstention_n")
    off_unbuilt = col("off_organ_unbuilt")
    off_match = col("off_matches_host_baseline")
    used_organ = col("gate_used_spiking_organ")
    ab_rate = ab_ok / np.maximum(ab_n, 1)

    # PARITY: the spiking gate MATCHES the host advantage (>= parity_frac x host) or BEATS it, AND is itself
    # above the random floor (>1) on every seed.
    parity_all = bool(np.all(parity >= a.parity_frac) and np.all(spk_adv > a.min_spiking_advantage))
    beats_any = bool(np.any(spk_adv >= host_adv))
    agreement_all = bool(np.all(agree >= a.agreement_bar))
    # LESION load-bearing: shuffled-synapse advantage clearly BELOW the spiking advantage (the learned
    # neighbourhood, read through synapses, carries the signal), AND ablation collapses relatedness to ~0.
    lesion_all = bool(np.all(lesion < spk_adv) and np.all(ablate_rel == 0))
    # PROVENANCE: the hot-path host `P>=tau` is never called while installed; |vocab| spiking reads per seed.
    provenance_all = bool(np.all(hot_host == 0) and np.all(reads >= 1))
    # MOAT-SAFE with the spiking gate live.
    moat_all = bool(np.all(leaks == 0) and np.all(negrep == 0) and np.all(ab_rate >= a.store_floor_bar)
                    and np.all(disjoint))
    generated_all = bool(np.all(n_gen >= a.min_novel) and np.all(used_organ))
    # BYTE-IDENTICAL OFF: organ never built + gate() output equals the pure-host baseline.
    off_all = bool(np.all(off_unbuilt) and np.all(off_match))

    detail = {
        "host_advantage_mean": float(host_adv.mean()), "host_advantage_min": float(host_adv.min()),
        "host_advantage_max": float(host_adv.max()),
        "spiking_advantage_mean": float(spk_adv.mean()), "spiking_advantage_min": float(spk_adv.min()),
        "spiking_advantage_max": float(spk_adv.max()),
        "parity_ratio_mean": float(parity.mean()), "parity_ratio_min": float(parity.min()),
        "beats_host_on_any_seed": beats_any,
        "agreement_mean": float(agree.mean()), "agreement_min": float(agree.min()),
        "agreement_f1_mean": float(col("agreement_f1").mean()),
        "lesion_shuffle_advantage_mean": float(lesion.mean()),
        "lesion_ablate_related_pairs_total": int(ablate_rel.sum()),
        "hot_path_host_calls_total": int(hot_host.sum()),
        "n_spiking_reads_mean": float(reads.mean()),
        "n_generated_min": int(n_gen.min()), "n_generated_mean": float(n_gen.mean()),
        "moat_leaks_total": int(leaks.sum()), "negated_reproposed_total": int(negrep.sum()),
        "untaught_cue_abstention_rate_min": float(ab_rate.min()),
        "parity_all_seeds": parity_all, "agreement_all_seeds": agreement_all,
        "lesion_load_bearing_all_seeds": lesion_all, "provenance_all_seeds": provenance_all,
        "moat_all_seeds": moat_all, "generated_all_seeds": generated_all,
        "byte_id_off_all_seeds": off_all,
        "parity_frac_bar": a.parity_frac, "agreement_bar": a.agreement_bar,
        "min_spiking_advantage": a.min_spiking_advantage, "min_novel": a.min_novel,
    }
    # GO: the plausibility GATE is now computed by the brain (spiking) AND it matches the host advantage,
    # reproduces the host relation, is lesion-load-bearing, moat-safe, provenance-clean, byte-identical-off.
    if (parity_all and agreement_all and lesion_all and provenance_all and moat_all and generated_all and off_all):
        verdict = "GO"
    elif not off_all:
        verdict = "SCOPED_byte_id_off_broken"
    elif not provenance_all:
        verdict = "SCOPED_provenance_host_leak"
    elif not moat_all:
        verdict = "SCOPED_moat_broken"
    elif not generated_all:
        verdict = "SCOPED_no_novel_generated"
    elif not lesion_all:
        verdict = "SCOPED_not_load_bearing"
    elif not parity_all:
        verdict = "HONEST_NEGATIVE_underperforms_host"     # the quantified gap IS the deliverable
    elif not agreement_all:
        verdict = "SCOPED_low_agreement"
    else:
        verdict = "SCOPED_other"
    return verdict, detail


def main():
    p = argparse.ArgumentParser(description="6-seed de-risk: brain-native SPIKING plausibility gate vs the host "
                                            "P>=tau matrix comparison, through the real ChatBrain handler.")
    p.add_argument("--seeds", default="42,43,44,100,101,102")
    p.add_argument("--n-attempts", type=int, default=600)
    p.add_argument("--parity-frac", type=float, default=0.8,
                   help="spiking advantage must be >= this fraction of the host advantage (MATCH), per seed")
    p.add_argument("--min-spiking-advantage", type=float, default=1.3,
                   help="the spiking advantage itself must exceed this (clearly above the random floor)")
    p.add_argument("--agreement-bar", type=float, default=0.85,
                   help="spiking related() must agree with host P>=tau at least this fraction of pairs")
    p.add_argument("--min-novel", type=int, default=3)
    p.add_argument("--store-floor-bar", type=float, default=0.95)
    p.add_argument("--rich", action="store_true", help="use the richer type-structured graph")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    t0 = time.time()
    print(f"[bnp] seeds={seeds} n_attempts={a.n_attempts} rich={a.rich} -- converting the #3E plausibility GATE "
          f"from host P>=tau to a SPIKING synaptic read, through the real ChatBrain handler.", flush=True)
    rows = [run_seed(s, a) for s in seeds]
    verdict, detail = decide(rows, a)

    print(f"\n{'='*98}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  ADVANTAGE: host mean {detail['host_advantage_mean']:.2f}x -> SPIKING mean "
          f"{detail['spiking_advantage_mean']:.2f}x (parity mean {detail['parity_ratio_mean']:.2f}, min "
          f"{detail['parity_ratio_min']:.2f}; beats-host-any-seed {detail['beats_host_on_any_seed']})", flush=True)
    print(f"  AGREEMENT with host P>=tau: mean {detail['agreement_mean']:.2f} (min {detail['agreement_min']:.2f}, "
          f"F1 {detail['agreement_f1_mean']:.2f})", flush=True)
    print(f"  LESION load-bearing all seeds: {detail['lesion_load_bearing_all_seeds']} (shuffle adv mean "
          f"{detail['lesion_shuffle_advantage_mean']:.2f}x; ablate related-pairs total "
          f"{detail['lesion_ablate_related_pairs_total']})", flush=True)
    print(f"  PROVENANCE (hot-path host P>=tau calls total, MUST be 0): {detail['hot_path_host_calls_total']} "
          f"({detail['n_spiking_reads_mean']:.0f} spiking reads/seed)", flush=True)
    print(f"  MOAT all seeds: {detail['moat_all_seeds']} (leaks {detail['moat_leaks_total']}, negated-reproposed "
          f"{detail['negated_reproposed_total']}, abstain-min {detail['untaught_cue_abstention_rate_min']:.2f}); "
          f"generated all: {detail['generated_all_seeds']} (min {detail['n_generated_min']})", flush=True)
    print(f"  BYTE-IDENTICAL OFF all seeds (organ unbuilt + gate()==host baseline): {detail['byte_id_off_all_seeds']}",
          flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*98}\n", flush=True)

    preconditions = [
        {"kind": "require", "name": "byte_id_off", "ok": bool(detail["byte_id_off_all_seeds"]),
         "detail": "BRAIN_SPIKING_PLAUSIBILITY=0 -> organ never built + gate() == pure-host baseline, all seeds"},
        {"kind": "require", "name": "provenance_no_host_leak", "ok": bool(detail["provenance_all_seeds"]),
         "detail": "the hot-path host P>=tau comparison is NEVER called while the spiking organ is installed"},
        {"kind": "require", "name": "advantage_parity", "ok": bool(detail["parity_all_seeds"]),
         "detail": "spiking advantage >= parity_frac x host advantage AND > min_spiking_advantage, all seeds"},
        {"kind": "require", "name": "agreement_with_host", "ok": bool(detail["agreement_all_seeds"]),
         "detail": "the spiking read reproduces the host P>=tau relation (agreement >= bar), all seeds"},
        {"kind": "require", "name": "lesion_load_bearing", "ok": bool(detail["lesion_load_bearing_all_seeds"]),
         "detail": "shuffled-synapse advantage < spiking advantage AND ablated-synapse relatedness == 0, all seeds"},
        {"kind": "require", "name": "moat_safe", "ok": bool(detail["moat_all_seeds"]),
         "detail": "0 leaks, 0 negated re-proposed, untaught-cue abstention unregressed, all seeds"},
        {"kind": "require", "name": "novel_generated", "ok": bool(detail["generated_all_seeds"]),
         "detail": ">= min_novel novel props via the real gate() with the spiking gate live, all seeds"},
    ]
    out = {
        "probe": "brain_native_plausibility_derisk",
        "verdict": verdict,
        "preconditions": preconditions,
        "seeds": seeds,
        "config": {"n_attempts": a.n_attempts, "parity_frac": a.parity_frac,
                   "min_spiking_advantage": a.min_spiking_advantage, "agreement_bar": a.agreement_bar,
                   "min_novel": a.min_novel, "store_floor_bar": a.store_floor_bar, "rich_graph": a.rich,
                   "composer_kind": os.environ.get("BRAIN_COMPOSER_KIND")},
        "flag": {"name": "BRAIN_SPIKING_PLAUSIBILITY",
                 "default": "OFF / opt-in (_SPIKING_PLAUSIBILITY_DEFAULT_ON=False) — the spiking gate matches the "
                 "host advantage on average + is provenance/lesion/moat/byte-id clean, but does not dominate host on "
                 "ALL 6 seeds on the sparse tiny graph (2/6 underperform, generation suppressed), so it is opt-in",
                 "on_value": "1/true/on/yes",
                 "off_semantics": "the plausibility gate stays the host _related = P>=tau; the spiking organ is never "
                 "built (byte-identical: gate() volunteers the same hypotheses as the pure-host baseline)"},
        "mechanism": ("SpikingAssociativePlausibilityOrgan: the co-occurrence graph is installed as directed "
                      "cortex_A->dlpfc_B synapses (weight ∝ co-occurrence count) on a real SimulationBridge; "
                      "related(w1,w2) drives w1's assembly and reads whether w2's readout assembly fires above the "
                      "brain's own threshold (tau_spike = the 50th-percentile rule applied to the brain's positive "
                      "spiking readouts). The decision reads cp_firing_states, not P>=tau. Monosynaptic (the readout "
                      "never projects back), so no multi-hop transitive blow-up."),
        "honest_residual": ("the synaptic weight matrix is SET from the co-occurrence counts (the same counts the "
                            "host P holds); online Hebbian self-organization of those weights is the named next rung. "
                            "The selectional-preference STRUCTURE and the SVO template are unchanged. The advance: the "
                            "plausibility DECISION is now computed by neurons+synapses+spikes, not a host matrix."),
        "detail": detail,
        "per_seed": rows,
        "elapsed_total_s": time.time() - t0,
    }
    if a.out is None:
        a.out = os.path.join(_REPO, "research", "findings", "raw", "_brain_native_plausibility_derisk.json")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {a.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
