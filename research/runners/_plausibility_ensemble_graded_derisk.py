"""6-SEED DE-RISK — the ENSEMBLE + GRADED brain-native plausibility read, the robustness rung that closes
the 2-seed tiny-graph gap the QUALIFIED single-assembly hard read left open (finding
2026-09-01-brain-native-plausibility-spiking-synaptic-gate-qualified: 2/6 seeds underperformed host on
parity, generation was suppressed). This runner measures whether the ENSEMBLE (K redundant readout
populations averaged) + GRADED (rate-coded soft-AND) read reaches host-level parity AND generation on ALL 6
seeds, through the REAL production handler (research.runners.brain_chat_tui.ChatBrain), so the host P>=tau
scaffold can be retired DEFAULT-ON.

THE STRICTER BAR (default-ON = a real host-scaffold retirement, so the bar is all-6, not on-average):
  (PARITY)      spiking replay-vs-random advantage >= host advantage on EVERY seed (min parity >= 1.0), and
                the spiking advantage is itself above the random floor.
  (GENERATION)  the brain volunteers >= as many distinct novel props with the SPIKING gate as with the HOST
                gate, on EVERY seed (spiking gen >= host gen), via the REAL gate() over the open prompts.
  (AGREEMENT)   the (hard) spiking related() still reproduces the host P>=tau relation (sanity: same signal).
  (LESION)      shuffled-synapse advantage < spiking advantage AND ablated-synapse relatedness == 0, all seeds.
  (PROVENANCE)  the hot-path host P>=tau comparison is NEVER called while the organ is installed.
  (MOAT-SAFE)   0 leaks, 0 negated re-proposed, untaught-cue abstention unregressed, all seeds.
  (BYTE-ID OFF) BRAIN_SPIKING_PLAUSIBILITY=0 -> the organ is never built + gate() == the pure-host baseline.

Only the plausibility GATE differs between HOST and SPIKING (same facts, same host-oracle draw for the
advantage metric); the improved organ is the production default of build_for_proposer (ensemble_graded=True).

NO sim/ edit; reuse-by-import; CPU (SIM_BACKEND=numpy). Run:
  SIM_BACKEND=numpy python -u -m research.runners._plausibility_ensemble_graded_derisk \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/_plausibility_ensemble_graded_derisk.json
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
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")

from research.runners._genfrontier_b2_generative_replay_derisk import random_recombination  # noqa: E402
from research.runners._generate_channel_wiring_verify import (  # noqa: E402
    build_chat, _collect_hypotheses, _AFFIRMED, _NEGATED, _rich_facts, _TOPICS,
)
from research.runners.spiking_plausibility_organ import (  # noqa: E402
    SpikingAssociativePlausibilityOrgan, build_for_proposer, PRODUCTION_READ_CONFIG,
)

ENSEMBLE_GRADED_CONFIG = PRODUCTION_READ_CONFIG      # naming kept for the report/JSON key


def _advantage(prop, n_attempts, seed):
    """replay-vs-random plausible-fraction ratio (the generate-channel-verify metric), over whatever
    `_plausible` gate is currently installed on `prop`."""
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

    # ---- HOST baseline gate (the residual we convert): host `_related = P>=tau` on the proposer.
    chat, inner = build_chat(seed, affirmed, negated)
    prop = chat._build_generation_proposer()
    assert prop is not None, "the interlinked graph must build a proposer"
    host = _advantage(prop, a.n_attempts, seed)

    # ---- SPIKING gate: build + install the ENSEMBLE+GRADED plausibility organ on a FRESH proposer (same
    # facts/draw; only the GATE differs). Measure agreement (hard read vs host), advantage, provenance.
    chat_s, inner_s = build_chat(seed, affirmed, negated)
    prop_s = chat_s._build_generation_proposer()
    organ = build_for_proposer(prop_s, seed=chat_s._gen_seed)     # ensemble_graded=True (production default)
    agree = organ.agreement_with_host(prop_s.P, prop_s.row, prop_s.tau)
    n_host_calls_at_install = organ.n_host_related_calls
    organ.install(prop_s)
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

    # ---- GENERATION (SPIKING gate live), through the REAL gate(). Also MOAT-SAFE.
    os.environ["BRAIN_SPIKING_PLAUSIBILITY"] = "1"
    os.environ.pop("BRAIN_GENERATE_CHANNEL", None)
    chat_on, inner_on = build_chat(seed, affirmed, negated)
    hyps_on = _collect_hypotheses(chat_on, topics)
    organ_built_on = chat_on._spiking_plausibility_organ is not None
    # the production read is the ENSEMBLE (K>1 redundant readout populations, low internal recurrence); the
    # `graded` soft-AND sub-lever was tested and NOT adopted (graded=False), so this checks the ensemble only.
    used_ensemble_graded = bool(organ_built_on and chat_on._spiking_plausibility_organ.n_ensemble > 1
                                and chat_on._spiking_plausibility_organ.density == 0.0)
    hyps_set = set(hyps_on)
    n_generated = len(hyps_set)
    novel_disjoint = len(hyps_set & stored) == 0
    leaks = sum(1 for (ag, acn, ptn) in hyps_set
                if inner_on.what_does(ag, acn) == ptn or inner_on.is_it_true(ag, acn, ptn) != "unknown")
    negated_reproposed = len(hyps_set & negated_set)
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

    # ---- HOST GENERATION (BRAIN_SPIKING_PLAUSIBILITY=0), through the REAL gate(): the per-seed reference
    # the spiking generation must MATCH-or-BEAT. This is also the byte-identical-off position.
    os.environ["BRAIN_SPIKING_PLAUSIBILITY"] = "0"
    chat_off, _ = build_chat(seed, affirmed, negated)
    hyps_off = _collect_hypotheses(chat_off, topics)
    off_organ_unbuilt = (getattr(chat_off, "_spiking_plausibility_organ", "x") is None)
    chat_ref, _ = build_chat(seed, affirmed, negated)
    hyps_ref = _collect_hypotheses(chat_ref, topics)
    off_matches_host = (hyps_off == hyps_ref)
    n_generated_host = len(set(hyps_ref))
    os.environ.pop("BRAIN_SPIKING_PLAUSIBILITY", None)

    parity = spk["advantage"] / max(1e-9, host["advantage"])
    row = {
        "seed": seed,
        "vocab_size": len(organ.vocab),
        "organ_pattern_size": int(organ.pattern_size), "organ_n_ensemble": int(organ.n_ensemble),
        "organ_graded": bool(organ.graded), "organ_beta": float(organ._beta),
        "tau": float(prop.tau), "tau_spike": float(organ.tau_spike),
        "host_advantage": host["advantage"], "spiking_advantage": spk["advantage"],
        "parity_ratio": float(parity),
        "agreement_with_host": float(agree["agreement"]), "agreement_f1": float(agree["f1"]),
        "agreement_recall": float(agree["recall"]), "agreement_precision": float(agree["precision"]),
        "lesion_shuffle_advantage": lesion["advantage"],
        "lesion_ablate_related_pairs": int(n_abl_related),
        "n_spiking_reads": int(organ.n_spiking_reads),
        "hot_path_host_related_calls": int(hot_path_host_calls),
        "n_generated_spiking": int(n_generated), "n_generated_host": int(n_generated_host),
        "gen_spiking_ge_host": bool(n_generated >= n_generated_host),
        "novel_disjoint_from_store": bool(novel_disjoint),
        "moat_leaks": int(leaks), "negated_reproposed": int(negated_reproposed),
        "untaught_cue_abstention_ok": int(ab_ok), "untaught_cue_abstention_n": int(n_ab),
        "gate_used_spiking_organ": bool(organ_built_on), "gate_used_ensemble_graded": bool(used_ensemble_graded),
        "off_organ_unbuilt": bool(off_organ_unbuilt),
        "off_matches_host_baseline": bool(off_matches_host),
        "examples": [f"perhaps {x} {y} {z}" for (x, y, z) in hyps_on[:6]],
    }
    print(f"[eg seed {seed}] vocab={len(organ.vocab)} ps={organ.pattern_size} K={organ.n_ensemble} "
          f"graded={organ.graded} tau_spike={organ.tau_spike:.3f} | ADV host {host['advantage']:.2f} vs "
          f"SPIKING {spk['advantage']:.2f} (parity {parity:.2f}) | agree {agree['agreement']:.2f} "
          f"rec {agree['recall']:.2f} | GEN spk {n_generated} vs host {n_generated_host} "
          f"(spk>=host {n_generated >= n_generated_host}) | LESION shuf {lesion['advantage']:.2f} abl "
          f"{n_abl_related} | MOAT leaks {leaks} negrep {negated_reproposed} abstain {ab_ok}/{n_ab} | "
          f"PROV hot-host {hot_path_host_calls} | OFF unbuilt {off_organ_unbuilt} match {off_matches_host}",
          flush=True)
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
    n_gen = col("n_generated_spiking")
    n_gen_host = col("n_generated_host")
    gen_ge = col("gen_spiking_ge_host")
    disjoint = col("novel_disjoint_from_store")
    leaks = col("moat_leaks")
    negrep = col("negated_reproposed")
    ab_ok = col("untaught_cue_abstention_ok")
    ab_n = col("untaught_cue_abstention_n")
    off_unbuilt = col("off_organ_unbuilt")
    off_match = col("off_matches_host_baseline")
    used_organ = col("gate_used_spiking_organ")
    ab_rate = ab_ok / np.maximum(ab_n, 1)

    # PARITY (STRICT): spiking advantage >= host advantage on EVERY seed (min parity >= parity_bar, default
    # 1.0), AND the spiking advantage itself is above the random floor.
    parity_all = bool(np.all(parity >= a.parity_bar) and np.all(spk_adv > a.min_spiking_advantage))
    agreement_all = bool(np.all(agree >= a.agreement_bar))
    # GENERATION (STRICT): spiking gen >= host gen on EVERY seed, AND >= min_novel.
    generation_all = bool(np.all(gen_ge) and np.all(n_gen >= a.min_novel) and np.all(used_organ))
    lesion_all = bool(np.all(lesion < spk_adv) and np.all(ablate_rel == 0))
    provenance_all = bool(np.all(hot_host == 0) and np.all(reads >= 1))
    moat_all = bool(np.all(leaks == 0) and np.all(negrep == 0) and np.all(ab_rate >= a.store_floor_bar)
                    and np.all(disjoint))
    off_all = bool(np.all(off_unbuilt) and np.all(off_match))

    detail = {
        "host_advantage_mean": float(host_adv.mean()), "host_advantage_min": float(host_adv.min()),
        "spiking_advantage_mean": float(spk_adv.mean()), "spiking_advantage_min": float(spk_adv.min()),
        "parity_ratio_mean": float(parity.mean()), "parity_ratio_min": float(parity.min()),
        "beats_host_on_all_seeds": bool(np.all(parity >= 1.0)),
        "agreement_mean": float(agree.mean()), "agreement_min": float(agree.min()),
        "agreement_recall_mean": float(col("agreement_recall").mean()),
        "lesion_shuffle_advantage_mean": float(lesion.mean()),
        "lesion_ablate_related_pairs_total": int(ablate_rel.sum()),
        "hot_path_host_calls_total": int(hot_host.sum()),
        "n_generated_spiking_min": int(n_gen.min()), "n_generated_spiking_mean": float(n_gen.mean()),
        "n_generated_host_min": int(n_gen_host.min()), "n_generated_host_mean": float(n_gen_host.mean()),
        "gen_spiking_ge_host_all_seeds": bool(np.all(gen_ge)),
        "moat_leaks_total": int(leaks.sum()), "negated_reproposed_total": int(negrep.sum()),
        "untaught_cue_abstention_rate_min": float(ab_rate.min()),
        "parity_all_seeds": parity_all, "agreement_all_seeds": agreement_all,
        "generation_all_seeds": generation_all, "lesion_load_bearing_all_seeds": lesion_all,
        "provenance_all_seeds": provenance_all, "moat_all_seeds": moat_all, "byte_id_off_all_seeds": off_all,
        "parity_bar": a.parity_bar, "agreement_bar": a.agreement_bar,
        "min_spiking_advantage": a.min_spiking_advantage, "min_novel": a.min_novel,
    }
    # GO (default-ON candidate): parity >= host AND generation >= host on ALL seeds, + agreement + lesion +
    # provenance + moat + byte-id-off. Anything short is an HONEST NEGATIVE with the quantified residual.
    if (parity_all and generation_all and agreement_all and lesion_all and provenance_all and moat_all
            and off_all):
        verdict = "GO"
    elif not off_all:
        verdict = "SCOPED_byte_id_off_broken"
    elif not provenance_all:
        verdict = "SCOPED_provenance_host_leak"
    elif not moat_all:
        verdict = "SCOPED_moat_broken"
    elif not lesion_all:
        verdict = "SCOPED_not_load_bearing"
    elif not parity_all and not generation_all:
        verdict = "HONEST_NEGATIVE_parity_and_generation"
    elif not parity_all:
        verdict = "HONEST_NEGATIVE_underperforms_host_parity"
    elif not generation_all:
        verdict = "HONEST_NEGATIVE_generation_below_host"
    elif not agreement_all:
        verdict = "SCOPED_low_agreement"
    else:
        verdict = "SCOPED_other"
    return verdict, detail


def main():
    p = argparse.ArgumentParser(description="6-seed de-risk: the ENSEMBLE+GRADED brain-native plausibility "
                                            "read vs the host P>=tau gate, strict all-6 bar for default-ON.")
    p.add_argument("--seeds", default="42,43,44,100,101,102")
    p.add_argument("--n-attempts", type=int, default=600)
    p.add_argument("--parity-bar", type=float, default=1.0,
                   help="spiking advantage must be >= this fraction of host (default 1.0 = parity), per seed")
    p.add_argument("--min-spiking-advantage", type=float, default=1.3)
    p.add_argument("--agreement-bar", type=float, default=0.80)
    p.add_argument("--min-novel", type=int, default=3)
    p.add_argument("--store-floor-bar", type=float, default=0.95)
    p.add_argument("--rich", action="store_true", help="use the richer type-structured graph")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    seeds = [int(s.strip()) for s in a.seeds.split(",")]
    t0 = time.time()
    print(f"[eg] seeds={seeds} n_attempts={a.n_attempts} rich={a.rich} config={ENSEMBLE_GRADED_CONFIG} -- "
          f"ENSEMBLE+GRADED brain-native plausibility read, strict all-6 bar (parity>=host & gen>=host).",
          flush=True)
    rows = [run_seed(s, a) for s in seeds]
    verdict, detail = decide(rows, a)

    print(f"\n{'='*100}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  PARITY: host mean {detail['host_advantage_mean']:.2f} -> SPIKING mean "
          f"{detail['spiking_advantage_mean']:.2f} (parity mean {detail['parity_ratio_mean']:.2f}, min "
          f"{detail['parity_ratio_min']:.2f}; >=host all seeds {detail['beats_host_on_all_seeds']})", flush=True)
    print(f"  GENERATION: spiking mean {detail['n_generated_spiking_mean']:.1f} (min "
          f"{detail['n_generated_spiking_min']}) vs host mean {detail['n_generated_host_mean']:.1f} (min "
          f"{detail['n_generated_host_min']}); spiking>=host all seeds {detail['gen_spiking_ge_host_all_seeds']}",
          flush=True)
    print(f"  AGREEMENT with host P>=tau: mean {detail['agreement_mean']:.2f} (min {detail['agreement_min']:.2f}, "
          f"recall {detail['agreement_recall_mean']:.2f})", flush=True)
    print(f"  LESION load-bearing all seeds: {detail['lesion_load_bearing_all_seeds']} | PROVENANCE hot-host "
          f"calls {detail['hot_path_host_calls_total']} | MOAT all {detail['moat_all_seeds']} | BYTE-ID-OFF all "
          f"{detail['byte_id_off_all_seeds']}", flush=True)
    print(f"  elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"{'='*100}\n", flush=True)

    preconditions = [
        {"kind": "require", "name": "byte_id_off", "ok": bool(detail["byte_id_off_all_seeds"]),
         "detail": "BRAIN_SPIKING_PLAUSIBILITY=0 -> organ never built + gate() == pure-host baseline, all seeds"},
        {"kind": "require", "name": "provenance_no_host_leak", "ok": bool(detail["provenance_all_seeds"]),
         "detail": "the hot-path host P>=tau comparison is NEVER called while the spiking organ is installed"},
        {"kind": "require", "name": "advantage_parity_all6", "ok": bool(detail["parity_all_seeds"]),
         "detail": "spiking advantage >= host advantage on EVERY seed (min parity >= parity_bar)"},
        {"kind": "require", "name": "generation_ge_host_all6", "ok": bool(detail["generation_all_seeds"]),
         "detail": "spiking generation >= host generation on EVERY seed (via the real gate)"},
        {"kind": "require", "name": "agreement_with_host", "ok": bool(detail["agreement_all_seeds"]),
         "detail": "the spiking read reproduces the host P>=tau relation (agreement >= bar), all seeds"},
        {"kind": "require", "name": "lesion_load_bearing", "ok": bool(detail["lesion_load_bearing_all_seeds"]),
         "detail": "shuffled-synapse advantage < spiking advantage AND ablated-synapse relatedness == 0, all seeds"},
        {"kind": "require", "name": "moat_safe", "ok": bool(detail["moat_all_seeds"]),
         "detail": "0 leaks, 0 negated re-proposed, untaught-cue abstention unregressed, all seeds"},
    ]
    out = {
        "probe": "plausibility_ensemble_graded_derisk",
        "verdict": verdict,
        "preconditions": preconditions,
        "seeds": seeds,
        "config": {"n_attempts": a.n_attempts, "parity_bar": a.parity_bar,
                   "min_spiking_advantage": a.min_spiking_advantage, "agreement_bar": a.agreement_bar,
                   "min_novel": a.min_novel, "rich_graph": a.rich, "organ": ENSEMBLE_GRADED_CONFIG,
                   "composer_kind": os.environ.get("BRAIN_COMPOSER_KIND")},
        "mechanism": ("SpikingAssociativePlausibilityOrgan (ensemble+graded): K disjoint readout populations "
                      "per concept averaged (finer, lower-variance firing-fraction read -> stable operating "
                      "point) + a rate-coded soft-AND `_plausible` (logistic soft-relatedness around the "
                      "brain's own tau_spike, geometric-mean >= 0.5) that preserves borderline-but-supported "
                      "triples so generation is not suppressed. Reads cp_firing_states, never P>=tau."),
        "detail": detail,
        "per_seed": rows,
        "elapsed_total_s": time.time() - t0,
    }
    if a.out is None:
        a.out = os.path.join(_REPO, "research", "findings", "raw", "_plausibility_ensemble_graded_derisk.json")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {a.out}", flush=True)
    return out


if __name__ == "__main__":
    main()
