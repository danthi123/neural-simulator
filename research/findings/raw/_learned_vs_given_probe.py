"""Learned-vs-given audit: how much of concept separability is LEARNED vs GIVEN by the input encoding?

This is the rigorous answer to "are we cheating?". The 2026-05-31 drive-echo finding showed that a
bind-QA on captured pool codes is contaminated by the orthogonal INPUT encoding (distinct inputs ->
distinct codes even untrained). So that metric cannot prove learned concepts. The HONEST metric is the
POOL-LABEL recognition (argmax over concept pools of the pool firing rate): does the trained lang_input ->
pool routing send each word to its OWN pool? Drive-echo CANNOT pass this on an untrained bridge, because
random weights route every word to a random/dominant pool (~chance). So:

    LEARNED separability  =  pool-label(trained)  -  pool-label(untrained)

The untrained value is the drive-echo / structural-bias floor; the delta over it is genuinely learned.

Run (GPU), trained then untrained control:
  python -m research.findings.raw._learned_vs_given_probe --ckpt <16word.h5>
  python -m research.findings.raw._learned_vs_given_probe --ckpt <16word.h5> --untrained

Reuse-by-import only (concept_pool_demo builder + orthogonal_drive_pattern); no protected-module change.
"""
from __future__ import annotations
import argparse
import os
import numpy as np

import research.runners.concept_pool_demo as cpd
from sim.backend import get_backend, to_host


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="research/findings/raw/_learned16_seed42.simstate.h5")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-lang", type=int, default=2048)
    ap.add_argument("--sparsity", type=float, default=0.05)
    ap.add_argument("--drive-pa", type=float, default=200.0)
    ap.add_argument("--reset", type=int, default=50)
    ap.add_argument("--stim", type=int, default=100)
    ap.add_argument("--untrained", action="store_true",
                    help="CONTROL: random weights, do NOT load the checkpoint (isolates the drive-echo floor)")
    a = ap.parse_args()
    from sim.text_embeddings import orthogonal_drive_pattern
    xp, backend = get_backend()

    if not a.untrained and not os.path.exists(a.ckpt):
        print(f"CANNOT-CONCLUDE: {a.ckpt} not found (run the 16-word training first)", flush=True); return

    # 16-word vocab + pools, exactly as concept_pool_demo builds them (motor + noun + verb + adjective)
    words = (list(cpd.DIRECTION_VOCAB) + list(cpd.NOUN_VOCAB)
             + list(cpd.VERB_VOCAB) + list(cpd.ADJECTIVE_VOCAB))
    word_to_idx = {w: i for i, w in enumerate(words)}
    word_to_pool = {}
    for w, v in cpd.DIRECTION_VOCAB.items(): word_to_pool[w] = f"motor_{v}"
    for w, v in cpd.NOUN_VOCAB.items(): word_to_pool[w] = f"noun_pool_{v}"
    for w, v in cpd.VERB_VOCAB.items(): word_to_pool[w] = f"verb_pool_{v}"
    for w, v in cpd.ADJECTIVE_VOCAB.items(): word_to_pool[w] = f"adjective_pool_{v}"
    pools = ([f"motor_{v}" for v in cpd.DIRECTION_VOCAB.values()]
             + [f"noun_pool_{v}" for v in cpd.NOUN_VOCAB.values()]
             + [f"verb_pool_{v}" for v in cpd.VERB_VOCAB.values()]
             + [f"adjective_pool_{v}" for v in cpd.ADJECTIVE_VOCAB.values()])
    tag = "UNTRAINED-CONTROL" if a.untrained else "TRAINED"
    print(f"=== learned-vs-given ({len(words)} words, {len(pools)} pools, backend={backend}, {tag}) ===",
          flush=True)

    bridge = cpd.build_concept_bridge(seed=a.seed, n_lang_input=a.n_lang, n_per_pool=200, n_fs_per_pool=24,
                                      enable_adjective=True, weak_dynamics=True, verbose=False)
    if a.untrained:
        print("  [random weights; checkpoint NOT loaded -- this is the drive-echo / structural floor]",
              flush=True)
    else:
        bridge.load_checkpoint(a.ckpt)
    rm = bridge.region_manager
    pool_slices = {}; all_idx = []
    for p in pools:
        idx = list(rm.indices(p)); pool_slices[p] = (len(all_idx), len(all_idx) + len(idx)); all_idx += idx
    all_arr = xp.asarray(all_idx, dtype=xp.int64)
    lang_arr = xp.asarray(list(rm.indices("language_input")), dtype=xp.int64)

    def capture(word):
        drive = orthogonal_drive_pattern(cue_idx=word_to_idx[word], n_cues=len(words),
                                         n_neurons=a.n_lang, drive_max_pA=a.drive_pa, sparsity=a.sparsity)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(a.reset):
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[lang_arr] = xp.asarray(drive, dtype=xp.float32)
        acc = xp.zeros(len(all_idx), dtype=xp.float64)
        for _ in range(a.stim):
            bridge._run_one_simulation_step()
            acc += bridge.cp_firing_states[all_arr].astype(xp.float64)
        bridge.cp_external_input_current[:] = 0.0
        return to_host(acc) / a.stim

    label_ok = 0
    per_word = []
    for w in words:
        a_act = capture(w)
        rates = {p: a_act[s:e].mean() for p, (s, e) in pool_slices.items()}
        pred = max(rates, key=rates.get)
        ok = int(pred == word_to_pool[w]); label_ok += ok
        per_word.append((w, word_to_pool[w], pred, ok))
    label_acc = label_ok / len(words)
    print(f"  POOL-LABEL recognition: {label_acc:.3f} ({label_ok}/{len(words)})  chance={1.0/len(pools):.3f}",
          flush=True)
    for w, gt, pred, ok in per_word:
        print(f"    {'OK ' if ok else 'xx '} {w:8s} -> {pred:20s} (gt {gt})", flush=True)
    print(f"\nRESULT[{tag}]: pool-label={label_acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
