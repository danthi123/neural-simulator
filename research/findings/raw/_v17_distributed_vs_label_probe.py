"""Front-end angle (insight #5): at 28 words where the POOL-LABEL recognition is ~50% (v17 wall,
motor pools dominate the argmax), does the bind/QA on the DISTRIBUTED concept-pool activity EXCEED
the pool-label accuracy? If YES -> the front-end limit is partly a READOUT artifact and the effective
conversational vocabulary is larger than the label suggests. If NO (distributed also ~50%) -> the codes
are genuinely inseparable at 28 words and the limit is real. Either way a real finding.

The v17 finding measured ONLY the pool-label (50%); this measures the distributed-code bind-recovery on
the SAME bridge -- genuinely new. Loads the 28-word bridge trained by concept_pool_demo_v2 (architecture
matched; load_checkpoint validates so the 2026-05-14 monkey-patch mismatch bug is CAUGHT not silent).

Run AFTER the 28-word training: python -m research.findings.raw._v17_distributed_vs_label_probe
"""
from __future__ import annotations
import os
import numpy as np

import research.runners.concept_pool_demo_v2 as v2          # MUST import first: patches vocab to 28 words
import research.runners.concept_pool_demo as cpd            # vocab now patched
import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
from sim.backend import get_backend, to_host

CKPT = "research/findings/raw/_v17_28word_seed42.simstate.h5"
N_LANG = 2048
SPARSITY = 0.03
DRIVE_PA = 200.0
RESET, STIM = 50, 100


def _center(v):
    v = v.astype(np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    if not os.path.exists(CKPT):
        print(f"CANNOT-CONCLUDE: {CKPT} not found (run the 28-word training first)"); return
    from sim.text_embeddings import orthogonal_drive_pattern
    xp, backend = get_backend()
    print(f"=== v17 distributed-vs-label (28 words, backend={backend}) ===", flush=True)

    # word ordering + word->pool, exactly as concept_pool_demo builds them (patched 28-word vocab)
    words = list(cpd.DIRECTION_VOCAB) + list(cpd.NOUN_VOCAB) + list(cpd.VERB_VOCAB) + list(cpd.ADJECTIVE_VOCAB)
    word_to_idx = {w: i for i, w in enumerate(words)}
    word_to_pool = {}
    for w, v in cpd.DIRECTION_VOCAB.items(): word_to_pool[w] = f"motor_{v}"
    for w, v in cpd.NOUN_VOCAB.items(): word_to_pool[w] = f"noun_pool_{v}"
    for w, v in cpd.VERB_VOCAB.items(): word_to_pool[w] = f"verb_pool_{v}"
    for w, v in cpd.ADJECTIVE_VOCAB.items(): word_to_pool[w] = f"adjective_pool_{v}"
    pools = [f"motor_{v}" for v in cpd.DIRECTION_VOCAB.values()] + \
            [f"noun_pool_{v}" for v in cpd.NOUN_VOCAB.values()] + \
            [f"verb_pool_{v}" for v in cpd.VERB_VOCAB.values()] + \
            [f"adjective_pool_{v}" for v in cpd.ADJECTIVE_VOCAB.values()]
    print(f"  {len(words)} words, {len(pools)} pools", flush=True)

    # build matching architecture + load (load_checkpoint validates -> mismatch is caught)
    bridge = cpd.build_concept_bridge(seed=42, n_lang_input=N_LANG, n_per_pool=200, n_fs_per_pool=24,
                                      enable_adjective=True, weak_dynamics=True, verbose=False)
    bridge.load_checkpoint(CKPT)
    cpd._freeze_phase1_gates(bridge) if hasattr(cpd, "_freeze_phase1_gates") else None
    rm = bridge.region_manager
    all_idx = []
    pool_slices = {}
    for p in pools:
        idx = list(rm.indices(p)); pool_slices[p] = (len(all_idx), len(all_idx) + len(idx)); all_idx += idx
    all_arr = xp.asarray(all_idx, dtype=xp.int64)
    lang_arr = xp.asarray(list(rm.indices("language_input")), dtype=xp.int64)

    def capture(word):
        drive = orthogonal_drive_pattern(cue_idx=word_to_idx[word], n_cues=len(words),
                                         n_neurons=N_LANG, drive_max_pA=DRIVE_PA, sparsity=SPARSITY)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(RESET):
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[lang_arr] = xp.asarray(drive, dtype=xp.float32)
        acc = xp.zeros(len(all_idx), dtype=xp.float64)
        for _ in range(STIM):
            bridge._run_one_simulation_step()
            acc += bridge.cp_firing_states[all_arr].astype(xp.float64)
        bridge.cp_external_input_current[:] = 0.0
        return to_host(acc) / STIM

    # capture distributed codes + pool-label recognition
    codes = {}; label_ok = 0
    for w in words:
        a = capture(w); codes[w] = _center(a)
        rates = {p: a[s:e].mean() for p, (s, e) in pool_slices.items()}
        pred = max(rates, key=rates.get)
        label_ok += int(pred == word_to_pool[w])
    label_acc = label_ok / len(words)
    print(f"  POOL-LABEL recognition: {label_acc:.3f} ({label_ok}/{len(words)})", flush=True)

    # bind/QA on the DISTRIBUTED codes
    D = len(all_idx)
    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    rng = np.random.default_rng(42)
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in RM.ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    bbridge, bidx = P.build(42, D, xp)
    qa_ok = tot = 0
    for _ in range(10):
        pk = rng.choice(len(words), 6, replace=False)
        facts = [{"agent": words[pk[3*f]], "action": words[pk[3*f+1]], "patient": words[pk[3*f+2]]} for f in range(2)]
        bounds = [RM.bind_fact_spiking(bbridge, bidx, fc, codes, roles, D, xp) for fc in facts]
        f = facts[rng.integers(2)]
        who = None
        for b in bounds:
            if (RM.unbind_spiking(bbridge, bidx, b, "action", roles, codes, words, D, xp) == f["action"] and
                    RM.unbind_spiking(bbridge, bidx, b, "patient", roles, codes, words, D, xp) == f["patient"]):
                who = RM.unbind_spiking(bbridge, bidx, b, "agent", roles, codes, words, D, xp); break
        qa_ok += int(who == f["agent"]); tot += 1
    qa = qa_ok / tot
    print(f"  DISTRIBUTED-code bind/QA (who): {qa:.3f}", flush=True)
    print(f"\nRESULT: pool-label={label_acc:.3f} vs distributed-bind-QA={qa:.3f}", flush=True)
    if qa > label_acc + 0.15:
        print("VERDICT: DISTRIBUTED >> LABEL -- the 28-word front-end limit is partly a READOUT artifact; "
              "the bind on distributed codes exceeds the pool-label -> effective conversational vocab is "
              "larger than the label suggests. A real path past the v17 wall.", flush=True)
    elif qa >= label_acc - 0.1:
        print("VERDICT: distributed ~ label -- the distributed codes are about as separable as the label; "
              "the 28-word limit is a genuine code-separability limit, not just readout.", flush=True)
    else:
        print("VERDICT: distributed < label -- bind adds noise; the label readout is the better signal here.",
              flush=True)


if __name__ == "__main__":
    main()
