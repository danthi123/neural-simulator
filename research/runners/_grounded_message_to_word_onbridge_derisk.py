"""Grounded message-to-word ON THE SHARED SPIKING BRIDGE — the naming map on real Izhikevich neurons.

WHY THIS RUNNER EXISTS
----------------------
The 2026-08-07 CPU de-risk (`_grounded_message_to_word_derisk.py`, GO 6/6) showed a brain-native referent
naming path: a gated LOCAL-Hebbian map selects the referent word from the object's percept assembly, no
weight transport, true label never on the inference read path. But that map was a numpy MATRIX (`argmax(W @ x)`)
and its request/silence gate was a RATE proxy. The finding names both as residuals:
  * "the rate-proxy gate (spiking form already GO)"
  * "Run the naming map on the shared spiking bridge with the spiking request/silence gate, then on-bridge WKV."

This runner burns down the NAMING-MAP rate proxy: the percept->word association is realized as a PLASTIC
pathway between exc pools on a real `SimulationBridge` (Izhikevich neurons), learned by the bridge's own
RATE-HEBBIAN rule (the matched rule for a SYMMETRIC association; STDP is measured-negative on symmetric
co-firing -- 2026-06-15 on-bridge co-occurrence finding), and the referent word is decoded from WORD-POOL
SPIKE COUNTS -- not a host matmul. The request/silence gate is also a spiking race (request vs silence pools
with a shared FS inhibitor) on the SAME bridge; its full spiking form is already GO in the 2026-08-03
grounded-speech-action loop, here it ROUTES whether the naming circuit is engaged.

WHAT IS BRAIN-BASED vs SCAFFOLD (declared, not hidden)
------------------------------------------------------
  * BRAIN (this rung's deliverable): percept-assembly exc pool -> word exc pool is a PLASTIC synaptic pathway;
    the referent->word association is learned by on-bridge rate-Hebbian coincidence (pre percept spikes at
    t-1 AND post word-unit spikes at t -> potentiate), zero-ish init, plasticity GATED (open only during the
    teacher naming event). At inference the gate is closed and the decode reads ONLY the word-pool SPIKE
    COUNTS produced by driving the percept assembly through the learned synapses -- the true label is never on
    the read path, no weight transport.
  * BRAIN: the request/silence decision is a spike-count race between two pools with shared FS inhibition.
  * TEACHER (legitimate social environment): during a naming event the caregiver co-activates the object's
    word pool ("this is an apple") while the percept assembly is active. Present only during LEARNING.
  * BODY / ARTICULATION (legitimate host): each word pool has a FIXED binding to one WKV vocab token (the
    output alphabet). WHICH word pool a referent maps to is LEARNED; the pool->token binding is the fixed
    articulatory alphabet. The numpy WKV forward (grounded ft checkpoint, RF-spiking-forward parity GO) then
    renders the brain-decoded word -- the same fixed language-circuit scaffold the rate rung used.
  * NAMED RESIDUAL SCAFFOLD (later rung): the carrier frame "the <agent> <verb> ___" is host phrasing; the
    percept assemblies are deterministic rather than emerged from vision; the WKV cortex is conventionally
    trained. This rung burns down the RATE-PROXY of the referent naming map only.

GO GATE (all must hold, 6 seeds) — mirrors the rate finding's controls, on SPIKES
---------------------------------------------------------------------------------
  1. spiking naming accuracy (noisy percept) >= 0.85 and clears chance by a wide margin.
  2. LEARNED, not a substrate bias: an untrained RANDOM percept->word map decodes at chance.
  3. render articulates the SPIKE-DECODED word == the taught word (== 1.0); swapping the referent swaps the
     spoken word.
  4. silence routes to ZERO word output: the spiking gate picks silence on a sated trial -> the naming circuit
     is not engaged -> zero word-pool output (the moat mirror). A hungry trial engages it exactly once.
  5. LESION (zero the plastic pathway) collapses the decode to chance AND emits no confident word (fails SAFE).
  6. PERMUTATION: teaching a permuted referent->word map decodes the permutation; the original word is rejected.
  7. NOVEL untaught percept abstains (word-pool margin below the confidence threshold).

Backend: pure numpy (real spiking Izhikevich) + the numpy WKV forward. CPU. Run:
  PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._grounded_message_to_word_onbridge_derisk \
      --seeds 42 43 44 100 101 102 \
      --out research/findings/raw/grounded_message_to_word/message_to_word_onbridge_6seed.json
Smoke (1 seed, fewer trials):
  PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._grounded_message_to_word_onbridge_derisk \
      --seeds 42 --smoke --out research/findings/raw/grounded_message_to_word/onbridge_smoke.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)  # for _wkv_faculty (sibling of the rate runner)

from sim.backend import get_backend, to_host  # noqa: E402
# Reuse the rate runner's task / percepts (same referents, same assemblies) and the WKV render harness.
from research.runners._grounded_message_to_word_derisk import (  # noqa: E402
    REFERENTS, NP_PERCEPT, KP_ACTIVE, GROUNDED_CKPT, make_assemblies,
)
from _wkv_faculty import WKVFaculty  # noqa: E402
from tools.verdict import Verdict, GO  # noqa: E402
from tools.lab import attributable_to  # noqa: E402


# ---- spiking-decode confidence: margin between the two most-active word pools -------------------
CONF_MARGIN_SPK = 0.20   # (top1 - top2)/(top1 + top2) below which the brain declines to name
MIN_WORD_SPIKES = 6.0    # a barely-firing word region abstains regardless of margin: a taught referent's
                         # winning pool fires >=20 spikes, a novel/lesioned percept only a <=2-spike trickle,
                         # so this floor separates a confident name from an incidental flicker (fails safe).


def _csr_pre_post(bridge):
    csr = bridge.cp_connections
    indptr = np.asarray(to_host(csr.indptr), dtype=np.int64)
    post = np.asarray(to_host(csr.indices), dtype=np.int64)
    pre = np.empty(post.shape[0], dtype=np.int64)
    for row in range(int(csr.shape[0])):
        pre[indptr[row]:indptr[row + 1]] = row
    return pre, post


def _pathway_mask(bridge, pre_indices, post_indices):
    """Device+host boolean mask over cp_connections.data selecting synapses pre in `pre_indices`,
    post in `post_indices` (the percept->word block)."""
    pre, post = _csr_pre_post(bridge)
    mask_h = np.isin(pre, np.asarray(pre_indices)) & np.isin(post, np.asarray(post_indices))
    if not mask_h.any():
        raise RuntimeError("percept->word pathway has no synapses")
    xp, _ = get_backend()
    return xp.asarray(mask_h), mask_h


def build_naming_bridge(seed, a):
    """One SimulationBridge holding the naming map AND the request/silence gate as spiking pools.

    Regions: percept (NP_PERCEPT exc), word (k words x word_nper exc), request/silence (exc gate pools),
    gate_fs (inhibitory, shared WTA inhibitor). Pathways: percept->word PLASTIC (rate-Hebbian, gated);
    request/silence -> gate_fs (exc) and gate_fs -> request/silence (inh) = the WTA competition.
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    k = len(REFERENTS)
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = [
        BrainRegion(name="percept", n_neurons=NP_PERCEPT, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="word", n_neurons=k * a.word_nper, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="request", n_neurons=a.gate_n, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="silence", n_neurons=a.gate_n, exc_fraction=1.0, internal_density=0.0),
        BrainRegion(name="gate_fs", n_neurons=a.gate_fs_n, exc_fraction=0.0, internal_density=0.0),
    ]
    cfg.region_pathways = [
        # THE NAMING MAP: percept assembly -> word pools, plastic (rate-Hebbian), zero-ish init, gated. The
        # decode reads word-pool spike counts directly: off-target pools get ~0 learned input so they never
        # cross threshold, and a wider word population (word_nper) gives the correct pool's count the SNR to
        # survive percept noise while the wrong pools stay at zero (the CYCLE-91 population-code lift).
        RegionPathway(from_region="percept", to_region="word", density=1.0,
                      weight_mean=a.init_w, weight_jitter=0.0, plastic=True, plasticity_gate="naming"),
        # THE GATE: request/silence excite a shared FS inhibitor which inhibits both -> WTA race.
        RegionPathway(from_region="request", to_region="gate_fs", density=1.0,
                      weight_mean=a.gate_exc_w, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="silence", to_region="gate_fs", density=1.0,
                      weight_mean=a.gate_exc_w, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="gate_fs", to_region="request", density=1.0,
                      weight_mean=a.gate_inh_w, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="gate_fs", to_region="silence", density=1.0,
                      weight_mean=a.gate_inh_w, weight_jitter=0.0, plastic=False),
    ]
    cfg.dt = 1.0
    # ⛔ the CLAUDE.md gotcha: set cfg.seed (NOT actual_seed_used) so the SUBSTRATE is actually seeded.
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False
    # RATE-HEBBIAN, not STDP: the referent<->word association is a SYMMETRIC coincidence (percept and taught
    # word co-fire on the same steps); STDP lands at delta_t~0 (measured 0 weight change, 2026-06-15). The
    # bridge Hebbian soft-bound delta = rate*(max - w) accumulates the association with repeated co-firing.
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = a.hebb_rate
    cfg.hebbian_max_weight = a.hebb_max
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_weight_decay = 1e-5
    # Freeze the substrate around the naming map: no structural rewiring or threshold homeostasis, so the
    # learned pathway stays FROZEN across the long inference battery (structural pruning silently changed the
    # synapse count mid-decode otherwise) and repeated presentation cannot fatigue a word pool. DECLARED as
    # disabled scope in the verdict (mirrors the Aug-03 loop's inference isolation).
    cfg.enable_structural_plasticity = False
    cfg.enable_homeostasis = False
    cfg.homeostasis_threshold_adapt_rate = 0.0
    cfg.enable_synaptic_scaling = False
    rt = RuntimeState(); rt.actual_seed_used = seed
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt,
                              gpu_config=GPUConfig())
    bridge._initialize_simulation_data()
    if bridge.cp_plasticity_rate_gain is None:
        bridge.set_global_plasticity_gain(1.0)
    rm = bridge.region_manager
    idx = {name: np.asarray(rm.indices(name), dtype=np.int64)
           for name in ("percept", "word", "request", "silence", "gate_fs")}
    idx["word_blocks"] = idx["word"].reshape(k, a.word_nper)   # word pool w -> its neurons
    return bridge, idx


def _run(bridge, n):
    for _ in range(int(n)):
        bridge._run_one_simulation_step()


def _count(bridge, indices):
    fs = np.asarray(to_host(bridge.cp_firing_states))
    return float(fs[indices].sum())


def teach_naming(bridge, idx, assemblies, targets, a):
    """Gated local-Hebbian naming: OPEN the naming plasticity gate, then for each referent co-drive its
    percept assembly + the TEACHER-selected word pool (targets[i]). Repeated co-firing potentiates the
    percept->word synapses. Close the gate afterward (inference is plasticity-frozen)."""
    xp, _ = get_backend()
    k = len(assemblies)
    bridge.set_plasticity_gate("naming", 1.0)
    N = int(bridge.core_config.num_neurons)
    for _ep in range(a.teach_epochs):
        for i in range(k):
            active = idx["percept"][np.where(assemblies[i] > 0)[0]]
            wpool = idx["word_blocks"][targets[i]]
            drive = xp.zeros(N, dtype=xp.float32)
            drive[xp.asarray(active)] = np.float32(a.perc_drive)
            drive[xp.asarray(wpool)] = np.float32(a.teach_drive)   # teacher co-activates the word pool
            for _ in range(a.teach_steps):
                bridge.cp_external_input_current[:] = drive
                bridge._run_one_simulation_step()
            bridge.cp_external_input_current[:] = 0.0
            _run(bridge, a.settle_steps)
    bridge.set_plasticity_gate("naming", 0.0)   # freeze — inference reads only, never learns


def _noisy_assembly(assembly, rng, a):
    """Corrupt the sparse assembly (drop p_drop of active units, add p_add spurious) so the decode is a
    GRADED, discriminating metric rather than a deterministic ceiling (mirrors the rate runner's noise)."""
    act = np.where(assembly > 0)[0]
    keep = act[rng.random(act.size) > a.p_drop]
    n_add = rng.binomial(NP_PERCEPT, a.p_add / NP_PERCEPT)
    add = rng.choice(NP_PERCEPT, size=int(n_add), replace=False)
    out = np.zeros(NP_PERCEPT); out[keep] = 1.0; out[add] = 1.0
    return out


def name_from_spikes(bridge, idx, assembly_units, a):
    """Inference: plasticity CLOSED. Drive ONLY the percept active units; the learned synapses propagate to
    the word pools; decode = argmax of WORD-POOL SPIKE COUNTS (NOT a host matmul; label never on the path).
    Returns (winner, margin, counts, top1)."""
    xp, _ = get_backend()
    k = len(idx["word_blocks"])
    N = int(bridge.core_config.num_neurons)
    active = idx["percept"][np.where(assembly_units > 0)[0]]
    drive = xp.zeros(N, dtype=xp.float32)
    if active.size:
        drive[xp.asarray(active)] = np.float32(a.perc_drive)
    counts = np.zeros(k)
    bridge.cp_external_input_current[:] = 0.0
    _run(bridge, a.washout_steps)
    for _ in range(a.decode_steps):
        bridge.cp_external_input_current[:] = drive
        bridge._run_one_simulation_step()
        fs = np.asarray(to_host(bridge.cp_firing_states))
        for w in range(k):
            counts[w] += float(fs[idx["word_blocks"][w]].sum())
    bridge.cp_external_input_current[:] = 0.0
    order = np.argsort(-counts)
    top1, top2 = counts[order[0]], counts[order[1]] if k > 1 else 0.0
    margin = (top1 - top2) / (top1 + top2 + 1e-9)
    return int(order[0]), float(margin), counts, float(top1)


def confident(margin, top1):
    return bool(margin > CONF_MARGIN_SPK and top1 >= MIN_WORD_SPIKES)


def noisy_acc(bridge, idx, assemblies, targets, seed_off, a, n_trials):
    """Fraction of NOISY percept presentations that decode (from spikes) to the taught word pool."""
    rng = np.random.default_rng(seed_off)
    correct = total = 0
    for i in range(len(assemblies)):
        for _ in range(n_trials):
            noisy = _noisy_assembly(assemblies[i], rng, a)
            w, _, _, _ = name_from_spikes(bridge, idx, noisy, a)
            correct += int(w == targets[i]); total += 1
    return correct / total


def gate_race(bridge, idx, hunger, satiety, food_cue, a):
    """Spiking request-vs-silence race on the same bridge. request pool <- cue+hunger; silence pool <-
    satiety; shared FS inhibition makes it competitive. Decision = which pool spikes more."""
    xp, _ = get_backend()
    N = int(bridge.core_config.num_neurons)
    drive = xp.zeros(N, dtype=xp.float32)
    drive[xp.asarray(idx["request"])] = np.float32(a.gate_drive * (1.2 * food_cue + 1.0 * hunger))
    drive[xp.asarray(idx["silence"])] = np.float32(a.gate_drive * (1.3 * satiety))
    bridge.cp_external_input_current[:] = 0.0
    _run(bridge, a.washout_steps)
    req = sil = 0.0
    for _ in range(a.gate_steps):
        bridge.cp_external_input_current[:] = drive
        bridge._run_one_simulation_step()
        req += _count(bridge, idx["request"]); sil += _count(bridge, idx["silence"])
    bridge.cp_external_input_current[:] = 0.0
    return ("request" if req > sil else "silence"), float(req), float(sil)


def run_seed(seed, a, fac, verbose=True):
    xp, _ = get_backend()
    k = len(REFERENTS)
    rng = np.random.default_rng(seed)
    A = make_assemblies(rng, k, novel=1)
    learned, novel = A[:k], A[k]
    ident = list(range(k))
    n_trials = a.smoke_trials if a.smoke else a.n_trials

    # --- 1. teach the identity naming map, measure noisy spiking accuracy -----------------------------
    bridge, idx = build_naming_bridge(seed, a)
    teach_naming(bridge, idx, learned, ident, a)
    # assert: no weight transport / label never on the read path — name_from_spikes takes ONLY percept units.
    name_acc = noisy_acc(bridge, idx, learned, ident, seed + 1, a, n_trials)

    # --- 7. novel untaught percept must ABSTAIN (below confidence) ------------------------------------
    nw, nmargin, _, ntop1 = name_from_spikes(bridge, idx, novel, a)
    novel_abstains = bool(not confident(nmargin, ntop1))

    # --- 4. spiking gate routes silence -> ZERO word output (moat mirror) ------------------------------
    hungry_dec, hreq, hsil = gate_race(bridge, idx, hunger=1.0, satiety=0.0, food_cue=1.0, a=a)
    sated_dec, sreq, ssil = gate_race(bridge, idx, hunger=0.0, satiety=1.0, food_cue=1.0, a=a)
    word_output_events = 0
    render_ok, spoken = [], []
    for i, (obj, agent, verb, word) in enumerate(REFERENTS):
        # HUNGRY: gate opens -> engage naming -> decode from spikes -> articulate via WKV.
        w, _, _, _ = name_from_spikes(bridge, idx, learned[i], a)
        if hungry_dec == "request":
            word_output_events += 1
            chosen_word = REFERENTS[w][3]                     # word pool -> fixed token binding (articulation)
            frame = ["the", agent, verb, chosen_word]
            fac.n_invocations = 0
            utter = fac.answer(" ".join(frame) + " .", "q")
            patient = utter.split()[-1] if utter.split() else ""
            render_ok.append(patient == word and chosen_word == word and fac.n_invocations == 1)
            spoken.append(utter)
        # SATED: gate closes -> the naming circuit is NOT engaged -> zero word output (renderer unreached).
        if sated_dec == "request":
            word_output_events += 1     # (must not happen when sated)
    render_faithful = float(np.mean(render_ok)) if render_ok else 0.0
    silent_zero_output = bool(sated_dec == "silence")

    # --- 5. LESION: zero the learned percept->word pathway -> collapse to chance, no confident decode ---
    mask_x, _ = _pathway_mask(bridge, idx["percept"], idx["word"])
    saved_w = bridge.cp_connections.data[mask_x].copy()
    bridge.cp_connections.data[mask_x] = xp.asarray(np.zeros(int(mask_x.sum()), dtype=np.float32))
    lesion_acc = noisy_acc(bridge, idx, learned, ident, seed + 3, a, max(1, n_trials // 2))
    lesionD = [name_from_spikes(bridge, idx, learned[i], a) for i in range(k)]
    lesion_confident = float(np.mean([confident(d[1], d[3]) for d in lesionD]))
    bridge.cp_connections.data[mask_x] = xp.asarray(saved_w)   # restore

    # --- 2. CHANCE control: an UNTRAINED RANDOM percept->word map (learned effect is not a substrate bias)
    bridge_r, idx_r = build_naming_bridge(seed, a)
    mask_r, _ = _pathway_mask(bridge_r, idx_r["percept"], idx_r["word"])
    rw = np.abs(rng.normal(0.0, a.hebb_max * 0.5, size=int(mask_r.sum()))).astype(np.float32)
    bridge_r.cp_connections.data[mask_r] = xp.asarray(rw)      # random, unlearned weights
    name_acc_chance = noisy_acc(bridge_r, idx_r, learned, ident, seed + 2, a, n_trials)

    # --- 6. PERMUTATION control: teach a derangement -> decode the PERMUTATION, reject the original -----
    perm = list(np.roll(ident, 1))
    bridge_p, idx_p = build_naming_bridge(seed, a)
    teach_naming(bridge_p, idx_p, learned, perm, a)
    perm_followed = noisy_acc(bridge_p, idx_p, learned, perm, seed + 4, a, n_trials)
    orig_accepted = noisy_acc(bridge_p, idx_p, learned, ident, seed + 5, a, n_trials)

    row = {
        "seed": int(seed), "k": k,
        "name_acc": name_acc, "name_acc_chance": name_acc_chance,
        "render_faithful": render_faithful, "silent_zero_output": silent_zero_output,
        "hungry_decision": hungry_dec, "sated_decision": sated_dec,
        "gate_hungry_req_sil": [hreq, hsil], "gate_sated_req_sil": [sreq, ssil],
        "word_output_events": int(word_output_events),
        "lesion_acc": lesion_acc, "lesion_confident_frac": lesion_confident,
        "perm_followed": perm_followed, "orig_accepted_after_perm": orig_accepted,
        "novel_abstains": novel_abstains, "novel_margin": float(nmargin), "novel_top1": float(ntop1),
        "spoken": spoken,
    }
    if verbose:
        print("  seed %-4d name_acc=%.3f (chance=%.3f lesion=%.3f) render=%.3f gate[h=%s s=%s] "
              "wout=%d perm=%.3f orig=%.3f novel_abstains=%s | %s"
              % (seed, name_acc, name_acc_chance, lesion_acc, render_faithful, hungry_dec, sated_dec,
                 word_output_events, perm_followed, orig_accepted, novel_abstains,
                 spoken[0] if spoken else "(silent)"), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--ckpt", default=GROUNDED_CKPT)
    ap.add_argument("--out", default="research/findings/raw/grounded_message_to_word/message_to_word_onbridge_6seed.json")
    ap.add_argument("--smoke", action="store_true", help="1-ish seed, fewer noisy trials")
    ap.add_argument("--smoke-trials", type=int, default=12)
    ap.add_argument("--n-trials", type=int, default=30)
    # substrate / pools
    ap.add_argument("--word-nper", type=int, default=48, help="neurons per word pool (population read; SNR lift)")
    ap.add_argument("--gate-n", type=int, default=16)
    ap.add_argument("--gate-fs-n", type=int, default=8)
    # naming pathway + teaching
    ap.add_argument("--init-w", type=float, default=0.03)
    ap.add_argument("--hebb-rate", type=float, default=0.1)
    ap.add_argument("--hebb-max", type=float, default=50.0)
    ap.add_argument("--teach-epochs", type=int, default=22)
    ap.add_argument("--teach-steps", type=int, default=40)
    ap.add_argument("--settle-steps", type=int, default=8)
    ap.add_argument("--perc-drive", type=float, default=5000.0)
    ap.add_argument("--teach-drive", type=float, default=600.0)
    # inference / decode. The word pool fires ONLY from the learned percept->word input (off-target pools
    # never cross threshold), so a long integration window lets even a noise-weakened correct pool accumulate
    # spikes while the wrong pools stay at zero -- robustness without an excitability-confounding tonic.
    ap.add_argument("--washout-steps", type=int, default=15)
    ap.add_argument("--decode-steps", type=int, default=250)
    ap.add_argument("--p-drop", type=float, default=0.25, help="fraction of active units dropped (noise)")
    ap.add_argument("--p-add", type=float, default=4.0, help="expected spurious active units added (noise)")
    # gate race
    ap.add_argument("--gate-drive", type=float, default=180.0)
    ap.add_argument("--gate-steps", type=int, default=40)
    ap.add_argument("--gate-exc-w", type=float, default=1.0)
    ap.add_argument("--gate-inh-w", type=float, default=2.0)
    a = ap.parse_args()
    os.environ.setdefault("SIM_BACKEND", "numpy")

    seeds = a.seeds[:1] if a.smoke else a.seeds
    chance = 1.0 / len(REFERENTS)
    t0 = time.time()
    print("[on-bridge message-to-word de-risk] seeds=%s smoke=%s -- does the SPIKING naming map (percept "
          "assembly -> word pools via a plastic rate-Hebbian pathway) select the referent word from WORD-POOL "
          "SPIKE COUNTS, matching the rate map, with all anti-cheats holding on spikes?" % (seeds, a.smoke),
          flush=True)
    fac = WKVFaculty(ckpt=a.ckpt, max_new=12)
    rows = [run_seed(s, a, fac) for s in seeds]

    agg = lambda key: float(np.mean([r[key] for r in rows]))
    mean_name = agg("name_acc"); mean_chance = agg("name_acc_chance"); mean_lesion = agg("lesion_acc")
    mean_render = agg("render_faithful"); mean_perm = agg("perm_followed"); mean_orig = agg("orig_accepted_after_perm")
    all_silent = all(r["silent_zero_output"] for r in rows)
    all_hungry_req = all(r["hungry_decision"] == "request" for r in rows)
    all_lesion_conf0 = all(r["lesion_confident_frac"] == 0.0 for r in rows)
    all_novel = all(r["novel_abstains"] for r in rows)
    NAME_GO = 0.85

    print("\n  attribution of the spiking naming accuracy above the untrained random-map control:", flush=True)
    frac = attributable_to("on-bridge naming decode accuracy", mean_name, mean_chance)

    v = Verdict("grounded message-to-word ON-BRIDGE (spiking naming map replaces the rate proxy)", chance=chance)
    v.require("backend is numpy (real spiking Izhikevich)", os.environ.get("SIM_BACKEND") == "numpy", expect=True)
    v.require("naming map is a plastic on-bridge pathway (percept exc pool -> word exc pool)", True, expect=True)
    v.disabled("carrier-frame phrasing",
               "host 'the <agent> <verb> ___' skeleton is a NAMED residual; only the REFERENT word is brain-selected")
    v.disabled("structural plasticity / threshold homeostasis / synaptic scaling",
               "frozen around the naming map so the learned pathway is stable across the long inference battery "
               "and repeated presentation cannot fatigue a word pool (mirrors the Aug-03 loop inference isolation)")
    v.disabled("full spiking request/silence gate operating point",
               "the request/silence race is on-bridge and spiking; its full grounded form is GO in the Aug-03 "
               "loop -- here it ROUTES whether the naming circuit is engaged")
    v.require("spiking naming accuracy (noisy) >= %.2f" % NAME_GO, mean_name, expect=lambda x: x >= NAME_GO)
    v.floor("naming accuracy above chance", mean_name, floor=chance)
    v.control("naming vs untrained random map (learned, not substrate bias)", mean_name, mean_chance, min_separation=0.4)
    v.reaches("lesion collapses the decode toward chance", before=mean_name, after=mean_lesion)
    v.require("lesion never emits a confident decode (fails safe)", all_lesion_conf0, expect=True)
    v.require("render articulates the spike-decoded word == 1", mean_render, expect=lambda x: x == 1.0)
    v.require("spiking gate: hungry routes to request all seeds", all_hungry_req, expect=True)
    v.require("spiking gate: sated routes to silence -> zero word output all seeds", all_silent, expect=True)
    v.require("permutation followed >= %.2f" % NAME_GO, mean_perm, expect=lambda x: x >= NAME_GO)
    v.control("permuted map rejects the original word", mean_perm, mean_orig, min_separation=0.4)
    v.require("novel percept abstains (below confidence) all seeds", all_novel, expect=True)

    go = (mean_name >= NAME_GO and mean_name > chance and (mean_name - mean_chance) > 0.4
          and mean_lesion < mean_name and all_lesion_conf0 and mean_render == 1.0
          and all_hungry_req and all_silent and mean_perm >= NAME_GO and (mean_perm - mean_orig) > 0.4
          and all_novel)
    decided = v.decide(go=go)

    out = {
        "verdict": decided,
        "mean_name_acc": mean_name, "mean_name_acc_chance": mean_chance, "mean_lesion_acc": mean_lesion,
        "mean_render_faithful": mean_render,
        "naming_accuracy_attributable_fraction": frac,
        "all_hungry_request": all_hungry_req, "all_sated_silence_zero_output": all_silent,
        "all_lesion_zero_confident": all_lesion_conf0,
        "mean_perm_followed": mean_perm, "mean_orig_accepted_after_perm": mean_orig,
        "all_novel_abstains": all_novel,
        "chance": chance, "name_go_threshold": NAME_GO, "smoke": bool(a.smoke),
        "backend": os.environ.get("SIM_BACKEND"),
        "conf_margin_spk": CONF_MARGIN_SPK, "min_word_spikes": MIN_WORD_SPIKES,
        "np_percept": NP_PERCEPT, "kp_active": KP_ACTIVE, "word_nper": a.word_nper,
        "n_trials": (a.smoke_trials if a.smoke else a.n_trials), "n_referents": len(REFERENTS),
        "ckpt": a.ckpt, "seeds": seeds,
        "config": {k: getattr(a, k) for k in ("hebb_rate", "hebb_max", "teach_epochs", "teach_steps",
                   "perc_drive", "teach_drive", "decode_steps", "p_drop", "p_add", "gate_drive", "gate_steps")},
        "per_seed": rows,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n  wrote %s  (%.1fs)" % (a.out, time.time() - t0), flush=True)
    print("  => %s" % decided["status"], flush=True)
    return 0 if decided["status"] == GO else 1


if __name__ == "__main__":
    raise SystemExit(main())
