"""Activity-level integration: derive the composition layer's phasor
symbols from the project substrate's actual population activity, not
from a discrete recognized label.

The validated two-system compositional pipeline
(``spiking_phasor_integration.py``) joins the concept substrate to the
spiking-phasor FHRR composition layer at the concept-IDENTITY level:
the substrate reports one discrete recognized pool label, and a fixed
lookup table maps that label to a pre-assigned phasor symbol.

This runner is the activity-LEVEL integration: the phasor symbol is
DERIVED from the substrate's per-neuron concept-pool population
activity vector directly -- no discrete label, no lookup table. The
substrate's actual neural activity is the input to the composition
layer.

The cheap-first probe (``activity_level_integration_probe.py``,
verdict ACTIVITY_LEVEL_REACHABLE) established this is reachable AND
handed this build one hard requirement: the activity vector must be
the substrate's DISTRIBUTED per-neuron concept-pool activity (the
coarse per-pool aggregate fails -- too few degrees of freedom to
average the trial noise). This runner uses the full per-neuron
concept-pool population vector.

Pipeline (per seed):
1. Load a validated v14/v16 + hippocampus substrate (the 800-event
   Phase-1 cache -- the SAME substrate the identity-level integration
   used, for a clean comparison).
2. CAPTURE: drive each concept word M times through the validated
   v14/v16 drive path; each time, record the per-neuron firing-rate
   vector over the whole concept-pool population. The M observations
   differ by the substrate's genuine trial-to-trial variability.
3. DERIVE: a fixed random complex projection maps a population
   activity vector to a phasor symbol (the deriver the probe
   validated). The same fixed deriver is used for every symbol.
4. REGISTER: a clean-up vocabulary symbol per concept = the deriver
   applied to the MEAN activity over K registration observations (the
   consolidated, stable concept identity).
5. COMPOSE: encode (cue, filler) facts via the REUSED, byte-unchanged
   spiking-phasor FHRR subsystem; storage and query each draw an
   INDEPENDENT activity observation of the cue concept, so the
   storage-time and query-time symbols differ by real substrate noise.
6. Measure integrated accuracy against the project's frozen 0.80 bar.

Two accuracies are reported, honestly:
- integrated accuracy: the whole pipeline (recognition imperfection
  and all -- whatever activity the substrate produced).
- composition-only accuracy: restricted to facts whose cue and filler
  observations were both recognized correctly (per-pool argmax ==
  target pool) -- isolates whether the activity-derived FHRR
  composition itself works.

PRE-REGISTERED reading (fixed; never tuned):
- If integrated multi-seed mean >= 0.80 at loads {2,3,5}: activity-
  level integration WORKS end-to-end on the real substrate -- the
  phasor symbol can be derived from the substrate's actual population
  activity, no discrete-label lookup table needed.
- If integrated < 0.80 but composition-only is high: the activity-
  derived FHRR composition works; the bottleneck is the substrate's
  recognition (its population activity is too ambiguous on some
  words) -- which relocates the open problem and is itself the
  finding.

Reuse-by-import: the validated v14/v16 + hippocampus substrate builder
and the validated spiking-phasor FHRR subsystem, both byte-unchanged.
No protected/frozen/moat module modified. No autograd. The activity ->
phasor deriver is the only net-new piece and lives in this runner.

Kill-safe: the M-observation capture per seed is cached to disk; a
re-run skips already-captured seeds. The numpy composition is fast and
re-run from cache each time. --smoke shrinks the substrate + capture
for a fast end-to-end check (toy numbers NOT a result). Plain ASCII.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the validated substrate builder + Phase-1 helpers.
from research.runners.unified_per_regime_monitor_runner import (
    _build_bridge_with_phase1_recipe,
    _phase1_cache_path,
    _phase1_recipe,
    _freeze_phase1_gates,
    _all_words_word_to_idx,
    _all_pool_regions,
    _direct_pool_target,
    _N_WORDS_ORTHOGONAL,
)

# Reuse the validated spiking-phasor FHRR composition subsystem.
from research.runners.spiking_phasor_fhrr import (
    SpikingPhasorFHRR, phase_similarity, phases_to_spikes,
)

SEEDS = [42, 43, 44]
RECOG_CACHE = "research/findings/raw/unified_per_regime/phase1_800ev"
CACHE_DIR = "research/findings/raw/activity_level_integration_cache"
N_DIM = 512                 # phasor dimension (matches the validated subsystem)
LOADS = [2, 3, 5]
N_TRIALS = 300
BAR = 0.80                  # the project's frozen compositional bar
M_OBS = 16                  # activity observations captured per word
K_VOCAB = 8                 # registration observations averaged for vocabulary
DERIV_SEED = 12345          # fixed -- the deriver is a fixed interface property
DRIVE_PA = 200.0
STIM_STEPS = 100
RESET_STEPS = 50
SPARSITY = 0.05


def build_substrate(seed):
    """Build + load a validated v14/v16 + hippocampus substrate for one
    seed (reuse-by-import; byte-unchanged builder). The 800-event cache
    matches only the full architecture, so the substrate is always full
    scale; --smoke shrinks the capture, not the substrate."""
    bridge = _build_bridge_with_phase1_recipe(seed=seed, tiny_synth=False)
    cache_path = _phase1_cache_path(RECOG_CACHE, seed)
    bridge.load_checkpoint(str(cache_path))
    _freeze_phase1_gates(bridge)
    return bridge


def pool_layout(bridge):
    """Concept-pool index layout: the concatenated per-neuron index
    array over ALL concept pools, plus per-pool (start, end) slices.
    The activity vector is the substrate's firing over this population."""
    rm = bridge.region_manager
    all_pools = _all_pool_regions(enable_adjective=True)
    all_idx = []
    slices = {}
    cursor = 0
    for p in all_pools:
        idx = list(rm.indices(p))
        slices[p] = (cursor, cursor + len(idx))
        all_idx.extend(idx)
        cursor += len(idx)
    return np.asarray(all_idx, dtype=np.int64), slices, all_pools


def capture_activity(bridge, word, all_idx, recipe_dims, word_to_idx,
                      n_words_for_orthogonal):
    """Drive lang_input(word) via the validated v14/v16 path and record
    the per-neuron firing-rate vector over the concept-pool population.

    Mirrors the validated direct-binding drive (orthogonal codes,
    sparsity 0.05, drive 200 pA, 50-step reset + 100-step stim) -- the
    ONLY difference from ``measure_pool_firing`` is per-neuron capture
    instead of a per-pool sum."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    from sim.text_embeddings import orthogonal_drive_pattern

    rm = bridge.region_manager
    lang_input_idx = cp.asarray(list(rm.indices("language_input")),
                                dtype=cp.int64)
    idx_arr = cp.asarray(all_idx, dtype=cp.int64)

    drive_in = orthogonal_drive_pattern(
        cue_idx=word_to_idx[word], n_cues=n_words_for_orthogonal,
        n_neurons=int(recipe_dims["n_lang_input"]),
        drive_max_pA=DRIVE_PA, sparsity=SPARSITY,
    )
    drive_gpu = cp.asarray(drive_in, dtype=cp.float32)

    bridge.cp_external_input_current[:] = 0.0
    for _ in range(RESET_STEPS):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1

    bridge.cp_external_input_current[lang_input_idx] = drive_gpu
    counts = cp.zeros(idx_arr.shape[0], dtype=cp.float64)
    for _ in range(STIM_STEPS):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        counts += bridge.cp_firing_states[idx_arr]
    return np.asarray(to_host(counts), dtype=np.float64) / float(STIM_STEPS)


def recognized_pool(activity, slices, all_pools):
    """The substrate's own readout: the concept pool with the highest
    mean per-neuron firing in this activity vector (argmax over pools)."""
    best_pool, best_rate = None, -1.0
    for p in all_pools:
        s, e = slices[p]
        rate = float(np.mean(activity[s:e]))
        if rate > best_rate:
            best_rate, best_pool = rate, p
    return best_pool


def make_deriver(n_dim, d_act, deriv_seed):
    """The fixed activity -> phasor-symbol deriver: a fixed random
    complex projection of the normalized activity vector; the phase of
    each projected component is one phasor dimension. Deterministic and
    smooth. Returns spikes in the validated subsystem's native format."""
    drng = np.random.default_rng(deriv_seed)
    w_re = drng.normal(0.0, 1.0, size=(n_dim, d_act))
    w_im = drng.normal(0.0, 1.0, size=(n_dim, d_act))

    def derive(activity):
        a = np.asarray(activity, dtype=np.float64)
        norm = np.linalg.norm(a)
        a_hat = a / (norm + 1e-9)
        z = w_re @ a_hat + 1j * (w_im @ a_hat)
        phases = np.mod(np.angle(z) / (2.0 * np.pi), 1.0)
        return phases_to_spikes(phases)

    return derive


def activity_cv(obs_matrix, n_active):
    """Trial-to-trial coefficient of variation of the genuinely-active
    population: take the n_active highest-mean neurons, return the mean
    of (std across observations / mean across observations)."""
    mean_v = obs_matrix.mean(axis=0)
    std_v = obs_matrix.std(axis=0)
    active = np.argsort(mean_v)[-n_active:]
    m = mean_v[active]
    s = std_v[active]
    return float(np.mean(s / np.maximum(m, 1e-9)))


def capture_seed(seed, cache_path, m_obs):
    """Capture m_obs activity observations of every concept word for one
    seed. Cached to disk -- a re-run loads the cache and skips capture."""
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        words = [str(w) for w in data["__words__"]]
        obs = {w: data["obs__" + w] for w in words}
        clean = {w: data["clean__" + w] for w in words}
        all_pools = [str(p) for p in data["__pools__"]]
        slices = {p: tuple(int(x) for x in data["slice__" + p])
                  for p in all_pools}
        print(f"  [seed {seed}] loaded cached activity ({len(words)} words, "
              f"{obs[words[0]].shape[0]} obs/word)")
        return obs, clean, slices, all_pools, words

    print(f"  [seed {seed}] capturing {m_obs} activity observations/word "
          f"(no cache) ...")
    t0 = time.time()
    bridge = build_substrate(seed)
    all_idx, slices, all_pools = pool_layout(bridge)
    recipe_dims = _phase1_recipe(False)
    all_words, word_to_idx = _all_words_word_to_idx()
    n_words_for_orthogonal = max(_N_WORDS_ORTHOGONAL, len(all_words))

    obs = {}
    clean = {}
    for word in all_words:
        try:
            target = _direct_pool_target(word)
        except KeyError:
            continue
        rows = []
        flags = []
        for _ in range(m_obs):
            a = capture_activity(bridge, word, all_idx, recipe_dims,
                                  word_to_idx, n_words_for_orthogonal)
            rows.append(a)
            flags.append(recognized_pool(a, slices, all_pools) == target)
        obs[word] = np.asarray(rows, dtype=np.float64)
        clean[word] = np.asarray(flags, dtype=bool)
    words = list(obs.keys())

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save = {"__words__": np.asarray(words),
            "__pools__": np.asarray(all_pools)}
    for w in words:
        save["obs__" + w] = obs[w]
        save["clean__" + w] = clean[w]
    for p in all_pools:
        save["slice__" + p] = np.asarray(slices[p], dtype=np.int64)
    np.savez(cache_path, **save)
    print(f"  [seed {seed}] captured + cached in {time.time()-t0:.1f}s")
    return obs, clean, slices, all_pools, words


def run_one_seed(seed, tag, m_obs, k_vocab, n_trials):
    """Activity-level integration for one substrate seed."""
    print(f"\n--- seed {seed} ---")
    cache_path = os.path.join(CACHE_DIR, f"{tag}_seed{seed}.npz")
    obs, clean, slices, all_pools, words = capture_seed(
        seed, cache_path, m_obs)

    d_act = obs[words[0]].shape[1]
    deriver = make_deriver(N_DIM, d_act, DERIV_SEED)

    # Cue words = nouns + verbs; filler words = adjectives (mirrors the
    # identity-level integration runner's task definition exactly).
    cue_words = [w for w in words
                 if _direct_pool_target(w).startswith(("noun_pool_",
                                                        "verb_pool_"))]
    filler_words = [w for w in words
                    if _direct_pool_target(w).startswith("adjective_pool_")]

    # Clean-up vocabulary: the deriver applied to the MEAN activity over
    # K registration observations -- the consolidated concept identity.
    vocab = {fw: deriver(obs[fw][:k_vocab].mean(axis=0)) for fw in filler_words}

    # Diagnostics: measured trial-to-trial variability + recognition rate.
    n_active = max(1, d_act // len(all_pools))
    cvs = [activity_cv(obs[w], n_active) for w in words]
    clean_rate = float(np.mean([clean[w].mean() for w in words]))

    net = SpikingPhasorFHRR(N_DIM, np.random.default_rng(seed))
    qrng = np.random.default_rng(seed + 1)

    def true_pool(word):
        return _direct_pool_target(word)

    per_load = {}
    for load in LOADS:
        n_int_correct = n_int_total = 0
        n_comp_correct = n_comp_total = 0
        for _ in range(n_trials):
            cues = list(qrng.choice(cue_words, size=load, replace=False))
            fills = list(qrng.choice(filler_words, size=load, replace=True))
            # Each fact's storage symbols come from one observation;
            # the query symbol comes from an INDEPENDENT observation.
            facts = []
            for (c, f) in zip(cues, fills):
                ci_enc = int(qrng.integers(m_obs))
                fi_enc = int(qrng.integers(m_obs))
                ci_qry = int(qrng.integers(m_obs))
                facts.append((c, f, ci_enc, fi_enc, ci_qry))
            composite = net.encode([
                (deriver(obs[c][ci_enc]), deriver(obs[f][fi_enc]))
                for (c, f, ci_enc, fi_enc, ci_qry) in facts
            ])
            for (c, f, ci_enc, fi_enc, ci_qry) in facts:
                recovered = net.query(composite, deriver(obs[c][ci_qry]))
                sims = {fw: phase_similarity(recovered, vocab[fw])
                        for fw in filler_words}
                best = max(sims, key=sims.get)
                hit = (true_pool(best) == true_pool(f))
                n_int_correct += int(hit)
                n_int_total += 1
                if clean[c][ci_enc] and clean[f][fi_enc] and clean[c][ci_qry]:
                    n_comp_correct += int(hit)
                    n_comp_total += 1
        int_acc = n_int_correct / n_int_total
        comp_acc = (n_comp_correct / n_comp_total) if n_comp_total else float("nan")
        per_load[load] = {
            "integrated_accuracy": int_acc,
            "composition_only_accuracy": comp_acc,
            "n_composition_only": n_comp_total,
        }
        print(f"  L={load}: integrated acc={int_acc:.4f} | "
              f"composition-only acc={comp_acc:.4f} (n={n_comp_total})")

    print(f"  [seed {seed}] activity dim={d_act}; measured trial-to-trial "
          f"CV mean={np.mean(cvs):.4f}; recognition-clean rate={clean_rate:.4f}")
    return {
        "seed": seed,
        "activity_dim": int(d_act),
        "m_obs": int(m_obs),
        "activity_cv_mean": float(np.mean(cvs)),
        "recognition_clean_rate": clean_rate,
        "per_load": per_load,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="short capture (few obs/word, 1 seed, fewer "
                         "trials) on the full substrate -- a fast "
                         "end-to-end check; toy numbers, NOT a result")
    args = ap.parse_args()

    smoke = bool(args.smoke)
    tag = "smoke" if smoke else "full"
    m_obs = 4 if smoke else M_OBS
    k_vocab = 2 if smoke else K_VOCAB
    n_trials = 40 if smoke else N_TRIALS
    seeds = [42] if smoke else SEEDS

    print("=== activity-level integration: substrate population activity "
          "-> spiking-phasor FHRR composition ===")
    if smoke:
        print("  *** SMOKE MODE: short capture on the full substrate, "
              "toy numbers, NOT a result ***")
    print(f"recognition substrate: {RECOG_CACHE} seeds {seeds}; "
          f"FHRR N_dim={N_DIM}; loads={LOADS}; bar={BAR}; "
          f"obs/word={m_obs}; vocab-K={k_vocab}; trials={n_trials}")

    seed_results = [run_one_seed(s, tag, m_obs, k_vocab, n_trials)
                    for s in seeds]

    print(f"\n=== MULTI-SEED AGGREGATE ===")
    agg = {}
    all_pass = True
    for load in LOADS:
        int_accs = [r["per_load"][load]["integrated_accuracy"]
                    for r in seed_results]
        comp_accs = [r["per_load"][load]["composition_only_accuracy"]
                     for r in seed_results]
        mean_int = float(np.mean(int_accs))
        mean_comp = float(np.mean([c for c in comp_accs if c == c]))
        agg[load] = {"mean_integrated": mean_int,
                     "per_seed_integrated": int_accs,
                     "mean_composition_only": mean_comp}
        if mean_int < BAR:
            all_pass = False
        print(f"  L={load}: integrated per-seed="
              f"{['%.3f' % a for a in int_accs]} mean={mean_int:.4f} "
              f"({'>=' if mean_int >= BAR else '<'} {BAR}) | "
              f"composition-only mean={mean_comp:.4f}")

    print(f"\n=== VERDICT ===")
    if smoke:
        verdict = "SMOKE"
        print("  SMOKE run -- toy numbers, not propagated as a result.")
    elif all_pass:
        verdict = "ACTIVITY_LEVEL_INTEGRATED_PASS"
        print("  Activity-level integration clears the frozen 0.80 bar "
              "multi-seed mean at all loads: the spiking-phasor symbol is "
              "derived from the substrate's actual population activity -- "
              "no discrete-label lookup table.")
    else:
        verdict = "RECOGNITION_BOUNDED"
        print("  Integrated multi-seed mean is below 0.80 at some load; if "
              "composition-only stays high the bottleneck is the "
              "substrate's recognition (population activity too ambiguous).")

    out = {
        "seeds": seeds, "recognition_cache": RECOG_CACHE, "n_dim": N_DIM,
        "loads": LOADS, "n_trials": n_trials, "bar": BAR,
        "m_obs": m_obs, "k_vocab": k_vocab, "deriv_seed": DERIV_SEED,
        "smoke": smoke,
        "per_seed": seed_results,
        "aggregate": {str(k): v for k, v in agg.items()},
        "verdict": verdict,
    }
    out_path = f"research/findings/raw/activity_level_integration_{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
