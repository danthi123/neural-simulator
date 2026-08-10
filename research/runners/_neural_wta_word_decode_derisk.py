"""Neural WTA word-decode -- burn the host cosine-argmax in the PRODUCTION speaker (CPU de-risk).

The production speaker (`_realcorpus_full_frame_speech_derisk.ConceptFrameSpeaker.spell()`) drives a
word's concept pool, reads `language_output` ON SPIKES, then decodes the spoken word with a HOST
`best = max(vocab, key=lambda w: _cosine(spike, patterns[w]))`. That argmax is HOST-computed -- a
shortcut (the read-out selection is the host's bookkeeping, not the brain's).

This replaces the host argmax with a NEURAL FS-WTA read-out -- the SAME validated one-of-K spiking
WTA (`build_fswta_score_bridge`/`fswta_drive`) that the reslm read-out parity ran to K=200
(2026-07-13-reslm-SPIKING-readout-parity-12seed-GO: "parity tracks the score MARGIN, not K";
parity 1.000 at V=200 when scores are discriminable). Path:

    language_output spike pattern
      -> per-word SYNAPTIC drive = projection of the spike pattern onto each word-assembly's
         afferent weights (= its taught reference pattern; a Hebbian read-out matrix)   [16 scores]
      -> K=16 competing word-assembly pools, shared inhibitory FS pool (LATERAL INHIBITION)
      -> the winner fires first, recruits FS, FS suppresses runners-up
      -> a CLEAN one-of-K SPIKING winner == the decoded word.

Because every word's `orthogonal_drive_pattern` reference has the SAME active count at the SAME pA,
all reference patterns are EQUAL-NORM, so argmax(cosine) == argmax(dot): the host cosine's per-word
norm division is a constant across words and cannot change the winner. The neural WTA therefore
targets exactly the host cosine's winner; the open question the map flagged is whether the REAL
16-word decode MARGINS survive the FS-WTA at K=16 (vs the K=3 hand-calibrated soft-WTA).

Ceiling  = host cosine-argmax spell accuracy (the production speaker).
GO gate  = neural-WTA spell accuracy matches/approaches host on >=5/6 seeds
           (per seed: neural_acc >= host_acc - 1/16  AND  parity(WTA==host) > 0.9  AND
            shuffle_parity < 0.5).
Anti-cheat = SHUFFLE control: drive the WTA with PERMUTED scores -> its winner's agreement with the
             true host winner collapses to chance -> the WTA reads the ACTUAL scores (no host leak).

Honest scope: the 16 `language_output` spike patterns come from the single trained seed42 v16 bridge
(the production speaker substrate; only seed42 exists in bridges/v16/). The 6 seeds vary the FS-WTA
read-out bridge's NEURAL heterogeneity + shuffle RNG -- testing that a neural WTA resolves the 16 REAL
decode margins robustly. A 6-bridge SPEAKER sweep (retrain v16 at 6 seeds) is the GPU follow-on. The
per-word score = spike . reference is a synaptic projection computed on host here (as in the reslm
read-out parity); wiring it as real on-bridge synapses (language_output -> 16 word pools) is the
follow-on -- only the ARGMAX is the piece THIS rung moves onto spikes.

NO `sim/` edit; reuse-by-import. numpy backend (small FS-WTA bridge; the v16 speaker runs on numpy
too -- ~7s/word). Requires bridges/v16/seed42.simstate.h5.

Run:
  PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._neural_wta_word_decode_derisk \
    --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/neural_wta_word_decode/neural_wta_16word_6seed.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

N_LANG = 2048


# ---------------------------------------------------------------------------
# STAGE A -- the expensive part: drive the REAL v16 bridge, one pattern per word.
# Cached to .npz so the 6-seed FS-WTA loop does not re-run the bridge.
# ---------------------------------------------------------------------------
def _hard_reset_state(bridge):
    """Zero the Izhikevich state (membrane V, recovery u, firing) so each spell is an INDEPENDENT
    decode event -- the grandmother-architecture ceiling. Without this, driving 16 words back-to-back
    at reset_steps=50 carries adaptation between spells (the production `speak_frame` only spells 2
    words/frame; the spell() docstring flags a larger reset fully decays this)."""
    if getattr(bridge, "cp_izh_c_reset", None) is not None:
        bridge.cp_membrane_potential_v[:] = bridge.cp_izh_c_reset
    else:
        bridge.cp_membrane_potential_v[:] = -65.0
    if getattr(bridge, "cp_recovery_variable_u", None) is not None:
        bridge.cp_recovery_variable_u[:] = 0.0
    if getattr(bridge, "cp_firing_states", None) is not None:
        bridge.cp_firing_states[:] = False
    bridge.cp_external_input_current[:] = 0.0


def collect_spike_patterns(bridge_path: str, bridge_seed: int, reset_steps: int = 200,
                           hard_reset: bool = True):
    from research.runners._realcorpus_full_frame_speech_derisk import ConceptFrameSpeaker
    from research.runners.concept_speak_demo import drive_pool_and_read_lang_output

    spk = ConceptFrameSpeaker(bridge_path, seed=bridge_seed)
    vocab = list(spk.vocab)
    ref = {w: np.asarray(spk.patterns[w], dtype=np.float64) for w in vocab}
    spikes = {}
    for w in vocab:
        if hard_reset:
            _hard_reset_state(spk.bridge)
        spikes[w] = np.asarray(
            drive_pool_and_read_lang_output(
                spk.bridge, spk.pool_of[w], n_lang_output=N_LANG, reset_steps=reset_steps
            ),
            dtype=np.float64,
        )
    return vocab, ref, spikes


def load_or_collect(bridge_path: str, bridge_seed: int, cache: str, reset_steps: int = 200,
                    hard_reset: bool = True):
    cpath = Path(cache) if cache else None
    if cpath is not None and cpath.exists():
        d = np.load(cpath, allow_pickle=False)
        vocab = list(d["__vocab__"])
        spikes = {w: d["spike__" + w] for w in vocab}
        ref = {w: d["ref__" + w] for w in vocab}
        return vocab, ref, spikes, False
    vocab, ref, spikes = collect_spike_patterns(bridge_path, bridge_seed, reset_steps, hard_reset)
    if cpath is not None:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        save = {"__vocab__": np.asarray(vocab)}
        for w in vocab:
            save["spike__" + w] = spikes[w]
            save["ref__" + w] = ref[w]
        np.savez(cpath, **save)
    return vocab, ref, spikes, True


# ---------------------------------------------------------------------------
# Read-out scores + the two decoders (host cosine ceiling; neural FS-WTA).
# ---------------------------------------------------------------------------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def host_cosine_winner(spike: np.ndarray, ref: dict, vocab: list) -> int:
    return int(np.argmax([_cosine(spike, ref[w]) for w in vocab]))


def word_scores(spike: np.ndarray, ref: dict, vocab: list) -> np.ndarray:
    """SYNAPTIC drive per word-assembly = projection of the language_output spike pattern onto each
    word's afferent weight vector (= its taught reference pattern). spike>=0, ref>=0 -> scores>=0."""
    return np.asarray([float(np.dot(spike, ref[w])) for w in vocab], dtype=np.float64)


def condition_scores(s: np.ndarray, mode: str = "center") -> np.ndarray:
    """Afferent gain-control on the 16 word-assembly drives BEFORE the FS-WTA -- the missing companion
    process of the raw synaptic projection. The raw dot scores carry a large COMMON-MODE baseline (a
    word's language_output pattern overlaps EVERY reference pattern, so all 16 scores are large; the
    discriminative signal is their DIFFERENCE). This is precisely the common-mode-convergence trap the
    instrument_required gate warns of. A shared feedforward-inhibition pool (de Almeida-Idiart-Lisman
    E%-max; the FS interneurons the real circuit runs alongside the projection) SUBTRACTS that baseline
    so the relative margins land in the WTA's resolvable range. Every mode is MONOTONIC (preserves the
    host argmax by construction); the OPEN question the de-risk answers is whether the SPIKING WTA then
    resolves the conditioned margins -- the shuffle control proves it reads the drive, not a host leak.

      max     : s / max(s)            -- NO common-mode removal (the naive proxy; parity ~0.75-0.81).
      center  : (s - min) / (max-min) -- subtractive feedforward-inhibition baseline removal (PRIMARY).
      softmax : divisive-norm exp     -- sharper E%-max gain-control (parity ~1.0, = host exactly).
    """
    s = np.maximum(np.asarray(s, dtype=np.float64), 0.0)
    if mode == "max":
        return s / (s.max() + 1e-12)
    if mode == "center":
        z = s - s.min()
        return z / (z.max() + 1e-12)
    if mode == "softmax":
        e = np.exp((s - s.max()) / (0.5 * (s.std() + 1e-9)))
        return e / (e.max() + 1e-12)
    raise ValueError(f"unknown score mode: {mode!r}")


def run_wta_seed(vocab, ref, spikes, fs_seed, settle=25, input_gain=1200.0, score_mode="center"):
    """One FS-WTA read-out bridge (seed=fs_seed) decodes all 16 words on spikes; also the host cosine
    ceiling + the shuffle anti-cheat. Returns per-seed metrics."""
    from research.runners._d3_spiking_attractor_derisk import build_fswta_score_bridge, fswta_drive

    K = len(vocab)
    sb = build_fswta_score_bridge(seed=int(fs_seed), K=K)
    rng = np.random.RandomState(int(fs_seed) + 7)

    n_host = n_dot = n_neural = n_parity = n_shuf_parity = 0
    margins = []
    per = []
    for i, w in enumerate(vocab):
        spike = spikes[w]
        cos_win = host_cosine_winner(spike, ref, vocab)
        s = word_scores(spike, ref, vocab)
        dot_win = int(np.argmax(s))
        sn = condition_scores(s, mode=score_mode)  # afferent gain-control (feedforward-inhibition baseline); monotonic

        _, acc = fswta_drive(sb, K, sn, input_gain=input_gain, settle=settle)
        wta_win = int(np.argmax(acc)) if float(acc.max()) > 0.0 else -1

        perm = rng.permutation(K)
        _, acc_sh = fswta_drive(sb, K, sn[perm], input_gain=input_gain, settle=settle)
        sh_win = int(np.argmax(acc_sh)) if float(acc_sh.max()) > 0.0 else -1

        n_host += int(cos_win == i)
        n_dot += int(dot_win == i)
        n_neural += int(wta_win == i)
        n_parity += int(wta_win == dot_win)
        n_shuf_parity += int(sh_win == dot_win)

        srt = np.sort(sn)[::-1]
        margins.append(float(srt[0] - srt[1]))
        per.append({
            "word": w, "target_idx": i,
            "host_cosine_win": vocab[cos_win], "dot_win": vocab[dot_win],
            "neural_wta_win": vocab[wta_win] if wta_win >= 0 else None,
            "shuffle_win": vocab[sh_win] if sh_win >= 0 else None,
            "margin_top1_top2": round(float(srt[0] - srt[1]), 4),
        })

    K16 = float(K)
    return {
        "fs_seed": int(fs_seed),
        "host_cosine_acc": n_host / K16,
        "host_dot_acc": n_dot / K16,
        "neural_wta_acc": n_neural / K16,
        "parity_wta_vs_dot": n_parity / K16,
        "shuffle_parity": n_shuf_parity / K16,
        "mean_margin": float(np.mean(margins)),
        "min_margin": float(np.min(margins)),
        "per_word": per,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default="bridges/v16/seed42.simstate.h5")
    ap.add_argument("--bridge-seed", type=int, default=42,
                    help="seed used to REBUILD the v16 speaker before load_checkpoint (heterogeneity)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102],
                    help="FS-WTA read-out bridge seeds (neural heterogeneity of the WTA)")
    ap.add_argument("--reset-steps", type=int, default=200)
    ap.add_argument("--no-hard-reset", action="store_true",
                    help="disable the per-word Izhikevich state reset (=production reset_steps=50 carryover)")
    ap.add_argument("--settle", type=int, default=25)
    ap.add_argument("--input-gain", type=float, default=1200.0)
    ap.add_argument("--score-mode", choices=["max", "center", "softmax"], default="center",
                    help="afferent gain-control before the FS-WTA (center = feedforward-inhibition common-mode removal)")
    ap.add_argument("--cache", default="research/findings/raw/neural_wta_word_decode/_v16_seed42_spikes.npz")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    t0 = time.time()
    vocab, ref, spikes, collected = load_or_collect(
        a.bridge, a.bridge_seed, a.cache, a.reset_steps, hard_reset=not a.no_hard_reset)
    print(f"[patterns] {'collected from bridge' if collected else 'loaded from cache'}; "
          f"vocab={len(vocab)} ({time.time()-t0:.0f}s)", flush=True)

    seeds_out = []
    for sd in a.seeds:
        r = run_wta_seed(vocab, ref, spikes, sd, settle=a.settle, input_gain=a.input_gain,
                         score_mode=a.score_mode)
        matches = (r["neural_wta_acc"] >= r["host_cosine_acc"] - 1.0 / len(vocab)
                   and r["parity_wta_vs_dot"] > 0.9 and r["shuffle_parity"] < 0.5)
        r["seed_go"] = bool(matches)
        seeds_out.append(r)
        print(f"  fs_seed={sd:4d}  host_cos={r['host_cosine_acc']:.3f}  "
              f"neural_wta={r['neural_wta_acc']:.3f}  parity={r['parity_wta_vs_dot']:.3f}  "
              f"shuffle_parity={r['shuffle_parity']:.3f}  mean_margin={r['mean_margin']:.3f}  "
              f"[{'GO' if matches else 'MISS'}]", flush=True)

    n_go = sum(int(r["seed_go"]) for r in seeds_out)
    host_mean = float(np.mean([r["host_cosine_acc"] for r in seeds_out]))
    neural_mean = float(np.mean([r["neural_wta_acc"] for r in seeds_out]))
    parity_mean = float(np.mean([r["parity_wta_vs_dot"] for r in seeds_out]))
    shuf_mean = float(np.mean([r["shuffle_parity"] for r in seeds_out]))
    chance = 1.0 / len(vocab)

    # ANTI-CHEAT attribution: how much of the WTA's agreement with the host winner is attributable to it
    # READING the true scores (treatment=parity) vs a generic sorter (control=shuffled-score parity)?
    from tools.lab import attributable_to
    frac = attributable_to("FS-WTA reading the true drive (parity vs shuffled-score parity)",
                            parity_mean, shuf_mean)
    summary_attribution = None if frac is None else round(float(frac), 4)

    # EARNED VERDICT — the preconditions travel with the result (tools/gates/verdict_preconditions).
    from tools.verdict import Verdict
    n_seeds = len(a.seeds)
    v = Verdict("neural-WTA word decode vs host cosine @ vocab-16", chance=chance)
    v.floor("neural-WTA spell accuracy vs chance", neural_mean, chance)
    v.require("parity(WTA==host winner) > 0.9", parity_mean, expect=lambda x: x > 0.9)
    v.control("WTA reads TRUE drive (parity vs shuffled-score parity)", parity_mean, shuf_mean,
              min_separation=0.4)
    v.require("neural-WTA matches host within 1/16", neural_mean - (host_mean - chance),
              expect=lambda x: x >= 0.0)
    v.require(">=5/6 seeds match host", n_go, expect=lambda x: x >= max(5, n_seeds - 1))
    decided = v.decide(go=(n_go >= max(5, n_seeds - 1)), verbose=False)
    verdict = decided["status"]

    summary = {
        "runner": "research.runners._neural_wta_word_decode_derisk",
        "bridge": a.bridge, "vocab_size": len(vocab), "vocab": vocab,
        "n_seeds": len(a.seeds), "n_go": n_go, "verdict": verdict,
        "host_cosine_acc_mean": host_mean, "neural_wta_acc_mean": neural_mean,
        "parity_wta_vs_dot_mean": parity_mean, "shuffle_parity_mean": shuf_mean,
        "parity_attributable_to_true_drive": summary_attribution,
        "chance": chance,
        "settle": a.settle, "input_gain": a.input_gain, "score_mode": a.score_mode,
        "wall_clock_s": round(time.time() - t0, 1),
        "preconditions": decided["preconditions"],
        "undefined_reasons": decided["undefined_reasons"],
        "per_seed": seeds_out,
    }
    print(f"\n[VERDICT] {verdict} -- {n_go}/{len(a.seeds)} seeds match host. "
          f"host_cosine={host_mean:.3f}  neural_wta={neural_mean:.3f}  "
          f"parity={parity_mean:.3f}  shuffle_parity={shuf_mean:.3f} (chance {1.0/len(vocab):.3f})",
          flush=True)

    if a.out:
        op = Path(a.out)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(summary, indent=2))
        print(f"[OUT] wrote {op}", flush=True)


if __name__ == "__main__":
    main()
