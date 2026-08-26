"""B1 V1 self-org — 6-SEED FLIP-SOAK = the gate that decides whether BRAIN_V1_SELFORG may be flipped default-ON.

Flip criterion (the de-risk's self-org GO bar, on the production substrate, 6/6):
    all 6 seeds develop orientation selectivity: osi_post_frac >= 0.5 AND osi_post_frac >= osi_pre_frac + 0.15.
The LOAD-BEARING lesion oracle (seed 42): freeze (no learning) and shuffle (orientation-destroyed input) must NOT
reach that bar -- proving any orientation would come from the LEARNING on ORIENTED input, not the support/substrate.

This reuses-by-import the production organ (research.runners.v1_selforg_production_organ), so it soaks EXACTLY the
bank the production wiring would install. The organ carries the clock-advance fix, so if `--rule stdp` is passed the
timing rule is genuinely exercised (it was silently inert in every prior on-bridge run).

Run (production scale, cupy):
  SIM_BACKEND=cupy python -u -m research.runners._b1_v1_selforg_flip_soak \
      --seeds 42 43 44 45 46 47 --dev-steps 24000 \
      --out research/findings/raw/_b1_v1_selforg_flip_soak_6seed.json
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

from research.runners.v1_selforg_production_organ import selforg_v1_rf_bank  # noqa: E402
from sim.visual_cortex import N_ORIENTATIONS, N_FREQUENCIES, V1_POSITIONS_PER_DIM, RETINA_SIZE  # noqa: E402


def _bank_metrics(seed, a, rule, dog, lesion):
    t0 = time.time()
    _, _, _, m = selforg_v1_rf_bank(
        seed, n_orientations=a.n_orient, n_frequencies=a.n_freq,
        n_positions_per_dim=a.n_pos, retina_size=a.retina_size,
        receptive_field_radius=a.radius, dev_steps=a.dev_steps,
        drive_pA=a.drive_pA, rule=rule, dog=dog, lesion=lesion)
    m["elapsed_s"] = round(time.time() - t0, 1)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47])
    ap.add_argument("--n-orient", type=int, default=N_ORIENTATIONS)
    ap.add_argument("--n-freq", type=int, default=N_FREQUENCIES)
    ap.add_argument("--n-pos", type=int, default=V1_POSITIONS_PER_DIM)
    ap.add_argument("--retina-size", type=int, default=RETINA_SIZE)
    ap.add_argument("--radius", type=int, default=4)
    ap.add_argument("--dev-steps", type=int, default=24000)
    ap.add_argument("--drive-pA", type=float, default=1200.0)
    ap.add_argument("--rule", type=str, default="hebbian", choices=["hebbian", "stdp", "both"])
    ap.add_argument("--dog", type=int, default=0)
    ap.add_argument("--lesion-seed", type=int, default=42, help="seed to also run freeze+shuffle lesion controls on")
    ap.add_argument("--out", type=str, default="research/findings/raw/_b1_v1_selforg_flip_soak_6seed.json")
    a = ap.parse_args()

    os.environ.setdefault("SIM_BACKEND", "cupy")
    print(f"[B1 flip-soak] seeds={a.seeds} dev_steps={a.dev_steps} rule={a.rule} dog={bool(a.dog)} "
          f"arch={a.n_orient}x{a.n_freq}x{a.n_pos}x{a.n_pos}", flush=True)

    learn = []
    for s in a.seeds:
        m = _bank_metrics(s, a, a.rule, bool(a.dog), lesion=None)
        learn.append(m)
        print(f"  learn seed={s}: osi_post_frac={m['osi_post_frac']} (pre={m['osi_pre_frac']}) "
              f"on-off={m['on_minus_off_mean']} l2={m['l2_mean']} -> {m['verdict']} ({m['elapsed_s']}s)", flush=True)

    # lesion oracle on the representative seed
    freeze = _bank_metrics(a.lesion_seed, a, a.rule, bool(a.dog), lesion="freeze")
    shuffle = _bank_metrics(a.lesion_seed, a, a.rule, bool(a.dog), lesion="shuffle")
    print(f"  LESION seed={a.lesion_seed}: freeze osi_post_frac={freeze['osi_post_frac']} | "
          f"shuffle osi_post_frac={shuffle['osi_post_frac']} | learn osi_post_frac="
          f"{next(m['osi_post_frac'] for m in learn if m['seed'] == a.lesion_seed)}", flush=True)

    verdicts = [m["verdict"] for m in learn]
    flip = all(v == "GO" for v in verdicts)
    # load-bearing: the learn OSI must clear the lesion controls (else the coupling is not the source)
    learn_seed_osi = next(m["osi_post_frac"] for m in learn if m["seed"] == a.lesion_seed)
    load_bearing = bool(learn_seed_osi >= max(freeze["osi_post_frac"], shuffle["osi_post_frac"]) + 0.15)

    summary = dict(
        flip_decision=("FLIP-ON" if (flip and load_bearing) else "HOLD-OFF"),
        all_seeds_go=bool(flip),
        load_bearing=load_bearing,
        per_seed_verdicts=verdicts,
        osi_post_frac_mean=round(float(np.mean([m["osi_post_frac"] for m in learn])), 4),
        osi_pre_frac_mean=round(float(np.mean([m["osi_pre_frac"] for m in learn])), 4),
        osi_post_frac_min=round(float(np.min([m["osi_post_frac"] for m in learn])), 4),
        on_minus_off_mean=round(float(np.mean([m["on_minus_off_mean"] for m in learn])), 6),
        lesion=dict(seed=a.lesion_seed, freeze_osi_post_frac=freeze["osi_post_frac"],
                    shuffle_osi_post_frac=shuffle["osi_post_frac"], learn_osi_post_frac=learn_seed_osi),
        gate="all 6 osi_post_frac>=0.5 AND >=pre+0.15 AND learn>=lesion_ctrl+0.15",
        rule=a.rule, dog=bool(a.dog), dev_steps=a.dev_steps, seeds=a.seeds,
    )
    out = dict(summary=summary, learn=learn, lesion=dict(freeze=freeze, shuffle=shuffle))
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print("\n" + "=" * 90, flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[written] {outp}", flush=True)


if __name__ == "__main__":
    main()
