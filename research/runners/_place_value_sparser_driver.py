"""Thin FOREGROUND driver for the #5 self-org place value-grading δ de-risk
(SPARSER-fields sweep). Calls run_moving_goal_episode(stage_b_smoke=True)
directly, captures the returned dict, prints the STAGE-B VERDICT, and dumps
JSON. NOT committed. See task: research/findings/raw/_place_value_sparser_derisk.json.

Usage:
  SIM_BACKEND=cupy python -m research.runners._place_value_sparser_driver \
    --seed 42 --tag base --fs-to-place-weight 8 --fs-to-place-density 0.4 \
    --place-fs-weight 16 --n-place 200 --n-place-fs 24 \
    --value-train-stdp-w-max 40 --out research/findings/raw/_pvs_base_seed42.json
"""
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
import json
import time
import argparse
import numpy as np

from research.runners.g11_bg_runner import run_moving_goal_episode


def build_multi_schedule(gs):
    far = (max(0, gs - 2), max(0, gs - 2))
    far_west = (max(0, 1), max(0, gs - 2))
    sw = (max(0, 1), max(0, 1))
    far_se = (max(0, gs - 2), max(0, 1))
    return [(0, far), (450, far_west), (900, sw), (1350, far_se)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid-size", type=int, default=32)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--out", type=str, required=True)
    # sparsity levers
    ap.add_argument("--n-place", type=int, default=200)
    ap.add_argument("--n-place-fs", type=int, default=24)
    ap.add_argument("--place-fs-weight", type=float, default=16.0)
    ap.add_argument("--place-fs-density", type=float, default=0.4)
    ap.add_argument("--fs-to-place-weight", type=float, default=8.0)
    ap.add_argument("--fs-to-place-density", type=float, default=0.4)
    ap.add_argument("--place-sensors-to-place-weight", type=float, default=28.0)
    ap.add_argument("--place-sensors-to-place-density", type=float, default=0.5)
    ap.add_argument("--selforg-steps", type=int, default=2000)
    ap.add_argument("--selforg-n-positions", type=int, default=40)
    # value-train
    ap.add_argument("--value-train-trials", type=int, default=40)
    ap.add_argument("--value-train-stdp-w-max", type=float, default=40.0)
    ap.add_argument("--critic-fs-weight", type=float, default=16.0)
    ap.add_argument("--coincidence-threshold", type=int, default=12)
    # sparsify-during-selforg env lever (open the FS-PING during STEP-1)
    ap.add_argument("--sparsify-fs-selforg", action="store_true",
                    help="set N5_SPARSIFY_FS_DURING_SELFORG=1 (carve fields WITH recurrent FS inhibition)")
    # grid front end (the decorrelated/sparser afferent; #5b R1 SURPASS) + determinism
    ap.add_argument("--grid-frontend", action="store_true",
                    help="use the spatial-phase grid-cell metric as the place_sensors afferent (decorrelated => sparser/selective fields)")
    ap.add_argument("--grid-drive-scale", type=float, default=2.5)
    ap.add_argument("--grid-n-modules", type=int, default=6)
    ap.add_argument("--grid-n-per-module", type=int, default=33)
    ap.add_argument("--deterministic-selforg", action="store_true",
                    help="toggle cfg.deterministic_transpose_matvec during STEP-1 self-org (reproducible place code)")
    ap.add_argument("--deterministic-read", action="store_true",
                    help="hold deterministic_transpose_matvec ON through value-train + delta-read")
    args = ap.parse_args()

    if args.sparsify_fs_selforg:
        os.environ["N5_SPARSIFY_FS_DURING_SELFORG"] = "1"
    else:
        os.environ["N5_SPARSIFY_FS_DURING_SELFORG"] = "0"

    gs = args.grid_size
    sched = build_multi_schedule(gs)
    t0 = time.time()
    print("=" * 72, flush=True)
    print(f"[driver tag={args.tag} seed={args.seed}] SPARSER-fields de-risk: "
          f"n_place={args.n_place} n_fs={args.n_place_fs} place_fs_w={args.place_fs_weight} "
          f"fs_to_place_w={args.fs_to_place_weight} fs_to_place_d={args.fs_to_place_density} "
          f"sparsify_fs_selforg={args.sparsify_fs_selforg} vt_w_max={args.value_train_stdp_w_max}",
          flush=True)
    print("=" * 72, flush=True)

    res = run_moving_goal_episode(
        out_path=args.out + ".episode_unused.json",  # stage_b returns early; not written
        seed=args.seed,
        n_steps=1800,
        grid_size=gs,
        goal_schedule=sched,
        # --- neural critic + self-org place (the finding's config) ---
        enable_neural_critic=True,
        spiking_reward_us=True,
        enable_critic_homeostasis=True,
        enable_critic_fs_inhibition=True,
        critic_fs_weight=float(args.critic_fs_weight),
        neural_place_selforg=True,
        stage_b_smoke=True,
        value_train_trials=int(args.value_train_trials),
        value_train_stdp_w_max=float(args.value_train_stdp_w_max),
        coincidence_threshold=int(args.coincidence_threshold),
        # --- sparsity levers ---
        n_place=int(args.n_place),
        n_place_fs=int(args.n_place_fs),
        place_fs_weight=float(args.place_fs_weight),
        place_fs_density=float(args.place_fs_density),
        fs_to_place_weight=float(args.fs_to_place_weight),
        fs_to_place_density=float(args.fs_to_place_density),
        place_sensors_to_place_weight=float(args.place_sensors_to_place_weight),
        place_sensors_to_place_density=float(args.place_sensors_to_place_density),
        selforg_steps=int(args.selforg_steps),
        selforg_n_positions=int(args.selforg_n_positions),
        # --- grid front end (decorrelated afferent => sparser fields) + determinism ---
        nav_critic_grid_frontend=bool(args.grid_frontend),
        grid_drive_scale=float(args.grid_drive_scale),
        grid_n_modules=int(args.grid_n_modules),
        grid_n_per_module=int(args.grid_n_per_module),
        deterministic_selforg=bool(args.deterministic_selforg),
        deterministic_read=bool(args.deterministic_read),
        verbose=True,
    )
    elapsed = time.time() - t0

    out = {
        "tag": args.tag,
        "seed": args.seed,
        "grid_size": gs,
        "elapsed_s": elapsed,
        "levers": {
            "n_place": args.n_place,
            "n_place_fs": args.n_place_fs,
            "place_fs_weight": args.place_fs_weight,
            "place_fs_density": args.place_fs_density,
            "fs_to_place_weight": args.fs_to_place_weight,
            "fs_to_place_density": args.fs_to_place_density,
            "place_sensors_to_place_weight": args.place_sensors_to_place_weight,
            "place_sensors_to_place_density": args.place_sensors_to_place_density,
            "selforg_steps": args.selforg_steps,
            "selforg_n_positions": args.selforg_n_positions,
            "value_train_trials": args.value_train_trials,
            "value_train_stdp_w_max": args.value_train_stdp_w_max,
            "critic_fs_weight": args.critic_fs_weight,
            "coincidence_threshold": args.coincidence_threshold,
            "sparsify_fs_selforg": args.sparsify_fs_selforg,
            "grid_frontend": args.grid_frontend,
            "grid_drive_scale": args.grid_drive_scale,
            "grid_n_modules": args.grid_n_modules,
            "grid_n_per_module": args.grid_n_per_module,
            "deterministic_selforg": args.deterministic_selforg,
            "deterministic_read": args.deterministic_read,
        },
        "result": res,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # Pull the headline numbers for a compact one-line summary the controller reads.
    sb = (res or {}).get("stage_b_smoke") or {}
    so = (res or {}).get("selforg") or {}
    vt = (res or {}).get("value_train") or {}
    summary = {
        "tag": args.tag, "seed": args.seed,
        "selforg_sparsity": so.get("sparsity"),
        "selforg_diff_cos": so.get("diff_cos"),
        "w_near": sb.get("w_near"), "w_far": sb.get("w_far"),
        "w_near_over_far": sb.get("w_near_over_far"),
        "crit_near_hz": sb.get("crit_near_hz"), "crit_far_hz": sb.get("crit_far_hz"),
        "snc_pred": sb.get("snc_predicted_near_hz"),
        "snc_unpred": sb.get("snc_unpredicted_far_hz"),
        "delta_gap": sb.get("snc_gap_ratio"),
        "gabab_gap_pass": sb.get("gabab_gap"),
        "lesion_gap": sb.get("lesion_gap_ratio"),
        "lesion_collapses": sb.get("lesion_collapses"),
        "elapsed_s": round(elapsed, 1),
    }
    print("DRIVER_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
