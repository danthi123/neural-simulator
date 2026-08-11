"""TEACHER-LOOP SLEEP-REPLAY + bdsp_wmax DE-CLAMP DE-RISK (2026-08-11): push REPLAY-based continual retention
PAST the characterized ~0.55 self-replay cap by removing a SILENT SUBSTRATE CLAMP that was driving much of the
forgetting the prior arc attributed to engram FIDELITY.

THE CAP THIS ATTACKS (name it): 6-seed self-replay `frac_recalled` mean ~0.55 (range 0.20-0.90), no-replay ~0.13,
interleaved ceiling 0.8 -- finding 2026-08-08-teacher-loop-sleep-replay-consolidation-... (runner
_teacher_loop_sleep_replay_consolidation_derisk.py). The budget-sweep NEGATIVE
(2026-08-09-teacher-loop-sleep-replay-budget-sweep-...) mapped the residual to STORE-FIDELITY by showing MORE
replay is flat across 64x -- BUT it held the substrate config FIXED. It never varied the substrate weight clamp.

THE LEVER (candidate: "replay + the bdsp_wmax de-clamp"; consolidation-STRENGTH, NOT the sibling's generator).
The OnBridgeEpropNet parent sets bdsp_w_min/max = -6/+6 while ff_w_init=2000 and w_clip=4000. Even with e-prop as
the sole learner (BDSP lr=0), `fused_bdsp_update` ENDS in an UNCONDITIONAL `cp.clip(w_new, w_min, w_max)`
(kernels.py:485) that RUNS every forward pass on every FF synapse whose presyn fired (bridge.py active_bd), so it
silently CRUSHES the ~2000-scale FF weights toward |w|<=6 as teaching proceeds. Measured (seed 42, teach 5 facts,
numpy): default clamp -> FF |w|mean 229->82, frac|w|<=6 0.42->0.68, no-replay retention 0.40; de-clamped
(hp['bdsp_wmax']=1e9, the port's single-variable clamp lever, config-only) -> FF |w|mean 229->229 PRESERVED,
no-replay retention 0.80 -- immediate acquisition 1.000 in BOTH. The clamp was a hidden catastrophic-forgetting
term. This is the CLAUDE.md "the CLAMP owns the measurement" pattern (gap#5: 97% of a weight change was the clamp);
the prioritized-replay runner already de-clamped (bdsp_wmax=1e9) but the sleep-replay/budget-sweep line did not.

DE-CLAMP IS CONFIG, NOT A HOST SHORTCUT. bdsp_w_max is a substrate CONFIG scalar (the synaptic weight ceiling in
bridge units); the +-6 default is inconsistent with this pathway's ff_w_init=2000 (a units-scale artifact, not a
biological bound on this FF pathway). Widening it restores the intended operating point -- the same lever the port
exposes as `_bw` and the prioritized-replay finding used. Brain-based self-generation is inherited UNCHANGED: the
engram store is the brain's own captured trace and `_self_replay_consolidate` / `Hippocampus.generate_replay` take
NO env (teacher/world ABSENT during sleep). NO sim/ edit; reuse-by-import of every substantive piece.

FIVE ARMS, one world / seed / schedule / budget (the ONLY differences: replay on/off, content lesion, clamp):
  * noreplay_clamped   = the substrate baseline: +-6 clamp, no replay -> the crush + 1/N forgetting.
  * replay_clamped     = +-6 clamp + self-replay -> the RECORD'S ~0.55 cap (the in-run baseline to beat).
  * noreplay_declamped = de-clamp only, no replay -> ISOLATES the clamp's own contribution to forgetting.
  * replay_declamped   = de-clamp + self-replay -> THE CANDIDATE (does it beat the 0.55 cap?).
  * scramble_declamped = de-clamp + content-lesioned replay (labels shuffled, identical compute) -> anti-cheat:
                         if replay ADDS on top of the de-clamp, is that gain the stored CONTENT or just compute?

TEETH -> GO if:
  (a) DE-CLAMPED REPLAY beats the named cap: replay_declamped frac_recalled@N=10 > 0.55 AND > replay_clamped.
  (b) the CLAMP is load-bearing on forgetting: noreplay_declamped > noreplay_clamped (de-clamp alone recovers).
  (c) immediate acquisition stays perfect in replay_declamped (mean immediate acq >= 0.9).
  (d) [honest decomposition, not a gate] attributable_to() splits the declamped-replay retention into the CLAMP
      term (declamped vs clamped) and the replay-CONTENT term (replay vs scramble, both declamped). A NEGATIVE
      here (replay adds nothing once de-clamped) is a first-class result: it would relocate the whole ~0.55->? gap
      onto the clamp and retire replay-magnitude as the lever.

RUN (numpy; the net is ~48 neurons so numpy avoids GPU launch overhead -- verified for this line 2026-08-09):
  single-seed SMOKE:
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      .venv/bin/python -m research.runners._teacher_loop_sleep_replay_declamp_derisk --seed 42 \
        --n-max 10 --milestones 1 5 10 --epochs 40 --replay-epochs 24 --replay-per-fact 16 --n-draws 32 \
        --out research/findings/raw/sleep_replay_declamp_s42.json
  6-SEED POOL command is in the finding (one seed per process).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")   # ~48-neuron net; numpy avoids cupy launch overhead for this size
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
from research.runners._onbridge_eprop_port_derisk import OnBridgeEpropNet  # noqa: E402
# reuse-by-import: ALL substantive machinery (world, teach, held-out acc, corrective batch, engram store, replay).
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _fit_readout_norm_world, _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402
from research.runners._teacher_loop_sleep_replay_consolidation_derisk import (  # noqa: E402
    Hippocampus, _self_replay_consolidate,
)

OUT = _REPO / "research" / "findings" / "raw" / "sleep_replay_declamp.json"
CAP_BASELINE = 0.55   # the NAMED record cap: 6-seed clamped self-replay frac_recalled mean (finding 2026-08-08)


def _mk_net(n_in, k, seed, hidden, settle, eprop_lr, w_clip, bdsp_wmax=None):
    """Identical to _teacher_loop_scaling_derisk._mk_net, plus the port's single-variable CONFIG clamp lever
    hp['bdsp_wmax'] (widens bdsp_w_min/max; None = the +-6 default the banked runs used). No sim/ edit."""
    hp = dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=2000.0, pbar_alpha=0.05,
              in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0)
    if bdsp_wmax is not None:
        hp["bdsp_wmax"] = float(bdsp_wmax)
    return OnBridgeEpropNet(n_in, hidden, k, seed=seed, n_hidden_layers=1, settle_steps=settle,
                            eprop_lr=eprop_lr, eps_leak=0.9, surrogate="atan_vt", alpha_surr=0.15,
                            logit_source="leaky_readout", w_clip=w_clip, hp=hp)


def _ff_scale(net):
    data = np.asarray(net.br.cp_connections.data)
    return {"absmean": float(np.abs(data).mean()), "frac_le6": float(np.mean(np.abs(data) <= 6.0))}


def _run_arm(arm, do_replay, scramble, bdsp_wmax, seed, referents, env, K, n_in, hidden, settle, epochs, batch,
             eprop_lr, w_clip, n_draws, milestones, test_n, replay_epochs, replay_per_fact, replay_noise, chance):
    """Teach the referents SEQUENTIALLY into ONE brain (thin adaptation of the sleep-replay runner's _run_arm,
    parametrized by the clamp). For replay arms, after teaching each fact run the offline self-replay
    consolidation over the hippocampus so far (teacher + world ABSENT)."""
    net = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip, bdsp_wmax=bdsp_wmax)
    _fit_readout_norm_world(net, env, referents, seed)
    teach_rng = np.random.default_rng(seed + 777)
    brain_rng = np.random.default_rng(seed + 313)
    hippo = Hippocampus(seed, replay_noise=replay_noise)
    ff0 = _ff_scale(net)
    acquire_acc, retention = [], {}
    for i, r in enumerate(referents):
        X, y = _corrective_batch(env, r, i, n_draws)            # WAKE: teacher draws from the world (legitimate)
        _teach_fact(net, X, y, epochs, batch, teach_rng)
        acq = _fact_acc(net, env, r, i, n=test_n)               # immediate acquisition (teeth c), before replay
        acquire_acc.append(acq)
        hippo.encode(X, i)                                      # hippocampus captures the engram of this episode
        if do_replay:                                          # SLEEP: self-replay (env absent); noreplay SKIPS
            _self_replay_consolidate(net, hippo, replay_epochs, batch, brain_rng, replay_per_fact, scramble=scramble)
        N = i + 1
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {"frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                                 "mean_retained_acc": float(np.mean(accs)),
                                 "per_fact_acc": [float(a) for a in accs]}
    return {"arm": arm, "bdsp_wmax": bdsp_wmax, "do_replay": do_replay, "scramble": scramble,
            "acquire_acc_immediate": [float(a) for a in acquire_acc],
            "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
            "ff_scale_build": ff0, "ff_scale_final": _ff_scale(net), "retention_curve": retention}


def run(seed, n_max, milestones, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p, noise,
        test_n, replay_epochs, replay_per_fact, replay_noise, declamp_wmax):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))
    env = ReferentEnv(seed, d_p=d_p, noise=noise)
    referents = [f"ref{i}" for i in range(n_max)]
    for r in referents:
        env.proto(r)

    #        arm name             do_replay scramble  bdsp_wmax (None => +-6 default clamp)
    specs = [("noreplay_clamped",   False,   False,   None),
             ("replay_clamped",     True,    False,   None),
             ("noreplay_declamped", False,   False,   declamp_wmax),
             ("replay_declamped",   True,    False,   declamp_wmax),
             ("scramble_declamped", True,    True,    declamp_wmax)]
    arms = {}
    for name, do_replay, scramble, bw in specs:
        t0 = time.time()
        env.rng = np.random.default_rng(seed + 101)            # same teaching percepts across arms (like-for-like)
        arms[name] = _run_arm(name, do_replay, scramble, bw, seed, referents, env, K, n_in, hidden, settle,
                              epochs, batch, eprop_lr, w_clip, n_draws, milestones, test_n, replay_epochs,
                              replay_per_fact, replay_noise, chance)
        arms[name]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[name]["retention_curve"]
        big = max((int(k) for k in rc), default=None)
        fr = rc[str(big)]["frac_recalled"] if big else float("nan")
        print(f"[arm {name:18s}] {arms[name]['wall_seconds']:5.0f}s | immediate-acq "
              f"{arms[name]['mean_acquire_acc_immediate']:.3f} | FF|w|mean "
              f"{arms[name]['ff_scale_final']['absmean']:7.1f} | frac-recalled@N={big}: {fr:.2f}", flush=True)
    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "cap_baseline": CAP_BASELINE,
            "config": {"hidden": hidden, "settle_steps": settle, "epochs": epochs, "batch": batch,
                       "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p, "noise": noise,
                       "test_n": test_n, "replay_epochs": replay_epochs, "replay_per_fact": replay_per_fact,
                       "replay_noise": replay_noise, "declamp_wmax": declamp_wmax},
            "arms": arms}


def _verdict(result):
    from tools.lab import lever, attributable_to
    from tools.verdict import Verdict
    cap = result["cap_baseline"]
    rc = {a: result["arms"][a]["retention_curve"] for a in result["arms"]}
    big = max((int(k) for k in rc["replay_declamped"]), default=None)
    key = str(big)
    f = {a: rc[a][key]["frac_recalled"] for a in rc}
    acq_declamp = result["arms"]["replay_declamped"]["mean_acquire_acc_immediate"]
    chance = result["chance"]

    # the manipulation MOVED (the clamp differs across the arms compared) -- else the A/B is void.
    lever("bdsp_wmax (clamped vs declamped)", "pm6", str(result["config"]["declamp_wmax"]))

    # honest decomposition of the declamped-replay retention:
    attributable_to("de-clamp (replay: declamped vs clamped)", f["replay_declamped"], f["replay_clamped"])
    attributable_to("replay content on de-clamp (replay vs scramble, declamped)",
                    f["replay_declamped"], f["scramble_declamped"])
    attributable_to("replay-vs-noreplay on de-clamp", f["replay_declamped"], f["noreplay_declamped"])
    attributable_to("clamp alone on forgetting (noreplay: declamped vs clamped)",
                    f["noreplay_declamped"], f["noreplay_clamped"])

    v = Verdict("teacher-loop sleep-replay + bdsp de-clamp", chance=chance)
    v.reaches("(a1) de-clamped replay beats clamped replay", before=f["replay_clamped"],
              after=f["replay_declamped"])
    v.floor(f"(a2) de-clamped replay beats the named ~{cap:.2f} cap", f["replay_declamped"], floor=cap)
    v.reaches("(b) the CLAMP is load-bearing on forgetting (noreplay: declamped>clamped)",
              before=f["noreplay_clamped"], after=f["noreplay_declamped"])
    v.floor("(c) immediate acquisition stays perfect (replay_declamped)", acq_declamp, floor=0.9)
    go = (f["replay_declamped"] > cap and f["replay_declamped"] > f["replay_clamped"]
          and f["noreplay_declamped"] > f["noreplay_clamped"] and acq_declamp >= 0.9)
    decision = v.decide(go=go)
    return {"largest_N": big, "cap_baseline": cap, "frac_recalled": f, "replay_declamped_acq": acq_declamp,
            "declamp_gain_replay": float(f["replay_declamped"] - f["replay_clamped"]),
            "clamp_forgetting_gain_noreplay": float(f["noreplay_declamped"] - f["noreplay_clamped"]),
            "replay_content_margin_declamped": float(f["replay_declamped"] - f["scramble_declamped"]),
            "replay_over_noreplay_declamped": float(f["replay_declamped"] - f["noreplay_declamped"]),
            **decision}


def _aggregate(paths):
    """6-seed roll-up. GO = every seed's replay_declamped frac_recalled@N=10 > cap AND > its own replay_clamped
    AND noreplay_declamped > noreplay_clamped (the clamp is load-bearing on every seed)."""
    arms = ("noreplay_clamped", "replay_clamped", "noreplay_declamped", "replay_declamped", "scramble_declamped")
    rows, cap = [], CAP_BASELINE
    for p in paths:
        d = json.loads(Path(p).read_text())
        f = d["verdict"]["frac_recalled"]
        rows.append({"seed": d["seed"], "N": d["verdict"]["largest_N"], **{a: f[a] for a in arms},
                     "acq": d["verdict"]["replay_declamped_acq"],
                     "seed_go": bool(f["replay_declamped"] > cap and f["replay_declamped"] > f["replay_clamped"]
                                     and f["noreplay_declamped"] > f["noreplay_clamped"])})
    import numpy as _np
    means = {a: float(_np.mean([r[a] for r in rows])) for a in arms}
    n_go = sum(r["seed_go"] for r in rows)
    go = n_go == len(rows) and len(rows) >= 6
    print("\n" + "=" * 100)
    print(f"[AGG] {len(rows)} seeds | cap {cap:.2f} | GO needs replay_declamped>cap & >clamped & clamp-load-bearing, all seeds")
    print(f"{'seed':>5} " + " ".join(f"{a[:9]:>10}" for a in arms) + f" {'acq':>6} {'GO':>4}")
    for r in sorted(rows, key=lambda x: x["seed"]):
        print(f"{r['seed']:>5} " + " ".join(f"{r[a]:>10.2f}" for a in arms) + f" {r['acq']:>6.3f} {str(r['seed_go']):>4}")
    print(f"{'mean':>5} " + " ".join(f"{means[a]:>10.2f}" for a in arms))
    print(f"[AGG] replay_declamped mean {means['replay_declamped']:.3f} vs replay_clamped mean "
          f"{means['replay_clamped']:.3f} vs cap {cap:.2f} | seeds GO {n_go}/{len(rows)} | VERDICT "
          f"{'GO' if go else 'NO-GO'}")
    print("=" * 100)
    return 0 if go else 1


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop sleep-replay + bdsp_wmax de-clamp: push replay-based "
                                             "continual retention past the ~0.55 self-replay cap by removing a "
                                             "silent substrate weight clamp.")
    ap.add_argument("--aggregate", nargs="+", default=None, help="per-seed JSONs -> 6-seed GO roll-up")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--milestones", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--replay-epochs", type=int, default=24)
    ap.add_argument("--replay-per-fact", type=int, default=16)
    ap.add_argument("--replay-noise", type=float, default=0.10)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=32)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--declamp-wmax", type=float, default=1e9, help="bdsp_w_max for the de-clamped arms (no-op clip)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.aggregate:
        return _aggregate(a.aggregate)
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    result = run(a.seed, a.n_max, a.milestones, a.hidden, a.settle_steps, a.epochs, a.batch, a.eprop_lr,
                 a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.replay_epochs, a.replay_per_fact,
                 a.replay_noise, a.declamp_wmax)
    verdict = _verdict(result)
    summary = {"probe": "teacher_loop_sleep_replay_declamp", "seed": a.seed,
               "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": True,
               "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))

    f = verdict["frac_recalled"]
    print("\n" + "=" * 100, flush=True)
    print(f"[declamp] seed {a.seed} @ N={verdict['largest_N']} (cap {verdict['cap_baseline']:.2f}, "
          f"chance {result['chance']:.2f}):", flush=True)
    for arm in ("noreplay_clamped", "replay_clamped", "noreplay_declamped", "replay_declamped",
                "scramble_declamped"):
        print(f"    {arm:18s}: frac-recalled {f[arm]:.2f}", flush=True)
    print(f"[declamp] de-clamp gain on REPLAY {verdict['declamp_gain_replay']:+.2f} | clamp forgetting gain "
          f"(noreplay) {verdict['clamp_forgetting_gain_noreplay']:+.2f} | replay content margin "
          f"{verdict['replay_content_margin_declamped']:+.2f} | replay-over-noreplay(declamp) "
          f"{verdict['replay_over_noreplay_declamped']:+.2f} | VERDICT {verdict['status']}", flush=True)
    print(f"[declamp] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if verdict["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
