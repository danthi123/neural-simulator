"""EMERGENCE-ENGINE + WORKING-MEMORY HYBRID — the SCALE de-risk (rung 3b -> scale).

BANKED GO (do NOT re-derive; read 2026-08-11-emergence-WM-hybrid-separate-channel-GO-working-hybrid-6seed.md):
  At TOY scale (n_subj=4, n_fill=8, n_cls=2, L=3, n_cells=32 -> vocab_sep=26, N=832 neurons) the SEPARATE-CHANNEL
  WM+HTM fusion WORKS 6-seed: hybrid held-out exact 0.974 [min 0.938] BEATS HTM-alone 0.224 and WM-alone 0.516 by
  +0.46, subject PRESERVED 1.000, class clean 0.974, every lesion load-bearing. The variable-binding WM's held-subject
  (its own wm[] column) and the HTM emergence engine's local-class (a subject-agnostic clsrd column) are combined by a
  LEARNED dendritic conjunction (verb(s,c) potentiates coincidence synapses from BOTH channels).

THE OPEN QUESTION (this de-risk): does that neural integration HOLD AT SCALE toward real-language sizes, and where
  (if anywhere) does it break first? We scale BOTH:
    * the TASK  : more subjects (n_subj), more classes (n_cls), a larger filler vocabulary (n_fill), longer/harder
                  dependency spans (L). The named risks: does the SUBJECT latch degrade with more subjects? does the
                  conjunction bridge SATURATE with more verb(s,c) columns? does the HTM class channel degrade with L?
    * the NET   : n_cells/column up so the substrate (N = vocab*n_cells neurons, a DENSE cross-column coincidence pool
                  of ~N*(N-n_cells) synapses) is large enough that the GPU genuinely helps.

WHAT THIS FILE IS — a thin ORCHESTRATOR + INSTRUMENT around the banked rung-3b engine (`_emerge_wm_hybrid_sepchan_derisk`),
  NOT a re-implementation. For each scale point it spawns the rung-3b runner in its OWN process with:
    * the point's scale knobs (--n-subj --n-fill --n-cls --distances --n-cells ...), and
    * a BACKEND chosen by substrate size (SIM_BACKEND=numpy below the measured crossover N; cupy above it) so cupy is
      used ONLY where it beats numpy. The backend MUST be fixed before `sim` imports, which is exactly why each point
      runs in a fresh subprocess (also gives clean per-point VRAM).
  It measures WALL-TIME + PEAK VRAM (attributed to the child pid via nvidia-smi) + THROUGHPUT per point, reads back the
  point's per-arm metrics + GO from the rung-3b summary JSON, and writes a combined scale-ladder summary. The GO GATE,
  the arms (HTM-alone / WM-alone / hybrid-sep / lesion-WM-channel / lesion-HTM-channel / lesion-the-hold /
  subject-shuffle / conj-untrained) and the honest-negative verdict are the rung-3b runner's own (unchanged).

MEASURED substrate crossover (coincidence_predict = 6 bridge steps over the dense O(N^2) pool; RTX 3090 vs 1 CPU core):
  N=832 numpy 17.6ms / cupy 9.6ms (1.8x) | N=2176 211 / 15.3 (14x) | N=4224 765 / 15.3 (50x) | N=6400 2062 / 27.5 (75x).
  => cupy wins the coincidence op from N~=1000; below that numpy wins the FULL run (host per-column loops + the kernel
  round-trip dominate and don't move to GPU). Default crossover threshold = 1200 neurons.

GO GATE (per scale point, the rung-3b bar, unchanged): hybrid held-out exact >= max(HTM-alone, WM-alone) + 0.20 AND
  subject preserved >= 0.90 AND class clean >= 0.90, with both channel lesions + the hold load-bearing and the untrained
  conjunction at chance. A precisely-characterised scale CEILING (where/why it first breaks) IS the deliverable.

NO sim/ edit. 6-seed (42 43 44 100 101 102) at the decisive point(s). Reuse-by-import / subprocess of the banked runner.

Run (single-seed ladder, foreground):
  python -m research.runners._emerge_wm_hybrid_scale_derisk --ladder toy medium large xlarge --seeds 42 \
    --out research/findings/raw/_emerge_wm_hybrid_scale/ladder_1seed.json
6-seed decisive point (foreground):
  python -m research.runners._emerge_wm_hybrid_scale_derisk --ladder medium --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/_emerge_wm_hybrid_scale/medium_6seed.json
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER_MOD = "research.runners._emerge_wm_hybrid_sepchan_derisk"
RAW_DIR = REPO / "research/findings/raw/_emerge_wm_hybrid_scale"

# ---- the scale ladder. Each point scales BOTH the task (subj/fill/cls/L) and the net (n_cells). vocab_sep and N are
#      derived; the dense coincidence pool is ~N*(N - n_cells) synapses. ----
LADDER = {
    #        n_subj n_fill n_cls  L   n_cells   (n_train, n_test scale with the task)
    "toy":    dict(n_subj=4,  n_fill=8,   n_cls=2, L=3, n_cells=32,  n_train=240, n_test=64),
    "medium": dict(n_subj=8,  n_fill=16,  n_cls=4, L=4, n_cells=32,  n_train=480, n_test=96),
    "large":  dict(n_subj=16, n_fill=32,  n_cls=4, L=5, n_cells=32,  n_train=640, n_test=120),
    "xlarge": dict(n_subj=16, n_fill=32,  n_cls=8, L=5, n_cells=32,  n_train=800, n_test=128),
    "huge":   dict(n_subj=32, n_fill=64,  n_cls=8, L=6, n_cells=32,  n_train=960, n_test=160),
    # a NET-only axis (fixed toy task, bigger columns) to isolate network-size scaling from task difficulty
    "net64":  dict(n_subj=4,  n_fill=8,   n_cls=2, L=3, n_cells=64,  n_train=240, n_test=64),
    "net128": dict(n_subj=4,  n_fill=8,   n_cls=2, L=3, n_cells=128, n_train=240, n_test=64),
    "net256": dict(n_subj=4,  n_fill=8,   n_cls=2, L=3, n_cells=256, n_train=240, n_test=64),
}


def vocab_sep(n_subj, n_fill, n_cls):
    # subj + fill + verb(n_subj*n_cls) + wm(n_subj) + clsrd(n_cls)
    return 2 * n_subj + n_fill + n_subj * n_cls + n_cls


def point_sizes(cfg):
    V = vocab_sep(cfg["n_subj"], cfg["n_fill"], cfg["n_cls"])
    N = V * cfg["n_cells"]
    nnz = N * (N - cfg["n_cells"])       # dense cross-column pool
    return V, N, nnz


def gpu_used_mb_for_pid(pid):
    """VRAM (MiB) attributed to `pid` (and its children share the same context so the pid matches)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=10).decode()
    except Exception:
        return 0.0
    tot = 0.0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
            try:
                tot += float(parts[1])
            except ValueError:
                pass
    return tot


def run_point(name, cfg, seeds, backend_mode, xover_n, epochs, extra):
    V, N, nnz = point_sizes(cfg)
    backend = ("cupy" if N >= xover_n else "numpy") if backend_mode == "auto" else backend_mode
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seed_tag = "s" + "_".join(str(s) for s in seeds) if len(seeds) <= 3 else f"{len(seeds)}seed"
    out_json = RAW_DIR / f"point_{name}_{backend}_{seed_tag}.json"
    cmd = [sys.executable, "-u", "-m", RUNNER_MOD,
           "--seeds", *[str(s) for s in seeds],
           "--n-subj", str(cfg["n_subj"]), "--n-fill", str(cfg["n_fill"]), "--n-cls", str(cfg["n_cls"]),
           "--distances", str(cfg["L"]), "--n-cells", str(cfg["n_cells"]),
           "--n-train", str(cfg["n_train"]), "--n-test", str(cfg["n_test"]),
           "--epochs", str(epochs), "--out", str(out_json)]
    cmd += list(extra)
    env = dict(os.environ)
    env["SIM_BACKEND"] = backend
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    print(f"\n=== POINT {name} | backend={backend} | n_subj={cfg['n_subj']} n_fill={cfg['n_fill']} n_cls={cfg['n_cls']} "
          f"L={cfg['L']} n_cells={cfg['n_cells']} | vocab_sep={V} N={N} pool_synapses={nnz:,} | seeds={seeds}",
          flush=True)
    print(f"    cmd: SIM_BACKEND={backend} {' '.join(cmd)}", flush=True)

    peak = {"vram": 0.0}
    stop = threading.Event()
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env)

    def sampler():
        while not stop.is_set():
            if backend == "cupy":
                v = gpu_used_mb_for_pid(proc.pid)
                if v > peak["vram"]:
                    peak["vram"] = v
            stop.wait(2.0)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.time()
    rc = proc.wait()
    wall = time.time() - t0
    stop.set(); th.join(timeout=3)

    summ = None
    if out_json.exists():
        try:
            summ = json.loads(out_json.read_text())
        except Exception as e:
            print(f"    !! could not read point summary: {e}", flush=True)
    gp = (summ or {}).get("go_point") or {}
    n_sent = (cfg["n_train"] + cfg["n_test"]) * max(1, len(seeds))
    row = {
        "name": name, "backend": backend, "returncode": rc, "wall_seconds": round(wall, 1),
        "peak_vram_mb": round(peak["vram"], 1), "vocab_sep": V, "N_neurons": N, "pool_synapses": nnz,
        "throughput_sent_per_s": round(n_sent / wall, 2) if wall > 0 else None,
        "config": cfg, "seeds": seeds, "point_json": str(out_json),
        "chance": gp.get("chance"), "n_verb": cfg["n_subj"] * cfg["n_cls"],
        "gen_defined": gp.get("gen_defined"),
        "hybrid_sep_exact": gp.get("hybrid_sep_exact"), "hybrid_sep_exact_min": gp.get("hybrid_sep_exact_min"),
        "hybrid_sep_subj": gp.get("hybrid_sep_subj"), "hybrid_sep_subj_min": gp.get("hybrid_sep_subj_min"),
        "hybrid_sep_cls": gp.get("hybrid_sep_cls"),
        "htm_exact": gp.get("htm_exact"), "htm_subj": gp.get("htm_subj"), "htm_cls": gp.get("htm_cls"),
        "wm_exact": gp.get("wm_exact"), "wm_subj": gp.get("wm_subj"), "wm_cls": gp.get("wm_cls"),
        "old_fusion_exact": gp.get("old_fusion_exact"), "old_fusion_subj": gp.get("old_fusion_subj"),
        "lesion_wm_chan_exact": gp.get("lesion_wm_chan_exact"),
        "lesion_htm_chan_exact": gp.get("lesion_htm_chan_exact"),
        "lesion_hold_exact": gp.get("lesion_hold_exact"),
        "subj_shuffle_exact": gp.get("subj_shuffle_exact"),
        "conj_untrained_exact": gp.get("conj_untrained_exact"),
        "slot_decode_acc": gp.get("slot_decode_acc"), "cls_chan_acc": gp.get("cls_chan_acc"),
        "hold_alive": gp.get("hold_alive"), "ngram_floor_exact": gp.get("ngram_floor_exact"),
        "verdict": (summ or {}).get("verdict"),
    }
    # per-point GO evaluation (the rung-3b bar). smoke (<6 seeds) reports SMOKE-GO but never GO.
    row["go"] = _eval_go(row, len(seeds) >= 6)
    _print_point(row)
    return row


def _eval_go(r, six_seed):
    def g(k):
        return r.get(k)
    need = [g("hybrid_sep_exact"), g("htm_exact"), g("wm_exact"), g("chance"), g("hybrid_sep_subj"),
            g("hybrid_sep_cls"), g("lesion_wm_chan_exact"), g("lesion_htm_chan_exact"), g("lesion_hold_exact"),
            g("subj_shuffle_exact"), g("conj_untrained_exact"), g("gen_defined")]
    if any(x is None for x in need):
        return {"status": "NO-DATA", "core": False}
    base = max(g("htm_exact"), g("wm_exact"))
    chance = g("chance")
    checks = {
        "gen_defined": bool(g("gen_defined")),
        "beats_both(+0.20)": g("hybrid_sep_exact") >= base + 0.20,
        "above_chance(+0.30)": g("hybrid_sep_exact") >= chance + 0.30,
        "subject_preserved(>=0.90)": g("hybrid_sep_subj") >= 0.90,
        "class_clean(>=0.90)": g("hybrid_sep_cls") >= 0.90,
        "wm_chan_load_bearing": g("hybrid_sep_exact") >= g("lesion_wm_chan_exact") + 0.20
                                 and g("lesion_wm_chan_exact") <= g("htm_exact") + 0.10,
        "htm_chan_load_bearing": g("hybrid_sep_exact") >= g("lesion_htm_chan_exact") + 0.20
                                  and g("lesion_htm_chan_exact") <= g("wm_exact") + 0.10,
        "hold_load_bearing": g("hybrid_sep_exact") >= g("lesion_hold_exact") + 0.20,
        "no_leak": g("subj_shuffle_exact") <= base + 0.05,
        "bind_learned": g("conj_untrained_exact") <= chance + 0.10,
    }
    core = all(checks.values())
    status = ("GO" if six_seed else "SMOKE-GO") if core else "NO-GO"
    return {"status": status, "core": core, "checks": checks,
            "failed": [k for k, v in checks.items() if not v]}


def _print_point(r):
    go = r["go"]
    print(f"    -> {go['status']} | wall {r['wall_seconds']}s vram {r['peak_vram_mb']}MB "
          f"thru {r['throughput_sent_per_s']} sent/s | N={r['N_neurons']} pool={r['pool_synapses']:,}", flush=True)
    if r.get("hybrid_sep_exact") is not None:
        print(f"       hybrid-sep {r['hybrid_sep_exact']}"
              f"{('[min '+str(r['hybrid_sep_exact_min'])+']') if r.get('hybrid_sep_exact_min') is not None else ''} "
              f"(subj {r['hybrid_sep_subj']} cls {r['hybrid_sep_cls']}) | HTM {r['htm_exact']} (subj {r['htm_subj']}) | "
              f"WM {r['wm_exact']} (cls {r['wm_cls']}) | chance {r['chance']} n_verb {r['n_verb']}", flush=True)
        print(f"       LES-wm {r['lesion_wm_chan_exact']} | LES-htm {r['lesion_htm_chan_exact']} | "
              f"LES-hold {r['lesion_hold_exact']} | subj-shuf {r['subj_shuffle_exact']} | "
              f"conj-untr {r['conj_untrained_exact']} | slot {r['slot_decode_acc']} cls-chan {r['cls_chan_acc']}",
              flush=True)
        if go.get("failed"):
            print(f"       FAILED: {go['failed']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", nargs="+", default=["toy"], choices=list(LADDER.keys()),
                    help="which scale points to run (in order)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--backend", choices=["auto", "numpy", "cupy"], default="auto")
    ap.add_argument("--xover-n", type=int, default=1200, help="N (neurons) at/above which auto picks cupy")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--out", default=str(RAW_DIR / "scale_summary.json"))
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra args forwarded verbatim to the rung-3b runner (after --extra)")
    a = ap.parse_args()

    t0 = time.time()
    rows = []
    for name in a.ladder:
        rows.append(run_point(name, LADDER[name], a.seeds, a.backend, a.xover_n, a.epochs, a.extra))

    # ---- crossover verdict: the largest point that HOLDS (SMOKE-GO/GO core) and the first that BREAKS ----
    held = [r for r in rows if r["go"].get("core")]
    broke = [r for r in rows if r["go"].get("status") == "NO-GO"]
    largest_held = max(held, key=lambda r: r["N_neurons"], default=None)
    first_break = min(broke, key=lambda r: r["N_neurons"], default=None)

    summary = {
        "probe": "emerge_wm_hybrid_scale", "seeds": a.seeds, "backend_mode": a.backend, "xover_n": a.xover_n,
        "n_points": len(rows), "elapsed_seconds": round(time.time() - t0, 1),
        "largest_held": (largest_held or {}).get("name"),
        "largest_held_N": (largest_held or {}).get("N_neurons"),
        "first_break": (first_break or {}).get("name"),
        "first_break_N": (first_break or {}).get("N_neurons"),
        "points": rows,
        "HONEST_NOTE": "Orchestrator over the banked rung-3b runner (_emerge_wm_hybrid_sepchan_derisk); NO sim/ edit. "
                       "Backend auto-selected by substrate size (numpy < xover_n neurons else cupy). The dense "
                       "cross-column coincidence pool is O(N^2) synapses built by a host double-loop -> RAM+build time "
                       "is the scaling wall well before VRAM. <6 seeds is a SMOKE indicator; 6 seeds is decisive.",
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 110, flush=True)
    print("[emerge_wm_hybrid_scale] SCALE LADDER (per-point):", flush=True)
    print(f"{'point':8} {'backend':6} {'N':>7} {'pool':>13} {'wall_s':>8} {'vram_mb':>8} "
          f"{'hyb':>6} {'subj':>6} {'cls':>6} {'HTM':>6} {'WM':>6} {'status':>8}", flush=True)
    for r in rows:
        print(f"{r['name']:8} {r['backend']:6} {r['N_neurons']:>7} {r['pool_synapses']:>13,} "
              f"{str(r['wall_seconds']):>8} {str(r['peak_vram_mb']):>8} "
              f"{str(r.get('hybrid_sep_exact')):>6} {str(r.get('hybrid_sep_subj')):>6} "
              f"{str(r.get('hybrid_sep_cls')):>6} {str(r.get('htm_exact')):>6} {str(r.get('wm_exact')):>6} "
              f"{r['go']['status']:>8}", flush=True)
    print(f"\nlargest HELD: {summary['largest_held']} (N={summary['largest_held_N']}) | "
          f"first BREAK: {summary['first_break']} (N={summary['first_break_N']})", flush=True)
    print(f"[emerge_wm_hybrid_scale] wrote {a.out}\n" + "=" * 110, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
