"""Kill-safe / pausable multi-seed runner for the owner-authorized
fair-scale dendritic credit-assignment gate (literal Lillicrap-2016 /
GLR-2017 feedback-alignment on MNIST).

WHAT THIS IS
------------
A thin INTEGRATION wrapper. It owns NO learning rule and NO verdict
logic: it imports the network (``sim.dendritic_mlp.DendriticMLP``), the
THREE-STATE verdict (``research.runners.dendritic_fair_core``), and the
atomic checkpoint core (``sim.train_checkpoint``) and wires them into a
five-condition, multi-seed, KILL-SAFE training loop.

KILL-SAFETY (a HARD owner requirement)
--------------------------------------
A week-scale run must lose <= 1 epoch on interrupt and resume cleanly
just by re-running the script. Every (seed, condition) leg writes an
ATOMIC per-epoch checkpoint via ``sim.train_checkpoint.save_checkpoint``
(``.tmp`` + ``os.replace``). On re-run, ``resume_epoch`` skips already-
completed epochs and the network weights + history are restored. A
``KeyboardInterrupt`` mid-epoch flushes a final checkpoint for the
in-flight epoch and exits cleanly (return 0) so the next run resumes
from exactly there. The per-epoch mini-batch shuffle is a PURE FUNCTION
of (seed, condition, epoch), so a resumed epoch sees byte-identical
batches.

HONEST CEILING
--------------
This gates a literature-scale CREDIT-ASSIGNMENT ROOT only (does a
biologically-local, no-weight-transport feedback-alignment rule learn
MNIST, with the wrong-sign / global-scalar / permuted controls failing
and emergent gradient alignment present). It is explicitly NOT priority
#3 developmental/embodiment work, NOT "conversation solved", and NOT
the conversational stack. Wiring a PASS into the conversational
architecture is a SEPARATE later effort and is the controller's
post-run job, not this runner's.

PURITY
------
Pure numpy + stdlib. NO deep-learning framework and NO automatic-
differentiation library is imported or referenced anywhere on the
shipped path; the forbidden-substring invariant is enforced by a smoke
assertion (transitively through ``sim.dendritic_mlp``) and by the
runtime ``_biologically_local`` self-check. Even this docstring avoids
spelling those library names so the source itself stays clean.
ASCII-only prints (Windows cp1252 safe).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np

from sim.dendritic_mlp import DendriticMLP
from sim.train_checkpoint import load_checkpoint, resume_epoch, save_checkpoint
from research.runners.dendritic_fair_core import (
    dfair_aggregate_multiseed,
    dfair_verdict,
)

# Five conditions. local_correct is the biologically-local rule under
# test; oracle is the hand-derived-backprop positive control; the other
# three are the negative controls the verdict requires to FAIL.
_CONDITIONS = ("oracle", "local_correct", "local_wrongsign",
               "global_scalar", "permuted")

# Deterministic per-leg shuffle salt + permuted-label salt. Plain
# integers spelled WITHOUT alphabetic hex digits so there is no risk of
# an invalid-literal foot-gun on later edits.
_SHUFFLE_SALT = 1599193869
_PERMLABEL_SALT = 2718281828

_HONEST_CEILING = (
    "Gates a LITERATURE-SCALE credit-assignment ROOT only "
    "(biologically-local feedback-alignment learns MNIST + controls "
    "fail + emergent alignment). NOT priority-#3 developmental/"
    "embodiment. NOT 'conversation solved'. Integration into the "
    "conversational stack is a SEPARATE later effort (controller's "
    "post-run job), not this runner's."
)

# Canonical TF/Keras MNIST mirror (x_train/y_train/x_test/y_test,
# uint8, 60000/10000). Authoritative + stable public Google Cloud
# Storage bucket served by tf.keras.datasets.mnist. Fetched ONCE,
# then cached + reused (never re-downloaded).
_MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def _wrap(text, width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    if cur:
        lines.append(cur)
    return lines


def _print_banner():
    print("=" * 70)
    print("DENDRITIC FAIR-SCALE CREDIT-ASSIGNMENT GATE (kill-safe runner)")
    print("=" * 70)
    print("HONEST CEILING:")
    for line in _wrap(_HONEST_CEILING, 66):
        print("  " + line)
    print("=" * 70)


def _atomic_write_json(path, obj):
    """Atomic JSON write (.tmp + os.replace) -- kill-safe output."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------
def _tiny_synth():
    """Deterministic synthetic 3-class gaussian-cluster problem.

    Fully seed-fixed (its own RNG, independent of the model seed) so the
    whole pipeline turns in well under the smoke timeout. The toy
    verdict is NOT propagated -- it only proves the pipeline turns;
    expect VOID at this scale (a tiny net cannot clear the oracle>=0.95
    instrument-validity bar), which is fine and intended.
    """
    rng = np.random.default_rng(20260517)
    n_per = 100  # 300 total -> 200 train / 100 heldout after the split
    centers = np.array([[2.0, 0.0], [-1.0, 1.7], [-1.0, -1.7]])
    dim = 24
    proj = rng.normal(0, 1.0, (2, dim))
    Xs, ys = [], []
    for c in range(3):
        pts2 = centers[c] + 0.6 * rng.normal(0, 1.0, (n_per, 2))
        Xs.append(pts2 @ proj + 0.05 * rng.normal(0, 1.0, (n_per, dim)))
        ys.append(np.full(n_per, c, dtype=np.int64))
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    X = X - X.mean(0, keepdims=True)  # mirrors MNIST mean-centering
    cut = 200
    return (X[:cut], y[:cut], X[cut:], y[cut:],
            [dim, 16, 16, 3], 2,
            {"source": "tiny_synth(deterministic)",
             "sha_or_shape_ok": True})


def _finish_mnist(npz, source):
    """Normalize an MNIST .npz into the runner's contract or None."""
    keys = set(npz.files)
    if {"x_train", "y_train", "x_test", "y_test"} <= keys:
        xtr, ytr = npz["x_train"], npz["y_train"]
        xte, yte = npz["x_test"], npz["y_test"]
    elif {"X_train", "y_train", "X_test", "y_test"} <= keys:
        xtr, ytr = npz["X_train"], npz["y_train"]
        xte, yte = npz["X_test"], npz["y_test"]
    else:
        return None
    xtr = xtr.reshape(len(xtr), -1).astype(np.float32) / 255.0
    xte = xte.reshape(len(xte), -1).astype(np.float32) / 255.0
    mu = xtr.mean(0, keepdims=True)  # mean-center on train statistics
    xtr = xtr - mu
    xte = xte - mu
    ytr = np.asarray(ytr).astype(np.int64).reshape(-1)
    yte = np.asarray(yte).astype(np.int64).reshape(-1)
    if xtr.shape[1] != 784 or len(ytr) < 60000:
        return None
    prov = {"source": source, "sha_or_shape_ok": True,
            "n_train": int(len(ytr)), "n_test": int(len(yte))}
    return (xtr, ytr, xte, yte, [784, 512, 256, 128, 10], None, prov)


def _load_mnist(cache, allow_download=True):
    """Idempotent MNIST load (mirrors corpus_fetch cache discipline).

    Returns the dataset tuple on success, or None if the cache is
    absent AND a download could not be obtained (offline) -- the caller
    then prints NOT RUNNABLE and returns 2.

    A cache hit is NEVER re-downloaded. A fresh download streams to a
    ``.tmp``, is VALIDATED, then atomically renamed into the cache.
    Preprocessing: X -> float32 in [0,1] (/255) then mean-centered;
    y -> int.
    """
    # 1. Idempotent cache hit -- never re-download.
    if os.path.exists(cache):
        try:
            with np.load(cache, allow_pickle=False) as npz:
                res = _finish_mnist(npz, "cache:" + cache)
            if res is not None:
                print("[dendritic_fair_gate] MNIST cache hit: %s "
                      "(%d train) -- no download" % (cache, len(res[1])))
                return res
            print("[dendritic_fair_gate] MNIST cache present but "
                  "shape/keys invalid: %s" % cache)
        except (OSError, ValueError) as exc:
            print("[dendritic_fair_gate] MNIST cache unreadable (%s)"
                  % exc)

    if not allow_download:
        return None

    # 2. Fetch ONCE -> .tmp -> validate -> atomic rename into cache.
    print("[dendritic_fair_gate] downloading MNIST from %s" % _MNIST_URL)
    d = os.path.dirname(cache)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = cache + ".tmp"
    try:
        req = urllib.request.Request(
            _MNIST_URL,
            headers={"User-Agent": "neural-simulator-dfair/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(tmp, "wb") as fh:
            fh.write(data)
        with np.load(tmp, allow_pickle=False) as npz:  # validate .tmp
            res = _finish_mnist(npz, _MNIST_URL)
        if res is None:
            print("[dendritic_fair_gate] downloaded MNIST failed "
                  "shape/keys validation")
            _safe_remove(tmp)
            return None
        os.replace(tmp, cache)  # atomic promote
        print("[dendritic_fair_gate] cached MNIST -> %s (%d train)"
              % (cache, len(res[1])))
        with np.load(cache, allow_pickle=False) as npz:
            return _finish_mnist(npz, _MNIST_URL)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError,
            ValueError, TimeoutError) as exc:
        print("[dendritic_fair_gate] MNIST download failed (%s)" % exc)
        _safe_remove(tmp)
        return None


def _safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------------
# Training (kill-safe, resume-stable)
# ---------------------------------------------------------------------
def _epoch_shuffle(seed, condition, epoch, n):
    """Resume-stable mini-batch order.

    A PURE function of (seed, condition, epoch): a resumed epoch sees
    byte-identical batches because the permutation RNG is reseeded from
    these scalars only -- never from a process-global generator.
    """
    cond_id = _CONDITIONS.index(condition)
    rng = np.random.default_rng(
        [int(seed), int(cond_id), int(epoch), _SHUFFLE_SALT])
    return rng.permutation(n)


def _restore(net, ckpt):
    """Restore W + B from a checkpoint (packed as net.W + net.B)."""
    w = ckpt["weights"]
    nW = len(net.W)
    net.W = [np.asarray(a, dtype=float) for a in w[:nW]]
    net.B = [np.asarray(a, dtype=float) for a in w[nW:]]


def _pack_history(history):
    """Flatten [[acc, align], ...] -> [acc, align, acc, align, ...]."""
    return [float(v) for row in history for v in row]


def _train_leg(seed, condition, Xtr, ytr, Xte, yte, sizes, epochs,
               lr, ckpt_path, batch=128):
    """Train one (seed, condition) leg with per-epoch atomic checkpoint.

    Returns (final_heldout_acc, end_align_cos, net). ``end_align_cos``
    is meaningful only for ``local_correct`` (mean of the last ~20% of
    the measured layer-0 alignment history); 0.0 otherwise.

    KeyboardInterrupt mid-epoch -> flush a checkpoint marking the LAST
    COMPLETED epoch (an in-flight epoch's partial weights are discarded
    so resume re-runs that epoch from its deterministic start = lose
    <= 1 epoch), print [INTERRUPTED], and re-raise so main() exits
    cleanly with return 0.
    """
    net = DendriticMLP(sizes, seed=seed)

    # For 'permuted': a FIXED per-seed label permutation applied to the
    # TRAIN labels ONLY. Held-out labels stay unpermuted, so a
    # permuted-trained net cannot generalize -- the 2026-05-03
    # permuted-label catcher.
    ytr_used = ytr
    if condition == "permuted":
        pr = np.random.default_rng([int(seed), _PERMLABEL_SALT])
        ytr_used = pr.permutation(ytr)

    ckpt = load_checkpoint(ckpt_path)
    start = resume_epoch(ckpt)
    # history[i] = [heldout_acc, align_cos] for completed epoch i.
    history = []
    if ckpt is not None:
        _restore(net, ckpt)
        flat = ckpt["loss_history"]
        history = [[float(flat[i]), float(flat[i + 1])]
                   for i in range(0, len(flat) - 1, 2)]
        if start >= epochs:
            print("[dendritic_fair_gate] seed=%d cond=%s already "
                  "complete (%d epochs) -- skipping" %
                  (seed, condition, epochs))

    n = len(ytr_used)
    meas_n = min(256, n)  # fixed measurement slice for alignment

    def _flush(epoch_completed):
        save_checkpoint(
            ckpt_path, epoch_completed,
            weights=list(net.W) + list(net.B),
            rng_state={"seed": int(seed), "condition": condition},
            loss_history=_pack_history(history))

    last_completed = start - 1
    try:
        for ep in range(start, epochs):
            order = _epoch_shuffle(seed, condition, ep, n)
            for bi in range(0, n, batch):
                idx = order[bi:bi + batch]
                net.train_step(Xtr[idx], ytr_used[idx],
                               mode=condition, lr=lr)
            acc = net.accuracy(Xte, yte)
            if condition == "local_correct":
                align = float(net.hidden_grad_alignment(
                    Xtr[:meas_n], ytr_used[:meas_n]))
            else:
                align = 0.0
            history.append([float(acc), align])
            _flush(ep)  # atomic per-epoch checkpoint (kill-safe)
            last_completed = ep
    except KeyboardInterrupt:
        # Re-snapshot the LAST COMPLETED epoch. The in-flight epoch's
        # partial weights are intentionally NOT persisted: resume re-runs
        # that single epoch from its deterministic batch order, so at
        # most one epoch of compute is lost and state stays consistent.
        try:
            if history:
                _flush(last_completed)
            else:
                # Nothing completed this run; leave any prior checkpoint
                # untouched (re-running resumes from it / from scratch).
                pass
        except Exception:  # never let the flush mask the interrupt
            pass
        print("[INTERRUPTED] checkpointed seed=%d cond=%s epoch=%d; "
              "re-run to resume" %
              (seed, condition, max(last_completed, 0)))
        raise

    if history:
        final_acc = float(history[-1][0])
        if condition == "local_correct":
            tail = max(1, len(history) // 5)  # last ~20%
            end_align = float(np.mean([h[1] for h in history[-tail:]]))
        else:
            end_align = 0.0
    else:
        # Fully resumed with an empty history -> re-evaluate directly.
        final_acc = float(net.accuracy(Xte, yte))
        end_align = 0.0
    return final_acc, end_align, net


# ---------------------------------------------------------------------
# Verdict plumbing
# ---------------------------------------------------------------------
def _biologically_local(nets_by_cond, sizes):
    """Runtime self-check for the no-weight-transport invariant.

    True ONLY if, for every condition's trained net, every fixed
    feedback matrix B[i] is BYTE-IDENTICAL to a freshly regenerated
    DendriticMLP(sizes, seed).B[i] (B never learned, never derived from
    any forward W), AND neither this module nor sim.dendritic_mlp
    references a deep-learning / auto-differentiation library on the
    shipped path. The two forbidden tokens are assembled at runtime so
    the literal substrings never appear in this file's source.
    """
    import sim.dendritic_mlp as _dmm
    _forbidden = ("tor" + "ch", "auto" + "grad")
    for src in (inspect.getsource(_dmm),
                inspect.getsource(sys.modules[__name__])):
        if any(tok in src for tok in _forbidden):
            return False
    for seed, net in nets_by_cond:
        ref = DendriticMLP(sizes, seed=seed)
        if len(ref.B) != len(net.B):
            return False
        for bi in range(len(ref.B)):
            if not np.array_equal(ref.B[bi], net.B[bi]):
                return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Kill-safe dendritic fair-scale credit-assignment "
                    "gate (5 conditions, multi-seed, THREE-STATE).")
    ap.add_argument("--seeds", default="42,43,44",
                    help="CSV seed list (>= 3 MANDATORY).")
    ap.add_argument("--tiny-synth", action="store_true",
                    help="Tiny synthetic problem (pipeline smoke).")
    ap.add_argument("--epochs", type=int, default=60,
                    help="Epochs per (seed,condition) for the real run.")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument(
        "--out",
        default=os.path.join("research", "findings", "raw", "g11_bg",
                             "dendritic_fair_gate.json"))
    ap.add_argument(
        "--ckpt-dir",
        default=os.path.join("research", "findings", "raw", "g11_bg",
                             "dendritic_fair_ckpt"),
        help="Dir for per-(seed,condition) atomic checkpoints.")
    ap.add_argument("--mnist-cache",
                    default=os.path.join("data", "mnist.npz"))
    args = ap.parse_args(argv)

    _print_banner()

    try:
        seeds = [int(s) for s in str(args.seeds).split(",")
                 if s.strip() != ""]
    except ValueError:
        print("[NOT RUNNABLE] --seeds must be a CSV of integers")
        return 2

    if len(seeds) < 3:
        print("[NOT RUNNABLE] >= 3 seeds MANDATORY")
        return 2

    # Dataset.
    if args.tiny_synth:
        Xtr, ytr, Xte, yte, sizes, ep_override, prov = _tiny_synth()
        epochs = 2 if ep_override is None else ep_override
        print("[dendritic_fair_gate] tiny-synth: %d train / %d heldout, "
              "net=%s, epochs=%d" % (len(ytr), len(yte), sizes, epochs))
    else:
        loaded = _load_mnist(args.mnist_cache, allow_download=True)
        if loaded is None:
            print("[NOT RUNNABLE] MNIST cache absent and download "
                  "failed (offline?)")
            return 2
        Xtr, ytr, Xte, yte, sizes, _ep, prov = loaded
        epochs = int(args.epochs)
        print("[dendritic_fair_gate] MNIST: %d train / %d heldout, "
              "net=%s, epochs=%d" % (len(ytr), len(yte), sizes, epochs))

    os.makedirs(args.ckpt_dir, exist_ok=True)

    per_seed = []
    try:
        for seed in seeds:
            print("-" * 70)
            print("SEED %d" % seed)
            heldout = {}
            end_align_cos = 0.0
            nets_for_seed = []
            for cond in _CONDITIONS:
                ckpt_path = os.path.join(
                    args.ckpt_dir, "s%d_%s.npz" % (seed, cond))
                acc, align, net = _train_leg(
                    seed, cond, Xtr, ytr, Xte, yte, sizes, epochs,
                    args.lr, ckpt_path)
                heldout[cond] = float(acc)
                nets_for_seed.append((seed, net))
                if cond == "local_correct":
                    end_align_cos = float(align)
                print("  %-14s heldout=%.4f%s" %
                      (cond, acc,
                       (" align=%.4f" % align)
                       if cond == "local_correct" else ""))

            bio_local = _biologically_local(nets_for_seed, sizes)
            v = dfair_verdict(
                oracle_heldout=heldout["oracle"],
                correct_heldout=heldout["local_correct"],
                wrongsign_heldout=heldout["local_wrongsign"],
                globalscalar_heldout=heldout["global_scalar"],
                permuted_heldout=heldout["permuted"],
                end_align_cos=end_align_cos,
                biologically_local=bio_local,
                has_controls=True)
            # align_history: the local_correct alignment trajectory
            # (every 2nd packed float, offset 1).
            ck = load_checkpoint(os.path.join(
                args.ckpt_dir, "s%d_local_correct.npz" % seed))
            align_history = []
            if ck is not None:
                fl = ck["loss_history"]
                align_history = [float(fl[i])
                                 for i in range(1, len(fl), 2)]
            per_seed.append({
                "seed": seed,
                "heldout": heldout,
                "align_history": align_history,
                "biologically_local": bool(bio_local),
                "verdict": v,
            })
            print("  -> seed verdict: %s" % v["GATE"])
    except KeyboardInterrupt:
        # _train_leg already flushed the in-flight leg's checkpoint and
        # printed [INTERRUPTED]. Exit cleanly so re-running resumes
        # from exactly the last completed epoch (lose <= 1 epoch).
        print("[dendritic_fair_gate] clean interrupt -- checkpoints "
              "intact; re-run to resume.")
        return 0

    agg = dfair_aggregate_multiseed(per_seed)

    out_obj = {
        "task": "dendritic_fair_gate",
        "n_seeds": len(seeds),
        "seeds": seeds,
        "epochs": epochs,
        "lr": float(args.lr),
        "tiny_synth": bool(args.tiny_synth),
        "mnist_provenance": prov,
        "honest_ceiling": _HONEST_CEILING,
        "per_seed": per_seed,
        "aggregate_verdict": agg,
        "GATE": agg["GATE"],
    }
    _atomic_write_json(args.out, out_obj)

    print("=" * 70)
    print("VERDICT: %s" % agg["GATE"])
    print("  (VOID = instrument not sound; PASS/FAIL = science result)")
    if args.tiny_synth:
        print("  tiny-synth: VOID expected (toy scale cannot clear the "
              "oracle>=0.95 instrument-validity bar) -- the toy verdict "
              "is NOT propagated; it only proves the pipeline turns.")
    print("HONEST CEILING REMINDER:")
    for line in _wrap(_HONEST_CEILING, 66):
        print("  " + line)
    print("Honest propagation of a PASS into the conversational stack "
          "is the controller's post-run job, NOT this runner's.")
    print("Wrote %s" % args.out)
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
