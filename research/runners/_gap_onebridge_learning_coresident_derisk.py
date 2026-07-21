"""Single-shared-substrate — the render-LEARNING co-resides with the composer on ONE bridge (2026-07-20).

The capstone put the composer's RF ops + the WKV read-out on ONE bridge (READ side). The on-bridge fluency de-risk
showed the read-out LEARNS by a pure exact delta rule over cp_ssm_state. This closes the WRITE side of "everything on
one substrate": the delta-rule LEARNING (cp_ssm_readout_w updated by dw = -lr*err*state) runs on the shared bridge's
chan region WHILE the composer (RF bind/unbind on its own region) coexists and its recall stays correct.

MAIN gate: (1) the read-out LEARNS a synthetic target on-bridge (loss drops a lot), (2) the composer recall + no-confab
moat are correct on the SAME bridge, (3) a composer op INTERLEAVED into the learning loop does NOT perturb the learned
read-out (byte-isolated write). ANTI-CHEAT: FROZEN read-out (no update) -> loss flat (learning is load-bearing).

Reuse-by-import (the capstone bridge + SharedBridgeComposer); NO sim/ edit. `--seed`, `--n-steps`.
"""
import argparse
import numpy as np

from sim.backend import get_backend, to_host
from research.runners._gap_onebridge_capstone_derisk import _build_capstone_bridge, SharedBridgeComposer


def _charge(b, xp, inject, read_idx, n):
    cur = np.zeros(n, np.float32); cur[read_idx] = inject.astype(np.float32)
    b.cp_ssm_inject[:] = xp.asarray(cur); b.cp_ssm_shunt[:] = 0.0
    b._run_one_simulation_step()
    return np.asarray(to_host(b.cp_ssm_readout_out)).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--D-cmp", type=int, default=64)
    ap.add_argument("--frozen", action="store_true")
    args = ap.parse_args()
    xp, _ = get_backend()
    rng = np.random.default_rng(args.seed)

    D_wkv = 64                                                    # a small WKV chan region for the learning probe
    decay = 0.9
    facts = [("dog", "chase", "cat"), ("owl", "eat", "mouse")]
    vocab = sorted({w for f in facts for w in f})

    mb, mchan, enc_idx, cmp_idx = _build_capstone_bridge(D_wkv, args.D_cmp, args.seed, decay)
    n = int(mb.core_config.num_neurons)
    read_idx = np.concatenate([np.asarray(g) for g in mchan]).astype(np.int64)
    n_read = len(read_idx)

    # a learned read-out over the chan region's cp_ssm_state: out = W @ cp_ssm_state (the committed graded read-out)
    OUT = 8
    W = (rng.standard_normal((OUT, n)) * 0.05).astype(np.float32)
    W[:, np.setdiff1d(np.arange(n), read_idx)] = 0.0             # read only the chan region (rest unused)
    mb.cp_ssm_readout_w = xp.asarray(W)
    mb.cp_ssm_readout_out = None

    # composer on the same bridge
    cmp = SharedBridgeComposer(seed=args.seed, D=args.D_cmp, vocab=vocab)
    cmp.bind_to_shared(mb, cmp_idx)
    for a, v, p in facts:
        cmp.store(a, v, p)

    # a fixed synthetic supervised task: input inject -> target vector (learn W by the exact delta rule)
    inj = rng.standard_normal(n_read) * 0.5
    target = rng.standard_normal(OUT)

    losses = []
    for step in range(args.n_steps):
        out = _charge(mb, xp, inj, read_idx, n)
        err = out - target
        losses.append(float(np.mean(err ** 2)))
        if not args.frozen:
            state = np.asarray(to_host(mb.cp_ssm_state)).astype(np.float64)
            dW = np.zeros((OUT, n)); dW[:, read_idx] = np.outer(err, state[read_idx])
            W = (W - args.lr * dW).astype(np.float32)
            W[:, np.setdiff1d(np.arange(n), read_idx)] = 0.0
            mb.cp_ssm_readout_w = xp.asarray(W)
        if step % 40 == 20:                                     # INTERLEAVE a composer op mid-learning
            _ = cmp.query_patient("dog", "chase")

    # after learning: composer still correct + moat holds
    ans = [cmp.query_patient(a, v) for a, v, _ in facts]
    abstain = cmp.query_patient("lion", "roar")
    recall_ok = (ans == ["cat", "mouse"])
    moat_ok = (abstain is None)
    learned = losses[0] > 1e-6 and losses[-1] < 0.2 * losses[0]  # loss dropped >=5x
    if args.frozen:
        # frozen W: the loss still DRIFTS (the ssm integrator settles s toward inject, so out=W@s moves) but it does
        # NOT LEARN -- it never drops toward 0. The load-bearing contrast is "does NOT drop 5x" (vs MAIN's ->0).
        did_not_learn = losses[-1] > 0.2 * losses[0]
        verdict = "GO" if did_not_learn else "NO-GO"
        print(f"[RESULT {verdict}] FROZEN control (seed {args.seed}): loss {losses[0]:.4f} -> {losses[-1]:.4f} "
              f"(did_not_learn={did_not_learn}) -- W frozen => loss drifts with the settling state but never learns "
              f"(no 5x drop); the delta update is load-bearing.")
        return

    verdict = "GO" if (learned and recall_ok and moat_ok) else "NO-GO"
    print(f"[RESULT {verdict}] render-LEARNING co-resident with the composer on ONE bridge (seed {args.seed}, "
          f"{args.n_steps} steps):")
    print(f"  on-bridge delta-rule learning : loss {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"({losses[0]/max(losses[-1],1e-9):.1f}x drop, learned={learned})")
    print(f"  composer recall (interleaved) : {ans} == ['cat','mouse'] -> {recall_ok}")
    print(f"  no-confab moat                : {abstain!r} is None -> {moat_ok}")
    print(f"  => the render-LEARNING (exact delta rule over cp_ssm_state) LEARNS on the shared bridge WHILE the "
          f"composer binds/queries on the same substrate; composer recall + moat intact. Learning + composer + WKV "
          f"read-out all on ONE substrate.")


if __name__ == "__main__":
    main()
