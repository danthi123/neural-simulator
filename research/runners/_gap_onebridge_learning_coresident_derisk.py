"""Single-shared-substrate — the render read-out LEARNING runs co-resident with the composer on ONE bridge (2026-07-20).

Scope (honest, post adversarial-audit): the composer's RF ops + the WKV read-out FORWARD (out = cp_ssm_readout_w @
cp_ssm_state, committed in the step loop) + the read-out's ELIGIBILITY (cp_ssm_state) are ON the shared bridge; the
delta-rule WEIGHT ARITHMETIC (dw = -lr*err*state) is host numpy (a pure delta rule, NOT yet a spiking local rule — the
on-bridge BDSP graded-clean-error is the follow-on). This de-risks that the delta-update WIRE-UP runs co-resident with
the composer, reading the ON-BRIDGE cp_ssm_state, and GENERALIZES (a teacher-student task with a HELD-OUT set — so it
is genuine learning of a MAP, not memorization of one fixed point), while the composer binds/queries on the same bridge.

MAIN gate: (1) the read-out learns a fixed random teacher T on-bridge and the HELD-OUT loss drops a lot (generalizes);
(2) INTERLEAVE non-interference — training WITH a composer op every 40 steps gives an IDENTICAL held-out loss to
training WITHOUT (the composer op does not perturb the learning); (3) the composer recall + moat stay correct.
ANTI-CHEAT: FROZEN read-out -> held-out loss does NOT drop (the delta update is load-bearing).

Reuse-by-import (the capstone bridge + SharedBridgeComposer); NO sim/ edit. `--seed`, `--epochs`, `--frozen`.
"""
import argparse
import numpy as np

from sim.backend import get_backend, to_host
from research.runners._gap_onebridge_capstone_derisk import _build_capstone_bridge, SharedBridgeComposer

OUT = 8


def _state_and_out(b, xp, inj, read_idx, n):
    """Wash-free single charge: set cp_ssm_inject, one step -> the on-bridge read-out out = W@cp_ssm_state + the state.
    Callers WASH between distinct inputs so each input's settled state is independent."""
    cur = np.zeros(n, np.float32); cur[read_idx] = inj.astype(np.float32)
    b.cp_ssm_state[:] = 0.0                                    # fresh state per input (independent settle)
    b.cp_ssm_inject[:] = xp.asarray(cur); b.cp_ssm_shunt[:] = 0.0
    b._run_one_simulation_step()
    out = np.asarray(to_host(b.cp_ssm_readout_out)).astype(np.float64)
    st = np.asarray(to_host(b.cp_ssm_state)).astype(np.float64)
    return out, st


def _run(seed, epochs, lr, D_cmp, frozen, interleave):
    xp, _ = get_backend(); rng = np.random.default_rng(seed)
    D_wkv = 16                                                # n_read = 2*D_wkv = 32; n_train >> n_read => generalizes
    facts = [("dog", "chase", "cat"), ("owl", "eat", "mouse")]
    vocab = sorted({w for f in facts for w in (f[0], f[1], f[2])})
    mb, mchan, enc_idx, cmp_idx = _build_capstone_bridge(D_wkv, D_cmp, seed, 0.9)
    n = int(mb.core_config.num_neurons)
    read_idx = np.concatenate([np.asarray(g) for g in mchan]).astype(np.int64)
    n_read = len(read_idx)
    off = np.setdiff1d(np.arange(n), read_idx)

    # fixed random TEACHER over the chan-region state -> the read-out must LEARN this map (teacher-student).
    # n_train (96) >> n_read (32) makes it OVER-determined: the read-out must recover the true T (cannot memorize
    # individual points), so a low HELD-OUT loss is genuine generalization, not memorization.
    T = (rng.standard_normal((OUT, n_read)) * 0.3)
    n_train, n_test = 96, 16
    train_inj = [rng.standard_normal(n_read) * 0.5 for _ in range(n_train)]
    test_inj = [rng.standard_normal(n_read) * 0.5 for _ in range(n_test)]

    W = (rng.standard_normal((OUT, n)) * 0.02).astype(np.float32); W[:, off] = 0.0
    mb.cp_ssm_readout_w = xp.asarray(W); mb.cp_ssm_readout_out = None

    cmp = SharedBridgeComposer(seed=seed, D=D_cmp, vocab=vocab); cmp.bind_to_shared(mb, cmp_idx)
    for a, v, p in facts:
        cmp.store(a, v, p)

    def heldout_loss():
        tot = 0.0
        for x in test_inj:
            out, st = _state_and_out(mb, xp, x, read_idx, n)
            tgt = T @ st[read_idx]
            tot += float(np.mean((out - tgt) ** 2))
        return tot / len(test_inj)

    l0 = heldout_loss()
    order = rng.permutation(n_train)
    step = 0
    for _ in range(epochs):
        for i in order:
            out, st = _state_and_out(mb, xp, train_inj[i], read_idx, n)
            tgt = T @ st[read_idx]
            err = out - tgt
            if not frozen:
                dW = np.zeros((OUT, n)); dW[:, read_idx] = np.outer(err, st[read_idx])
                W = (W - lr * dW).astype(np.float32); W[:, off] = 0.0
                mb.cp_ssm_readout_w = xp.asarray(W)
            if interleave and step % 40 == 20:
                cmp.query_patient("dog", "chase")             # composer op interleaved into the learning loop
            step += 1
    l1 = heldout_loss()
    ans = [cmp.query_patient(a, v) for a, v, _ in facts]
    abstain = cmp.query_patient("lion", "roar")
    return l0, l1, ans, abstain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--D-cmp", type=int, default=64)
    ap.add_argument("--frozen", action="store_true")
    args = ap.parse_args()

    if args.frozen:
        l0, l1, _, _ = _run(args.seed, args.epochs, args.lr, args.D_cmp, frozen=True, interleave=True)
        did_not_learn = l1 > 0.5 * l0
        v = "GO" if did_not_learn else "NO-GO"
        print(f"[RESULT {v}] FROZEN control (seed {args.seed}): held-out loss {l0:.4f} -> {l1:.4f} "
              f"(did_not_learn={did_not_learn}) -- W frozen => held-out loss does not drop; the delta update is load-bearing.")
        return

    # MAIN: with the composer interleaved
    l0, l1, ans, abstain = _run(args.seed, args.epochs, args.lr, args.D_cmp, frozen=False, interleave=True)
    # INTERLEAVE counterfactual: identical training WITHOUT the composer op -> the held-out loss must be IDENTICAL
    l0b, l1b, _, _ = _run(args.seed, args.epochs, args.lr, args.D_cmp, frozen=False, interleave=False)

    learned = l0 > 1e-6 and l1 < 0.2 * l0                     # held-out loss dropped >=5x (generalizes)
    recall_ok = (ans == ["cat", "mouse"])
    moat_ok = (abstain is None)
    interleave_noperturb = abs(l1 - l1b) < 1e-6               # composer op does NOT perturb the learning
    verdict = "GO" if (learned and recall_ok and moat_ok and interleave_noperturb) else "NO-GO"
    print(f"[RESULT {verdict}] render read-out LEARNING (teacher-student, HELD-OUT) co-resident with the composer on "
          f"ONE bridge (seed {args.seed}, {args.epochs} epochs, {n_read_note(args)}):")
    print(f"  on-bridge delta-rule HELD-OUT loss : {l0:.4f} -> {l1:.4f} ({l0/max(l1,1e-9):.1f}x drop, generalizes={learned})")
    print(f"  INTERLEAVE non-interference        : held-out with-composer {l1:.6f} == without {l1b:.6f} -> {interleave_noperturb}")
    print(f"  composer recall (interleaved)      : {ans} == ['cat','mouse'] -> {recall_ok}")
    print(f"  no-confab moat                     : {abstain!r} is None -> {moat_ok}")
    print(f"  => the render read-out's delta-update WIRE-UP runs on the shared bridge (on-bridge forward + on-bridge "
          f"cp_ssm_state eligibility; host delta arithmetic), GENERALIZES to held-out, and the composer op interleaved "
          f"into the loop does not perturb it. Learning wire-up + composer + WKV read-out on ONE substrate.")


def n_read_note(args):
    return "teacher-student 96 train / 16 held-out, n_read=32 (over-determined => generalization is genuine)"


if __name__ == "__main__":
    main()
