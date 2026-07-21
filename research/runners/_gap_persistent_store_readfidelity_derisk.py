"""Fact-store Phase 2 — cheap-first de-risk of the ONE open question: staged vs PERSISTENT store read FIDELITY.

The composer's current read STAGES the store: install the store operator (trigger->readout = composite), kick + settle,
then SWAP it out for the unbind operator and read. A persistent store (cp_rf_store_re/im, Phase 1) instead KEEPS
driving the readout via the additive matvec term DURING the unbind/cleanup windows. Does that change what the unbind
reads? Reasoned answer: NO, because the RF read is PHASE-based + magnitude-invariant -- a refreshed-at-full-magnitude
readout and a decaying one carry the SAME phase. This probe confirms it on a minimal FHRR store->unbind->decode.

Layout on a pure-RF bridge (n = 1 + 2D): trigger [0]; readout [1..1+D); Q [1+D..1+2D). composite = r (+) f (phase sum).
  store: readout_k <- trigger, weight = exp(i2pi*composite_k)   (drives readout to the composite phase)
  unbind: Q_k <- readout_k, weight = exp(-i2pi*r_k)             (Q = composite (-) r = f)  -- rows [1+D..) DISJOINT from readout
STAGED: install store op -> kick trigger -> resonate; SWAP to unbind op -> resonate; read Q.
PERSISTENT: rf_set_store_weights(store) [persistent] + rf_set_complex_weights(unbind) -> kick -> resonate; read Q.

GO: (1) both decode the true filler f (phase err small), and (2) persistent Q phase ~= staged Q phase (read-fidelity
clean). Reuse-by-import (_build_rf_encoder); the Phase-1 sim/ mechanism; `--seed`, `--D`.
"""
import argparse
import numpy as np

from sim.backend import to_host
from research.runners._emerge_wkv_onbridge_derisk import _build_rf_encoder


def _phasor(phase):
    return np.exp(2j * np.pi * np.asarray(phase))


def _circ_absdiff(a, b):
    d = np.abs((np.asarray(a) - np.asarray(b) + 0.5) % 1.0 - 0.5)   # circular |phase difference| in [0, 0.5]
    return float(np.mean(d))


def _read_Q(b, D):
    ph = np.asarray(to_host(b.rf_read_phases()), np.float64)
    return ph[1 + D:1 + 2 * D]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--period", type=int, default=200)
    args = ap.parse_args()
    D = args.D; n = 1 + 2 * D; per = args.period
    rng = np.random.default_rng(args.seed)
    r = rng.uniform(0, 1, D); f = rng.uniform(0, 1, D)             # role + filler phases
    composite = (r + f) % 1.0                                      # bind = phase sum
    store_conns = [(1 + k, 0, _phasor(composite[k])) for k in range(D)]        # readout_k <- trigger
    unbind_conns = [(1 + D + k, 1 + k, _phasor(-r[k])) for k in range(D)]      # Q_k <- readout_k
    kick = np.zeros(n, np.complex128); kick[0] = 1.0 + 0.0j                    # fire the trigger (unit phasor)

    # --- STAGED (the composer's current staged path): install store op, settle, SWAP to unbind, read ---
    bS = _build_rf_encoder(n, seed=args.seed)
    bS.rf_set_complex_weights(store_conns); bS.rf_kick(kick, period=per, lam=0.0)
    bS.rf_resonate_steps(per + 8)
    bS.rf_set_complex_weights(unbind_conns); bS.rf_resonate_steps(per + 8)     # store SWAPPED OUT
    Q_staged = _read_Q(bS, D)

    # --- PERSISTENT (Phase 1 store): store stays via the additive term DURING the unbind ---
    bP = _build_rf_encoder(n, seed=args.seed)
    bP.rf_set_store_weights(store_conns)                          # PERSISTENT (cp_rf_store_*), never swapped out
    bP.rf_set_complex_weights(unbind_conns)                      # the per-op unbind on cp_rf_w_* (disjoint rows)
    bP.rf_kick(kick, period=per, lam=0.0)
    bP.rf_resonate_steps(per + 8)
    Q_persist = _read_Q(bP, D)

    err_staged = _circ_absdiff(Q_staged, f)                      # does staged decode the true filler?
    err_persist = _circ_absdiff(Q_persist, f)                    # does persistent decode the true filler?
    fidelity = _circ_absdiff(Q_persist, Q_staged)               # persistent vs staged read (the open question)

    # controls: chance-level circular error ~ 0.25; a correct decode is << 0.25
    staged_ok = err_staged < 0.08
    persist_ok = err_persist < 0.08
    fidelity_ok = fidelity < 0.05                                # persistent ~= staged
    verdict = "GO" if (staged_ok and persist_ok and fidelity_ok) else "NO-GO"
    print(f"[RESULT {verdict}] persistent-store read fidelity (seed {args.seed}, D={D}):")
    print(f"  STAGED decodes filler f     : mean circular |dphase| = {err_staged:.4f}  ({'ok' if staged_ok else 'BAD'}; chance ~0.25)")
    print(f"  PERSISTENT decodes filler f : mean circular |dphase| = {err_persist:.4f}  ({'ok' if persist_ok else 'BAD'})")
    print(f"  PERSISTENT vs STAGED read   : mean circular |dphase| = {fidelity:.4f}  ({'CLEAN (same read)' if fidelity_ok else 'DIVERGES'})")
    print(f"  => a persistent store (kept driving the readout via the additive term through the unbind window) reads "
          f"the SAME filler as the staged store swapped out -- the RF read is phase-based + magnitude-invariant. "
          f"Phase 2's staged->continuous read-fidelity risk is {'retired' if verdict=='GO' else 'REAL (characterize)'}.")


if __name__ == "__main__":
    main()
