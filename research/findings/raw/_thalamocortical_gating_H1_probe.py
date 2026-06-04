"""Thalamocortical gating cheap-first H1 (deep-research Track 2, Logiaco-Abbott-Escola 2021): does a
MULTIPLICATIVE GATE give deterministic VARIABLE BINDING (same role, different fillers, on command, no
retraining) where GROWN STATIC WEIGHTS are seed-fragile and overwrite?

This is the gate-keeper before any bridge build. Reduced rate-model, NumPy (CPU), minutes. N roles x M
fillers; structurally pre-wired routes role_i -> filler_j.
  - GATING model: bind(i,j) opens gate g[i,j]=1 (and closes the other gates for role i); query(i) = the
    filler whose gated route from role i is open. Binding = which gate is open (dynamical state selection).
  - GROWN-WEIGHT model: bind(i,j) Hebbian-grows w[i,j] += 1 (co-fire); query(i) = argmax_j w[i,j]. Binding =
    which weight grew (synaptic storage).

PRE-REGISTERED, FROZEN gate (the discriminator is RE-BINDING + interference, which grown weights cannot do):
  Protocol: a random sequence of bind commands over N roles x M fillers that INCLUDES re-bindings (a role is
  re-bound to a new filler later). After the sequence, query every role; correct = the role's LATEST commanded
  binding. 200 random sequences x N=M=4, multi-seed.
  THREE-STATE:
    RESOLVES := gating accuracy >= 0.99 AND grown-weight accuracy materially lower (< 0.80) -> gating gives
                on-command variable binding that grown weights cannot; the lever is real. Proceed to the
                spiking bridge integration (per-pathway multiplicative transmission gate).
    BOUNDARY := both high (the re-binding protocol didn't expose a difference) -> redesign the discriminator.
    DOES-NOT-RESOLVE := gating not ~perfect -> the gate model is wrong.

  python -m research.findings.raw._thalamocortical_gating_H1_probe
"""
import numpy as np

N_ROLE = 4
M_FILL = 4
N_SEQ = 200
SEEDS = (42, 43, 44)


def run_sequence(rng):
    """A random sequence of bind(role, filler) commands with re-bindings; returns the ground-truth latest map."""
    n_cmds = rng.integers(N_ROLE + 2, 3 * N_ROLE)        # enough to force re-bindings
    cmds = [(int(rng.integers(N_ROLE)), int(rng.integers(M_FILL))) for _ in range(n_cmds)]
    latest = {}
    for i, j in cmds:
        latest[i] = j                                    # the LATEST command for role i is the truth
    return cmds, latest


def gating_model(cmds):
    """Binding = which gate is open. Re-binding closes the role's other gates and opens the new one."""
    gate = np.zeros((N_ROLE, M_FILL))
    for i, j in cmds:
        gate[i, :] = 0.0                                 # close all of role i's gates...
        gate[i, j] = 1.0                                 # ...open the commanded one (dynamical state selection)
    # query role i: the filler whose gated route is open (routes pre-wired with equal fixed weight)
    return {i: int(np.argmax(gate[i])) if gate[i].max() > 0 else -1 for i in range(N_ROLE)}


def grown_weight_model(cmds, hebb=1.0):
    """Binding = which weight grew. Re-binding grows the new weight but the OLD weight persists (no unlearning)."""
    w = np.zeros((N_ROLE, M_FILL))
    for i, j in cmds:
        w[i, j] += hebb                                  # Hebbian co-fire potentiation (additive, no decay)
    return {i: int(np.argmax(w[i])) if w[i].max() > 0 else -1 for i in range(N_ROLE)}


def main():
    print(f"=== thalamocortical gating H1: variable binding (N={N_ROLE} roles x M={M_FILL} fillers) ===", flush=True)
    print("    re-binding protocol: a role is re-bound to a new filler later; correct = the LATEST binding.", flush=True)
    g_accs, w_accs = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        g_ok = w_ok = total = 0
        for _ in range(N_SEQ):
            cmds, latest = run_sequence(rng)
            gq = gating_model(cmds)
            wq = grown_weight_model(cmds)
            for i, j_true in latest.items():
                total += 1
                g_ok += int(gq[i] == j_true)
                w_ok += int(wq[i] == j_true)
        g_accs.append(g_ok / total)
        w_accs.append(w_ok / total)
    g = float(np.mean(g_accs))
    w = float(np.mean(w_accs))
    print(f"\n  GATING (which gate is open):     latest-binding accuracy {g:.3f}", flush=True)
    print(f"  GROWN WEIGHTS (which weight grew): latest-binding accuracy {w:.3f}", flush=True)
    print(f"  (grown weights fail on RE-BINDING: the first binding's weight persists and wins ties / dominates)",
          flush=True)
    if g >= 0.99 and w < 0.80:
        verdict = ("RESOLVES -- a multiplicative gate gives deterministic on-command variable binding; grown "
                   "static weights cannot re-bind (the old weight persists). The thalamocortical gating lever "
                   "is real at toy scale -> justify the spiking-bridge per-pathway multiplicative transmission gate.")
    elif g >= 0.99:
        verdict = f"BOUNDARY -- grown weights also scored {w:.2f}; the re-binding protocol didn't expose a gap."
    else:
        verdict = f"DOES-NOT-RESOLVE -- gating only {g:.2f}; the gate model is wrong."
    print(f"\nVERDICT: {verdict}", flush=True)
    return verdict


if __name__ == "__main__":
    main()
