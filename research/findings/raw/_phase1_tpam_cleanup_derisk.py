"""Phase 1 (cheat B) de-risk: a biology-grounded TPAM cleanup (complex Hopfield / Threshold Phasor Associative
Memory, Frady-Sommer 2019 -- CA3 pattern completion + striatal WTA) gives the SAME winner as the RF composer's
current numpy phase-cosine argmax, on the composer's REAL noisy unbinds. GATE: TPAM winner == argmax winner,
multi-seed. If GO -> the numpy argmax is replaceable by the spiking-realizable TPAM (W=SS* is the bridge's complex
synapse matvec; the |Z|>floor resonate-and-fire IS the magnitude threshold). Plan:
docs/plans/2026-06-05-conversational-cheat-conversion-plan.md.
"""
import numpy as np
from research.runners.rf_phasor_composer import RFPhasorComposer


def tpam_cleanup(rec_phases, codebook, theta_frac, max_iters=12):
    """TPAM settle: vocabulary in W = S S*/D (S = codebook phasors as columns); iterate the magnitude-thresholded
    phase-preserving transfer z <- phase(W z) for neurons whose |W z| > theta (else silenced); winner =
    argmax|S* z| at convergence. theta = theta_frac * D (the magnitude gate, the WTA discretizer)."""
    words = list(codebook.keys())
    S = np.stack([np.exp(2j * np.pi * codebook[w]) for w in words], axis=1)   # D x V
    D = S.shape[0]
    W = (S @ S.conj().T) / float(D)                                            # D x D complex (= the bridge matvec)
    theta = float(theta_frac) * D / float(S.shape[1]) if False else float(theta_frac)
    z = np.exp(2j * np.pi * np.asarray(rec_phases))
    for _ in range(max_iters):
        u = W @ z
        mag = np.abs(u)
        active = mag > theta
        z_new = np.where(active & (mag > 1e-12), u / np.where(mag > 1e-12, mag, 1.0), 0.0)
        if np.allclose(z_new, z, atol=1e-6):
            break
        z = z_new
    sims = np.abs(S.conj().T @ z)
    return words[int(np.argmax(sims))]


def numpy_argmax_cleanup(rec_phases, codebook):
    words = list(codebook.keys())
    sims = [float(np.mean(np.cos(2.0 * np.pi * (rec_phases - codebook[w])))) for w in words]
    return words[int(np.argmax(sims))]


def run(seed, D, theta_frac):
    comp = RFPhasorComposer(seed=seed, D=D, period=200)
    comp.store("dog", "go", "north")
    comp.store("cat", "run", "south")
    comp.store("river", "look", "apple")
    n = n_match = n_argmax_correct = 0
    for (a, v, p), comp_phases in zip([("dog", "go", "north"), ("cat", "run", "south"), ("river", "look", "apple")],
                                      [c for _, c in comp.kb]):
        for role, truth in (("agent", a), ("action", v), ("patient", p)):
            rec = comp._unbind_phases(comp_phases, role)        # the REAL noisy spiking unbind
            w_np = numpy_argmax_cleanup(rec, comp.concepts)
            w_tp = tpam_cleanup(rec, comp.concepts, theta_frac)
            n += 1
            n_match += int(w_tp == w_np)                        # PARITY: TPAM == argmax (the gate)
            n_argmax_correct += int(w_np == truth)
    return n_match, n, n_argmax_correct


if __name__ == "__main__":
    # The dense-phasor capacity wall is the suspect -> sweep D (the capacity dial) at a fixed theta.
    for D in (128, 256, 512, 1024):
        rows = []
        for seed in (42, 43, 44):
            m, n, c = run(seed, D, 0.1)
            rows.append((seed, m, n, c))
        tot_m = sum(m for _, m, _, _ in rows)
        tot_n = sum(n for _, _, n, _ in rows)
        tot_c = sum(c for _, _, _, c in rows)
        print(f"D={D}: TPAM==argmax parity {tot_m}/{tot_n}  (argmax-correct {tot_c}/{tot_n})  "
              + "  ".join(f"s{s}:{m}/{n}" for s, m, n, _ in rows), flush=True)
