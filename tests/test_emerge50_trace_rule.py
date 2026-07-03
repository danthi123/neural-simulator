"""CI guard for EMERGE-50 — the FÖLDIÁK (1991) TRACE / temporal-continuity rule against the EMERGE-46 fully-spiking-
stacked-pooler boundary. These tests pin the DECISIVE mechanism facts, kept fast:
  (1) NUMPY PROXY (pure numpy, ~sub-second): the trace rule over TEMPORALLY-GROUPED same-super bouts routes held-out
      inheritance (held-out within-super L2 overlap >> cross-super), i.e. a held-out sub-category shares its same-super
      L2 columns via the trace. This is the mechanism that the on-substrate port carries.
  (2) NUMPY PROXY — the LOAD-BEARING control: SHUFFLED-TEMPORAL-ORDER (the SAME members, randomized order) COLLAPSES the
      discrimination (within ~= cross) -> temporal continuity is proven to be doing the work, not the member statistics.
  (3) ON-SUBSTRATE mechanism fact (~one small bridge build + a short trace pass): feeding a GRADED (traced) pre_last into
      the committed potentiation kernel binds the winner column to RECENTLY-ACTIVE (traced) inputs that are NOT in the
      current sample -- i.e. the traced potentiation reaches beyond the instantaneous input (the structural basis of the
      trace rule on the substrate). The kernels are byte-unchanged.
The DECISIVE full 3-seed on-substrate super-acc port is validated by the runner + finding, not pinned here (slow). Skip
if deps missing.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pytest

pytest.importorskip("sim.bridge")
_e50 = pytest.importorskip("research.runners._emerge50_trace_rule_derisk")


def _numpy_disc(seed, trace_decay=0.8, bout_len=12, epochs=60, shuffle_temporal=False):
    mem, l1codon = _e50._build_l1codons(seed)
    stream = _e50._make_stream(mem, l1codon, seed, bout_len=bout_len, shuffle_temporal=shuffle_temporal)
    l2 = _e50.TraceNumpyL2Pooler(seed + 1, trace_decay=trace_decay)
    l2.train(stream, epochs)
    l2codon = {m: l2.codon(l1codon[m]) for m in mem}
    return _e50._held_within_cross(mem, l2codon)


def test_numpy_trace_rule_routes_held_out_within_over_cross():
    """The trace rule over TEMPORALLY-GROUPED same-super bouts routes held-out inheritance: a held-out ENTIRE sub-category
    shares its same-superordinate L2 columns (within-super overlap high) WITHOUT flooding cross-super (cross low). This is
    the structural mechanism that surpasses the EMERGE-46 over-selective boundary."""
    ws, cs = [], []
    for s in (42, 43, 44):
        w, c = _numpy_disc(s)
        ws.append(w); cs.append(c)
    within, cross = float(np.mean(ws)), float(np.mean(cs))
    assert within >= 0.50, f"grouped held-out within-super overlap must be high (trace binds same-super), got {within:.3f}"
    assert within - cross >= 0.30, f"within must clearly exceed cross (generalization, not collision): {within:.3f} vs {cross:.3f}"


def test_numpy_shuffled_temporal_control_collapses_discrimination():
    """THE LOAD-BEARING CONTROL: present the SAME members in RANDOMIZED order (shuffled-temporal) so the trace can no
    longer bind same-super. The within>cross discrimination must COLLAPSE relative to the grouped stream -> temporal
    continuity (not the member multiset) is doing the work."""
    gw, gc, sw, sc = [], [], [], []
    for s in (42, 43, 44):
        w, c = _numpy_disc(s, shuffle_temporal=False); gw.append(w); gc.append(c)
        w2, c2 = _numpy_disc(s, shuffle_temporal=True); sw.append(w2); sc.append(c2)
    grouped_disc = float(np.mean(gw)) - float(np.mean(gc))
    shuffled_disc = float(np.mean(sw)) - float(np.mean(sc))
    assert grouped_disc - shuffled_disc >= 0.30, (
        f"shuffling the temporal order must collapse the trace's within>cross discrimination "
        f"(grouped disc {grouped_disc:+.3f} vs shuffled disc {shuffled_disc:+.3f})")


def test_onsubstrate_traced_potentiation_reaches_beyond_current_input():
    """ON-SUBSTRATE mechanism fact: with a slow trace, the committed potentiation kernel (pre_last = trace) potentiates a
    winner column's synapses from RECENTLY-ACTIVE inputs that are NOT in the CURRENT sample -- the structural basis of the
    trace rule on the substrate (a winner binds beyond the instantaneous input). Small bridge + a 2-step trace pass."""
    Probe = _e50._build_onsubstrate_probe()
    from research.runners._emerge46_spiking_stacked_pooler_derisk import NCOL1 as N_IN, NCOL2 as N_COL, K2 as KW
    from research.runners._emerge14_stageC_onbridge_learning_derisk import _host
    # Build a bare TraceOnSubstratePooler via the probe's inner class by constructing a tiny probe is heavy; instead build
    # the pooler directly through EMERGE-46 + attach the trace methods by running one train_trace over a crafted stream.
    # Two disjoint inputs A (step 1) then B (step 2); the winner on step 2 (driven by B) should have its synapses from A
    # (still in the trace) potentiated above their init -- reach beyond the current input.
    import research.runners._emerge50_trace_rule_derisk as e50
    # Access the inner TraceOnSubstratePooler by building a 1-member probe would train L1; too heavy. Rebuild the pooler:
    from research.runners._emerge46_spiking_stacked_pooler_derisk import OnSubstratePooler
    # Recreate the trace subclass locally (mirrors _build_onsubstrate_probe's inner class) to keep the test light.
    from sim.kernels import fused_htm_winner_inactive_depression

    class _TP(OnSubstratePooler):
        def train_two_step(self, A, B, trace_decay=0.8):
            trace = np.zeros(self.n_in)
            for feats in (A, B):
                x = np.zeros(self.n_in); x[list(feats)] = 1.0
                trace = np.clip(trace * trace_decay + x, 0, 1)
                win = self._winners(feats)
                e50._apply_traced_potentiation(self, trace, e50._sdr(win), self.lp)
            return win

    p = _TP(seed=7, n_in=N_IN, n_col=N_COL, k_win=KW)
    A = set(range(0, 6)); B = set(range(50, 56))
    init = _host(p.b.cp_connections.data)[p.ff_pos].copy()
    win = p.train_two_step(A, B)
    perm = _host(p.b.cp_connections.data)[p.ff_pos]
    # synapses from an A-feature (in the trace but NOT in the current input B) into a step-2 winner must have risen
    win_arr = np.fromiter((int(c) for c in win), int)
    a_feat = np.isin(p.ff_feat, np.fromiter(A, int))
    to_win = np.isin(p.ff_col, win_arr)
    reached = a_feat & to_win
    assert reached.any(), "there should be A-feature -> step-2-winner synapses to test"
    rose = float(np.mean(perm[reached] > init[reached] + 1e-6))
    assert rose >= 0.5, (f"traced potentiation must reach beyond the current input: A-feature->winner synapses should "
                         f"rise (fraction risen {rose:.2f}) even though A is not in the step-2 sample")
