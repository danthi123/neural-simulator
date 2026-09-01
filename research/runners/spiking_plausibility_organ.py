"""BRAIN-NATIVE PLAUSIBILITY — a spiking/synaptic read of the fact-association graph that REPLACES the host
`GenerativeReplayProposer._related(w1, w2) = P[w1, w2] >= tau` matrix-comparison in the #3E open-ended
GENERATE channel of the production `/api/brain-chat` turn.

THE HOST SHORTCUT (what this converts).  In the generate channel the brain VOLUNTEERS a novel grounded
proposition and each candidate SVO triple is gated by PLAUSIBILITY:

    _plausible(a, ac, p) = _related(a, ac) and _related(ac, p)          # selectional preference
    _related(w1, w2)     = P[row[w1], row[w2]] >= tau                    # <-- HOST float comparison

`P` is the brain's own concept co-occurrence over its stored facts; `tau` = the 50th percentile of the
positive edges. The DRAW is already spiking (VocabAgnosticSpikingSampler), but this GATE — the decision
"is `a` plausibly related to `ac`" — sits between sensation and action and is computed by a host matrix
comparison, not by the brain. The generate-channel finding (2026-08-18-generate-channel-wired-brain-chat-GO)
names it explicitly: "the plausibility LIKELIHOOD is a host co-occurrence matrix over the brain's own facts".

THE BRAIN-NATIVE MECHANISM (this organ).  The co-occurrence graph is embodied as SYNAPTIC WEIGHTS and the
relatedness decision is computed by SPIKES propagating across those synapses — a monosynaptic associative
read:

  1. Two Izhikevich populations (cortex_ctx "input" + dlpfc_wm "readout"), one disjoint neuron assembly per
     concept, on a real `SimulationBridge` (reuses the validated `build_loop_wm_bridge`).
  2. The association graph is installed as DIRECTED synapses cortex_A -> dlpfc_B with weight PROPORTIONAL to
     the co-occurrence count P[A, B] (Hebbian: association strength == synaptic strength). The readout layer
     never projects back -> the read is strictly MONOSYNAPTIC (no multi-hop / transitive-closure blow-up
     that made a recurrent spreading read saturate — see the module's calibration finding).
  3. related(A, B) = drive input assembly A with a brief current pulse; B's readout assembly FIRES iff the
     A->B synapse carries enough current to bring it to threshold. A weak co-occurrence -> weak synapse ->
     sub-threshold EPSP -> B stays silent -> "not related"; a strong co-occurrence -> supra-threshold -> B
     fires -> "related". The tau boundary EMERGES from the spike threshold, it is not a host `>=`.
  4. The readout threshold is the brain's OWN: `tau_spike` = the 50th percentile of the POSITIVE readout
     firing-fractions (the SAME operating-point rule the host applies to P, applied to the brain's spiking
     output). So the hot-path decision `firing_frac[A][B] >= tau_spike` reads a SPIKE COUNT, never P.

WHAT REMAINS HOST (declared).  The synaptic weight matrix is SET from the co-occurrence counts (the same
counts the host P holds); online Hebbian self-organization of those weights is the named next rung. The
selectional-preference STRUCTURE (`related(a,ac) and related(ac,p)`) and the SVO template are unchanged.
The advance this organ delivers: the plausibility DECISION is computed by neurons + synapses + their
spikes, not by a host matrix comparison.

Additive, master-switched, byte-identical when OFF (the organ is never built unless installed). Reuse-by-
import, NO sim/ edit, CPU (numpy backend fine).

LESION handles (for load-bearing attribution):
  lesion="shuffle"  -> install a shuffled-off-diagonal P (marginals preserved, neighbourhoods destroyed):
                       the learned structure is gone, so replay's plausibility advantage must collapse to
                       the random floor (the b2 shuffled-graph anti-cheat, now in synapses).
  lesion="ablate"   -> zero the association synapses: no A->B pathway -> nothing downstream fires ->
                       relatedness collapses (proves the synaptic association is what carries the signal).
"""
from __future__ import annotations

import numpy as np


def _shuffle_offdiag(P, rng):
    """Shuffle the OFF-DIAGONAL upper-triangle entries of the symmetric co-occurrence matrix and
    re-symmetrize (marginal edge-weight multiset preserved, every neighbourhood destroyed). Mirrors
    `_genfrontier_b2_generative_replay_derisk.shuffle_graph` so the synaptic lesion is the exact
    spiking analogue of the b2 host anti-cheat."""
    Nm = P.shape[0]
    iu = np.triu_indices(Nm, k=1)
    vals = P[iu].copy()
    rng.shuffle(vals)
    Ps = np.zeros_like(P)
    Ps[iu] = vals
    return Ps + Ps.T


class SpikingAssociativePlausibilityOrgan:
    """A spiking monosynaptic associative read of the co-occurrence graph. `related(w1, w2)` is computed by
    driving w1's assembly and reading whether w2's readout assembly fires above the brain's own threshold.
    `install(prop)` swaps it in for a `GenerativeReplayProposer`'s host `_related` (so `_plausible` becomes
    spiking); byte-identical when never installed."""

    def __init__(self, P, row, vocab=None, seed=42, tau_pct=50.0, pattern_size=12, gain=16.0,
                 drive_pA=2500.0, stim_steps=6, read_window=10, lesion=None, verbose=False):
        import sim.backend as _B
        self.B = _B
        self.xp, _ = _B.get_backend()
        self.row = dict(row)
        self.vocab = list(vocab) if vocab is not None else sorted(row.keys(), key=lambda w: row[w])
        self.seed = int(seed)
        self.tau_pct = float(tau_pct)
        self.pattern_size = int(pattern_size)
        self.gain = float(gain)
        self.drive_pA = float(drive_pA)
        self.stim_steps = int(stim_steps)
        self.read_window = int(read_window)
        self.lesion = lesion
        # the association weight matrix the SYNAPSES encode (Hebbian: co-occurrence -> synaptic strength).
        Pw = np.array(P, dtype=float)
        if lesion == "shuffle":
            Pw = _shuffle_offdiag(Pw, np.random.default_rng(self.seed * 101 + 7))
        elif lesion == "ablate":
            Pw = np.zeros_like(Pw)
        self._Pw = Pw
        # provenance: every related() decision reads this SPIKE-derived matrix, never P>=tau.
        self.n_spiking_reads = 0
        self.n_host_related_calls = 0     # MUST stay 0 while installed (a host P>=tau leak would increment)
        self._build_bridge()
        self._firing_frac = self._precompute_readout()      # {A: {B: readout firing fraction}}
        self.tau_spike = self._self_threshold()
        # cache the boolean relatedness (deterministic given the installed synapses)
        self._related_cache = {}
        if verbose:
            print(f"[spiking-plausibility] |vocab|={len(self.vocab)} gain={self.gain} "
                  f"tau_spike={self.tau_spike:.4f} lesion={self.lesion}", flush=True)

    # ---- the substrate: two Izhikevich pools + a co-occurrence-weighted feedforward synapse matrix ----
    def _build_bridge(self):
        from research.runners.content_selection_spiking import build_loop_wm_bridge
        n = max(600, 2 * self.pattern_size * len(self.vocab))
        # inert generic loop (loop_weight=0 -> the dlpfc_wm readout NEVER projects back to cortex_ctx, so the
        # read stays monosynaptic); a small density is kept only so the region framework generates synapses.
        self.bridge = build_loop_wm_bridge(n=n, density=0.05, loop_weight=0.0, loop_density=0.02,
                                           seed=self.seed, enable_ou=False, verbose=False)
        rm = self.bridge.region_manager
        cin = np.asarray(rm.indices("cortex_ctx"))
        dout = np.asarray(rm.indices("dlpfc_wm"))
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(min(len(cin), len(dout)))
        ps = self.pattern_size
        self._cin, self._dout = {}, {}
        for i, c in enumerate(self.vocab):
            idx = perm[i * ps:(i + 1) * ps]
            self._cin[c] = cin[idx]
            self._dout[c] = dout[idx]
        # install cortex_A -> dlpfc_B synapses, weight PROPORTIONAL to the co-occurrence count P[A, B].
        pre_all, post_all, w_all = [], [], []
        for a in self.vocab:
            ra = self.row.get(a)
            for b in self.vocab:
                if a == b:
                    continue
                rb = self.row.get(b)
                if ra is None or rb is None:
                    continue
                w = self._Pw[ra, rb]
                if w <= 0:
                    continue
                pre_all.append(np.repeat(self._cin[a], ps))
                post_all.append(np.tile(self._dout[b], ps))
                w_all.append(np.ones(ps * ps, np.float32) * (float(w) * self.gain))
        if pre_all:
            self.bridge.set_pathway_weights(
                "c2d", pre_indices=np.concatenate(pre_all).astype(np.int64),
                post_indices=np.concatenate(post_all).astype(np.int64),
                weights=np.concatenate(w_all), add_missing=True)
        self._v0 = self.bridge.cp_membrane_potential_v.copy()
        self._u0 = self.bridge.cp_recovery_variable_u.copy()

    def _reset(self):
        b = self.bridge
        b.cp_membrane_potential_v[:] = self._v0
        b.cp_recovery_variable_u[:] = self._u0
        for a in ("cp_firing_states", "cp_prev_firing_states"):
            arr = getattr(b, a, None)
            if arr is not None:
                arr[:] = False
        for a in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda",
                  "cp_conductance_g_nmda_rise", "cp_refractory_timers", "cp_synapse_pulse_timers",
                  "cp_synapse_pulse_progress"):
            arr = getattr(b, a, None)
            if arr is not None:
                arr[:] = 0

    def _drive_read(self, a):
        """Drive input assembly `a` for `stim_steps`, then read every concept's readout assembly firing
        fraction over the post-drive window (the monosynaptic wave). Returns {concept: firing fraction}."""
        self._reset()
        b = self.bridge
        drv = self._cin[a]
        acc = {c: 0.0 for c in self.vocab}
        for t in range(self.stim_steps + self.read_window):
            b.cp_external_input_current[:] = 0.0
            if t < self.stim_steps:
                b.cp_external_input_current[drv] = self.drive_pA
            b._run_one_simulation_step()
            b.runtime_state.current_time_ms += b.core_config.dt_ms
            if t >= self.stim_steps:
                fs = np.asarray(self.B.to_host(b.cp_firing_states)).astype(float)
                for c in self.vocab:
                    acc[c] += float(fs[self._dout[c]].sum())
        b.cp_external_input_current[:] = 0.0
        denom = float(self.pattern_size * self.read_window)
        return {c: acc[c] / denom for c in self.vocab}

    def _precompute_readout(self):
        """One monosynaptic drive-read per concept (|vocab| spiking probes) -> the full readout matrix. The
        read is deterministic (OU noise OFF), so caching the SPIKE-derived matrix is exact reuse, not a
        shortcut (it is computed by the brain once, exactly as the host computes P once)."""
        out = {}
        for a in self.vocab:
            out[a] = self._drive_read(a)
            self.n_spiking_reads += 1
        return out

    def _self_threshold(self):
        """The brain's OWN operating point: `tau_spike` = the `tau_pct`-th percentile of the POSITIVE readout
        firing-fractions — the SAME percentile rule the host applies to the positive P edges, applied to the
        brain's spiking output. No host P enters this."""
        vals = [self._firing_frac[a][b] for a in self.vocab for b in self.vocab
                if a != b and self._firing_frac[a][b] > 0.0]
        if not vals:
            return 1e9      # nothing fired (e.g. ablate lesion) -> nothing is ever related
        return float(np.percentile(np.asarray(vals), self.tau_pct))

    # ---- the drop-in for GenerativeReplayProposer._related ----
    def related(self, w1, w2):
        """SPIKING relatedness: w2's readout assembly fired at/above the brain's own threshold when w1 was
        driven. Reads the spike-derived readout matrix, NEVER P>=tau."""
        key = (w1, w2)
        cached = self._related_cache.get(key)
        if cached is not None:
            return cached
        fr = self._firing_frac.get(w1)
        val = bool(fr is not None and fr.get(w2, 0.0) >= self.tau_spike)
        self._related_cache[key] = val
        return val

    def install(self, prop):
        """Route the proposer's `_plausible` gate through this spiking read: save the host `_related` and
        replace it with `self.related`. `_plausible` (unchanged) now decides via spikes. Idempotent."""
        if getattr(prop, "_related_host", None) is None:
            prop._related_host = prop._related
        prop._related = self.related
        prop._spiking_plausibility_organ = self
        return self

    @staticmethod
    def uninstall(prop):
        """Restore the host `_related` (byte-identical to never-installed)."""
        host = getattr(prop, "_related_host", None)
        if host is not None:
            prop._related = host
            prop._related_host = None
            prop._spiking_plausibility_organ = None

    # ---- diagnostics ----
    def host_related(self, P, row, tau, w1, w2):
        """The host relation, for AGREEMENT diagnostics only (NOT used on the hot path)."""
        self.n_host_related_calls += 1
        return float(P[row[w1], row[w2]]) >= float(tau)

    def agreement_with_host(self, P, row, tau):
        """Fraction of ordered concept pairs where the spiking read agrees with host `P>=tau`, plus F1."""
        pairs = [(a, b) for a in self.vocab for b in self.vocab if a != b]
        tp = fp = fn = tn = 0
        for a, b in pairs:
            s = self.related(a, b)
            h = float(P[row[a], row[b]]) >= float(tau)
            if s and h:
                tp += 1
            elif s and not h:
                fp += 1
            elif (not s) and h:
                fn += 1
            else:
                tn += 1
        n = max(1, len(pairs))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return {"agreement": (tp + tn) / n, "f1": f1, "precision": prec, "recall": rec,
                "n_pairs": len(pairs), "spk_related": tp + fp, "host_related": tp + fn}


def build_for_proposer(prop, seed=42, lesion=None, **kw):
    """Build a `SpikingAssociativePlausibilityOrgan` from a live `GenerativeReplayProposer` (reads its P/row).
    The vocab is the proposer's graph vocabulary (row keys)."""
    vocab = sorted(prop.row.keys(), key=lambda w: prop.row[w])
    return SpikingAssociativePlausibilityOrgan(prop.P, prop.row, vocab=vocab, seed=seed, lesion=lesion, **kw)
