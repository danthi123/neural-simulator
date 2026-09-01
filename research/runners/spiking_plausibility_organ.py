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

THE ROBUSTNESS RUNG — ENSEMBLE + GRADED READ (2026-09-01, this file's second arc).  The QUALIFIED (default-
OFF) organ used a SINGLE small readout assembly (pattern_size 12) per concept and a HARD fire/no-fire read
at the median. On the SPARSE tiny own-facts graph that made the read (i) more VARIABLE across seeds (a
12-neuron patch's firing fraction is coarse; the median operating point jitters) and (ii) too SELECTIVE (a
double-jeopardy hard AND over two point-neuron thresholds suppressed generation — seed 44 volunteered 1 of
5). Two additive levers close that gap, staying fully on-substrate (both read `cp_firing_states`, never P):

  ENSEMBLE / larger readout population (`n_ensemble` K disjoint assembly assignments per concept + a larger
    `pattern_size`).  A real cortex reads a REDUNDANT population, not a 12-neuron patch. related(A,B) drives
    ALL K of A's input assemblies and averages B's firing fraction over ALL K of B's readout assemblies
    (each B^k receives only from the matching A^k, so the K reads are independent parallel monosynaptic
    reads). Spatial averaging over a bigger, redundant population -> a finer, lower-variance firing-fraction
    estimate -> the median operating point `tau_spike` stabilises across seeds.

  GRADED (rate-coded soft-AND) plausibility (`graded=True`).  Instead of thresholding EACH leg to a hard
    boolean and AND-ing them (which double-jeopardies borderline pairs and over-selects on a sparse graph),
    keep each leg's rate-coded relatedness as a SOFT probability r in [0,1] — a logistic around the brain's
    OWN operating point `tau_spike`, with softness `beta` set from the SPREAD of the brain's positive firing
    fractions — and require the GEOMETRIC MEAN of the two legs' soft-relatedness to clear 0.5 (the smooth
    analogue of "both legs above the median"). A strong leg partially compensates a borderline leg
    (preserving borderline-but-supported triples -> restores generation), while a weak-weak pair still
    fails (discrimination preserved). The DECISION reads firing fractions (spikes); the soft-AND is the SAME
    host selectional-preference STRUCTURE the QUALIFIED organ already used (`related and related`), applied
    to rate-coded scores instead of booleans. As `beta -> 0` and K=1 this recovers the hard read exactly.

WHAT REMAINS HOST (declared).  The synaptic weight matrix is SET from the co-occurrence counts (the same
counts the host P holds); online Hebbian self-organization of those weights is the named next rung. The
selectional-preference STRUCTURE (`related(a,ac) and related(ac,p)` / its graded soft-AND) and the SVO
template are unchanged. The advance this organ delivers: the plausibility DECISION is computed by neurons +
synapses + their spikes, not by a host matrix comparison.

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
    spiking); byte-identical when never installed.

    ENSEMBLE + GRADED (2026-09-01): `n_ensemble` K redundant readout populations per concept (averaged) cut
    the small-graph operating-point variance; `graded=True` installs a rate-coded soft-AND `_plausible`
    (logistic soft-relatedness, geometric-mean >= 0.5) that preserves borderline-but-supported triples so
    generation is not suppressed. n_ensemble=1 + graded=False reproduces the QUALIFIED hard read EXACTLY."""

    def __init__(self, P, row, vocab=None, seed=42, tau_pct=50.0, pattern_size=12, gain=16.0,
                 drive_pA=2500.0, stim_steps=6, read_window=10, n_ensemble=1, graded=False,
                 beta_frac=0.6, density=0.05, lesion=None, verbose=False):
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
        self.n_ensemble = max(1, int(n_ensemble))
        self.graded = bool(graded)
        self.beta_frac = float(beta_frac)
        self.density = float(density)
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
        self.tau_spike, self._beta = self._self_threshold()
        # cache the boolean relatedness (deterministic given the installed synapses)
        self._related_cache = {}
        if verbose:
            print(f"[spiking-plausibility] |vocab|={len(self.vocab)} gain={self.gain} ps={self.pattern_size} "
                  f"K={self.n_ensemble} graded={self.graded} tau_spike={self.tau_spike:.4f} "
                  f"beta={self._beta:.4f} lesion={self.lesion}", flush=True)

    # ---- the substrate: two Izhikevich pools + a co-occurrence-weighted feedforward synapse matrix ----
    def _build_bridge(self):
        from research.runners.content_selection_spiking import build_loop_wm_bridge
        ps, K = self.pattern_size, self.n_ensemble
        # enough neurons per region for K disjoint assemblies of ps neurons for every concept (input pool in
        # cortex_ctx, readout pool in dlpfc_wm), with the original 2x head-room.
        n = max(600, 2 * ps * len(self.vocab) * K)
        # inert generic loop (loop_weight=0 -> the dlpfc_wm readout NEVER projects back to cortex_ctx, so the
        # read stays monosynaptic). `density` is the INTERNAL region recurrence: at the QUALIFIED 0.05 it spreads
        # a driven input assembly's activation to other cortex neurons (which fire their OWN co-occurrence
        # synapses -> CROSS-TALK that contaminates the read and CAPS recall, worse for larger assemblies). A low
        # density suppresses that spread so the read is the PURE monosynaptic c2d wave (higher-fidelity read).
        self.bridge = build_loop_wm_bridge(n=n, density=self.density, loop_weight=0.0, loop_density=0.02,
                                           seed=self.seed, enable_ou=False, verbose=False)
        rm = self.bridge.region_manager
        cin = np.asarray(rm.indices("cortex_ctx"))
        dout = np.asarray(rm.indices("dlpfc_wm"))
        rng = np.random.default_rng(self.seed)
        perm_in = rng.permutation(len(cin))
        perm_out = rng.permutation(len(dout))
        # per-concept assemblies: K disjoint input assemblies (cortex) + K disjoint readout assemblies (dlpfc).
        # _cin[c]/_dout[c] = the FULL (concatenated over K) assembly used to drive/read; _cin_k/_dout_k keep the
        # per-ensemble split so synapses only wire the MATCHING assignment (A^k -> B^k), keeping the K reads
        # independent parallel monosynaptic reads.
        self._cin, self._dout = {}, {}
        self._cin_k, self._dout_k = {c: [] for c in self.vocab}, {c: [] for c in self.vocab}
        slot = 0
        for k in range(K):
            for c in self.vocab:
                ai = cin[perm_in[slot * ps:(slot + 1) * ps]]
                bi = dout[perm_out[slot * ps:(slot + 1) * ps]]
                self._cin_k[c].append(ai)
                self._dout_k[c].append(bi)
                slot += 1
        for c in self.vocab:
            self._cin[c] = np.concatenate(self._cin_k[c])
            self._dout[c] = np.concatenate(self._dout_k[c])
        # install cortex_A^k -> dlpfc_B^k synapses, weight PROPORTIONAL to the co-occurrence count P[A, B].
        pre_all, post_all, w_all = [], [], []
        for a in self.vocab:
            ra = self.row.get(a)
            if ra is None:
                continue
            for b in self.vocab:
                if a == b:
                    continue
                rb = self.row.get(b)
                if rb is None:
                    continue
                w = self._Pw[ra, rb]
                if w <= 0:
                    continue
                for k in range(K):
                    pre_all.append(np.repeat(self._cin_k[a][k], ps))
                    post_all.append(np.tile(self._dout_k[b][k], ps))
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
        """Drive input assembly `a` (ALL K ensemble copies) for `stim_steps`, then read every concept's
        readout assembly firing fraction over the post-drive window (the monosynaptic wave), averaged over
        the K redundant readout populations. Returns {concept: firing fraction}."""
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
        # denom counts EVERY readout neuron of the concept (ps * K) across the read window -> the ensemble
        # average is a plain firing fraction in [0, 1].
        denom = float(self.pattern_size * self.n_ensemble * self.read_window)
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
        brain's spiking output. `beta` (graded softness) = `beta_frac` x `tau_spike` — the softness scales
        with the operating point itself (seed-robust, unlike the raw std, which varies wildly with the
        firing-fraction spread and can make the soft gate flood on high-spread seeds). No host P enters this."""
        vals = [self._firing_frac[a][b] for a in self.vocab for b in self.vocab
                if a != b and self._firing_frac[a][b] > 0.0]
        if not vals:
            return 1e9, 1e-9      # nothing fired (e.g. ablate lesion) -> nothing is ever related
        arr = np.asarray(vals)
        tau = float(np.percentile(arr, self.tau_pct))
        beta = max(1e-9, self.beta_frac * tau)
        return tau, beta

    # ---- the drop-in for GenerativeReplayProposer._related ----
    def related(self, w1, w2):
        """SPIKING relatedness (HARD): w2's readout assembly fired at/above the brain's own threshold when w1
        was driven. Reads the spike-derived readout matrix, NEVER P>=tau."""
        key = (w1, w2)
        cached = self._related_cache.get(key)
        if cached is not None:
            return cached
        fr = self._firing_frac.get(w1)
        val = bool(fr is not None and fr.get(w2, 0.0) >= self.tau_spike)
        self._related_cache[key] = val
        return val

    def _soft_related(self, s):
        """Rate-coded soft relatedness in [0,1]: a logistic on the firing fraction `s` around the brain's own
        operating point `tau_spike`, softness `beta`. r=0.5 at s==tau_spike; -> the hard step as beta->0."""
        return 1.0 / (1.0 + float(np.exp(-(s - self.tau_spike) / self._beta)))

    def related_score(self, w1, w2):
        """The graded (soft) relatedness score in [0,1] for a pair — reads the spike-derived readout, never
        P. A completely silent leg (no synaptic pathway) scores 0."""
        fr = self._firing_frac.get(w1)
        s = 0.0 if fr is None else fr.get(w2, 0.0)
        if s <= 0.0:
            return 0.0
        return self._soft_related(s)

    def plausible_graded(self, a, ac, p):
        """GRADED (rate-coded soft-AND) selectional-preference plausibility: PASS iff the GEOMETRIC MEAN of
        the two legs' soft-relatedness clears 0.5 (== r(a,ac) * r(ac,p) >= 0.25) — the smooth analogue of
        'both legs above the median'. A strong leg partially compensates a borderline leg (preserves
        borderline-but-supported triples); a genuinely silent leg (no synaptic pathway) hard-fails. Reads
        firing fractions (spikes), never P; the soft-AND is the same host selectional-preference STRUCTURE
        the hard read used, on rate-coded scores instead of booleans."""
        r1 = self.related_score(a, ac)
        r2 = self.related_score(ac, p)
        if r1 <= 0.0 or r2 <= 0.0:
            return False
        return (r1 * r2) >= 0.25

    def install(self, prop):
        """Route the proposer's plausibility gate through this spiking read. Saves the host `_related` and
        replaces it with `self.related`; in GRADED mode ALSO replaces `_plausible` with the rate-coded
        soft-AND `self.plausible_graded`. `_plausible` (host structure) then decides via spikes. Idempotent."""
        if getattr(prop, "_related_host", None) is None:
            prop._related_host = prop._related
        prop._related = self.related
        if self.graded:
            if getattr(prop, "_plausible_host", None) is None:
                prop._plausible_host = prop._plausible
            prop._plausible = self.plausible_graded
        prop._spiking_plausibility_organ = self
        return self

    @staticmethod
    def uninstall(prop):
        """Restore the host `_related` / `_plausible` (byte-identical to never-installed)."""
        host = getattr(prop, "_related_host", None)
        if host is not None:
            prop._related = host
            prop._related_host = None
        ph = getattr(prop, "_plausible_host", None)
        if ph is not None:
            prop._plausible = ph
            prop._plausible_host = None
        prop._spiking_plausibility_organ = None

    # ---- diagnostics ----
    def host_related(self, P, row, tau, w1, w2):
        """The host relation, for AGREEMENT diagnostics only (NOT used on the hot path)."""
        self.n_host_related_calls += 1
        return float(P[row[w1], row[w2]]) >= float(tau)

    def agreement_with_host(self, P, row, tau):
        """Fraction of ordered concept pairs where the spiking read agrees with host `P>=tau`, plus F1.
        Uses the HARD `related()` (the boolean relation), so the agreement metric is comparable across the
        hard and graded organs."""
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


# PRODUCTION read config — the robustness rung that reaches host-level parity + generation on ALL 6 seeds.
# What closed the gap (6-seed de-risk _plausibility_ensemble_graded_derisk): (1) an ENSEMBLE of K=8 redundant
# readout populations per concept (each ps=16 neurons), averaged -> a finer, low-variance firing-fraction read
# with a STABLE operating point; (2) `density=0.0` — no internal region recurrence, so a driven input assembly
# cannot spread activation into other concepts' assemblies (the CROSS-TALK that capped recall on the sparse
# graph and was worse for larger assemblies); (3) `gain=12` — the synaptic gain in the MONOTONIC (non-saturating)
# regime: at gain 16 the strongest pairs saturated on 2 seeds -> top-of-distribution ties -> near-median rank
# inversions (agreement 0.96-0.98); gain 12 keeps the median-P pair on the steep part of the f-I curve so the
# firing fraction stays rank-faithful to P. Together they lift the read's agreement with host P>=tau to ~1.0
# (recall == precision == 1.0 on ALL 6 seeds), so the SPIKING gate reproduces the host relation EXACTLY — parity
# and generation match host by construction. The `graded` soft-AND read was BUILT + MEASURED and is NOT adopted:
# on the sparse graph it rescues the wrong borderline pairs and FLOODS (advantage collapses on high-spread
# seeds), a documented honest-negative on that sub-lever. graded=False here (the hard median read, now
# high-fidelity). This is what the production wiring builds when BRAIN_SPIKING_PLAUSIBILITY is enabled.
PRODUCTION_READ_CONFIG = dict(pattern_size=16, n_ensemble=8, read_window=12, graded=False, density=0.0, gain=12.0)
# back-compat alias (older callers / the QUALIFIED de-risk import this name).
ENSEMBLE_GRADED_CONFIG = PRODUCTION_READ_CONFIG


def build_for_proposer(prop, seed=42, lesion=None, production=True, **kw):
    """Build a `SpikingAssociativePlausibilityOrgan` from a live `GenerativeReplayProposer` (reads its P/row).
    The vocab is the proposer's graph vocabulary (row keys). `production=True` (default) builds the robust
    ENSEMBLE + low-recurrence read (`PRODUCTION_READ_CONFIG`); pass production=False (or override the kwargs)
    for the QUALIFIED single-assembly hard read (the oracle/comparison baseline)."""
    vocab = sorted(prop.row.keys(), key=lambda w: prop.row[w])
    cfg = dict(PRODUCTION_READ_CONFIG) if production else {}
    cfg.update(kw)      # explicit kwargs win (lets a caller pin any single knob)
    return SpikingAssociativePlausibilityOrgan(prop.P, prop.row, vocab=vocab, seed=seed, lesion=lesion, **cfg)
