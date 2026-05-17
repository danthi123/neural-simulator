"""Increment G1 ignition + self-comprehension adapter.

The ONLY bridge between the pure SongHVC controller (sim/song_hvc.py +
research/runners/song_g1_core.py) and the UNCHANGED, multi-seed-validated
catalog G.20 sparse-distributed substrate (the 320-concept production tier:
5 sparse bridges x 64 concepts).

Three functions, all DRY (they reuse the validated G.20 surface; they do
NOT reimplement recall or sparse-pattern generation):

  load_members(seed)        -> build + load the 5 sparse 320-tier bridges,
                               copying the EXACT loader idiom of the
                               320-tier caller g20_xbridge_benchmark.main().
  ignite_sequence(...)      -> WRITE-ONLY drive of a concept's sparse
                               pattern into cp_external_input_current,
                               then advance the bridge. Mirrors the inner
                               drive idiom of shared_pool_chat.
                               stim_recall_sparse_rates. It registers NO
                               RegionPathway, adds NO feedback, calls NO
                               commit_engram_tag, and modifies NO weights.
                               This "write-only, no feedback" constraint
                               is load-bearing: the v12/v13/v15 dlpfc
                               failure was caused by a region that fed
                               activity back, which broke per-concept
                               selectivity.
  self_comprehend(...)      -> read which concept is firing in
                               shared_concept_pool using the SAME
                               validated per-sparse-pattern accumulation
                               loop as stim_recall_sparse_rates (read
                               cp_firing_states[pattern_arr].sum() over a
                               window). Returns the argmax concept + its
                               accumulated rate. That rate is a NO-DRIVE
                               residual readout (no external current is
                               written during decode) -- a DIFFERENT
                               magnitude regime from the continuous-drive
                               stim_recall_sparse_rates regime that the
                               abstention_gate 650 threshold was
                               calibrated on. It is therefore NOT a
                               drop-in for the literal 650; Task 9/10
                               must pre-register a regime-specific
                               abstention floor from a control
                               distribution measured in THIS exact
                               self_comprehend regime (see
                               self_comprehend's docstring).

Validation: import/signature smoke only (tests/test_song_g1_ignite_smoke.py).
The real validation is Task 8's no-harm probe + Task 10's gate -- NOT a
contrived orchestration unit test here. Heavy imports / IO are lazy (inside
the functions) so the smoke test stays instant and Task 8 can import
cheaply.

ASCII-only output.
"""
from __future__ import annotations

# --- 320-tier loader constants (source of truth: the 320-tier callers
#     g20_sparse_5bridge_chain_320.ps1 + g20_xbridge_benchmark.main()).
#     These are NOT heavy imports -- just literals -- so module-top is fine.
_BRIDGE_DIR = "research/findings/raw/g11_bg/g20_sparse_bridges_320"

# Per-bridge (name, checkpoint file). Order matches g20_vocab_spec_320
# ALL_BRIDGES_64 + the chain's foreach order.
_BRIDGE_SPEC = [
    ("bridgeA_nouns", "bridgeA_nouns_sparse64.simstate.h5"),
    ("bridgeB_verbs", "bridgeB_verbs_sparse64.simstate.h5"),
    ("bridgeC_adj", "bridgeC_adj_sparse64.simstate.h5"),
    ("bridgeD_spatial", "bridgeD_spatial_sparse64.simstate.h5"),
    ("bridgeE_functional", "bridgeE_functional_sparse64.simstate.h5"),
]

# Bridge kwargs copied VERBATIM from the 320-tier caller
# g20_xbridge_benchmark.main() (argparse defaults: --n-lang-input 8192,
# --n-shared-pool 2000, --sparsity 0.007, --pattern-size 100, --sparse).
# These are also exactly what g20_sparse_5bridge_chain_320.ps1 trained
# with (8192 / 2000 / pattern-size 100 / sparsity 0.007). The 160 tier
# uses sparsity 0.02; this is the 320 tier so 0.007 is correct.
_N_LANG_INPUT = 8192
_N_SHARED_POOL = 2000
# _SPARSITY is used only by encode_partial (which this adapter never
# calls); kept for loader-idiom fidelity with the 320-tier caller.
_SPARSITY = 0.007
_PATTERN_SIZE = 100


def load_members(seed: int = 42) -> list:
    """Build + load the 5 sparse G.20 320-tier bridges.

    Copies the loader idiom of the 320-tier caller
    g20_xbridge_benchmark.main() exactly (DRY):

        m = SharedPoolMember(
            bridge_path=bp, vocab=read_vocab_file(vp), name=nm,
            n_lang_input=8192, n_shared_pool=2000,
            sparsity=0.007, sparse=True, pattern_size=100)
        m.load(seed)

    The only deviation is the vocab source: instead of reading the
    g20_<name>_vocab64.txt files via read_vocab_file (which the chain
    writes), this uses g20_vocab_spec_320.ALL_BRIDGES_64 directly. Those
    are byte-identical (verified: spec == vocab64.txt == each bridge's
    trained .json sibling) and the spec module carries a global-
    uniqueness assert, making it the safer DRY source of truth.

    SharedPoolMember.load() -> build_sparse_pool_bridge(seed, ...) then
    load_checkpoint(path), and regenerates the per-concept sparse
    patterns deterministically from `seed` via generate_sparse_patterns
    (byte-identical to the patterns used at training time -- a drift
    would silently read the wrong neurons).

    Returns the list of 5 loaded SharedPoolMember instances (vocab order
    A nouns / B verbs / C adj / D spatial / E functional).
    """
    from pathlib import Path

    from research.runners.g20_multibridge import SharedPoolMember
    from research.runners.g20_vocab_spec_320 import ALL_BRIDGES_64

    members = []
    for name, ckpt_file in _BRIDGE_SPEC:
        bridge_path = str(Path(_BRIDGE_DIR) / ckpt_file)
        vocab = ALL_BRIDGES_64[name]
        m = SharedPoolMember(
            bridge_path=bridge_path,
            vocab=vocab,
            name=name,
            n_lang_input=_N_LANG_INPUT,
            n_shared_pool=_N_SHARED_POOL,
            sparsity=_SPARSITY,
            sparse=True,
            pattern_size=_PATTERN_SIZE,
        )
        m.load(seed)
        members.append(m)
    return members


def _pattern_global_arrs(member, cp):
    """Map every concept's pool-local sparse pattern to GLOBAL neuron
    indices in shared_concept_pool, exactly as
    shared_pool_chat.stim_recall_sparse_rates does:

        shared_indices = list(rm.indices("shared_concept_pool"))
        pattern_arrs = [cp.asarray([shared_indices[k] for k in pat], ...)
                        for pat in sparse_patterns]

    Reused by both ignite_sequence (drive) and self_comprehend (readout)
    so they index the IDENTICAL neurons.
    """
    rm = member.bridge.region_manager
    shared_indices = list(rm.indices("shared_concept_pool"))
    return [
        cp.asarray([shared_indices[k] for k in pat], dtype=cp.int64)
        for pat in member.sparse_patterns
    ]


def ignite_sequence(member, concept_indices, drive_pA: float = 1500.0,
                     steps_per: int = 100, recovery_steps: int = 20):
    """WRITE-ONLY ignition of a sequence of concepts.

    For each concept index in `concept_indices` (in order):
      1. Zero cp_external_input_current.
      2. Set cp_external_input_current to `drive_pA` at that concept's
         sparse-pattern GLOBAL indices in shared_concept_pool ONLY.
      3. Advance the bridge `steps_per` simulation steps.
      4. Zero cp_external_input_current and run `recovery_steps` free
         steps to let activity settle before the next concept.

    Mirrors the INNER drive idiom of shared_pool_chat.
    stim_recall_sparse_rates (which maps sparse_patterns -> global
    shared_concept_pool indices, writes cp_external_input_current there,
    steps the bridge, then clears). The difference: that helper drives a
    committed engram TAG via stimulate_tag; here we write the concept's
    raw sparse pattern directly (the ignition primitive the SongHVC
    controller needs).

    LOAD-BEARING CONSTRAINT -- this is strictly WRITE-ONLY:
      * writes ONLY cp_external_input_current at the concept pattern
        indices (never lang_input, never any other region);
      * registers NO RegionPathway;
      * adds NO feedback connection;
      * calls NO commit_engram_tag / start_engram_recording;
      * modifies NO weights / plasticity gates.
    (The v12/v13/v15 dlpfc failure: a region that fed activity back
    broke per-concept selectivity. The G.20 substrate stays UNCHANGED.)

    Returns the number of concepts ignited (len of the input order).
    """
    from sim.backend import get_backend
    cp, _ = get_backend()

    bridge = member.bridge
    pattern_arrs = _pattern_global_arrs(member, cp)
    n_concepts = len(pattern_arrs)

    order = list(concept_indices)
    for idx in order:
        if idx < 0 or idx >= n_concepts:
            raise IndexError(
                "concept index %d out of range [0, %d)"
                % (idx, n_concepts))
        parr = pattern_arrs[idx]
        # write-only: clear, then drive ONLY this concept's pattern
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[parr] = drive_pA
        for _ in range(steps_per):
            bridge._run_one_simulation_step()
        # zero external current + a few recovery steps before next slot
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(recovery_steps):
            bridge._run_one_simulation_step()
    return len(order)


def self_comprehend(member, decode_window: int = 100) -> list:
    """Read which concept is currently firing in shared_concept_pool.

    Uses the EXACT validated per-sparse-pattern accumulation loop from
    shared_pool_chat.stim_recall_sparse_rates -- for `decode_window`
    steps, advance the bridge and accumulate, per concept,
    cp_firing_states[pattern_arr].sum() into a per-concept rate vector
    (the same readout that reproduces the training-time discrimination).
    NO tag is stimulated and NO external current is written here.

    The returned rate is a per-concept accumulated-firing readout of
    the pool's RESIDUAL state AFTER the ordered production (no drive
    during decode). This no-drive residual is what carries the ORDER
    signal: a different ignition order leaves a different residual
    attractor in the pool; driving during decode would clamp the pool
    and DESTROY that order signal -- so the no-drive regime is correct
    and is deliberately NOT changed.

    Because nothing is driven during decode, the magnitude regime of
    this rate DIFFERS from stim_recall_sparse_rates' continuous-drive
    regime, which is the regime the abstention_gate 650 threshold was
    calibrated on (encoded ~796 / control ~584, AUC 0.990). Therefore
    Task 10 must NOT apply the literal 650 here: it must pre-register a
    regime-specific abstention threshold derived from a CONTROL
    distribution measured in THIS exact self_comprehend regime, using
    the same encoded-vs-control AUC methodology that produced 650 --
    decided BEFORE the true-order eval and never tuned afterward
    (anti-cheat: the pre-registered RULE, not the literal number, is
    the invariant).

    M3 -- INTEGRATED post-sequence decode (not per-slot): this decodes
    the integrated post-sequence pool state. Task 9/10 produce the FULL
    ordered sequence via ignite_sequence, THEN call self_comprehend ONCE
    on the integrated residual; order enters through the pool's
    sequence-dependent settling. Do NOT decode per-slot-then-average --
    that would erase the order signal.

    Returns a single-element list [(concept_idx, rate)] = the argmax
    concept index over the window + its accumulated firing count, so the
    caller treats it as the decode for the integrated post-sequence
    state. (List return shape keeps the door open for a top-k variant
    later without an API break.)
    """
    import numpy as np

    from sim.backend import get_backend
    cp, _ = get_backend()

    bridge = member.bridge
    pattern_arrs = _pattern_global_arrs(member, cp)

    # Identical accumulation loop to stim_recall_sparse_rates' inner
    # body (minus the stimulate_tag/clear_tag_drive that helper wraps it
    # in -- self-comprehension reads the NO-DRIVE residual pool state
    # left by the just-finished ordered production; see docstring re:
    # why this no-drive regime is correct + not 650-comparable).
    rates = np.zeros(len(pattern_arrs), dtype=np.float32)
    for _ in range(decode_window):
        bridge._run_one_simulation_step()
        for j, parr in enumerate(pattern_arrs):
            firing = bridge.cp_firing_states[parr]
            s = firing.sum() if hasattr(firing, 'sum') else 0
            if hasattr(s, 'item'):
                s = s.item()
            rates[j] += float(s)

    if len(rates) == 0:
        return []
    best_idx = int(np.argmax(rates))
    return [(best_idx, float(rates[best_idx]))]


def ignite_prediction(member, concept_idx, drive_pA: float = 1500.0,
                       steps_per: int = 100):
    """Generator-P write-only top-down-prediction ignition (Task 7a).

    The P (PredictiveCoder, sim/predictive_coding.py) layer's ONLY
    write into concept pools: deliver a single top-down PREDICTED
    concept into shared_concept_pool by driving that concept's sparse
    pattern, using the EXACT same allowed-write surface as
    ignite_sequence's inner per-concept drive. P predicts the next
    concept (active inference); this function realizes that prediction
    as a (P-T) top-down prediction current on the predicted concept's
    pattern -- nothing else.

    Thin reuse, NOT a reimplementation: this delegates to
    ignite_sequence([concept_idx], drive_pA=..., steps_per=...,
    recovery_steps=0). That means the concept->global-neuron mapping
    is _pattern_global_arrs (the IDENTICAL mapping ignite_sequence /
    self_comprehend / ignite_and_trajectory_decode index), and the
    ONLY bridge state write is cp_external_input_current (drive at the
    predicted concept's sparse pattern + zeroing). recovery_steps=0
    keeps a single prediction-step atomic (the P rollout caller, Task
    8, sequences predictions slot-by-slot and owns inter-slot
    settling, exactly as it does for ignite_and_trajectory_decode);
    no behavior of ignite_sequence is changed (default recovery_steps
    is unaffected for its existing callers).

    LOAD-BEARING CONSTRAINT -- strictly WRITE-ONLY, identical
    guarantee to ignite_sequence (inherited verbatim by delegation):
      * the ONLY bridge state write is cp_external_input_current
        (drive at the predicted concept's pattern + zeroing); never
        lang_input, never any other region;
      * registers NO RegionPathway; adds NO feedback connection;
      * calls NO commit_engram_tag / start_engram_recording /
        stimulate_tag;
      * modifies NO weights / plasticity gates / tags.
    (The v12/v13/v15/G1 "first, do no harm" lesson: a region that fed
    activity back broke per-concept selectivity. P feeds in a
    SPECIFIC predicted-concept pattern via the validated write-only
    surface only -- never non-specific feedback. The G.20 substrate
    stays UNCHANGED; the no-harm probe re-proves this empirically.)

    Range-checks concept_idx (raises IndexError, same as
    ignite_sequence / ignite_and_trajectory_decode).

    Task 8/10 NOTE (for the gate author -- do NOT act on it here):
    PredictiveCoder.select_next tie-breaks to the FIRST candidate on
    logit ties (deterministic, NOT uniform-random). The Task 8/10
    gate's per-candidate lists MUST NOT be target-ordered, else a
    degenerate "always pick the first candidate" rollout could score
    above chance without learning. The pre-registered permuted-ORDER
    control + frozen abstention floor are designed to catch this --
    but the gate author must keep candidate ordering target-agnostic
    explicitly.

    Returns the number of predicted concepts ignited (always 1).
    """
    return ignite_sequence(
        member,
        [concept_idx],
        drive_pA=drive_pA,
        steps_per=steps_per,
        recovery_steps=0,
    )


def ignite_and_trajectory_decode(member, concept_indices,
                                  drive_pA: float = 1500.0,
                                  steps_per: int = 100,
                                  gap_steps: int = 20,
                                  decode_steps: int = 30) -> tuple:
    """G1.5 order-sensitive trajectory readout (write-only).

    For each concept index in `concept_indices` IN ORDER, this produces
    one ordered decode slot:

      1. WRITE-ONLY drive: clear cp_external_input_current, then set it
         to `drive_pA` at that concept's sparse-pattern GLOBAL indices
         in shared_concept_pool ONLY (the EXACT inner drive idiom of
         ignite_sequence -- same _pattern_global_arrs mapping, same
         backend). Advance the bridge `steps_per` steps.
      2. Zero cp_external_input_current, then free-run `gap_steps`
         UN-DRIVEN steps. This gap is LOAD-BEARING: it prevents the
         trivial "read back the concept you just drove" circularity, so
         the subsequent read reflects the pool's order-dependent
         integration / RESPONSE, not the clamped input. `gap_steps`
         MUST be > 0 (enforced).
      3. Over `decode_steps` further UN-DRIVEN steps, accumulate
         cp_firing_states[pattern_arr].sum() per concept pattern using
         the SAME validated per-sparse-pattern accumulation loop as
         self_comprehend / stim_recall_sparse_rates. argmax over the
         per-concept accumulated vector is `decoded[t]` -- the concept
         the pool expresses after integrating slots 0..t (order-
         dependent); its accumulated rate is `rate[t]`.

    Unlike self_comprehend (a single argmax of the final integrated
    residual, length-1, 0.5-capped under score_order), this returns an
    ORDERED length-N decoded list so song_g1_core.score_order can reach
    1.0 and reflect ORDER. A scrambled ignition order yields a different
    per-slot trajectory.

    LOAD-BEARING CONSTRAINT -- strictly WRITE-ONLY, identical guarantee
    to ignite_sequence / self_comprehend:
      * the ONLY bridge state write is cp_external_input_current (drive
        at concept patterns + zeroing); never lang_input, never any
        other region;
      * cp_firing_states is READ only;
      * registers NO RegionPathway; adds NO feedback connection;
      * calls NO commit_engram_tag / start_engram_recording /
        stimulate_tag;
      * modifies NO weights / plasticity gates / tags.
    (The v12/v13/v15 dlpfc failure: a region that fed activity back
    broke per-concept selectivity. The G.20 substrate stays UNCHANGED.)

    DRY: reuses _pattern_global_arrs (the same pool-local -> global
    mapping ignite_sequence and self_comprehend index), and the same
    validated accumulation loop as self_comprehend. Heavy imports are
    lazy so the module top stays import-light.

    Returns (decoded_list, rates_list) where both have length ==
    len(concept_indices); decoded_list[t] is an int concept index and
    rates_list[t] is a float accumulated rate.
    """
    if gap_steps <= 0:
        raise ValueError(
            "gap_steps must be > 0 (the un-driven gap is load-bearing: "
            "it prevents the read-back-what-you-drove circularity); got "
            "%d" % gap_steps)

    import numpy as np

    from sim.backend import get_backend
    cp, _ = get_backend()

    bridge = member.bridge
    pattern_arrs = _pattern_global_arrs(member, cp)
    n_concepts = len(pattern_arrs)

    order = list(concept_indices)
    decoded_list = []
    rates_list = []
    for idx in order:
        if idx < 0 or idx >= n_concepts:
            raise IndexError(
                "concept index %d out of range [0, %d)"
                % (idx, n_concepts))
        parr = pattern_arrs[idx]
        # (1) write-only drive of THIS concept's pattern (ignite_sequence
        #     inner drive idiom -- same allowed write surface)
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[parr] = drive_pA
        for _ in range(steps_per):
            bridge._run_one_simulation_step()
        # (2) un-driven gap: zero current, free-run so the read reflects
        #     the pool's RESPONSE, not the driven input (gap_steps > 0)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(gap_steps):
            bridge._run_one_simulation_step()
        # (3) un-driven decode: SAME accumulation loop as self_comprehend
        #     (cp_firing_states READ only; no external current written)
        rates = np.zeros(n_concepts, dtype=np.float32)
        for _ in range(decode_steps):
            bridge._run_one_simulation_step()
            for j, pj in enumerate(pattern_arrs):
                firing = bridge.cp_firing_states[pj]
                s = firing.sum() if hasattr(firing, 'sum') else 0
                if hasattr(s, 'item'):
                    s = s.item()
                rates[j] += float(s)
        best_idx = int(np.argmax(rates))
        decoded_list.append(best_idx)
        rates_list.append(float(rates[best_idx]))

    # optional brief settle for cleanliness (still write-only: zero only)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(min(gap_steps, 5)):
        bridge._run_one_simulation_step()

    return (decoded_list, rates_list)
