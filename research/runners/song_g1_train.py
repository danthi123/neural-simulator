"""Increment G1 Task 9 -- the self-supervised SONGBIRD training loop.

The songbird babble -> self-comprehend -> dopamine-reinforce loop
(Fee & Goldberg 2011) on the UNMODIFIED, multi-seed-validated catalog
G.20 sparse-distributed substrate. Kill-safe + resumable (Inc-3
sim/train_checkpoint atomic pattern: re-running the script resumes;
a kill mid-write never corrupts the checkpoint).

This trainer is PURELY ADDITIVE and reuses (DRY) -- it does NOT
reimplement chain/scoring/ignition/checkpoint/loading:

  * sim.song_hvc.SongHVC                  -- the pure synfire chain
                                             (rollout / babble /
                                             reinforce / intention bias)
  * research.runners.song_g1_core         -- score_order / compose_reward
                                             (the permuted-ORDER anti-cheat
                                             scoring; gate semantics)
  * research.runners.song_g1_ignite       -- load_members / ignite_sequence
                                             (WRITE-ONLY) / self_comprehend
                                             (INTEGRATED post-sequence
                                             decode, M3)
  * research.runners.g20_xbridge_benchmark.sample_xbridge_pairs
                                          -- the VALIDATED deterministic
                                             pair sampler (seed 42,
                                             exclude_idx=12) -- the same
                                             idiom the no-harm probe /
                                             xbridge benchmark use
  * sim.train_checkpoint                  -- atomic kill-safe checkpoint

THREE LOAD-BEARING PRE-REGISTRATION CONSTRAINTS (violating any one
invalidates the G1 result -- see the implementation plan
"Pre-registration corrections 2/3/4" + Task 9):

(M3) INTEGRATED, NOT per-slot, self-comprehension. For each babble
     candidate ORDERED sequence we ignite the WHOLE ordered production
     via ignite_sequence(...) ONCE, THEN call self_comprehend(...) ONCE
     on the integrated post-sequence residual. We NEVER decode per-slot
     then average -- that erases the order signal. The order enters the
     readout through the pool's sequence-dependent settling (different
     ignition ORDER -> different residual attractor). self_comprehend
     returns a single integrated decode [(concept_idx, rate)]; that one
     concept is THE decode for the whole ordered production, and
     score_order([decoded_idx], intended) scores ordered match against
     the intended position sequence.

(corr 2) CONTROL-CALIBRATED abstention floor, NOT the literal 650. The
     literal 650 was calibrated on stim_recall_sparse_rates'
     CONTINUOUS-DRIVE regime; self_comprehend reads a NO-DRIVE
     integrated residual (a different magnitude regime), so 650 is NOT
     comparable here. At TRAIN START (Step 0, ONCE) we measure the
     self_comprehend integrated-residual top-rate distribution for
     (i) intended-order productions of the TRAIN propositions [proxy
     "encoded"] and (ii) a CONTROL set [random/unencoded concept
     sequences AND permuted-order productions], in the IDENTICAL
     self_comprehend regime, and set g1_abstain = the encoded-vs-control
     separation point at the SAME operating criterion the original 650
     used (control-max: the max control top-rate -- the 650 doc set the
     gate just above control-max ~584; AUC also recorded for
     transparency). Computed ONCE, recorded in the checkpoint + a
     sidecar JSON, and NEVER tuned during training. gate_cleared for
     compose_reward = (integrated decode top-rate >= this provisional
     g1_abstain).

(frozen) The TRAIN propositions are a SMALL fixed set (4-6) derived
     deterministically from the validated sample_xbridge_pairs(...,
     seed=42, exclude_idx=12) sampler and FROZEN. A DISJOINT held-out
     set is reserved for Task 10 (NEVER trained here). Both sets +
     the Step-0 g1_abstain are persisted (sidecar JSON next to the
     checkpoint) so a resumed run AND Task 10 use the SAME values.

G1.5 --readout MODE (additive; default 'final' = G1 byte-identical):

  --readout final (DEFAULT) keeps the EXACT G1 behavior: each babble
  candidate is decoded via the M3 _integrated_decode path (ignite the
  WHOLE ordered sequence ONCE, self_comprehend ONCE on the integrated
  residual -> a length-1 decode). Nothing about the G1 negative's
  reproducibility changes when --readout is absent or 'final'.

  --readout trajectory (G1.5) decodes via
  song_g1_ignite.ignite_and_trajectory_decode: per concept slot, a
  write-only ignition -> brief UN-DRIVEN gap -> argmax read, returning
  an ORDERED length-N decoded list + a per-slot rate list. This lets
  song_g1_core.score_order reach 1.0 and reflect ORDER (vs G1's
  length-1 0.5-capped readout). The "top rate" used for gate_cleared
  is the trajectory-rate aggregate `traj_top_rate(rates_list)` = the
  MIN per-slot accumulated rate (sidecar key traj_rate_rule="min"):
  the production is only "confident" if EVERY slot cleared the
  abstention floor (matches compose_reward's no-confabulation moat:
  "any produced slot below the abstention gate -> 0.0"). This ONE rule
  is used IDENTICALLY in Step-0 calibration, training, and the gate
  (recorded in the sidecar so the gate uses the same).

  The trajectory regime is a DIFFERENT readout magnitude regime, so
  Step 0 RE-CALIBRATES the control-max abstention floor IN the
  trajectory regime (same AUC/control-max methodology that produced
  G1's 72.0, re-derived here -- NOT 72.0, NOT the literal 650).

GENERATOR-P --mode MODE (additive; default 'songbird' = G1/G1.5
byte-identical; Task 8 of the Generator-P plan):

  --mode songbird (DEFAULT) keeps the ENTIRE existing path BYTE-
  IDENTICAL: both --readout final AND --readout trajectory keep
  working exactly as before, so G1 (the recorded negative) and G1.5
  stay reproducible. --mode only diverges behavior when 'p'.

  --mode p (Generator-P) changes ONLY how the ordered concept sequence
  is CHOSEN: by a net-new pure-numpy PredictiveCoder
  (sim/predictive_coding.py) active-inference rollout -- predict the
  next concept given the prefix-so-far (the order-sensitive learning
  signal the recognition-only G.20 substrate provably does NOT
  provide) -- instead of the songbird synfire chain. Each predicted
  concept is REALIZED into shared_concept_pool via the EXISTING
  write-only P top-down (P-T) prediction surface
  song_g1_ignite.ignite_prediction (the ONLY P write into concept
  pools; NO RegionPathway, NO weight/tag mutation, NO non-specific
  feedback -- the v12/v13/v15/G1 "first, do no harm" lesson; the
  no-harm probe re-proves this). The gate-cleared "top rate" is the
  SAME comprehension readout songbird/final uses: the EXISTING
  integrated self_comprehend NO-DRIVE residual top-rate. So the gate
  uses the SAME readout machinery; the P difference is ONLY the
  ordered-sequence CHOICE. Training (Task 9) self-supervises the
  PredictiveCoder on EACH prefix->next of the frozen TRAIN
  propositions' intended order (PredictiveCoder.learn; the substrate
  is UNCHANGED -- learning touches ONLY the small P weights).

  LOAD-BEARING ANTI-CHEAT (carry-forward Minor #4):
  PredictiveCoder.select_next tie-breaks to the FIRST candidate on
  logit ties. The candidate list passed to select_next at each rollout
  slot is the FIXED canonical vocab-index order _canonical_candidates(
  len(member.vocab)) = range(n) -- INDEPENDENT of the intended/target
  sequence -- so a degenerate untrained "always pick the first
  candidate" rollout can NEVER correlate with the targets. This same
  fixed ordering is used IDENTICALLY in Step-0 calibration, training,
  and the gate.

  The P regime is a DIFFERENT regime than final/trajectory, so Step 0
  RE-CALIBRATES the control-max abstention floor IN the P regime (same
  AUC/control-max methodology that produced G1's 72.0, re-derived
  here -- NOT 72.0, NOT the trajectory regime's 46.0, NOT the literal
  650). Step-0 measures the SUBSTRATE's integrated response to a
  realized ORDER (intended vs permuted vs random) via ignite_prediction
  + self_comprehend -- it does NOT use PredictiveCoder.select_next
  (exactly as final/trajectory Step-0 do for their paths), so the rate
  aggregate is IDENTICAL across Step-0, training, and the gate.

  PRECEDENCE: --mode p SUPERSEDES --readout for BOTH the ckpt-path
  infix ('.pc' first, NOT '.traj'; then '.smoke') AND the decode path
  (the P decode supersedes the readout dispatch). The sidecar records
  "mode" (top-level + mirrored in calibration{}); the gate REFUSES a
  sidecar whose mode != its --mode (same HARD-refusal class as the
  smoke/readout cross-mode refusal). The pure g1_verdict /
  score_order / permuted_order_controls and the FIXED bars are
  UNMODIFIED; the P-regime floor is sidecar-frozen ONLY (never
  recomputed at gate time, never 650).

SMOKE / FULL + MODE/READOUT CKPT NAMESPACE ISOLATION (pre-launch
defense; the mode + readout layers reuse the EXACT smoke idioms, DRY):

  (a-traj) When --readout trajectory AND --ckpt was NOT explicitly
      overridden, the path is redirected to an ISOLATED namespace with
      a ".traj" infix (song_g1.traj.ckpt.npz / its sidecar) so the
      trajectory-regime frozen floor + weights NEVER collide with G1's
      canonical song_g1.ckpt.npz (the recorded G1 negative). This
      composes with (a): --smoke --readout trajectory with a default
      --ckpt -> song_g1.traj.smoke.ckpt.npz. An explicitly-passed
      --ckpt is honored verbatim (defense (b) still guards it).

  (a) A default --smoke run (one whose --ckpt was NOT explicitly
      overridden) is redirected to an ISOLATED checkpoint+sidecar with
      a ".smoke" infix (song_g1.smoke.ckpt.npz / its sidecar). So a
      default smoke run NEVER writes the canonical song_g1.ckpt.npz
      that the multi-hour full run + Task 10 consume. The default
      --ckpt CONSTANT is unchanged; only smoke-with-default-path is
      redirected. If --ckpt is passed explicitly with --smoke, that
      path is honored (defense (b) still applies).

  (b) The sidecar records "smoke": bool AND "readout": str. On
      resume/reuse, if an existing checkpoint/sidecar's smoke flag
      differs from the current run's OR its recorded readout differs
      from this run's --readout, its calibration.g1_abstain is NOT
      reused AND SongHVC.W / epoch are NOT resumed -- the run starts
      fresh (recompute Step 0, fresh init, epoch 0) with an explicit
      [fresh] warning. This guarantees a full run / Task 10 can never
      inherit a smoke-calibrated OR cross-readout-regime abstention
      floor or weights even if paths collide (e.g. an explicit shared
      --ckpt across modes). A SAME-mode resume (smoke->smoke or
      full->full, SAME readout) is unchanged: W + rng + loss + frozen
      props + calibration are reused exactly. The sidecar "smoke" /
      "readout" keys are additive JSON; Task 10 reads the sidecar and
      will see smoke:false + the matching readout for the real run.

WHY single-bridge propositions: ignite_sequence / self_comprehend
operate on ONE SharedPoolMember's bridge using THAT member's concept
indices (its sparse_patterns). A proposition is therefore an ORDERED
concept-index sequence WITHIN a single chosen bridge -- selected via
the SAME validated deterministic sample_xbridge_pairs idiom (seed 42,
exclude_idx=12) but constrained to one bridge so the single-member
ignite/decode contract + the M3 integrated decode hold. The "A rel B"
ORDER is exactly what the songbird chain must learn; the permuted-ORDER
control (Task 10) has the same concept multiset, order scrambled.

song_hvc ONLY writes drive (via ignite_sequence's WRITE-ONLY path); it
NEVER adds a feedback pathway into concept pools (the documented
v12/v13/v15 dlpfc failure mode). Task 8's no-harm probe already proved
the silent controller does not regress the validated path; this trainer
keeps ignition strictly write-only.

Heavy imports / IO are lazy (inside main): loading 5 sparse 320 bridges
is slow + GPU-bound (several minutes expected, even for --smoke).

ASCII-only output (Windows cp1252 safe).
"""
from __future__ import annotations

import argparse

# --- Bridge each robust word lives in is irrelevant to the loop; what
#     matters is that ignite/decode run on ONE member. Loader constants
#     live in song_g1_ignite (DRY) -- not duplicated here. ------------

# Default checkpoint + sidecar (sidecar persists frozen props +
# Step-0 g1_abstain so resume + Task 10 reuse the SAME values).
_CKPT_DEFAULT = "research/findings/raw/g11_bg/song_g1.ckpt.npz"

# SongHVC chain geometry. n_concepts MUST be >= the per-bridge vocab
# size (64 for the 320 tier) so any concept index is representable;
# n_states >= the longest proposition length. (Same shape the no-harm
# probe constructed: SongHVC(8, 64, seed=42).)
_SONG_N_STATES = 8
_SONG_N_CONCEPTS = 64

# Deterministic proposition derivation (the VALIDATED idiom):
#   sample_xbridge_pairs([m.vocab ...], n_pairs=_PAIR_POOL, seed=42,
#                         exclude_idx=12)
# -> deterministic cross-bridge (bi, word_a, bj, word_b) pairs. We then
# build ONE-bridge ORDERED concept-index propositions from the prefix
# (see _build_frozen_propositions) and split:
#   first _N_TRAIN              -> TRAIN (reinforced)
#   next  _N_HELDOUT            -> HELD-OUT (reserved for Task 10,
#                                  NEVER trained here)
_PAIR_SEED = 42
_EXCLUDE_IDX = 12
_PAIR_POOL = 60          # sampler prefix (superset; strictly det.)
_N_TRAIN = 4             # 4 frozen TRAIN propositions
_N_HELDOUT = 2           # 2 disjoint HELD-OUT (Task 10 only)
_PROP_LEN = 2            # ordered 2-concept "A rel B" propositions

# Step-0 control-calibration sample sizes (self_comprehend regime).
_CTRL_N_RANDOM = 6       # random/unencoded concept sequences
# (permuted-order productions of the TRAIN props are ALSO added to the
#  control distribution -- the permuted-ORDER control is load-bearing.)

# Decode / drive windows for ignite_sequence + self_comprehend. These
# mirror the validated stim_recall_sparse_rates magnitudes (drive_pA
# 1500, 100-step windows) the comprehension path uses.
_DRIVE_PA = 1500.0
_STEPS_PER = 100
_IGNITE_RECOVERY = 20    # ignite_sequence's own inter-slot settle
_DECODE_WINDOW = 100


def traj_top_rate(rates_list) -> float:
    """PURE: the G1.5 trajectory-rate aggregate (no IO, deterministic).

    The single, pre-registered `traj_rate_rule` = "min": the trajectory
    "top rate" used for `gate_cleared` is the MINIMUM per-slot
    accumulated rate over the ordered decode. Rationale: the production
    is only "confident" (non-confabulated) if EVERY slot cleared the
    abstention floor -- this mirrors compose_reward's no-confabulation
    moat ("any produced slot below the abstention gate -> 0.0"). The
    MIN aggregate enforces "every slot >= floor" with a single scalar
    comparison against the frozen g1_abstain.

    This ONE rule MUST be used IDENTICALLY in Step-0 calibration,
    training, and the gate (it is recorded in the sidecar as
    traj_rate_rule so the gate uses the same). Empty list -> 0.0.
    """
    if not rates_list:
        return 0.0
    return float(min(float(r) for r in rates_list))


# The pre-registered trajectory-rate aggregate name, persisted to the
# sidecar so train + gate provably use the IDENTICAL rule (anti-cheat:
# the gate is invalid if its aggregate differs from Step-0's).
_TRAJ_RATE_RULE = "min"


# --- Generator-P (--mode p) constants --------------------------------
# The PredictiveCoder recurrent-prefix state width (Task 8 binding spec:
# "reasonable default e.g. 64"). Pure-numpy P core; the substrate is
# UNCHANGED -- P only writes the predicted concept's pattern via the
# write-only song_g1_ignite.ignite_prediction surface.
_P_STATE_DIM = 64
# Drive window for the per-slot P top-down prediction current (the SAME
# stim_recall magnitudes the comprehension path / ignite_sequence use --
# ignite_prediction delegates to ignite_sequence's inner per-slot drive
# with recovery_steps=0; the P rollout caller owns inter-slot settling).
_P_DRIVE_PA = 1500.0
_P_STEPS_PER = 100


def _canonical_candidates(n_concepts: int) -> list:
    """PURE: the FIXED canonical candidate ordering for the P
    active-inference rollout = list(range(n_concepts)) -- the natural
    per-bridge vocab index order, INDEPENDENT of any target/intended
    sequence (no IO, deterministic).

    LOAD-BEARING ANTI-CHEAT (carry-forward Minor #4; see
    song_g1_ignite.ignite_prediction's "Task 8/10 NOTE"):
    PredictiveCoder.select_next tie-breaks to the FIRST candidate on
    logit ties. If the candidate list passed to select_next were
    target-ordered (the intended/target sequence), a degenerate
    untrained "always pick the first candidate" rollout could correlate
    with the targets and score above chance WITHOUT any learning. This
    builder therefore returns range(n)-style fixed order that can NEVER
    be the target order, so a degenerate rollout is target-agnostic.
    Used IDENTICALLY by Step-0 calibration, training, and the gate.
    """
    return list(range(int(n_concepts)))


def _build_frozen_propositions(members):
    """Derive the FROZEN train + held-out propositions deterministically.

    Reuses the VALIDATED sample_xbridge_pairs sampler (seed 42,
    exclude_idx=12) -- the exact idiom the no-harm probe / xbridge
    benchmark use -- then constrains each to a SINGLE bridge so the
    single-member ignite_sequence/self_comprehend contract + the M3
    integrated decode hold.

    For each sampled cross-bridge pair (bi, word_a, bj, word_b) we take
    the home bridge `bi` and form a length-2 ORDERED concept-index
    proposition [idx(word_a), idx(word_b_in_bridge_bi)] where the SECOND
    concept is a deterministic distinct concept index in the SAME bridge
    (so the whole ordered production lives in one pool). word_b's own
    bridge is irrelevant to ignite/decode; we only borrow the sampler's
    determinism to pick the pair, then keep both concepts in `bi`.

    The second concept index = (idx(word_a) + 1 + (pair_rank %
    (vocab-2))) wrapped into the bridge, skipping _EXCLUDE_IDX and
    idx(word_a) -- a pure deterministic function of the sampler output
    (no RNG, no data peeking). Returns a list of dicts:
        {"intention": int, "bridge": name, "bridge_idx": int,
         "concept_seq": [i, j], "words": [wi, wj], "kind": "train"|"heldout"}
    The first _N_TRAIN are train, the next _N_HELDOUT are held-out
    (DISJOINT). Frozen the moment this runs (deterministic in members'
    vocab order + seed 42); persisted to the sidecar so resume + Task 10
    reuse the identical set.
    """
    from research.runners.g20_xbridge_benchmark import sample_xbridge_pairs

    vocabs = [m.vocab for m in members]
    pairs = sample_xbridge_pairs(
        vocabs, n_pairs=_PAIR_POOL, seed=_PAIR_SEED,
        exclude_idx=_EXCLUDE_IDX)

    props = []
    seen_keys = set()
    need = _N_TRAIN + _N_HELDOUT
    for rank, (bi, wa, _bj, _wb) in enumerate(pairs):
        if len(props) >= need:
            break
        member = members[bi]
        vocab = member.vocab
        n_v = len(vocab)
        i = member.word_to_idx[wa]
        # deterministic distinct second concept index in the SAME
        # bridge: walk forward from i+1, skipping i and _EXCLUDE_IDX,
        # offset by the pair rank so different pairs pick different j.
        offset = 1 + (rank % max(1, n_v - 2))
        j = i
        steps = 0
        while steps < 2 * n_v:
            j = (i + offset + steps) % n_v
            if j != i and j != _EXCLUDE_IDX:
                break
            steps += 1
        if j == i or j == _EXCLUDE_IDX:
            continue
        key = (bi, i, j)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        kind = "train" if len(props) < _N_TRAIN else "heldout"
        props.append({
            "intention": len(props),
            "bridge": member.name,
            "bridge_idx": bi,
            "concept_seq": [int(i), int(j)],
            "words": [vocab[i], vocab[j]],
            "kind": kind,
        })
    return props


def _seed_intention_biases(song, props):
    """Seed each proposition's intended ORDERED concept sequence as the
    SongHVC intention bias so rollout(intention, len) initially returns
    the intended order (the babble loop then explores around it and
    DA-reinforces matches). Pure controller call (no bridge)."""
    for p in props:
        song.set_intention_bias(p["intention"], p["concept_seq"])


def _integrated_decode(member, ignite_sequence, self_comprehend,
                        concept_seq, drive_pA, steps_per, decode_window):
    """M3: ignite the WHOLE ordered sequence ONCE, THEN self_comprehend
    ONCE on the integrated post-sequence residual. Returns
    (decoded_list, top_rate). decoded_list is [concept_idx] (a single
    integrated decode -- NOT a per-slot list); order entered via the
    pool's sequence-dependent settling. NEVER per-slot-then-average."""
    ignite_sequence(member, concept_seq, drive_pA=drive_pA,
                     steps_per=steps_per,
                     recovery_steps=_IGNITE_RECOVERY)
    dec = self_comprehend(member, decode_window=decode_window)
    if not dec:
        return [], 0.0
    idx, rate = dec[0]
    return [int(idx)], float(rate)


def _decode_candidate(readout, member, ignite_sequence, self_comprehend,
                       trajectory_decode, concept_seq, drive_pA,
                       steps_per, decode_window):
    """Readout-mode dispatch for ONE ordered candidate. Returns
    (decoded_list, top_rate) in BOTH modes so the caller (Step-0
    calibration, training, gate) is readout-agnostic and the SAME rule
    flows everywhere.

      readout == "final"      -> the EXACT G1 path: _integrated_decode
          (ignite the WHOLE ordered sequence ONCE, self_comprehend ONCE
          on the integrated residual). decoded_list is length-1; the
          top_rate is that single integrated residual rate. BYTE-
          IDENTICAL to G1 -- this branch is the default and does not
          change the recorded G1 negative's reproducibility.

      readout == "trajectory" -> ignite_and_trajectory_decode: an
          ORDERED length-N decoded list + per-slot rates. top_rate =
          traj_top_rate(rates_list) (the pre-registered MIN-per-slot
          `traj_rate_rule`), used IDENTICALLY for gate_cleared in
          Step-0, training, and the gate.
    """
    if readout == "trajectory":
        decoded_list, rates_list = trajectory_decode(
            member, concept_seq, drive_pA=drive_pA,
            steps_per=steps_per)
        return [int(x) for x in decoded_list], traj_top_rate(rates_list)
    # default: byte-identical G1 M3 integrated decode (length-1).
    return _integrated_decode(
        member, ignite_sequence, self_comprehend,
        concept_seq, drive_pA, steps_per, decode_window)


def _decode_candidate_p(pc, member, ignite_prediction, self_comprehend,
                         intended, drive_pA, steps_per, decode_window):
    """Generator-P (--mode p) decode for ONE proposition. Returns
    (produced_list, top_rate) -- SAME shape as _decode_candidate so
    Step-0 calibration, training, and the gate stay decode-agnostic and
    the SAME rate aggregate flows everywhere.

    The P difference is ONLY how the ordered concept sequence is CHOSEN:
    by the PredictiveCoder active-inference rollout (predict next concept
    given the prefix-so-far) instead of the songbird synfire chain. The
    gate-cleared "top rate" is obtained via the EXISTING `self_comprehend`
    INTEGRATED readout on the produced sequence -- IDENTICAL to G1's M3
    readout regime (so the gate uses the SAME comprehension readout as
    songbird/final). Per slot t, for length == len(intended):

      1. c = pc.select_next(_canonical_candidates(<#concepts in this
         bridge's vocab>))  -- LOAD-BEARING: the candidate list is the
         FIXED canonical vocab-index order, NOT target-ordered (so a
         degenerate untrained "pick the first candidate" rollout cannot
         correlate with the intended sequence; see _canonical_candidates
         + ignite_prediction's "Task 8/10 NOTE").
      2. ignite_prediction(member, c, ...) -- the write-only top-down
         (P-T) prediction current (the ONLY P write into concept pools;
         reuses the validated ignite_sequence inner per-slot drive,
         recovery_steps=0; the rollout owns inter-slot settling).
      3. pc.update_state(c) -- feed the realized prediction back into
         the recurrent prefix (order-dependent).

    After the produced sequence is realized into the pool, the
    gate-cleared top rate is the EXISTING self_comprehend integrated
    NO-DRIVE residual readout (the SAME machinery final/G1 uses for the
    rate -- only the ORDERED-sequence CHOICE differs in P mode).
    score_order(produced, intended) (UNMODIFIED song_g1_core) scores it
    at the call site, NOT here.
    """
    n_concepts = len(member.vocab)
    candidates = _canonical_candidates(n_concepts)
    length = len(intended)
    pc.reset(intention=list(intended))
    produced = []
    for _t in range(length):
        # active inference: emit the concept the top-down generative
        # model most predicts given the prefix-so-far. `candidates` is
        # the FIXED canonical vocab-index order (range(n)) -- NOT the
        # intended/target order -- so PredictiveCoder.select_next's
        # first-candidate tie-break can never correlate with the target
        # for an untrained/degenerate predictor (anti-cheat Minor #4).
        c = int(pc.select_next(candidates))
        produced.append(c)
        # write-only top-down (P-T) prediction current (the ONLY P
        # write into concept pools; recovery_steps=0 inside, the rollout
        # owns inter-slot settling exactly as the trajectory path does).
        ignite_prediction(member, c, drive_pA=drive_pA,
                           steps_per=steps_per)
        # feed the realized prediction back into the recurrent prefix.
        pc.update_state(c)
    # gate-cleared rate via the EXISTING integrated self_comprehend
    # NO-DRIVE residual readout (IDENTICAL to G1/final's rate machinery;
    # only the ordered-sequence CHOICE differs in P mode).
    dec = self_comprehend(member, decode_window=decode_window)
    top_rate = float(dec[0][1]) if dec else 0.0
    return [int(x) for x in produced], top_rate


def _realize_order_p(member, ignite_prediction, self_comprehend,
                       concept_seq, drive_pA, steps_per, decode_window):
    """Step-0/control realization for --mode p: realize a GIVEN ordered
    concept sequence into the pool via the SAME write-only P top-down
    (P-T) prediction surface (ignite_prediction per slot), THEN the
    EXISTING integrated self_comprehend NO-DRIVE residual readout.
    Returns (decoded_list, top_rate) -- SAME shape + SAME rate aggregate
    as _decode_candidate_p / _decode_candidate.

    This is _decode_candidate_p WITHOUT the PredictiveCoder.select_next
    step: Step-0 calibration must measure the SUBSTRATE's integrated
    response to a SPECIFIC realized ORDER (intended vs permuted vs
    random) -- exactly as final/trajectory Step-0 does for their decode
    paths -- so the floor does NOT depend on an (untrained) predictor's
    choices. The rate aggregate (integrated self_comprehend top-rate
    after realizing the order through ignite_prediction) is therefore
    IDENTICAL across Step-0, training, and the gate in the P regime
    (anti-cheat: the gate is invalid if its rate aggregate differs from
    Step-0's). decoded_list is the integrated self_comprehend argmax
    (length-1, the same M3 integrated-residual readout final/G1 uses).
    """
    n_concepts = len(member.vocab)
    for c in concept_seq:
        ci = int(c)
        if ci < 0 or ci >= n_concepts:
            raise IndexError(
                "concept idx %d out of [0,%d)" % (ci, n_concepts))
        ignite_prediction(member, ci, drive_pA=drive_pA,
                           steps_per=steps_per)
    dec = self_comprehend(member, decode_window=decode_window)
    if not dec:
        return [], 0.0
    idx, rate = dec[0]
    return [int(idx)], float(rate)


def _recover(member, recover_steps):
    """Inter-production free-run so adaptation/STP recovers between
    ignitions (the documented Stage-1 inter-turn remedy; the no-harm
    probe relied on this too). Pure free steps -- no drive, no decode,
    no weight change."""
    if recover_steps <= 0:
        return
    bridge = member.bridge
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(recover_steps):
        bridge._run_one_simulation_step()


def _step0_calibrate(members, props_train, ignite_sequence,
                      self_comprehend, rng, drive_pA, steps_per,
                      decode_window, recover_steps, readout="final",
                      trajectory_decode=None, mode="songbird",
                      ignite_prediction=None):
    """Pre-registered Step 0 (ONCE at train start; NOT 650).

    Measure the decode TOP-RATE distribution in the IDENTICAL regime
    (the SAME `mode` + `readout` + `traj_rate_rule` aggregate the run
    uses) for:
      (i)  ENCODED proxy : intended-order productions of the TRAIN
           propositions.
      (ii) CONTROL       : random/unencoded concept sequences AND
           permuted-ORDER productions of the TRAIN propositions
           (the permuted-ORDER control is load-bearing).

    g1_abstain = the encoded-vs-control separation at the SAME operating
    criterion the original 650 used: control-max (the 650 doc set the
    gate just above control-max ~584). A simple AUC (P[encoded_rate >
    control_rate]) is ALSO recorded for transparency, matching the
    encoded-vs-control AUC methodology that produced 650 -- but the
    OPERATING POINT is control-max, fixed here, never tuned.

    G1.5: when readout=="trajectory" the encoded/control TOP-RATE is
    the trajectory-regime traj_top_rate(rates_list) (MIN per slot),
    NOT G1's final-residual rate -- a DIFFERENT magnitude regime, so
    the floor is RE-DERIVED here (same control-max methodology that
    produced G1's 72.0; the result is NOT 72.0, NOT 650). readout=="
    final" is byte-identical to G1's Step 0.

    Generator-P: when mode=="p" the encoded/control TOP-RATE is the P
    regime -- a GIVEN ordered concept sequence (intended / permuted /
    random) is realized into the pool via the write-only P top-down
    (P-T) prediction surface (ignite_prediction per slot), THEN the
    EXISTING integrated self_comprehend NO-DRIVE residual readout (the
    SAME comprehension readout final/G1 uses for the rate; only the
    ORDER-realization surface differs). This is a DIFFERENT regime than
    final/trajectory, so the control-max floor is RE-DERIVED here with
    the SAME control-max methodology that produced G1's 72.0 -- the
    result is NOT 72.0, NOT 650, NOT the trajectory regime's 46.0. The
    rate aggregate is IDENTICAL across Step-0, training, and the gate in
    the P regime (anti-cheat: the gate is invalid if its rate aggregate
    differs from Step-0's). Step-0 does NOT use PredictiveCoder.
    select_next (it measures the SUBSTRATE's response to a realized
    ORDER, exactly as final/trajectory Step-0 do for their paths).

    Returns a calibration dict (also written to the sidecar)."""
    encoded_rates = []
    control_rates = []

    def _decode_seq(member, concept_seq):
        """Mode/readout-agnostic Step-0 decode of a GIVEN ordered
        sequence -> (decoded_list, top_rate). P mode realizes the order
        through the write-only P-T prediction surface; songbird mode
        uses the existing final/trajectory _decode_candidate dispatch.
        The SAME function is used for ENCODED and ALL controls so the
        rate aggregate is identical everywhere."""
        if mode == "p":
            return _realize_order_p(
                member, ignite_prediction, self_comprehend,
                concept_seq, drive_pA, steps_per, decode_window)
        return _decode_candidate(
            readout, member, ignite_sequence, self_comprehend,
            trajectory_decode, concept_seq, drive_pA, steps_per,
            decode_window)

    # (i) ENCODED proxy: intended order of each TRAIN proposition.
    for p in props_train:
        member = members[p["bridge_idx"]]
        _dec, rate = _decode_seq(member, p["concept_seq"])
        encoded_rates.append(rate)
        _recover(member, recover_steps)

    # (ii-a) CONTROL: permuted-ORDER productions of the TRAIN props
    #        (same concept multiset, order scrambled). Load-bearing.
    from research.runners.song_g1_core import permuted_order_controls
    for p in props_train:
        member = members[p["bridge_idx"]]
        perms = permuted_order_controls(p["concept_seq"], rng, n=2)
        for perm in perms:
            _dec, rate = _decode_seq(member, perm)
            control_rates.append(rate)
            _recover(member, recover_steps)

    # (ii-b) CONTROL: EXACTLY _CTRL_N_RANDOM random/unencoded ordered
    #        concept-index sequences (FIX 2). The count is decoupled
    #        from len(props_train) (was props_train[:_CTRL_N_RANDOM],
    #        which capped at 4 full / 2 smoke -- making the named
    #        constant unreachable and feeding the smoke/full divergence).
    #        Each draw picks a bridge from the TRAIN propositions'
    #        bridges (regime-matched) via the DEDICATED calibration rng
    #        (does NOT touch the training rng stream; deterministic
    #        given --seed). Same in-regime random/unencoded controls
    #        fed through the SAME _integrated_decode path.
    train_bridge_idxs = [p["bridge_idx"] for p in props_train]
    for _k in range(_CTRL_N_RANDOM):
        bidx = train_bridge_idxs[
            int(rng.integers(0, len(train_bridge_idxs)))]
        member = members[bidx]
        n_v = len(member.vocab)
        a = int(rng.integers(0, n_v))
        b = int(rng.integers(0, n_v))
        if b == a:
            b = (a + 1) % n_v
        _dec, rate = _decode_seq(member, [a, b])
        control_rates.append(rate)
        _recover(member, recover_steps)

    enc = sorted(encoded_rates)
    ctl = sorted(control_rates)
    control_max = max(ctl) if ctl else 0.0
    control_mean = (sum(ctl) / len(ctl)) if ctl else 0.0
    encoded_mean = (sum(enc) / len(enc)) if enc else 0.0
    # operating criterion = control-max (exactly what produced 650:
    # gate set just above control-max). g1_abstain is that point.
    g1_abstain = float(control_max)
    # transparency-only AUC = P[encoded > control] over the cross-pairs.
    if enc and ctl:
        wins = sum(1 for e in enc for c in ctl if e > c)
        ties = sum(1 for e in enc for c in ctl if e == c)
        auc = (wins + 0.5 * ties) / float(len(enc) * len(ctl))
    else:
        auc = 0.0
    return {
        "g1_abstain": g1_abstain,
        "operating_criterion": "control_max",
        # G1.5/Generator-P: which mode + readout regime this floor was
        # calibrated IN, and the trajectory-rate aggregate used. The
        # gate REFUSES a sidecar whose mode != its --mode OR (in
        # songbird mode) whose readout != its --readout, and uses the
        # SAME traj_rate_rule (anti-cheat: identical rule everywhere).
        "mode": str(mode),
        "readout": str(readout),
        "traj_rate_rule": (_TRAJ_RATE_RULE
                           if (mode != "p" and readout == "trajectory")
                           else None),
        "encoded_rates": [round(x, 2) for x in enc],
        "control_rates": [round(x, 2) for x in ctl],
        "encoded_mean": round(encoded_mean, 2),
        "control_mean": round(control_mean, 2),
        "control_max": round(control_max, 2),
        "auc_encoded_vs_control": round(auc, 4),
        "n_encoded": len(enc),
        "n_control": len(ctl),
        "note": (
            ("Generator-P regime: a GIVEN ordered concept sequence "
             "realized via the write-only P top-down (P-T) prediction "
             "surface (ignite_prediction per slot) THEN the EXISTING "
             "integrated self_comprehend NO-DRIVE residual readout. "
             "RE-CALIBRATED in THIS regime -- NOT G1's final 72.0, NOT "
             "the trajectory regime's 46.0, NOT the literal 650. "
             "control-max operating point, fixed at train start, never "
             "tuned (anti-cheat: pre-registered RULE).")
            if mode == "p" else
            ("trajectory-regime per-slot decode, traj_rate_rule='%s' "
             "(MIN per slot); RE-CALIBRATED in THIS regime -- NOT G1's "
             "72.0, NOT 650. control-max operating point, fixed at "
             "train start, never tuned (anti-cheat: pre-registered "
             "RULE)." % _TRAJ_RATE_RULE)
            if readout == "trajectory" else
            ("self_comprehend NO-DRIVE integrated-residual regime; "
             "NOT the literal 650 (continuous-drive regime). "
             "control-max operating point, fixed at train start, "
             "never tuned (anti-cheat: pre-registered RULE).")),
    }


def _smoke_ckpt_path(ckpt_path):
    """Isolated smoke checkpoint path = the base with a '.smoke' infix
    before the '.ckpt.npz' (or '.npz') suffix. Pure string transform
    (deterministic; no IO). Used ONLY when --smoke is set and --ckpt
    was left at its default so a default smoke run can never write the
    canonical full-run checkpoint."""
    for suffix in (".ckpt.npz", ".npz"):
        if ckpt_path.endswith(suffix):
            return ckpt_path[: -len(suffix)] + ".smoke" + suffix
    return ckpt_path + ".smoke"


def _traj_ckpt_path(ckpt_path):
    """Isolated trajectory-regime checkpoint path = the base with a
    '.traj' infix before the '.ckpt.npz' (or '.npz') suffix. Pure
    string transform (deterministic; no IO -- the EXACT idiom as
    _smoke_ckpt_path with a '.traj' tag instead of '.smoke').

    Used ONLY when --readout trajectory AND --ckpt was left at its
    default so a trajectory run can NEVER write the canonical
    song_g1.ckpt.npz (G1's recorded NEGATIVE) or its sidecar/floor.
    Composes with _smoke_ckpt_path: applying this first then the smoke
    transform yields '<base>.traj.smoke.ckpt.npz' (the trajectory smoke
    namespace) -- both layers reuse the same pure-string-transform
    machinery (DRY)."""
    for suffix in (".ckpt.npz", ".npz"):
        if ckpt_path.endswith(suffix):
            return ckpt_path[: -len(suffix)] + ".traj" + suffix
    return ckpt_path + ".traj"


def _p_ckpt_path(ckpt_path):
    """Isolated Generator-P (--mode p) checkpoint path = the base with a
    '.pc' infix before the '.ckpt.npz' (or '.npz') suffix. Pure string
    transform (deterministic; no IO -- the EXACT idiom as
    _smoke_ckpt_path / _traj_ckpt_path with a '.pc' tag).

    Used ONLY when --mode p AND --ckpt was left at its default so a P
    run can NEVER write the canonical song_g1.ckpt.npz (G1's recorded
    NEGATIVE) or the trajectory regime's song_g1.traj.ckpt.npz (G1.5's
    recorded result) or their sidecars/floors.

    PRECEDENCE (documented): P mode SUPERSEDES readout for the path
    infix. The caller applies '.pc' FIRST (NOT '.traj'), then '.smoke'
    if smoke -- so --mode p -> song_g1.pc.ckpt.npz and
    --mode p --smoke -> song_g1.pc.smoke.ckpt.npz, regardless of
    --readout (P mode's decode path supersedes readout; the readout
    flag does not change the P-regime frozen floor). Both layers reuse
    the same pure-string-transform machinery (DRY)."""
    for suffix in (".ckpt.npz", ".npz"):
        if ckpt_path.endswith(suffix):
            return ckpt_path[: -len(suffix)] + ".pc" + suffix
    return ckpt_path + ".pc"


def _sidecar_mode(meta):
    """PURE: the mode a sidecar was calibrated IN (no IO).

    Single source of truth = meta["calibration"]["mode"] (where
    _step0_calibrate writes it); falls back to top-level meta["mode"]
    (the trainer mirrors it there too), then to "songbird" -- legacy
    G1/G1.5 sidecars predate the mode key and are the songbird regime
    (the trainer's own additive-JSON contract, exactly mirroring
    _sidecar_readout's legacy-absent -> "final"). `meta` must be a dict
    (the caller validates non-dict first)."""
    calib = meta.get("calibration")
    if isinstance(calib, dict) and calib.get("mode") is not None:
        return str(calib["mode"])
    if meta.get("mode") is not None:
        return str(meta["mode"])
    return "songbird"


def _sidecar_path(ckpt_path):
    return ckpt_path + ".meta.json"


def _load_sidecar(ckpt_path):
    import json
    import os
    sp = _sidecar_path(ckpt_path)
    if not os.path.exists(sp):
        return None
    with open(sp, "r") as f:
        return json.load(f)


def _save_sidecar(ckpt_path, meta):
    """Persist frozen props + Step-0 g1_abstain atomically (same
    kill-safe .tmp + os.replace idiom as sim/train_checkpoint)."""
    import json
    import os
    sp = _sidecar_path(ckpt_path)
    os.makedirs(os.path.dirname(sp) or ".", exist_ok=True)
    tmp = sp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, sp)


def main():
    p = argparse.ArgumentParser(
        description="Increment G1 Task 9 self-supervised songbird "
                    "babble->self-comprehend->DA training loop "
                    "(kill-safe resumable).")
    p.add_argument("--epochs", type=int, default=60,
                   help="total epochs (resume-aware: counts from 0; a "
                        "checkpoint at epoch k resumes at k+1).")
    p.add_argument("--n-babble", type=int, default=8,
                   help="k babble candidates per proposition per epoch "
                        "(LMAN variability).")
    p.add_argument("--temperature0", type=float, default=0.5,
                   help="initial babble temperature (decays linearly to "
                        "~0 over epochs).")
    p.add_argument("--lr", type=float, default=0.5,
                   help="SongHVC.reinforce learning rate (DA-gated "
                        "three-factor; reward<=0 is a no-op by design).")
    p.add_argument("--recover-steps", type=int, default=200,
                   help="inter-production free-run steps so "
                        "adaptation/STP recovers (documented Stage-1 "
                        "inter-turn remedy).")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for bridge load + the training RNG "
                        "(babble/control sampling).")
    p.add_argument("--ckpt", type=str, default=_CKPT_DEFAULT,
                   help="checkpoint .npz path (sidecar .meta.json holds "
                        "frozen props + Step-0 g1_abstain + 'smoke' + "
                        "'readout' + 'mode' flags). NAMESPACE ISOLATION "
                        "when --ckpt is left at its default: --mode p "
                        "redirects to an isolated '.pc' namespace "
                        "(song_g1.pc.ckpt.npz); --readout trajectory "
                        "redirects to '.traj'; --smoke adds a '.smoke' "
                        "infix. PRECEDENCE: --mode p SUPERSEDES "
                        "--readout for the path infix -- the order is "
                        "'.pc' FIRST (NOT '.traj'), then '.smoke', so "
                        "--mode p -> song_g1.pc.ckpt.npz and "
                        "--mode p --smoke -> song_g1.pc.smoke.ckpt.npz "
                        "regardless of --readout. So a P run NEVER "
                        "writes G1's canonical song_g1.ckpt.npz (the "
                        "recorded G1 negative) or the trajectory "
                        "regime's song_g1.traj.ckpt.npz; a default "
                        "songbird/trajectory smoke run NEVER writes the "
                        "canonical full-run checkpoint. An explicit "
                        "--ckpt is honored verbatim (the cross-mode "
                        "refusal still guards it).")
    p.add_argument("--mode", type=str, default="songbird",
                   choices=("songbird", "p"),
                   help="generation regime. 'songbird' (DEFAULT) = the "
                        "ENTIRE existing path BYTE-IDENTICAL (both "
                        "--readout final AND --readout trajectory keep "
                        "working exactly as before; G1/G1.5 stay "
                        "reproducible). 'p' (Generator-P) = the ordered "
                        "concept sequence is CHOSEN by a net-new "
                        "PredictiveCoder active-inference rollout "
                        "(predict next concept given the prefix-so-far) "
                        "instead of the songbird synfire chain; each "
                        "predicted concept is realized via the "
                        "write-only P top-down (P-T) prediction surface "
                        "(song_g1_ignite.ignite_prediction) and the "
                        "gate-cleared rate is the EXISTING integrated "
                        "self_comprehend readout (SAME comprehension "
                        "readout as songbird). P mode uses an ISOLATED "
                        "'.pc' ckpt/sidecar namespace + its OWN Step-0 "
                        "control-max floor RE-CALIBRATED in the P regime "
                        "(NOT G1's final 72.0, NOT the trajectory "
                        "regime's 46.0, NOT the literal 650); the gate "
                        "refuses a cross-mode sidecar. P mode SUPERSEDES "
                        "--readout for the ckpt path infix.")
    p.add_argument("--readout", type=str, default="final",
                   choices=("final", "trajectory"),
                   help="decode regime (songbird mode only -- in "
                        "--mode p the P decode path supersedes this and "
                        "the '.pc' namespace is used regardless). "
                        "'final' (DEFAULT) = the EXACT G1 M3 integrated "
                        "decode (length-1, byte-identical to the "
                        "recorded G1 negative -- its reproducibility is "
                        "unchanged when this flag is absent). "
                        "'trajectory' (G1.5) = per-slot ignite -> "
                        "un-driven gap -> argmax read, an ORDERED "
                        "length-N decode (score_order can reach 1.0 / "
                        "reflect ORDER); gate_cleared uses the "
                        "MIN-per-slot traj_rate_rule. The trajectory "
                        "regime uses an ISOLATED '.traj' ckpt/sidecar "
                        "namespace and its OWN Step-0 control-max floor "
                        "RE-CALIBRATED in that regime (NOT G1's 72.0, "
                        "NOT 650); the gate refuses a cross-readout "
                        "sidecar.")
    p.add_argument("--smoke", action="store_true",
                   help="tiny build/kill-safe-resume validation: 2 "
                        "epochs, 2 train props, k=2 babbles. NOT the "
                        "G1 result (Task 10 is the gate). Uses an "
                        "ISOLATED ckpt/sidecar namespace (default path "
                        "gets a '.smoke' infix, composing with the "
                        "'.pc' infix when --mode p or the '.traj' infix "
                        "when --readout trajectory in songbird mode); a "
                        "sidecar from the OTHER mode (different smoke "
                        "flag OR different readout OR different --mode) "
                        "is refused (no full verdict can be gated on a "
                        "smoke-calibrated / cross-readout / cross-mode "
                        "floor or weights).")
    args = p.parse_args()

    # --- lazy/heavy imports (loading 5 sparse bridges is slow + GPU) -
    import time

    import numpy as np

    from research.runners.song_g1_core import compose_reward
    from research.runners.song_g1_ignite import (
        ignite_and_trajectory_decode, ignite_prediction, ignite_sequence,
        load_members, self_comprehend,
    )
    from sim.predictive_coding import PredictiveCoder
    from sim.song_hvc import SongHVC
    from sim.train_checkpoint import (
        load_checkpoint, resume_epoch, save_checkpoint,
    )

    t0 = time.time()
    smoke = bool(args.smoke)
    readout = str(args.readout)
    mode = str(args.mode)

    # --- isolated ckpt namespace when --ckpt not overridden (DRY: the
    #     SAME pure-string-transform idiom for the '.pc', '.traj', and
    #     '.smoke' layers; they COMPOSE with a documented PRECEDENCE):
    #
    #   FIX 1(a-pc): if --mode p AND --ckpt is default, redirect to a
    #     distinct '.pc' ckpt+sidecar so a Generator-P run can NEVER
    #     write G1's canonical song_g1.ckpt.npz (the recorded G1
    #     NEGATIVE) or the trajectory regime's song_g1.traj.ckpt.npz
    #     (G1.5's recorded result) or their frozen floors.
    #   FIX 1(a-traj): if --readout trajectory AND --ckpt is default,
    #     redirect to a distinct '.traj' ckpt+sidecar so a trajectory
    #     run can NEVER write G1's canonical song_g1.ckpt.npz.
    #   FIX 1(a): if --smoke AND --ckpt is default, redirect to a
    #     distinct '.smoke' ckpt+sidecar so a smoke run NEVER writes
    #     the canonical full-run checkpoint.
    #
    #   PRECEDENCE (documented): --mode p SUPERSEDES --readout for the
    #   path infix. When mode==p we apply '.pc' and SKIP '.traj'
    #   entirely (the P decode path supersedes readout; the readout
    #   flag does not change the P-regime frozen floor), then '.smoke'
    #   if smoke. So --mode p -> song_g1.pc.ckpt.npz; --mode p --smoke
    #   -> song_g1.pc.smoke.ckpt.npz; --mode p --readout trajectory ->
    #   song_g1.pc.ckpt.npz (NOT '.pc.traj'). In songbird mode the
    #   pre-existing '.traj'/'.smoke' composition is UNCHANGED (so
    #   --smoke --readout trajectory -> song_g1.traj.smoke.ckpt.npz,
    #   byte-identical to before). An explicitly-passed --ckpt is
    #   honored verbatim (defense (b)'s cross-mode refusal -- smoke OR
    #   readout OR mode mismatch -- still guards it).
    ckpt_explicit = (args.ckpt != _CKPT_DEFAULT)
    if not ckpt_explicit:
        isolated = _CKPT_DEFAULT
        if mode == "p":
            # P mode supersedes readout: '.pc' FIRST, NOT '.traj'.
            isolated = _p_ckpt_path(isolated)
        elif readout == "trajectory":
            isolated = _traj_ckpt_path(isolated)
        if smoke:
            isolated = _smoke_ckpt_path(isolated)
        if isolated != _CKPT_DEFAULT:
            args.ckpt = isolated
            tags = []
            if mode == "p":
                tags.append("mode=p ('.pc'; supersedes --readout)")
            elif readout == "trajectory":
                tags.append("readout=trajectory ('.traj')")
            if smoke:
                tags.append("smoke ('.smoke')")
            print(f"[CKPT-ISOLATION] --ckpt left at default; "
                  f"redirecting {' + '.join(tags)} run to isolated "
                  f"namespace:\n  ckpt    = {args.ckpt}"
                  f"\n  sidecar = {_sidecar_path(args.ckpt)}\n  (the "
                  f"canonical {_CKPT_DEFAULT} -- the recorded G1 "
                  f"negative -- is UNTOUCHED)", flush=True)

    n_epochs = 2 if smoke else int(args.epochs)
    n_babble = 2 if smoke else int(args.n_babble)
    n_train_cap = 2 if smoke else _N_TRAIN

    print("=" * 64, flush=True)
    print("SONG G1 TASK 9 -- SELF-SUPERVISED SONGBIRD TRAINING LOOP",
          flush=True)
    print("(babble -> self-comprehend -> DA-reinforce; kill-safe "
          "resumable)", flush=True)
    if smoke:
        print("[SMOKE] tiny build/kill-safe-resume validation -- NOT "
              "the G1 result (Task 10 is the gate)", flush=True)
    print("=" * 64, flush=True)

    # --- load the 5 validated G.20 320-sparse bridges (DRY) ----------
    print(f"Loading 5 sparse 320-tier bridges via "
          f"song_g1_ignite.load_members(seed={args.seed}) ...",
          flush=True)
    members = load_members(seed=int(args.seed))
    print(f"  loaded: {[m.name for m in members]} "
          f"({int(time.time()-t0)}s)", flush=True)

    # --- frozen propositions (deterministic; reuse validated sampler) -
    all_props = _build_frozen_propositions(members)
    props_train_full = [p for p in all_props if p["kind"] == "train"]
    props_heldout = [p for p in all_props if p["kind"] == "heldout"]
    # smoke trains only the first 2 train props. The sidecar still
    # records the SAME full train/held-out lists, BUT a smoke run
    # writes to an ISOLATED namespace (FIX 1(a): default --ckpt gets a
    # '.smoke' infix) and a cross-mode sidecar is REFUSED (FIX 1(b)),
    # so a non-smoke resume / Task 10 NEVER inherits the smoke sidecar
    # or its smoke-calibrated g1_abstain -- they recompute Step 0 fresh
    # on the canonical namespace and freeze the canonical set there.
    props_train = props_train_full[:n_train_cap]

    print(f"\nFrozen propositions (deterministic via "
          f"sample_xbridge_pairs seed={_PAIR_SEED} "
          f"exclude_idx={_EXCLUDE_IDX}):", flush=True)
    print(f"  TRAIN ({len(props_train_full)}; this run trains "
          f"{len(props_train)}):", flush=True)
    for p in props_train_full:
        mark = "*" if p in props_train else " "
        print(f"   [{mark}] int={p['intention']} {p['bridge']} "
              f"seq={p['concept_seq']} words={p['words']}", flush=True)
    print(f"  HELD-OUT ({len(props_heldout)}; RESERVED for Task 10, "
          f"NEVER trained here):", flush=True)
    for p in props_heldout:
        print(f"       int={p['intention']} {p['bridge']} "
              f"seq={p['concept_seq']} words={p['words']}", flush=True)

    # --- SongHVC controller (pure; write-only via ignite_sequence) ---
    #     Always constructed (songbird mode uses it; P mode keeps it
    #     allocated but does NOT train/checkpoint it -- the P decode
    #     path supersedes the synfire chain). The intention biases are
    #     harmless in P mode (rollout via PredictiveCoder, not song).
    song = SongHVC(n_states=_SONG_N_STATES,
                   n_concepts=_SONG_N_CONCEPTS,
                   seed=int(args.seed))
    _seed_intention_biases(song, all_props)

    # --- Generator-P controller (pure-numpy PredictiveCoder; the ONLY
    #     learned machinery in --mode p; the G.20 substrate is
    #     UNCHANGED -- P only writes via the write-only ignite_prediction
    #     surface). n_concepts == _SONG_N_CONCEPTS (64 = the per-bridge
    #     320-tier vocab size, exactly as SongHVC; every 320-tier bridge
    #     is 64-concept so _canonical_candidates(len(member.vocab)) ==
    #     range(64) for every bridge). Seed = the run seed (deterministic
    #     init; checkpointed/resumed in P mode). In songbird mode it is
    #     constructed but UNUSED (no behavior change -- byte-identical).
    pc = PredictiveCoder(n_concepts=_SONG_N_CONCEPTS,
                         state_dim=_P_STATE_DIM,
                         seed=int(args.seed))

    # --- FIX 1(b): cross-mode reuse refusal -------------------------
    #     Load the sidecar FIRST so the cross-mode decision is made
    #     atomically for BOTH W/epoch resume AND calibration reuse. The
    #     run REFUSES to inherit anything from a namespace whose regime
    #     differs -- regime = (smoke flag, readout, mode). Mismatch iff
    #     the existing sidecar's "smoke" flag differs from this run's
    #     smoke OR its recorded "mode" differs from this run's --mode
    #     OR its recorded "readout" differs from this run's --readout.
    #     A cross-mode (songbird vs p) OR cross-readout floor is a HARD
    #     refusal class (a P-regime control-max floor must NEVER gate a
    #     songbird run or vice versa -- a DIFFERENT decode magnitude
    #     regime; same class as the smoke-tag rejection). An absent
    #     "smoke" key -> full=False; an absent "readout" key -> "final";
    #     an absent "mode" key -> "songbird" (the trainer's own
    #     additive-JSON contract: legacy G1/G1.5 sidecars predate the
    #     mode key and are songbird). Conservatively, ANY recorded
    #     (smoke, readout, mode) != current forces a fresh start
    #     (recompute Step 0, fresh init, epoch 0) -- no smoke-calibrated
    #     / cross-readout / cross-mode floor or weights can ever gate a
    #     full verdict, even on a colliding explicit --ckpt.
    meta = _load_sidecar(args.ckpt)
    cross_mode_mismatch = False
    if meta is not None:
        sidecar_smoke = bool(meta.get("smoke", False))
        sidecar_readout = str(meta.get("readout", "final"))
        # mode mirrored top-level + inside calibration{} (the single
        # source the gate's _check_sidecar_usable reads); legacy/absent
        # -> "songbird" (the additive-JSON back-compat contract so
        # existing G1/G1.5 sidecars stay usable by songbird-mode runs).
        sidecar_mode = _sidecar_mode(meta)
        if sidecar_smoke != smoke:
            cross_mode_mismatch = True
            print(f"\n[fresh] existing ckpt mode={sidecar_mode} "
                  f"readout={sidecar_readout} smoke={sidecar_smoke} "
                  f"but this run smoke={smoke}; ignoring it (will not "
                  f"gate a full verdict on a smoke-calibrated floor)",
                  flush=True)
        elif sidecar_mode != mode:
            cross_mode_mismatch = True
            print(f"\n[fresh] existing ckpt mode={sidecar_mode} but "
                  f"this run mode={mode}; ignoring it (a {sidecar_mode}"
                  f"-regime control-max floor must NEVER gate a {mode} "
                  f"run -- different decode regime)", flush=True)
        elif sidecar_readout != readout:
            cross_mode_mismatch = True
            print(f"\n[fresh] existing ckpt readout={sidecar_readout} "
                  f"but this run readout={readout}; ignoring it (a "
                  f"{sidecar_readout}-regime control-max floor must "
                  f"NEVER gate a {readout} run -- different magnitude "
                  f"regime)", flush=True)

    # --- resume (kill-safe Inc-3 pattern) ----------------------------
    ckpt = load_checkpoint(args.ckpt)
    rng = np.random.default_rng(int(args.seed))
    loss_history = []
    if ckpt is not None and not cross_mode_mismatch:
        # SAME-mode resume: restore controller weights + RNG state +
        # loss history exactly as before. songbird mode restores
        # SongHVC.W from weights[0] (BYTE-IDENTICAL to before). P mode
        # restores the PredictiveCoder W_in/W_pred/b_pred (the trained
        # P weights -- NOT re-randomized; the exact kill-safe resume
        # idiom, just a different weights list shape per mode).
        start_epoch = resume_epoch(ckpt)
        if mode == "p":
            pc.W_in = np.asarray(ckpt["weights"][0],
                                 dtype=np.float32).copy()
            pc.W_pred = np.asarray(ckpt["weights"][1],
                                   dtype=np.float32).copy()
            pc.b_pred = np.asarray(ckpt["weights"][2],
                                   dtype=np.float32).copy()
        else:
            song.W = np.asarray(ckpt["weights"][0],
                                 dtype=np.float32).copy()
        rng.bit_generator.state = ckpt["rng_state"]
        loss_history = list(ckpt["loss_history"])
        print(f"\n[RESUME] checkpoint at epoch {ckpt['epoch']} -> "
              f"resuming at epoch {start_epoch} (mode={mode}, "
              f"loss_history len={len(loss_history)})", flush=True)
    elif ckpt is not None and cross_mode_mismatch:
        # CROSS-mode collision: a checkpoint exists but it belongs to
        # the other mode. Do NOT resume W/epoch from it -- start fresh
        # (epoch 0, fresh SongHVC init, fresh rng) so a full/Task-10
        # verdict can never inherit smoke-trained weights.
        start_epoch = 0
        print(f"\n[FRESH] checkpoint at {args.ckpt} is the OTHER mode "
              f"-> NOT resuming its weights/epoch; starting fresh at "
              f"epoch 0", flush=True)
    else:
        start_epoch = resume_epoch(ckpt)
        print(f"\n[FRESH] no checkpoint at {args.ckpt} -> starting "
              f"at epoch 0", flush=True)

    # --- sidecar: frozen props + Step-0 g1_abstain (persist ONCE;
    #     SAME-mode resume + Task 10 MUST reuse the SAME values; a
    #     cross-mode sidecar is REFUSED -> Step 0 recomputed fresh) ---
    if (meta is not None and not cross_mode_mismatch
            and "g1_abstain" in meta.get("calibration", {})):
        calib = meta["calibration"]
        g1_abstain = float(calib["g1_abstain"])
        _calib_mode = _sidecar_mode(meta)
        _calib_readout = str(calib.get("readout", "final"))
        print(f"[SIDECAR] reusing pre-registered Step-0 g1_abstain="
              f"{g1_abstain:.2f} (control-max, mode={_calib_mode}"
              f"{', readout=' + _calib_readout if _calib_mode != 'p' else ''} "
              f"regime, computed at train start; NEVER retuned)",
              flush=True)
    else:
        # Step 0 -- run ONCE at train start (NOT 650; NOT G1's 72.0
        # when readout=trajectory -- a DIFFERENT magnitude regime, so
        # the control-max floor is RE-DERIVED here). Uses a dedicated
        # RNG so the calibration is independent of (and does not
        # consume) the training rng stream. Decodes via the SAME
        # readout path + SAME traj_rate_rule the run trains/gates with.
        print(f"\n[STEP 0] PRE-REGISTERED control-calibrated "
              f"abstention floor (mode={mode}"
              f"{', readout=' + readout if mode != 'p' else ''} "
              f"regime; NOT literal 650"
              f"{'; NOT G1 final-regime 72.0' if (mode == 'p' or readout == 'trajectory') else ''}"
              f"{'; NOT trajectory-regime 46.0' if mode == 'p' else ''}"
              f") ...", flush=True)
        calib_rng = np.random.default_rng(int(args.seed) * 7 + 1)
        calib = _step0_calibrate(
            members, props_train, ignite_sequence, self_comprehend,
            calib_rng, _DRIVE_PA, _STEPS_PER, _DECODE_WINDOW,
            int(args.recover_steps), readout=readout,
            trajectory_decode=ignite_and_trajectory_decode,
            mode=mode, ignite_prediction=ignite_prediction)
        g1_abstain = float(calib["g1_abstain"])
        meta = {
            "task": "song_g1 Task 9 self-supervised training",
            "substrate": "G.20 320-sparse 5-bridge",
            "seed": int(args.seed),
            # FIX 1(b): record the regime so a resume/Task-10 read can
            # REFUSE a cross-mode sidecar. Additive JSON keys (Task 10
            # tolerates them; it will see smoke:false + the matching
            # readout + mode for the real run). 'readout' AND 'mode'
            # are ALSO mirrored inside calibration{} by _step0_calibrate
            # (the single source the gate's _check_sidecar_usable /
            # _sidecar_mode / _sidecar_readout read).
            "smoke": bool(args.smoke),
            "mode": str(mode),
            "readout": str(readout),
            "traj_rate_rule": (_TRAJ_RATE_RULE
                               if (mode != "p"
                                   and readout == "trajectory")
                               else None),
            "pair_sampler": {
                "fn": "sample_xbridge_pairs",
                "seed": _PAIR_SEED,
                "exclude_idx": _EXCLUDE_IDX,
                "n_pairs_prefix": _PAIR_POOL,
            },
            "prop_len": _PROP_LEN,
            "train_propositions": props_train_full,
            "heldout_propositions": props_heldout,
            "calibration": calib,
        }
        _save_sidecar(args.ckpt, meta)
        print(f"  encoded mean={calib['encoded_mean']:.1f} "
              f"(n={calib['n_encoded']})  "
              f"control mean={calib['control_mean']:.1f} "
              f"max={calib['control_max']:.1f} "
              f"(n={calib['n_control']})", flush=True)
        print(f"  AUC(encoded>control)={calib['auc_encoded_vs_control']}"
              f"  (transparency only; operating point=control-max)",
              flush=True)
        print(f"  -> PRE-REGISTERED g1_abstain = {g1_abstain:.2f} "
              f"(control-max; NOT 650; frozen, never retuned)",
              flush=True)
        print(f"  [sidecar saved] {_sidecar_path(args.ckpt)}",
              flush=True)

    if start_epoch >= n_epochs:
        print(f"\n[DONE] start_epoch {start_epoch} >= target "
              f"{n_epochs} -- nothing to do (already complete). "
              f"g1_abstain={g1_abstain:.2f}", flush=True)
        print("=" * 64, flush=True)
        return 0

    # --- training loop (kill-safe; KeyboardInterrupt -> ckpt+exit) ---
    print(f"\n[TRAIN] epochs {start_epoch}..{n_epochs-1}  "
          f"props={len(props_train)}  k={n_babble}  lr={args.lr}  "
          f"recover={args.recover_steps}  g1_abstain="
          f"{g1_abstain:.2f}", flush=True)

    # FIX 3: track the last-completed epoch in-process so the
    # KeyboardInterrupt handler does NOT re-read the npz it just wrote
    # (cosmetic disk read). Initialized to one-before-start so a kill
    # before any epoch completes reports the correct resume baseline
    # (start_epoch - 1 == the epoch already on disk, or -1 if fresh).
    last_completed_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, n_epochs):
            ep_t0 = time.time()
            # linear temperature decay temperature0 -> ~0.
            if n_epochs > 1:
                frac = epoch / float(max(1, n_epochs - 1))
            else:
                frac = 0.0
            temperature = float(args.temperature0) * (1.0 - frac)

            rewards_this_epoch = []
            n_gate_cleared = 0

            for p in props_train:
                intention = p["intention"]
                member = members[p["bridge_idx"]]
                intended = p["concept_seq"]

                if mode == "p":
                    # --- Generator-P (Task 9): self-supervise the
                    #     PredictiveCoder on EACH prefix->next of the
                    #     intended order (Task 4 pc.learn), then eval ONE
                    #     active-inference rollout for the per-epoch
                    #     metric. The substrate is UNCHANGED -- learning
                    #     touches ONLY the small P weights; the rollout's
                    #     ONLY pool write is the write-only P-T
                    #     ignite_prediction surface.
                    for t in range(len(intended)):
                        prefix = list(intended[:t])
                        pc.learn(prefix=prefix,
                                 target_next_idx=int(intended[t]),
                                 lr=float(args.lr))
                    # active-inference rollout realized through the
                    # validated substrate; gate-cleared rate via the
                    # EXISTING integrated self_comprehend readout (SAME
                    # rate aggregate as Step-0 + the gate). The
                    # candidate list inside _decode_candidate_p is the
                    # FIXED canonical vocab-index order (NOT
                    # target-ordered) -- anti-cheat Minor #4.
                    decoded, top_rate = _decode_candidate_p(
                        pc, member, ignite_prediction, self_comprehend,
                        intended, _P_DRIVE_PA, _P_STEPS_PER,
                        _DECODE_WINDOW)
                    gate_cleared = bool(top_rate >= g1_abstain)
                    if gate_cleared:
                        n_gate_cleared += 1
                    reward = compose_reward(
                        decoded, intended, gate_cleared)
                    rewards_this_epoch.append(reward)
                    # inter-production recovery so adaptation/STP
                    # recovers before the next proposition's ignition.
                    _recover(member, int(args.recover_steps))
                    continue

                # --- songbird (G1/G1.5): BYTE-IDENTICAL to before. ----
                base = song.rollout(intention, len(intended))

                best_reward = -1.0
                best_cand = None
                for _ in range(n_babble):
                    cand = song.babble(base, rng, temperature)
                    # Readout-mode dispatch (SAME path + SAME
                    # traj_rate_rule as Step-0 + the gate):
                    #   final      = M3 integrated decode (ignite the
                    #     WHOLE ordered candidate ONCE, self_comprehend
                    #     ONCE; length-1; byte-identical to G1).
                    #   trajectory = per-slot ordered decode; top_rate
                    #     = traj_top_rate (MIN per slot).
                    decoded, top_rate = _decode_candidate(
                        readout, member, ignite_sequence,
                        self_comprehend, ignite_and_trajectory_decode,
                        cand, _DRIVE_PA, _STEPS_PER, _DECODE_WINDOW)
                    gate_cleared = bool(top_rate >= g1_abstain)
                    if gate_cleared:
                        n_gate_cleared += 1
                    reward = compose_reward(
                        decoded, intended, gate_cleared)
                    rewards_this_epoch.append(reward)
                    if reward > best_reward:
                        best_reward = reward
                        best_cand = cand
                    # inter-production recovery so adaptation/STP
                    # recovers before the next ignition.
                    _recover(member, int(args.recover_steps))

                # DA-gated three-factor reinforce on the BEST candidate
                # (reward<=0 is a no-op inside reinforce by design).
                if best_cand is not None and best_reward > 0.0:
                    song.reinforce(intention, best_cand,
                                   best_reward, float(args.lr))

            mean_reward = (
                sum(rewards_this_epoch) / len(rewards_this_epoch)
                if rewards_this_epoch else 0.0)
            loss_history.append(float(mean_reward))

            # per-epoch kill-safe checkpoint (host numpy arrays only).
            # songbird mode: [SongHVC.W] -- BYTE-IDENTICAL to before.
            # P mode: [PredictiveCoder.W_in, W_pred, b_pred] (the only
            # learned P machinery; resumed by mode above). Same atomic
            # kill-safe save_checkpoint idiom either way.
            if mode == "p":
                ckpt_weights = [
                    np.asarray(pc.W_in, dtype=np.float32),
                    np.asarray(pc.W_pred, dtype=np.float32),
                    np.asarray(pc.b_pred, dtype=np.float32),
                ]
            else:
                ckpt_weights = [np.asarray(song.W, dtype=np.float32)]
            save_checkpoint(
                args.ckpt, epoch, ckpt_weights,
                rng.bit_generator.state, loss_history)
            last_completed_epoch = epoch  # FIX 3: track in-process

            print(f"[epoch {epoch}] mean_reward={mean_reward:.4f} "
                  f"n_gate_cleared={n_gate_cleared} "
                  f"temp={temperature:.3f} "
                  f"({int(time.time()-ep_t0)}s) [ckpt saved]",
                  flush=True)

    except KeyboardInterrupt:
        # kill-safe: the last COMPLETED epoch is already checkpointed
        # (save_checkpoint runs at the END of each epoch). FIX 3: use
        # the in-process last_completed_epoch instead of re-reading the
        # npz we just wrote. Behavior otherwise unchanged -- the
        # handler writes NO checkpoint (the last completed epoch was
        # already checkpointed at epoch-end); re-running resumes from
        # that epoch.
        print(f"\n[INTERRUPT] KeyboardInterrupt -- last completed "
              f"epoch checkpointed = {last_completed_epoch}. Re-run "
              f"to resume. Exiting cleanly.", flush=True)
        print("=" * 64, flush=True)
        return 0

    print("\n" + "=" * 64, flush=True)
    print("SONG G1 TASK 9 TRAINING COMPLETE", flush=True)
    print(f"  mode               : {mode}"
          f"{' (Generator-P active-inference rollout)' if mode == 'p' else ' (songbird; G1/G1.5 byte-identical)'}",
          flush=True)
    print(f"  readout            : {readout}"
          f"{' (P mode: superseded by P decode path)' if mode == 'p' else (' (traj_rate_rule=' + _TRAJ_RATE_RULE + ')' if readout == 'trajectory' else ' (G1 byte-identical)')}",
          flush=True)
    print(f"  epochs run         : {start_epoch}..{n_epochs-1}",
          flush=True)
    print(f"  g1_abstain (frozen): {g1_abstain:.2f} "
          f"(control-max, {mode} regime; NOT 650"
          f"{'; NOT G1 final 72.0; NOT trajectory 46.0' if mode == 'p' else ('; NOT G1 72.0' if readout == 'trajectory' else '')})",
          flush=True)
    if loss_history:
        print(f"  mean_reward first  : {loss_history[0]:.4f}",
              flush=True)
        print(f"  mean_reward last   : {loss_history[-1]:.4f}",
              flush=True)
    print(f"  checkpoint         : {args.ckpt}", flush=True)
    print(f"  sidecar            : {_sidecar_path(args.ckpt)}",
          flush=True)
    print(f"  elapsed            : {int(time.time()-t0)}s", flush=True)
    print("  NOTE: smoke/training rewards are NOT the G1 verdict -- "
          "Task 10's pre-registered gate is.", flush=True)
    print("=" * 64, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
