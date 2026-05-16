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
                      decode_window, recover_steps):
    """Pre-registered Step 0 (ONCE at train start; NOT 650).

    Measure the self_comprehend integrated-residual TOP-RATE
    distribution in the IDENTICAL regime for:
      (i)  ENCODED proxy : intended-order productions of the TRAIN
           propositions.
      (ii) CONTROL       : random/unencoded concept sequences AND
           permuted-order productions of the TRAIN propositions
           (the permuted-ORDER control is load-bearing).

    g1_abstain = the encoded-vs-control separation at the SAME operating
    criterion the original 650 used: control-max (the 650 doc set the
    gate just above control-max ~584). A simple AUC (P[encoded_rate >
    control_rate]) is ALSO recorded for transparency, matching the
    encoded-vs-control AUC methodology that produced 650 -- but the
    OPERATING POINT is control-max, fixed here, never tuned.

    Returns a calibration dict (also written to the sidecar)."""
    encoded_rates = []
    control_rates = []

    # (i) ENCODED proxy: intended order of each TRAIN proposition.
    for p in props_train:
        member = members[p["bridge_idx"]]
        _dec, rate = _integrated_decode(
            member, ignite_sequence, self_comprehend,
            p["concept_seq"], drive_pA, steps_per, decode_window)
        encoded_rates.append(rate)
        _recover(member, recover_steps)

    # (ii-a) CONTROL: permuted-ORDER productions of the TRAIN props
    #        (same concept multiset, order scrambled). Load-bearing.
    from research.runners.song_g1_core import permuted_order_controls
    for p in props_train:
        member = members[p["bridge_idx"]]
        perms = permuted_order_controls(p["concept_seq"], rng, n=2)
        for perm in perms:
            _dec, rate = _integrated_decode(
                member, ignite_sequence, self_comprehend,
                perm, drive_pA, steps_per, decode_window)
            control_rates.append(rate)
            _recover(member, recover_steps)

    # (ii-b) CONTROL: random/unencoded concept sequences (per train
    #        bridge so the regime matches; deterministic via rng).
    for p in props_train[:_CTRL_N_RANDOM]:
        member = members[p["bridge_idx"]]
        n_v = len(member.vocab)
        a = int(rng.integers(0, n_v))
        b = int(rng.integers(0, n_v))
        if b == a:
            b = (a + 1) % n_v
        _dec, rate = _integrated_decode(
            member, ignite_sequence, self_comprehend,
            [a, b], drive_pA, steps_per, decode_window)
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
        "encoded_rates": [round(x, 2) for x in enc],
        "control_rates": [round(x, 2) for x in ctl],
        "encoded_mean": round(encoded_mean, 2),
        "control_mean": round(control_mean, 2),
        "control_max": round(control_max, 2),
        "auc_encoded_vs_control": round(auc, 4),
        "n_encoded": len(enc),
        "n_control": len(ctl),
        "note": ("self_comprehend NO-DRIVE integrated-residual regime; "
                 "NOT the literal 650 (continuous-drive regime). "
                 "control-max operating point, fixed at train start, "
                 "never tuned (anti-cheat: pre-registered RULE)."),
    }


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
                        "frozen props + Step-0 g1_abstain).")
    p.add_argument("--smoke", action="store_true",
                   help="tiny build/kill-safe-resume validation: 2 "
                        "epochs, 2 train props, k=2 babbles. NOT the "
                        "G1 result (Task 10 is the gate).")
    args = p.parse_args()

    # --- lazy/heavy imports (loading 5 sparse bridges is slow + GPU) -
    import time

    import numpy as np

    from research.runners.song_g1_core import compose_reward
    from research.runners.song_g1_ignite import (
        ignite_sequence, load_members, self_comprehend,
    )
    from sim.song_hvc import SongHVC
    from sim.train_checkpoint import (
        load_checkpoint, resume_epoch, save_checkpoint,
    )

    t0 = time.time()
    smoke = bool(args.smoke)
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
    # smoke trains only the first 2 train props (still freezes the SAME
    # full train/held-out lists in the sidecar so a non-smoke resume /
    # Task 10 see the canonical set).
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
    song = SongHVC(n_states=_SONG_N_STATES,
                   n_concepts=_SONG_N_CONCEPTS,
                   seed=int(args.seed))
    _seed_intention_biases(song, all_props)

    # --- resume (kill-safe Inc-3 pattern) ----------------------------
    ckpt = load_checkpoint(args.ckpt)
    start_epoch = resume_epoch(ckpt)
    rng = np.random.default_rng(int(args.seed))
    loss_history = []
    if ckpt is not None:
        # restore controller weights + RNG state + loss history.
        song.W = np.asarray(ckpt["weights"][0],
                             dtype=np.float32).copy()
        rng.bit_generator.state = ckpt["rng_state"]
        loss_history = list(ckpt["loss_history"])
        print(f"\n[RESUME] checkpoint at epoch {ckpt['epoch']} -> "
              f"resuming at epoch {start_epoch} "
              f"(loss_history len={len(loss_history)})", flush=True)
    else:
        print(f"\n[FRESH] no checkpoint at {args.ckpt} -> starting "
              f"at epoch 0", flush=True)

    # --- sidecar: frozen props + Step-0 g1_abstain (persist ONCE;
    #     resume + Task 10 MUST reuse the SAME values) ----------------
    meta = _load_sidecar(args.ckpt)
    if meta is not None and "g1_abstain" in meta.get("calibration", {}):
        calib = meta["calibration"]
        g1_abstain = float(calib["g1_abstain"])
        print(f"[SIDECAR] reusing pre-registered Step-0 g1_abstain="
              f"{g1_abstain:.2f} (control-max, computed at train "
              f"start; NEVER retuned)", flush=True)
    else:
        # Step 0 -- run ONCE at train start (NOT 650). Uses a dedicated
        # RNG so the calibration is independent of (and does not
        # consume) the training rng stream.
        print("\n[STEP 0] PRE-REGISTERED control-calibrated abstention "
              "floor (self_comprehend regime; NOT literal 650) ...",
              flush=True)
        calib_rng = np.random.default_rng(int(args.seed) * 7 + 1)
        calib = _step0_calibrate(
            members, props_train, ignite_sequence, self_comprehend,
            calib_rng, _DRIVE_PA, _STEPS_PER, _DECODE_WINDOW,
            int(args.recover_steps))
        g1_abstain = float(calib["g1_abstain"])
        meta = {
            "task": "song_g1 Task 9 self-supervised training",
            "substrate": "G.20 320-sparse 5-bridge",
            "seed": int(args.seed),
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
                base = song.rollout(intention, len(intended))

                best_reward = -1.0
                best_cand = None
                for _ in range(n_babble):
                    cand = song.babble(base, rng, temperature)
                    # M3: ignite the WHOLE ordered candidate ONCE, THEN
                    # self_comprehend ONCE on the integrated residual
                    # (NEVER per-slot then average).
                    decoded, top_rate = _integrated_decode(
                        member, ignite_sequence, self_comprehend,
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

            # per-epoch kill-safe checkpoint (host numpy array only).
            save_checkpoint(
                args.ckpt, epoch,
                [np.asarray(song.W, dtype=np.float32)],
                rng.bit_generator.state, loss_history)

            print(f"[epoch {epoch}] mean_reward={mean_reward:.4f} "
                  f"n_gate_cleared={n_gate_cleared} "
                  f"temp={temperature:.3f} "
                  f"({int(time.time()-ep_t0)}s) [ckpt saved]",
                  flush=True)

    except KeyboardInterrupt:
        # kill-safe: the last COMPLETED epoch is already checkpointed
        # (save_checkpoint runs at the END of each epoch). Just exit
        # cleanly; re-running resumes from that epoch.
        last = (resume_epoch(load_checkpoint(args.ckpt)) - 1)
        print(f"\n[INTERRUPT] KeyboardInterrupt -- last completed "
              f"epoch checkpointed = {last}. Re-run to resume. "
              f"Exiting cleanly.", flush=True)
        print("=" * 64, flush=True)
        return 0

    print("\n" + "=" * 64, flush=True)
    print("SONG G1 TASK 9 TRAINING COMPLETE", flush=True)
    print(f"  epochs run         : {start_epoch}..{n_epochs-1}",
          flush=True)
    print(f"  g1_abstain (frozen): {g1_abstain:.2f} "
          f"(control-max; NOT 650)", flush=True)
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
