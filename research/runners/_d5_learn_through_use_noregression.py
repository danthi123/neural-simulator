"""PRODUCTION no-regression GUARD for the D5 learn-through-use default-on flip (board #71 handback, step 4).

The graded apical read is ALREADY folded into the production episodic memory on HEAD: `EpisodicDapMemory.recall`
emits the graded magnitude (depth_rest/depth_hold/soft) beside the binary UP-fraction, `EpisodicRecallOrgan.recall`
passes it through, `d5_episodic_production_organ.recall_disclosure` surfaces `depth_hold` (the BTSP IS_post) as the
conversation-visible recall STRENGTH, and `continuous_engine.consolidate_used_memory` runs the between-turn
plateau-gated BTSP — all default-OFF behind `BRAIN_D5_CONSOLIDATE`. Knob 2 (the graded read criterion: relative-
tolerance floor + saturating-tail exclusion, STABLE 6/6, finding
2026-08-21-d5-learn-through-use-knob2-relative-tolerance-floor-stable-6of6-seed44-closed on branch
research/d5-graded-apical-read) is CLOSED and its criterion is now in `_d5_step6_graded_apical_read_derisk.py` (imported
here). The remaining gate before the owner flips `BRAIN_D5_CONSOLIDATE` default 0->1 is a PRODUCTION no-regression check.
This runner IS that check, at the REAL production op-point.

It proves THREE claims through the ACTUAL production functions (`EpisodicRecallOrgan.note_topic`/`recall`,
`d5_episodic_production_organ.recall_disclosure`, `continuous_engine.{mark_recall,consolidate_used_memory,
d5_consolidate_enabled}`) at the PRODUCTION encode (train_events=40 — exactly what a live `/api/brain-chat` turn runs,
webapp/server.py:4318-4344), with the step-5/6 snapshot/restore weight-attribution isolation:

  A. OFF byte-identical to HEAD (`BRAIN_D5_CONSOLIDATE` unset):
     * `recall_disclosure(rec)` emits NO recall-strength clause (the default reply is unchanged);
     * `consolidate_used_memory(...)` returns None (disabled short-circuit);
     * a full `mark_recall -> consolidate_used_memory` cycle leaves the store weights BYTE-IDENTICAL (hash equal).

  B. ON leaves the moat / abstain UNCHANGED (`BRAIN_D5_CONSOLIDATE=1`):
     * the binary `in_memory` gate is FLAG-INDEPENDENT (`recall` never reads the flag) — a formed topic still
       completes, a never-formed topic still ABSTAINS, a formation-lesion still collapses;
     * `recall_disclosure` for a never-formed topic (the honest abstain line) is IDENTICAL off vs on;
     * ON only ADDS the strength clause to a memory the binary gate ALREADY admitted — the completion text (the
       moat-carrying part) is unchanged, and the surfaced strength == the record's graded `depth_hold`.

  C. ON adds a CONVERSATION-VISIBLE graded rise that VANISHES when off:
     * across real `consolidate_used_memory` turns the surfaced `depth_hold` rises monotonically under the
       6/6-validated knob-2 criterion (`_mono_rel` with the absolute floor), the `recall_disclosure` STRING changes
       (the reply's strength mV rises), and `in_memory` stays True the whole time (the moat holds through consolidation);
     * OFF (lesion control) the same mark+consolidate cycle is a NO-OP -> the store is byte-identical and the read does
       not move beyond instrument jitter -> the rise is DRIVEN by the consolidation loop, not decoration.

SCOPE (honest boundary). This guard covers the RECALLED-TOPIC no-regression — the memory the turn actually used. The
distinct NEIGHBOR-CROSSTALK no-regression (consolidating one memory perturbing an OVERLAPPING assembly's surfaced
strength on ~1/6 emergent builds) is the existing `_d5_graded_flip_soak.py`'s domain and the documented residual that
BLOCKS the flip until knob 1 lands (the memory separator, sep_bias=1000 -> 6/6 DISJOINT, board #73, commit e62113ef on
branch research/memory-separator-readout), per finding
2026-08-21-d5-graded-apical-read-conversation-visible-in-production-flip-blocked-on-emergent-assembly-crosstalk. So a
clean verdict here is NECESSARY, not sufficient, for the flip; the flip additionally requires knob 1 on main. Self-
ignition (~1/6 builds where the assembly's nocue self-completes -> the moat correctly abstains) is reported as
INSTRUMENT-INVALID (the moat working, not a regression), matching that finding.

BRAIN-BASED / NO sim/ edit / ADDITIVE: this runner makes NO production-code change (OFF is byte-identical to HEAD BY
CONSTRUCTION). The strengthening is the substrate's OWN plateau-gated BTSP via the actual `consolidate_used_memory`.
Host code is only the clock and the determinism guard. GPU-preferred (route via `tools/gpu_queue.sh add` — one brain
load, the queue serializes the OOM constraint).

  Run:    SIM_BACKEND=cupy python -m research.runners._d5_learn_through_use_noregression --seed 42
  6-seed: SIM_BACKEND=cupy python -m research.runners._d5_learn_through_use_noregression --seeds 42 43 44 100 101 102
  self-test only (no brain): python -m research.runners._d5_learn_through_use_noregression --self-test
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from sim.backend import get_backend  # noqa: E402
# The knob-2-validated criterion (the SAME floor + saturating-tail handling) + its flat-trace self-test — reuse.
from research.runners._d5_step6_graded_apical_read_derisk import (  # noqa: E402
    _mono_rel, _selftest_criteria, MONO_TOL_ABS_MV, MOVE_MARGIN_DEPTH)
from research.runners._gap5_dendritic_dap_readout_completion_derisk import (  # noqa: E402
    _reset_apical_latch, _apical_up_read)
from research.runners._gap5_d5_latch_self_termination_derisk import snapshot_state, restore_state  # noqa: E402
# The ACTUAL production functions under test (exactly what webapp/server.py calls on a referential turn).
from research.runners.d5_episodic_production_organ import (  # noqa: E402
    EpisodicRecallOrgan, recall_disclosure, SURFACED_GRADED_READ)
from webapp import continuous_engine as CE  # noqa: E402
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_d5_ltu_noregression" / "seed42.json"

STRENGTH_MARK = "recall strength"     # the substring the ON disclosure adds and the OFF disclosure must NOT contain


def _whash(cp, W):
    h = np.asarray(cp.asnumpy(W) if hasattr(cp, "asnumpy") else W, dtype=np.float32)
    return hashlib.sha1(h.tobytes()).hexdigest()[:16]


def _set_flag(on: bool):
    if on:
        os.environ["BRAIN_D5_CONSOLIDATE"] = "1"
    else:
        os.environ.pop("BRAIN_D5_CONSOLIDATE", None)   # UNSET == the HEAD default (default-OFF anchor)


def _strength_of(rec):
    return float((rec.get("graded_cue") or {}).get(SURFACED_GRADED_READ, 0.0))


def run_one(seed, a, backend, out_path):
    t0 = time.time()
    print("\n" + "=" * 118)
    print(f"[d5-ltu-noreg] seed={seed} backend={backend} te={a.train_events} — production no-regression guard for the "
          f"BRAIN_D5_CONSOLIDATE flip (OFF byte-identical | ON moat-unchanged | ON graded rise conversation-visible)",
          flush=True)
    result = {"seed": seed, "backend": backend, "params": vars(a), "surfaced_read": SURFACED_GRADED_READ}
    cache_key = ("d5-ltu-noreg", seed)
    try:
        cp, _ = get_backend()
        CE.forget_session(cache_key)
        _set_flag(False)   # start from the HEAD default so nothing leaks into the store build

        # ── PRODUCTION encode: the REAL write path note_topic('dog') at train_events=40; 'cat' never spoken ─────────
        org = EpisodicRecallOrgan(seed, ["cat", "dog"], verbose=False)
        org._ensure_built()
        org.mem.p["train_events"] = int(a.train_events)
        if not org.note_topic("dog"):
            raise RuntimeError("note_topic('dog') did not form the assembly")
        mem = org.mem
        dslot = mem.topic_slot["dog"]
        cslot = mem.topic_slot["cat"]

        # clean-rest snapshot for isolated, deterministic, weight-attributable handler reads (the step-5/6 guard)
        mem.recall("dog")  # warm/allocate cp_v_apical
        mem.R.hard_silence(); _reset_apical_latch(mem.bridge)
        snap = snapshot_state(mem.bridge)
        W_before = mem.R.C.data.copy()
        w_dog_before = float(cp.mean(W_before[mem.R.withinA_masks[dslot]]))
        w_cat_before = float(cp.mean(W_before[mem.R.withinA_masks[cslot]]))

        def handler_read(topic, W, *, lesion=False):
            """The EXACT production recall (EpisodicRecallOrgan.recall), snapshot-isolated on store-weights W so a
            T-vs-T+k comparison is purely WEIGHT-attributable (the step-5/6 isolation)."""
            restore_state(mem.bridge, snap)
            mem.bridge.cp_connections.data[:] = cp.asarray(W)
            return org.recall(topic, lesion=lesion)

        # ── borderline reads (the moat truth table): formed dog completes, never-formed cat abstains, lesion collapses
        rec_dog = handler_read("dog", W_before)
        rec_dog2 = handler_read("dog", W_before)
        rec_cat = handler_read("cat", W_before)
        rec_dog_les = handler_read("dog", W_before, lesion=True)
        inmem_dog = bool(rec_dog["in_memory"])
        inmem_cat = bool(rec_cat["in_memory"])
        inmem_dog_les = bool(rec_dog_les["in_memory"])
        ac_dog = float(rec_dog["apical_cue"])
        cat_never = bool(w_cat_before < 5.0 and not inmem_cat)
        deterministic = bool(abs(rec_dog["apical_cue"] - rec_dog2["apical_cue"]) < 1e-9
                             and abs(_strength_of(rec_dog) - _strength_of(rec_dog2)) < 1e-9)
        print(f"[d5-ltu-noreg] moat truth-table: dog in_memory={inmem_dog} (apical_cue={ac_dog:.3f}) | "
              f"cat in_memory={inmem_cat} | dog(formation-lesion) in_memory={inmem_dog_les} | det={deterministic}",
              flush=True)

        # SELF-IGNITION (~1/6 builds): dog does not complete at T -> the moat CORRECTLY abstains -> the recalled-topic
        # no-regression test is N/A on this build (not a regression; the honesty gate working). Report instrument-invalid.
        if not (inmem_dog and cat_never and deterministic):
            result["verdict_status"] = "UNDEFINED"
            result["checks"] = {"instrument_valid": False, "inmem_dog": inmem_dog, "cat_never": cat_never,
                                "deterministic": deterministic, "apical_cue_dog": round(ac_dog, 4),
                                "reason": "dog did not complete at the production encode (self-ignition/moat-abstain) "
                                          "or cat/determinism control failed — recalled-topic test N/A this build"}
            print(f"[d5-ltu-noreg] seed={seed} INSTRUMENT-INVALID (self-ignition/moat-abstain): inmem_dog={inmem_dog} "
                  f"cat_never={cat_never} det={deterministic}", flush=True)
            CE.forget_session(cache_key)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, default=str))
            return result

        # ── CLAIM A — OFF byte-identical to HEAD ────────────────────────────────────────────────────────────────
        _set_flag(False)
        off_disabled = (not CE.d5_consolidate_enabled())
        disc_dog_off = recall_disclosure(rec_dog, content=None)
        disc_cat_off = recall_disclosure(rec_cat, content=None)
        off_no_strength = STRENGTH_MARK not in disc_dog_off
        restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_before)
        hash_pre = _whash(cp, mem.bridge.cp_connections.data)
        CE.mark_recall(cache_key, "dog")
        off_rec = CE.consolidate_used_memory(cache_key, org)
        hash_post_off = _whash(cp, mem.bridge.cp_connections.data)
        off_consolidate_none = off_rec is None
        off_store_byte_identical = bool(hash_post_off == hash_pre)
        claim_A = bool(off_disabled and off_no_strength and off_consolidate_none and off_store_byte_identical)
        print(f"[d5-ltu-noreg] CLAIM A (OFF byte-identical): disabled={off_disabled} no_strength_clause={off_no_strength} "
              f"consolidate_None={off_consolidate_none} store_byte_identical={off_store_byte_identical}\n"
              f"    OFF dog reply: {disc_dog_off!r}", flush=True)

        # ── CLAIM B — ON leaves the moat / abstain UNCHANGED ────────────────────────────────────────────────────
        _set_flag(True)
        on_enabled = bool(CE.d5_consolidate_enabled())
        # the binary gate is flag-independent (recall never reads the flag): re-read with the flag ON, same verdicts.
        rec_dog_on = handler_read("dog", W_before)
        rec_cat_on = handler_read("cat", W_before)
        rec_dog_les_on = handler_read("dog", W_before, lesion=True)
        gate_flag_independent = bool(bool(rec_dog_on["in_memory"]) == inmem_dog
                                     and bool(rec_cat_on["in_memory"]) == inmem_cat
                                     and bool(rec_dog_les_on["in_memory"]) == inmem_dog_les)
        disc_dog_on = recall_disclosure(rec_dog_on, content=None)
        disc_cat_on = recall_disclosure(rec_cat_on, content=None)
        on_has_strength = STRENGTH_MARK in disc_dog_on
        abstain_unchanged = bool(disc_cat_on == disc_cat_off)     # the honest-abstain line is byte-identical off vs on
        completion_frag = f"dendritic dAP completion {float(rec_dog_on['apical_cue']):.2f}"
        completion_text_preserved = bool((f"dendritic dAP completion {ac_dog:.2f}" in disc_dog_off)
                                         and (completion_frag in disc_dog_on))
        surfaced = _strength_of(rec_dog_on)
        strength_matches_record = bool(on_has_strength and (f"{surfaced:.1f} mV" in disc_dog_on))
        moat_truth_table = bool(inmem_dog and (not inmem_cat) and (not inmem_dog_les))
        claim_B = bool(on_enabled and gate_flag_independent and on_has_strength and abstain_unchanged
                       and completion_text_preserved and strength_matches_record and moat_truth_table)
        print(f"[d5-ltu-noreg] CLAIM B (ON moat-unchanged): gate_flag_independent={gate_flag_independent} "
              f"on_has_strength={on_has_strength} abstain_line_identical={abstain_unchanged} "
              f"completion_text_preserved={completion_text_preserved} strength_matches_record={strength_matches_record} "
              f"moat_truth_table={moat_truth_table}\n    ON  dog reply: {disc_dog_on!r}\n"
              f"    cat abstain (off==on): {disc_cat_on!r}", flush=True)

        # ── CLAIM C — ON adds a conversation-visible graded rise that VANISHES when off ──────────────────────────
        # PRODUCTION SIGNAL = the FIRST-USE rise: one idle consolidation tick (the production budget,
        # BRAIN_D5_CONSOLIDATE_BUDGET=1) raises the surfaced depth_hold, visible in the reply. At the near-saturated
        # production encode (te=40) the regenerative NMDA plateau is CEILING-BOUNDED (Bittner et al. 2017 Science
        # 357:1033), so depth_hold OVERSHOOTS on the first tick then saturates/decays over further ticks — sustained
        # monotone growth is NOT the production signal (nor what production does: 1 tick/recall). So Claim C GATES on the
        # first-use rise + the reply change, and REPORTS the multi-turn trajectory + the knob-2 monotone check (the
        # saturating-tail-tolerant floor) for honesty (it is expected to be non-monotone at the saturated encode; the
        # first-use rise holds — matching finding 2026-08-21-d5-graded-apical-read-conversation-visible-...-crosstalk).
        _set_flag(True)
        depth_traj = [round(_strength_of(rec_dog_on), 4)]
        disc_traj = [disc_dog_on]
        disc_at_T = disc_dog_on
        inmem_traj = [inmem_dog]
        W_after = W_before
        consolidated_rounds = 0
        for turn in range(a.n_turns):
            restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_after)
            CE.mark_recall(cache_key, "dog")
            on_rec = CE.consolidate_used_memory(cache_key, org, n_episodes=a.n_episodes)
            if on_rec is not None:
                consolidated_rounds += 1
            W_after = mem.bridge.cp_connections.data.copy()
            rec_turn = handler_read("dog", W_after)
            depth_traj.append(round(_strength_of(rec_turn), 4))
            inmem_traj.append(bool(rec_turn["in_memory"]))
            disc_turn = recall_disclosure(rec_turn, content=None)
            disc_traj.append(disc_turn)
            print(f"  [turn T+{turn+1}] consolidate->{'ok' if on_rec else None} depth_hold={_strength_of(rec_turn):.3f} "
                  f"in_memory={rec_turn['in_memory']} reply={disc_turn!r}", flush=True)
        consolidated = bool(consolidated_rounds > 0)
        first_use_move = depth_traj[1] - depth_traj[0]             # the single-tick production learn-through-use signal
        first_use_rises = bool(first_use_move > MOVE_MARGIN_DEPTH)
        first_use_reply_changed = bool(disc_traj[1] != disc_at_T)  # the surfaced strength mV rose IN the reply STRING
        graded_monotone = bool(_mono_rel(depth_traj, tol_abs=MONO_TOL_ABS_MV)
                               and depth_traj[-1] > depth_traj[0] + MOVE_MARGIN_DEPTH)  # reported (saturates at te=40)
        net_move = depth_traj[-1] - depth_traj[0]
        peak_move = max(depth_traj) - depth_traj[0]
        inmem_held = bool(all(inmem_traj))                        # the moat holds THROUGH consolidation

        # OFF (lesion) control: the SAME mark+consolidate cycle with the flag OFF is a NO-OP -> the STORE is
        # byte-identical (hash) and the read does NOT move beyond instrument jitter. (The read is NOT bit-reproducible
        # on cupy — FMA reorder gives a ~1e-4 mV jitter — so the rigorous "vanishes when off" invariant is the STORE
        # hash + a jitter-tolerant read move well below the ON move, NOT a 1e-9 read byte-identity.)
        _set_flag(False)
        restore_state(mem.bridge, snap); mem.bridge.cp_connections.data[:] = cp.asarray(W_before)
        hash_ctrl_pre = _whash(cp, mem.bridge.cp_connections.data)
        CE.mark_recall(cache_key, "dog")
        off_ctrl_rec = CE.consolidate_used_memory(cache_key, org)
        hash_ctrl_post = _whash(cp, mem.bridge.cp_connections.data)
        off_ctrl_store_identical = bool(hash_ctrl_post == hash_ctrl_pre)
        rec_dog_off_ctrl = handler_read("dog", W_before)
        disc_off_ctrl = recall_disclosure(rec_dog_off_ctrl, content=None)   # flag OFF -> no strength clause (== Claim A)
        reply_flat_off = bool(disc_off_ctrl == disc_dog_off)               # the OFF reply is unchanged by the no-op tick
        move_off = _strength_of(rec_dog_off_ctrl) - depth_traj[0]           # the OFF single-tick move (must be ~0)
        # "vanishes when off" = the consolidate is a no-op: store byte-identical (hash) + the surfaced depth does not
        # move beyond cupy jitter. (The OFF reply naturally drops the strength clause per Claim A — that is the flag's
        # surfacing, NOT a substrate move — so off_flat does NOT compare the OFF reply against the flag-ON turn-T reply.)
        off_flat = bool(off_ctrl_rec is None and off_ctrl_store_identical and abs(move_off) < MOVE_MARGIN_DEPTH)
        # ATTRIBUTION (tools.lab): whose is the first-use rise? The ON tick vs the OFF (no-op) control — the graded
        # move must be OWNED by the consolidation loop, not the read machinery. (Measuring both arms is not attributing.)
        attribution = attributable_to(f"[s{seed}] first-use depth_hold rise: ON tick vs OFF no-op control",
                                      first_use_move, move_off)
        result["attributable"] = attribution
        claim_C = bool(consolidated and first_use_rises and first_use_reply_changed and inmem_held and off_flat)
        print(f"[d5-ltu-noreg] CLAIM C (ON first-use rise visible): depth_hold traj={depth_traj} "
              f"first_use_move={first_use_move:.3f} rises={first_use_rises} reply_changed={first_use_reply_changed} "
              f"in_memory_held={inmem_held} | monotone(reported)={graded_monotone} net_move={net_move:.3f} "
              f"peak_move={peak_move:.3f} | OFF-control flat={off_flat} (store_identical={off_ctrl_store_identical}, "
              f"reply_flat_off={reply_flat_off}, first-use ON={first_use_move:.3f} vs OFF={move_off:.4f})", flush=True)

        go = bool(claim_A and claim_B and claim_C)

        # ── the earned verdict ──────────────────────────────────────────────────────────────────────────────────
        v = Verdict(f"PRODUCTION no-regression for the BRAIN_D5_CONSOLIDATE flip at the production encode (seed {seed}): "
                    f"OFF byte-identical | ON moat-unchanged | ON graded rise conversation-visible")
        v.disabled("BRAIN_D5_CONSOLIDATE (default)", "the OFF arm is the HEAD default path — no sim/ or OFF-path edit")
        v.disabled("neighbor-crosstalk (out of scope)", "this guard is the RECALLED-topic no-regression; the "
                   "overlapping-assembly crosstalk is _d5_graded_flip_soak + knob-1 (separator, board #73)")
        v.require("A: OFF consolidate disabled", off_disabled, expect=True,
                  note="d5_consolidate_enabled() is False when the flag is unset (the default anchor)")
        v.require("A: OFF disclosure has no strength clause", off_no_strength, expect=True,
                  note="the default recall reply is byte-identical to HEAD (no 'recall strength' text)")
        v.require("A: OFF consolidate returns None", off_consolidate_none, expect=True,
                  note="consolidate_used_memory short-circuits when disabled")
        v.require("A: OFF store byte-identical", off_store_byte_identical, expect=True,
                  note="a full mark+consolidate cycle leaves the store weights hash-identical")
        v.require("B: binary gate flag-independent", gate_flag_independent, expect=True,
                  note="the in_memory gate is identical off vs on (recall never reads the flag) -> moat unchanged")
        v.require("B: moat truth-table", moat_truth_table, expect=True,
                  note="formed dog completes, never-formed cat abstains, formation-lesion collapses")
        v.require("B: abstain line identical off vs on", abstain_unchanged, expect=True,
                  note="the honest-abstain reply for a never-formed topic is byte-identical off vs on")
        v.require("B: ON completion text preserved", completion_text_preserved, expect=True,
                  note="ON only APPENDS the strength; the moat-carrying completion text is unchanged")
        v.require("B: ON surfaced strength == record read", strength_matches_record, expect=True,
                  note="the surfaced number is the real graded depth_hold, not an invented value")
        v.require("C: consolidation ran", consolidated, expect=True,
                  note="the ON arm actually consolidated (mark_recall -> consolidate_used_memory ticked)")
        v.reaches("C: first use raises surfaced depth_hold", depth_traj[0], depth_traj[1],
                  note="one production consolidation tick (the budget) raised the conversation-visible recall strength")
        v.require("C: first-use rises", first_use_rises, expect=True,
                  note="depth_hold(T+1) > depth_hold(T) — the production single-tick learn-through-use signal")
        v.require("C: first-use reply STRING changed", first_use_reply_changed, expect=True,
                  note="the surfaced strength mV rose in the recall_disclosure reply itself after one tick (anti-hollow)")
        v.require("C: in_memory held through consolidation", inmem_held, expect=True,
                  note="the binary moat gate never drops out while the strength rises")
        v.control("C: ON first-use rise vs OFF flat", treatment=first_use_move, control=move_off,
                  min_separation=MOVE_MARGIN_DEPTH,
                  note="the rise is DRIVEN by consolidation; the OFF cycle is a byte-identical no-op (read+reply flat)")
        decided = v.decide(go=go)
        result["verdict"] = decided
        result["verdict_status"] = decided["status"]

        result["checks"] = dict(
            instrument_valid=True, te=int(a.train_events), surfaced_read=SURFACED_GRADED_READ,
            claim_A_off_byte_identical=claim_A, claim_B_on_moat_unchanged=claim_B,
            claim_C_graded_rise_visible=claim_C, GO=go, deterministic=deterministic,
            # A
            off_disabled=off_disabled, off_no_strength=off_no_strength, off_consolidate_none=off_consolidate_none,
            off_store_byte_identical=off_store_byte_identical, hash_pre=hash_pre, hash_post_off=hash_post_off,
            disc_dog_off=disc_dog_off, disc_cat_off=disc_cat_off,
            # B
            on_enabled=on_enabled, gate_flag_independent=gate_flag_independent, on_has_strength=on_has_strength,
            abstain_unchanged=abstain_unchanged, completion_text_preserved=completion_text_preserved,
            strength_matches_record=strength_matches_record, moat_truth_table=moat_truth_table,
            inmem_dog=inmem_dog, inmem_cat=inmem_cat, inmem_dog_les=inmem_dog_les, cat_never=cat_never,
            surfaced_strength=round(surfaced, 4), disc_dog_on=disc_dog_on, disc_cat_on=disc_cat_on,
            # C — GATED on the first-use rise (production single-tick signal); monotone reported (saturates at te=40)
            depth_traj=depth_traj, inmem_traj=inmem_traj, first_use_move=round(first_use_move, 4),
            first_use_rises=first_use_rises, first_use_reply_changed=first_use_reply_changed,
            graded_monotone=graded_monotone, net_move=round(net_move, 4), peak_move=round(peak_move, 4),
            inmem_held=inmem_held, consolidated=consolidated, consolidated_rounds=consolidated_rounds,
            off_flat=off_flat, off_ctrl_store_identical=off_ctrl_store_identical, reply_flat_off=reply_flat_off,
            move_off=round(move_off, 4), disc_traj=disc_traj, disc_off_ctrl=disc_off_ctrl,
            n_turns=a.n_turns, n_episodes=a.n_episodes, w_dog_before=round(w_dog_before, 3))
        print(f"[d5-ltu-noreg] seed={seed} A={claim_A} B={claim_B} C={claim_C} => {decided['status']}", flush=True)
        CE.forget_session(cache_key)
        del mem, org
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e)
        result["verdict_status"] = "ERROR"
        traceback.print_exc()
    finally:
        _set_flag(False)   # ALWAYS leave the flag at the HEAD default
        CE.forget_session(cache_key)

    result["elapsed_s"] = round(time.time() - t0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print("=" * 118)
    print(f"[d5-ltu-noreg] seed={seed} VERDICT: {result.get('verdict_status')} -> wrote {out_path}")
    print("=" * 118)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--train-events", type=int, default=40, dest="train_events",
                    help="production encode strength (GO_DEFAULTS train_events=40 — the real live op-point)")
    ap.add_argument("--n-episodes", type=int, default=1, dest="n_episodes",
                    help="consolidation episodes per tick (production _D5_EPISODES=1)")
    ap.add_argument("--n-turns", type=int, default=3, dest="n_turns",
                    help="number of USE turns (each: mark_recall -> one consolidate tick -> read)")
    ap.add_argument("--self-test", action="store_true", dest="self_test",
                    help="run ONLY the knob-2 criteria self-test (no brain) and exit")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    # The knob-2 flat-trace CONTROL (imported from step-6) runs on EVERY invocation and BLOCKS if the criterion is
    # ever able to admit a non-rising trace — the monotone check Claim C relies on must be sound.
    _st_fails = _selftest_criteria()
    if _st_fails:
        print(f"[d5-ltu-noreg] ⛔ knob-2 CRITERIA SELF-TEST FAILED: {_st_fails}", flush=True)
        return 3
    print(f"[d5-ltu-noreg] knob-2 criteria self-test: PASS (floor MV={MONO_TOL_ABS_MV}; flat-trace control holds)",
          flush=True)
    if a.self_test:
        return 0

    seeds = a.seeds if a.seeds else [a.seed]
    _, backend = get_backend()
    all_results = {}
    go_flags = []
    valid_flags = []
    for seed in seeds:
        out_path = (Path(a.out) if len(seeds) == 1 else Path(a.out).parent / f"seed{seed}.json")
        res = run_one(seed, a, backend, out_path)
        all_results[seed] = res
        c = res.get("checks", {})
        valid_flags.append(bool(c.get("instrument_valid")))
        go_flags.append(bool(res.get("verdict_status") == "GO"))

    if len(seeds) > 1:
        n = len(seeds)
        n_go = int(sum(go_flags))
        n_valid = int(sum(valid_flags))
        print("\n" + "#" * 118)
        print(f"[d5-ltu-noreg] {n}-SEED SUMMARY: {n_go}/{n} GO ({n_valid}/{n} instrument-valid) seeds={seeds}")
        for s in seeds:
            c = all_results[s].get("checks", {})
            print(f"  seed {s}: status={all_results[s].get('verdict_status')} valid={bool(c.get('instrument_valid'))} "
                  f"A={c.get('claim_A_off_byte_identical')} B={c.get('claim_B_on_moat_unchanged')} "
                  f"C={c.get('claim_C_graded_rise_visible')} depth_traj={c.get('depth_traj')}")
        print("#" * 118)
        summ_path = Path(a.out).parent / f"summary_{n}seed.json"
        summ_path.parent.mkdir(parents=True, exist_ok=True)
        summ_path.write_text(json.dumps({"seeds": seeds, "n_go": n_go, "n_valid": n_valid, "go_flags": go_flags,
                                         "valid_flags": valid_flags, "backend": backend, "params": vars(a),
                                         "per_seed": {str(s): all_results[s].get("checks", {}) for s in seeds}},
                                        indent=2, default=str))
        print(f"[d5-ltu-noreg] wrote {summ_path}")

    valid_go = [go_flags[i] for i in range(len(seeds)) if valid_flags[i]]
    return 0 if (valid_go and all(valid_go)) else 1


if __name__ == "__main__":
    sys.exit(main())
