"""CONTINUOUS LEARN-WITHIN-A-CONVERSATION -- a minimal CLOSED learn-from-a-turn loop on ONE PERSISTENT brain.

THE PIECE OF THE CLOSED LOOP THIS DE-RISKS. INTEGRATION #7 (+ the one-brain merge, burn-down #1) gave the chat a
CO-RESIDENT e-prop acquisition net that LEARNS facts into the shared weights -- but its eval TEACHES every fact UP
FRONT (`_teach` before `run_chat`) and then reads a FIXED 14-turn script. It is NOT a loop where a fact told at turn
K changes the brain's reply at turn K+M via the on-substrate weight change. This de-risk closes that specific gap:

  a fact learned MID-CONVERSATION (turn K), by the brain's OWN e-prop plasticity on the SHARED spiking substrate,
  changes the brain's LATER reply (turn K+M) -- within the SAME persistent run, NO rebuild between K and K+M --
  while the no-confab MOAT still holds (it abstains on what it was NOT taught).

WHY THIS IS A REAL CLOSED LOOP (not the batch eval re-labelled). The SAME probe ("Tell me about the dax.") is issued
BEFORE the learning event and AFTER it, on the SAME persistent `SimulationBridge`:
  * SEGMENT 1 (turns 1..K-1): the target cue (dax, a genuinely-new referent absent from the curated kb) is queried
    and the brain is HONESTLY IGNORANT -> SILENCE (net untrained on it + the familiarity gate empty -> abstain).
  * LEARNING EVENT (turn K): the teacher tells dax->grass (+ dog->bone, cat->fish, the contrastive background).
    `_train_eprop` moves the co-resident net's readout weights -- a REAL e-prop weight change in the SAME
    `cp_connections` array as every conversational synapse.
  * SEGMENT 2 (turns K+1..K+M-1): M intervening OOD/small-talk turns run on the SAME bridge. run_chat's per-turn
    `_restore_state` washes the NEURON dynamical state (v/u/firing/conductances) but NOT `cp_connections.data`, so
    the weight change PERSISTS across the washing turns (the mechanism a continuous loop needs; asserted here as the
    readout-norm being byte-invariant K -> K+M).
  * QUERY (turn K+M): the same "Tell me about the dax." now answers "Dax eats grass." -- read from the LEARNED
    weight change (`AcquiredReadComposer.query_patient` consults the LIVE net). The only thing about the brain that
    changed between the silent turn and the answered turn is the on-substrate weight (+ the gate imprint).

THE CAUSAL / ANTI-CHEAT BATTERY (tools.lab + tools.verdict):
  * WITHIN-RUN DELTA          : taught-recall pre-teach == 0, post-teach >= 1 (headline dax; ideally == K) -- the
                                reply at K+M reflects the learning at K, same persistent bridge.
  * PERSISTENCE               : readout-norm(net) at K+M == readout-norm right after the teach (the weight survived
                                the M intervening washing turns -- the loop is genuinely continuous, not a re-teach).
  * LESION (load-bearing)     : restore the pre-teach FF weights (undo ONLY the turn-K e-prop weight change) -> the
                                K+M reply reverts (taught-recall -> 0; the taught patient is gone) -- the reply RODE
                                the learning, not a host buffer.
  * FROZEN control (no e-prop): a SECOND persistent run, identical teaching but eprop_lr=0 (weights never move, gate
                                still imprinted) -> the K+M query does NOT recall the taught fact -> the CONTENT rode
                                the weight change (the #7 KEY anti-cheat, carried to the continuous loop).
  * MOAT intact               : untaught cues (dax+chases, wug+eats + the battery) abstain; every OOD turn abstains;
                                0 confabulations -- throughout the persistent run, before AND after the learning.
  * kb-unchanged              : acquisition is a WEIGHT change, not a host `comp.store` append (len(comp.kb) fixed).
  * attributable              : the K+M recall is attributed to the mid-conversation weight change (treatment vs
                                frozen; treatment vs weight-lesion).

HONEST SCOPE (this is a PIECE, not the whole closed loop). The teacher/curriculum + the per-turn appraisal are the
LEGITIMATE host social environment (AI-teacher). The familiarity gate is a numpy anti-Hebbian projector (host-
idealized; the spiking v320 gate is the swap-in, burn-down #2). The conjunctive cue codebook + patient argmax are the
composer-idealization/neural-motor-readout targets. The learned-fact SET is the reliable K=3 joint-contrastive regime
(sequential/continual breadth is the OPEN continual-learning arc -- frac_recalled ~ 1/N; NOT re-litigated here). What
this de-risk adds over INTEGRATION #7: the learning event happens INSIDE the persistent conversation (not before it)
and its causal effect on a LATER reply, ON THE SAME BRIDGE WITH NO REBUILD, is measured + lesioned. If the mid-
conversation weight change did NOT persist or did NOT change the later reply, that is the first-class HONEST NEGATIVE
(it would name what the continuous loop needs next -- e.g. weight-state carryover or interference control).

DISCIPLINE: SIM_BACKEND=numpy, reuse-by-import (build_one_brain + CoResidentEpropNet + I7's teach/chat/moat/recall
machinery, all unchanged), NO `sim/` edit, cfg.seed (set by build_one_brain), additive. Single-seed SMOKE ->
VERDICT in ONE foreground process; the parent runs the 6-seed self-sweep.

Run (cheap-first single-seed SMOKE, foreground):
  PYTHONPATH=$PWD SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    .venv/bin/python -u -m research.runners._continuous_learn_in_conversation_derisk --smoke --seed 42
6-seed self-sweep (GO needs 6/6):
  PYTHONPATH=$PWD SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    .venv/bin/python -u -m research.runners._continuous_learn_in_conversation_derisk \
      --seeds 42,43,44,100,101,102 \
      --out research/findings/raw/lanes/stageA/continuous_learn_in_conversation_6seed.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
import logging as _logging  # noqa: E402
for _n in ("SIM_BRIDGE", "sim.bridge", "root"):
    _logging.getLogger(_n).setLevel(_logging.ERROR)

import numpy as np  # noqa: E402

from sim.backend import get_backend, to_host  # noqa: E402

# reuse-by-import: the ONE-brain builder + the co-resident e-prop net + INTEGRATION #7's teach/chat/moat/recall glue.
from research.runners import _stageA_full_integration_derisk as SA  # noqa: E402
from research.runners import _conversation_turing_test_derisk as TT  # noqa: E402
from research.runners import _corpus_facts_into_live_chat_derisk as CF  # noqa: E402
from research.runners import _teacher_loop_facts_into_live_chat_derisk as I7  # noqa: E402
from research.runners._i7_burndown1_one_brain_merge_derisk import _mk_merged_net  # noqa: E402
from research.runners._teacher_loop_contrastive_familiarity_moat_derisk import (  # noqa: E402
    _readout_norm, TAUGHT, HEADLINE_REFERENT,
)
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

V = I7.V                          # the #7 vocab (DEFAULT_VOCAB U the plasticity words)
HEAD = HEADLINE_REFERENT          # 'dax' -- the genuinely-NEW referent (absent from the curated kb)

# The persistent conversation is SEGMENTED around the learning event: some conversational turns, then the teach,
# then M intervening turns, then the query. All segments run on the SAME bridge/shim/snap (no rebuild) -> what
# persists (weights) carries across; the per-turn neuron-state wash is identical to one continuous run_chat.
PRE_TURNS = list(TT.HUMAN_TURNS[:7])       # turns 1..7 (small talk + in-domain) BEFORE the learning event
INTERVENING_TURNS = list(TT.HUMAN_TURNS[7:14])  # M=7 OOD/experiential turns BETWEEN the teach (K) and the query (K+M)


def _pre_probe():
    return [("Tell me about the %s." % HEAD,
             "PRE-TEACH subject probe (%s) -- must ABSTAIN (not taught yet, mid-run)" % HEAD)]


def _post_probes():
    # the headline new referent FIRST, then the contrastive background (dog/cat) -- all "subject probe" tagged so the
    # per-turn appraisal reads friendly (== I7._turns) and I7._probe_answer can pull each reply.
    order = [HEAD] + [r for r in TAUGHT if r != HEAD]
    return [("Tell me about the %s." % r,
             "POST-TEACH subject probe (%s) -- recall from the mid-conversation weight change" % r) for r in order]


def _ff_snapshot(bridge, net):
    """Copy the e-prop FF-slot weights (both pathways) out of the SHARED cp_connections -- the turn-K learning target."""
    return [np.asarray(to_host(bridge.cp_connections.data[s])).copy() for s in net._data_idx_flat]


def _ff_restore(bridge, net, snap):
    """Write FF-slot weights back into the SHARED cp_connections (used to LESION the turn-K weight change)."""
    xp, _ = get_backend()
    data = bridge.cp_connections.data
    for s, arr in zip(net._data_idx_flat, snap):
        data[s] = xp.asarray(arr)


def _learn_event(net, fam, env, seed):
    """THE MID-CONVERSATION LEARNING EVENT (turn K): the brain's OWN e-prop plasticity moves the persistent net's
    readout over the K=3 contrastive facts (a REAL weight change on the shared substrate) + the source-monitor
    imprints the taught cues. Returns |readout moved| (the acquisition signal; ~0 when frozen)."""
    for r in TAUGHT:
        fam.imprint(env, r, "eats")
    Xtr, ytr = I7._contrastive_batch(env, int(seed), I7.N_DRAWS)
    ro0 = _readout_norm(net)
    I7._train_eprop(net, Xtr, ytr, I7.EPOCHS, I7.BATCH, int(seed))
    return float(abs(_readout_norm(net) - ro0))


def run_continuous(seed, frozen=False):
    """One PERSISTENT continuous-loop run: build ONE brain, construct ONE co-resident e-prop net + ONE gate (never
    rebuilt), then run pre-teach -> learn -> intervening -> query as SEGMENTS on the SAME bridge. frozen=True sets
    eprop_lr=0 for the learning event (the no-e-prop control). Returns the full metric record."""
    xp, backend = get_backend()

    bridge, comp, idx, snap = SA.build_one_brain(int(seed), with_faculties=True, co_resident_affect_ladder=True,
                                                 vocab=V, co_resident_eprop=True,
                                                 eprop_dims=(I7.N_IN, I7.HIDDEN, I7.K))
    kb0 = len(comp.kb)
    _vc, curated = SA._store_facts(comp)               # the 6 curated host-stored facts (#6 baseline / moat context)
    kb_after_store = len(comp.kb)
    env = I7._make_env(int(seed))

    # ONE persistent co-resident e-prop net + ONE persistent familiarity gate (constructed ONCE, never rebuilt).
    net = _mk_merged_net(bridge, snap, idx["eprop"], int(seed), freeze=frozen)
    fam = I7._make_fam(int(seed))                       # empty gate (nothing taught yet)
    shim = I7.ChatShim(comp, env, net=net, fam=fam, enabled=True, use_gate=True)
    facts_all = list(curated) + list(I7.TAUGHT_FACTS)   # cue bookkeeping (dax/dog/cat classify as topics)

    # ---- SEGMENT 1: PRE-TEACH (turns 1..K-1) on the persistent bridge; the NEW cue must ABSTAIN (honest ignorance) ----
    tr_pre = CF.run_chat(bridge, xp, idx, snap, shim, facts_all, PRE_TURNS + _pre_probe())
    recall_pre, _ = I7._taught_recall(tr_pre)
    pre_reply = I7._probe_answer(tr_pre, HEAD)
    sum_pre = CF._chat_summary(tr_pre)

    # ---- LEARNING EVENT at turn K (mid-conversation): the on-substrate e-prop weight change ----
    ff_pre = _ff_snapshot(bridge, net)                  # snapshot the pre-teach FF weights (for the lesion)
    readout_moved = _learn_event(net, fam, env, int(seed))
    readout_after_teach = _readout_norm(net)
    kb_after_teach = len(comp.kb)

    # ---- SEGMENT 2: M INTERVENING turns + the POST-TEACH QUERY (turn K+M) -- NO rebuild between K and K+M ----
    tr_post = CF.run_chat(bridge, xp, idx, snap, shim, facts_all, INTERVENING_TURNS + _post_probes())
    recall_post, recalled = I7._taught_recall(tr_post)
    post_reply = I7._probe_answer(tr_post, HEAD)
    sum_post = CF._chat_summary(tr_post)
    readout_at_KplusM = _readout_norm(net)              # PERSISTENCE: must equal readout_after_teach (weights survived)

    # ---- MOAT at chat scale (treatment): the untaught cues the learned gate must abstain on ----
    moat_fa, moat_ex = I7._moat_battery(shim)

    # ---- LESION the turn-K weight change (treatment only): undo ONLY the e-prop FF change -> the K+M reply reverts ----
    lesion = None
    if not frozen:
        _ff_restore(bridge, net, ff_pre)               # restore the pre-teach FF weights (undo the learning)
        readout_after_lesion = _readout_norm(net)
        tr_les = CF.run_chat(bridge, xp, idx, snap, shim, facts_all, _post_probes())
        recall_les, _ = I7._taught_recall(tr_les)
        les_reply = I7._probe_answer(tr_les, HEAD)
        lesion = {
            "recall_after_weight_lesion": int(recall_les),
            "dax_reply_after_weight_lesion": les_reply,
            "readout_norm_after_lesion": readout_after_lesion,
            "reverts_to_ignorance": bool(recall_les == 0 and (TAUGHT[HEAD] not in (les_reply or ""))),
        }

    rec = {
        "seed": int(seed), "frozen": bool(frozen),
        "sim_backend": os.environ.get("SIM_BACKEND", "numpy"),
        "backend_module": type(backend).__module__ if backend is not None else str(type(xp).__module__),
        "num_neurons": int(bridge.core_config.num_neurons),
        "cfg_seed": int(bridge.core_config.seed),
        # the closed-loop headline
        "recall_pre_teach": int(recall_pre), "pre_teach_dax_reply": pre_reply,
        "recall_post_teach": int(recall_post), "post_teach_dax_reply": post_reply,
        "recalled_referents_post": recalled,
        # persistence of the weight change across the M intervening washing turns
        "readout_moved_at_teach": readout_moved,
        "readout_norm_after_teach": readout_after_teach, "readout_norm_at_K_plus_M": readout_at_KplusM,
        "weight_persisted_K_to_KplusM": bool(abs(readout_at_KplusM - readout_after_teach) < 1e-9),
        "n_intervening_turns_M": len(INTERVENING_TURNS),
        # moat / honesty
        "moat_false_accepts": int(moat_fa), "moat_examples": moat_ex[:3],
        "pre_confab": int(sum_pre["confabulated"]), "post_confab": int(sum_post["confabulated"]),
        "pre_ood_abstained": int(sum_pre["ood_abstained"]), "pre_ood_turns": int(sum_pre["ood_turns"]),
        "post_ood_abstained": int(sum_post["ood_abstained"]), "post_ood_turns": int(sum_post["ood_turns"]),
        "ungrounded_word_total": int(sum_pre["ungrounded_word_total"] + sum_post["ungrounded_word_total"]),
        # substrate discipline
        "kb_len_before": int(kb0), "kb_len_after_store": int(kb_after_store), "kb_len_after_teach": int(kb_after_teach),
        "kb_unchanged_by_teaching": bool(kb_after_teach == kb_after_store),
        "lesion": lesion,
    }
    return rec


def smoke_seed(seed):
    """The FULL single-seed closed-loop smoke: the treatment continuous run (with the weight-lesion) + the frozen
    control run, then the per-seed GO flags + attributions. Foreground, ONE process."""
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        treat = run_continuous(int(seed), frozen=False)
        frozen = run_continuous(int(seed), frozen=True)

    recall_pre = treat["recall_pre_teach"]
    recall_post = treat["recall_post_teach"]
    recall_frozen_post = frozen["recall_post_teach"]
    recall_lesion = treat["lesion"]["recall_after_weight_lesion"]

    # ---- attributions (tools.lab): the K+M recall rode the MID-CONVERSATION weight change ----
    attrib_vs_frozen = attributable_to(
        "K+M taught-recall rode the mid-conversation e-prop weight change (treatment vs frozen-readout, identical teaching)",
        float(recall_post), float(recall_frozen_post))
    attrib_vs_lesion = attributable_to(
        "K+M taught-recall rode the turn-K weight change (post-teach vs post weight-lesion, same persistent run)",
        float(recall_post), float(recall_lesion))
    attrib_within_run = attributable_to(
        "within-run learning delta (post-teach vs pre-teach recall, ONE persistent bridge)",
        float(recall_post), float(recall_pre))

    # ---- per-seed GO flags ----
    closed_loop_ok = bool(recall_pre == 0 and recall_post >= 1)          # silent before -> answered after (headline)
    full_recall = bool(recall_post == len(TAUGHT))                       # all K taught facts recalled at K+M
    persistence_ok = bool(treat["weight_persisted_K_to_KplusM"] and treat["readout_moved_at_teach"] > 1e-3)
    lesion_ok = bool(treat["lesion"]["reverts_to_ignorance"] and recall_lesion == 0)
    frozen_ok = bool(frozen["readout_moved_at_teach"] <= 1e-3 and recall_frozen_post == 0)
    moat_ok = bool(treat["moat_false_accepts"] == 0)
    honesty_ok = bool(treat["pre_confab"] == 0 and treat["post_confab"] == 0 and treat["ungrounded_word_total"] == 0
                      and treat["pre_ood_abstained"] == treat["pre_ood_turns"]
                      and treat["post_ood_abstained"] == treat["post_ood_turns"])
    kb_ok = bool(treat["kb_unchanged_by_teaching"])

    smoke_go = bool(closed_loop_ok and persistence_ok and lesion_ok and frozen_ok and moat_ok and honesty_ok and kb_ok)

    return {
        "seed": int(seed), "elapsed_s": round(time.time() - t0, 1),
        "sim_backend": treat["sim_backend"], "backend_module": treat["backend_module"],
        "num_neurons": treat["num_neurons"], "cfg_seed": treat["cfg_seed"],
        "recall_pre_teach": recall_pre, "recall_post_teach": recall_post,
        "recall_frozen_post": recall_frozen_post, "recall_after_weight_lesion": recall_lesion,
        "pre_teach_dax_reply": treat["pre_teach_dax_reply"], "post_teach_dax_reply": treat["post_teach_dax_reply"],
        "lesion_dax_reply": treat["lesion"]["dax_reply_after_weight_lesion"],
        "frozen_dax_reply": frozen["post_teach_dax_reply"],
        "readout_moved_at_teach": treat["readout_moved_at_teach"],
        "readout_moved_frozen": frozen["readout_moved_at_teach"],
        "weight_persisted_K_to_KplusM": treat["weight_persisted_K_to_KplusM"],
        "n_intervening_turns_M": treat["n_intervening_turns_M"],
        "moat_false_accepts": treat["moat_false_accepts"], "moat_examples": treat["moat_examples"],
        "kb_unchanged_by_teaching": treat["kb_unchanged_by_teaching"],
        "recall_attributable_to_weight_change_vs_frozen": attrib_vs_frozen,
        "recall_attributable_to_weight_change_vs_lesion": attrib_vs_lesion,
        "within_run_learning_delta": attrib_within_run,
        "GO_flags": {
            "closed_loop (pre==0 -> post>=1)": closed_loop_ok, "full_recall (post==K)": full_recall,
            "persistence (weight survives M turns + moved)": persistence_ok,
            "lesion_load_bearing (undo weight -> reverts)": lesion_ok,
            "frozen_control (no e-prop -> 0 recall)": frozen_ok,
            "moat_0_false_accepts": moat_ok, "honesty (0 confab, OOD abstains)": honesty_ok,
            "kb_unchanged": kb_ok,
        },
        "SMOKE_GO": smoke_go,
        "treatment": treat, "frozen": frozen,
    }


def _print_smoke(sm):
    print("=== CONTINUOUS LEARN-IN-CONVERSATION SMOKE (seed %d) ===" % sm["seed"], flush=True)
    print("  backend=%s (%s) | neurons=%d | cfg.seed=%d | %.1fs"
          % (sm["sim_backend"], sm["backend_module"], sm["num_neurons"], sm["cfg_seed"], sm["elapsed_s"]), flush=True)
    print("  CLOSED LOOP: pre-teach dax='%s' (recall=%d) -> [learn turn K] -> +%d turns -> K+M dax='%s' (recall=%d/%d)"
          % (sm["pre_teach_dax_reply"], sm["recall_pre_teach"], sm["n_intervening_turns_M"],
             sm["post_teach_dax_reply"], sm["recall_post_teach"], len(TAUGHT)), flush=True)
    print("  PERSISTENCE: weight moved=%.3f, survived K->K+M=%s"
          % (sm["readout_moved_at_teach"], sm["weight_persisted_K_to_KplusM"]), flush=True)
    print("  LESION (undo turn-K weights): dax='%s' recall=%d | FROZEN (no e-prop): dax='%s' recall=%d moved=%.4f"
          % (sm["lesion_dax_reply"], sm["recall_after_weight_lesion"], sm["frozen_dax_reply"],
             sm["recall_frozen_post"], sm["readout_moved_frozen"]), flush=True)
    print("  MOAT: false_accepts=%d %s | kb_unchanged=%s"
          % (sm["moat_false_accepts"], sm["moat_examples"], sm["kb_unchanged_by_teaching"]), flush=True)
    for k, v in sm["GO_flags"].items():
        print("    [%s] %s" % ("PASS" if v else "FAIL", k), flush=True)
    print("  SMOKE_GO: %s" % sm["SMOKE_GO"], flush=True)


def _build_verdict(recs, go):
    r0 = recs[0]
    v = Verdict("continuous learn-within-a-conversation (K=%d joint, %d seeds)" % (len(TAUGHT), len(recs)), chance=None)
    v.require("all seeds SMOKE_GO", int(sum(1 for r in recs if r["SMOKE_GO"])), expect=len(recs))
    v.require("closed loop: pre-teach recall == 0 (all seeds)",
              int(max(r["recall_pre_teach"] for r in recs)), expect=0)
    v.require("closed loop: post-teach recall >= 1 (all seeds)",
              int(min(r["recall_post_teach"] for r in recs)), expect=lambda m: m >= 1)
    v.control("within-run learning delta (post-teach vs pre-teach recall, same bridge)",
              r0["recall_post_teach"], r0["recall_pre_teach"], min_separation=0.0)
    v.require("PERSISTENCE: weight survives the M intervening washing turns (all seeds)",
              bool(all(r["weight_persisted_K_to_KplusM"] for r in recs)), expect=True)
    v.control("K+M recall: treatment vs FROZEN-readout (content rode the weight change)",
              r0["recall_post_teach"], r0["recall_frozen_post"], min_separation=0.0)
    v.control("K+M recall: post-teach vs weight-LESION (reply rode the turn-K weights)",
              r0["recall_post_teach"], r0["recall_after_weight_lesion"], min_separation=0.0)
    v.require("FROZEN control recalls 0 at K+M (all seeds)",
              int(max(r["recall_frozen_post"] for r in recs)), expect=0)
    v.require("weight-LESION reverts to ignorance at K+M (all seeds)",
              int(max(r["recall_after_weight_lesion"] for r in recs)), expect=0)
    v.require("moat 0 false-accepts (untaught cues, all seeds)",
              int(sum(r["moat_false_accepts"] for r in recs)), expect=0)
    v.require("kb unchanged by mid-conversation teaching (all seeds)",
              bool(all(r["kb_unchanged_by_teaching"] for r in recs)), expect=True)
    v.disabled("spiking familiarity gate (v320)",
               "the source-monitor is a numpy anti-Hebbian projector; the spiking v320 gate is the swap-in (burn-down #2)")
    v.disabled("sequential/continual breadth (frac_recalled ~ 1/N)",
               "the learned-fact SET is the reliable K=3 joint-contrastive regime; continual breadth is the open arc")
    return v.decide(go=bool(go), verbose=False)


def sweep(seeds):
    """SELF-SWEEP (the parent's ONE command): per seed run the full closed-loop smoke, aggregate. No per-seed
    orchestration by Claude."""
    per = []
    for sd in seeds:
        sm = smoke_seed(int(sd))
        per.append(sm)
        print("seed %d: SMOKE_GO=%s | pre-recall=%d post-recall=%d/%d frozen=%d lesion=%d | persisted=%s moat_fa=%d (%.1fs)"
              % (sd, sm["SMOKE_GO"], sm["recall_pre_teach"], sm["recall_post_teach"], len(TAUGHT),
                 sm["recall_frozen_post"], sm["recall_after_weight_lesion"], sm["weight_persisted_K_to_KplusM"],
                 sm["moat_false_accepts"], sm["elapsed_s"]), flush=True)
    n = len(per)
    go = bool(n == len(seeds) and all(r["SMOKE_GO"] for r in per))
    decided = _build_verdict(per, go)
    agg = {
        "probe": "continuous_learn_in_conversation", "seeds": list(seeds), "n_seeds": n,
        "n_smoke_go": int(sum(1 for r in per if r["SMOKE_GO"])),
        "GO_6of6": go, "verdict_earned": decided["status"],
        "sim_backend": per[0]["sim_backend"] if per else os.environ.get("SIM_BACKEND", "numpy"),
        "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
        "per_seed": [{k: r[k] for k in (
            "seed", "SMOKE_GO", "recall_pre_teach", "recall_post_teach", "recall_frozen_post",
            "recall_after_weight_lesion", "weight_persisted_K_to_KplusM", "readout_moved_at_teach",
            "readout_moved_frozen", "moat_false_accepts", "kb_unchanged_by_teaching",
            "pre_teach_dax_reply", "post_teach_dax_reply", "lesion_dax_reply", "frozen_dax_reply",
            "GO_flags", "elapsed_s")} for r in per],
    }
    return agg


def main():
    ap = argparse.ArgumentParser(description="Continuous learn-within-a-conversation: a fact taught mid-chat changes "
                                             "the brain's later reply on ONE persistent spiking bridge.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None, help="comma-separated -> SELF-SWEEP, aggregated + earned verdict")
    ap.add_argument("--smoke", action="store_true", help="run the 1-seed full closed-loop smoke, then exit")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.seeds:
        seeds = [int(x) for x in args.seeds.replace(",", " ").split()]
        agg = sweep(seeds)
        print("\n=== %d-SEED SELF-SWEEP AGGREGATE ===" % len(seeds), flush=True)
        print(json.dumps({k: agg[k] for k in ("n_seeds", "n_smoke_go", "GO_6of6", "verdict_earned")}, indent=2), flush=True)
        if args.out:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as fh:
                json.dump(agg, fh, indent=2, default=str)
            print("[saved] %s" % args.out, flush=True)
        return 0 if agg["GO_6of6"] else 1

    sm = smoke_seed(int(args.seed))
    _print_smoke(sm)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(sm, fh, indent=2, default=str)
        print("[saved] %s" % args.out, flush=True)
    return 0 if sm["SMOKE_GO"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
