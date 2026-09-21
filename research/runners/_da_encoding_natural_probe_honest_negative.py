"""NATURAL-PROBE honest-negative test for the DA-GATED ENCODING coupling (board load-bearing lane, gap-da-gated-
encoding-v2). Is there ANY NATURAL conversational condition -- no tuned read-damage sigma sweep, no induce-then-sweep
DA -- where DA-gated encoding CHANGES THE PRODUCED REPLY (the load-bearing definition: lesioning the brain's
contribution changes the reply)?

CONTEXT (why this runner exists). A first fix (branch research/hollow-da-gated-encoding-drive) made da-gated-encoding
read "load-bearing" ONLY under a SWEPT read-damage operating point (BRAIN_ONEBRAIN_RETRIEVE_DAMAGE_SIGMA ascending
across a knee) + induced arousal -- a TUNED condition, rejected as metric-tuning. This runner asks the honest question
the rejection implies: does the coupling bite on a CLEAN, natural recall at all?

THE MECHANISM under test (from the rejected diagnosis, re-verified by reading one_brain_composer.py):
  * DA-gated encoding scales the STORED MAGNITUDE of a trace at write time: `_write_block` writes `complex(g)*zc[k]`
    for every k in the block (a COMMON scale g over the whole D-run of that fact's readout synapses).
  * A CLEAN recall on the production-default magnitude-carrying OneBrainComposer is MAGNITUDE-INVARIANT at every
    decision point:
      - `_read_block` kicks ONE block's trigger, resonates, unbinds, cleans up -> membrane `scores` that all scale
        by ~g (a common factor).
      - `_select` = argmax over scores (invariant to a common positive scale) OR `_spiking_select`, whose drive is
        `(scores/peak)*_cleanup_drive_pA` -- PEAK-NORMALIZED, so the WTA winner is invariant to g.
      - `_margin` = (peak - runner_up)/(peak+eps) -- a RATIO, so the confidence_gate decision is invariant to g.
      - `query_role`/`query_patient`/`query_agent` return the FIRST cue-matching block (POSITIONAL first-match), never
        a magnitude-weighted competition among matching facts.
      - blocks are per-block-independent (store_base + i*block; each fact = its own trigger + D readout neurons), so
        adding facts does NOT degrade a given fact's read SNR -- there is no natural interference-based degradation.
  * The ONLY magnitude-sensitive element is the ABSOLUTE RF read floor (sim/bridge.py:5589 `_rf_mag2 > ...`): a block
    whose readout magnitude falls BELOW the floor decodes to noise -> abstain. A clean read at g>=1 never approaches
    it, and the DEFAULT-ON homeostatic floor (webapp/da_encoding_drives_chat: g_floor = 1.0) keeps every write at or
    above unit -> above the floor. Crossing the floor requires DAMAGING the read (the rejected tuned knob) or writing
    BELOW unit (which default homeostasis prevents).

WHAT THIS RUNNER PROVES (a real brain build; SIM_BACKEND=numpy; the production-default OneBrainComposer store_conns):
  TEST-1  STORE-STRENGTH -> CLEAN-RECALL INVARIANCE. Store the same fact-set at a range of encoding gains spanning the
          ENTIRE natural gain-map range g in {1.0(tonic/unengaged==lesion-pin), 1.5, 2.0, 3.0(g_max, a maximally
          salient turn)} and do a CLEAN recall (query_patient/query_agent/query_role -- NO damage knob, NO
          confidence-gate change, NO sweep-to-flip). The recalled SVO must be IDENTICAL across ALL gains. This is an
          INVARIANCE demonstration (the opposite of tuning-until-it-flips): if even the CEILING boost g=3.0 does not
          change the clean recall, no natural DA level can. The STORED |w| ratio is reported alongside to prove the
          WRITE genuinely differs (mean|w| of block == g) -- the coupling IS load-bearing on the STORE, invisible on
          the clean READ.
  TEST-2  MULTI-FACT COMPETITION. Store two facts sharing a cue (agent+action) but differing in the queried role, one
          SALIENT (g_high) one UNIT, then query the shared cue. Repeat with the store ORDER swapped. If recall were
          magnitude-arbitrated the SALIENT fact would win in BOTH orders; if it is positional first-match the winner
          FOLLOWS THE ORDER. This directly tests "do salient memories win natural competition" on this substrate.

VERDICT. LOAD-BEARING (natural) iff a clean recall reply DIFFERS between a salient and a unit store WITHOUT any tuned
read-damage. Otherwise HONEST-NEGATIVE: da-gated-encoding is an encoding-magnitude effect invisible to a clean read;
its behavioral bite requires a degraded/stress-gated recall the natural conversational path never constructs. The
honest-negative is a genuine deliverable (it maps what the substrate does on its own; brain-based-only; the write-side
coupling remains a real, validated, on-store mechanism -- see 2026-08-21-da-gated-encoding-wired-into-chat-GO).

Run (numpy-CPU, foreground, ~a minute; ALWAYS under memcap):
  tools/memcap.sh 12 -- .venv/bin/python -u -m research.runners._da_encoding_natural_probe_honest_negative \
      --out research/findings/raw/_da_encoding_natural_probe/verdict.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np  # noqa: E402

from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

# a small, clean vocab (all tiny-demo content words); the facts below are built ONLY from these.
VOCAB = ["dog", "cat", "bird", "worm", "fish", "grass", "eat", "chase", "run", "see"]
# the natural gain-map RANGE end-to-end: tonic/unengaged (== the lesion pin g=1.0) .. g_max (a maximally salient turn).
# NOT a search grid -- the whole point is to show the clean recall is INVARIANT across ALL of it.
GAINS = [1.0, 1.5, 2.0, 3.0]
G_MAX = 3.0


def _build(seed=42, D=64):
    """A production-default-shaped OneBrainComposer (the magnitude-carrying store_conns path -- the composer the
    load-bearing harness builds via _COMPOSER_KIND_DEFAULT='onebrain'). Defaults left at production (enable_spiking_
    cleanup=True, persistent_loop=True, confidence_gate=0.0)."""
    from research.runners.one_brain_composer import OneBrainComposer
    return OneBrainComposer(seed=seed, D=D, vocab=list(VOCAB), k_max=16)


def _store_mean_mag(comp, block_idx):
    """The mean |w| of stored block `block_idx` on the composer's own store_conns (== the encoding gain g for that
    fact; a unit write has mean|zc|==1.0). Direct read of the production magnitude-carrying store."""
    D = comp.D
    sl = comp.store_conns[block_idx * D:(block_idx + 1) * D]
    if not sl:
        return None
    return float(np.mean([abs(complex(w)) for (_p, _q, w) in sl]))


def _clean_recall(comp, facts):
    """Every role of every fact, decoded on a CLEAN read (no damage, no confidence-gate change). Returns a stable,
    order-independent view: for each fact's (agent, action) cue the recalled patient (query_patient) and, for the
    (action, patient) cue, the recalled agent (query_agent) -- the two production recall entry points -- plus a
    full-block decode via query_role for every role. This is the 'produced reply' proxy at the composer layer."""
    out = {}
    for (a, act, p) in facts:
        patient = comp.query_patient(a, act)
        agent = comp.query_agent(act, p)
        role_all = {r: comp.query_role(r, agent=a, action=act) for r in ("agent", "action", "patient")}
        out[f"{a}|{act}|{p}"] = {"query_patient": patient, "query_agent": agent, "query_role": role_all}
    return out


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=repr).encode()).hexdigest()[:16]


def test1_store_strength_invariance(seed=42):
    """TEST-1: store the same fact-set at each natural gain; CLEAN recall must be identical across ALL gains."""
    facts = [("dog", "chase", "cat"), ("bird", "eat", "worm")]
    arms = {}
    for g in GAINS:
        comp = _build(seed=seed)
        comp.encoding_gain_fn = (lambda gg=g: gg)   # a salient turn writes at gg; g=1.0 == unengaged == the lesion pin
        for f in facts:
            comp.store(*f)
        recall = _clean_recall(comp, facts)
        mags = [_store_mean_mag(comp, i) for i in range(len(facts))]
        arms[f"g={g}"] = {"recall": recall, "recall_hash": _hash(recall),
                          "stored_mean_mag": mags}
    base_h = arms[f"g={GAINS[0]}"]["recall_hash"]    # g=1.0 (unengaged / lesion pin) is the baseline
    all_identical = all(arm["recall_hash"] == base_h for arm in arms.values())
    # the write genuinely differs across gains (ratio of stored mag at g_max vs g=1 ~= g_max) -> the coupling is not
    # vacuous on the STORE, only on the clean READ.
    mag_lo = arms[f"g={GAINS[0]}"]["stored_mean_mag"][0]
    mag_hi = arms[f"g={G_MAX}"]["stored_mean_mag"][0]
    write_ratio = (mag_hi / mag_lo) if (mag_lo and mag_hi) else None

    # the reply-change SCALAR the load-bearing question turns on: how many decoded (fact, role) words differ on a
    # CLEAN recall between the SALIENT store (g_max, intact) and the UNIT store (g=1, == the lesion pin)? (treatment)
    # vs the same count for a UNIT rebuild (deterministic control -> 0). This is the pair the attribution reads.
    def _divergence(recall_a, recall_b):
        n = 0
        for fk in recall_a:
            ra, rb = recall_a[fk]["query_role"], recall_b[fk]["query_role"]
            n += sum(1 for r in ra if ra.get(r) != rb.get(r))
        return n
    recall_salient = arms[f"g={G_MAX}"]["recall"]
    recall_unit = arms[f"g={GAINS[0]}"]["recall"]
    recall_unit_rebuild = _clean_recall(_apply_gain_and_store(seed, GAINS[0], facts), facts)
    div_salient_vs_unit = _divergence(recall_salient, recall_unit)     # treatment: the DA-boost's reply effect
    div_unit_vs_rebuild = _divergence(recall_unit, recall_unit_rebuild)  # control: deterministic-rebuild null
    return {
        "facts": facts, "gains": GAINS, "arms": arms,
        "clean_recall_identical_across_all_gains": bool(all_identical),
        "write_mag_ratio_gmax_over_unit": write_ratio,
        "write_genuinely_differs": bool(write_ratio is not None and write_ratio > 1.5),
        "reply_divergence_salient_vs_unit": div_salient_vs_unit,       # the intact-vs-lesion reply-change count
        "reply_divergence_unit_vs_rebuild": div_unit_vs_rebuild,       # the null control
        "stored_mag_salient": mag_hi, "stored_mag_unit": mag_lo,
    }


def _apply_gain_and_store(seed, g, facts):
    comp = _build(seed=seed)
    comp.encoding_gain_fn = (lambda gg=g: gg)
    for f in facts:
        comp.store(*f)
    return comp


def test2_competition_is_positional(seed=42):
    """TEST-2: two facts share the (agent, action) cue, differ in patient; one salient, one unit. The shared cue's
    recall follows STORE ORDER (positional first-match), not magnitude -- so a salient memory does NOT win a natural
    competition on this substrate."""
    fA = ("dog", "chase", "cat")     # salient candidate
    fB = ("dog", "chase", "bird")    # unit candidate
    g_high = G_MAX

    def _run(order_salient_first):
        comp = _build(seed=seed)
        holder = {"g": 1.0}
        comp.encoding_gain_fn = (lambda: holder["g"])
        if order_salient_first:
            holder["g"] = g_high; comp.store(*fA)     # A salient, stored FIRST
            holder["g"] = 1.0;    comp.store(*fB)     # B unit,    stored SECOND
        else:
            holder["g"] = 1.0;    comp.store(*fB)     # B unit,    stored FIRST
            holder["g"] = g_high; comp.store(*fA)     # A salient, stored SECOND
        winner = comp.query_patient("dog", "chase")   # the shared cue -> whichever block wins
        mags = {"block0": _store_mean_mag(comp, 0), "block1": _store_mean_mag(comp, 1)}
        return winner, mags

    win_salient_first, mag_sf = _run(True)    # A(salient) stored first -> expect 'cat' (A) if positional
    win_unit_first, mag_uf = _run(False)      # B(unit)   stored first -> expect 'bird' (B) if positional
    # positional => the winner FOLLOWS ORDER (first-stored wins), so the two runs give DIFFERENT winners even though
    # the SALIENT fact (A='cat') is the same in both. magnitude-arbitrated would give 'cat' (A) in BOTH.
    is_positional = (win_salient_first == "cat" and win_unit_first == "bird")
    is_magnitude = (win_salient_first == "cat" and win_unit_first == "cat")
    return {
        "fact_salient": fA, "fact_unit": fB, "g_high": g_high,
        "winner_when_salient_stored_first": win_salient_first, "stored_mags_that_order": mag_sf,
        "winner_when_unit_stored_first": win_unit_first, "stored_mags_that_order": mag_uf,
        "recall_is_positional_first_match": bool(is_positional),
        "recall_is_magnitude_arbitrated": bool(is_magnitude),
    }


def test3_end_to_end_reply(induce=None):
    """END-TO-END through the REAL webapp.server.brain_chat handler on the tiny-demo brain (onebrain composer, the
    magnitude-carrying production default). A natural store->recall pair in one session; compare the PRODUCED REPLY
    intact (da-encoding armed, natural self-produced DA -- NO induce by default) vs BRAIN_DA_ENCODING_LESION=1 (the
    DA->encoding-gain link severed, g pinned to 1.0). load-bearing would require the recall reply to DIFFER; identical
    -> the coupling does not change the produced reply on a natural conversational turn. (This is the exact intact-vs-
    lesion comparison the load-bearing harness runs, made explicit as a store->recall pair.)"""
    # do NOT set BRAIN_COMPOSER_KIND -> tiny-demo uses _COMPOSER_KIND_DEFAULT='onebrain' (magnitude-carrying store).
    quiet = {
        "BRAIN_AFFECT": "0", "BRAIN_WORLDMODEL": "0", "BRAIN_SURPRISE": "0", "BRAIN_METACOG": "0",
        "BRAIN_MULTIREF": "0", "BRAIN_NONCONTRADICTION_GATE": "0", "BRAIN_RECONSOLIDATION": "0",
        "BRAIN_EPISODIC_STORE": "0", "BRAIN_CURIOSITY": "0", "BRAIN_RICH": "0", "BRAIN_GNW_BUS": "0",
        "BRAIN_CONTINUOUS": "0", "BRAIN_CONTINUOUS_DRIVES": "0", "BRAIN_SWAP_DRIVES": "0",
    }

    def _arm(session, lesion):
        from webapp.server import brain_chat, BrainChatRequest as Req
        for k, v in quiet.items():
            os.environ[k] = v
        os.environ["BRAIN_DA_ENCODING"] = "1"                                  # coupling armed (intact arm)
        if lesion:
            os.environ["BRAIN_DA_ENCODING_LESION"] = "1"                       # sever DA->encoding-gain (g pinned 1.0)
        else:
            os.environ.pop("BRAIN_DA_ENCODING_LESION", None)
        if induce is not None:
            os.environ["BRAIN_DA_DRIVES_INDUCE"] = str(induce)
        else:
            os.environ.pop("BRAIN_DA_DRIVES_INDUCE", None)

        def _turn(msg):
            r = brain_chat(Req(session=session, message=msg, brain="tiny-demo", renderer="stub", rich=False))
            return json.loads(bytes(r.body).decode("utf-8"))

        store = _turn("the wolf bites the apple")                             # a natural teach turn
        recall = _turn("what does the wolf bite")                            # a plain later recall
        g_at_store = (store.get("da_encoding") or {}).get("g")
        return {"store_g": g_at_store,
                "recall_answer": recall.get("answer"),
                "recall_svo": recall.get("recalled_svo"),
                "recall_abstained": recall.get("abstained")}

    intact = _arm("dae_e2e_intact", lesion=False)
    lesion = _arm("dae_e2e_lesion", lesion=True)
    reply_identical = (intact["recall_answer"] == lesion["recall_answer"]
                       and intact["recall_svo"] == lesion["recall_svo"]
                       and intact["recall_abstained"] == lesion["recall_abstained"])
    return {"induce": induce, "intact": intact, "lesion": lesion,
            "produced_reply_identical_intact_vs_lesion": bool(reply_identical)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="research/findings/raw/_da_encoding_natural_probe/verdict.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--handler", action="store_true",
                    help="ALSO run the end-to-end brain_chat handler test (onebrain recall is ~minutes/turn on numpy; "
                         "prefer cupy). Default OFF: TEST-1's g=1(lesion-pin)-vs-g>1(salient) clean recall on the "
                         "production OneBrainComposer IS the intact-vs-lesion comparison at the reply-determining layer.")
    args = ap.parse_args()

    t1 = test1_store_strength_invariance(seed=args.seed)
    t2 = test2_competition_is_positional(seed=args.seed)
    t3 = None
    if args.handler:
        try:
            t3 = test3_end_to_end_reply()
        except Exception as e:
            t3 = {"error": f"{type(e).__name__}: {e}"}
    else:
        t3 = {"skipped": "opt-in via --handler (onebrain handler recall ~minutes/turn on numpy; run on cupy). "
              "TEST-1 g=1(lesion-pin) vs g>1(salient) clean recall on the production OneBrainComposer is the "
              "intact-vs-lesion comparison at the reply-determining layer; handler-level recall byte-identity is "
              "additionally the existing GO research/findings/2026-08-21-da-gated-encoding-wired-into-chat-GO.md."}

    # ── ATTRIBUTION (tools.lab): whose is the effect? The DA boost's effect is ENTIRELY in the STORE, NONE in the
    #    clean-read reply. (a) the store-magnitude effect is real; (b) the reply effect is a genuine NULL. ──
    store_attr = attributable_to("DA-encoding boost -> STORED magnitude (salient vs unit)",
                                 treatment_value=t1["stored_mag_salient"], control_value=t1["stored_mag_unit"])
    reply_attr = attributable_to("DA-encoding boost -> CLEAN-READ reply change (salient vs unit vs rebuild null)",
                                 treatment_value=t1["reply_divergence_salient_vs_unit"],
                                 control_value=t1["reply_divergence_unit_vs_rebuild"])   # 0 vs 0 -> UNDEFINED null

    # ── VERDICT (tools.verdict): the honest-negative must travel with what earned it (the validity preconditions). ──
    v = Verdict("da-gated-encoding load-bearing on a NATURAL conversational probe")
    v.require("the write genuinely differs (mag ratio > 1.5) — the coupling is wired, not a dead knob",
              t1["write_genuinely_differs"], expect=True,
              note="a negative from a vacuous no-op would be void; the store effect IS real (ratio %.2f)"
                   % (t1["write_mag_ratio_gmax_over_unit"] or float("nan")))
    v.require("the read is CLEAN (no read-damage knob set)",
              os.environ.get("BRAIN_ONEBRAIN_RETRIEVE_DAMAGE_SIGMA") in (None, "", "0"), expect=True,
              note="the rejected first fix needed a swept BRAIN_ONEBRAIN_RETRIEVE_DAMAGE_SIGMA; this probe sets none")
    v.require("clean recall IDENTICAL across the whole natural gain range (invariance, not sweep-to-flip)",
              t1["clean_recall_identical_across_all_gains"], expect=True)
    v.require("multi-fact competition is POSITIONAL first-match (not magnitude-arbitrated)",
              t2["recall_is_positional_first_match"], expect=True)
    v.require("the reply-change null control reproduces (unit vs deterministic rebuild == 0)",
              t1["reply_divergence_unit_vs_rebuild"] == 0, expect=True)
    v.disabled("read-damage sweep", "no BRAIN_ONEBRAIN_RETRIEVE_DAMAGE_SIGMA — the effect is measured on a CLEAN read")
    v.disabled("induced-then-swept DA", "no DA sweep; store strength is set across the natural gain-map range directly")
    # go=False -> NO-GO on "is it load-bearing on a natural probe" == the HONEST-NEGATIVE (all preconditions hold).
    decided = v.decide(go=(not t1["clean_recall_identical_across_all_gains"]))

    # VERDICT. natural-load-bearing would require a clean-recall DIFFERENCE across store strength with NO tuned damage,
    # OR the end-to-end produced reply to differ intact-vs-lesion on a natural turn.
    t3_ran = bool(t3 and "error" not in t3 and "skipped" not in t3)
    t3_reply_differs = bool(t3_ran and not t3.get("produced_reply_identical_intact_vs_lesion", True))
    natural_flip = ((not t1["clean_recall_identical_across_all_gains"])
                    or t2["recall_is_magnitude_arbitrated"] or t3_reply_differs)
    t3_ok = bool(t3_ran and t3.get("produced_reply_identical_intact_vs_lesion"))
    honest_negative = (t1["clean_recall_identical_across_all_gains"]
                       and t1["write_genuinely_differs"]
                       and t2["recall_is_positional_first_match"]
                       and (not t3_ran or t3_ok))
    verdict = ("NATURAL-LOAD-BEARING" if natural_flip
               else ("HONEST-NEGATIVE" if honest_negative else "INCONCLUSIVE"))

    result = {
        "coupling": "DA-gated encoding (encoding_gain_fn <- live self-produced tonic DA), production-default onebrain store",
        "question": "does DA-gated encoding change the produced reply on a NATURAL (clean, un-tuned) conversational recall?",
        "backend": os.environ.get("SIM_BACKEND"),
        "test1_store_strength_clean_recall": t1,
        "test2_multifact_competition": t2,
        "test3_end_to_end_handler_reply": t3,
        "attribution": {
            "store_magnitude_effect_fraction": store_attr,           # ~0.67: the boost DOES own the stored magnitude
            "clean_read_reply_effect_fraction": reply_attr,          # None (UNDEFINED null): no reply effect to attribute
        },
        "preconditions": decided["preconditions"],                   # gate: a verdict must travel with what earned it
        "disabled_processes": decided["disabled_processes"],
        "verdict_status": decided["status"],
        "verdict": verdict,
        "natural_flip_found": bool(natural_flip),
        "honest_negative": bool(honest_negative),
        "interpretation": (
            "The write-side coupling is real (mean|w| of a salient block == g, ratio ~%.2f at g_max), but a CLEAN "
            "recall on the production-default magnitude-carrying OneBrainComposer is magnitude-invariant at every "
            "decision point (peak-normalized WTA, ratio margin, argmax, positional first-match), so store strength "
            "does not change the reply. da-gated-encoding is NOT load-bearing on a natural conversational probe: it "
            "is an encoding-magnitude effect invisible to a clean read; its bite requires a degraded/stress-gated "
            "recall (a read-damage knee) the natural path never constructs -- a genuine honest-negative."
            % (t1["write_mag_ratio_gmax_over_unit"] or float("nan"))
        ),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=repr)

    print("=" * 100, flush=True)
    print("  DA-GATED ENCODING — NATURAL-PROBE honest-negative test (production-default onebrain store, numpy-CPU)", flush=True)
    print("=" * 100, flush=True)
    print(f"  TEST-1 clean recall IDENTICAL across all natural gains {GAINS}: "
          f"{t1['clean_recall_identical_across_all_gains']}", flush=True)
    print(f"         (write genuinely differs: mean|w| ratio g_max/unit = "
          f"{t1['write_mag_ratio_gmax_over_unit']:.3f})", flush=True)
    for k, arm in t1["arms"].items():
        print(f"           {k}: recall_hash={arm['recall_hash']} stored_mean_mag={['%.3f'%m for m in arm['stored_mean_mag']]}", flush=True)
    print(f"  TEST-2 competition is POSITIONAL first-match (not magnitude): "
          f"{t2['recall_is_positional_first_match']}", flush=True)
    print(f"         winner salient-stored-first={t2['winner_when_salient_stored_first']!r}  "
          f"unit-stored-first={t2['winner_when_unit_stored_first']!r}  "
          f"(magnitude-arbitrated would be 'cat' in BOTH)", flush=True)
    if t3 is not None:
        if "skipped" in t3:
            print("  TEST-3 end-to-end handler reply: SKIPPED (opt-in via --handler; see note in artifact)", flush=True)
        elif "error" in t3:
            print(f"  TEST-3 end-to-end handler reply: ERROR ({t3['error']})", flush=True)
        else:
            print(f"  TEST-3 produced REPLY identical intact-vs-lesion (real brain_chat, onebrain): "
                  f"{t3['produced_reply_identical_intact_vs_lesion']}", flush=True)
            print(f"         intact:  g@store={t3['intact']['store_g']} svo={t3['intact']['recall_svo']} "
                  f"answer={t3['intact']['recall_answer']!r}", flush=True)
            print(f"         lesion:  g@store={t3['lesion']['store_g']} svo={t3['lesion']['recall_svo']} "
                  f"answer={t3['lesion']['recall_answer']!r}", flush=True)
    print("-" * 100, flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  artifact: {args.out}", flush=True)
    print("=" * 100, flush=True)
    return 0 if verdict in ("HONEST-NEGATIVE", "NATURAL-LOAD-BEARING") else 1


if __name__ == "__main__":
    sys.exit(main())
