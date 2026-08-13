"""Burn-down B3 — the NON-CONTRADICTION MOAT fires on the PRODUCTION default (OneBrainComposer).

CONTEXT (the moat hole B3 names). The production conversational composer is `OneBrainComposer`
(composer_kind="onebrain"). B3's premise was that "the onebrain composer doesn't store negations
retrievably, so the non-contradiction check only fires on rf". On reading the code that premise is
STALE: `OneBrainComposer` already binds a POLARITY role (AFFIRM/NEGATE, `pol_words`) as a 4th role
and decodes it from the SPIKING-substrate cleanup membrane (`_decode_batched_mem` ->
`_select(ps, self.pol_words)`), and `ask_yes_no(a,v,p)` already returns yes/no/unknown from that
substrate read. The `_contradicts` gate used on the brain-GENERATION path is literally
`composer.ask_yes_no(a,ac,p) == "no"`.

THE GENUINE RESIDUAL GAP (what is actually missing at production). Two things, both real:
  (1) the production first-chat console stores ZERO negated facts (`negated = []` in
      first_chat_console.py), so the gate has nothing to fire against; and
  (2) there is NO non-contradiction gate on the USER-ASSERTION path -- when a user asserts a fact,
      the console `hear`/`store` just stores it; a user asserting "the dog eats grass" (AFFIRM) when
      the brain was told "a dog does NOT eat grass" (NEGATE) is NOT caught.

This de-risk closes that residual: a non-contradiction gate on the ASSERTION path, driven by the
onebrain substrate's own polarity recall. The LOAD-BEARING recall (the stored negation) is on the
spiking substrate; the gate itself is the SAME thin host comparison the project already accepts as
the moat (`_contradicts` == `ask_yes_no == "no"`) -- one boolean, `stored_polarity != asserted`.

MECHANISM (brain-based). On an incoming assertion (SVO, polarity):
    yn = composer.ask_yes_no(agent, action, patient)     # spiking substrate polarity recall
    if yn == "unknown":  accept   (novel -- no stored belief; the no-confab moat: never fabricate)
    stored = AFFIRM if yn == "yes" else NEGATE
    if stored != asserted:  REJECT (contradicts a stored belief)   else accept (consistent)

PRE-REGISTERED GO GATE (all must hold on ALL 6 seeds 42/43/44/100/101/102):
  INTACT onebrain:
    * recall_neg_ok == 1.0 AND recall_aff_ok == 1.0   (negations + affirmations recalled on substrate)
    * moat_false_accepts == 0        (every contradicting assertion REJECTED -- 0 slips through)
    * over_blocks == 0               (every consistent restatement + every novel assertion ACCEPTED)
    * n_rejections == n_contradictions  (the gate actually FIRED -- not inert)
  LESION (disable negation storage: store all facts AFFIRM):
    * moat_false_accepts_lesion > 0  (contradictions now slip through -> the gate goes INERT).
      VERIFY: ask_yes_no on a would-be-NEGATE fact reads "yes" in the lesion (the negation is
      genuinely gone -> the false-accept is REAL, not a bug). LOAD-BEARING.
  ANTI-CHEAT no-store (store nothing):
    * n_rejections_nostore == 0      (the gate CANNOT reject without a stored belief -> the
      rejection is driven by the stored negation, not a fixed template).
  ANTI-CHEAT shuffle (permute which facts are negated):
    * shuffle_recall_ok == 1.0       (ask_yes_no tracks the SHUFFLED store, not a memorized answer)
    * shuffle_tracks_store == 1.0    (every gate decision == (shuffled_pol != asserted_pol))
    * shuffle_reject_set != intact_reject_set  (the reject set MOVES with the store -> not a template)

numpy-CPU only (SIM_BACKEND=numpy). NO sim/ edit (reuse-by-import). Uses tools.lab.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("NEURAL_SIM_DISABLE_LLM", "1")
import argparse
import json
import numpy as np

from research.runners.one_brain_composer import OneBrainComposer
from tools.lab import lever, void_if, assert_backend
from tools.verdict import Verdict

SEEDS = [42, 43, 44, 100, 101, 102]

VOCAB = ["dog", "cat", "bird", "wolf", "child", "fish",
         "eat", "chase", "fear", "like", "see", "hear",
         "grass", "meat", "bone", "water", "sky", "ball"]

# 6 canonical facts, 3 NEGATE / 3 AFFIRM. Fact 0 is B3's exact scenario: "a dog does NOT eat grass".
FACTS = [
    ("dog", "eat", "grass", "NEGATE"),      # the canonical B3 scenario
    ("cat", "eat", "fish", "AFFIRM"),
    ("wolf", "chase", "dog", "AFFIRM"),
    ("bird", "fear", "cat", "NEGATE"),
    ("child", "like", "ball", "AFFIRM"),
    ("fish", "see", "water", "NEGATE"),
]
# Novel (unstored) assertions -- the gate must ACCEPT these (no basis to contradict; the no-confab
# moat abstains from a rejection it cannot justify). Cue pairs deliberately absent from the store.
NOVEL = [
    ("dog", "chase", "ball", "AFFIRM"),
    ("wolf", "eat", "meat", "NEGATE"),
    ("child", "see", "sky", "AFFIRM"),
    ("bird", "like", "water", "NEGATE"),
]

FLIP = {"AFFIRM": "NEGATE", "NEGATE": "AFFIRM"}


def _build(seed, D):
    return OneBrainComposer(seed=seed, D=D, k_max=16, vocab=list(VOCAB))


def _store(comp, facts):
    """Ingest facts (agent, action, patient, polarity) on the onebrain substrate. The canonical
    fact 0 goes through the COMPREHENSION path (`hear`, active voice) to show a heard negation stores;
    the rest through `store` (resolved-role API parity). polarity is the passed tag either way."""
    for i, (a, v, p, pol) in enumerate(facts):
        if i == 0:
            comp.hear(f"{a} {v} {p}", voice="active", polarity=pol)
        else:
            comp.store(a, v, p, polarity=pol)


def _assert_gate(comp, agent, action, patient, asserted_polarity):
    """The NON-CONTRADICTION gate on the assertion path. Returns (decision, recalled_yn) where
    decision in {"accept","reject"}. The load-bearing recall (ask_yes_no) is on the spiking
    substrate; the comparison is the thin host boolean the project already accepts as the moat."""
    yn = comp.ask_yes_no(agent, action, patient)          # <-- SPIKING SUBSTRATE polarity recall
    if yn == "unknown":
        return "accept", yn                                # novel -- no stored belief to contradict
    stored = "AFFIRM" if yn == "yes" else "NEGATE"
    if stored != asserted_polarity:
        return "reject", yn                                # contradicts a stored belief -> REJECT
    return "accept", yn                                    # consistent restatement -> accept


def _measure(comp, facts):
    """Run the recall + non-contradiction battery against `comp` (already loaded with `facts`).
    Returns a metrics dict. Contradictions = each fact asserted with the FLIPPED polarity."""
    # --- recall on the substrate ---
    neg = [f for f in facts if f[3] == "NEGATE"]
    aff = [f for f in facts if f[3] == "AFFIRM"]
    recall_neg = sum(1 for (a, v, p, _) in neg if comp.ask_yes_no(a, v, p) == "no")
    recall_aff = sum(1 for (a, v, p, _) in aff if comp.ask_yes_no(a, v, p) == "yes")

    # --- CONTRADICTIONS: assert each fact with the OPPOSITE polarity -> must REJECT ---
    reject_set = []
    false_accepts = 0
    for (a, v, p, pol) in facts:
        dec, _ = _assert_gate(comp, a, v, p, FLIP[pol])
        reject_set.append(1 if dec == "reject" else 0)
        if dec == "accept":
            false_accepts += 1                              # a contradiction slipped through = MOAT BREACH
    n_rejections = sum(reject_set)

    # --- CONSISTENT restatements (true polarity) + NOVEL assertions -> must ACCEPT (no over-block) ---
    over_blocks = 0
    for (a, v, p, pol) in facts:
        dec, _ = _assert_gate(comp, a, v, p, pol)
        if dec == "reject":
            over_blocks += 1
    for (a, v, p, pol) in NOVEL:
        dec, _ = _assert_gate(comp, a, v, p, pol)
        if dec == "reject":
            over_blocks += 1

    return {
        "n_facts": len(facts), "n_neg": len(neg), "n_aff": len(aff),
        "recall_neg_ok": recall_neg / max(1, len(neg)),
        "recall_aff_ok": recall_aff / max(1, len(aff)),
        "moat_false_accepts": int(false_accepts),
        "n_rejections": int(n_rejections),
        "n_contradictions": len(facts),
        "over_blocks": int(over_blocks),
        "reject_set": reject_set,
    }


def run_seed(seed, D):
    assert_backend("numpy", "B3 negation moat de-risk runs CPU (GPU is busy).")

    # ---------- INTACT ----------
    comp = _build(seed, D)
    _store(comp, FACTS)
    intact = _measure(comp, FACTS)
    # instrument check: the canonical negation genuinely reads "no" on the substrate
    canon_yn = comp.ask_yes_no("dog", "eat", "grass")

    # ---------- LESION: disable negation storage (store everything AFFIRM) ----------
    lesion_facts = [(a, v, p, "AFFIRM") for (a, v, p, _) in FACTS]
    comp_les = _build(seed, D)
    _store(comp_les, lesion_facts)
    # measure against the ORIGINAL polarities: asserting FLIP(original) now hits an AFFIRM store,
    # so the 3 originally-NEGATE contradictions ("assert AFFIRM") become CONSISTENT -> slip through.
    les = _measure(comp_les, FACTS)
    lesion_canon_yn = comp_les.ask_yes_no("dog", "eat", "grass")   # must read "yes" (negation gone)

    # ---------- ANTI-CHEAT no-store ----------
    comp_ns = _build(seed, D)                                       # nothing stored
    ns = _measure(comp_ns, FACTS)

    # ---------- ANTI-CHEAT shuffle: permute which facts are negated ----------
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(FACTS))
    shuffled_pols = [FACTS[perm[i]][3] for i in range(len(FACTS))]
    # ensure the shuffle ACTUALLY moved at least one polarity (else it is not a control)
    if all(shuffled_pols[i] == FACTS[i][3] for i in range(len(FACTS))):
        shuffled_pols[0], shuffled_pols[1] = shuffled_pols[1], shuffled_pols[0]
    shuffled_facts = [(FACTS[i][0], FACTS[i][1], FACTS[i][2], shuffled_pols[i]) for i in range(len(FACTS))]
    comp_sh = _build(seed, D)
    _store(comp_sh, shuffled_facts)
    # recall must track the SHUFFLED store (not a memorized answer)
    want = {"NEGATE": "no", "AFFIRM": "yes"}
    shuffle_recall = sum(1 for (a, v, p, pol) in shuffled_facts
                         if comp_sh.ask_yes_no(a, v, p) == want[pol])
    # every gate decision must equal (shuffled_pol != asserted_pol), asserted = FLIP(ORIGINAL pol)
    shuffle_tracks = 0
    shuffle_reject_set = []
    for i, (a, v, p, orig_pol) in enumerate(FACTS):
        asserted = FLIP[orig_pol]
        dec, _ = _assert_gate(comp_sh, a, v, p, asserted)
        rej = 1 if dec == "reject" else 0
        shuffle_reject_set.append(rej)
        expected_reject = 1 if shuffled_pols[i] != asserted else 0
        shuffle_tracks += int(rej == expected_reject)
    n = len(FACTS)

    row = {
        "seed": seed, "D": D,
        "intact": intact,
        "canon_neg_recall_no": (canon_yn == "no"), "canon_yn": canon_yn,
        "lesion": les,
        "lesion_canon_yn": lesion_canon_yn,
        "lesion_canon_negation_gone": (lesion_canon_yn == "yes"),
        "nostore": ns,
        "shuffle_recall_ok": shuffle_recall / n,
        "shuffle_tracks_store": shuffle_tracks / n,
        "shuffle_reject_set": shuffle_reject_set,
        "shuffle_reject_set_moved": int(shuffle_reject_set != intact["reject_set"]),
        "shuffled_pols": shuffled_pols,
    }

    # ---- pre-registered per-seed GO ----
    go_intact = (intact["recall_neg_ok"] == 1.0 and intact["recall_aff_ok"] == 1.0
                 and intact["moat_false_accepts"] == 0 and intact["over_blocks"] == 0
                 and intact["n_rejections"] == intact["n_contradictions"] and (canon_yn == "no"))
    go_lesion = (les["moat_false_accepts"] > 0 and lesion_canon_yn == "yes")
    go_nostore = (ns["n_rejections"] == 0)
    go_shuffle = (row["shuffle_recall_ok"] == 1.0 and row["shuffle_tracks_store"] == 1.0
                  and row["shuffle_reject_set_moved"] == 1)
    row["go_intact"] = bool(go_intact)
    row["go_lesion"] = bool(go_lesion)
    row["go_nostore"] = bool(go_nostore)
    row["go_shuffle"] = bool(go_shuffle)
    row["go"] = bool(go_intact and go_lesion and go_nostore and go_shuffle)

    print(f"  [seed {seed} D={D}] INTACT: recall neg={intact['recall_neg_ok']:.2f} "
          f"aff={intact['recall_aff_ok']:.2f} | moat FA={intact['moat_false_accepts']} "
          f"rejections={intact['n_rejections']}/{intact['n_contradictions']} "
          f"over_block={intact['over_blocks']} | canon('dog!eat grass')={canon_yn}", flush=True)
    print(f"           LESION(no-neg): FA={les['moat_false_accepts']} (want >0) "
          f"canon->{lesion_canon_yn}(want yes) | NOSTORE: rejections={ns['n_rejections']}(want 0) "
          f"| SHUFFLE: recall={row['shuffle_recall_ok']:.2f} tracks={row['shuffle_tracks_store']:.2f} "
          f"moved={row['shuffle_reject_set_moved']} => GO={row['go']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--out", type=str, default="research/findings/raw/_burndown_B3_onebrain_negation_moat.json")
    args = ap.parse_args()

    # LEVER self-check: the lesion MUST change the stored polarity of the canonical fact (NEGATE->AFFIRM),
    # else the A/B is void (both arms identical).
    lever("lesion strips negation", before="NEGATE", after="AFFIRM")

    rows = []
    for s in args.seeds:
        rows.append(run_seed(s, args.D))

    n = len(rows)
    go_all = all(r["go"] for r in rows)
    go_intact_all = all(r["go_intact"] for r in rows)
    go_lesion_all = all(r["go_lesion"] for r in rows)
    go_nostore_all = all(r["go_nostore"] for r in rows)
    go_shuffle_all = all(r["go_shuffle"] for r in rows)

    # instrument verification: the "inert lesion" claim is not a narrow bug -- in the lesion the
    # canonical negation genuinely reads "yes" on EVERY seed (the storage really lost the negation).
    lesion_negation_gone_all = all(r["lesion_canon_negation_gone"] for r in rows)
    void_if(not lesion_negation_gone_all,
            "a lesion seed did NOT read the canonical fact as AFFIRM -> the false-accept is a bug, "
            "not a genuine loss of negation storage")

    total_fa_intact = sum(r["intact"]["moat_false_accepts"] for r in rows)
    total_fa_lesion = sum(r["lesion"]["moat_false_accepts"] for r in rows)
    total_rej_intact = sum(r["intact"]["n_rejections"] for r in rows)
    total_rej_nostore = sum(r["nostore"]["n_rejections"] for r in rows)

    # ---- EARN the verdict: preconditions travel with the GO (tools.verdict.Verdict) ----
    v = Verdict("B3 non-contradiction moat on onebrain")
    v.require("recall_neg_ok == 1.0 all seeds", all(r["intact"]["recall_neg_ok"] == 1.0 for r in rows),
              note="stored negations recall as 'no' on the spiking substrate")
    v.require("recall_aff_ok == 1.0 all seeds", all(r["intact"]["recall_aff_ok"] == 1.0 for r in rows),
              note="stored affirmations recall as 'yes'")
    v.require("moat 0 false-accepts intact", total_fa_intact, expect=0,
              note="every contradicting assertion rejected")
    v.require("gate fired (rejections == contradictions) all seeds",
              all(r["intact"]["n_rejections"] == r["intact"]["n_contradictions"] for r in rows),
              note="the gate is not inert")
    v.require("0 over-blocks all seeds", all(r["intact"]["over_blocks"] == 0 for r in rows),
              note="consistent restatements + novel assertions accepted")
    # LESION load-bearing: the lesion (negation storage disabled) must BREACH where intact does not
    v.control("lesion breaches vs intact false-accepts", treatment=total_fa_lesion,
              control=total_fa_intact, min_separation=0.0,
              note="disable negation storage -> contradictions slip through (moat inert)")
    v.require("lesion negation genuinely gone all seeds", lesion_negation_gone_all,
              note="ask_yes_no reads 'yes' on the would-be-NEGATE fact in the lesion (not a bug)")
    # ANTI-CHEAT no-store: rejections collapse relative to intact
    v.control("no-store rejections collapse vs intact", treatment=total_rej_intact,
              control=total_rej_nostore, min_separation=0.0,
              note="store nothing -> the gate cannot reject (rejection is store-driven, not templated)")
    v.require("shuffle tracks store all seeds", all(r["shuffle_tracks_store"] == 1.0 for r in rows),
              note="every gate decision == (shuffled_pol != asserted_pol)")
    v.require("shuffle reject set moved all seeds", all(r["shuffle_reject_set_moved"] == 1 for r in rows),
              note="the reject set follows the permuted store -> not a fixed template")
    decided = v.decide(go=go_all)

    summary = {
        "runner": "_burndown_B3_onebrain_negation_moat_derisk",
        "n_seeds": n, "seeds": args.seeds, "D": args.D,
        "status": decided["status"],
        "preconditions": decided["preconditions"],
        "undefined_reasons": decided["undefined_reasons"],
        "GO": bool(decided["go"]),
        "go_intact_all": bool(go_intact_all),
        "go_lesion_all": bool(go_lesion_all),
        "go_nostore_all": bool(go_nostore_all),
        "go_shuffle_all": bool(go_shuffle_all),
        "total_false_accepts_intact": int(total_fa_intact),
        "total_false_accepts_lesion": int(total_fa_lesion),
        "lesion_negation_gone_all": bool(lesion_negation_gone_all),
        "rows": rows,
        "verdict": ("GO -- the non-contradiction moat FIRES on the onebrain (production-default) "
                    "substrate: stored negations recall, contradicting assertions are rejected "
                    "(0 false-accepts), consistent+novel assertions accepted; the negation storage "
                    "is load-bearing (lesion breaches) and the rejection is store-driven "
                    "(no-store + shuffle controls)."
                    if go_all else
                    "NO-GO -- see per-seed rows; the moat did not fire cleanly on all seeds."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==== B3 NON-CONTRADICTION MOAT ON ONEBRAIN ====")
    print(f"  seeds={args.seeds}  D={args.D}")
    print(f"  INTACT go (all seeds):   {go_intact_all}   (recall + 0 false-accepts + 0 over-block + gate fired)")
    print(f"  LESION go (all seeds):   {go_lesion_all}   (false-accepts intact={total_fa_intact} -> lesion={total_fa_lesion}; negation gone on all: {lesion_negation_gone_all})")
    print(f"  NO-STORE go (all seeds): {go_nostore_all}   (rejections collapse to 0)")
    print(f"  SHUFFLE go (all seeds):  {go_shuffle_all}   (recall tracks shuffled store; reject set moves)")
    print(f"  >>> GO = {go_all}")
    print(f"  wrote {args.out}")
    return summary


if __name__ == "__main__":
    main()
