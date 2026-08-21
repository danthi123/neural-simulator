---
type: finding
status: contributing
date: 2026-08-21
mechanism: open-ended-known-supplement-contradiction-filter
lane: integration
seeds: [42]
seed-waiver: A deterministic filter-logic de-risk over the SAVED open-ended replies (no new render) — the evidence
  is the catch/leak counts against a fixed wrong-supplement ground truth, not a stochastic effect.
instrument: research/runners/_open_ended_known_supplement_filter_derisk.py — runs a contradiction filter over the
  saved known-topic open-ended replies and counts wrong-supplement catch vs leak vs empties, with tools.verdict.Verdict.
runner: research/runners/_open_ended_known_supplement_filter_derisk.py
external: NO-EXTERNAL-NEEDED — reuses the prior de-risk's saved replies + the topic store facts.
artifacts:
  - research/findings/raw/_open_ended_known_supplement_filter_derisk.json
---
# A CONTRADICTION filter catches the known-topic WRONG-supplement residual of open-ended chat (GO, validated on saved replies)

Artifact: research/findings/raw/_open_ended_known_supplement_filter_derisk.json (GO).

**One line.** The wired open-ended mode + post-filter fully resolved the UNKNOWN-topic 100% fabrication, but left a
named residual ([[2026-08-21-verify-post-filter-restores-honesty-to-open-ended-generation-GO]]): on a KNOWN topic
the free Qwen reply keeps the grounded facts AND adds confident WRONG supplements on the SAME relations (Canada
"borders … Mexico" [store: united states]; France "bordered by Italy/Germany/Switzerland" [store: spain]; +
unsupported numbers/dates). This de-risks the fix — a per-sentence CONTRADICTION filter — and it catches all of them.

## The filter
Per sentence, drop it if it (a) carries a specific number or year (never in the SVO store), or (b) asserts a stored
relation (borders / continent / capital) with a DIFFERENT object than the store holds (using a small country/
continent gazetteer + the topic's store facts). Otherwise keep it.

## The verdict (on the saved known-topic replies — no new render) — GO
<!--derived-->
Over the 3 saved known-topic replies (canada/france/morocco), the 10 wrong supplements (mexico, 35 million, 1867,
italy, germany, switzerland, algeria, tunisia, libya, egypt): **caught 10/10 (catch rate 1.0), leaked 0**, and NO
reply was emptied (still conversational — the grounded content survives). So the contradiction filter cleanly
removes the known-topic wrong supplements while keeping the honest grounded prose.

## Honest scope
Validated on 3 saved replies (small sample; a fresh multi-topic render is the confirmation, deferred to avoid GPU
contention with a live test session). It catches the CONTRADICTION class only — a stored relation with a wrong
object, plus numbers/dates. DISABLED next rung: ungrounded-but-non-contradicting supplements (France "on the
Mediterranean" — true, not stored) survive, and the gazetteer should generalize to a store-backed entity check (any
store-known entity that isn't the stored object) or an NLI model. NEXT: wire this contradiction check into
`webapp/open_ended_chat.py`'s post-filter known-topic path (replacing the stub `contradicts`), generalized via the
store, and confirm with a fresh render. The primary unknown-topic moat is already GO + wired live; this closes most
of the known-topic honesty gap. NO sim/ edit.
