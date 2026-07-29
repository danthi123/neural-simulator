# Document structure rules — two rules, both greppable

**Two rules.** Each is earned by a currently-live failure in this repository, and each is checked by one command
with no judgment call. The drafted version had five; adversarial review cut three (see the changelog at the end).

**Companion to [`docs/TERMS.md`](TERMS.md).** `TERMS.md` governs *which words* may be used and under what code
condition. This file governs *where a dead claim's death is recorded*. **Neither file governs whether a claim is
true** — see "What this does NOT do".

**Credit.** The approach comes from **ASD-STE100** (Simplified Technical English, maintained by ASD — free to
obtain, not redistributable), which demonstrates that technical prose can be made machine-checkable. None of its
rule text or dictionary is reproduced here. `TERMS.md` adopts its one-term-one-meaning principle; the rest of the
specification was assessed and not adopted, for the reasons recorded there.

---

## Scope

**Governed** (6 files, ~5,200 lines):

    CLAUDE.md
    GAP_CLOSURE_MISSION.md
    ROADMAP.md
    README.md
    docs/TERMS.md
    docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md

**Not governed:** the 1,800+ existing files in `research/findings/`, the rest of `docs/plans/`,
`docs/project-history-archive.md`, `AUTONOMOUS_STATE.md`, code comments, docstrings, scratch notes. **There is no
retrofit of findings.** Findings are an append-only research record; rewriting them would destroy the audit trail
that makes retractions traceable. `sim/` is never edited by documentation work.

---

## W1 — A voided document is registered, and no governed file cites it silently

**Rule, two parts.**

1. When a document's central claim dies, add one row to [`docs/RETRACTED.md`](RETRACTED.md):

       | path or commit | date | superseded by | why (<=20 words) |

2. No governed file may cite a registered path without `⛔` on the same line or bullet.

**Why.** A finding voided a re-attribution **by description and never by path**: it lists what is void in prose,
while the document it kills asserts that attribution in its own filename and H1. Measured at adoption: the voiding
doc never names its target, 7 files still cited the target, and no marker convention existed anywhere in
`research/findings/`. This is CLAUDE.md drift #12 — stale pointers, the documented #1 cause of re-derived work.

Retraction is a **standing property** of this project, not one bad session: findings containing a retraction number
63 in 2026-07, 42 in 2026-05, 37 in 2026-06.

**Why a registry rather than an in-place marker.** A retraction's target is almost always an older, ungoverned
findings doc, so an in-place rule could not be satisfied without editing outside this file's scope. A registry is
written by the person writing the retraction, inside the governed set, and the check fires at the **citing** site —
where the cost is actually paid. It also handles **partial** voids, which a heading stamp cannot: retraction #9
deliberately preserves three still-valid measurements under a void headline, and a block marker would discard them.

---

## W2 — Prose lines in governed files are at most 800 characters

**Rule.** A prose line is at most 800 characters. Table rows and fenced code are exempt.

**This is a precondition for W1, not a readability rule.** W1's marker must sit next to the claim it kills, and it
cannot when a bullet is 14,222 characters long. One governed line carries a dead number at offset 865 and a
hand-written `⚠️ DO NOT cite` at offset 14,098 — **the author knew a claim ~13,200 characters earlier was dead and
had nowhere to put the marker.** W1's check would read that line as "marked" while a reader landing at offset 865
gets no signal at all.

**Not claimed:** that this improves RAG retrieval. The index splits on tokens
(`SentenceSplitter(chunk_size=1024, chunk_overlap=100)`, `tools/rag/update_indexes.py`), so a claim and its
refutation ~1,400 tokens apart land in different chunks whether the source is one line or twenty. Splitting the
line does not change chunk membership.

---

## Checks

Both run in `tests/test_doc_rules.py` and via `tools/check_docs.py`. After the adoption retrofit both are green, so
any later failure is a real regression rather than a warning nobody acts on.

---

## What this does NOT do

**It cannot catch instrument failures — the dominant failure mode.** Six of the nine retractions on 2026-07-28 were
broken instruments, and **all six would have produced prose that passes both rules**: a lesion that did not persist
(zeroed weights back to 0.05 within five steps, during the read meant to measure their absence); a measurement
placed before the thing it measured; a control comparing two identical configurations; `ast.parse` missing
symbol-table errors; a metric too coarse to resolve a real lever; a type error reported as a scientific null.

Those are `.claude/skills/verify-go/SKILL.md`'s job. **A spec that implied otherwise would be exactly the overclaim
this project keeps retracting.**

It also does not govern truth, hedging, sentence length, voice, or vocabulary beyond `TERMS.md`.

---

## Changelog — what adversarial review CUT, and why

| Drafted rule | Verdict | Reason (measured) |
|---|---|---|
| Retraction marks in place | **rebuilt as W1** | Its check matched **0 lines** in all six governed files, and part of it required writing markers into files the scope explicitly exempts. The naming half was earned and survives as the registry. |
| `Status:` line at line 2 | **CUT** | Premise false — it claimed the H1 could not be edited while the neighbouring rule prescribed inserting a line above it. 20 of 164 recent findings carry both a GO and a retraction; one token cannot express both. |
| 800-char cap | **KEPT, narrowed** | Measurement confirmed, but table rows had to be exempted and "one claim per bullet" / "ends with its verdict" were unenforceable. |
| 72-char commit subject | **CUT** | No cited failure, and an active regression: `git log -40 --format=%b \| grep -c .` → **0**. Every retraction's evidence currently lives in the subject line, which is the surface injected into an agent's context. |
| 25-word verdict sentences | **CUT** | Not reproducible from its own stated convention, ~481 day-one warnings, and its own exemplar (`docs/TERMS.md`) failed the rule on 2 of 5 verdict sentences — including the founding-evidence sentence. |

**Adding a rule** requires a cited, live failure and a check with no judgment call. This file records earned
lessons, not anticipated ones.
