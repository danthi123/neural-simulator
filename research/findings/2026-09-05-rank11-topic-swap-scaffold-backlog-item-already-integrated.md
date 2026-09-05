---
type: finding
status: complete
date: 2026-09-05
mechanism: scaffold-retirement backlog rank-11 ("topic-swap regex+host", research/coordination/scaffold_retirement_backlog.md)
  audited against current code. The swap-vs-hold DETECTION this item asks to neuralize is already the spiking
  mismatch/salience `mm_k` + prediction-veto `pred_k` comparator
  (`research/runners/_gnw_neural_swap_intention_derisk.py`), wired to production DEFAULT-ON since 2026-08-19/20
  (`webapp/gnw_thought_swap.py` board #77, `webapp/swap_drives_chat.py` board #85, `_SWAP_DRIVES_DEFAULT_ON=True` in
  `webapp/server.py`), and reused a SECOND time by the ACC/BG STOP-trigger circuit (rank-12,
  `webapp/gnw_acc_bg_stop_trigger.py` / `webapp/gnw_global_stop.py`).
lane: integration-audit (research/coordination/scaffold_retirement_backlog.md rank 11)
verdict: NO NEW BUILD. The retirement rank-11 describes is ALREADY DONE — the swap-vs-hold DECISION is `wired` +
  `on-by-default` + `scaffold-retired` (docs/TERMS.md's `integrated` bar) for that specific sub-capability, and this
  is RE-VERIFIED on today's code, not merely re-cited from the old finding. Re-running the exact 6-seed reproduce
  command this task specifies (seeds 42/43/44/100/101/102) against TODAY's code reproduces the tracked artifact
  BYTE-IDENTICAL to the committed 2026-08-19 result (`git diff` empty on the data file). The real `/api/brain-chat`
  handler still fires the correct `swapped`/`held_topic`/`lead` triad on every turn of the original board-#85
  conversation script, unchanged. `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s own Check-D taxonomy (dated
  2026-09-02, three days before this backlog was generated) already tracks this exact mechanism under rows
  `gnw-thought-swap` / `swap-drives-response` and already names its true residual scaffolding precisely — none of
  which is "a regex performing the swap DETECTION." The backlog entry is MIS-SCOPED: a stale/imprecise
  re-description of an already-tracked, already-integrated capability, the same class of drift rank-24 found (2 of
  its 3 "quick flips" were also mis-scoped against current code).
artifacts:
  - research/runners/_gnw_neural_swap_intention_derisk.py
  - webapp/gnw_thought_swap.py
  - webapp/swap_drives_chat.py
  - webapp/gnw_acc_bg_stop_trigger.py
  - webapp/gnw_global_stop.py
  - docs/PRODUCTION_INTEGRATION_LEDGER.yaml (rows gnw-thought-swap, swap-drives-response)
  - research/findings/raw/_gnw_neural_swap_intention_6seed.json (re-run TODAY, byte-identical to the committed file)
  - research/findings/raw/_swap_drives_chat/verify_swap_drives.py (re-run TODAY against the real handler)
verification: |
  SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -u -m research.runners._gnw_neural_swap_intention_derisk --six-seed \
    --json research/findings/raw/_gnw_neural_swap_intention_6seed.json
    -> six-seed verdict=GO, seed_go 6/6 on every anti-cheat (swap/specificity/non_salient_holds/match_holds/
    load_bearing/lesion_holds/timing/reignite/reversible/no_reset/det), swap rates salient=1.00 non_salient=0.00
    match=0.00, POOLED_GO=True. `git diff` on the written artifact is EMPTY (byte-identical to the file committed
    2026-08-19) -- the mechanism has not drifted.
  SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -u research/findings/raw/_swap_drives_chat/verify_swap_drives.py
    -> through the REAL `webapp.server.brain_chat` handler, part (A)'s 7-turn conversation reproduces the exact
    swapped/held_topic/lead triad the 2026-08-19 finding tabulated (change->True/brain, hold->False, change->True/
    cat, hold->False, change->True/dog, with leads 'On brain, then -- '/'On cat, then -- '/'On dog, then -- ').
    With the script's isolation list extended to cover faculties flipped default-ON after 2026-08-19
    (BRAIN_GNW_STOP/BRAIN_AFFECTIVE_TOM/BRAIN_DA_DRIVES/BRAIN_SILENT_WM/BRAIN_BG_SELECT/BRAIN_CG_DRIVES/
    BRAIN_VISION_IDENTITY=0), the script's OWN verdict is confirmed: `VERDICT (A) tracks=True (B) drives+lesion=True
    (C) no-regression=True => GO`, all five checks [PASS] -- see "What actually changed since 2026-08-19" below for
    why the unmodified script prints FAIL today (test staleness) and the confirmation that restoring its assumed
    isolation reproduces its original clean GO.
---

# Backlog rank-11 ("topic-swap regex+host") describes a capability that is ALREADY neural, wired, and on-by-default — re-verified against TODAY's code, not just re-cited from the old finding. The backlog entry is mis-scoped.

**One line.** `research/coordination/scaffold_retirement_backlog.md` rank 11 says "detecting a topic-swap ('the
user changed the subject') currently uses a host regex + Python logic" and asks for it to be retired with a
neural detector reusing the GNW thought-swap mismatch machinery's `mm_peak`. That neural detector is not a
proposal — it is the mechanism already shipping: `research/runners/_gnw_neural_swap_intention_derisk.py` (6/6-seed
GO, 2026-08-19) wired into `/api/brain-chat` by `webapp/gnw_thought_swap.py` (board #77) and made load-bearing on
the reply by `webapp/swap_drives_chat.py` (board #85, `_SWAP_DRIVES_DEFAULT_ON=True`). No separate host-regex
topic-swap DECISION exists anywhere in the current codebase (see "What I searched," below).

## What the task asked to verify, and what I actually checked

Per `tools/before_you_build.sh "topic-swap regex host neural mismatch detector"` and
`rag_search.py "topic shift detection GNW thought swap mismatch mm_peak spiking"`, the top corpus hit was
`2026-08-19-gnw-neural-swap-intention-GO.md` itself — the RAG surfaced the answer directly. Rather than trust
that pointer alone (the map has been wrong before — rank-24 found 2 of 3 "quick flips" mis-scoped against current
code on this SAME day), I did four independent checks:

1. **Exhaustive grep sweep for a separate host topic-swap DECISION.** Searched `webapp/*.py`, `sim/*.py`,
   `research/runners/*.py`, `tests/*.py`, and the whole repo for topic-change phrase lists ("by the way", "anyway",
   "speaking of", "new subject"...), topic-comparison operators (`topic != `, `topic_changed`, `last_topic`,
   `topic_break`), and named functions (`detect_topic_*`, `is_topic_swap`, `topic_shift*`). The only "regex +
   topic" hits are (a) `webapp/gnw_thought_swap.py::_extract_topic`'s `re.findall(r"[a-zA-Z']+", ...)` word
   tokenizer — TOPIC EXTRACTION, not the swap DECISION, explicitly declared in that module's own docstring as "the
   SAME host-comprehension boundary the SVO question parser... occupies" (a different, already-separately-tracked
   residual — see below), and (b) `research/runners/brain_chat_tui.py::_parse_open_ended`'s regex router, which
   classifies whether a turn is an EXPLICIT open-ended-generation prompt ("what might X...", "guess...") — a
   different classification task, cited as an ANALOGOUS example in `_selfschema_authorship_neural_turnclass_derisk.py`'s
   scope note, not itself rank-11's target. No keyword-list or `!=`-style host swap-DECISION exists.
2. **Re-ran the underlying 6-seed de-risk against today's code** (this task's exact seed set, 42/43/44/100/101/102)
   writing to the tracked artifact path — `git diff` on the output is EMPTY, i.e. BYTE-IDENTICAL to the file
   committed 2026-08-19. The mismatch/salience comparator has not drifted or regressed since.
3. **Re-ran the production-level verify script** (`research/findings/raw/_swap_drives_chat/verify_swap_drives.py`)
   against the REAL `webapp.server.brain_chat` handler on today's code (see below for the one wrinkle this
   surfaced).
4. **Checked `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s Check-D scaffold-retirement taxonomy** (dated 2026-09-02,
   i.e. already current three days before this backlog's 2026-09-05 audit ran) — it already carries rows
   `gnw-thought-swap` and `swap-drives-response`, both `de_risked: YES`, `wired: YES`, and `swap-drives-response`
   `on_by_default: YES`, with a hand-written `host_scaffold_in_default` field that already names the TRUE residual
   scaffolding precisely (quoted below). Also checked `docs/RETRACTED.md` — neither finding is retracted.

## The mechanism, as it actually stands today (unchanged from 2026-08-19, re-verified)

Two per-pattern spiking pools ride the neural-vacancy-gate substrate: `mm_k` (60 excitatory mismatch/salience
detectors per topic-slot) fires when topic `k` is proposed with salience; `pred_k` (40 inhibitory interneurons)
is excited by the currently-HELD workspace pattern and vetoes `mm_k` when `k` is already held. `mm_k`'s firing
rate sets the incumbent coalition's short-term-depression boost — no host `if` — so a genuine mismatch
self-evicts the old topic and the neural vacancy gate admits the new one; a repeat/held topic is vetoed by `pred`
and nothing moves. The per-turn `mm_peak` this produces is exactly the read `webapp/gnw_acc_bg_stop_trigger.py`
(rank-12's ACC/BG STOP circuit) ALSO consumes off `chat._last_swap_drives` — i.e. this backlog's own premise
("the same mismatch read the swap detector already exposes as `mm_peak`") is describing a signal that is already
being reused by a second production consumer, not proposing its first use.

Production wiring (`webapp/swap_drives_chat.py::observe_turn`, called from `webapp/server.py`'s
`_swap_drives_on()` block): each turn's grounded topic is presented to a per-session `ThoughtSwapWorkspace`; the
neural `swapped` verdict prepends a transition lead ("On `<topic>`, then — ") to the answer on a genuine topic
change and stays silent on a hold; `BRAIN_SWAP_DRIVES=0` removes the whole block byte-identically;
`BRAIN_SWAP_DRIVES_LESION=1` silences `mm` and the transition lead vanishes even though the world input (a real
topic change) is unchanged — the load-bearing proof, previously established and reconfirmed by today's re-run
(part A, below).

## Byte-identical reproduction, today

```
SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -u -m research.runners._gnw_neural_swap_intention_derisk --six-seed \
  --json research/findings/raw/_gnw_neural_swap_intention_6seed.json
```
<!--derived-->
`[neural-swap-intent six-seed] verdict=GO seed_go 6/6 swap 6/6 spec 6/6 non_salient_holds 6/6 match_holds 6/6
load_bearing 6/6 lesion_holds 6/6 timing 6/6 reignite 6/6 reversible 6/6 no_reset 6/6 det 6/6` — every per-seed
number rounds from the SAME full-precision values stored in the artifact (e.g. seed 101:
`mm_peak=0.3055555555555556` -> displayed `0.306`, `boost_max=0.16`; seed 102: `mm_peak=0.2666...` -> displayed
`0.267`, `boost_max=0.159...` -> displayed `0.159`), matching the committed 2026-08-19 table's own rounded
display. `git diff research/findings/raw/_gnw_neural_swap_intention_6seed.json` is empty; only the `.prov.json`
sidecar (this run's own timestamp/argv) changed.

## What actually changed since 2026-08-19 (why the OLD verify script now prints FAIL, and why that is not a regression)

Re-running `verify_swap_drives.py` unmodified against today's code, part (A)'s per-turn `swap_drives` fields are
ALL correct — `swapped=True` on every genuine topic-change turn with the exact expected `held` topic and the
exact expected lead string (`'On brain, then — '`, `'On cat, then — '`, `'On dog, then — '`), `swapped=False` on
every same-topic/no-topic turn — yet the script's own summary prints `(A) FAIL` / `(B) FAIL`. The cause is
**test staleness, not mechanism drift**: the script's isolation list (17 env vars it forces to `0` to get a
tractable, decoration-free answer string) was written 2026-08-19 and does not know about faculties flipped
default-ON afterward — `BRAIN_CG_DRIVES` (common-ground audience-design lead, flipped 2026-09-01: this is exactly
the `'As for it — '` prefix now observed wrapping the swap lead) and `BRAIN_DA_DRIVES`/`BRAIN_GNW_STOP`/
`BRAIN_AFFECTIVE_TOM`/`BRAIN_SILENT_WM`/`BRAIN_BG_SELECT`/`BRAIN_VISION_IDENTITY` (all flipped default-ON in the
2026-08-26/2026-09-01 waves, per `_DEFAULT_ON = True` anchors in `webapp/server.py`). These faculties add their
OWN unrelated leads/suffixes to the SAME answer string, so the script's exact-whole-string equality checks (which
assume ONLY the swap lead can appear) now legitimately fail even though every `swap_drives`-specific field is
correct. **Confirmed, not just diagnosed:** re-running the SAME unmodified script with those newer flags ALSO
forced to 0 (`BRAIN_GNW_STOP=0 BRAIN_AFFECTIVE_TOM=0 BRAIN_DA_DRIVES=0 BRAIN_SILENT_WM=0 BRAIN_BG_SELECT=0
BRAIN_CG_DRIVES=0 BRAIN_VISION_IDENTITY=0` added on top of the script's own list, writing to the tracked default
artifact path `research/findings/raw/_swap_drives_chat/verify.json`) reproduces the ORIGINAL clean verdict
exactly: `VERDICT (A) tracks=True (B) drives+lesion=True (C) no-regression=True => GO`, all five named checks
`[PASS]` — part (B)'s INTACT probe shows `swapped=True lead='On dog, then — '`, the SAME probe with
`trigger_lesion` shows `swapped=False lead=''` reverting the answer to the un-led base, and part (C)'s no-swap
content-md5 is identical across {off, swap, hold} on every turn. The refreshed `verify.json` differs from the
2026-08-19 file ONLY in surface grammar (e.g. `"The dog chases cat."` -> `"the dog chases the cat"` — an
intervening, unrelated article/capitalization change in the recall-render surface) and the run timestamp; every
`mm_peak`/`boost_max`/`reason`/`swapped` value per turn is UNCHANGED. So this is not merely a diagnosis of why
the old script misfires today — restoring its assumed isolation makes it pass exactly as it did on 2026-08-19.
This is recorded here as a small, separate, NOT-GATEABLE-by-this-task item: `verify_swap_drives.py`'s hardcoded
isolation list should be extended to cover every `_..._DEFAULT_ON` flag added after 2026-08-19, or (better)
derive its isolation set from `webapp/server.py`'s own `_DEFAULT_ON` anchors so it cannot go stale again.
Flagged, not fixed here (out of rank-11's scope, and fixing a verify script is not itself a scaffold-retirement).

## The ledger already tracks this, and already names the true residual precisely

`docs/PRODUCTION_INTEGRATION_LEDGER.yaml` (Check-D scaffold-retirement taxonomy, 2026-09-02 — three days BEFORE
this backlog's audit ran) row `swap-drives-response`:
> `host_scaffold_in_default: "the topic EXTRACTION is host (the SVO-parser comprehension boundary); the
> verdict->transition-STRING is a host conditioned-articulation template (the discourse mouth), DRIVEN by the
> neural swap verdict; the swap runs on its own co-resident bridge (not merged with the recall composer's
> bridge); inherited #77 host scaffolds (mm->boost rate read-out, held_slot continuity label, labeled-line
> routing). The load-bearing spiking part is the swap-vs-hold VERDICT + the winning coalition's identity off the
> #77 mismatch/eviction/admit chain (lesion-proven) driving the transition lead."

This is precise and matches everything I independently re-derived: the DETECTION/DECISION is the lesion-proven
spiking part; the residuals are extraction, articulation, the mm-to-boost read-out, cross-turn continuity
bookkeeping, and labeled-line routing — NOT a regex performing the swap detection. `retire_status:
"BLOCKED:gnw-thought-swap"` on both rows reflects those NAMED residuals (each a real, separate, larger research
item — topic-extraction neuralization is essentially backlog rank-14's scope, not rank-11's), not an unclosed
swap-DETECTION gap.

## Terminology check (`docs/TERMS.md`)

The swap-vs-hold DECISION specifically clears the `integrated` bar: `wired` (reachable from `/api/brain-chat` on
every turn), `on-by-default` (`_SWAP_DRIVES_DEFAULT_ON=True`, no opt-in needed), and `scaffold-retired` for that
narrow sub-result (no host code computes "did the topic change" on the default turn — grep-verified, section
above). I am NOT claiming the broader bundled "hold a topic across turns and talk about it" capability is fully
`scaffold-retired` — the ledger's own `BLOCKED` status for the surrounding extraction/articulation/coupling/
continuity/routing pieces stands, unchanged by this finding.

## Why the backlog says "host regex + Python logic" at all (a plausible account, not verified)

The scaffold-shortcut-map workflow (`w9sn9wn4b`) that produced `scaffold_retirement_backlog.md` most likely
conflated `_extract_topic`'s tokenization regex (a genuinely-host, but different and already-named, residual) with
the swap DECISION itself, and/or ran without cross-referencing the Check-D ledger rows that already track this
exact capability under a different name (`gnw-thought-swap` / `swap-drives-response`). Either way, re-deriving a
"neural swap reusing `mm_peak`" mechanism here would duplicate, not extend, existing production code.

## Scope honesty

No `sim/` edit. No new runner, no new production wiring — this is a verification-only finding. The only repo
changes are: this finding, one re-run artifact (byte-identical, so effectively a no-op data-wise) plus its
provenance sidecar, and a `scaffold_retirement_backlog.md` status-update line (append-only, per that file's own
convention) recording the audit outcome so a future session does not re-derive this. What WOULD be a genuine next
rung — closing the named ledger residuals (topic-extraction neuralization, a real synaptic mm-to-boost mechanism,
continuous cross-turn ignition, learned/composed topic routing) — is each a separate, substantial mechanism-design
item, not attempted here.
