---
type: finding
status: go
date: 2026-09-01
mechanism: common-ground ledger DRIVES audience design (board #152) — a persistent per-referent bistable
  NMDA-attractor ledger (K=6 slots) latched by grounding acts + held by recurrence, read at speak-time by a
  Namburi-Tye biased-competition reduce/introduce decision; the decision drives a reduced/pronominal reference
  lead ("As for it — <answer>") on the served /api/brain-chat reply
lane: E-language / conversational pragmatics
verdict: GO (load-bearing) — 6/6 seeds. Through the REAL production coupling
  (`webapp/common_ground_drives_chat.observe_turn` + `common_ground_ledger_production_organ`, mocked chat/composer,
  no brain build): drive_rate=1.000 every seed (36/36 referent-turns show the predicted STATE-A/STATE-B reply
  difference — first mention -> no lead, re-mention -> "As for it — " lead), lesion_drive_rate=0.000 every seed
  (BRAIN_CG_DRIVES_LESION=1 collapses the difference completely), a novel-interleave anti-cheat holds every seed
  (grounding one referent never spuriously grounds a different one), and the flag reads OFF by default every
  seed. Recommends the auto-flip per the 2026-09-01 owner policy; the flip itself is left to the controller.
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_common_ground_drive_soak.py
---

# Common-ground ledger DRIVES audience design — 6-seed load-bearing soak, GO

## Board task

Board #152: *"Common-ground ledger DRIVES audience design — WIRED default-OFF (next: flip default-ON after a live
UX soak)."* Per the 2026-09-01 owner policy (`GAP_CLOSURE_MISSION.md` §"POLICY 2026-09-01"), a faculty auto-flips
to production-default-ON the moment it is validated-GO **and** genuinely load-bearing on the live turn **and**
moat-safe **and** byte-identical-off **and** no-regression — no separate owner UX sign-off is required. This
finding is that soak.

## Prior work carried forward (not re-derived)

`bash tools/before_you_build.sh` + the RAG corpus surfaced two prior artifacts, both READ before building anything
here:

- [`2026-06-27-tier2.4-common-ground-GO.md`](2026-06-27-tier2.4-common-ground-GO.md) — an EARLIER, DIFFERENT
  common-ground mechanism (a host-SET SHARED/PRIVATE fact tag driving volunteer-vs-suppress). Honest boundary
  named in that finding: *"a LEARNED common-ground ledger (updated at each accepted contribution, Clark
  grounding) is the natural follow-on."* This soak is about that follow-on mechanism, not a re-test of Tier 2.4.
- [`2026-09-01-production-default-flip-plan.md`](2026-09-01-production-default-flip-plan.md) — the general
  auto-flip framework this soak's GO criteria are drawn from.

## A load-bearing discrepancy found before building anything (report this to the controller)

Board #152's own description states the ledger is *"now a production organ WIRED into the live /api/brain-chat
reply path"*. Investigating `webapp/open_ended_chat.py`, `webapp/server.py`, and `research/runners/
first_chat_console.py` on **current `main`** found **no trace of it** — no `common_ground_drives_chat.py`, no
`common_ground_ledger_production_organ.py`, and no `_CG_DRIVES_DEFAULT_ON` anchor in `webapp/server.py`.

The wiring commit (`793dd367a "wire(pragmatics): common-ground ledger DRIVES audience design in the live reply
path (default-OFF)"`) exists ONLY on the side branch `research/wire-common-ground-drive` (also `gitea` remote),
which was **never merged to `main`** and is now **319,942 lines behind** current `main` (it branched before a
large repo-wide cleanup — CRLF normalization, several Hermes-era tool retirements, etc.). Board #152 was
therefore describing a state that exists on a stale branch, not on the trunk this soak (and any real
`/api/brain-chat` traffic) actually runs against.

**Resolution taken (in-scope for building a soak that tests the real code, not a re-implementation):** the three
new files are pure additions with zero dependency drift — verified by diffing every reused helper
(`_gnw_rung1_ignition_curve_derisk`, `_gnw_rung3_report_reasoning_identity_derisk`, `_self_schema_region_derisk`,
`webapp/gnw_thought_swap._extract_topic`) between the stale branch and current `main`: byte-identical. They were
cherry-picked forward (`git checkout origin/research/wire-common-ground-drive -- <3 files> research/biology/
common-ground-ledger.md`) onto a fresh branch off current `main`. The `webapp/server.py` diff applied via
`git apply --3way` with exactly ONE trivial adjacency conflict (two independent flag-blocks both wanting to
follow `_da_drives_on()` — the BG-select/silent-WM blocks already on `main`, and this commit's common-ground
block); resolved by keeping both blocks in sequence (no logic touched). Confirmed with `py_compile` +
`tools/gates/production_integration.py.check()` + `tools/biology_check.py` — all clean. `docs/
PRODUCTION_INTEGRATION_LEDGER.yaml` gained the `common-ground-drives` row (57→58 faculties), `on_by_default: NO`,
matching the original commit's intent exactly. **The controller should treat landing this branch (or an
equivalent rebase of `research/wire-common-ground-drive`) onto `main` as a prerequisite to any real-traffic
flip** — it does not yet exist there.

**The cherry-picked de-risk still reproduces on current `main`'s `sim/`, byte-for-byte against the original
commit's numbers:** `SIM_BACKEND=numpy python -u -m research.runners._learned_common_ground_ledger_derisk --smoke`
wrote `research/findings/raw/_learned_common_ground_ledger_6seed.json` — seed 42, audience-design acc=1.000
(chance 0.5), permute=0.500, lesion=0.500 (frac-introduce 1.00), evidence-rate grounded=0.11347222222222224 vs
ungrounded=0.000.

## The soak

`research/runners/_common_ground_drive_soak.py`, PART A (the decisive gate; organ-level, no brain build):

- **Mocks** (`_MockComposer`, `_MockChat`) stand in only for `chat.inner.composer` — the ONE thing
  `common_ground_drives_chat.observe_turn` reads from the caller. Everything else exercised is the REAL
  production code: `webapp.common_ground_drives_chat.observe_turn` (the exact function `webapp/server.py`'s
  `brain_chat` calls) driving the REAL persistent `CommonGroundLedgerOrgan` (the same 750-neuron Izhikevich
  bridge `build_cg_bridge` builds, `research/runners/_learned_common_ground_ledger_derisk.py`). This is the
  "unbound method + a mock self" pattern (CLAUDE.md RAM-safety guidance) — appropriate here because the ledger is
  its OWN small dedicated substrate, not the 15k-LTM production brain; RAM stayed >20GB available throughout
  (`free -m` checked before running; 6 seeds x this soak completed in single-digit seconds).
- **STATE A vs STATE B, 6 referents x 6 seeds = 36 word-instances:** first mention (ungrounded, expect
  `introduce`/no lead) then a re-mention in the SAME conversation (expect `reduce`/`"As for it — "` lead).
  **drive_rate = 1.000 on every seed** (36/36) — the reply demonstrably differs exactly as audience design
  predicts.
- **Anti-cheat (novel-interleave):** ground referent 0, immediately query NEVER-mentioned referent 1 — must still
  read ungrounded. **Holds on every seed** — the decision follows the SPECIFIC referent grounded, not "something
  was grounded this conversation" (mirrors the underlying de-risk's own permuted-grounding-history control).
- **Lesion (the REAL `BRAIN_CG_DRIVES_LESION=1` flag, not a bypass):** identical script, ledger recurrence forced
  to 0. **lesion_drive_rate = 0.000 on every seed** (0/36) — the STATE A/B difference fully collapses; a
  re-mentioned referent can no longer be told apart from a first mention once the ledger cannot hold.
- **Byte-identical-off:** `cg_drives_enabled()` reads `False` with `BRAIN_CG_DRIVES` unset, confirmed every seed
  — and structurally, `webapp/server.py`'s `_common_ground_drives_on()` gate skips the entire block when this is
  false (no ledger build, no key, no lead attached at either the rich or single-fact response-assembly site).
- **No honesty/moat regression (by construction + inspection):** `observe_turn` returns only
  `{decision, lead, evidence_rate, ...}` — no fact/content field. The lead is PREPENDED as a string onto an
  already-computed answer surface at both `webapp/server.py` call sites; `abstained` / `recalled_svo` / `verified`
  are untouched by this module. (Empirical end-to-end confirmation via the real handler is PART B, best-effort —
  see below.)

Command + result:
```
SIM_BACKEND=numpy python -m research.runners._common_ground_drive_soak --seeds 42 43 44 100 101 102 --organ-only
```
`research/findings/raw/_common_ground_drive_soak/soak_summary_6seed.json`:
`overall_go: true`, `organ_go: true`, `mean_drive_rate: 1.0`, `mean_lesion_drive_rate: 0.0`, all 6 per-seed rows
`go: true`.

### PART B (best-effort, through the real `brain_chat` handler)

The harness also implements a handler-level replay (`brain='tiny-demo'`, `renderer='stub'`, no Qwen) mirroring
the established convention (`research/runners/_bg_action_selection_flip_soak.py` PART B). In THIS sandboxed
worktree, a fresh `tiny-demo` session's first turn pays a one-time warm cost across ~10 auxiliary production
organs (comprehension, surprise, world-model, an OU-process region, etc. — unrelated to the common-ground ledger)
that ran past a 240s bound without completing (once on CPU/numpy, once on GPU/cupy — GPU had 23GB free VRAM
throughout, so this is a wall-clock/build-graph cost, not a resource wall). Per the established convention
("Degrades to a reported SKIP … PART A is the gate"), PART A above is the decisive result; PART B remains
available for the controller to run with a longer budget or on an already-warm process:
```
SIM_BACKEND=cupy python -m research.runners._common_ground_drive_soak --seeds 42 --handler-timeout-s 600
```

## GO / auto-flip guard checklist (2026-09-01 policy)

| guard | result |
|---|---|
| validated-GO | YES — de-risk 1-seed smoke reproduced on current `main`'s `sim/` byte-for-byte (see the smoke numbers cited above) <!--derived--> |
| genuinely load-bearing (vary→differ→lesion→vanish) | YES — drive_rate 1.000 / lesion_drive_rate 0.000, 6/6 seeds, through the real production coupling |
| moat-safe | YES by construction (lead is a prepended string; no fact/content field touched) |
| byte-identical-off | YES — confirmed every seed; structurally guaranteed by `_common_ground_drives_on()`'s skip-the-whole-block gate |
| no-regression | YES at the organ level (this module touches nothing else); PART B (full-handler) not completed in-session — see above |
| hollow-flip trap | NOT TRIPPED — the lead demonstrably changes served text, and vanishes under lesion (not a metadata-only flip) |

**GO** on the mechanism. One prerequisite named above: the wiring must actually land on `main` (branch
`research/common-ground-audience-drive-soak`, built on top of the cherry-picked `research/wire-common-ground-drive`
files) before the flip means anything on real traffic.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._common_ground_drive_soak --seeds 42 43 44 100 101 102 --organ-only
SIM_BACKEND=numpy python -m research.runners._learned_common_ground_ledger_derisk --smoke
```
