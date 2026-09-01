---
type: finding
status: audit
date: 2026-09-01
mechanism: production-default flip plan (chat-path wired-default-OFF faculty audit)
lane: integration
seeds: [42]
seed-waiver: This is a READ-ONLY code-trace + ledger + board + FAILURE_LOG audit, not a stochastic
  result. The claims are structural facts about which flag is read where, and which reads reach a
  load-bearing coupling on the default /api/brain-chat turn — established by grepping the source and
  reading the gating control-flow, not by a seed sweep. No flag was flipped and no faculty code was
  edited (the task is to PRODUCE the plan; flips execute carefully afterward). Each per-faculty verdict
  cites the exact source file:line + the ledger row + the FAILURE_LOG entry it rests on.
instrument: grep/read over docs/PRODUCTION_INTEGRATION_LEDGER.yaml, webapp/server.py,
  webapp/open_ended_chat.py, webapp/gnw_two_organ_bus.py, the research/runners/*_production*.py flag
  readers, research/FAILURE_LOG.md, and the Vikunja board (tools/vikunja.sh list-tasks 2).
external: NO-EXTERNAL-NEEDED — an internal integration-state audit of this repo's own flags/ledger/board.
---

# Production-default flip plan — the chat-path wired-default-OFF faculties

## Headline (the decision-relevant finding)

**Class A (FLIP-READY NOW, genuinely load-bearing, low risk) is EMPTY on the chat path today.** Every
wired-default-OFF faculty that touches the default `/api/brain-chat` reply is currently **hollow if
flipped now** — flipping its flag either changes no reply text at all, or would light up a flag whose
coupling the parent gate never reaches. This is exactly the failure mode the audit exists to prevent, so
the honest verdict is: there is no free flip to bank this cycle, and the ~40-de-risk backlog on the chat
path is a set of *blocked* items, not *ready* ones.

The single highest-value action is therefore an **owner-UX decision on the `BRAIN_OPEN_ENDED` family**
(the free open-ended generation + VERIFY-post-filter reply path, board #112), flipped **as a bundle with
its moat-hardening children** (`BRAIN_OPEN_ENDED_NP_ENTAILMENT` just-merged 2026-09-01, and
`BRAIN_OPEN_ENDED_GEN_TIME_HONESTY`) — never the parent alone and never a child alone. The one genuinely
*buildable* near-term Class-A is fixing `confidence-forthcomingness`'s content-exhaustion residual.

## Method

Extracted every row from the 57-row ledger and its levels; only 3 rows are `wired=YES` with
`on_by_default!=YES` (`gnw-thought-swap`, `source-provenance-honesty`, `confidence-forthcomingness`). The
other named chat faculties (the open-ended family, the cross-edges, common-ground #152) are **wired via
env flags in source but are NOT yet ledger rows** — found them by grepping `_DEFAULT_ON = False` and
`os.environ.get("BRAIN_*"` across `webapp/` and the server-imported `research/runners/*_production*.py`,
then reading each reader and the server-side gating control-flow. Cross-checked every "flip would be
hollow" claim against `research/FAILURE_LOG.md`, which independently documents the recent hollow-flip bug
class.

## Evidence artifacts the per-faculty verdicts rest on

These are the measured verify artifacts (from the ledger rows' own `evidence` fields) that the
"hollow-if-flipped" verdicts below are grounded in — not re-run here, cited:

- `research/findings/raw/_confidence_forthcoming_retest/verify_real_traffic_FINAL.json` — the
  real/unpatched-confidence + noise-degraded run through the REAL handler showing the un-overridden
  production floor has NO visible HIGH-vs-LOW difference (grounds the #1 content-exhaustion verdict).
- `research/findings/raw/lanes/metacog/_129_source_provenance_honesty_wirein_6seed.json` — the 6-seed
  source-provenance wire-in GO (grounds #6: the mechanism is real; the block is HTTP-exposure #140, not
  the substrate).
- `research/findings/raw/_onebrain_xedge_surprise_episodic_production_frozen_6seed.json` — the 6-seed
  frozen cross-edge GO whose own residual #3 states the read is content-neutral at the decision level
  (grounds #2 "diagnostic-only, flip is hollow").

## The hollow-flip patterns the FAILURE_LOG already caught (why rigor here matters)

- **os.environ.pop-as-OFF staleness** (FAILURE_LOG 2026-08-27, 6 soaks): a flip-soak's OFF arm did
  `os.environ.pop(FLAG)`, which silently reads ON once that flag's production default flips ON — every
  ON-vs-OFF comparison collapsed to ON-vs-ON. Fixed; the class is real.
- **caller/callee env-default mismatch (inner-gate-defaults-off)** (FAILURE_LOG 2026-08-27, GNW 3-organ):
  the outer install-gate read default `"1"` (ON) but the inner `three_organ_enabled()` read default `""`
  (OFF), so a "flipped default-ON" faculty **never installed** — the ledger's `on_by_default:YES` was
  FALSE in production from 2026-08-21 to the fix.
- **GNW two-organ bus vetoes every LTM fact** (FAILURE_LOG 2026-08-27, #78): organ B needs a
  buffer-tier expectation, so the consensus bus structurally vetoed all ~15k Wikidata LTM facts. Closed
  by `BRAIN_GNW_ORGANB_LTM_EXEMPT` (see "Latent issues" — it is now default-ON in code but its docstrings
  still say default-OFF).
- **confidence saturation self-referential ratio** (FAILURE_LOG 2026-08-27, #181): `confidence =
  s[argmax]/max(s) == 1.0` always, so the metacog hedge never fired on real traffic. Fixed (margin-keyed,
  board #181 now DONE) — but this is exactly why `confidence-forthcomingness` still reads hollow (below).
- **TieredFactStore.__setattr__ missing** (FAILURE_LOG 2026-08-27, #82): metacog/activity reads were
  silently `None` on the production `tiny-demo +LTM` brain since the 2026-08-26 knowledge-core flip. Fixed.

Every one of these was a flag that *looked* on but reached nothing. The verdicts below apply the same lens.

## Classified, prioritized flip table

| # | Faculty | Flag (+ default reader) | Class | Hollow-flip risk | What the flip needs | Blast radius |
|---|---------|-------------------------|-------|------------------|---------------------|--------------|
| — | *(none clean)* | — | **A** | — | — | — |
| 1 | confidence-forthcomingness (#94) | `_CONFIDENCE_FORTHCOMING_DEFAULT_ON=False` (webapp/confidence_forthcoming_chat.py:103) | **C** | HIGH — wiring now non-hollow (its 2 blockers #181/#82 fixed) but the coupling has **nothing to add on real traffic**: elaboration candidate pool is BUFFER-tier only, so the un-overridden production floor shows no visible HIGH-vs-LOW difference (content-exhaustion). Documented NOGO/PARTIAL. | Give the elaboration planner an **LTM-tier candidate source** (or a real differentiated-confidence traffic source), then re-verify the HIGH-vs-LOW difference on unforced real content + lesion. | Additive; only ever DROPS an already-verified tail fact. Moat-safe. |
| 2 | xedge surprise->episodic | `_XEDGE_SE_DEFAULT_ON=False` (research/runners/onebrain_xedge_surprise_episodic_production.py:93) | **C** | HIGH — the live read is a **content-neutral additive DIAGNOSTIC field** (`resp["surprise"]["source_provenance_crossedge"]`, module residual #3: "not wired to flip any decision-level text"), over the construction's OWN fixed ambiguous pattern (residual #1), not an arbitrary live fact. Flip = metadata-only hollow checkbox. | Build the decision-level coupling: bind the bias to an arbitrary live chat fact + drive reply text; THEN a load-bearing flip. | None today (additive metadata). |
| 3 | xedge self-schema->provenance | `_XEDGE_SS_DEFAULT_ON=False` (research/runners/onebrain_xedge_selfschema_production.py) | **C** | HIGH — same content-neutral diagnostic (`resp["authorship"]["source_provenance_crossedge"]`, "content-neutral at the decision level"). Flip = hollow. | Same as #2 (decision-level coupling first). | None today. |
| 4 | BRAIN_CONTINUOUS_IDEATE_SPIKING (#104 rung 2) | `os.environ.get("BRAIN_CONTINUOUS_IDEATE_SPIKING","0")` (webapp/continuous_engine.py:979) | **B** | MEDIUM — spiking upgrade of the already-ON host continuous-ideation; genuinely swaps a mechanism, so not structurally hollow, but never verified load-bearing-equal to the host version on real between-turn traffic. | A spiking-vs-host equivalence + load-bearing verify on the live idle tick (mind the background-thread race, FAILURE_LOG 2026-08-26 #75), then flip as a **scaffold-retirement**. | Between-turn ideation only; additive. |
| 5 | **BRAIN_OPEN_ENDED** (#112) — the family PARENT | `os.environ.get("BRAIN_OPEN_ENDED","0")` (webapp/open_ended_chat.py:197 + the sole server gate webapp/server.py:4519) | **D** | n/a as a flag (it IS load-bearing) — but it is a **product decision**: it fully REPLACES the strict/rich substrate composer with Qwen/WKV free generation + moat post-filter as the default reply (server.py:4567 `return`s the open-ended response). | OWNER call + a real-traffic **moat-safety soak** (fabrication rate on brain-unknown/Qwen-known topics). Flip the moat children WITH it. | LARGE — changes the default reply generator for every turn. |
| 5a | BRAIN_OPEN_ENDED_NP_ENTAILMENT (merged 2026-09-01) | `os.environ.get(...,"0")` (webapp/open_ended_chat.py:238) | **D-bundle** | **PROVABLY HOLLOW if flipped alone** — `post_filter`/`np_entailment_enabled()` are referenced ONLY inside the `BRAIN_OPEN_ENDED` block (server.py:4519 is the module's sole import site); parent OFF => module never imported. | Flip ONLY together with the parent (it is the parent's moat hardening). | Adds a monotonic-only drop; never restores. |
| 5b | BRAIN_OPEN_ENDED_GEN_TIME_HONESTY | `os.environ.get(...,"0")` (webapp/open_ended_chat.py:225) | **D-bundle** | PROVABLY HOLLOW if flipped alone (same sole-import-site argument). | Flip with the parent + a live organ-wired `chat`. | Generation-time suppression; additive safety net. |
| 5c | BRAIN_OPEN_ENDED_WKV_MOUTH | `os.environ.get(...,"1")` — already default-ON (webapp/open_ended_chat.py:210) | **D-bundle** | Default-ON but HOLLOW because the parent is OFF (module never imported). Becomes live only when the parent flips. | Nothing extra — it rides the parent flip. | Swaps the in-vocab FORM generator to the from-scratch WKV spiking mouth. |
| 5d | BRAIN_HONESTY_SKIP_CONTINUE | `os.environ.get(...,"0")` (open_ended_chat.py) | **D-bundle** | HOLLOW without BOTH parent + GEN_TIME_HONESTY (grandchild). | Flip only after 5 + 5b are on and verified. | Drop-and-continue within a reply. |
| 6 | source-provenance-honesty (#129) | `BRAIN_SOURCE_PROVENANCE_HONESTY` default-OFF (research/runners/source_provenance_production_organ.py:48; read webapp/server.py:5884) | **D** | HIGH — a PERCEIVED (recalled) claim renders **byte-identical**; the GENERATED half (the actual value) **has no live HTTP exposure at all** (board #140). So on the real endpoint, flip ON = byte-identical = hollow until #140. | OWNER-gated design decision **#140** (expose GENERATED-provenance framing / reason_chain as an answer channel), THEN flip. | Reframes TEXT only; never flips an abstain / manufactures a fact. Moat-safe. |
| 7 | common-ground audience-design (#152) | no `/api/brain-chat` flag — wired only in research/runners/first_chat_console.py (guarded by whether `cg_comp` builds) | **D** | Not on the production spine: it is wired into the interactive CONSOLE, NOT `webapp/server.py /api/brain-chat`, so it fails the ledger's own "wired" definition for production. Board next step = "flip after a live UX soak". | (a) WIRE into `/api/brain-chat` first; (b) OWNER call on audience-design behavior; (c) live UX soak. | Audience-design (how much shared context to assume) — a UX behavior. |
| — | gnw-thought-swap | `_GNW_SWAP_DEFAULT_ON=False` (webapp/server.py:3081) | **EXCLUDE** | SUPERSEDED + RETIRED: `swap-drives-response` (default-ON) is the sole live swap path; the observe-only fallback was REMOVED from `brain_chat`. Flipping the flag now only re-enables an additive observability key — no behavior change = hollow. | Nothing — leave OFF. The live faculty is already on via swap-drives. | None. |
| — | XEDGE_SS_DECLARATIVE | `_XEDGE_SS_DECLARATIVE_DEFAULT_ON=False` | **N/A** | Not a faculty flip — an internal construction-path refactor (declarative vs bespoke pool build) with **byte-identical output** either way. | Nothing owner-facing. | None (byte-identical). |

## Per-class summary

- **A — FLIP-READY NOW:** none. Bank nothing blind this cycle.
- **B — NEEDS-VERIFY:** #4 `BRAIN_CONTINUOUS_IDEATE_SPIKING` (spiking-vs-host equivalence on the live idle
  tick). The open-ended moat children (5a/5b) become B *the moment the parent decision is YES* — verify
  moat-safety, then flip together.
- **C — NEEDS-FIX:** #1 confidence-forthcomingness (content-exhaustion: LTM-tier elaboration candidates);
  #2/#3 the two cross-edges (build a decision-level coupling; today they are diagnostic-only, so a flip is
  the exact "neural verdict stashed as metadata" hollow-checkbox the owner named 2026-08-19).
- **D — OWNER-UX-CALL:** #5 `BRAIN_OPEN_ENDED` bundle (the big one, board #112); #6 source-provenance
  GENERATED half (board #140); #7 common-ground #152 (needs production wiring + a UX call).

## The single highest-value flip to do first

**The `BRAIN_OPEN_ENDED` family, flipped as one bundle** — parent `BRAIN_OPEN_ENDED=1` **together with**
`BRAIN_OPEN_ENDED_NP_ENTAILMENT=1` and `BRAIN_OPEN_ENDED_GEN_TIME_HONESTY=1` (WKV mouth is already
default-ON and rides along). It is the one flip that most advances *genuine spiking one-brain
conversation*: it turns the default reply into a free, first-person, multi-sentence generation whose FORM
mouth is the from-scratch WKV spiking cortex (in-vocab) with Qwen as the sanctioned articulation fallback,
guarded by the no-confab VERIFY post-filter + the spiking NP-entailment moat gate.

But it is **NOT flip-ready** — it is a **Class-D owner decision**, for two reasons the audit is obligated
to surface: (1) it fully replaces the strict/rich substrate composer as the default generator (big blast
radius, and it leans harder on the Qwen/WKV FORM scaffold — which the owner *has* sanctioned as the
articulation mouth 2026-08-19, but making it the DEFAULT is the owner's call); (2) flipping the parent
with the moat children OFF is the strictly LESS-safe configuration, so the children must ride the same
flip. **Exact next action:** run a real-traffic moat-safety soak of the bundle (fabrication rate on
brain-unknown and Qwen-known-brain-unknown topics, with vs without the entailment/gen-time children),
present the fabrication-rate delta to the owner, and flip only on an explicit yes.

If a genuinely *buildable* Class-A win is wanted this cycle instead: **fix confidence-forthcomingness's
content-exhaustion** (wire an LTM-tier candidate pool into the elaboration planner so a HIGH-confidence
turn actually has an extra grounded fact to chain), which converts #1 from C to a real, load-bearing,
moat-safe A.

## Latent issues found in passing (not flips — correctness)

1. **`BRAIN_GNW_ORGANB_LTM_EXEMPT` docstring drift.** The reader defaults to `"1"` (ON) at
   `webapp/gnw_two_organ_bus.py:123`, so LTM knowledge recall corroborates through the consensus bus by
   default — good, and it is what makes the 15k-fact knowledge base answerable in chat. But the same
   module's prose (lines 51, 114) still say "DEFAULT-OFF (2026-08-27)", while `gnw_three_organ_bus.py:78`
   says "DEFAULT-ON since 2026-08-27". This is the exact docstring-vs-fallback-literal inconsistency that
   produced the #77/#80 hollow-flip bugs. Recommend: a one-line docstring correction + **add a ledger
   row** for this coupling (it is load-bearing and default-ON but untracked).
2. **The open-ended family has no ledger rows at all** despite being wired (5 flags). If the owner flips
   the bundle, the ledger must gain rows in the same cycle (the `board_sync_on_status_change` gate will
   otherwise be the only thing tracking it). Same for the two cross-edges and common-ground #152.
3. `XEDGE_SS_DECLARATIVE` is a byte-identical refactor flag, not a faculty — worth deleting or clearly
   labelling so a future audit does not mistake it for a pending faculty flip.

## Bottom line

No blind flip is safe this cycle. The chat-path integration frontier is gated on ONE owner decision (the
`BRAIN_OPEN_ENDED` bundle) plus a small number of real builds (confidence content-exhaustion; the
cross-edge decision-couplings; common-ground production wiring). Present the open-ended moat-safety soak
to the owner as the highest-leverage next step; build the confidence LTM-candidate fix if a bankable
Class-A is wanted first.
