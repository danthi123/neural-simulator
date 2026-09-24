---
type: finding
status: live
date: 2026-09-24
lane: A5 (theory of mind: false-belief wire into chat)
mechanism: PRE-REGISTRATION of a live-chat wire-in for the already-GO'd W3 agent-keyed FALSE-BELIEF register
  (research/runners/_false_belief_register_derisk.py, 6/6-seed GO, b5804d092). New default-OFF flags
  BRAIN_FALSE_BELIEF_CHAT (install the consumer) and BRAIN_FALSE_BELIEF_LESION (force the witnessing gate
  open, collapsing the belief store onto reality). No new spiking mechanism; this reuses-by-import the
  validated bridge/write/query primitives and adds only (a) a live incremental orchestration that lets a
  conversation narrate an arbitrary sequence of place/leave/return/move/query sentences instead of the
  derisk's fixed two-event trial, and (b) a host comprehension boundary that turns chat sentences into
  witnessed/unwitnessed events.
seeds: [7]
verdict: PRE-REGISTRATION only. No result is claimed here. The 6-seed capability gate is deferred to B2b
  (per the midnight plan S15); this document also fixes the seed-7 smoke criteria and the byte-identity-off
  criterion this build lane runs tonight.
runner: research/runners/_tom_false_belief_chat_gate.py
builds_on:
  - research/findings/2026-07-24-W3-false-belief-register-ToM-6seed-GO-adversarially-verified-immunity-claim-corrected.md
---

# Theory-of-mind false-belief wire into chat: PRE-REGISTRATION (2026-09-24)

Branch `research/tom-false-belief-chat`, from `origin/main` at `d7b2a2bb5`. Lane A5 of the 2026-09-24
midnight plan, step S15 item (a).

## What this reuses, unedited

`research/runners/_false_belief_register_derisk.py` (6/6-seed GO, commit `b5804d092`,
`research/findings/2026-07-24-W3-false-belief-register-ToM-6seed-GO-adversarially-verified-immunity-claim-
corrected.md`, banked raw artifact `research/findings/raw/_false_belief_6seed_GO.json`): three GNW
single-content attractor stores (`world`, `belief`, `self`) of the self-schema
meta-schema class, wired with `sim`'s own `transmission_gate="witness_other"` gating the `world -> belief`
write. That file is not modified; every constant and helper it exports (`build_tom_bridge`, `_restore_slice`,
`IGNITE_PA`, `HOLD_STEPS`, `WRITE_DRIVE_STEPS`, `W_WRITE`, `K_LOC`, `STORE_ASSEMBLY`, `FREE_STEPS`,
`_argmax_loc`, `DEFAULT_THRESHOLDS`) is imported by reference, mirroring how `affective_tom_production_organ.py`
reuses `_affective_tom_derisk.py` and how `d3_discourse_event_register_production_organ.py` reuses
`_d3_event_pair_agent_derisk.py`.

## What is new

1. `research/runners/tom_false_belief_chat_organ.py` — a `FalseBeliefChatOrgan` that keeps ONE bridge alive
   across an arbitrary sequence of live events (not the derisk's fixed place-then-move trial), by calling the
   SAME clear-then-ignite write the derisk's `_write_event` closure performs, extracted as a standalone method
   that calls the derisk's own `_restore_slice`/`set_transmission_gate` primitives verbatim. A `query()` method
   replays the derisk's query window (zero-current settle, late-window `cp_firing_states` rate read, host
   `argmax`) at any point in the sequence, so the organ answers a belief query after however many events have
   been narrated so far.
2. `webapp/false_belief_chat.py` — the production glue: a host regex boundary maps chat sentences to four
   event classes (PLACE, LEAVE, RETURN, MOVE) plus a disjoint QUERY class, per-`cache_key` conversation state
   (which agent is tracked, which locations have been named and their ordinal slot, which agents are
   currently "present"), and `observe_turn()` that folds narration sentences into the organ (write-only, reply
   unchanged) and short-circuits a query sentence with the belief-store read-out.
3. `webapp/server.py` — a one-line-guarded hook (`BRAIN_FALSE_BELIEF_CHAT` truthy) placed in `brain_reply`,
   in the same disjoint-turn-class position as the discourse-event-register block (after affect/episodic/
   worldmodel/multiref/discourse, before causal/comprehension), plus a `_SESSION_FALSE_BELIEF.pop(cache_key,
   None)` line in the existing per-session reset block.
4. `research/runners/lbf_rows/tom_false_belief_chat.py` — `EXTRA_LESIONS["tom-false-belief"]` and
   `EXTRA_PROBES` entries in the exact shapes `FACULTY_LESIONS`/`FACULTY_PROBES` use
   (`research/runners/load_bearing_fraction.py`), plus an `EXTRA_TURNS` list of new
   `PROBE_TURNS`-shaped tuples (label, message, session, reset, percept, rich) for the AG-REG import hook (or
   a follow-on integrator) to merge into `_TURN_BY_LABEL`, since the plan's row contract does not name a
   turns-export convention and the closest precedent (`_EXTRA_TURNS` in
   `research/runners/onebrain_regression_battery.py`) already exists for exactly this purpose.
5. `research/runners/_tom_false_belief_chat_gate.py` — the capability-gate runner this document's criteria
   below are scored against.

## Honest scope, declared up front (never claimed closed)

- **Witnessing is host-parsed.** Whether the tracked agent is "present" when an event is narrated comes from
  a regex over "X leaves the room" / "X returns", not from a spiking organ. No existing production organ
  computes third-party physical presence from text (the discourse-event register tracks WHO did WHAT, not
  who is IN THE ROOM), so this is a genuine comprehension-boundary residual, not a shortcut of convenience.
  Per the plan's own framing this wire is credited as **"belief-store read-out given host-parsed witnessing"**
  unless a future rung derives presence from a spiking organ.
- **The action read-out is a host `argmax`** over the belief store's late-window firing rate (identical
  instrument the derisk itself uses, `_argmax_loc`) — the nav-readout-scaffold precedent the W3 finding
  itself already declares.
- **One concurrent (agent, object) scenario per conversation.** A second simultaneous false-belief scenario
  in the same `cache_key` is out of scope; a new PLACE sentence for a different tracked agent starts a fresh
  scenario (the old bridge is discarded). At most `K_LOC=4` distinct locations per scenario (the derisk's own
  geometry); a fifth named location aliases onto slot 3.
- **The open-ended chat path is not wired** (only the normal `brain_reply` turn path). This mirrors the
  codebase's own staged-rung convention for newer faculties (E2/D6/silent-WM were wired into the normal path
  first, open-ended folded in a later rung) rather than a held-back shortcut.
- **Change-of-location only** (Sally-Anne script), not unexpected-contents — same scope limit the underlying
  W3 register already carries.

## Flags

- `BRAIN_FALSE_BELIEF_CHAT` (default unset = OFF): installs the hook. OFF -> the module is never imported and
  the block is skipped -> **byte-identical** to pre-wiring on every turn, including a turn whose text happens
  to match one of the new regexes.
- `BRAIN_FALSE_BELIEF_LESION` (default unset = OFF): forces the witnessing gate open at every write AND
  through the query (mirrors the derisk's `lesion_other=True`) -> the belief store collapses onto reality ->
  an unwitnessed-move query answers with the TRUE location instead of the stale one.

## Byte-identity-off criterion (this lane's own gate)

Ten turns (five ordinary + a full five-sentence Sally-Anne narration ending in a belief query) are run twice
through the SAME fresh numpy build at `cfg.seed=7`: once with `BRAIN_FALSE_BELIEF_CHAT` unset, once with it
explicitly `"0"`. Both `answer` sequences and every other response field must be identical, reported as a `byte_identical_off`
boolean field in this lane's own byte-identity artifact (not yet run as of this pre-registration).

## Seed-7 smoke criterion (dev seed only; not a GO, not one of 42/43/44/100/101/102)

`research/runners/_tom_false_belief_chat_gate.py --seed 7 --n-items 8` drives >=8 change-of-location items
through the LIVE `FalseBeliefChatOrgan` (not the standalone derisk trial runner) via the SAME host-sentence
parser `webapp/false_belief_chat.py` uses, reusing the derisk's own anti-cheats:

1. **false-belief accuracy** (intact, unwitnessed-move items): predicted belief-location == the STALE
   (pre-move) location, well above chance (`1/K_LOC=0.25`).
2. **true-belief control** (intact, witnessed-move items): predicted belief-location == the NEW location, and
   agrees with the reality read (the belief updates when the tracked agent is present — not "always predict
   the first location").
3. **reality baseline must FAIL**: the WORLD store's argmax read on unwitnessed-move items must NOT recover
   the false belief at better than the false-belief-acc bar (it should track the true, current location).
4. **other-lesion collapses**: with `BRAIN_FALSE_BELIEF_LESION=1`, the same unwitnessed-move items' predicted
   belief-location collapses toward the reality/chance floor (the derisk's `chance_loc<=0.45` bar).
5. **scramble-witnessing collapses**: permuting which events are marked "witnessed" (independent of the
   text's actual leave/return narration) collapses accuracy scored against the TRUE witnessing schedule
   toward its 0.5 floor (the derisk's `scramble_chance<=0.70` bar).

A per-item verdict is scored with `tools.verdict.Verdict`, mirroring the derisk's own gate structure
(`false_belief_acc`, `reality_baseline_max`, `chance_loc`, `scramble_chance` thresholds carried over
unchanged from `DEFAULT_THRESHOLDS`). This is a **de-risk (1 seed)**, not a claim of GO; the 6-seed
capability gate over seeds 42/43/44/100/101/102 is explicitly deferred to B2b (S28) per the plan, and is
never run from this branch tonight.

## What is NOT claimed tonight

No GO/NO-GO verdict on the wire-in as a whole. No 6-seed run. No merge to `main`. If the seed-7 smoke and the
byte-identity-off check both pass, the branch is pushed default-OFF and named in the B2b prereg as a
conditional flag per the plan's own rule (S15's success/fallback criteria).
