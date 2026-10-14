---
type: finding
status: provisional
claim_check: measured (exact integer rss_kb/hwm_kb values read from the two g6b RSS sidecar copies in artifacts; select/circuits_built read from g6b_on_rss.jsonl; reply bytes diffed between the two g6b jsons; delegate verdict per RULING models; preconditions per RULING c5-next-1012; awaiting Claude review)
date: 2026-10-12
lane: c5 / g6 replay-fired pair (the idle-replay organ fires at an idle turn; C5 memory bar vs the <= 2 GB/turn target)
mechanism: The g6b OFF/ON whole-brain battery pair on wspool (rev e3ac0da4cd2e685b9aa376860ee418c6aa11eaa2, seed 42, sequential arms, caps 19/22 GB, both rc=0) with the turn set emo,bc_a,bc_b,selfinit so an idle-trigger turn exists; ON carries outer-env BRAIN_IDLE_REPLAY=1 and the organ's leaklog; both arms carry the flag-independent per-turn rss sidecar
seeds: [42]
preconditions:
  - replay fired in ON: exactly one select event (k=5, circ_k=5, circuits_built=1) at the selfinit turn in g6b_on_rss.jsonl; zero select events in g6b_off_rss.jsonl — MET
  - ON-OFF <= 2 GB per turn: max +6324 KiB rss (+7580 KiB hwm) at selfinit — MET
artifacts:
  - research/findings/raw/_regression_battery/c5_offcurve_diag/g6b_off.json
  - research/findings/raw/_regression_battery/c5_offcurve_diag/g6b_on.json
  - research/findings/raw/_regression_battery/c5_offcurve_diag/g6b_off_rss.jsonl
  - research/findings/raw/_regression_battery/c5_offcurve_diag/g6b_on_rss.jsonl
verdict: C5 g6b pair = PASS (agent-filed PROVISIONAL): replay fired in ON and the max ON-OFF is 6324 KiB rss vs the 2 GB bar; single seed 42, tiny-demo brain — no six-seed or scale claim. <!--derived-->
---

# RESULT: g6b replay-fired pair — replay FIRES, owns its reply, and costs ~6 MB: C5 memory bar MET

**Date:** 2026-10-12 · **Status:** PROVISIONAL (agent-filed, verdict pending Claude's review) ·
**Seeds:** 42 (single seed; no generalization claim)

**C5 STATUS: PASS (agent-filed PROVISIONAL) — both `c5-next-1012` (a) preconditions are met:** (1)
replay fired in ON (exactly one `select` event, `k=5, circ_k=5, circuits_built=1`, at the selfinit
idle turn; zero in OFF), and (2) the max ON−OFF is +6324 KiB rss <!--derived--> (0.0060 GB
<!--derived-->; +7580 KiB hwm) against the 2 GB/turn bar — ~330x <!--derived--> headroom. One MiMo
Pro delegate (RULING `models`) re-read all four artifacts and confirmed every number, the select
counts, and the reply bytes before issuing the verdict line "C5 g6b pair: PASS".

## Per-turn numbers (exact integers)

<!--derived--> each cell below is a readout of the two sidecars research/findings/raw/_regression_battery/c5_offcurve_diag/g6b_off_rss.jsonl (OFF) and research/findings/raw/_regression_battery/c5_offcurve_diag/g6b_on_rss.jsonl (ON); the Δ columns are their differences.

| turn | OFF rss_kb | ON rss_kb | Δ rss (KiB) | OFF hwm_kb | ON hwm_kb | Δ hwm (KiB) |
|---|---|---|---|---|---|---|
| emo | 6231220 | 6230184 | −1036 | 6388168 | 6387096 | −1072 |
| bc_a | 11685200 | 11687724 | +2524 | 11776668 | 11778768 | +2100 |
| bc_b | 11684336 | 11686860 | +2524 | 11776668 | 11778768 | +2100 |
| selfinit | 16914020 | 16920344 | **+6324** | 16914020 | 16921600 | +7580 |

Max Δ rss +6324 KiB = 0.0060 GB <!--derived--> vs the 2 GB bar (bar MET). Both arms show the same
per-kind retention staircase (emo → bc → selfinit ≈ +5.3 GB per new turn kind, bc_a→bc_b same kind
= +0) — the retention defect is arm-independent, so it cancels in the ON−OFF column (see the
separate battery-worker retention RESULT of 2026-10-12).

## Precondition 1 — replay fired, and it owns the reply

The ON sidecar's only organ record of note, immediately before the selfinit battery_turn:

    {"event":"select","circ_k":5,"k":5,"circuits_built":1,"turn":1,"rss_kb":16916284}

The OFF sidecar research/findings/raw/_regression_battery/c5_offcurve_diag/g6b_off_rss.jsonl has zero `select` events (4 battery_turn records only). The selfinit reply differs
by arm and is organ-authored in ON: composer=`brain_store`, path=`idle-replay`,
`winner_fact=['cat','eat','fish']`, `decode_matches_winner=true` → "Something's been on my mind —
cat eat fish." (43 chars), versus OFF composer=`onebrain`, path=`light-numpy-deferred-wander` →
"Something's been on my mind — cat eat worm. What does cat eat?" (62 chars). A full `diff` of
research/findings/raw/_regression_battery/c5_offcurve_diag/g6b_off.json against research/findings/raw/_regression_battery/c5_offcurve_diag/g6b_on.json confines every byte difference to the selfinit block: emo/bc_a/bc_b are byte-identical
across arms — the organ does not contaminate active chat.

## Caveats (delegate-verified honesty boundary)

- Single seed 42 — no six-seed replication, so no generalization claim (non-negotiable 6).
- `tiny-demo` brain (5–6 fact KB, template renderer), not full-brain or GPU-mode.
- Per-kind retention runs in BOTH arms, so ON−OFF bounds only the organ against a retaining
  baseline; it does not bound retention cost itself.
- One pair, sequential (ON later); run-order effects are only controlled by the byte-identical
  active turns.
- The select is count-driven (`winner_is_most_excitable=false`), fine for the precondition but
  relevant to later excitability work.
- Exactly ONE replay event was observed — multi-idle accumulation is unmeasured here.
- Agent-filed PROVISIONAL: rc=0/caps/exits come from the dispatcher exits.json, not these artifacts.

## Scope

Establishes: at an idle turn the organ fires, builds one circuit, authors its own reply, costs
≤ 6324 KiB rss over OFF, and leaves active turns byte-identical. Does NOT establish six-seed
generality, scale/full-brain behavior, replay-improves-memory, or retention cost bounds.