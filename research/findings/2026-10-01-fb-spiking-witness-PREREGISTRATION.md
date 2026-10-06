---
type: preregistration
status: preregistration
date: 2026-10-01
packet: fb-spiking-witness
runner: research/runners/tom_spiking_witness_organ.py
scorer: tools/s2p_score.py
---
# fb-spiking-witness: who saw the move, decided by spikes (PRE-REGISTRATION, 2026-10-01)

Written by the cloud Claude session 2 (packet factory) before any gate-seed run. HANDOFF:
`docs/drafts/fb-spiking-witness/HANDOFF.md`. Owner ruling (runbook section 6, `iw2-stage1-credit`): the
false-belief flip counts toward Stage 1 as PARTIAL until the witnessing decision moves into the spiking substrate.
This packet is that move.

## 1. The host decision being moved

`webapp/false_belief_chat.py` keeps `_Scenario.present`, a host dict flipped by the LEAVE/RETURN regexes, and
passes `witnessed = present[tracked_agent]` to the W3 organ at every move. That boolean decides whether the belief
store is cleared, whether the helper ignites the new member, and the `witness_other` gate. Runbook RULING
`iw2-verify-go`: "Who saw the move is decided by the host witnessing parse, and that decision gates the write."

## 2. The mechanism (code at the pinned revision; default OFF behind `BRAIN_FALSE_BELIEF_SPIKING_WITNESS`)

One bridge: the W3 register's three stores, unchanged, plus a presence/absence FLIP-FLOP. `presence` and
`absence` are self-recurrent NMDA attractors of the store class. A perceived arrival (the tracked agent places
the object, or returns) drives `ev_arrive`, which ignites `presence` and drives `arrive_inh` onto `absence`. A
perceived departure drives `ev_leave`, which ignites `absence` and drives `leave_inh` onto `presence`. Each
attractor keeps the other's inhibitory pool firing, so the state holds with no further input. The T13 exponential-Euler
conductance guard (`enable_bounded_conductance_integration`) is ON in this bridge only. Without it, explicit-Euler
GABA overshoot made inhibition raise the attractor's rate (dev seed 7). With a lone presence attractor and no flip-flop,
presence re-ignited 25-50 ms after the inhibition ended (dev seed 7). Both are recorded in the organ's docstring.

Two write modes, frozen as GRID entries (operating points):
- `relay` (GRID 0-3): fully synaptic. `relay_k` fires on the coincidence of `world_k` and `presence`; it ignites
  `belief_k` and drives an eviction pool onto the other locations' belief members. No gate is opened and no host
  clear touches the belief store.
- `readout` (GRID 4-7): `presence` is read over 20 ms with a frozen threshold (`PRES_MIN`), and that read sets the
  W3 register's validated write (clear, helper, `witness_other` gate). The decision state is spiking. The
  actuation is a host threshold read of one population, the same class as every store's argmax read-out. The
  finding must say which mode ran.

Lesion `BRAIN_FALSE_BELIEF_WITNESS_LESION=1` (and the older `BRAIN_FALSE_BELIEF_LESION`) removes presence's influence.
In `relay` mode the bridge is built without presence -> relay and with a tonic relay drive. In `readout` mode the
read is ignored. Either way every move is written.

## 3. Step 1: operating point (dev seeds 1-8 only; no verdict)

Every GRID index runs the organ battery (section 4) on dev seeds 1-8 (pool, `--sweep`). Selection, by `tools/s2p_score.py
select` only: the FIRST index in grid order that passes on all 8 dev seeds. Relay indices come first, so the
fully synaptic write is chosen whenever one passes. No index passes: the method is banked, and NEEDS-CLAUDE names
the next method (STD-terminated presence, Mongillo-style synaptic state). The capability is not deferred.

Dev seed record before registration (local numpy, Claude, 2026-10-01; carries no gate weight): GRID 4, 5 and 6
passed all 10 cases on dev seeds 1-8. GRID 7 (GABA_B leave receptor) failed on every dev seed tried. GRID 0 failed on
seeds 2-6 and 8 under an earlier 6-case battery. Relay indices 1-3 were not run.

## 4. Step O: the organ gate (gate seeds 42/43/44/100/101/102, the selected index)

Battery (the `CASES` list in the runner; A = arrival, L = departure, W = 300 ms with nothing perceived, digits =
the object perceived at a location, the first being the placement). Expected belief slot in brackets:
false_belief A0L1 [0] · true_belief A01 [1] · return_then_move A0LA1 [1] · two_moves_one_seen A0L1A2 [2] ·
two_unseen_moves A0L12 [0] · absent_after_wait A0LW1 [0] · present_after_wait A0W1 [1] · return_after_wait A0LWAW1 [1]
· lesion_false_belief (lesion) A0L1 [1] · silence_true_belief (presence clamped during the move's read) A01 [0].
A case passes iff the belief argmax is the expected slot AND that slot's late-window rate is at least 0.02 and at least
2x every other slot (`store_read_ok`; a silent store answering slot 0 by default never passes).

**Bar: PASS iff all 10 cases pass on 6/6 gate seeds.** A cell is DEFINED only if its sidecar passes
`tools/b2d_score.py` validity at the pin, `op_index` equals the selection, and `prereg` names this file.
UNDEFINED never counts. One re-run per UNDEFINED cell, with the line unchanged.

## 5. Step R: reply-level gate (real handler, LB shards, the same six seeds)

Rows `fb-spiking-witness` (unwitnessed move: intact "basket", lesion "box") and its control `fb-spiking-true-belief-ctl`
(witnessed move: intact "box", lesion unchanged), research/runners/lbf_rows/fb_spiking_witness.py. Both arms set
`BRAIN_FALSE_BELIEF_CHAT=1`, `BRAIN_FALSE_BELIEF_SPIKING_WITNESS=1` and `BRAIN_FALSE_BELIEF_WITNESS_OP=<selected>`.
**Bar: PASS iff the capability cell is reply-level load-bearing on 6/6 AND the control holds 6/6**, under every rule of
`tools/iw2_score.py` step R (DEFINED cell, S2 `regressed`, degenerate-intact guard, intact content "basket" and not
"box", the store-read guard on `false_belief_tom.belief_rates`, Amendment 1's `lesion-kept`). Every sidecar must carry
the operating point = the selection. Scored only by `python tools/s2p_score.py R`.

## 6. What a PASS lets the finding say (fixed sentences)

- Relay mode: "Whether the tracked agent saw the move is the state of a spiking presence/absence flip-flop, and a
  coincidence population writes the belief only when presence and the new location fire together. No host variable
  carries presence. The sentence classifier and agent-name routing stay host (comprehension residuals)."
- Readout mode: "Whether the tracked agent saw the move is the state of a spiking presence/absence flip-flop. A
  frozen threshold read of that population gates the W3 write (an actuation residual of the read-out class). The host
  presence dict no longer decides."
- Either mode: the host-decision row for the false-belief chat moves "witnessing" from host-owned to substrate-owned
  (relay) or substrate-decided/host-actuated (readout). The Stage-1 credit becomes FULL only when P4 makes this the
  faculty's default path and steps B/F pass there.

## 7. Predictions (expectations, not bars)

Selection picks GRID 4 (readout) unless a relay index passes all 8 dev seeds; O PASS; R PASS. A relay FAIL is a
finding about the coincidence write: its belief ignition is the open defect.

## 8. Seeds, cost, files

Dev seeds 1-8 (step 1); gate seeds 42, 43, 44, 100, 101, 102 (steps O and R). Step 1: 8 pool lines, about 10 min
each. Step O: 6 pool lines, about 2 min each. Step R: 12 LB shards, `mem_gb=8`, about 30 min each. Outputs go under
`research/findings/raw/_s2p/fb-spiking-witness/<P7>/{sweep,O,verdicts}/` and the LB shards under
`research/findings/raw/_load_bearing/_shards/s2p-R-fb-<P7>/`.
