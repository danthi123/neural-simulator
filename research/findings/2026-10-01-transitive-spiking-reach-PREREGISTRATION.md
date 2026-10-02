---
type: preregistration
status: preregistration
date: 2026-10-01
packet: transitive-spiking-reach
runner: research/runners/transitive_spiking_reach_organ.py
scorer: tools/s2p_score.py
---
# transitive-spiking-reach: the chain walk and the reachability test on spikes (PRE-REGISTRATION, 2026-10-01)

Written by the cloud Claude session 2 (packet factory) before any gate-seed run. HANDOFF:
`docs/drafts/transitive-spiking-reach/HANDOFF.md`. Owner ruling (runbook section 6, `iw2-stage1-credit`): the
transitive-chat flip counts toward Stage 1 as PARTIAL until the chain walk moves into the spiking substrate.

## 1. The host decisions being moved

`webapp/reasoning_transitive_chat.resolve_transitive_query` walks the chain in a host loop
(`confidence_gated_chase`: each hop calls `composer.query_patient`, picks distractors, drives a workspace slot). It
then answers "yes" with host code, `target in path[1:]`. Runbook RULING `iw2-verify-go`: "The next node is the
composer's read, and the reachability test is host code." The workspace ignition already decides when the chase
stops, and that part is not at issue here.

## 2. The mechanism (code at the pinned revision; default OFF behind `BRAIN_TRANSITIVE_SPIKING_REACH`)

One bridge per (fact set, relation). `wave[c]` is one excitatory assembly per concept, with a directed dense
projection `wave[x] -> wave[y]` for each stored fact R(x, y). `goal[c]` receives the question's target word as a
sensory current. `match[c]` is a coincidence population that fires only when `wave[c]` and `goal[c]` are both active.
`detect` is ONE population fed by every `match[c]`. The question's subject word starts a wave at `wave[start]`. The
wave runs down the chain and dies at a leaf. The host reads only `detect`'s spike count against a frozen
`DETECT_MIN`. It never reads which concept fired to decide yes or no. The `derived_from` narration (after the
decision) is read off the wave's first-spike order, backtracked along stored edges.

Lesion `BRAIN_TRANSITIVE_REACH_LESION=1`: the hop projections' transmission gate is closed, so the wave cannot leave
the start, `detect` stays silent, and the route abstains.

Declared residuals: the edges are TRANSCRIBED from the composer's fact index (`composer.kb`, relation-matched with
`lexical_lemma.lemma_verb`) when the bridge is built. That is a host wiring transcription; learning the edges by STDP
from the teach turns is the next rung. The question parse (`_TRANSITIVE_RE`) and the direct-fact short-circuit are
unchanged. The organ is its own bridge, not co-resident with the composer (the same convention as the W3 and GNW
organs). This is a one-brain integration debt, named and not hidden.

## 3. Step 1: operating point (dev seeds 1-8 only; no verdict)

Every GRID index (7) runs the organ battery on dev seeds 1-8 (pool, `--sweep`). Selection, by `tools/s2p_score.py
select` only: the FIRST index in grid order that passes on all 8 dev seeds. If none passes, the method is banked and
NEEDS-CLAUDE names the next method (lateral inhibition on `match`, then depth). Dev record before registration
(the build agent, local numpy, 2026-10-01; no gate weight): GRID 0-4 passed on dev seeds 1-8, and GRID 5 and 6 failed
as documented. Limit, already known: chains beyond 3 hops reach weakly at the chosen `W_HOP`. The battery tests up
to 3 hops, the depth of the iw2 probe.

## 4. Step O: the organ gate (gate seeds 42/43/44/100/101/102, the selected index)

Cases (chain cat -> ball -> worm -> river, relation "precede"): derived2 (cat reaches worm), derived3 (cat reaches
river), reversed (river does NOT reach cat), unrelated (with an isolated dog -> bone, cat does NOT reach bone),
branch_river + branch_sun (with cat -> sun added, both reached), lesion_derived3 (lesion: cat does NOT reach river),
path3 (the read-out path for cat -> river is exactly [cat, ball, worm, river]). **Bar: PASS iff `all_pass` on 6/6 gate
seeds.** Cell validity is as in the fb-spiking-witness prereg section 4: the sidecar is valid at the pin, `op_index`
equals the selection, and `prereg` names this file. UNDEFINED never counts. One re-run per UNDEFINED cell.

## 5. Step R: reply-level gate (real handler, LB shards, the same six seeds)

Rows `transitive-spiking-reach` (teach cat -> ball -> worm -> river, ask "does cat precede river?": intact states
"river" and "reasoned", lesion loses it) and the control `transitive-spiking-reversed-ctl` (the chain taught in
reverse: intact abstains, lesion unchanged), in research/runners/lbf_rows/transitive_spiking_reach.py. Both arms set
`BRAIN_TRANSITIVE_CHAT=1`, `BRAIN_TRANSITIVE_SPIKING_REACH=1` and `BRAIN_TRANSITIVE_REACH_OP=<selected>`. **Bar:
PASS iff the capability is reply-level load-bearing on 6/6 AND the control holds 6/6**, under every rule of
`tools/iw2_score.py` step R plus the operating-point sidecar check. Scored only by `python tools/s2p_score.py R`.

## 6. What a PASS lets the finding say (fixed sentences)

"The chain walk is a spiking wave over the relation's facts, and whether the queried target was reached is read
from one detector population fed by goal-gated coincidence units. No host loop walks the chain, and no host test
compares the target to a path. The edges are transcribed from the composer's fact index (a wiring residual), and the
organ is a separate bridge (a one-brain debt)." The Stage-1 credit becomes FULL only when P4 makes this the faculty's
default path and steps B/F pass there.

## 7. Predictions

Selection picks GRID 0; O PASS; R PASS. The risk is the handler path: the probe's facts must reach `composer.kb`
with the relation lemma "precede".

## 8. Seeds, cost, files

Dev seeds 1-8 (step 1); gate seeds 42, 43, 44, 100, 101, 102. Step 1: 8 pool lines, a few minutes each. Step O: 6 pool
lines. Step R: 12 LB shards, `mem_gb=8`. Outputs go under `research/findings/raw/_s2p/transitive-spiking-reach/<P7>/`
and `research/findings/raw/_load_bearing/_shards/s2p-R-tr-<P7>/`.
