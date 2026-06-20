# Phase C — Task 2 (the cheap-first K=2 WHOLE-TURN loop): the host `_scan` is GONE, control on-substrate — GO (2026-06-19)

**Verdict: the WHOLE who/what conversational turn runs as ONE persistent loop on the production `OneBrainComposer`
bridge, with the host `_scan` `for/if/return` orchestrator REMOVED — the spiking basal-ganglia WTA decides
answer-vs-abstain. `==host` on who/what at K=2 (6/6 seeds), the no-confab MOAT holds (0 false-accepts across all 6
seeds), and the full anti-cheat battery passes (6/6 each). HONEST SCOPE: one residual host DATA read at S5 (the cleanup
score → the decoded-line drive; Task 1 ruled the on-bridge projection out — a NEF-thresholded on-bridge cleanup is the
separable lever to close it). NO `sim/` edit (reuse-by-import). numpy is the exact oracle path (the FHRR algebra is
exact + the sequencer match cascade is deterministic given the build seed); the CuPy co-resident confirm is the
controller's decisive run.**

This is the cheap-first END-TO-END loop the Phase C design (`2026-06-19-tier2-phaseC-integrated-loop-design.md` §1.2,
§5 Task 2) called the bar for "Phase C reached": comprehend → store → reconstruct/unbind/cleanup → match →
answer/abstain, sequenced by the substrate, host control GONE.

## What the loop composes (the seam wiring — confirm the host `_scan` is gone)

`LoopComposer` (`research/runners/_phaseC_task2_wholeturn_loop.py`) subclasses Phase A's `SynapticH4Composer` (so the
STORE front is the synaptic bind→store hand-off, inherited verbatim) and OVERRIDES `query_patient`/`query_agent` to run
the whole turn through Phase B's on-substrate sequencer. The seam table (design §2.2), as wired:

| seam | from → to | mechanism (as wired in Task 2) | source |
|---|---|---|---|
| S0 | sentence → role firing | the on-bridge `BridgeParser` (`OneBrainComposer.hear`) — the role it FIRES selects each bind | ALREADY-SYNAPTIC (reused) |
| S2 | `fill_i` → `bound_i` → `acc` | diagonal + unit complex synapses; the bind/bundle stays on-bridge | ALREADY-SYNAPTIC (reused) |
| **S3** | **`acc` → store block (the WRITE)** | **Phase A's `acc → store-readout` unit complex synapse (`SynapticH4Composer`)** | **PHASE A — inherited** |
| S4 | store block → reconstruct → unbind 4 roles → cleanup → membrane | fire the trigger; block-diagonal unbind + cleanup complex synapses; `Re(c)` on the cleanup neurons | ALREADY-SYNAPTIC (`block_role_scores`) |
| **S5** | **cleanup membrane (the RESULT) → decoded word-line drive** | **OPTION (b): read the cleanup scores to host (`scores_to_drive`); the residual DATA read (Task 1 ruled option a out)** | **HOST DATA read (1)** |
| **S6** | **cue + decoded lines → spiking match → BG production rule → {ans0/ans1/abstain}** | **Phase B's gated-disinhibition match cascade + BG WTA (`run_sequencer`)** | **PHASE B — reused** |
| S7 | won BG channel → the answer-role word | the mechanical body read (which channel fired → that block's answer role) | HOST body read |

**The host `_scan` is GONE from the query path.** `OneBrainComposer.query_patient`/`query_agent` end in
`self._scan(...)`'s `for got in self._read_blocks(): if all(got[r]==want...): return got[answer_role]` (a Python
control loop that decides which block answers and whether to abstain). `LoopComposer.query_patient`/`query_agent` do
NOT call `_scan` — they call `_loop_answer`, which runs S4 (on-bridge cleanup) → S5 (host read of the scores) → S6
(`run_sequencer`: the spiking match cascade settles, the BG production rule selects the channel in spikes) → S7 (read
the won channel). The `for/if/return` that branched answer-vs-abstain is replaced by the spiking BG WTA — the only
data-dependent branch (which block answers / abstain) is now on the substrate.

**The sequencer is role-agnostic.** Phase B's `build_sequencer_bridge` matches on two cue roles (`cueA`, `cueX`) and
the body-read picks a third. `what_does` drives the cue with (agent, action) and reads back the patient; `who_does`
drives the cue with (patient, action) and reads back the agent — the SAME spiking circuit, only which cleanup roles
feed the cue/decoded lines differs (set per query in `_loop_answer`). One sequencer serves who AND what.

## Result (numpy oracle, D=64, real `OneBrainComposer`; runner `_phaseC_task2_wholeturn_loop.py`)

Seed 42 (the smoke, every gate GREEN):
```
seed 42 D64: GO  [==host=True correct=True MOAT=OK seq-lesion-safe=True store-lesion-collapse=True
                  perm-rule-inverts=True perm-store-carries=True]
  what/blk0-present:loop=north|host=north   what/blk1-present:loop=river|host=river
  what/absent-agent:loop=None|host=None      what/absent-action:loop=None|host=None
  what/cross-no-block:loop=None|host=None
  who/blk0-present:loop=dog|host=dog          who/blk1-present:loop=cat|host=cat
  who/absent-patient:loop=None|host=None       who/absent-action:loop=None|host=None
  who/cross-no-block:loop=None|host=None
```

6-seed summary (42–47), numpy, the GO bar (every gate GREEN every seed):
```
SUMMARY (6 seeds, K=2, host `_scan` GONE, S5=option-b host-read, control on-substrate):
  ==host 6/6  present-correct 6/6  MOAT 6/6 (total false-accepts 0)  seq-lesion-safe 6/6
  store-lesion-collapse 6/6  perm-rule-inverts 6/6  perm-store-carries 6/6  -> GO
```
(seeds 42/43/44/45/46/47 each: `==host=True correct=True MOAT=OK seq-lesion-safe=True store-lesion-collapse=True
perm-rule-inverts=True perm-store-carries=True`.)

The GO bar (design §4.1), checked every seed:
1. **==host on who/what at K=2.** The loop's `query_patient`/`query_agent` == the host `_scan` path on the SAME store,
   for both present cues (block 0 AND block 1 — the scan reaches block 1) and all moat cues.
2. **The no-confab MOAT — the HARD gate — holds: 0 false-accepts.** Every no-confab cue abstains (the BG WTA selects
   the `abstain` channel; the emitted answer is `None`): absent-agent / absent-action / cross (agent-of-block-0 +
   action-of-block-1) for `what`, and absent-patient / absent-action / cross for `who`. A single false-accept at any
   seed is a FAIL — the moat is never traded for a pass.

## The anti-cheat battery (design §4.2 — all on the loop)

- **sequencer-LESION fails SAFE:** cut the S5 result→sequencer drive on a PRESENT cue → the decoded lines get zero
  drive → the match can't fire → the loop ABSTAINS (`None`), never confabulates a wrong block. (Phase B's lesion, in
  the loop.)
- **store-LESION collapses recall:** a `LoopComposer` built with the Phase-A `acc→store` synapse SEVERED (`lesion=True`)
  → the store readout gets nothing → the cleanup is garbage → the present cues no longer recall their facts. Proves the
  on-bridge store hand-off is load-bearing in the loop (not a parallel host write). (Phase A's lesion, in the loop.)
- **permuted-RULE INVERTS:** swap the BG match→answer production rule at QUERY time (`run_sequencer(permute=True)`,
  Phase B's exact anti-cheat) → the block-0-present cue routes to block 1's patient and block-1 to block 0's. Proves
  the BG selection carries the conditional, not a fixed scan order. (Subtlety: the permute is on the production rule
  that READS the spiking match, not the bridge wiring — building the bridge permuted too would double-cancel; this was
  the one bug found + fixed during the build.)
- **permuted-STORE carries content:** synaptically route a DISTINCT fact (`fox see tree`) into block 0, read block 0
  directly → it holds the routed fact. Proves the synaptic store write carries content. (Phase A's anti-cheat, in the
  loop.)

## The honest scope (what the host still does after Task 2)

After the loop, the host does exactly three things in a who/what turn — all legitimate per the BRAIN-BASED-ONLY
standard (the sensory/body boundaries) PLUS one residual DATA read:

1. **text-in** (the sentence string) — the legitimate sensory boundary.
2. **the concept codes at parse time** (S1, `comp.concepts[word]` → the kick) — comprehension/sensory boundary, like a
   retina rendering input.
3. **the S5 result-read** — the cleanup score → the decoded-line drive (`block_role_scores` → `scores_to_drive`). This
   is the ONE residual host DATA read. Task 1 (`2026-06-19-phaseC-task1-S5-seam-derisk.md`) ruled out the on-bridge
   projection (option a): a graded cleanup score through a binary Izhikevich spike loses the relative magnitude the
   match needs (the point-neuron graded-magnitude limit) — a fixed projection either washes the match out or breaks the
   moat. So the cheap-first loop uses option (b): one number is read to host between the cleanup and the sequencer.
4. **the S7 body-read** (which channel won → that block's answer-role word) — the body read, like the nav cascade
   reading the winning motor pool.

**The CONTROL FLOW between ops — which op next, match-the-cue, answer-vs-abstain — is on the substrate.** That is the
owner's "real one brain" target for the who/what turn, reached at the cheap-first option. The lever to close the last
host read (S5) is a **NEF-thresholded on-bridge cleanup** (Stewart-Tang-Eliasmith — already the composer's cleanup
mechanism) with on-bridge input normalization; that is a separable next step (the controller's, per the Task 1 finding),
NOT on the Task 2 cheap-first path.

## What is OUT of cheap-first scope (later phases, per design §1.3)

negation/yes-no gating through the sequencer, the patient RENDER as a spiking word-order emission, recursive embedded
CLAUSES, multi-hop reasoning, multi-turn anaphora, reconsolidation, extend to K∈{4,8} + all 4 roles (Task 3), the
fold-into-`OneBrainComposer`-opt-in + CI guard (Task 4). The polarity ROLE is still bound/stored (it's in the 4-role
bundle); it is just not sequencer-gated at K=2.

## `sim/` edit

**NONE.** Reuse-by-import: `SynapticH4Composer` (Phase A) + `build_sequencer_bridge`/`run_sequencer` (Phase B) + the
public bridge API + a generalized per-role cleanup read (`block_role_scores`, a parameterization of Phase B's
`block_cleanup_scores`). Matches Phase A and Phase B (both NO `sim/` edit). The deferred `rf_kick` tracker-mask edit
(design §3.2 edit 1, Task 5) was NOT needed: the K=2 loop's micro-schedule does not re-kick an RF group while a disjoint
RF group must hold its trackers (each query's per-block cleanup is sequential, the store is in synapses, and the
sequencer is a disjoint Izhikevich bridge).

## Reproduce

```bash
# numpy oracle (the exact algebra path):
SIM_BACKEND=numpy python -u -m research.runners._phaseC_task2_wholeturn_loop --seeds 42,43,44,45,46,47 --dim 64
# CuPy co-resident confirm (the on-bridge parser trains on the CuPy substrate — the controller's decisive run):
SIM_BACKEND=cupy  python -u -m research.runners._phaseC_task2_wholeturn_loop --seeds 42,43,44,45,46,47 --dim 64 \
    --out research/findings/raw/_phaseC_task2_wholeturn_loop_gpu.json
```

Design: `2026-06-19-tier2-phaseC-integrated-loop-design.md` §1.2, §2, §4, §5 Task 2. Phase A
(`2026-06-19-onebrain-bindstore-handoff-derisk.md`, commit `21bec31c`); Phase B
(`2026-06-19-onebrain-sequencer-derisk.md`, commit `6043101b`); Task 1 / S5 verdict
(`2026-06-19-phaseC-task1-S5-seam-derisk.md`, commit `27c6422e`); the production composer (`one_brain_composer.py`).
