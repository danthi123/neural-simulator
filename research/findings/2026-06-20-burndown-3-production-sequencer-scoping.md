# Burndown #3 — scale the on-substrate sequencer K=2 → production K=32 + fold into the production query path (retire the host `_scan`): build scoping (2026-06-20)

**Type:** READ-ONLY design / staging doc. NO code written, NO experiments run. One design document. Stayed on `main`.
**The shortcut (#3):** the SHIPPED production `OneBrainComposer` orchestrates its who/what (and chain) queries with a
host `_scan` — a Python `for`-over-stored-facts + `if cue-matches then emit else continue/abstain`. That cue-match
COMPARISON + answer/abstain routing is a COGNITIVE control-flow op done by host bookkeeping = a brain-based-only
shortcut (`2026-06-20-shortcut-burndown-inventory.md` #3).
**The proven replacement:** the **Phase-B spiking sequencer** (gated-disinhibition match cascade + BG production rule),
validated GO replacing `_scan` for **K=2 who/what** in the Phase-C Task-2 whole-turn loop
(`2026-06-19-onebrain-sequencer-derisk.md` + `2026-06-19-phaseC-task2-wholeturn-loop.md` +
`research/runners/_phaseC_task2_wholeturn_loop.py`).
**The gap (#3):** scale that K=2 sequencer to the PRODUCTION scale (K up to 32 stored facts; the production vocab V and
D) and fold it into the production query path, retiring the host `_scan`.
**Framing honesty:** this is ENGINEERING on a PROVEN mechanism, not new research. So it is a staging/design pass, not a
deep-research gate. The one place it could hit a substrate boundary — the 1-of-K=32 match-cascade discrimination — is
flagged with a predicted failure mode + the exact per-K sweep that would test it.

---

## 0. TL;DR (the load-bearing answers)

- **Exact scope of #3:** the host `_scan` / inline cue-match `for got in self._read_blocks(): if ...` orchestrates
  **seven** production read-paths in `one_brain_composer.py`: `query_agent` (→`_scan`), `query_patient`, `ask_yes_no`,
  `render_fact`, `query_chain` (rides on `query_patient`), `_find_cued_block`/`update_on_mismatch` (reconsolidation),
  and `count_facts`. **#3's core is the who/what pair (`query_patient`/`query_agent`) + the chain (`query_chain`,
  which is just iterated `query_patient`).** `ask_yes_no` and `render_fact` are the SAME cue-match op (one more
  body-read role) and fold in for free. `_find_cued_block`/`count_facts` are reconsolidation control-flow — a SEPARATE
  item (out of #3's who/what core; flagged below). The deep-frontier #12 (the FHRR bind algebra) is untouched.
- **The staged plan:** 5 bite-sized stages — **(S0)** generalize the sequencer builder K=2→K (the K-way priority WTA +
  the K-way production rule), CPU; **(S1)** wire the already-validated on-bridge S5 divisive-norm into the loop (close
  the one residual host read, CPU); **(S2)** scale K∈{2,8,16,32} at D=128 with the per-K margin sweep (the boundary
  test), CPU then GPU; **(S3)** fold into `OneBrainComposer` as `integrated_loop=True` (default-off, byte-identical) +
  CI guard, covering who/what/yes-no/render/chain; **(S4)** the production 320-scale GO on the real
  `consolidated_320_conversation_demo` path, GPU. Each stage's GO = `==host` + moat-0-FA.
- **The likely BOUNDARY:** the **1-of-K=32 match-cascade discrimination** (each query lights ≤K decoded blocks; the
  worst no-match leak vs the true-match rate must stay separated as K grows — Phase-B's own scope note flags this). The
  #1 spiking cleanup already does **1-of-V at V=320** (a HARDER selection), so the SELECTION primitive is in proven
  range; the open question is whether the K-way match CASCADE (K parallel gated-disinhibition matches + a K-way
  priority/abstain WTA) keeps the margin. **Predicted:** GO to K≈8–16, then a margin-squeeze risk at K=32 from (a) the
  abstain channel needing to be suppressed by ANY of K matches (more leak summed into the default channel) and (b) the
  K-way priority chain's inhibitory fan-out. **Test:** the per-K sweep K∈{2,8,16,32}, reporting `worst-leak vs
  threshold vs true-match` at each K; the honest-negative if it squeezes (below).
- **Reuse vs `sim/` edit:** **reuse-by-import, NO `sim/` edit expected** (matching Phase A/B/Task-2 + the S5
  divisive-norm de-risk, all NO `sim/` edit). The per-RF-synapse transmission gain the scoping once flagged is
  **confirmed NOT needed** (the sequencer gates Izhikevich routes, already gated by `cp_transmission_gain`). The one
  conditional edit (mask the `rf_kick` tracker re-init) is deferred until a multi-op de-risk demonstrates it; flag for
  byte-review only if reached.
- **GO bar + honest-negative:** `==host` on the production who/what(/chain) matrix multi-seed (6 seeds), the no-confab
  moat **0 false-accepts** as the HARD gate. The honest-negative form: if the cascade can't match `_scan` at K=32, the
  residual host op is the cue-match COMPARISON and the boundary is the spike-SNR of the 1-of-K=32 match at scale — a
  characterized cost, the deliverable.
- **Effort:** a **days arc** (≈3–6 focused working days), NOT weeks — every piece is proven and reuse-by-import; the
  only real unknown is the K=32 margin (one sweep). The risk that turns it into >1 week is a K=32 margin squeeze that
  needs the NEF-FS-pool S5 fallback or a hierarchical match — bounded, with the retreat named.

---

## 1. EXACT SCOPE — which production queries the host `_scan` actually orchestrates

The cue-match host control flow in `one_brain_composer.py` is two syntactic forms of the SAME op: the named `_scan`
(line 492) and the inline `for ... in self._read_blocks(): if cue-match: ...` repeated at the call sites. Verified
line-by-line (grep `_read_blocks()` / `_scan(` / `for got in` / `for i, got in`):

| production method | line | host control-flow form | the cognitive op (what `_scan` does) | in #3's who/what core? |
|---|---|---|---|---|
| `query_agent(action, patient)` | 566–567 | `return self._scan({"action":…, "patient":…}, "agent")` | who: cue=(patient,action) → answer agent / abstain | **YES (core)** |
| `query_patient(agent, action, …)` | 554–564 | `for i, got in enumerate(self._read_blocks()): if agent&action match: return …` | what: cue=(agent,action) → answer patient / abstain | **YES (core)** |
| `ask_yes_no(agent, action, patient)` | 569–575 | `for got in …: if full-SVO match: return yes/no` else `unknown` | yes/no/unknown: cue=(agent,action,patient) → polarity tag | **YES (same op, +1 cue role)** |
| `render_fact(agent, …)` | 577–592 | `for i, got in …: if agent match: return rendered fact` | describe: cue=(agent) → emit the matched fact (1-role cue) | **YES (same op, 1 cue role)** |
| `query_chain(cue, actions)` | 594–603 | iterates `query_patient` per hop; abstains on first miss | multi-hop: chained what; the moat holds at every hop | **YES (rides on `query_patient`)** |
| `_find_cued_block(agent, action)` | 647–653 | `for i, got in …: if agent&action match: return i` | reconsolidation: find the block to relabel | **NO — separate item (see §1.2)** |
| `count_facts(agent, action)` | 678–681 | `sum(1 for got in … if agent&action match)` | reconsolidation/test: count matching blocks | **NO — separate item (see §1.2)** |

### 1.1 The core of #3 = ONE role-agnostic op (the Phase-B sequencer already is this)

`query_patient` / `query_agent` / `ask_yes_no` / `render_fact` are the SAME operation at different cue-arities and
answer roles:
- **what** = cue {agent, action} → answer patient (2-role cue, 1 answer);
- **who** = cue {patient, action} → answer agent (2-role cue, 1 answer);
- **yes/no** = cue {agent, action, patient} → answer polarity (3-role cue, 1 answer); `unknown`-on-miss IS abstain;
- **describe** = cue {agent} → answer the whole fact (1-role cue, multi-answer body-read).

The Phase-B sequencer is **already role-agnostic** (`_phaseC_task2_wholeturn_loop.py:82–84`,
`block_role_scores(c, b, role_a, role_x)` generalizes the K=2 `block_cleanup_scores`): it matches on two cue roles and
the body-read picks a third. `query_patient` and `query_agent` already both run through the one `run_sequencer` in the
Task-2 loop (`LoopComposer:159–167`), differing only in WHICH cleanup roles drive the cue lines. So #3's core
(who/what) needs ZERO new mechanism beyond scaling K — the role-agnosticism is built. yes/no folds in by driving a
THIRD cue role into the match (a 3-input gated conjunction, the same gated-disinhibition primitive); describe folds in
by reading back ALL the matched block's roles (the body-read already returns the full row).
`query_chain` is `query_patient` iterated — once the single who/what turn is sequencer-driven, the chain is
sequencer-driven by composition (the no-confab moat holds at every hop, the validated property).

### 1.2 What is a SEPARATE shortcut (NOT part of #3)

- **`_find_cued_block` / `count_facts` (reconsolidation control flow).** These are the cue-match for the
  reconsolidation path (`update_on_mismatch`), which is a documented FOLLOW-ON capability, NOT in the shipped
  who/what core turn (the Phase-C design §1.3 lists reconsolidation as OUT of cheap-first scope; the inventory's
  EXCLUSIONS note flags the reconsolidation PE gate as a follow-on, below the shipped core). They use the same
  `_read_blocks()` cue-match, so the sequencer would extend to them naturally AFTER #3 — but #3's bar is the who/what
  core. **Flag: a small separate item** ("reconsolidation control-flow on the sequencer"), do NOT let it expand #3's
  scope or the K=32 gate.
- **The S5 result-read (the cleanup-score → decoded-line drive).** This was the inventory's item #4 (a SEPARATE
  CLEAN-CONVERSION) — and it is **already CLOSED on-bridge** (the `input_divisive_norm` Carandini-Heeger primitive,
  `2026-06-20-S5-divisive-norm-derisk.md`, GO 3 seeds CPU+GPU, NO `sim/` edit), just not yet WIRED into `LoopComposer`.
  #3 should ADOPT it (stage S1) so the scaled production loop has zero host round-trips, but the S5 mechanism itself is
  not #3's research — it is a wire-in. (Treat the S5 wire-in as a bundled prerequisite of #3, not a separate gate.)
- **The cleanup SELECTION (#1)** is already a fully-on-substrate spiking WTA in `OneBrainComposer`
  (`enable_spiking_cleanup`, `2026-06-20-burndown-1-onebrain-spiking-cleanup.md`, default-off byte-identical). #3 turns
  it ON in the integrated path. Not #3's research; a flag flip.

**So #3's precise boundary:** retire the host `for/if/return` cue-match COMPARISON + answer/abstain ROUTING for the
who/what core (`query_patient`/`query_agent`/`ask_yes_no`/`render_fact`/`query_chain`), at production K=32 + V/D, on the
real `OneBrainComposer`, with the moat intact. Reconsolidation control-flow is a deferred separate item; S5 + #1 are
bundled wire-ins (already de-risked).

---

## 2. THE STAGED PLAN — bite-sized stages to scale K=2 → K=32 + fold into production

Ordered cheapest/CPU-first; each stage ends GREEN against the host `_scan` oracle, with an anti-cheat and the moat
checked. The naming follows the established `research/runners/_phaseC_*` / `_phaseB_onebrain_*` convention. The
load-bearing pair is S0 (generalize the sequencer to K) + S2 (the K=32 margin sweep — the boundary test).

### Stage S0 — generalize the sequencer builder K=2 → K (the K-way priority WTA + K-way rule). CPU.

**What it builds.** The current `build_sequencer_bridge` (`_phaseB_onebrain_sequencer_derisk.py:117`) is hard-coded to
K=2: explicit `m0`/`m1` match pools, `ans0`/`ans1` answer channels, `inh0`/`inh1` interneurons, a 2-element
`blocks_scores[:2]` truncation, the priority chain `ans0→inh0→{ans1,abstain}` / `ans1→inh1→abstain`, and a 2-way
production-rule dict `{(f0,f1)→…}`. Generalize ALL of these to a parameter `K`:
- **K match cascades** (`mw{b}{role}_w`, `mA{b}`/`mX{b}`/`m{b}` for `b∈0..K-1`) — the per-block gated-disinhibition
  match, replicated K times (the wiring loop already iterates `range(V)`; extend the block loop to `range(K)`).
- **K answer channels** `ans{b}` + the **K-way BG priority WTA**: the K=2 block-0-priority chain becomes a standard
  competitive WTA over the K answer channels (lateral inhibition among the K answers; the BG/GPi disinhibition motif
  the project already runs — `g11_bg_runner` `--bg-lateral-inhibition`, catalog A.04). The first-match priority of the
  host `_scan` (return the FIRST matching block) maps to a fixed priority ordering in the WTA inhibition (block i
  inhibits blocks >i), OR — cleaner for K=32 — a single winner-take-all where exactly-one-block-matches is the
  expected case (facts are distinct; a unique cue matches one block), so priority only matters for the rare
  degenerate multi-match. **Design choice:** keep the host `_scan`'s first-match semantics by a graded priority bias
  (earlier blocks a hair stronger), but the common case (unique match) needs only a plain WTA.
- **The abstain channel** stays the tonic default suppressed by ANY of the K matches (the K-way OR into `abstain`'s
  inhibition).
- **The production rule** generalizes from the 2-element dict to "the won `ans{b}` channel → block b's answer role;
  no channel won → abstain" (read the K-way WTA winner; the `permute` anti-cheat becomes a cyclic shift of the
  match→answer map).
- **`run_sequencer`** generalizes `blocks_scores[:2]` → `blocks_scores[:K]` and reads the K-way WTA winner.

**The GO check.** `==host` on the K=4 who/what battery + abstain on absent/cross cues, **3 seeds** (the kernel parity,
CPU/numpy oracle); the moat 0-FA. Anti-cheat: sequencer-lesion fails safe (sever the S5 drive → abstain); permuted-rule
(cyclic shift) inverts the routing. **CPU-first** (the FHRR algebra + the deterministic match cascade run numpy; this
is the exact-oracle parity path). Reuse: extend `_phaseB_onebrain_sequencer_derisk.py` in place / a new
`_phaseB_onebrain_sequencerK_derisk.py`.

### Stage S1 — wire the validated on-bridge S5 divisive-norm into the loop (close the residual host read). CPU.

**What it builds.** The Task-2 loop still reads the cleanup score to host at S5 (option b, the documented residual).
S5 is **already closed on-bridge** by `input_divisive_norm` (Option 4, `2026-06-20-S5-divisive-norm-derisk.md`: GO 3
seeds CPU+GPU at K=2/D=64, operating point `sigma=1.0, gain=0.05, input_gain=1.0`, NO `sim/` edit). This stage WIRES
that divisive-norm-flagged score pool into `LoopComposer` so the scaled loop has zero host round-trips between cleanup
and sequencer (the divnorm-flagged pool's firing replaces the host `scores_to_drive`). The de-risk's own
"Follow-on (NOT on this de-risk's path)" names exactly this drop-in.

**The GO check.** The S5-on-bridge loop `==host` + moat-0-FA at K=2 (parity with the existing Task-2 loop, now with
the host read gone), **3 seeds** CPU. Anti-cheat: the RAW (no-norm) negative control breaks the moat (norm
load-bearing); lesion fails safe; OFF==byte-identical. **Note the scale caveat the de-risk flagged:** the divnorm
operating point was validated at K=2/D=64; the saturated ratio `peak/(gain·mean)` is scale-free, so `gain≈0.05`
SHOULD transfer to D=128/V=320 — but the firing-band placement must be re-confirmed at scale (folded into S2/S4). The
NEF input-norm FS pool (`2026-06-05-composer-cleanup-NEF-GO.md`, Option 1) is the named graded-pool FALLBACK if the
mean-pool firing-band basin needs the matched-filter structure at 320 concepts. **This stage can be merged with S0**
(both CPU, both small) if convenient.

### Stage S2 — scale K∈{2,8,16,32} at D=128 + the per-K margin sweep (THE BOUNDARY TEST). CPU → GPU.

**What it builds.** Grow the store to K∈{2,8,16,32} at the production D=128, all four roles bound
(agent/action/patient/polarity), and run the per-K margin sweep: at each K, report `worst no-match leak` vs
`match_thresh` vs `true-match rate` (the Phase-B scope note explicitly flags re-verifying this as K grows — "more
blocks → more leak lines"). This is the one place #3 could hit a substrate boundary (§3).

**The GO check.** `==host` on who/what + abstain at K=8, K=16, K=32, **6 seeds** (the standing rule for the
noise-sensitive cascade), moat 0-FA at every K; the margin table reported and still separated (true-match comfortably
above threshold, worst leak comfortably below). Anti-cheats: the full battery (sequencer-lesion fails safe;
store-lesion collapses; permuted-rule inverts; permuted-store carries content) at K=32. **CPU for the exact-algebra
parity sub-steps; GPU (`SIM_BACKEND=cupy`) for the real co-resident confirm** (the on-bridge parser trains on CuPy; the
megakernel + CSR cache are GPU). If the margin squeezes at K=32 → the honest-negative branch (§5) with the NEF-FS-pool
or hierarchical-match retreat (§3).

### Stage S3 — fold into `OneBrainComposer` as `integrated_loop=True` (default-off) + CI guard. GPU.

**What it builds.** Promote the loop from the `_phaseC_*` runner subclass (`LoopComposer`) to an opt-in mode on the
production `OneBrainComposer` (`integrated_loop=True`, default off = byte-identical to today), so
`BrainConversationalAgent(composer_kind="onebrain")` can use it. Route the FIVE who/what-core methods
(`query_patient`/`query_agent`/`ask_yes_no`/`render_fact`/`query_chain`) through the sequencer when the flag is on; keep
the host `_scan` as the default (the oracle/CPU path) until the production GO. yes/no = the 3-cue-role match;
describe = the full-row body-read; chain = iterated `query_patient` (composes for free).

**The GO check.** `tests/test_one_brain_composer_agent.py` (the 11–13 tests incl. the three `is None` no-confab
assertions) + `tests/test_brain_conversational_agent.py` pass VERBATIM with the flag OFF (byte-unregressed); a NEW
`tests/test_onebrain_integrated_loop.py` asserts the K∈{2,8} who/what + yes-no + describe + the moat with the flag ON
(GPU-gated, skips gracefully without GPU/the concept cache, like the sibling guards `test_onebrain_spiking_cleanup.py`).
**GPU** for the flag-on assertions (the parser trains on CuPy).

### Stage S4 — the production 320-scale GO on the real demo path. GPU.

**What it builds.** Run the integrated loop on the real `consolidated_320_conversation_demo.py --composer onebrain`
(D=128, V up to 320, k_max=32, the stream-learned cortex codes) — the flagship production conversation — with
`integrated_loop=True`. This is the same battery the 2026-06-18 320-scale GO used (recall 1.00, abstain 1.00, 0
false-accepts), now with the host `_scan` retired.

**The GO check.** `==host` (the host-`_scan` path on the same store) on the 320-scale who/what(/chain) matrix +
the no-confab moat 0 false-accepts, **3 seeds** (42/43/44, the production-demo precedent — the 320-scale recall/abstain
GO was 3-seed). If GO → flip `integrated_loop=True` as the production default (keeping the host `_scan` as an explicit
`--host-scan` oracle / numpy-CPU fallback, mirroring how `rf` is retained as the oracle). **GPU only.**

---

## 3. THE LIKELY BOUNDARY — the 1-of-K=32 match-cascade discrimination

### 3.1 What the boundary IS

Each query fires every stored trigger and the cleanup lands per-block-per-word scores; the sequencer's K match
cascades each compute "does block b's decoded cue-roles == the cue" by gated disinhibition, and the K-way BG WTA picks
the winner / abstains. At K=32 there are 32 parallel match cascades and a 32-channel WTA. **Two margin pressures grow
with K:**
1. **The abstain channel must be suppressed by ANY of K matches** (the K-way OR into the abstain interneuron). On a
   PRESENT cue, one match fires and correctly suppresses abstain — fine. But on an ABSENT cue (the moat case), all K
   match pools should be silent; if K-1 of them leak a little (a near-miss block's partial decoded overlap), the
   summed leak into the abstain-suppression path could spuriously suppress abstain → a moat breach. **This is the
   moat-relevant failure mode and the one the per-K sweep must catch.**
2. **The K-way priority/WTA inhibitory fan-out.** A K=32 winner-take-all has a 32-channel lateral-inhibition fabric;
   the project has been burned by recurrent-WTA instability before (the hand-WTA cleanup NEGATIVE, Rutishauser α>1).
   The retreat here is the BG disinhibition motif (feed-forward, the validated `g11_bg_runner` selection), not a
   recurrent soft-WTA.

### 3.2 Is it within the spiking WTA's proven range?

**The SELECTION primitive: YES, comfortably.** The #1 cleanup spiking WTA already does **1-of-V at V=320** (a 320-way
argmax-by-firing, validated `==` host argmax, moat 0-FA — `2026-06-20-burndown-1-onebrain-spiking-cleanup.md`). Picking
1-of-32 blocks is a SMALLER selection than 1-of-320 words. So the final WTA/priority stage is well inside proven range.

**The match CASCADE: the genuine open question.** The #1 cleanup is a single matched-filter-then-WTA; the sequencer's
match is a CASCADE (gated disinhibition: cue-gate × decoded-line → per-role match → gated AND → per-block match), and
#3 needs K of them in parallel feeding a K-way WTA. Phase-B validated this cascade is CLEAN at K=2 (true match
~0.22–0.25 vs no-match ≤0.10, threshold 0.15). The open question is whether the per-block match stays that clean when
32 blocks are reconstructed in parallel and 32 cleanup read-outs drive 32 decoded-line sets — i.e. whether the
cross-block leak (a near-miss block's decoded words partially overlapping the cue) stays below threshold as K grows.

### 3.3 The prediction + how the de-risk tests it

**Prediction:** GO to K≈8–16 with the K=2 operating point largely intact; a **margin-squeeze risk at K=32**, driven by
(1) the abstain-suppression OR summing more near-miss leak and (2) the cleanup's own per-block fidelity at D=128 (the
batched reconstruct-all-K-blocks-in-parallel read may degrade slightly with K, the documented per-block isolation
property to re-verify). Most-likely outcome: **GO at K=32 with a re-tuned threshold and the S5 divnorm `gain`
re-confirmed at scale** (the score normalization is scale-free, so this is a re-confirmation, not a redesign).

**The test (the per-K sweep, Stage S2):** K∈{2,8,16,32} at D=128, 6 seeds, reporting at each K:
- `true-match rate` (the matching block's `m{b}` firing fraction) — should stay ≳0.20;
- `worst no-match leak` (the highest `m{b}` among non-matching blocks, across all moat cues) — should stay ≲0.10;
- the `threshold` separating them — fixed across K (the GO) or needing per-K re-tune (a yellow flag, still a pass if
  one threshold covers the production K=32);
- **the moat 0-FA at every K** (the HARD gate).

**If it squeezes (the retreats, cheapest-first):** (a) re-confirm the S5 divnorm `gain` at D=128/V=320 (the firing-band
placement); (b) swap the mean-pool divnorm for the NEF input-norm FS pool (Option 1, validated at D=2048/V=320) so the
per-block decoded drive is sharper; (c) a HIERARCHICAL match (coarse pre-filter on the agent cue role → only the
surviving blocks run the full gated conjunction → fewer parallel cascades into the K-way WTA) — a wiring change, still
no `sim/` edit. If NONE recover the margin without a moat breach → the honest-negative (§5).

---

## 4. REUSE vs `sim/` EDIT

### 4.1 Reuse-by-import (the default — everything)

| machinery | file:line | contributes to #3 |
|---|---|---|
| `OneBrainComposer` (persistent co-resident bridge; store-in-synapses; cached operators; megakernel + CSR cache; `enable_spiking_cleanup`) | `one_brain_composer.py:87` | the substrate; S0/S2/S4 already synaptic; #1 cleanup folds in via the existing flag |
| `LoopComposer` + `block_role_scores` (the role-agnostic K=2 loop) | `_phaseC_task2_wholeturn_loop.py:84,118` | the loop to generalize K=2→K (S0) |
| the Phase-B sequencer (`build_sequencer_bridge`, `wire_sequencer_couplings`, `reset_sequencer_state`, `run_sequencer`) | `_phaseB_onebrain_sequencer_derisk.py:117,215,229,254` | the K=2 sequencer to generalize to K (the explicit `m0/m1`, `ans0/ans1`, `inh0/inh1`, `blocks_scores[:2]`, 2-way rule → K-way) |
| `input_divisive_norm` (Carandini-Heeger, validated for S5) | `regions.py:240`; `config.py:440`; `bridge.py:6048` | the on-bridge S5 normalizer to wire into the loop (S1) — already de-risked, NO `sim/` edit |
| NEF input-norm FS pool (the S5 graded-pool fallback) | `research/findings/raw/_spiking_cleanup_nef.py`; op `NEF_CLEANUP_OP` | the S5 retreat IF mean-pool divnorm squeezes at 320 |
| `couple_gate_to_pool` / `_apply_gate_couplings` (gated disinhibition) | `bridge.py` (`couple_gate_to_pool`) | the match primitive, K times in parallel |
| `set_transmission_gate` / `cp_transmission_gain` | `bridge.py` | the sequencer's Izhikevich match routes (already gateable) |
| BG WTA / lateral inhibition (`--bg-lateral-inhibition`, catalog A.04) | `g11_bg_runner.py` | the K-way priority/abstain WTA template (feed-forward, not recurrent) |
| the on-bridge `BridgeParser` | `brain_conversational_agent.py` (`BridgeParser`); `OneBrainComposer.hear` | comprehension (S0), reused verbatim |
| the masked `rf_kick` / `_rf_advance_one` (RF ops sliced) | `bridge.py` (`rf_kick`, `_rf_advance_one`) | the co-residence guarantee |
| CI guards (`test_one_brain_composer_agent.py`, `test_onebrain_spiking_cleanup.py`) | `tests/` | the no-regression + capability pins (S3) |

### 4.2 The `sim/` edit picture — flag each for byte-review

**Honest assessment: #3 needs ZERO `sim/` edit** (matching Phase A, Phase B, Task-2, and the S5 divisive-norm de-risk —
all NO `sim/` edit). Specifically:

1. **The per-RF-synapse transmission gain — CONFIRMED NOT NEEDED.** The scoping once flagged this as the arc's biggest
   potential edit (the RF complex matvec has no gain multiply, so "gate an RF route on a fixed weight set" isn't free).
   **Phase B obviated it** (`2026-06-19-onebrain-sequencer-derisk.md:120–123`): the sequencer gates IZHIKEVICH routes
   (already gated by `cp_transmission_gain`), driven by the cleanup result; it never gates an RF route. #3 inherits
   this — the K-way generalization is more Izhikevich match cascades + a bigger Izhikevich WTA, all on `cp_connections`,
   all gateable today. **No edit.**
2. **The `rf_kick` tracker-mask edit (the conditional one) — DEFERRED, flag only if reached.** `cp_rf_prev_im` /
   `cp_rf_fired` / `cp_rf_spike_step` are re-init whole-array even under a `neuron_mask` (`bridge.py` `rf_kick`). It
   becomes necessary ONLY if the loop's micro-schedule re-kicks an RF group while a DISJOINT RF group must hold its
   trackers across the kick. **Phase A and Task-2 did NOT need it** (each query's per-block cleanup is sequential; the
   store is in synapses). The K-scale loop reads K blocks via the BATCHED path (fire all triggers → reconstruct in
   parallel → one cleanup) — which is the existing `_read_all_blocks` op, already validated to K=32 for the STORE
   (GAP-A). So the batched read does not re-kick-while-holding. **Predict: still not needed.** IF a K=32 micro-schedule
   surfaces it: the edit is ~6 lines (mask the three tracker writes like `v`/`u` already are), default `None` =
   byte-identical, `test_rf_*` pins bit-identity — **isolated commit for byte-review.** Deferred until demonstrated.
3. **The S5 divisive-norm wire-in — NO `sim/` edit.** The primitive is already in `sim/` (built 2026-06-15 for PPMI);
   the de-risk used it with NO `sim/` edit; #3 flips it on from the runner.

**Summary: reuse-by-import, NO `sim/` edit expected; the one conditional edit (rf_kick tracker mask) is deferred and
flagged for byte-review only if a K=32 multi-op schedule demonstrates it.**

---

## 5. THE GO BAR + THE HONEST-NEGATIVE FORM

### 5.1 The GO bar (the moat is the HARD gate)

#3 is GO when, on the **real `OneBrainComposer`** (the production object, `composer_kind="onebrain"`):
1. **`==host` on the production who/what(/chain) matrix.** The sequencer-driven `query_patient` / `query_agent` /
   `ask_yes_no` / `render_fact` / `query_chain` return the SAME answer as the host `_scan` path on the SAME store, at
   **K∈{2,8,16,32}** + the production V/D (D=128, V to 320). **Multi-seed: 6 seeds** for the noise-sensitive
   match/cleanup cascade (the standing rule); the production 320-scale demo confirm at 3 seeds (the demo precedent).
2. **The no-confab MOAT — the HARD gate — holds: 0 false-accepts.** Every absent/cross cue abstains (the K-way WTA
   selects abstain; the emitted answer is `None`/`unknown`). **A single false-accept at any seed, any K, is a FAIL** —
   the moat is never traded for a pass (per `feedback_moat_not_hard_lossy_memory_ok`: the moat is kept where free, and
   here it is free, so it is not weakened).
3. **The full CI suite passes verbatim with the flag OFF** (byte-unregressed), and the new flag-ON guard asserts the
   who/what + yes-no + describe + moat.

The anti-cheat battery at every stage: sequencer-lesion fails SAFE (cut the S5 drive → abstain, never a wrong block);
store-lesion collapses recall (the synaptic store is load-bearing); permuted-rule INVERTS (the K-way cyclic-shift of
the match→answer map routes a present cue elsewhere); permuted-store carries content; the RAW-no-norm control breaks
the moat (the S5 norm is load-bearing).

### 5.2 The honest-negative form (a clean failure IS the deliverable)

Per the top-level goal (`project_actual_goal_artificial_life_brain_analogue`), an honest negative maps where a
point-neuron loop breaks. The two clean failure modes:
- **NEGATIVE at the K=32 match margin.** If the 1-of-K=32 match cascade cannot match `_scan` at K=32 without a moat
  breach (the worst no-match leak crosses threshold, or the abstain-suppression OR spuriously fires) AND none of the
  retreats (§3.3: divnorm re-tune / NEF-FS pool / hierarchical match) recover it, then **the deliverable is:** the
  residual host op is the cue-match COMPARISON at scale, and the boundary is the spike-SNR of the 1-of-K=32 match — a
  characterized cost (the per-K margin table showing exactly where it squeezes). The production loop then runs the
  sequencer up to the K where it holds (e.g. K≤16) and the host `_scan` above it, with the boundary documented. This
  maps the point-neuron parallel-match ceiling — a biology-translatable insight.
- **NEGATIVE at the S5-at-scale wire-in.** If the divnorm firing-band placement does not transfer to D=128/V=320 and
  the NEF-FS-pool fallback also can't (the low-probability branch the S5 deep-research named), the loop keeps the one
  host DATA read at S5 (the documented Task-1 ceiling) while the CONTROL stays on-substrate — #3's who/what control is
  still retired, with one residual DATA read. (S5 GO at K=2 makes this branch unlikely; flagged for completeness.)

**The moat is NEVER the negotiable axis:** if the only way to pass `==host` is by weakening the moat, #3 is a FAIL, not
a softer GO. The negatives above are about WHERE the K-scale match breaks, not about trading away abstention.

---

## 6. EFFORT HONESTY — days, not weeks

**This is a DAYS arc (≈3–6 focused working days), not a weeks arc.** The reasoning:
- **Every piece is proven + reuse-by-import.** The sequencer (K=2 GO), the loop (Task-2 GO), the S5 norm (GO,
  not-yet-wired), the #1 cleanup (wired), the K-way WTA primitive (the BG selection, validated). #3 is generalizing one
  parameter (K=2→K) + wiring three already-GO pieces together + one boundary sweep — no new mechanism.
- **The only real unknown is the K=32 match margin** (§3) — one per-K sweep (Stage S2). That sweep is a CPU/GPU run,
  not a research arc.
- **NO `sim/` edit expected** (the slow/risky path); the conditional rf_kick edit is deferred and unlikely.

**Stage-by-stage estimate:**
| stage | what | effort |
|---|---|---|
| S0 | generalize the sequencer K=2→K + K-way WTA, CPU parity to K=4 | ~1 day |
| S1 | wire the S5 divnorm into the loop, CPU parity (mergeable with S0) | ~0.5 day |
| S2 | the K∈{2,8,16,32} margin sweep, 6 seeds, CPU→GPU (the boundary test) | ~1–1.5 days (GPU run time + any re-tune) |
| S3 | fold into `OneBrainComposer` (opt-in) + CI guard, GPU | ~1 day |
| S4 | the 320-scale production GO, 3 seeds, GPU | ~0.5–1 day (mostly GPU run time) |

**Total: ~4–5 working days nominal**; the variance is the K=32 margin. **What turns it into >1 week:** a genuine K=32
margin squeeze that needs the NEF-FS-pool S5 fallback (re-tuning the input-norm basin at V=320) or the hierarchical
match (a sequencer re-wire) — both bounded, with the retreat named in §3.3. The honest-negative (the sequencer holds to
K≤16, host `_scan` above) is reachable in the same days budget if the margin walls.

**Ordering recommendation vs the other burndown items:** #3 is the **ONE big remaining conversational arc** (CYCLE 299:
"REMAINING REAL WORK: #3 = the ONE big arc"), it is a days-not-weeks engineering pass on proven mechanisms, and it
retires the single most pervasive conversational host shortcut (the cue-match control flow under 5 production methods +
the chain). It should rank ABOVE the nav reward/value loop (#6–#9, OPEN-RESEARCH, NO-GO closed loop) and the deep
frontier (#12, owner-deferred). It can run in PARALLEL with the cheap default-flips (#5 PPMI-neural code re-gen, #10
nav-decision CLI flip) since those touch disjoint files.

---

## 7. SOURCES (file:line verified)

- **Scope:** `research/runners/one_brain_composer.py` (`_scan`:492; `query_patient`:554–564; `query_agent`:566–567;
  `ask_yes_no`:569–575; `render_fact`:577–592; `query_chain`:594–603; `_find_cued_block`:647–653; `count_facts`:678–681;
  `enable_spiking_cleanup`/`_select`/`_spiking_select`:289–327; `D=128, k_max=32` default:93).
- **The proven mechanism:** `research/runners/_phaseB_onebrain_sequencer_derisk.py` (the K=2 hard-coding: `m0/m1`,
  `ans0/ans1`, `inh0/inh1`:156–159; `blocks_scores[:2]`:279; the 2-way rule dict:298–299; the block-0 priority
  chain:193–204); `research/runners/_phaseC_task2_wholeturn_loop.py` (`LoopComposer`:118; the role-agnostic
  `block_role_scores`:84; the who/what wiring:159–167).
- **Findings:** `2026-06-19-onebrain-sequencer-derisk.md` (Phase B GO, the gated-disinhibition fix + resting-reset
  discipline + the K-margin scope note:116–119), `2026-06-19-phaseC-task2-wholeturn-loop.md` (Task-2 K=2 loop GO + the
  Task-3/4 breakdown:118), `2026-06-19-tier2-phaseC-integrated-loop-design.md` (the seam table + the IN/OUT scope + the
  §5 task breakdown + the `sim/`-edit picture), `2026-06-19-phaseC-task1-S5-seam-derisk.md` (the fixed-projection
  WALL → option b), `2026-06-19-S5-on-bridge-normalization-deep-research.md` (the S5-is-divisive-gain-not-whitening
  call + the ranked options), `2026-06-20-S5-divisive-norm-derisk.md` (S5 CLOSED on-bridge via `input_divisive_norm`,
  GO 3 seeds CPU+GPU, NO `sim/` edit + the follow-on wire-in), `2026-06-20-burndown-1-onebrain-spiking-cleanup.md`
  (the 1-of-V=320 spiking WTA cleanup, the SELECTION-in-proven-range evidence),
  `2026-06-20-shortcut-burndown-inventory.md` (#3 + #4(S5) + #1, the bucket split), `AUTONOMOUS_STATE.md` CYCLE 299
  (#3 = the ONE big arc).
- **Production path:** `research/runners/consolidated_320_conversation_demo.py` (`--composer` default `onebrain`:205;
  D=128, V to 320, k_max=32, the 320-scale 3-seed GO precedent).
- **Reusable primitives:** `sim/regions.py:240` / `sim/config.py:440` / `sim/bridge.py:6048` (`input_divisive_norm`);
  the BG WTA (`g11_bg_runner.py`, `--bg-lateral-inhibition`, catalog A.04); `couple_gate_to_pool` /
  `set_transmission_gate` (`sim/bridge.py`); `tests/test_one_brain_composer_agent.py` /
  `tests/test_onebrain_spiking_cleanup.py` (the CI guards).

_Read-only design deliverable. No code written, no experiments run. Every cited file:line + finding verified against
the source. The K=2 hard-coding of the sequencer and the role-agnosticism of the loop were both confirmed directly in
the code; the S5 closure and the #1 cleanup wire-in were confirmed in their findings + the production composer._
