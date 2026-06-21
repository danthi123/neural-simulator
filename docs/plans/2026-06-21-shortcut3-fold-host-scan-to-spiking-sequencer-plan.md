# Shortcut #3 — fold the spiking K-way sequencer into the production composer, retire the host `_scan` (build plan, 2026-06-21)

**Type:** WIRING build (compose already-de-risked pieces — NOT a new mechanism). The deep-research gate does **NOT**
fire (no boundary to overcome; the K=32 capability + moat are de-risked GO at the production threshold —
`research/findings/2026-06-21-shortcut3-K32-capability-surpass.md`). This doc maps the wiring precisely so a build
subagent can execute it test-first, committing each task.

**Goal:** convert the production `OneBrainComposer`'s **host cue-matching scan** (a Python `for / if / return` loop over
stored fact blocks that decides which stored fact answers a who/what query, and answer-vs-abstain) into a **spiking
decision** — the validated on-bridge K-way sequencer (a gated-disinhibition match cascade + a BG first-match priority
WTA) chooses the matching block in spikes. This closes a brain-based-only host shortcut: the cue-match comparison and
the answer/abstain routing are currently host computation; after the fold they are neurons firing.

**Scope guardrails:**
- **NO `sim/` edit.** The sequencer reuses already-shipped bridge primitives (`couple_gate_to_pool`,
  `_apply_gate_couplings`, `set_transmission_gate`, `cp_external_input_current`, `enable_vectorized_gate_couplings`) and
  is reuse-by-import. The fold is a runner-side addition to `research/runners/one_brain_composer.py`.
- **Default-OFF, byte-identical.** The host `_scan` path stays the default and the test oracle. The spiking sequencer is
  an opt-in flag (`integrated_loop=True`). When OFF, every byte of the current behavior is preserved.
- **NEVER weaken the no-confab moat.** Abstention (`is None` / `"unknown"`) on absent/cross cues is the hard gate; the
  sequencer's `abstain` channel must map to it 0-false-accept.

---

## 1. The host `_scan` to retire (file:line + what it computes)

The host shortcut is **the cue-match-and-route control flow**, NOT the block reconstruction. Crucial distinction: the
per-block spiking reconstruction (`_read_blocks` → `_read_all_blocks`, the FHRR resonate/unbind/cleanup) is **already
on-bridge** and is unchanged by this fold. The host residual is the Python loop that consumes the decoded `{role: word}`
dicts and decides which block answers.

All in `research/runners/one_brain_composer.py`. There are **6 host cue-match sites** (one named `_scan` helper + 5
inlined copies of the same loop). Each computes: iterate `self._read_blocks()` (the decoded blocks, in store order);
return the FIRST block (or its answer field / index) whose cue roles equal the query; else `None` (abstain).

| # | Site (file:line) | Method | What it computes (the host control op) |
|---|---|---|---|
| 1 | `one_brain_composer.py:543` | `_scan(cue, answer_role)` | First block where ALL `cue` roles match → its `answer_role`; else `None`. The canonical helper. |
| 2 | `one_brain_composer.py:609` | `query_patient(agent, action)` | First block where `agent`+`action` match → routes to patient/clause/attributed-patient; else `None`. (Inlined loop — does NOT call `_scan` because it then branches on the kb-stored patient TYPE.) |
| 3 | `one_brain_composer.py:618` | `query_agent(action, patient)` | `_scan({"action","patient"}, "agent")` — the only caller of #1. |
| 4 | `one_brain_composer.py:623` | `ask_yes_no(agent, action, patient)` | First block matching the full SVO → `"yes"`/`"no"` by polarity; else `"unknown"`. |
| 5 | `one_brain_composer.py:635` | `render_fact(agent)` | First block where `agent` matches → renders `agent action patient`; else `None`. |
| 6 | `one_brain_composer.py:698` | `_find_cued_block(agent, action)` | First block where `agent`+`action` match → its index; else `None`. Used by reconsolidation (`update_on_mismatch`). |

`count_facts` (`:729`) is a related but distinct op — it COUNTS matching blocks (not first-match-and-stop). It is a
bookkeeping read, not the routing control flow; leave it on the host read (it does not pick a single answer or abstain).
The plan does NOT retire it.

**The unifying primitive** the 6 sites share: `first_block_where(agent==·, action==·)` (and #1/#4 additionally
constrain `patient` / read `polarity`). The host `_scan`'s semantic is exactly the de-risk's `host_scan_block(c, qa, qx)`
(`_phaseB_onebrain_sequencerK_derisk.py:290`): "the INDEX of the first block whose decoded agent+action match the cue,
or None." That is the function the spiking sequencer's `decision_to_block` reproduces.

**Therefore the fold is a single insertion point:** replace the host `agent+action` cue-match-and-first-match with a
spiking-sequencer call that returns the selected block INDEX (or `None`). The 6 callers then read their downstream field
(patient / agent / polarity / index) from that block on BOTH paths (the kb routing + the decoded `{role:word}` row are
read identically; only WHICH block is selected moves from host to spikes).

---

## 2. The validated spiking sequencer (API + how it competes the K blocks in spikes)

**Files (reuse-by-import):**
- `research/runners/_phaseB_onebrain_sequencerK_derisk.py` — the **S0 K-way sequencer fabric** (builder + run + parity
  helpers). This is the core.
- `research/runners/_phaseB_onebrain_sequencer_derisk.py` — the **composer-side op-result reader** `block_cleanup_scores`
  + `scores_to_drive` (the cleanup membrane → decoded-word-line drive). Imported by S0.
- `research/runners/_phaseB_onebrain_sequencerK_k32_margin_derisk.py` — the **S2 production-scale** runner: the divnorm
  score bridge + `run_sequencerK_with_drive` + the K=32 battery. This is the production op-point (`retreat=divnorm`,
  `gain=0.11`, `sigma=1.0`, `match_thresh=0.06`).
- `research/runners/_phaseC_S5_divnorm_derisk.py` — `build_divnorm_score_bridge`, `onbridge_divnorm_drive` (the
  on-bridge divisive-normalization that lifts K=16→K=32 by dropping the runner-up below rheobase).
- `research/runners/_phaseB_onebrain_sequencerK_divnorm_derisk.py` — `run_sequencerK_with_drive` (the S1 drive runner;
  the decoded-line drive is supplied directly, no host `scores.max()` in the drive path).

### How it competes K blocks in spikes (the mechanism)

For a store of K facts, a spiking Izhikevich subnetwork (all on `cp_connections`, the standard step + transmission
gates drive it):

1. **Per-block word match (gated disinhibition).** Each block `b` has decoded word-lines `d{b}A_w` / `d{b}X_w` (agent /
   action), driven from THAT block's cleanup scores (`scores_to_drive(ag)` / `scores_to_drive(ax)`). A decoded line
   passes to `mw{b}{role}_w` only THROUGH a transmission gate `g{b}{role}_w` that the CUE word-line `cue{role}_w`
   FIRING opens (via `couple_gate_to_pool`, the shipped `_apply_gate_couplings` hook). So `mw{b}{role}_w` fires iff the
   block's decoded word on role == the cue word.
2. **Role OR-pool.** `mA{b}` / `mX{b}` ← OR over `mw{b}{role}_w` (only the cue word's match line can fire it).
3. **Block AND (gated).** `m{b}` ← `mX{b}` THROUGH a gate `gblk{b}` that `mA{b}` firing opens (action-match passes iff
   the agent ALSO matched — a gated conjunction, the robust point-neuron primitive).
4. **BG first-match priority WTA + abstain.** `m{b}` drives `ans{b}`; `ans{b}` → `inh{b}` → inhibits every lower-priority
   `ans{j>b}` AND the tonic `abstain` channel (the canonical BG default-suppression; first-match priority = the host's
   "return the FIRST matching block"). `abstain` is the tonic default channel suppressed by any match (the moat: no match
   → abstain stays lit).
5. **The production rule (the legitimate body read over the spiking result).** Read the K spiking match-pool rates
   `m{0..K-1}`; the LOWEST-index block with `m{b} > match_thresh` answers; none → abstain. `match_thresh = 0.06` (the
   validated production threshold — the no-match floor is exactly 0.000 at K=32, so 0.06 admits the weakest real match
   (0.116) with zero false-accept).

### The API (the functions the fold calls)

```python
# build (once per store size K, on the composer's vocab):
sb, meta = build_sequencerK_bridge(seed, V, K)                 # the spiking control fabric (+ couplings wired)
score_sb  = build_divnorm_score_bridge(seed, V, enable_divnorm=True, sigma=1.0, gain=0.11)  # the divnorm drive source

# per store (recomputed when the store changes):
bscores = [block_cleanup_scores(c, b) for b in range(K)]       # the op RESULT per block (agent_scores, action_scores)
drives, lit = make_block_drives(score_sb, V, bscores, input_gain=1.0, retreat="divnorm", peak_mult=1.0)
# (make_block_drives lives in _phaseB_onebrain_sequencerK_k32_margin_derisk.py — the divnorm-normalized decoded-line drive)

# per query (the spiking decision that replaces the host first-match):
dec, rates = run_sequencerK_with_drive(sb, meta, cue_agent_idx, cue_action_idx, drives, match_thresh=0.06)
block_idx  = decision_to_block(dec, K)                          # the selected block INDEX, or None (abstain)
```

**Inputs:** the cue `(agent_idx, action_idx)` (word indices into the composer's vocab `c.words`) + the K stored blocks'
decoded-line drives (derived from the composer's own on-bridge cleanup `bscores`). **Output:** `decision ∈
{"ans0".."ans{K-1}", "abstain"}` → `decision_to_block` → the selected block index or `None`. This index IS the host
`_scan`'s answer (the same control decision), now produced by spikes.

**Reset discipline (load-bearing):** `reset_sequencerK_state(sb)` runs at the start of every query (called inside
`run_sequencerK_with_drive`) — it drains prior recurrent/delayed activity, zeros the gate-coupling EMAs, closes all
gates, and resets membranes to the resting `c`-reset, so consecutive queries on the SAME persistent bridge don't leak.
This is per-query housekeeping; the match comparison stays entirely in spikes.

---

## 3. The precise fold wiring (opt-in flag + insertion point + default-OFF byte-identical)

### 3a. The opt-in flag (mirrors the existing flag pattern, e.g. `enable_spiking_cleanup` at `:130`)

Add to `OneBrainComposer.__init__`:

```python
def __init__(self, ..., integrated_loop=False, sequencer_match_thresh=0.06,
             sequencer_gain=0.11, sequencer_sigma=1.0, sequencer_input_gain=1.0):
    ...
    # integrated_loop (shortcut #3, default OFF = byte-identical = the host-_scan oracle + numpy-CPU + test-oracle path):
    # make the CUE-MATCH ROUTING fully on-substrate. The per-block reconstruction (_read_blocks) is ALREADY spiking; the
    # residual host op is the Python first-match loop that picks WHICH stored block answers a who/what query (and answer
    # vs abstain). When ON, that loop is replaced by the validated K-way sequencer (gated-disinhibition match cascade +
    # BG first-match priority WTA): the cue + each block's cleanup scores drive a spiking control fabric whose winning
    # channel IS the selected block (the legitimate body read), == host_scan_block multi-seed at match_thresh 0.06
    # (2026-06-21-shortcut3-K32-capability-surpass.md). The no-confab moat is preserved by construction: the abstain
    # channel maps to the same None/"unknown" the host returned, 0 false-accept on absent/cross cues.
    self.integrated_loop = bool(integrated_loop)
    self.sequencer_match_thresh = float(sequencer_match_thresh)
    self.sequencer_gain = float(sequencer_gain)
    self.sequencer_sigma = float(sequencer_sigma)
    self.sequencer_input_gain = float(sequencer_input_gain)
    self._seq = None            # (sb, meta) — the sequencer control bridge, built lazily on first query
    self._seq_score = None      # the divnorm score bridge
    self._seq_K = None          # the store size the current sequencer/drives were built for
    self._seq_drives = None     # the per-block decoded-line drives (recomputed when the store changes)
    self._seq_dirty = True      # the store changed since the drives were built -> rebuild
```

**Byte-identity when OFF:** all new attributes are inert; no sequencer bridge is constructed (lazy). The existing
construction path (`self.comp`, `self.b`, `self.parser`, the layout) is untouched. Mark `self._seq_dirty = True` in
`store` / `_write_block` (the two layout/content-changing ops) so the drives rebuild after a write — but ONLY when
`integrated_loop` is on (guard the rebuild call, not the flag set, so a flag-off composer never pays the cost).

### 3b. The single helper that produces the spiking decision

Add one method that the 6 sites delegate to:

```python
def _seq_block(self, agent, action):
    """The SELECTED block index for cue (agent, action) — the spiking K-way sequencer decision (or None = abstain),
    replacing the host first-match loop. integrated_loop OFF -> the host read (byte-identical). Built lazily; the
    sequencer + drives are (re)built only when the store size changes or a write dirtied them."""
    if not self.integrated_loop:
        # the host path: the EXACT same first-match loop the 6 sites used (read here once so all callers share it)
        for i, got in enumerate(self._read_blocks()):
            if got.get("agent") == agent and got.get("action") == action:
                return i
        return None
    # the spiking path (lazy build; rebuild drives on a dirtied/grown store)
    K = len(self.kb)
    if K == 0:
        return None
    self._ensure_sequencer(K)                  # builds sb/meta/score_sb for K; recomputes drives if dirty
    if agent not in self._word_index or action not in self._word_index:
        return None                            # an absent cue word -> no block -> abstain (the moat)
    dec, _rates = run_sequencerK_with_drive(self._seq[0], self._seq[1],
                                            self._word_index[agent], self._word_index[action],
                                            self._seq_drives, match_thresh=self.sequencer_match_thresh)
    return decision_to_block(dec, K)
```

with `self._word_index = {w: i for i, w in enumerate(self.words)}` cached in `__init__`, and `_ensure_sequencer(K)`
building `build_sequencerK_bridge(self.seed, self.V, K)` + `build_divnorm_score_bridge(...)` + `make_block_drives(...)`
the first time for a given K (or after `_seq_dirty`), caching them on `self`.

### 3c. The insertion at the 6 sites

Each site replaces ONLY its `agent+action` first-match with `idx = self._seq_block(agent, action)`, then reads its
downstream field from `idx`. The decoded `{role:word}` row is still read from `self._read_blocks()[idx]` (the body read
the agent already does), so the patient/polarity/clause logic is unchanged. Concretely:

- **`query_patient`** (`:609`): `idx = self._seq_block(agent, action); if idx is None: return None`; then the existing
  `stored = self.kb[idx][0].get("patient")` + clause/attributed branching, reading `got = self._read_blocks()[idx]` for
  the decoded patient. *(The clause/attribute decode `_decode_clause` / `_attributed_patient` already address by index
  — no change.)*
- **`query_agent`** (`:618`): keep as a thin wrapper, but route through the sequencer on the (action, patient) cue. NOTE:
  the sequencer's match cascade is built for the **(agent, action)** cue. `query_agent` matches on **(action,
  patient)**. Two clean options, pick per the risk note below: (a) keep `query_agent` on the host read when
  `integrated_loop` (it is not the production who/what hot path and is not in the 320 GO battery), documenting it as a
  bounded follow-on; or (b) add a second sequencer instance keyed on (action, patient) lines. **Recommend (a)** — the
  K=32 de-risk validated the (agent, action) cue; do not widen scope.
- **`ask_yes_no`** (`:623`): `idx = self._seq_block(agent, action); if idx is None: return "unknown"`; then check the
  decoded `got = self._read_blocks()[idx]` matches `patient` (full-SVO) and return `"yes"`/`"no"` by polarity. *(The
  sequencer matches agent+action; the patient equality + polarity are the body read over the selected block, identical
  to today.)*
- **`render_fact`** (`:635`): `idx = self._seq_block(agent, None)` — but `render_fact` matches on **agent only**. Same
  cue-arity question as `query_agent`. **Recommend:** route `render_fact` through `_seq_block(agent, action=<the decoded
  action of the first agent-matching block>)` is circular; instead keep `render_fact` selecting via the agent-only host
  read when `integrated_loop`, OR (cleaner) note that `describe` in the 320 GO battery uses `render_fact(agent)` and the
  agent-only match is a 1-role cascade. **Recommend (a)** for the first build: keep agent-only `render_fact`/`describe`
  on the host read, document as a bounded follow-on (the production who/what + yes-no + reason hot paths go fully
  spiking; agent-only describe is the residual). This keeps the build minimal and the gate decisive.
- **`_find_cued_block`** (`:698`): `return self._seq_block(agent, action)` directly (it already returns the index).
  Reconsolidation (`update_on_mismatch`) then routes through the spiking decision for free.
- **`query_chain`** (`:645`): no change — it iterates `query_patient`, which now routes through the sequencer; the moat
  holds at every hop because `query_patient` abstains on a sequencer miss.

**Net:** the production who/what hot path (`query_patient`, `ask_yes_no`, `query_agent`-via-host, `reason_chain`,
`update_on_mismatch`) routes the **(agent, action) cue-match through spikes** under `integrated_loop=True`; agent-only
`describe`/`render_fact` and (action,patient) `query_agent` are documented bounded follow-ons (a 1-role and a swapped-cue
cascade) deliberately left on the host read so the first build's HARD GATE is the cleanly-validated (agent, action)
cascade. **This is an honest scope, not a moat hole** — those follow-ons still ABSTAIN correctly (they keep the host
read, which is the test oracle).

---

## 4. The HARD GATE test design (answer-identity + moat + 320-scale + multi-seed)

A new de-risk runner `research/runners/_phaseB_onebrain_integrated_loop_fold_derisk.py` + a CI test
`tests/test_onebrain_integrated_loop_fold.py`. The gate is: **the spiking-sequencer composer is capability-equivalent to
the host-`_scan` composer on the full who/what matrix INCLUDING every `is None`/`"unknown"` abstention, the moat is
0-false-accept, at production scale (320 concepts, K up to 32), multi-seed, with default-OFF byte-identical.**

### (a) Answer-identity / capability-equivalence

For each seed, build TWO composers on the SAME facts + codes: `c_host = OneBrainComposer(..., integrated_loop=False)`
and `c_seq = OneBrainComposer(..., integrated_loop=True, sequencer_match_thresh=0.06)`. For every query in the battery,
assert `c_seq.<method>(...) == c_host.<method>(...)` (the host is the oracle). Battery (mirrors
`tests/test_one_brain_composer_agent.py` + the K=32 `_build_queries`):
- **present cues:** every stored fact's `(agent, action)` → `query_patient` returns its patient (and reaches the LAST
  block — the scan must traverse all K); `ask_yes_no` full-SVO → `"yes"`/`"no"`.
- **the `is None` / `"unknown"` moat assertions (the hard ones):** an unstored cue → `query_patient(...) is None`;
  an absent-agent / absent-action / cross cue → `is None`; a never-stored SVO → `ask_yes_no(...) == "unknown"`; a
  never-stored reconsolidation cue → `update_on_mismatch(...)["action"] == "abstain"`.
- **reason_chain:** a valid 2-hop chain answers; a broken hop → `is None` (moat at every hop).
- Any **capability delta** (e.g. agent-only `describe` / (action,patient) `query_agent` kept on host per §3c) is
  asserted EXPLICITLY as "host == seq because both use the host read for this op" and DOCUMENTED in the runner output as
  the characterized residual — not a silent gap.

### (b) The moat (HARD — never traded)

`fa_total == 0` across all seeds: NO absent/cross cue selects a block (the sequencer's `abstain` channel must fire, or
the cue word is absent → `None`). This is the K=32 de-risk's `moat_ok` re-asserted at the composer API level. If ANY
seed admits a false-accept at `match_thresh=0.06`, the build is REJECTED (do NOT raise the threshold to mask it; the
threshold is fixed at the validated 0.06 — investigate the wiring). Reuse the de-risk's anti-cheats at the runner level:
sequencer-LESION fails safe (sever the result→op drive → abstain), permuted-rule inverts, the NO-DIVNORM raw control
fails (the divnorm is load-bearing).

### (c) 320-scale

Run the answer-identity + moat gate at **V=320** (the production stream-learned cortex codes, via
`grounded_codes=`), **K=32** (32 facts on a 320-word vocab; the 8 actions each shared by 4 facts — the maximal
shared-action stress the K=32 de-risk used). This is the production tier the `consolidated_320_conversation_demo`
default exercises. GPU (`SIM_BACKEND=cupy`) for the V=320 run; the numpy-CPU path is the smaller smoke.

### (d) Multi-seed

Seeds 42/43/44 minimum for the capability gate; the standing 6-seed rule (42/43/44/100/101/102) for the
final GO claim (the K=32 parallel launcher `_phaseB_onebrain_sequencerK_k32_parallel.py` already fans these out — model
the fold runner on it: run seeds SEQUENTIALLY in the CI test, fan out in the de-risk launcher).

### (e) Default-OFF byte-identical (regression — the existing suite must pass VERBATIM)

`integrated_loop=False` is the default, so the SHIPPED tests must pass unchanged:
- `tests/test_one_brain_composer_agent.py` (11 tests) — the core matrix/moat, negation, describe/reason, clause parity,
  reconsolidation, grounded-codes, multi-turn. ALL must stay GREEN (the default path is byte-untouched).
- `tests/test_consolidated_320_conversation.py` — the production demo.
- The new test additionally asserts an explicit `integrated_loop=False` composer == the pre-fold behavior on a fixed
  battery (a byte-identity guard, mirroring the K=32 `check_off_byte_identical` discipline).

---

## 5. Bite-sized task list (failing-test → minimal-impl → run → commit per task)

**Anti-rest discipline (load-bearing — prior fronts rested and lost work):** run seeds SEQUENTIALLY and **commit after
EACH task** (a green test or a landed finding). Never batch multiple tasks into one commit; never end a turn on a
promise. The next concrete step starts immediately after each commit. Use STRICT narrow `git add <pathspec>` (only the
files the task touched) so concurrent work is not cross-attributed.

1. **Flag + lazy plumbing (no behavior change).** Add `integrated_loop` + the sequencer config/cache attrs +
   `self._word_index` to `__init__`; add the `_seq_block` host-path branch only (spiking branch stubbed to raise
   `NotImplementedError` if reached). Write a test: a flag-OFF composer is byte-identical on a fixed battery, and
   `integrated_loop` defaults False. Run `tests/test_one_brain_composer_agent.py` (must stay green). **Commit:**
   `one_brain_composer.py` + the new test (the OFF-byte-identical guard).
2. **`_ensure_sequencer` + the spiking `_seq_block` branch.** Wire `build_sequencerK_bridge` +
   `build_divnorm_score_bridge` + `make_block_drives` + `run_sequencerK_with_drive` + `decision_to_block` (imports).
   Write a failing test: at K=2, V (small vocab), `integrated_loop=True`, `query_patient` present-cue == host and an
   unstored cue `is None`. Minimal-impl until green (numpy-CPU). **Commit.**
3. **Route `query_patient` + `_find_cued_block` through `_seq_block`.** Failing test: the K=2 + K=4 answer-identity +
   moat battery (`query_patient`, `reason_chain`, `update_on_mismatch` abstain) == host, `fa_total 0`. Impl the
   insertion at `:609` + `:698`. Run. **Commit.**
4. **Route `ask_yes_no` through `_seq_block`.** Failing test: affirmative→`yes`, negated→`no`, unstored→`unknown`,
   multi-seed K=4. Impl `:623`. Run. **Commit.**
5. **The K=32 answer-identity + moat de-risk runner** `_phaseB_onebrain_integrated_loop_fold_derisk.py` (model on
   `_phaseB_onebrain_sequencerK_k32_margin_derisk.py` + its parallel launcher). Two composers per seed (host oracle vs
   `integrated_loop`); the full who/what + moat battery at K=32, D=128; the lesion/permuted/raw anti-cheats; seeds
   42/43/44 first. Run on numpy-CPU (or GPU). Land a findings doc
   `research/findings/2026-06-21-shortcut3-fold-integrated-loop-derisk.md`. **Commit** runner + finding.
6. **320-scale GO (GPU).** Run the de-risk at V=320 (grounded codes) K=32, seeds 42/43/44 (then the 6-seed launcher).
   Assert recall == host, `fa_total 0`. Update the finding with the 320 GO row. **Commit.**
7. **CI guard** `tests/test_onebrain_integrated_loop_fold.py` (sequential seeds, small V for CI speed): the
   answer-identity matrix + the `is None`/`"unknown"` moat + the OFF-byte-identical guard. Run the full conversational
   test suite (`pytest tests/test_one_brain_composer_agent.py tests/test_consolidated_320_conversation.py
   tests/test_onebrain_integrated_loop_fold.py`). **Commit.**
8. **(Conditional, after the gate is GREEN) wire the opt-in through the demos + agent.** Add `--integrated-loop` to
   `consolidated_320_conversation_demo.py` (default OFF — do NOT flip the production default in the same build; a
   default flip is a separate, gated step exactly like the `enable_spiking_cleanup` burndown). Optionally plumb
   `integrated_loop` through `BrainConversationalAgent(composer_kind="onebrain")`'s `OneBrainComposer(...)` construction
   (`brain_conversational_agent.py:189`) as a default-OFF kwarg. **Commit.**

Each task: write the failing test FIRST, implement the minimum to pass, run it, then commit (narrow pathspec). After the
final commit, push BOTH remotes.

---

## 6. Risks / capability-delta flags

- **The `abstain` → `is None`/`"unknown"` mapping is clean (LOW risk).** The sequencer's `decision == "abstain"` →
  `decision_to_block → None`; the 6 callers already treat `None` as their abstain (`return None` / `"unknown"` /
  `{"action": "abstain"}`). The de-risk validated `moat_ok` (0 false-accept) at K=32 0.06. An absent cue WORD is caught
  before the sequencer (`agent not in self._word_index → None`), matching the host read (a never-seen word matches no
  block). **No moat hole.**
- **Cue-arity delta — `query_agent` (action,patient) and agent-only `render_fact`/`describe` (CHARACTERIZED residual,
  MEDIUM scope risk).** The K=32 sequencer is built + validated for the **(agent, action)** cue. `query_agent` matches
  (action, patient); `render_fact`/`describe` match agent-only. §3c recommends keeping these on the host read under
  `integrated_loop` for the first build (they still abstain correctly — the host read is the oracle), and DOCUMENTING
  them as bounded follow-ons (a swapped-cue cascade and a 1-role cascade). This is an honest partial conversion, not a
  silent gap: the production who/what + yes-no + reason hot paths go fully spiking; the gate asserts host==seq on the
  residual ops explicitly. **Flag for the owner: the fold makes the (agent, action) routing spiking; the (action,
  patient) + agent-only routings are a named, validated follow-on.** Widening the sequencer to those cues is a clean
  second build (a second cascade keyed on the other role pair) IF prioritized.
- **Per-query cost / lazy rebuild (LOW-MEDIUM, perf not correctness).** The sequencer adds a spiking settle per query
  (and a per-store drive rebuild). This is the brain-based-only cost (the point of the conversion); the A5 speed levers
  already in `OneBrainComposer` are orthogonal. The de-risk runs on numpy-CPU/GPU; the CI test uses a small V for speed.
  Cache the sequencer + score bridges per K (built once, reused across queries) and rebuild drives only on a dirtied
  store (`_seq_dirty`) — mirror the `enable_csr_cache` discipline. **No correctness risk; flag the latency as the
  expected brain-based cost.**
- **No 320-scale capability delta expected (the de-risk confirms it).** `2026-06-21-shortcut3-K32-capability-surpass.md`
  established K=32 `eq_n 3/3` + `fa_total 0` at V=320-representative scale + `match_thresh 0.06`; the single
  low-fidelity-code row (0.116) clears 0.06 with zero competing block. The 320 GO run (task 6) is the confirmation, not
  an open question.
- **Store reset between queries (LOW, already solved).** `reset_sequencerK_state` (with its `drain_steps`) handles
  consecutive-query leak on the persistent sequencer bridge — the K=8 stale-carryover bug is already fixed in S0. The
  fold reuses it verbatim. **Flag only:** ensure the composer builds ONE persistent sequencer bridge per K and reuses it
  (do not rebuild per query — that would be correct but slow); the reset handles inter-query isolation.

---

## Summary

- **Retire:** the host first-match cue-match loop at 6 sites in `research/runners/one_brain_composer.py` (`_scan:543`,
  `query_patient:609`, `query_agent:618`, `ask_yes_no:623`, `render_fact:635`, `_find_cued_block:698`) — the
  `first_block_where(agent==·, action==·)` control op. (`count_facts` stays — it counts, not routes.)
- **Replace with:** the validated K-way spiking sequencer (`build_sequencerK_bridge` + `build_divnorm_score_bridge` +
  `run_sequencerK_with_drive` + `decision_to_block`, reuse-by-import; the gated-disinhibition match cascade + BG
  first-match priority WTA), via one `_seq_block(agent, action)` helper the 6 sites delegate to.
- **Flag:** `integrated_loop=False` default (byte-identical, host-`_scan` oracle preserved); ON routes (agent, action)
  cue-match through spikes at `match_thresh=0.06`.
- **HARD GATE:** answer-identity (== host on the full who/what matrix INCLUDING every `is None`/`"unknown"`) + moat
  (`fa_total 0`, never traded) + 320-scale (V=320, K=32, GPU) + multi-seed + default-OFF byte-identical (the shipped
  suites pass verbatim).
- **Honest scope:** the (action,patient) `query_agent` + agent-only `render_fact`/`describe` are characterized bounded
  follow-ons (kept on the host read, still abstaining) so the first build's gate is the cleanly-validated (agent,
  action) cascade.
- **Anti-rest:** sequential seeds, commit each task (narrow pathspec), next step immediately after each commit.
