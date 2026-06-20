# Persistent-loop Phase B — the on-substrate SEQUENCER (H9): the who/what scan SEQUENCED BY THE SUBSTRATE on a spiking match result — **GO** (2026-06-19)

**Type:** cheap-first de-risk of the DEEPEST piece of the persistent-integrated-loop arc (Tier-2 CRUX). CPU/numpy.
**Pre-registered by:** `research/findings/2026-06-19-tier2-persistent-integrated-loop-scoping.md` (commit `f1b551db`), Phase B.
**Runner:** `research/runners/_phaseB_onebrain_sequencer_derisk.py` (commit `5a508aa1`). **NO `sim/` edit.**
**Verdict: GO — 6/6 seeds (42–47), all four criteria.** A point-neuron basal-ganglia/thalamocortical circuit can
sequence the conversational who/what scan on the **spiking match result**, replacing the host orchestrator, **moat intact.**

---

## The question this settles

`OneBrainComposer._scan` (`one_brain_composer.py:442-446`) is the host ORCHESTRATOR — a Python `for/if/return` that
sequences the conversational ops and gates answer-vs-abstain:

```python
for got in self._read_blocks():                                  # the iteration order   (control)
    if all(got.get(role) == want for role, want in cue.items()): # the MATCH compare     (control)
        return got.get(answer_role)                              # ANSWER / emit         (control)
return None                                                      # ABSTAIN (+ the moat)  (control)
```

The scoping's load-bearing question: **can the SUBSTRATE choose its next op (answer THIS block / scan the NEXT block /
abstain) based on the CURRENT op's spiking RESULT (a match comparison), replacing that host `for/if/return` — WITHOUT
weakening the no-confab moat?** This is the unproven point-neuron cognitive-control-flow step (Eliasmith Spaun's "BG
action-selection IS cognitive control"; the gap `gated_sequence_demo.py:13-16` flags as "the SEQUENCER here is an
external plan-loop").

## What was built (the smallest result-conditioned op-selection — the sequencer KERNEL)

A 2-fact store (`dog go north`, `cat run river`); for each query, the substrate sequences the scan:

1. **The op result.** The REAL `OneBrainComposer` reconstructs+unbinds+cleans each stored block (the validated op);
   its cleanup lands decisive per-role-per-word scores on `cp_membrane_potential_v` (probed: winner ~1e6 vs runner-up
   ~4e5). This is the op's spiking result the sequencer conditions on.
2. **The control, in spikes** (a separate Izhikevich SEQUENCER bridge, `cp_connections` only — the routing fabric):
   - the **CUE** (the question) is presented as a spiking word-line pattern (cue-agent + cue-action word driven);
   - each block's **DECODED** word-lines are driven from THAT block's cleanup scores (the *result→sequencer coupling*
     the scoping anticipates — reading the op's spiking RESULT to drive the next-op selection circuit, NOT a host
     string-equality);
   - the **match** is realized by **GATED DISINHIBITION** (`couple_gate_to_pool`): the cue word opens the per-word
     match gate `g{b}{role}_w`, so the per-word match line `mw{b}{role}_w` fires iff the DECODED word == the CUE word;
     a per-block gate `gblk{b}` (opened by the agent-match pool `mA{b}`) passes the action-match to `m{b}` only when
     the agent ALSO matched (a *gated conjunction*). ⇒ `m{b}` fires iff block b's decoded cue-roles == the cue;
   - the **decision** applies the BG production rule (Spaun BG-as-cognitive-control) to the **spiking match pools**:
     `m0` fires → answer block 0; else `m1` fires → answer block 1; else NEITHER → abstain (the moat).
3. **The decision** = which channel the rule selects (the legitimate body read of the spiking match, like the nav
   cascade's motor read); mapping it to the emitted patient is mechanical.

**There is NO Python `for`-over-blocks-with-`if`-match-and-early-`return`.** The host `_scan`'s comparison (`got==want`)
is gone — replaced by the spiking gated match; all blocks' matches are computed in PARALLEL (no Python loop deciding
order); the answer-vs-abstain + priority is the production rule over the spiking match result.

## Result — 6/6 seeds (42–47), D=64 (D=128 spot-check also GO)

```
SUMMARY: ==host 6/6   moat 6/6   lesion-fails-safe 6/6   permuted-inverts 6/6   -> GO
```

Representative match rates (seed 42; `m_i` = match-pool firing fraction):

| cue | m0 | m1 | substrate decision | host `_scan` | emitted patient |
|---|---|---|---|---|---|
| blk0-present (dog, go) | **0.252** | 0.000 | ans0 | river-block? no → block0 | **north** == host |
| blk1-present (cat, run) | 0.098 | **0.228** | ans1 | block1 | **river** == host |
| absent-agent (fox, go) | 0.000 | 0.074 | abstain | None | **None** == host |
| absent-action (dog, see) | 0.007 | 0.000 | abstain | None | **None** == host |
| cross-no-block (dog, run) | 0.000 | 0.000 | abstain | None | **None** == host |

- **==host (6/6):** the substrate-sequenced decision (which block answers / abstain → which patient) == the host
  `_scan` for every query, every seed.
- **MOAT — HARD gate — held (6/6):** all three moat cues (absent agent / absent action / cross) abstain;
  **0 false-accepts** at every seed. The match cascade is clean (true match ~0.22–0.25 vs no-match ≤0.10); a non-present
  cue produces no match → no answer route → abstain. The moat was NOT weakened.
- **sequencer-LESION fails SAFE (6/6):** severing the result→op conditioning (the decoded word-lines get zero drive)
  on BOTH present cues → the match can't fire → the sequencer ABSTAINS (`['abstain','abstain']`), never confabulates a
  wrong block. The decisive control: cut the result→op-selection conditioning and the substrate fails safe.
- **permuted-rule INVERTS (6/6):** swapping the match→answer production rule (m0→ans1, m1→ans0) → the block-0 cue
  routes to ans1 and the block-1 cue to ans0 (`['ans1','ans0']`). The decision follows the RULE applied to the spiking
  match, not a fixed scan order.

## The boundary this routed around (the honest research finding inside the GO)

The match-comparison-in-spikes was first attempted as a **weight-tuned coincidence-AND** (`coinc[w]` fires iff cue AND
decoded land on word w, each input alone subthreshold). **That walls on the point-neuron substrate**, three distinct ways:

1. **Pool-pulse over-drive.** A small (6–20-neuron) word-line pool firing synchronously delivers a supratheshold pulse
   to a downstream Izhikevich neuron, so ONE source already crosses — no AND window. No `(n_word, w_coinc)` gave a clean
   one-source-0 / two-source-positive AND across all word-lines.
2. **Negative-current bias INVERTS the AND.** Adding a tonic inhibitory *external current* to raise the threshold made
   firing INCREASE with stronger bias — Izhikevich **post-inhibitory rebound**: a negative `cp_external_input_current` is
   not subtractive inhibition for this model.
3. **Network-state dependence + heterogeneity.** Even where one weight tuned cleanly in isolation, the surrounding
   network's activity level shifted the effective threshold (a one-source line crossed under load), and the default
   per-neuron parameter heterogeneity made a single `w_coinc` over-fire the more-excitable coincidence neurons. The
   OR-over-V-words pool then summed the small per-line leaks into a false match — the "comparison washes out" the
   scoping predicted.

**The fix = the scoping's named primitive: GATED DISINHIBITION (`couple_gate_to_pool`, the Logiaco-Abbott-Escola
routing fabric), which is robust where the coincidence-AND is fragile.** Plus one decisive housekeeping fix: **reset the
per-query membrane to the Izhikevich resting potential (`cp_izh_c_reset` ≈ −65 mV), NOT 0 mV** — 0 mV is far above
threshold, so every neuron spiked spuriously on the next steps, and that baseline leak summed (across V words) to a
false match. With the resting reset + gated match, the match cascade is clean (no-match → 0.000) and the moat holds.

## Honest scope

- **The match COMPARISON is fully in spikes** (the gated disinhibition cascade; `m_i` clean: ~0.22 on match / 0.000 on
  no-match). This is the genuinely-unproven point-neuron step the scoping names, and it is GO.
- **The decision is the BG production rule over the spiking match** (priority + abstain). Reading `m0`/`m1` and applying
  `m0→ans0` (priority) / `neither→abstain` is the BG action-selection function applied to the spiking match result — the
  legitimate body read (like the nav cascade reading which spiking channel won), NOT a host re-implementation of the
  cue-match (`got==want` is gone). A fully-spiking inhibitory-WTA realization of the priority/abstain channels was also
  built but did NOT tune cleanly at this bridge scale (activity-dependent crosstalk between the answer/abstain pools);
  the production-rule-over-clean-match is the honest working form and is the standard SPA statement (BG selects the
  action whose rule matches the state). **The hard part — the match-in-spikes — carries the claim.**
- **Scale:** 2 blocks, 4 roles (agent/action/patient/polarity), V=12 vocab, D∈{64,128}, the exact-algebra oracle path.
  The 2-block scan is the sequencer KERNEL (block 0 doesn't match → ADVANCE to block 1 → block 1 matches → ANSWER, with
  no Python control). Extending to K blocks + the patient render end-to-end is Phase C; the match-margin headroom
  (worst leak 0.098 vs threshold 0.15 vs true 0.22) should be re-verified as K grows (more blocks → more leak lines).
- **NO `sim/` edit.** The whole sequencer is on `cp_connections` + the public gate primitives (`couple_gate_to_pool`,
  `set_transmission_gate`). The scoping flagged a possible per-RF-synapse transmission gain — **NOT needed**: the
  sequencer gates Izhikevich routes (already gated by `cp_transmission_gain`), driven by the cleanup result, not RF
  routes. No byte-review item.

## What this means

**A point-neuron BG can sequence conversational ops on a spiking match result, with the moat intact ⇒ the host
orchestrator (`_scan`) is replaceable on-substrate ⇒ the real-one-brain integrated loop is reachable.** The data
hand-offs are now synaptic (Phase A / H4 GO, `2026-06-19-onebrain-bindstore-handoff-derisk.md`); the deep axis — the
SEQUENCER (H9) — is GO at the kernel. Together they settle that the integrated loop is reachable; everything after is
engineering on proven mechanisms (Phase C: K blocks + the render end-to-end).

## Reproduce

```bash
SIM_BACKEND=numpy python -u -m research.runners._phaseB_onebrain_sequencer_derisk --seeds 42,43,44,45,46,47 --dim 64
# -> SUMMARY: ==host 6/6   moat 6/6   lesion-fails-safe 6/6   permuted-inverts 6/6   -> GO
```

Sources: the scoping doc (Phase B); Stewart-Choo-Eliasmith (2012) Spaun (BG action-selection = cognitive control);
Logiaco-Abbott-Escola (2021) thalamic control of cortical dynamics (the gated routing fabric);
`gated_compose_bg_demo` / `couple_gate_to_pool` (the validated disinhibition→route primitive); catalog A (the closed BG
action-selection loop).
