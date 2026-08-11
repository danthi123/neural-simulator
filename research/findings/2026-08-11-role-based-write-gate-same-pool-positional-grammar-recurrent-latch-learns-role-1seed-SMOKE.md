---
type: finding
status: smoke
date: 2026-08-11
mechanism: ROLE-BASED (syntactic, not token-class) write-gate — a reward-driven write-gate with a RECURRENT onset-latch ("have I loaded the controller yet") gates the role-defined subject by POSITION on a same-pool positional grammar where token identity does NOT reveal the subject
lane: emergence engine / working memory (the precisely-named residual of the variable-binding WM GO)
verdict: 1-SEED SMOKE. (1) The MEMORY composition holds on the HARDER same-pool positional stream (marker scaffold held-out 1.000, all anti-cheat teeth bite). (2) The reward-driven RECURRENT-latch gate LEARNED genuine syntactic ROLE at L=3 (held-out 1.000, fires pos0 1.00/pos>0 0.00, token-identity gap +1.00 — PASSES the crux; NOT an identity artifact, subjects & distractors share the pool) — but it is HIGH-VARIANCE (same seed at L=2 collapsed to fire-everything, gap +0.00). The EXISTING code-only identity gate robustly FALLS BACK to identity (gap +0.00, fails the crux). Role induction from reward alone is POSSIBLE but not yet RELIABLE; the 6-seed sweep quantifies the seed-fraction.
seeds: [42]
runner: research/runners/_var_bind_role_gate_derisk.py
artifacts:
  - research/findings/raw/_var_bind_role_gate/role_gate_6seed.json
  - research/findings/raw/_var_bind_role_gate/role_gate.json
instrument: reuse-by-import of the banked SpikingSlot (D3 slow-NMDA bistable HOLD, `_var_bind_gated_slot_derisk.SpikingSlot`) + the RUNG6c sparse barcodes (`_novel_referent_hebbian_fastweight_derisk._mint_codes`) + the EMERGE-14 on-bridge HTM engine and the n-gram HELD-OUT floor (`_emerge14_stageC_onbridge_learning_derisk`, `_emerge_stream_language_derisk`). The write CONTENT is the gated token's agreement feature (a host lexicon scaffold, exactly as verb_of was) — the GATE is the object of study. SIM_BACKEND=numpy device=cpu; NO sim/ edit.
---
<!--derived-->
## ⚠️ 6-SEED CORRECTION (coordinator-run, `research/findings/raw/_var_bind_role_gate/role_gate_6seed.json`) — the 1-seed existence-proof does NOT hold; role-gate is an HONEST NEGATIVE, and the residual is CREDIT ASSIGNMENT (deeper than the signal)

The 1-seed smoke below (recurrent latch 0.857, gap +0.80) was the OPTIMISTIC TAIL. At 6 seeds:
- **The MEMORY composition is still a GO** on the harder same-pool positional stream: MARKER-ROLE (gate timing given)
  held-out **1.000** >> HTM 0.000, and it is POSITIONAL (permuted-position 0.259 ≈ chance, recency-gate 0.250 ≈ chance),
  all teeth biting (lesion 0.046, always-open 0.225, feat-scramble 0.000, referent-shuffle 0.000). The memory is NOT the
  residual.
- **The reward-driven RECURRENT role-gate is an HONEST NEGATIVE:** held-out **0.602** (chance 0.250), token-identity gap
  **+0.45** (over 12 matched nouns) — it gates by position SOMEWHAT (positive gap, above the identity gate's +0.00) but
  does NOT cleanly induce syntactic role. The identity-only gate fails the crux (gap +0.00, gates by token class).
- **The decisive finding — the residual is the CREDIT ASSIGNMENT, not the positional signal:** a HOST POS-ORACLE (raw
  position fed IN, only the sign learned) STILL reached only acc 0.265 / gap +0.01. So even when position is HANDED to
  it, reward-over-token-statistics with plain REINFORCE cannot cleanly induce role. Only the hand-wired MARKER drives
  the WM to 1.000.
- **Named next mechanism (now precise + connected):** role induction needs a POSITIONAL/STRUCTURAL substrate signal
  (a spiking ordinal/phase code or a recurrent controller-seen population) **AND better CREDIT ASSIGNMENT** — the
  gap#4 three-factor DA-gated distal-credit machinery we surpassed on spikes at de-risk level — NOT reward over token
  statistics. This directly links the role-gate residual to the gap#4 deep-credit lane.


# Role-based write-gate on a same-pool positional grammar — the reward-driven recurrent onset-latch learns syntactic ROLE (position, not identity), but it is high-variance; the code-only identity gate falls back to identity (1-seed SMOKE)

The variable-binding working-memory GO
(`2026-08-11-variable-binding-working-memory-gated-slot-surpasses-HTM-heldout-1.000-vs-0.000-6seed-GO.md`) left one
precisely-named residual: its reward-driven write-gate learned to fire on the subject with held-out precision/recall
1.00 — but on a stream where the subject is a BARCODE-SEPARABLE CLASS (linearly separable from fillers), so it learned a
token-CLASS boundary, NOT syntactic ROLE. In real language the SAME token is subject-or-not by POSITION/SYNTAX. This
de-risk builds the harder ROLE stream and asks whether a reward-driven gate can induce role from POSITION/STRUCTURE.

## The harder stream — the subject is NOT a token class

ONE shared noun pool of N nouns; EVERY noun appears as BOTH the subject (position 0, the agreement-controlling slot) AND
as an intervening DISTRACTOR (positions 1..L), drawn i.i.d. from the SAME pool. Each noun carries an agreement FEATURE in
{0..F-1} (its number/gender class); the verb agrees with the SUBJECT's feature. Which token is the subject is fixed by
POSITION, not identity, and the distractors' features are LURES. The write content (the gated token's feature) is the
agreement lexicon — a host scaffold, exactly as `verb_of` was; the GATE is the object of study. Reuse-by-import; NO sim/
edit. The runner path is `research/runners/_var_bind_role_gate_derisk.py`.

## The memory composition holds on the harder stream (marker scaffold; all teeth bite)

<!--derived-->
At N=12, F=4 (chance 0.250), L=3 (distance 4, 1728 paths, held-out NOVEL distractors), 1 seed: the MARKER-ROLE spiking WM
(gate timing given = position 0) scores held-out branch(verb) **1.000**, versus the HTM emergence engine **0.000** (its
memorise-not-generalise baseline, reproduced in-runner on the identical harder stream), the best fixed-order n-gram
HELD-OUT floor 0.261, and chance 0.250. Every anti-cheat tooth bites: LESION-the-hold **0.000** (the slow-NMDA slot is
load-bearing; hold-alive 0.1086 with external input ASSERTED zero across the span), ALWAYS-OPEN gate **0.167** (the last
distractor's feature overwrites — a recency lure the gate must protect against), FEATURE-SCRAMBLE **0.000** (the
pool->feature deref/bind is load-bearing), REFERENT-SHUFFLE **0.000** (no topic->answer leakage), PERMUTED-POSITION
**0.200** and RECENCY-gate **0.167** (both ~ chance — the task and the gate are POSITIONAL: destroy the position->role
mapping, or fire a fixed-but-wrong position, and it collapses). Artifact:
`research/findings/raw/_var_bind_role_gate/role_gate.json`. This confirms the MEMORY is NOT the residual on
the harder stream; the LEARNED role gate is.

## The role-gate result — the recurrent onset-latch learns ROLE, but high-variance; identity falls back to identity

<!--derived-->
The CRUX tooth is the TOKEN-IDENTITY control: over held-out nouns appearing at BOTH position 0 (subject) and position >0
(distractor), does the gate fire DIFFERENTLY by POSITION? token_identity_gap = mean(fire-rate@pos0 - fire-rate@pos>0). A
gate firing on token IDENTITY has gap 0 (its decision depends only on the code); a ROLE gate has gap ~ 1.

- **RECURRENT-latch gate (the candidate — code + a self-generated onset-latch state, REINFORCE on the verb-prediction
  reward, NO role/position label):** at L=3 it LEARNED genuine role: held-out **1.000** (chance 0.250), fires pos0
  **1.00** / pos>0 **0.00**, token-identity gap **+1.00** over 11 matched nouns — it LOADs the SAME nouns at position 0
  and IGNORES them as distractors. This is NOT an identity artifact: subjects and distractors are the SAME pool, so the
  subject is not separable by identity. But it is HIGH-VARIANCE: the SAME seed at L=2 collapsed to fire-everything
  (held-out 0.367, fires pos0 0.93 / pos>0 0.88, gap **+0.00**). Larger L gives a stronger anti-recency gradient (firing
  any of more distractors is more likely to overwrite with a wrong feature), which explains the L-dependence.
- **IDENTITY gate (the EXISTING mechanism — code-only, p_load(code), no position):** robustly FALLS BACK to identity —
  gap **+0.00** at both L (its decision is code-deterministic, identical at both positions), held-out 0.167 (L=3) /
  0.533 (L=2), FAILING the token-identity control. This reproduces the finding's residual on the harder stream: a
  token-class gate does NOT transfer to a same-pool positional grammar.
- **POS-ORACLE gate (a host positional oracle — raw normalised position fed in, sign LEARNED):** did NOT cleanly induce
  role — held-out 0.133 (L=3), gap +0.00, sometimes firing MORE at pos>0. The raw-position scalar is a WORSE inductive
  bias than the self-generated recurrent latch.

## Scope / honesty — what is de-risked, what is the residual (brain-based-only)

<!--derived-->
NO-EXTERNAL-NEEDED: grounded in our OWN verified components (D3 hold, RUNG6c barcodes, EMERGE-14 HTM/stream floors); this
is a method-negative-turned-partial-positive (existence proof of role induction), not a capability wall.

- **DE-RISKED:** the residual named by the prior GO. (1) The memory composition solves the ROLE-defined task (marker
  scaffold, all teeth bite) on a stream where the subject is NOT a token class. (2) The reward-driven RECURRENT onset-latch
  is a MECHANISM that CAN induce syntactic role (position, not identity) from the verb-prediction reward alone, with NO
  role label — an existence proof (L=3: held-out 1.000, token-identity gap +1.00). (3) The existing code-only gate
  robustly fails role, confirming the residual is real.
- **THE RESIDUAL, precisely named:** RELIABILITY. Plain REINFORCE with a self-generated onset-latch learns role but is
  high-variance across L/episodes/seeds (existence, not reliability). The 6-seed sweep quantifies the seed-fraction. The
  recurrent latch + REINFORCE math are HOST; their on-substrate spiking realisations — a recurrent "controller-seen"
  population that provides the ordinal/structural signal, and three-factor DA-gated plasticity for the update — are the
  named next rungs (gap#4 distal-credit territory). If the 6-seed sweep shows role is NOT reliable from reward alone, the
  honest next mechanism is a POSITIONAL/STRUCTURAL substrate signal (a spiking ordinal/phase code), supervised syntax, or
  the emergence-engine's own sequence code — NOT reward over token statistics.
- **Named next build (dependency-ordered):** (a) the 6-seed sweep (the exact command below) to quantify role reliability;
  then (b) the spiking recurrent "controller-seen" ordinal population that drives the gate; then (c) the on-substrate
  three-factor DA-gated write-gate; then (d) wire into the emergence stream. Reuse-by-import; NO sim/ edit here.

## Reproduce

<!--derived-->
1-seed smoke (FOREGROUND):
`SIM_BACKEND=numpy python -m research.runners._var_bind_role_gate_derisk --seeds 42 --distances 2 3 --n-test 30 --learned-episodes 15`

The decisive 6-seed sweep (evaluated at the largest distance); point `--out` at a `role_gate_6seed.json` inside the same
`_var_bind_role_gate` raw directory as the smoke artifact:
`SIM_BACKEND=numpy python -m research.runners._var_bind_role_gate_derisk --seeds 42 43 44 100 101 102 --distances 2 3 4 --n-test 90 --out role_gate_6seed.json`
