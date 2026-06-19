# DA salience-gate -> conversational composer: PRODUCTION WIRE-UP — **GO**

**Date:** 2026-06-18
**Type:** production wire-up (reuse-by-import packaging of a de-risked mechanism). Verdict: **GO** (wire-up smoke PASS,
both GPU acceptance suites green, zero `sim/` edits).
**Direction:** TRUE-ONE-BRAIN roadmap **#6** — the SHARED spiking-SNc dopamine (the same DA the BG actor / limbic core
learns from) now MODULATES the deployed merged conversational agent's recall PRECISION. The de-risk
(`2026-06-18-DA-composer-precision-derisk-GO.md`, commit `b76959be`, GO 6/6) is now wired into production behind an
opt-in flag.

---

## TL;DR for the controller

- **Wired into `MergedNavConvAgent`** (`research/runners/nav_conv_merged_bridge.py`) — the deployed merged
  conversational path AND the only agent that owns BOTH the limbic core (the shared `dopamine` SNc modulator on the
  merged bridge, via `co_resident_limbic` / `co_resident_nav_critic` / `co_resident_td_cueshift`) AND the composer.
- **Flag: `enable_da_salience_gate: bool = False`** (+ the de-risk's validated knobs `da_gate_g0=0.06`,
  `da_gate_k=2.0`, `da_gate_cap=0.25`). **Default OFF = byte-identical** to the current agent (the conversational read
  path is unchanged).
- **When ON:** before each conversational READ op (`what_does` / `who_does` / `is_it_true` / `describe`), the agent
  reads the shared spiking dopamine off ITS OWN merged bridge
  (`self._merged_bridge.neuromodulator_manager.get_concentration("dopamine")`), maps it CLAMPED-TO-SHARPEN
  (`g_eff = clip(g0, g_cap, g0 + k*(DA - DA_baseline))`, the de-risk's `da_to_gate` **reused verbatim**) onto the
  composer's cue-role confidence gate, and ABSTAINS on a noise-dominated cue read
  (`min(margin(agent), margin(action)) < g_eff`, the EXACT margin the de-risk gates on, via the imported
  `OneBrainComposer._margin`). A higher gate => STRICTER abstention, so this can ONLY TIGHTEN the no-confab moat,
  never loosen it (**moat-safe by construction**).
- **NO `sim/` edit** (confirmed `git diff --stat -- sim/` empty). NO composer edit either — the gate reuses the
  composer's own cleanup primitives (`RFPhasorComposer._unbind_phases` + the matched filter == `_cleanup`'s cosine
  scores). The whole packaging is +92/-4 in the one runner.

---

## What was wired (the packaging)

`MergedNavConvAgent.__init__` gains `enable_da_salience_gate` (+ `da_gate_g0`/`da_gate_k`/`da_gate_cap`). Three small
helpers + a gate at the four read ops:

- **`_da_confidence_gate()`** — reads DA off the merged bridge's `neuromodulator_manager` (SAFE: returns `g0` if no
  manager / no `dopamine` modulator → no-op) and maps it via the de-risk's `da_to_gate` (imported from
  `research/runners/_da_composer_salience_cleanup_derisk.py`). At DA baseline `g_eff = g0` (the no-modulation floor);
  only a salient/high-DA turn raises it (capped at `g_cap` = the inverted-U ceiling).
- **`_gated_out(match_fn, g_eff)`** — the de-risk's gate on THIS agent's composer reads: scans `composer._iter_facts()`
  (the same cue-matching scan `query_*` uses), and for the FIRST block matching the cue returns
  `min(margin(agent_scores), margin(action_scores)) < g_eff`. Short-circuits to `False` (do NOT gate) when
  `g_eff <= g0` (the floor → BYTE-IDENTICAL read path) and when no block matches (the composer abstains anyway → the
  moat is unchanged).
- **`_role_cleanup_scores(composite, role)`** — the composer's matched-filter cleanup scores for a role (identical, up
  to the argmax-irrelevant `/D`, to `RFPhasorComposer._cleanup`'s `cos(rec - concepts[w])`), rectified (the NEF
  off-target-zero convention) so the margin is the de-risk's `(peak - runner_up)/peak`.
- The four read ops gate when `enable_da_salience_gate` is on: `what_does`/`who_does`/`describe` → `None`,
  `is_it_true` → `"unknown"` (abstention), each only on a noise-dominated cue read; otherwise delegate to the
  composer unchanged.

**Why the agent layer, not `composer.confidence_gate`:** `MergedNavConvAgent`'s composer is `RFPhasorComposer` /
`MergedRFComposer` (the slice-bound RF composer), which has no `confidence_gate` attribute (that lives only on
`OneBrainComposer`). To keep the wire-up reuse-by-import with NO composer/`sim` edit, the gate is applied at the agent
layer using the composer's OWN cleanup primitives + the de-risk's exact `_margin`/`da_to_gate` — a faithful production
realization of the de-risk's `confidence_gate` mechanism (the blanked-block-abstains behavior is reproduced: a gated
cue read returns the abstention answer).

---

## Validation (smoke, GPU)

`research/runners/_da_salience_gate_wireup_smoke.py` (`SIM_BACKEND=cupy`). This is a SMOKE (the 6-seed precision
numbers are the committed de-risk); it confirms the wire-up works + the moat holds.

**(1) Default-OFF byte-identity — PASS.** A vanilla `MergedNavConvAgent(seed=42)` (no gate, no limbic) reproduces the
conversational matrix + the no-confab moat EXACTLY: `what_does('dog','go')=='north'`, `what_does('cat','come')=='south'`,
`who_does('go','north')=='dog'`, `describe('dog')=='dog go north'`, and the three abstentions
(`what_does('river','look') is None`, `describe('river') is None`, `is_it_true('apple','stop','west')=='unknown'`).

**(2) ON + a co-resident limbic core — PASS.** With `co_resident_limbic=True, enable_da_salience_gate=True`, the gate
reads the shared `dopamine` off the merged bridge; the limbic SNc is driven to two operating points (the de-risk's
drive-SNc-and-read recipe, here on the merged `limbic_snc`):

| DA level | limbic_snc | DA | g_eff | recall_ok | moat_ok (0 false-accepts) |
|----------|-----------|------|-------|-----------|---------------------------|
| DA_low (tonic, 80 pA)   | 0 Hz   | 0.500 | 0.060 (floor) | True | True |
| DA_high (salient, 600 pA) | 359 Hz | 0.843 | 0.250 (cap)   | True | True |

- **Gate raises with DA:** `g_low=0.060 -> g_high=0.250` (monotone with the spiking dopamine — a salient turn raises
  the gate).
- **No-confab moat held at BOTH DA levels:** 0 false-accepts at DA_low AND DA_high (the three abstentions hold).
- **Clean recall preserved:** the decisive canonical facts are NOT over-abstained at DA_high (their cue-role margins,
  ~0.666/0.834, are >> the cap 0.25), so the gate sharpens the marginal tail without going silent on confident reads.

**Gate-logic unit checks (numpy, CPU):** the gate is monotone in `g_eff` — for a clean stored fact (cue-role
min-margin 0.666), `gated_out` is `False` at any `g_eff <= 0.666` and `True` at any `g_eff > 0.666`. ⇒ a higher
DA-driven gate ⇒ stricter abstention ⇒ moat-safe by construction, and the floor (`g_eff == g0`) is a no-op.

**Regression (GPU, verbatim):** `tests/test_nav_conv_merged_agent.py` 8/8 + `tests/test_nav_conv_step2b_coresident.py`
7/7 PASS — the default-OFF read path is byte-unregressed (the merged-agent acceptance gate, incl. the three `is None`
no-confab assertions, and the step-2b co-resident composer path).

---

## Discipline / scope

- **Reuse-by-import, NO `sim/` edit** (`git diff --stat -- sim/` empty), NO composer edit. Default-OFF byte-identical.
- **The no-confab moat is never weakened** — DA can ONLY raise the gate (clamped below at `g0`), which can only
  tighten abstention (moat-safe by construction + held empirically at both DA levels).
- **Is:** the read-side "one self" hook (Option A) wired into production behind `enable_da_salience_gate` — the shared
  spiking dopamine now shapes how the deployed merged conversational agent recalls. **Isn't:** a new precision
  measurement (the 6-seed numbers are the committed de-risk); the ENCODING hook (Option B, novelty-gated write /
  reconsolidation labilization — a ranked follow-on with a two-directional moat risk); the deep RF-dynamics `sim/`
  edit (a continuous DA-modulated resonate gain — deferred).
- **Activation:** the hook is only meaningful when a `dopamine` modulator is co-resident (a limbic / nav-critic / TD
  slice on the merged bridge); without one DA reads as baseline → `g_eff = g0` → the gate is a no-op (so turning the
  flag on without a limbic core is harmless, not a config error).

## Files

- `research/runners/nav_conv_merged_bridge.py` — `MergedNavConvAgent` gains `enable_da_salience_gate` (+ knobs) + the
  three gate helpers + the gate at the four read ops (+92/-4).
- `research/runners/_da_salience_gate_wireup_smoke.py` — the GPU wire-up smoke (default-OFF byte-identity + ON
  moat-safe at two DA levels).
- `research/findings/2026-06-18-DA-salience-gate-production-wireup-GO.md` — this doc.
