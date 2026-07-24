# DR-1 (new-direction Phase-0): the no-confab moat INVERTED into honest CURIOSITY — 6-seed GO (2026-07-23)

The owner's headline point ("don't refuse when unsure — crave knowledge + growth, seek to learn") realized:
the SAME uncertainty signal that drives the no-confab abstention becomes a CURIOSITY drive that ASKS + LEARNS,
kept honest by a learning-progress reward. Reuse-by-import (the Bogacz-Brown `RealAntiHebbianFamiliarity` gate +
the RPE/value machinery). **`sim/` edit scope (CORRECTED — see Adversarial verification below):** the numpy cheap-first
probe makes **no `sim/` edit** (it never imports the edited code, so its JSON numbers are untainted); the SEPARATE
on-bridge realization (`_curiosity_seek_learn_onbridge_derisk.py`, owned by the on-bridge subagent) makes **ONE
additive, default-off `sim/` edit** — the `from_novelty` production rule in `sim/neuromodulators.py` + the
`current_novelty_signal`/`novelty_baseline` fields in `sim/config.py` (default 0.0 → byte-identical when no
`from_novelty` modulator is registered, which none is). The original blanket "NO `sim/` edit (verified)" is therefore
CORRECTED to that scoped form. Runner `_curiosity_seek_learn_cheap_first_probe.py`.

## Mechanism
The familiarity gate supplies the epistemic gap g=novelty(x) (~1 novel / ~0 learned) → a curiosity modulator; when
NOVEL the policy ASKS a teacher, INGESTS the answer (imprint → raises familiarity → lowers future novelty), and the
intrinsic REWARD = LEARNING PROGRESS (g_before − g_after), TD-tracked as a per-concept expected-LP value. On-bridge
this fills the reserved `from_novelty` neuromodulator rule (now built additively + default-off by the on-bridge subagent
in `sim/neuromodulators.py`); the CPU probe here proxies it at rate level (importing no `sim/` module).

## 6-seed result — GO 6/6 (seeds 42/43/44/100/101/102)
- **Curiosity drives asking (load-bearing):** corr(gap, modulator) **+0.99**; post-answer confidence rises **+0.57**
  above the 0.03-0.04 abstain floor. Reward IS learning-progress: LP(learn) ≈ +0.21 vs LP(noise) ≈ +0.003.
- **Noise is NEVER spuriously learned AND is VETOED (load-bearing honesty guard):** for un-learnable "noisy-TV"
  concepts the gate novelty stays HIGH (g ≈ 0.96) — they are never learned — AND their learned value falls below the
  veto floor (noisy expected-LP ≈ 0.033 ≤ 0.05), so the policy stops paying to ask them; real spends fewer noisy asks
  (~8/30) than yoked (up to 15). *(The temporal "late-rate ≪ early-rate" decay originally sold as the "DECISIVE"
  anti-cheat is a BUDGET artifact — `ASK_BUDGET=30` is spent in turns 0-29, so the "late" bin 146-219 is never entered
  and `noisy_late_rate ≡ 0` in every mode; it is NON-load-bearing. The honesty claim rests on the never-learned g +
  the ELP-veto + the real-vs-yoked noisy-ask contrast above.)*
- **Controls collapse (6/6):** lesion (modulator=0 → 0 asks); yoked-random control (masters 3-7/8 vs real 8/8, robust
  over ask-budgets 26-40 — NB the yoke moves TWO variables, the targeting drive AND the reward, so the collapse is real
  but not cleanly reward-isolated); permuted-gap (corr → 0, masters 5-6/8); moat-by-construction (confident set ⊆
  ingested set, every seed).
- *(NON-load-bearing gate, dropped: `gate_b` "ask-rate unknown ≫ known" ≈ 8.3e7 passes BY CONSTRUCTION — the candidate
  filter hard-requires novelty, so `rate_known ≡ 0` in every run; it restates the moat, it is not an emergent preference.)*

## Rigor
Two real modeling bugs found + fixed in the smoke: (1) at small D the gate's span fills so noise becomes spuriously
learnable → D=1024 keeps noise genuinely novel; (2) `OBS_NOISE·√D` jitter swamps the unit code at large D → replaced
with a dimension-independent unit-direction jitter.

## Adversarial verification (2026-07-23): QUALIFIED — core mechanism holds, one build claim + two GO-gates corrected
An independent 2-skeptic refutation pass could NOT break the core dissociation: every control is genuinely invoked
per-seed and genuinely collapses (lesion → 0 asks; permuted corr −0.18…+0.08 vs real +0.99; yoked masters 3-7 vs 8/8;
noise never learned g ≈ 0.96, vetoed ELP 0.20→0.033); blind seeds 100/101/102 all ran and pass (no dev-seed selection);
no forbidden fixed-random-code control (D=1024 = large space). Three packaging defects corrected (numbers unaffected):
1. **Build claim** — the original blanket "NO `sim/` edit (verified)" was FALSE (the on-bridge realization edits
   `sim/config.py` + `sim/neuromodulators.py`, additive/default-off, owned + tested by the on-bridge subagent via
   `tests/test_from_novelty_curiosity.py`). Corrected to the scoped form in the header: the numpy probe is genuinely
   sim-free; the on-bridge `from_novelty` is the one additive default-off edit.
2. **`gate_b`** (ask-ratio unknown/known ≥ 2×) is DEGENERATE / non-load-bearing — `rate_known ≡ 0` by construction
   (the candidate filter hard-requires novelty), so the ratio ≈ 8.3e7 in ALL runs; dropped as load-bearing.
3. **The docstring's "DECISIVE anti-cheat"** (noisy late-rate ≪ early-rate) is a BUDGET artifact — the budget is spent
   in turns 0-29, the "late" bin (146-219) is never entered, so `noisy_late_rate ≡ 0` in every mode; relabelled
   non-load-bearing. The honest noisy-honesty evidence is `noisy_g ≈ 0.96` + the ELP-veto + the real-vs-yoked contrast.
Neither degenerate gate falsely flips the verdict (the real gates — gate_a corr +0.99, gate_c conf-rise, noisy_g,
ELP-veto, lesion/yoked/permuted collapse, moat — all hold), so **the 6-seed GO stands on its load-bearing gates.**

## Net
The moat's uncertainty signal becomes a curiosity drive that seeks a teacher + learns — and the learning-progress
reward keeps it HONEST (it stops chasing noise + never confabulates). This is the biological resolution of the
owner's "don't refuse, grow instead." Phase-0 P0.2. Follow-on: on-bridge (`from_novelty` + spiking-SNc RPE + A→W
question) + wire into the develop-loop teacher hook (P2.1). (`sim/` edit scope: the numpy probe is sim-free; the
on-bridge realization adds ONE additive default-off `from_novelty` rule — see Adversarial verification above.)
