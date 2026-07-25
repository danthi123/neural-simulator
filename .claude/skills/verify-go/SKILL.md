---
name: verify-go
description: Before COMMITTING or REPORTING a positive result — a GO, a "surpass", a milestone, an "it works / it's inert / byte-identical / tests pass" — run independent ADVERSARIAL skeptics that each try to REFUTE it from a distinct angle, then commit only what survives (commit-with-caveat, never silently drop, if a mischaracterization is found). Use whenever a result would change a wall/gap status, flip a default, or land in a findings doc / the board. This operationalizes drift-mode #8 (believing a surpass without verification) + the SILENT-FAILURE CLASS in the neural-simulator skill, which alone are a checklist — this skill makes running them a reflex before the commit.
---

# Verify-GO — adversarially verify a positive result before it lands

A GO that looks clean is the most dangerous thing in this project: it gets committed, flips a default, and mis-aims the next session. The 2026-07-24 session produced **four** over-claims caught only by adversarial verification (W3 "structural immunity", P1.2 affect-scalar, 2 gap#5 lucky-draws) and **six** silent-failure retractions — **three self-authored**. Vigilance doesn't catch these (I authored a bare-`except` swallowing my own warning the day I documented that pattern five times). A *procedure* does.

**Announce at start:** "Running verify-go: adversarially probing this result before it lands."

## When to run
- Before COMMITTING a finding that reports a GO / surpass / milestone / capability close.
- Before FLIPPING a default or claiming a cheat/shortcut is closed.
- Before writing "inert / byte-identical / no-op / tests pass / pushed / on GPU" as fact.
- NOT for routine mechanical edits (a typo, a refactor with a passing test you watched pass).

## The procedure — dispatch independent skeptics, one per lens
Spawn N skeptics (a `Workflow` parallel stage, or concurrent `Agent`s) — each gets the result + the runner/finding and is told: **"Try to REFUTE this from the <lens> angle. Default to REFUTED if uncertain."** Independence matters: redundant identical skeptics miss failure modes diverse lenses catch. Assign distinct lenses:

1. **Reproducibility / power** — does it hold at 6 seeds (42/43/44/100/101/102)? Is the effect bigger than seed-to-seed noise? A 3-seed indicator is not a GO. Is one lucky seed carrying it?
2. **Gate-cheat** — can the gate PASS without its key control? Is the anti-cheat control DEFAULT-ON and actually INVOKED (grep the call site), not just defined? A gate that passes without its control IS the bug.
3. **Control-integrity** — is the A/B lever ONE variable? (One flag ≠ one variable — a global clamp touched both a spiking synapse and a host readout.) Does the DEFAULT arm genuinely differ from the treatment (print the lever's effect; if both arms froze, the verdict is void while looking plausible)?
4. **Instrument-trust** — read the runner's OWN verdict line; NEVER lift a metric from a run that printed `SIGNAL=False` / `HONEST NEGATIVE`. Is the metric quantized/rounded so the effect is unfalsifiable by construction (differencing `round(x,4)` values)? A refutation needs the instrument verified exactly as much as a confirmation.
5. **Seeding** — is `cfg.seed` set (NOT `actual_seed_used`, which seeds nothing)? Build twice at one seed and hash `cp_neuron_firing_thresholds` — identical ⇒ actually seeded.
6. **Infra** — pushed? verify with `tools/push_both.sh` (ls-remote, not `echo pushed`). On GPU? `tools/monitor_runs.py` reports `[GPU]`/`ON CPU`. "byte-identical when off"? that's an ASSERTION (a test), not a comment.

Not every lens applies to every result — pick the ones with teeth for this claim, but a GO that flips a default earns at least reproducibility + gate-cheat + control-integrity.

## Synthesize + act
- **Survives all lenses** → commit as a GO. Say which lenses were run (the verification is part of the deliverable).
- **A lens finds a mischaracterization** → do NOT silently drop or quietly downgrade. Commit-WITH-CAVEAT: state precisely what the result IS vs what it was claimed to be (W3 was a real GO but not "structurally immune"; the affect wiring was real but not the scalar claimed). An honest narrowed result is a first-class deliverable.
- **A lens refutes it** → retract the GO, bank the method as a NO-GO with the root cause (per THE LAW the capability stays open), take the next mechanism.

## What this skill MUST NOT do
- Rubber-stamp — a skeptic that "confirms" without trying to break it did nothing. The prompt must push to REFUTE.
- Verify only the happy path — test the claim against the case you'd EXPECT to break it (a run you KNOW is broken, the seed you fear).
- Treat a refutation as needing less scrutiny than a confirmation (rule 3 of the SILENT-FAILURE CLASS).
- Touch the science verdict itself dishonestly — this skill SHARPENS a claim to the truth, it never launders a weak result into a strong one.

## Why this skill exists
2026-07-24 (evolve-skills): adversarial verification was run ad-hoc (via one-off Workflow panels) and reliably caught real over-claims — but it lived only in-session, not in a skill, so it depended on remembering to do it. This encodes it as a reflex triggered by the commit itself. Pairs with the neural-simulator skill's SILENT-FAILURE CLASS (the specific checks) — this is the procedure that runs them before a GO lands.
