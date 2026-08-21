---
type: finding
status: contributing
date: 2026-08-21
mechanism: open-ended-generation-verify-postfilter
lane: integration
seeds: [42]
seed-waiver: A rate de-risk of a deterministic post-filter over a fixed probe panel through the real spiking Qwen
  render — the load-bearing evidence is the before/after fabrication contrast (1.0 -> 0.0), not a stochastic effect.
instrument: research/runners/_open_ended_verify_postfilter_derisk.py — generates the SAME free open-ended reply as
  the prior de-risk, then applies a VERIFY post-filter, and re-scores fabrication (= not uncertainty_signaled, the
  prior runner's own metric) + known-topic substance BEFORE vs AFTER, with tools.verdict.Verdict.
runner: research/runners/_open_ended_verify_postfilter_derisk.py
external: NO-EXTERNAL-NEEDED — composes the prior open-ended generator + the store's grounding + the moat helpers.
artifacts:
  - research/findings/raw/_open_ended_verify_postfilter_derisk.json
---
# A VERIFY post-filter RESTORES honesty to open-ended generation — Qwen=FORM + VERIFY=honesty (GO), resolving the prompt-only NO-GO

Artifact: research/findings/raw/_open_ended_verify_postfilter_derisk.json (GO).

**One line.** The prior de-risk proved open-ended state-driven generation reads conversational (V1 GO) but
prompt-only state-fidelity fails — Qwen confabulates its own parametric knowledge 8/8 on brain-unknown/Qwen-known
topics ([[2026-08-21-open-ended-state-driven-generation-conversational-but-prompt-only-honesty-FAILS-verify-moat-must-stay]]).
This builds the named fix — MOVE the moat from a pre-hoc SVO constraint to a POST-FILTER on the free reply — and it
works: **fabrication drops 1.0 -> 0.0 while known-topic substance stays 1.0.**

## The post-filter (Qwen wrote freely for FORM; the moat strips what the brain can't stand behind)
Per reply: strip persona-leak sentences ("As an AI language model …"); for a BRAIN-UNKNOWN topic (empty retrieval),
keep only the uncertainty/hedge/question sentences and, if none remain, prepend an honest abstain — so the reply
SIGNALS uncertainty (the metric flips); for a KNOWN topic, drop sentences that CONTRADICT the retrieved facts (the
wrong parametric supplements), keep the rest. NO `sim/` edit; composes the existing generator + store + moat helpers.

## The verdict (cupy, real spiking-Qwen render, seed 42) — GO
<!--derived-->
- **fabrication on Qwen-known/brain-unknown: RAW 1.0 -> FILTERED 0.0** (8/8 → 0/8) — e.g. "paris" went from
  *"As an artificial intelligence … Paris is a city in western France …"* (confident + persona leak) to
  *"I'm not sure about paris — I don't have anything about it in what I've actually learned, so I'd only be
  guessing."*
- **known-topic substance: RAW 1.0 -> FILTERED 1.0** — the filter does NOT destroy the good grounded answers.
- **persona-leak rate (filtered): 0.0**; **replies emptied by the filter: 0** (still conversational).

## The live-wiring recipe (the integration this unblocks)
Open-ended generation (conversational, V1 GO) + this VERIFY post-filter (honest, GO) = the default-off
`BRAIN_OPEN_ENDED` live mode: assemble the brain STATE + retrieved knowledge -> free Qwen reply (FORM) -> VERIFY
post-filter (HONESTY) -> the reply. This is the answer to the open-ended-vs-honesty tension: neither strict SVO (not
conversational) nor free-Qwen (not honest), but free-Qwen-behind-the-moat.

## Honest scope
Single-seed rate panel through the real render. The DISABLED next rung: per-clause SVO grounding of KNOWN-topic
supplements — v1 drops only sentences the moat's `contradicts` flags, so a wrong supplement on a known topic
("Canada borders Mexico") can still survive if it does not directly contradict a stored fact; the primary failure
(unknown-topic 100% fabrication) is fully resolved. NEXT: wire `BRAIN_OPEN_ENDED` into live brain_chat + the OpenAI
shim (default-off), verify through the real handler, then the owner-UX flip. (Parent-built + verified from the
artifact.)
