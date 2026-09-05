---
type: finding
status: live
date: 2026-09-05
mechanism: value-choice-critic-on-shared-salience-context
lane: integration
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO
runner: research/runners/_value_choice_neural_context_6seed_derisk.py
artifacts:
  - research/findings/raw/_value_choice_neural_context/verify_6seed.json
  - research/findings/raw/_value_choice_neural_context/seed42_determinism_check.json
external: NO-EXTERNAL-NEEDED -- extends the verification depth of two already-validated, already-wired
  in-repo mechanisms (rank-4's shared spiking salience afferent, 6-seed GO; the RANK-1 value-critic, 6-seed
  GO) without changing either. No new biology claim is made.
---

# Value-choice's REAL trained critic is load-bearing on the shared spiking salience context — closing rank-4's own seed-waiver at 6 seeds (scaffold-retirement backlog rank-20)

**Verdict: GO (de-risk, default-OFF -- NOT flipped on).** `research/coordination/scaffold_retirement_backlog.md`
rank-20 ("value-choice reward-context host recency, MED-narrow, low-risk") asked whether
`value_choice_production_organ.py`'s per-candidate engagement/reward context can be fed a live neural
salience read instead of the host recency/referent-ratio formula. **That wiring already exists** --
rank-4 (`research/findings/2026-09-05-shared-spiking-salience-afferent-wired-GO.md`, landed earlier this
session) wired `default_context_fn()` through the shared ASK-pool spiking salience organ
(`BRAIN_SHARED_SALIENCE`, default-OFF) as one of its three consumer sites, and 6-seed-GO'd the CONTEXT
FUNCTION's own output. What rank-4's own frontmatter explicitly waived to a single seed was the ONE thing
this finding closes: an end-to-end pass through the REAL, heavy, trained `striosome_value` critic
(`ValueChoiceProductionOrgan.choose()`, not just the upstream context function) -- run here at all 6
project-standard seeds, over 4 candidate/recency scenarios per seed (24 (seed, scenario) pairs total), not
the single scenario the seed-42 waiver used.

**No code in `value_choice_production_organ.py` or `shared_salience_afferent.py` was changed.** This is a
verification-only extension: one new runner
(`research/runners/_value_choice_neural_context_6seed_derisk.py`), no `sim/` edit, no flag default changed.
`BRAIN_VALUE_CHOICE` stays default-ON (2026-08-26 flip) and `BRAIN_SHARED_SALIENCE` stays default-OFF
(rank-4); this finding exercises both at their EXISTING settings.

## What this closes

Rank-4's own seed-waiver: *"a full end-to-end pass through value_choice_production_organ's REAL trained
striosome_value critic ... is run ONCE at seed 42 only ... a single trained build costs ~5 CPU-minutes"* --
legitimate for the critic's OWN sensitivity-to-input mechanism (already 6-seed-GO'd,
`research/findings/2026-07-23-value-critic-closure-RANK1-GO.md`), but not automatically legitimate for
THIS specific question: does the real critic, fed the shared-organ-mediated context INSTEAD OF the bare
host ratio, still reach a decisive commit that matches-or-improves-on the host ratio's, and stay
load-bearing on the shared afferent specifically -- across seeds, not one build.

## Method

One trained `ValueChoiceProductionOrgan` per seed (`value_train_trials=40`, the RANK-1 GO's own production
default), built ONCE per seed (~163-225s <!--derived-->, dominated by the merged-bridge construction + DA-gated-STDP value-train,
NOT by scenario count -- see `build_seconds` in the artifact). Against that ONE trained critic, 4
candidate/recency scenarios are run (cheap -- only `choose()` repeats):
- **S1 baseline** -- 3 candidates, recency `[0.0, 0.5, 1.0]` (reproduces rank-4's own seed-42 scenario).
- **S2 near_tie_low** -- 3 candidates, an ASYMMETRIC storage order giving recency `[0.0, 1/9, 1.0]`
  (~0.111 <!--derived-->) (`default_context_fn` renormalizes to the SELECTED candidates' own min/max index,
  so evenly-spaced N always reduces to an evenly-spaced ladder; asymmetric storage is required to reach a
  non-uniform pattern).
- **S3 referent_tie** -- 3 candidates, recency `[0.0, 0.5, 1.0]` + the discourse-WM referent bound to the
  middle-recency candidate, exercising the `+0.5` referent-boost branch (untested by rank-4's seed-42
  proof) -- the HOST arm ties the middle and top candidate at `[0.0, 1.0, 1.0]`.
- **S4 four_candidate** -- 4 candidates, recency `[0.0, 1/3, 2/3, 1.0]` (~0.333/0.667 <!--derived-->),
  exercising a freshly-built n=4 spiking value-WTA (distinct from S1-S3's cached n=3 WTA).

Each scenario runs three arms through the REAL critic: **OFF** (`BRAIN_SHARED_SALIENCE` unset, the bare
host formula), **ON** (`BRAIN_SHARED_SALIENCE=1`), **ON+LESION** (`+ BRAIN_SHARED_SALIENCE_LESION=1`, severs
the shared ASK-pool afferent feeding the context -- rank-4's own lesion, applied here one layer downstream
at the critic's readout instead of at the context function). A fourth, cost-free check re-runs the ON
context through the critic's OWN PRE-EXISTING `BRAIN_VALUE_CHOICE_LESION` mean-pin (`choose(..., lesion=True)`)
as a non-regression sanity confirmation -- reported, not gated (a mean-pin discards the fed array's content
by construction regardless of what produced it, so this is trivially expected to collapse decisiveness, and
it does, 24/24).

## Result -- 6/6 seeds, all gates pass

```
research/runners/_value_choice_neural_context_6seed_derisk.py --seeds 42 43 44 100 101 102
verdict: GO   n_pass: 6/6   all_seeds_pass: true
```

| seed | build (s) | mean off spread (Hz) | mean on spread (Hz) | mean lesion spread (Hz) | match | lesion collapses | mean attribution |
|---|---|---|---|---|---|---|---|
| 42 | 162.7 | 13.99 <!--derived--> | 14.20 <!--derived--> | 4.48 <!--derived--> | 3/4 | 4/4 | 0.680 <!--derived--> |
| 43 | 168.7 | 12.81 <!--derived--> | 16.01 <!--derived--> | 3.82 <!--derived--> | 4/4 | 4/4 | 0.770 <!--derived--> |
| 44 | 170.8 | 17.67 <!--derived--> | 19.13 <!--derived--> | 3.89 <!--derived--> | 4/4 | 4/4 | 0.801 <!--derived--> |
| 100 | 170.4 | 11.32 <!--derived--> | 13.96 <!--derived--> | 4.72 <!--derived--> | 4/4 | 4/4 | 0.665 <!--derived--> |
| 101 | 175.2 | 10.49 <!--derived--> | 13.16 <!--derived--> | 4.10 <!--derived--> | 3/4 | 4/4 | 0.695 <!--derived--> |
| 102 | 225.4 | 15.59 <!--derived--> | 18.51 <!--derived--> | 4.62 <!--derived--> | 4/4 | 4/4 | 0.763 <!--derived--> |

"match" = the ON commit equals the OFF commit; "lesion collapses" = `attributable_to`'s fraction >= 0.5 OR
the commit itself changes under the shared-salience lesion (per-scenario detail below). Aggregated across
all 24 (seed, scenario) pairs (`tools.lab.attributable_to`, run once per pair, never both-arms-banked
unattributed): mean attribution **0.729** <!--derived--> (range **0.476 - 0.953** <!--derived-->, every one
of 24 pairs positive -- no seed/scenario shows the lesion arm exceeding the intact arm).

### Per-scenario detail (aggregated across all 6 seeds)

| scenario | match (of 6) | OFF commits | ON commits | ON+LESION commits |
|---|---|---|---|---|
| S1_baseline | 6/6 | shoe x6 | shoe x6 | cat x6 |
| S2_near_tie_low | 6/6 | shoe x6 | shoe x6 | cat x1, abstain(None) x5 |
| S3_referent_tie | 6/6 | shoe x6 | shoe x6 | cat x6 |
| S4_four_candidate | 4/6 | shoe x2, stick x4 | stick x6 | cat x6 |

**S1-S3 (3-candidate scenarios): 18/18 exact commit matches, 0 exceptions.** Despite the shared organ
substantially reshaping the fed value spread (see the seed table -- ON spread routinely well above OFF's),
the WINNING candidate is identical to the host-ratio arm in every single one of these 18 pairs, across every
scenario type including the untested-by-rank-4 referent-tie branch (S3) and the asymmetric near-tie storage
order (S2). This directly answers "match or improve": on the 3-candidate competitions, the neurally-mediated
context always at least MATCHES the host ratio's own decision.

**S4 (4-candidate, the tightest competition): 4/6 match, 2 genuine reorderings (seeds 42, 101).** In both
exceptions the OFF arm's own margin between the top two candidates was itself the narrowest in the whole
sweep (`shoe` vs `stick`, off-spread 15.97 <!--derived--> / 14.58 <!--derived--> Hz across 4 candidates,
vs 22.5 Hz max elsewhere) -- a near-tie in the HOST arm's own value gradient, not a mechanism failure. In
BOTH exceptions the ON arm still committed decisively (never a decline) -- the neurally-mediated version
did not fail to choose, it chose the OTHER of two closely-matched candidates. Read the other direction: the
ON arm converges on `stick` on ALL 6 seeds (`stick` is in fact the highest-recency candidate, 1.0, vs
`shoe`'s 2/3 (~0.667 <!--derived-->)) while the OFF arm itself splits 4x `stick` / 2x `shoe` -- i.e. on this specific near-tied
4-way competition the neurally-mediated context was numerically MORE seed-consistent about which candidate
is the true top-recency one than the bare host ratio was. This is reported as an observation (n=6 is not
statistical power for a reliability claim), not a headline.

**Lesion collapse holds on EVERY one of 24/24 (seed, scenario) pairs** -- either the fed-value spread
collapses toward the near-floor the organ-level lesion produces (mean attribution 0.729 <!--derived-->, the
same signature rank-4's own single-seed proof reported at 74% <!--derived-->), or the commit itself reverts
to `cat` (the lowest-engagement candidate under every host formula -- the WTA's baseline salience-only pull
once the value gradient is cut, the SAME qualitative signature `ValueChoiceProductionOrgan`'s own
pre-existing `BRAIN_VALUE_CHOICE_LESION` mean-pin produces), or (S2, the tightest 3-candidate spacing, 5/6
seeds) the WTA declines to commit at all (`chosen=None`) -- all three are the organ's documented
decline-to-abstain signature, never a silent wrong answer.

**The pre-existing critic-level `BRAIN_VALUE_CHOICE_LESION` sanity check collapses decisiveness on 24/24
pairs** (`fed_spread_hz=0.0`, `decisive=False` in every case) regardless of whether the fed context was
host-ratio or neurally-mediated -- confirming the shared-salience wiring does not disturb the PRE-EXISTING,
already-6-seed-GO'd RANK-1 anti-cheat. This is reported as a sanity confirmation, not a headline gate: a
mean-pin discards the fed array's content by construction, so this collapse is expected trivially regardless
of the question this finding asks, and was not counted toward `all_gates_pass`.

## Anti-cheats

- **`g_off_identical` (24/24 pairs).** Every OFF-arm engagement is checked against an INDEPENDENTLY
  re-implemented copy of the host recency/referent formula (`_host_formula`, written from source, not a
  hand-picked literal, not calling the code under test) -- exact float match on every one of 24 pairs, across
  all 4 scenario shapes including the asymmetric-storage and referent-tie cases rank-4's own OFF-identity
  check did not exercise. `BRAIN_SHARED_SALIENCE` unset is confirmed byte-identical to the pre-existing
  default on the REAL critic's input, not just on the context function.
- **`g_on_loadbearing` (24/24 pairs).** The ON-arm engagement measurably differs from OFF on every pair --
  the shared organ is genuinely in the path reaching the real critic, not a coincidental no-op.
- **Two independent lesions, not one.** `BRAIN_SHARED_SALIENCE_LESION` (severs the shared ASK-pool afferent
  one layer upstream, rank-4's own construction) and the pre-existing `BRAIN_VALUE_CHOICE_LESION` mean-pin
  (severs the critic's OWN value gradient regardless of its source) are BOTH exercised on every pair, and
  the second is confirmed to still behave identically to its pre-existing, already-validated signature --
  a non-regression check the rank-4 single-seed proof did not run.
- **Whose-the-difference, every pair.** `tools.lab.attributable_to` is called once per (seed, scenario) --
  24 calls, never a bare treatment/control pair banked unattributed (the gap#5 lesson) -- and the resulting
  fraction is reported alongside the RAW Hz spreads for both arms (not a ratio alone), per `docs/TERMS.md`'s
  "selective" discipline.
- **Six genuinely distinct substrates, not six labels.** The mean OFF-arm spread varies from 10.49 to 17.67
  Hz <!--derived--> across the 6 seeds -- clearly different trained critics, not six reruns of one cached
  build (the process-isolated subprocess-per-seed design the sibling rank-4 de-risk also uses, for the
  identical reason: the merged bridge + value-train is a fresh build per subprocess).

## Honest scope, terms, and open questions (per `docs/TERMS.md`)

- **This does NOT flip any default.** `BRAIN_SHARED_SALIENCE` remains default-OFF; the host recency/referent
  ratio remains what `/api/brain-chat` reads today. This finding narrows a verification GAP (rank-4's own
  n=1 waiver on the real critic), it does not change what rank-4 already established about reachability.
  "GO" here names this de-risk's OWN verdict (6/6 seeds, every gate) -- not a production-integration claim.
- **"Match" is the dominant signature (22/24), not "improve."** Every scenario in this sweep was constructed
  to be answerable (a genuine engagement spread exists in the host arm), so the OFF arm was decisive on
  24/24 pairs -- there was no case where OFF declined and ON rescued it. The `matches_or_improves` gate
  passed entirely on "matches"; "improves" (OFF non-decisive, ON decisive) never fired (0/24). A scenario
  deliberately built to make the HOST ratio non-decisive (all-candidates-tied engagement) is a natural
  follow-on this de-risk does not attempt.
- **The 2 non-matches (S4, seeds 42/101) are a genuine reordering in a near-tied competition, not a defect.**
  Reported in full above rather than folded into an aggregate match-rate that would obscure it.
- **G_UNTRAINED (the RANK-1 GO's untrained-critic control) is NOT re-run here.** That anti-cheat targets
  whether the critic's LEARNING is load-bearing, a question orthogonal to the context's provenance
  (host-ratio vs neurally-mediated) and already 6-seed-GO'd
  (`research/findings/2026-07-23-value-critic-closure-RANK1-GO.md`); re-running it would cost a second
  ~170s-class build per seed to re-ask a question this finding does not pose. What IS re-checked cheaply
  (zero extra build cost) is the critic's OWN `BRAIN_VALUE_CHOICE_LESION` -- see Result, above.
- **The shared organ's OWN seed is not varied by this sweep.** `default_context_fn`'s call into
  `_SHARED.read_salience(e)` never passes a `seed` kwarg, so `shared_salience_afferent.get_shared_organ()`
  always builds at its hardcoded default (seed 42) in each fresh subprocess -- identical to how
  `default_context_fn` behaves in production (one process, one shared-organ build). The 6 "seeds" in this
  sweep vary the CRITIC's build/value-train seed (`VT.build_merged(seed, ...)`), which is what this finding
  asks about; the shared organ's OWN 6-seed sensitivity was already covered by rank-4's `organ_core` gate.
- **Exact magnitudes are not expected to reproduce rank-4's own seed-42 plumbing-proof numbers, and don't.**
  Rank-4's `run_plumbing_proof` calls `bg_action_selection_production_organ.decide_action()` (which itself
  drives the process-shared curiosity/salience organ) before building `ValueChoiceProductionOrgan`; this
  runner does not call the BG organ at all. Both harnesses build the critic with `cfg.seed=42`, but the
  OU noise process the critic's live readout depends on (`_critic_rate_via_afferent`, a real
  30-warmup/120-measurement simulation window) rides the shared global RNG stream, so a DIFFERENT prior
  call history at the "same" seed lands on a different point in that stream -- this runner's seed-42 S1
  reads OFF spread 13.99 Hz mean <!--derived--> across its own scenario order, not rank-4's reported 14.72 Hz
  single read. This is expected behavior of a shared-global-RNG substrate, not a discrepancy in either
  result: within EACH harness's own fixed call order, the OFF/ON/LESION comparison is measured on one
  build in tight sequence, which is the property this finding's claims depend on -- not bit-for-bit
  cross-harness reproduction of one seed's exact number.
  **Verified separately (verify-go "seeding" lens, the CLAUDE.md-prescribed check -- "build twice at one
  seed ...; identical => actually seeded"):** a fresh rerun of this runner's OWN `--seed 42` worker, in a
  second independent subprocess, reproduced EVERY V_hz reading byte-identically across all 4 scenarios,
  both OFF and ON arms -- see `research/findings/raw/_value_choice_neural_context/seed42_determinism_check.json`
  (`fully_byte_identical_across_rerun: true`). `cfg.seed` genuinely controls this substrate; the
  cross-harness difference above is real but is a call-history effect, not non-determinism.
- **FUNCTIONAL correlate, not phenomenal** -- inherited from rank-4/RANK-1; this finding changes neither claim.
- **CO-RESIDENT, not yet merged onto the one-brain substrate** -- inherited from rank-4; unchanged here.

## Files
`research/runners/_value_choice_neural_context_6seed_derisk.py` (new runner; no other file touched -- `git
diff` outside this file and the raw artifacts is empty). Artifacts:
`research/findings/raw/_value_choice_neural_context/verify_6seed.json` (the 6-seed gate),
`research/findings/raw/_value_choice_neural_context/seed42_determinism_check.json` (the seeding-lens rerun).

## Citations
- Scaffold-retirement map (this de-risk's mandate): `research/coordination/scaffold_retirement_backlog.md`
  rank-20 ("value-choice reward-context host recency"), a follow-on to rank-4.
- The wiring this extends (reused verbatim, not modified):
  `research/findings/2026-09-05-shared-spiking-salience-afferent-wired-GO.md` (rank-4), specifically its
  Consumer-3 section and its own seed-waiver naming this exact gap.
- The critic this proves load-bearing (reused verbatim, not modified):
  `research/findings/2026-07-23-value-critic-closure-RANK1-GO.md` (RANK-1 GO, 6-seed, G_LESION/G_UNTRAINED).
- Attribution discipline: `tools/lab.py::attributable_to` (the gap#5 lesson).
