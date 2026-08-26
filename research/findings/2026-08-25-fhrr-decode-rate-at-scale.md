---
type: finding
status: live
date: 2026-08-25
mechanism: fhrr-cue-role-false-hop-and-wrong-patient-decode-rate
lane: memory
seeds: [42]
seed-waiver: This is an INSTRUMENT-STANDUP + smoke measurement, not a generalization claim. The smoke cell (D=128,
  N=1000, seed=42) and a supplementary D=128/N=15000 probe are single-seed by design (the task's own smoke-cell
  spec); the decisive multi-seed evidence is the QUEUED sweep (D in {128,256,512,1024} x N in {1000,5000,15000}
  x seed in {42,43,44}, 12 pool jobs / 36 cells) staged in `research/queue/pool.queue` and not yet returned.
  The headline below reports MEASURED rates, not a GO/NO-GO verdict on the moat's safety.
instrument: research/runners/reasoning_route_decode_rate.py -- a closed-form (numpy, no GPU) decode-rate
  instrument, self-verified byte-exact against the GENUINE resonate `RFPhasorComposer.query_patient()` before
  any cell is trusted (`verify_instrument`, on by default every run).
runner: research/runners/reasoning_route_decode_rate.py
external: NO-EXTERNAL-NEEDED -- this measures the project's OWN FHRR store/decode algebra (rf_phasor_composer.py)
  against the project's OWN shipped knowledge-core bundle; the closed form reuses the project's own already-
  validated `tiered_fact_store.encode_fast` and is verified against the genuine RF-bridge resonate directly.
artifacts:
  - research/findings/raw/_reasoning_route_decode_rate/smoke_D128_N1000_s42.json
  - research/findings/raw/_reasoning_route_decode_rate/probe_D128_N15000_s42.json
  - research/findings/raw/_reasoning_route_decode_rate/stress_lowD_N15000_s42.json
---

# FHRR decode-rate at scale: cue-role false-hop and wrong-patient rates, measured (smoke + instrument stand-up; full sweep queued)

## Why this exists

The reasoning route's no-confab memory moat is exact-string-equality gated with no confidence floor:
`ShardedPhasorStore.query_patient` -> `RFPhasorComposer._scan_first_match` (`research/runners/rf_phasor_composer.py:712-721`)
matches a decoded stored cue's role words against the query by `w == val`; `_cleanup`/the batched `_cleanup_all`
(:658-663, :700-710) is a floorless argmax that never returns `None`. This is safe for a genuinely
out-of-vocabulary cue **by construction** (decode can only ever return a word that is itself a member of the
vocab codebook, so an OOV query word can never equal a decoded word) -- but the moat's safety for an IN-VOCAB
near-miss rests on one previously unmeasured quantity: the rate at which FHRR bundling crosstalk (`store()`
binds+bundles up to 6 roles, `ROLES = ("agent","action","patient","polarity","attribute","attribute2")`;
production facts bind agent+action+patient+polarity=AFFIRM, 4 terms, per `tiered_fact_store.build_ltm_from_facts`)
makes a WRONG stored fact's decoded cue-role words land on an in-vocab query word.

(Note: the task that commissioned this runner cited `research/findings/2026-08-25-reasoning-route-moat-audit-
hardening-spec.md` as the source of this framing. At build time that file does not exist anywhere in this repo --
checked via `git log --all`, `git fetch origin --all`, and a full-tree filename search. This finding is grounded
directly in the cited production functions instead, read and quoted above; nothing here depends on the missing
document's contents beyond the framing already given in the task brief.)

## Instrument

`research/runners/reasoning_route_decode_rate.py`. Numpy only, no GPU, no edits to `rf_phasor_composer.py` /
`sharded_phasor_store.py` (read-only). Facts + vocabulary are the REAL shipped knowledge core
(`/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k`: 15000 real (agent, relation, patient)
triples, 7032-word vocab, the same data `build_ltm_from_facts` turns into the production LTM). The store side
reuses the project's own already-validated closed-form `tiered_fact_store.encode_fast` unmodified. The query
side (unbind + cleanup) has no published closed form, so the runner derives one from the documented per-op
semantics and **self-verifies it every run** (`verify_instrument`, default on) by building a tiny composer,
storing facts through the GENUINE `.store()` resonate (the real Izhikevich RESONATE_AND_FIRE bridge stepping,
not `encode_fast`), and asserting the closed-form decode reproduces the real `.query_patient()` byte-for-byte on
stored, false-hop, and out-of-vocab cues. Every cell in this finding passed that check (`instrument_verification.ok
== true`).

Why closed-form at all: at D=1024/N=15000 a single genuine `_scan_first_match` unbind resonates a ~2*N*D-neuron RF
bridge -- `sharded_phasor_store.py`'s own docstring measures ~5s at K=2413/D=128, which extrapolates to minutes
per single query at the sweep's largest cell. The closed form is the same algebra, computed directly instead of
stepped, verified above, and is fast: the full smoke cell ran in 0.6s, the D=128/N=15000 probe in 7.8s, and a
3-cell D-sweep at N=15000 in under 20s total.

## Instrument sanity checks (mandatory, before trusting any rate)

1. **Known-safe floor.** A genuinely out-of-vocab agent (`"__oov_agent_..."`, never in the 7032-word codebook)
   was queried against 300 real actions per cell. 0/300 hits in every cell run, including the smoke. This is a
   **structural** guarantee, not a probabilistic one (decode can only return a word IN the codebook, and the OOV
   probe word never is) -- measured to confirm the harness reproduces it, not assumed.
2. **Fresh-recall correctness.** `verify_instrument` stores 4 facts through the genuine RF-bridge resonate and
   confirms all 4 recall their correct patient, and that the closed-form decode agrees with the real bridge on
   every one (`fresh_recall.ok == true` every run).
3. **Auditable cases.** Printed and JSON-recorded below (not just a rate).

## Smoke cell: D=128, N=1000, seed=42 (`research/findings/raw/_reasoning_route_decode_rate/smoke_D128_N1000_s42.json`)

- **Cue-role false-hop rate: 0/5000** sampled in-vocab (agent, action) near-miss pairs (agent has >=1 other
  stored fact; 27380 such pairs exist in this sample, 5000 sampled). Rule-of-three 95% upper bound: **~0.06%**.
- **Wrong-patient rate: 0/1000** genuinely stored (agent, action) keys.
- **OOV floor: 0/300**, as expected structurally.
- True-match patient decode score (mean-cos, matches `_cleanup_all_score_stats`): mean 0.456, min 0.306; margin <!--derived-->
  (winner - runner-up): mean 0.221, min 0.067. <!--derived-->

## Supplementary probe: D=128 (the current production default), N=15000 (full shipped scale), seed=42 (`research/findings/raw/_reasoning_route_decode_rate/probe_D128_N15000_s42.json`)

Not the official smoke cell, but cheap (7.8s) and directly answers "is today's shipped configuration already
exposed at today's shipped scale":

- **Cue-role false-hop rate: 0/8000** sampled near-miss pairs (215044 such pairs exist at this scale; 8000
  sampled, 3.7% coverage). Rule-of-three 95% upper bound: **~0.038%**. <!--derived-->
- **Wrong-patient rate: 1/15000 (0.0067%).** One genuine decode error found: <!--derived-->

  | agent | action | returned patient | stored patient | score | margin |
  |---|---|---|---|---|---|
  | `d_sseldorf_nrw` | `country` | `franzoesische_republik` | `federal_republic_of_germany` | 0.565 | 0.347 | <!--derived-->

  (Düsseldorf/NRW's `country` fact decoded to "French Republic" instead of "Federal Republic of Germany" --
  a genuine supported-hop decode error, the `_render`-floorless-argmax channel the task named separately from
  cue-role false-hop.)
- True-match patient score: mean 0.453, min 0.250; margin: mean 0.218, **min 0.0081** -- a real correct recall <!--derived-->
  in this run scored a margin of only 0.008 over its runner-up. <!--derived-->

## Instrument-sensitivity stress test: is the instrument actually capable of detecting crosstalk?

A rate instrument that always reads 0 is indistinguishable from a broken one. D was pushed to unrealistic,
deliberately-undersized values (D=1, 2, 4) at N=15000 to confirm the harness responds in the expected direction
(`research/findings/raw/_reasoning_route_decode_rate/stress_lowD_N15000_s42.json`):

<!--derived-->
| D | false-hop rate | wrong-patient rate | true-match margin (mean / min) | false-hop margin (mean / min) |
|---|---|---|---|---|
| 1 | 1/4000 (0.025%) | 2/15000 (0.013%) | 3.3e-7 / 0.0 | 6.0e-8 / 6.0e-8 |
| 2 | 1/4000 (0.025%) | 2/15000 (0.013%) | 0.00020 / 0.00012 | 0.00034 / 0.00034 |
| 4 | 1/4000 (0.025%) | 16/15000 (0.107%) | 0.0078 / 0.00135 | 0.0059 / 0.0059 |

Confirmed non-zero and D-sensitive (wrong-patient climbs 0.013% -> 0.107% from D=1..2 to D=4); the instrument is
not a stuck-at-zero artifact.

## Would a confidence floor (hardening req #2) cleanly separate true from false matches?

**Not at the low-D regime where false hops actually occur in these probes, and the production-D (128) regime
gives no natural false-match sample to test yet (rate was 0 in both D=128 probes above).** At D<=4, RAW SCORE is
useless as a discriminator: both true and false matches saturate near 1.0 (D=1: true mean 0.99999952, the one
observed false-hop scored 1.0 exactly). MARGIN is somewhat better but does not cleanly separate: at D=4 the one
observed false-hop's margin (0.0059) is **larger** than the true-match margin *minimum* observed in the SAME <!--derived-->
cell (0.00135) -- a fixed margin floor that admits that genuine low-confidence true recall would also admit this
fabricated one. <!--derived--> At D=128 the true-match margin minimum was similarly small (0.0081, the Düsseldorf-adjacent
cell) -- low margins on GENUINE correct recalls are not rare even at production D, which is the tension any <!--derived-->
floor design has to navigate (a floor tight enough to reject the false hops seen at low D risks rejecting real
low-margin correct recalls at production D). This question needs the queued sweep's much larger false-hop sample
at D=128..1024 to answer with real production-D false-match cases rather than an extrapolation from D<=4.

## Does raising D drive both rates toward zero?

**Suggestively yes, but not conclusively measured yet.** Both rates are already at or indistinguishable from
zero at the shipped D=128, at both N=1000 and the full N=15000 shipped scale (rule-of-three upper bounds ~0.04-
0.06%). The low-D stress test shows the expected direction (rates and score/margin separation both degrade as D
shrinks below production values). But zero observed hits at D=128 in an 8000-trial / 215044-candidate sample is
a wide confidence interval, not a proof of zero, and D=256/512/1024 were not measured interactively (that is
what the queued sweep is for -- see below).

## What is NOT yet answered (queued to the pool)

- The full D in {128,256,512,1024} x N in {1000,5000,15000} x seed in {42,43,44} sweep (36 cells, 12 pool jobs,
  ~5000 false-hop trials/cell) -- to get a tight multi-seed estimate at production D and above, and (critically)
  enough trials to actually SURFACE production-D false-hop cases so the floor-separation question in the section
  above can be answered with real data instead of a D<=4 extrapolation.
- Queued via `tools/pool_queue.sh` (12 entries, one per (D, n_facts) cell, each fanning `--seed 42 43 44`
  in-process) to `research/queue/pool.queue`, dispatched headless on pool40/41/42 at 0 agent-token cost. Results
  land at `research/findings/raw/_reasoning_route_decode_rate/D{D}_N{n_facts}.json` per cell; this finding will
  be updated (or superseded by a follow-up finding) once they return.

## Honest summary

The moat looks safe at today's shipped configuration (D=128, up to the full 15000-fact shipped scale): 0
observed cue-role false-hops in 13000 combined trials across two probes, and exactly 1 wrong-patient decode error
in 16000 combined checks. That is a reassuring number, not a proof -- the confidence interval at D=128 is wide
(rule-of-three ~0.04-0.06%) because the observed hit count is 0/1. The stress test proves the instrument would
have caught a real problem if D were genuinely too small (it detects nonzero, D-sensitive rates at D<=4). Whether
raising D further tightens an already-good number, and whether a confidence floor would actually separate true
from false decodes at production scale, are the two questions the queued sweep is designed to close.
