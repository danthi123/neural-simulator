---
type: plan-draft
status: PROVISIONAL
date: 2026-10-03
lane: prove-who-owns-the-computation
mechanism: symmetric CA3-CA3 night for sleep transitive inference (the successor method to the asymmetric-STDP night)
runner: research/runners/_ca3_symmetric_ti_symrule.py (to be created; copy of the registered runner)
seeds: [8, 9, 10, 11] (dev only; the gate seeds stay Claude's)
---

# PLAN — the symmetric CA3-CA3 night as the sleep-TI successor (PROVISIONAL, dev-only draft)

## 1. The question

On the registered CA3 replay store, does changing **only the night's plasticity rule** — from the engine's asymmetric
Bi-Poo STDP to a symmetric, order-free, co-activation CA3-CA3 rule — make the SWR night write the A→D inference
**while keeping the A→B and B→D premises** (the STI7 repair), under a within-class weight-permutation control that is
valid from the start (the STI3 repair)? One question, one delta: the night's rule. Everything else (store, episodes,
night protocol, probe, reads) is held to the registered runner's behavior; the full code delta — including the arm-set
reduction and the strengthened gate-seed refusal — is enumerated in section 12.

## 2. Why now (the evidence that forces this method)

From `research/findings/2026-10-01-sleep-transitive-inference-6seed-UNDEFINED.md` (the family word is the scorer's
UNDEFINED; inspected, grader re-derives `aggregate.json` byte-identical):

- **STI7 fails 4/4 (as registered):** the night that writes A→D erodes the premises. PAB 0.59→0.43, PBD 0.52→0.41 on
  dev; an A→D link bought by erasing B→D is a lossy rewire, not relational integration.
- **STI3b fails 3/4 (not predicted):** a night replayed in **reverse order also writes the inference**. The asymmetric
  rule's temporal order is not what does the work; the effect is co-activation-driven integration — the regime of the
  symmetric CA3-CA3 rule.
- **U6 fires on seeds 42 and 101:** the global weight-permutation control lost the premises or flooded the foils, so
  STI3 had no valid control. Prereg section 10 names the fix: permute **within classes of synapses of equal pre-night
  weight**.
- **RULING T3-sleep-ti** (`research/coordination/LOCAL_LLM_RUNBOOK.md`): no standalone U6 re-run or U6 prereg; the next
  family registers section 10's STI7 companions with the within-class permutation as its STI3 control **from the
  start**; the first companion is the symmetric CA3-CA3 rule (Mishra 2016), "the cheapest faithful step" because of
  STI3b; then sleep downscaling. This draft executes exactly that first companion on dev seeds, then stops.

The registered family's verdict is a NO-GO **for the method** (asymmetric-STDP night, PV loop on, no downscaling), not
for the capability (no defer). This draft is the named next method.

## 3. The biology it follows

- **CA3-CA3 plasticity is symmetric and broad.** Mishra, Kim, Guzman & Jonas 2016, Nat Commun 7:11552
  (doi 10.1038/ncomms11552): CA3-CA3 STDP is LTP **independent of temporal order**, symmetric, half-width ~150 ms —
  "the only STDP measured in CA3" (`research/findings/2026-07-24-gap5-encode-only-derisk-NEGATIVE-...`). Under it,
  co-reactivation potentiates **both directions whatever the order**, so replay cannot depress a premise
  (prereg section 10 item (1); bound in `research/biology/sleep-relational-integration.md`, note at line 31 — biology
  is bound there once, not re-bound here).
- **CA3 recurrent LTP is associative/Hebbian** — Kandel 6e Ch 54; Marr 1971 autoassociator
  (`research/findings/2026-07-08-riii-ca3-attractor-formation-symmetric-hebbian.md`), and the engine already carries a
  guarded, default-off symmetric (offset-free) co-activity Hebbian for exactly this reason (`sim/config.py`
  `hebbian_symmetric`, `sim/bridge.py:1240`, `:10128`).
- **Premise retention binds to the capability** (Ellenbogen 2007: premises kept by every group, over 85 %) — that is
  why STI7's shape is read here at all.
- **The brake is a separate companion:** sleep downscaling (`research/biology/sleep-load-dependent-renormalization.md`,
  prereg section 10 item (2)) is registered as the second companion, not smuggled in as a decay constant (section 12).

## 4. What this draft does NOT claim

- **PROVISIONAL, dev-only.** Four dev seeds (8–11). No gate seed (42/43/44/100/101/102) runs here; the copy refuses
  them in code. No generalization claim is made or possible (non-negotiable 6: six seeds before any generalization).
- **Not the capability verdict.** Nothing here reads GO/NO-GO, passes a gate, or closes anything. The verdict
  vocabulary of this draft is the dev reading words of section 7 and nothing else. Gate seeds, the family
  preregistration, any amendment to a constant/bar/arm after this commit, and the merge are **Claude's**
  (RULING T3-sleep-ti).
- **No `sim/` edit. No edit of the registered runner, `sleep_transitive_inference_grade.py`, or any gate.** The only
  new code is the runner copy named in the frontmatter (plus its own record-assembly/report mode; no new scorer file).
- **This is a CA3 store, not the chat brain** (WM11 carries the question to the conversation at Step 7 — Claude's).
- **Host roles unchanged** from the registered runner's declaration: the world (which assemblies are driven), the body
  clock, the SWR initiation cue, the instruments, and the experimenter's interventions. No host code relates A to D.
- **Honesty boundary:** every read is a functional read-out of the network's own activity and weights; no phenomenal
  claim of any kind.

## 5. Seeds

**8, 9, 10, 11** — the ruling's named dev set ("dev seeds 8-11 only"), four seeds inside the 1–15 dev range, never a
gate seed. Disclosed split (prereg section 2's precedent): **seed 8 is the operating-point seed** (constants, section
11), **seeds 9–11 are the read seeds**; both are reported separately and together. Each seed = one process, numpy.

## 6. The arms (minimal faithful set)

Same store, same episodes, same night protocol, same probe as the registered runner; the delta is the night's rule
(section 12). Fresh build per phase, weights carried; `cfg.seed` set to the seed.

| arm | night | the question it answers |
|---|---|---|
| `wake` | none (probe straight after the episodes) | premises learned; the pre-sleep read (PAB/PBD/DI/WAD at wake) |
| `sym` | SWR replay, engine's **symmetric** rule on (STDP off) | does the symmetric night let the premises survive (STI7 repaired) **and** still write A→D? does it avoid false inference? |
| `wpermc` | the `sym` night's own weight changes, permuted **within classes of equal pre-night weight** | valid STI3 control from the start: writes no inference, keeps premises |
| `dis` | SWR replay with the symmetric rule, each event cut 4.2 ms after start (the registered `dis` lesion) | replay necessary: removes the link, keeps premises (the lesion is verified on raster+weights at measurement, per `docs/TERMS.md`) |
| `rev` | instrument: the `sym` raster time-reversed inside every SWR period, through a replica of the symmetric rule | order-independence **expected-pass**: the rule writes whatever the order (STI3b's 3/4 finding); also the replica-vs-engine identity check |

Instrument (no probe): the same replica applied to the **un-reversed** `sym` raster must reproduce the engine's `sym`
Δw (max abs ≤ 1e-3 pA) — the U3-analogue for this rule. `rev`'s expected-pass is the Δw identity between the reversed
and un-reversed replica passes; since the rule is same-step, the identity is by construction, and its value is
catching an accidental asymmetric term in the rule as executed.

## 7. Outcome-to-reading table (FIXED BEFORE ANY RUN)

This table is fixed at this commit and governs every run of this draft, including smokes. It maps each outcome to the
reading/field it is read on and to its interpretation. No cell claims GO or a gate result.

| # | outcome (dev reads) | reading / field | interpretation |
|---|---|---|---|
| 1 | `sym`: PAB and PBD ≥ max(0.30, 0.9×wake) **and** DI(`sym`)−DI(`wake`) ≥ 0.15 **and** WAD rise ≥ 20 pA | PAB, PBD, DI, WAD | **REPAIR-SHAPED**: the symmetric night integrates without eroding premises. The successor rule's hypothesis survives dev; Claude's family prereg may register STI7 with this rule. Not a capability verdict. |
| 2 | `sym`: premises kept but DI gain < 0.15 (WAD rise < 20 pA) | PAB, PBD, DI, WAD, `cospike` counts | **WRITE-ABSENT**. If total same-step co-spike pairs over the whole night `cospike_total` == 0 → **UNDEFINED(dev): rule never exercised** (U8; the 2026-07-08 same-step co-spike bottleneck), never "the rule is wrong". If `cospike_total` ≥ 1 → the symmetric write is too weak at these constants; dev evidence for the broad-window stand-in (section 13), Claude's call. The raw counts are reported either way. |
| 3 | `sym`: DI gain ≥ 0.15 but PAB or PBD < max(0.30, 0.9×wake), with that arm's B_k read below its own wake value | PAB/PBD components, per-arm recruited fractions | **EROSION-PERSISTS**: the erosion is not rule-asymmetry; the night needs the brake → the sleep-downscaling companion (section 14) is the necessary next method. |
| 4 | `sym`: DI gain ≥ 0.15 but PAB or PBD < max(0.30, 0.9×wake), with that arm's B_k read ≥ its wake value, and the foil/other reads rise | PAB/PBD components, foil read, C read | **SPECIFICITY-FLOOD**: potentiation-only without a brake raises the foils (the Wang plasticity-plus-noise confound). Downscaling / PV companions again. |
| 4b | `sym`: DI gain ≥ 0.15 (≥ 0.15) with premises kept but WAD rise < 20 pA | DI, WAD | Reported as-is under row 1's fields with the WAD number stated; no separate interpretation word — the WAD bar is part of row 1's REPAIR-SHAPED shape, so this outcome is row 1 **not** REPAIR-SHAPED on the WAD clause (DI-only gain, read cautiously). |
| 5 | `wpermc`: DI(`wpermc`)−DI(`wake`) ≤ 0.05, premises ≥ 0.30, foil ≤ 0.05 above `wake`, C < 0.10 | DI, PAB/PBD, foil, C, class histogram | **VALID-CONTROL**: STI3 is readable; if also DI(`sym`)−DI(`wpermc`) ≥ 0.5×the gain, the inference rides the night's weight structure (the STI3 shape), at matched plasticity. |
| 6 | `wpermc`: premises lost or foil/C flooded, or most night \|Δw\| mass sits in singleton classes | control validity reads + class histogram | **CONTROL-DEGENERATE** (U6-analogue): no STI3 reading is taken from this seed — UNDEFINED(dev), never a pass and never a zero. |
| 7 | `dis`: DI(`dis`)−DI(`wake`) ≤ 0.05 **and** premises ≥ 0.30, with the lesion verified on raster+weights | DI, PAB/PBD, U2-analogue checks | **REPLAY-NECESSARY** shape: the link needs the replay night; the premises do not. |
| 8 | `dis`: DI(`dis`)−DI(`wake`) > 0.05 while the lesion held | DI(`dis`), pre-sleep WAD | **ROUTE-FIRST**: the write is not carried by replay events; read pre-sleep WAD for encode-time integration (Zeithamova 2012) before any claim; no interpretation is taken from `sym`'s DI. |
| 9 | `sym`: foil read ≤ 0.05 above `wake`'s and every never-encoded C < 0.10 | foil read, C read | **NO-FALSE-INFERENCE** shape (the STI5 shape) holds under the symmetric night. |
| 10 | `rev`: max \|Δw(rev)−Δw(sym)\| ≤ 1e-3 pA (and the un-reversed replica reproduces the engine) | Δw identity, replica check | **ORDER-INDEPENDENT-CONFIRMED** (expected-pass): the rule writes whatever the order, matching STI3b's 3/4. STI3b is not carried as a gate in this family. |
| 11 | `rev` identity fails | same | **UNDEFINED(dev): instrument** (replica/rule mismatch); no biology reading from that seed. |
| 12 | any U-analogue fires (lesion not held at measurement, rule never exercised (`cospike_total` == 0), missing arm, run error, part disagreement, no SWR event) | per section 13 | **UNDEFINED(dev)** for that seed. One re-run only for the run-error analogue, at the same pinned revision; anything else waits for a Claude amendment. |
| 13 | any outcome not mapped by rows 1–12 | — | **UNDEFINED(dev)** for that seed; the mapping itself is amended only by Claude (never mid-draft). |

## 8. Reads and metrics available in the existing outputs

All from the registered runner's read set (`sleep_transitive_inference.py` section-5 semantics, field names preserved
in the copy's records so a later Claude grader can consume them):

- **DI** (direct inference, hub-silenced), **FI** (free inference, reported), **PAB / PBD** (premises), **PBP** (the
  minor detail, reported), each as chain-matched differences against the other chain.
- **Weights:** per-ordered-pair means; WAD and WDA; between-pair mean; mean absolute Δw over plastic synapses vs the
  arm's own pre-night store; per-arm before/after.
- **Replay:** per-assembly SWR participation counts, A_k/D_k and A_k/D_other co-participation, per-event reach.
- **New instrument fields in the copy:** `cospike` (same-step co-firing pair counts per assembly pair during the
  night), `wpermc` class histogram (class sizes, Δw-mass share per class), replica identity residuals.

PM1–PM3 stay with the Claude-registered family; this draft reports PBP and replay counts as observations only.

## 9. Bar thresholds, in dev / PROVISIONAL terms

These are the **registered gate shapes, used here only as dev reference points**. A dev read is one of the section-7
interpretation words; nothing passes or fails anything, and the words GO / NO-GO / gate / verdict are not used for
this draft's own results.

| shape (registered origin) | value used as a dev reference point |
|---|---|
| premises learned (STI0) | PAB ≥ 0.30 and PBD ≥ 0.30 in `wake` |
| premise retention (STI7) | PAB/PBD(`sym`) ≥ max(0.30, 0.9 × `wake`) |
| inference written (STI1/STI6) | DI(`sym`)−DI(`wake`) ≥ 0.15; WAD rise ≥ 20 pA |
| replay necessary (STI2) | DI(`dis`)−DI(`wake`) ≤ 0.05; DI(`sym`)−DI(`dis`) ≥ 0.15 |
| structure necessary (STI3) | DI(`sym`)−DI(`wpermc`) ≥ 0.5 × the gain, gain > 0 |
| no false inference (STI5) | foil ≤ 0.05 above `wake`; never-encoded C < 0.10 |
| control validity (U6) | premises ≥ 0.30 in `wpermc`, foil ≤ 0.05 above `wake`, C < 0.10 |

## 10. Queue and compute plan

**Placement: the pool, numpy — RULING compute-placement: anything over ~10 minutes goes to `pool_queue.sh`, never
this server.** Every job below exceeds 10 minutes, so all of them run on the mini-PC pool at a pinned revision,
provisioned isolated (`bash tools/pool_provision.sh --revision <sha> --isolated <nodes>`), each line checked by
`bash tools/queue_job_shape_check.sh` and queued with `bash tools/pool_queue.sh add`. Backend numpy only (the copy
keeps the registered runner's refusal of any other backend for a recorded run); no GPU, no `gpu_queue.sh`, no local
heavy job.

Per seed, two parts (each part re-runs the same deterministic waking episodes; the U7-analogue compares their
post-waking weight hashes):

    cd ~/derisk-pool/revisions/<sha> && env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u \
      -m research.runners._ca3_symmetric_ti_symrule --seed S --arms <PART> \
      --out research/findings/raw/_ca3_symmetric_ti/parts/seed<S>_<tag>.json

| part | arms | content | est. sim steps |
|---|---|---|---|
| a | `sym wpermc rev` | encode, symmetric night (240 000 steps), raster → `wpermc` permutation + `rev`/replica passes, probes for `sym` and `wpermc` | ~0.41 M + 2 replica scans of a 240 k-step raster |
| b | `wake dis` | encode, `wake` probes, disrupted night (240 000 steps), `dis` probes | ~0.41 M |

Per-seed total ≈ 0.82 M simulation steps (nights 2 × 240 k; encodes 2 × ~37 k; probes 4 arms × 6 probes × 8 trials ×
1400 steps ≈ 269 k). Estimated from the registered family's measured numbers (prereg section 11: 3–6 ms per step on
one unloaded pool core; a ~1 M-step part took 0.5–1.7 h): **each part ≈ 20–50 min; 8 jobs total (4 seeds × 2 parts);
≈ 3–7 core-hours.** Peak memory ≈ 1.5 GB per job (the night raster is 240 000 × 1260 cells as bytes ≈ 302 MB, plus
surrogates and the engine) — trivially inside a pool node; no `memcap.sh` interplay (nothing runs on this server).

**Order:** first run the section-11 `hebbian_learning_rate` rung runs (up to 3 standalone seed-8 sym-night jobs;
their selection criteria are the non-DI section-11 ones only — prereg §2 item-7 `--devcheck` precedent: none of them
reads DI), freeze the constant, then queue the 8 standard jobs (4 seeds × 2 parts; **11 jobs total including the
rungs**), and only then inspect per section 9. The section-7 table was fixed before this and governs every read,
smoke included. Record assembly is the copy's own `--merge`-style mode (a new file's code; the registered grader is not
touched); the finding is written PROVISIONAL by the local model from those records and inspected by Claude, per the
T3-sleep-ti precedent.

## 11. Operating-point constants (chosen without reading DI; frozen before seeds 9–11)

The night's constants are the registered ones verbatim (240 000 steps, SWR every 1500 steps, 9000 pA cue on 60 % of a
non-Z assembly for 40 steps, dt 0.1 ms). The rule-scale constants for the engine's symmetric Hebbian are set
explicitly in the copy's `DEFAULTS`, with the scale mismatch of that rule's defaults declared:

- `hebbian_max_weight = 900.0`, `hebbian_min_weight = 0.0` (the store's weight space is pA; the rule's own defaults
  are 0.05–1.0 and would clip the store to nothing), `hebbian_weight_decay = 0.0` (section 14: decay is the
  downscaling companion's job, not an unregistered brake).
- `hebbian_learning_rate`: ladder **{0.002, 0.006, 0.02}**, evaluated as up to 3 standalone seed-8 sym-night rung
  jobs (counted in section 10), selecting the **largest value that satisfies the no-saturation criterion** — the
  registered section-3 style criteria that do not depend on the inference read: premises learned in `wake`; the `sym`
  night does not saturate the store (foil and never-encoded reads below 0.05 in `sym`, between-pair mean below
  0.8 × w_max). The choice and every number are logged in the finding; frozen before seeds 9–11 and before any
  section-9 read. If the ladder is exhausted without satisfying the criterion, the draft reports SPECIFICITY-FLOOD
  and stops (no further tuning).

## 12. Runner-copy build notes — the symmetric-rule delta

`research/runners/_ca3_symmetric_ti_symrule.py` is a copy of `research/runners/sleep_transitive_inference.py`. The
name follows runbook §5 rule 4's `_<topic>_*.py` pattern; RULING T3-sleep-ti's `_sti_symmetric_*` was its example
name — §5's pattern governs here (Claude confirms at review). Everything is byte-identical except the delta below;
the copy must ship a diff against the original, and **every** hunk is enumerated here: the items below plus the
arm-set reduction (the registered `off/yok/scr/cross/nohub/nostdp/nocomp/noadapt` arms are dropped from the copy),
`wperm` → `wpermc`, `rev` demoted from a gate arm to an instrument, the new instrument fields (§8), and the
gate-seed refusal strengthening in item 6. No other hunk may appear in the shipped diff.

1. **The night's rule (the whole point).** The `sym` night runs with `enable_stdp = False` and
   `enable_hebbian_learning = True`, `hebbian_symmetric = True` — the engine's own symmetric (offset-free)
   co-activity potentiation on same-step co-spikes (`sim/bridge.py:10128` compacting path;
   `enable_branchless_plasticity` stays False). **Engine-run plasticity: no host code writes a weight on the
   capability path.** Waking encoding keeps the engine's Bi-Poo STDP unchanged (the failing component per STI7/STI3b
   is the night's rule; changing encoding would also confound against the registered dev baselines). Declared
   deviation from Mishra 2016: same-step coincidence, not a ~150 ms half-width window — a `sim/` rule is out of scope
   for this draft (no `sim/` edit); see risks.
2. **Mask enforcement.** `cfg.enforce_plastic_mask_in_hebbian = True` (`sim/config.py:456`) so the Hebbian rule acts
   only on the plastic between-assembly synapses (the build already asserts the mask: between plastic, within and PV
   fixed). Per-seed check: within-assembly and PV weights byte-identical across the night, else UNDEFINED(dev).
3. **One rule per build.** The waking builds run STDP with Hebbian off (as registered); the night build runs Hebbian
   with STDP off; the config echo of each build lands in the record (a double-update would be UNDEFINED(dev)).
4. **`wpermc`** takes the `sym` night's own Δw multiset and permutes it **within classes of equal pre-night weight**
   (exact float32 equality on the post-waking weights the night starts from). The class histogram is reported. This
   is the registered `wperm` construction restricted to classes — the prereg section-10 U6 fix — applied as an
   experimenter intervention, as before.
5. **`rev` + replica.** A `replica_sym` mirroring `replica_stdp`'s structure (step-wise, incremental, on the plastic
   between synapses) implements the same symmetric step as `sim/bridge.py:10128` (potentiate `lr·(w_max − w)` on
   same-step co-spikes, decay 0, clip [0, 900]). On the un-reversed raster it must reproduce the engine's `sym` Δw
   (≤ 1e-3 pA) or the seed is UNDEFINED(dev)-instrument; on the raster time-reversed inside each SWR period it must
   give the identical Δw (the order-independence expected-pass).
6. **Guards kept:** `--dev` semantics, `cfg.seed` setting, fresh-build-per-phase, provenance sidecars (run as a
   module), and the numpy-only refusal carry over verbatim. The gate-seed refusal is **strengthened** to
   unconditional (the registered runner makes it `--dev`-conditional: `sleep_transitive_inference.py:744-745`) —
   gate seeds stay Claude's under every invocation. Output root: `research/findings/raw/_ca3_symmetric_ti/`.
7. **Reads** (DI/FI/PAB/PBD/PBP/WAD/WDA/replay) and the probe protocol are the registered functions verbatim, so the
   field names and semantics match the family the Claude grader will later score.

## 13. Risks that could make a seed UNDEFINED (or the reading empty)

- **U8, the rule never fires (highest).** Same-step co-spikes may be rare (2026-07-08: the symmetric rule formed only
  a ~1.14× weak attractor when members fire asynchronously). **The condition is pinned here: U8 fires only if total
  same-step co-spike pairs over the whole night `cospike_total` == 0.** Any nonzero count reads WRITE-ABSENT
  honestly; the raw counts are reported either way. The `cospike` instrument exists precisely for this.
- **Control degeneracy.** Exact-equality weight classes are dominated by the untouched 15.0 pA synapses (including
  A→D, which is correct — its night Δw must be movable) while premise synapses sit near-singletons. If the histogram
  shows most Δw mass unmoved, STI3 is unreadable → CONTROL-DEGENERATE.
- **Replica mismatch** (engine incremental update vs replica float32 ordering) → UNDEFINED(dev)-instrument; the
  registered `yok` check shows ≤ 1e-3 pA is achievable for the STDP rule, but this is a different rule and is checked
  fresh.
- **Mask enforcement gap.** If the enforced-mask path is not the one executed (branchless vs compacting), the
  Hebbian block could touch within-assembly weights → confound; caught by the per-seed byte-identical check.
- **Saturation / foil flood** is a real outcome (SPECIFICITY-FLOOD), not an error — it is the expected failure mode of
  potentiation-only without a brake and must not be dressed up as UNDEFINED.
- **Scale trap.** The Hebbian rule's default weight space (0.05–1.0) differs from the store's pA space; a missed
  override silently clips the store → caught by the WAD/between-pair reads and the config echo.
- **Wall-time drift.** Step estimates inherit the registered family's 3–6 ms/step; a part that exceeds its window is
  re-queued at the same pinned revision (that is a run error analogue), never tuned around.

## 14. Next step on failure (registered in prereg section 10; execution stays Claude's)

- **EROSION-PERSISTS or SPECIFICITY-FLOOD** (STI7's shape not repaired, or the foils rise): the **sleep downscaling
  companion** — the brake on the night's net potentiation
  (`research/biology/sleep-load-dependent-renormalization.md`, prereg section 10 items (2)–(3); then a stronger or
  plastic PV loop). This is the second companion of the ruling and the next family after this one.
- **WRITE-ABSENT with healthy co-spikes:** the in-engine broad-window stand-in (`hebbian_rate_window`, co-activity
  trace) as the closer approximation to Mishra's ~150 ms half-width; a rule change is a Claude amendment.
- **WRITE-ABSENT / UNDEFINED with no co-spikes (U8):** a faithful broad symmetric window needs a guarded `sim/`
  addition (a symmetric-window STDP rule) — out of scope here, registered as the work package it is. No defer: the
  capability stays open; a characterized boundary is a finding, not a stopping point.
- **ROUTE-FIRST:** read the pre-sleep WAD for encode-time integration before anything else is interpreted.

Whatever the dev reading, the deliverable is a PROVISIONAL finding from the section-7 table, inspected by Claude, who
then writes the next family preregistration with the gate seeds — RESULT first, per RULING T3-sleep-ti.
