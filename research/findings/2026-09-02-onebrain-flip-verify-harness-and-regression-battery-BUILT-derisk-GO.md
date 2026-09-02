---
type: finding
status: live
date: 2026-09-02
mechanism: onebrain-flip-verify-harness
lane: integration
claim_check: synthesis
seeds: [42]
seed-waiver: The harness de-risk is a DETERMINISTIC byte-identical reproduction of BANKED verdicts (the reference
  `_aggregate` vs the generalized `aggregate` on the same per-worker JSON) — no new stochastic measurement, so a seed
  population measures nothing; the banked verdicts it reproduces were themselves 6-seed (42/43/44/100/101/102). The
  gate is a static-code check (seedless). The regression-battery no-op demo is a single deterministic build-pair at
  seed 42 with an UNUSED sentinel flag (identical builds); its claim is decision-STABILITY across a guaranteed-no-op
  flip, not a seeded effect size.
artifacts:
  - research/findings/raw/_flip_verify_harness/derisk_xedge_reproduction.json
  - research/findings/raw/_regression_battery/battery_demo.json
  - research/findings/raw/_xedge_flip_verify/flip_verify_cupy_6seed.json
  - research/findings/raw/_xedge_flip_verify/flip_verify_cupy_6seed_indirection.json
runner: research/runners/onebrain_flip_verify_harness.py, research/runners/onebrain_regression_battery.py,
  tools/gates/flip_offarm_staleness.py
---

# One-brain flip-verify harness + shipped-faculty regression battery + OFF-arm-staleness gate — BUILT, de-risk GO

**One-line:** the one-brain INTEGRATION program's Phase-1 verification infrastructure is built and de-risked: a
reusable three-arm flip-verify harness (generalizing `_xedge_flip_production_verify.py`) that reproduces the banked
d6-WM→comprehension verdict BYTE-FOR-BYTE, the genuinely-new cross-faculty REGRESSION BATTERY (no per-faculty
flip-verify ever tested whether a flip breaks the OTHER default-on faculties), and the 2026-08-27 audit's deferred
OFF-arm-staleness GATE — which, on its first run, caught a 7th live instance of the bug the audit could not have seen.

## What was built (three modules, all CPU/numpy)

1. **`research/runners/onebrain_flip_verify_harness.py`** — generalizes the reference runner's ARM A
   (byte-identical-off), ARM B (visible-on-real-traffic through the REAL `webapp.server.brain_chat`,
   lesion-attributable, `n_hollow=0`), ARM C (no-regression) into ONE `EdgeSpec`-parameterized entry. A future flip is
   specified by its flag / `*_LEARN` / `*_LESION` flags, its real-traffic probe items, per-arm env configs, and a few
   edge-specific read-outs; the harness runs all three arms through the real handler on fresh per-(config,seed) brains.
   The concrete `xedge_edge_spec()` deliberately REUSES the reference runner's own constants + helpers (`_w0_role`,
   `_decisions_equal`, item lists) so the generalized aggregate is byte-identical to the reference for that edge.

2. **`research/runners/onebrain_regression_battery.py`** — the cross-faculty no-regression instrument the program
   named as *the single most load-bearing missing instrument*. Given a flag flipped ON-vs-OFF, it runs a representative
   deterministic probe per default-on faculty through the real `brain_chat` (each arm a FRESH subprocess build at one
   seed, so the shared background-noise trajectory is identical between arms — comparing two in-process sequential arms
   would diverge on noise) and asserts each still DECIDES identically, comparing categorical DECISION variables only
   (booleans / labels / ids), excluding continuous measurements (rates, levels, margins, firing, `ema_*`, seconds, pA)
   — the same instrument choice ARM A makes. The harness's ARM C now calls it.

3. **`tools/gates/flip_offarm_staleness.py`** (CLASS OS, BLOCKS) — the 2026-08-27 flip-soak-off-arm-staleness audit's
   explicitly-deferred `tools/gates/` follow-up. Flags a non-`*_LESION* `BRAIN_ flag popped as an OFF arm (an explicit
   `="1"` ON sibling present, no explicit `="0"` for that flag) in a research/runners soak/flip/verify runner while the
   flag's owning-module reader default resolves ON (a literal `.get(F,"1")` fallback or a `_*_DEFAULT_ON=True`
   constant). `selftest()` fails in the failing direction. Registered in `docs/FAILURE_GATE_MATRIX.md` (CLASS OS) and
   `research/FAILURE_LOG.md`.

## De-risk 1 — the harness reproduces the banked d6→comprehension verdict BYTE-FOR-BYTE (GO)

`python -m research.runners.onebrain_flip_verify_harness --derisk` feeds the BANKED xedge per-worker data to both the
reference `_aggregate` and the harness `aggregate`, and requires byte-identical verdicts. It uses banked data (no brain
builds), so the generalization is isolated from the brain. Result (`derisk_xedge_reproduction.json`), on all three
banked cupy artifacts including both a GO and a NO-GO outcome (so ARM A/B/C pass AND fail paths are exercised):

| banked artifact | banked GO | harness GO | harness aggregate == reference == banked |
|---|---|---|---|
| `flip_verify_cupy_6seed.json` | False | False | byte-identical |
| `flip_verify_cupy_6seed_strengthened.json` | False | False | byte-identical |
| `flip_verify_cupy_6seed_indirection.json` (the landed flip) | True | True | byte-identical |

`DERISK_GO = True` on 3/3 (artifact `research/findings/raw/_flip_verify_harness/derisk_xedge_reproduction.json`). The
generalization changed NOTHING for the one edge with a known-good answer. Note the `aggregate` is an INDEPENDENT
re-implementation (it does not call `_aggregate`); the byte-identical match across both GO and NO-GO artifacts, which
exercise different code paths, is what makes the reproduction load-bearing rather than trivial.

## De-risk 2 — the regression battery runs + reports per-faculty, catches a break

`onebrain_regression_battery.py` registers 38 faculties mapped to (probe turn, decision fields). The response surface
was discovered by running the real default `brain_chat` (numpy) and dumping every faculty metadata dict; each faculty's
DECISION fields are grounded in what the handler actually returns (not guessed).

- **Synthetic comparison logic (no builds):** an identical response-pair → `all_pass=True`; a deliberately-broken probe
  (one faculty's decision field mutated in the OFF copy) → caught, and ONLY that faculty is reported regressed
  (`regressed=['da-mode-drives-response']`). This proves both the all-pass path and that the detector localizes the
  regression to the responsible faculty rather than smearing it.
- **Real no-op flip through the real handler (numpy, fresh per-arm builds at seed 42):** an UNUSED sentinel flag
  (`BRAIN_REGRESSION_BATTERY_NOOP`, nothing reads it → the ON and OFF arms build identically) over the
  `well/unknown/hold/held` probe turns (a 36-faculty subset of the registry). Result `all_pass=True`: **22 faculties
  exercised and decided identically, 0 regressed, 14 not-exercised** (their driving fields need a trigger this subset
  does not supply — reported honestly rather than silently counted as covered). See
  `research/findings/raw/_regression_battery/battery_demo.json`. (This is a guaranteed-no-op — identical builds — so
  it isolates the two-arm plumbing + the all-pass path; a real answer-preserving flip would additionally exercise the
  RNG-trajectory-shift case.)

## De-risk 3 — the OFF-arm-staleness gate: selftest + a genuine live catch

`selftest()` passes (empty), demonstrating all four directions: a pop-based OFF arm on a default-ON flag IS flagged; the
explicit-`="0"` fix PASSES; a `*_LESION` flag pop is NOT flagged; a still-default-OFF flag's pop is NOT flagged. On its
first full-tree run the gate flagged exactly one live instance the 2026-08-27 audit could not have seen:
`research/runners/_wkv_mouth_open_ended_wiring_verify.py`'s FLAG-OFF arm popped `BRAIN_OPEN_ENDED_WKV_MOUTH`, which
flipped **default-ON on 2026-08-30** (`webapp/open_ended_chat.py:236` reads `.get(..., "1")`) — three days AFTER the
audit marked that file safe. So its "FLAG OFF, WKV module never imported" arm had silently been reading ON since the
flip. Fixed the same way (explicit `="0"`), matching the audit's own 5-fix discipline (not re-run end-to-end). Two false
positives found during calibration were real gate bugs, both fixed before landing: the value-set parser missed the
`os.environ[F] = "1" if on else "0"` ternary's `else "0"` (so two comprehension verifies looked OFF-arm-less), and
`*_LESION` flags had to be excluded.

## Honest residuals (what is NOT claimed)

- **The harness worker/orchestrator were not run end-to-end in this worktree.** The xedge cross-edge cannot build here
  (`data/corpus/tinystories.txt` is an untracked file present only in the main checkout, not the worktree), so the ARM
  B live-visibility path degrades to standalone organs. The load-bearing proof is the aggregate de-risk (byte-identical
  reproduction against banked data); the worker/orchestrator are faithful ports of the reference structure, verified to
  construct + parse but not driven through a full brain here. A cupy run on the main checkout is the natural follow-up.
- **The regression battery has thin coverage.** 38 faculties are registered; 22 carry a static `thin` flag (a
  conservative a-priori marking that their driving field may not populate on a generic turn). Empirically on the 4-turn
  no-op demo subset (36 faculties): 22 were exercised (fields present, all decided identically), 14 were `not-exercised`
  — their driving decision fields need a trigger this subset does not supply: a mismatch turn for `surprise-monitor`, a
  2-turn intention for `prospective-memory`, a visual percept for `vision-identity-spiking-hmax`, a between-turn idle
  tick for `self-initiated-utterance`, and the special-probe faculties (`value-driven-choice`, `bg-action-selection`,
  `worldmodel-forward`, `metacog-monitor`, `curiosity-followup`, `reconsolidation`, `episodic-memory`,
  `wm-binding-advanced`, `discourse-register`, `confidence-forthcomingness`). The full probe set adds `scalar` (drives
  `pragmatic-implicature`) and `open` (drives `open-ended-generation`) turns. The battery reports every not-exercised
  faculty as such (counted, honest, not claimed as covered); lifting a thin probe to a driving one is the mechanical
  Phase-1 follow-on named in the program plan.
- **The battery + harness compare DECISIONS, not correctness.** They are reachability + decision-stability instruments:
  they catch a flip that changes a faculty's decided output on a turn the set already drives; they cannot catch a
  regression a probe never reaches, nor prove any faculty computes the right thing.

## External methodology frame

The regression battery's core relation — a NO-OP flip must preserve every faculty's DECIDED output — is a *metamorphic
relation* acting as a pseudo-oracle: the standard external technique for testing a system whose exact correct output is
intractable to specify (you assert the input→output *relationship* across runs instead). The OFF-arm-staleness bug is
the dual failure — a control input transformation that silently became the identity, so the relation was vacuously
satisfied. See metamorphic testing (Sun et al. 2024, *Softw: Pract Exper* 54(3):394-418, doi:10.1002/spe.3280;
overview https://en.wikipedia.org/wiki/Metamorphic_testing).

## Why this matters for the program

Every merge wave and cross-edge flip in the integration program (Phases 3–5) was to be gated by "does this flip break
anything else on the roster" — a test that did not exist; ARM C checked only one faculty's fixed items. That gap is now
closed by a reusable, extensible instrument, and the discipline it depends on (an OFF arm that stays OFF after a default
flips) is now enforced by a gate rather than remembered — which the wkv catch shows was already being violated.
