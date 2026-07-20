# gap#4 RUNG 5 — RESULT: the zero-DC DoG FAILS. The zero-DC property does NOT hold as deployed.

Pre-registered at `672c94d6` before the run, seeds 400-405, parameters derived (`tau_slow=3000`, `a_dep=0.4444`).

## Scored against the pre-registration

| prediction | bar | result |
|---|---|---|
| **P0** stage 1 survives | >= 5/6 | **3/6 — FAILED** |
| **P1** adjacent >= 1.60x | >= 5/6 | **unmeasurable / invalid** (see below) |
| P2 far >= 2.0x | >= 5/6 | FAILED (0.40-0.63 where map formed) |
| P3 pedestal lower | >= 5/6 | FAILED (pedestal is HIGHER — see below) |
| **P4** rule-OFF reproduces | 6/6 | **6/6 — PASSED** |

## ⚠️ The trap P0 was written to catch, caught

Three seeds reported `c_adj` of **3.085, 1.642, 2.010** — all at or above the 1.60 target, with `map_ok=1`.
Read alone, that is "P1 passes on 3/6 and is trending". **It is worthless**, because stage-2 `dw = 0 on all six
seeds`: no layer-2 learning occurred at all, so those contrasts describe an unlearned read-out. Had P0 and `dw`
not been reported alongside, this run would have produced a plausible, entirely false positive.

## Diagnosed: the DoG is not inert — it is MASSIVELY over-potentiating

Direct instrumentation on a real lap (seed 400):

| arm | stage-1 dw | fast trace max | slow trace max |
|---|---|---|---|
| DoG OFF | +206,768 | 0.022659 | — |
| **DoG ON** | **+602,632** (~3x) | 0.022659 | **0.008061** |

drive = `fast - 0.4444*slow`: **180 synapses positive, 68 negative** — overwhelmingly potentiating.

**So stage 1 saturates the weights to `w_max`, which explains stage-2 `dw = 0` exactly: there is no headroom left
to move.** The mechanism is not inert; it is running away.

## ROOT CAUSE — my kernel validation used idealized traces that the implementation does not produce

I verified the zero-DC property in isolation and it passed (`sum(dw) = +0.000381`). **That test fed the kernel
EQUAL-AMPLITUDE idealized profiles** — `ef = 0.023*exp(-lag/tau_p)`, `es = 0.023*exp(-lag/tau_d)`.

The real traces are normalized EMAs, and their per-spike increments differ by construction:
`(1 - lam_fast) = 1e-3` vs `(1 - lam_slow) = 3.33e-4`. So the slow trace reaches only **0.008061** against the
fast trace's **0.022659** — an amplitude ratio of **0.36**, not 1.0. With `a_dep = 0.4444`, the subtracted term
`0.4444 * 0.008` = 0.0036 cannot cancel a DC component of order 0.02. **The zero-DC condition was derived and
validated for a trace pair the implementation never generates.**

That is the same instrument-validity failure as earlier today, one level up: I validated the *kernel* correctly and
never checked that its *inputs* matched the assumption. A kernel test with synthetic inputs is not a test of the
deployed mechanism.

## Honoring the cap

The pre-registration stated: *"one derivation ... if this fails I do not re-derive them — the next step is the
remaining ranked candidate (Rank 4 mean-subtracted increment), not a second DoG parameterization."*

The corrected `a_dep` is derivable (it must absorb the 0.36 amplitude ratio, giving `a_dep ~ 1.23` rather than
0.4444). **I am recording that and NOT acting on it.** The cap exists because "one more parameter value" is exactly
how a mechanism gets fitted to its outcome, and the fact that I can now name a value that would plausibly work is
precisely when the cap is load-bearing rather than decorative. A corrected DoG is available to a FUTURE
pre-registered test; it is not this one.

**Next per the cap: Rank 4 — mean-subtracted increment (Miller-MacKay), where `sum_j dw_ij = 0` holds by
construction with NO free parameter to mis-derive.** That property is structural rather than analytic, so it cannot
fail the way this one did.

## What stands

- `P4_ruleOFF` reproduced **1.213 / 2.609 on 6/6 seeds** across three separate runs today (rung 4, 4b, 5). The
  contrast asymmetry is the most robust measurement in this arc.
- The blocker remains adjacent-band contrast. Two mechanism families have now failed pre-registered tests:
  eligibility-magnitude band selection (geometric collision) and zero-DC subtraction (trace-amplitude mismatch).
