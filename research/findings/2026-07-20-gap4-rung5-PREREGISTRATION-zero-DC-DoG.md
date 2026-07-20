# gap#4 RUNG 5 — PRE-REGISTRATION: zero-DC difference-of-exponentials (filed BEFORE the run)

**Filed 2026-07-20, before any rung-5 result exists.** Seeds **400-405**, never used.
Parameters DERIVED from the zero-DC condition, not fitted: `tau_slow = 3000 ms`, `a_dep = 0.4444`
(`a_dep = [tau_p(1-e^{-W/tau_p})]/[tau_d(1-e^{-W/tau_d})]`, tau_p=1000, W=4000).

## Why this mechanism and not another band

Two pre-registered band attempts failed, and the cap fired. The diagnosed reason was NOT that eligibility magnitude
fails to encode lag (it encodes it cleanly: **corr(eligibility, lag) = -0.9445**) but that **at this geometry the
adjacent field and the lag where the field forms are the SAME lag** (spacing 4 bins == backward shift 4-6 bins).
No band in eligibility space can separate what lag space does not separate.

The zero-DC rule **does not attempt lag selection at all**. It drives the update with the signed difference of a
fast and a slow trace; a kernel with zero DC gain cannot build a pedestal, algebraically. The geometric collision
therefore does not apply to it.

## ⚠️ THE CAVEAT, STATED BEFORE THE RUN — this mechanism may improve the WRONG axis

Measured on the built kernel: the DoG potentiates at lag 0 and **depresses at LONG lag (~2800 ms)**. Its contrast
comes from *never building a pedestal*, NOT from carving an adjacent trough. But the measured deficit is
**adjacent-only** — far contrast is already healthy at 2.60x while adjacent sits at 1.21x.

**So there is a real, named possibility that this mechanism lowers the pedestal, improves far contrast further, and
leaves adjacent contrast untouched.** If that happens it is a **FAILURE on the stated goal**, not a partial success,
and it will be recorded as one. Writing this down now removes the option of reinterpreting a null as a win on a
different axis later.

## PRE-REGISTERED PREDICTIONS

0. **P0 — stage 1 survives:** `map_ok = 1` on >= 5/6. *(Both band attempts died here.)*
1. **P1 — adjacent contrast rises (THE GOAL):** contrast vs the ADJACENT field goes from 1.213 to **>= 1.60x** on >= 5/6.
2. **P2 — far contrast not sacrificed:** contrast vs the FAR field stays **>= 2.0x** on >= 5/6.
3. **P3 — the pedestal is actually lower:** mean layer-2 weight moves closer to the BTSP equilibrium than the
   rule-OFF arm's 1.331x above it, on >= 5/6. *(This is the mechanism's own claim; it can pass while P1 fails —
   which is exactly the caveat above.)*
4. **P4 — rule-OFF reproduces:** the `P4_ruleOFF` arm returns 1.213 / 2.609, 6/6.

**FALSIFIED if** P0 fails (the rule breaks field formation like the bands did), or **P1 fails while P3 passes**
(the mechanism works as designed but on the wrong axis — the outcome flagged above).

## The bar, restated

Weight contrast 1.73x currently yields only 1.09-1.21x response contrast; the transfer eats ~1.5x. P1's 1.60x
response target therefore implies **>= 2.5x adjacent weight contrast**.

## Cap

As with rung 4: **one derivation.** `a_dep` and `tau_slow` come from the zero-DC condition and are not free
parameters. If this fails, I do not re-derive them — the next step is the remaining ranked candidate
(Rank 4 mean-subtracted increment), not a second DoG parameterization.
