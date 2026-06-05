# De-risk (A) RESOLVED — NEF thresholded cleanup reaches seed-robust numpy parity → GO — 2026-06-05

**Verdict: GO.** The literature-grounded NEF thresholded cleanup (Stewart-Tang-Eliasmith 2011, the Spaun spiking VSA
cleanup) reaches **seed-robust numpy parity** on the composer's real noisy unbind est: worst-case **0.978**, mean
**0.993** across seeds 42/43/44 at a FIXED operating point — clearing the ≥0.95 bar that defeated three prior
mechanisms. The numpy `argmax` cleanup CAN be replaced by a fully-spiking circuit without regressing the capability
matrix. This resolves the (A) readout shortcut at the mechanism level; the composer integration is the next build.

## The arc (parameter-guessing → literature-grounded)
Owner steer (after the divisive-norm + hand-WTA attempts stalled): stop guessing; ground the cleanup in the science.
A deep-research synthesis (`2026-06-05-spiking-cleanup-memory-literature-synthesis.md`) diagnosed all three failures
and prescribed the NEF thresholded cleanup. It worked first try at the mechanism level and cleared seed-robustness
with the synthesis's own robustness lever (more neurons/concept).

| approach | seed-robust worst-case (min across 42/43/44) | mean | note |
|---|---|---|---|
| matched filter + temporal integration | ~0.78 | — | rate readout = linear reconstructor, not argmax |
| + Carandini-Heeger divisive norm | 0.844 | 0.904 | Krotov-Hopfield Model-C (L2, n=2) plateau |
| two-stage (input + output divisive norm) | 0.911 | 0.926 | input-norm helps but graded readout caps it |
| hand-tuned hard-WTA | 0.13-0.16 | — | violated Rutishauser α>1 stability |
| **NEF threshold, n_per=6** | 0.889 | 0.948 | broke the plateau: **per-seed 1.000**; seed-44 laggard |
| **NEF threshold, n_per=12** | **0.978** | **0.993** | **GO** — more neurons closed seed-44 (0.889→1.000) |
| numpy oracle | 1.000 | 1.000 | — |

## The mechanism (literature-grounded, NO sim/ edits)
`research/findings/raw/_spiking_cleanup_nef.py` — feed-forward, no recurrent WTA:
1. **Input normalization** — a spiking inhibitory-trait FS pool shunts the est ON/OFF input population so the
   matched-filter drive is ~scale-invariant (≈ cosine) across seeds. (This is why our earlier input-norm helped
   0.844→0.911; here it makes the THRESHOLD seed-invariant — the load-bearing role.)
2. **Matched filter** — `n_per` neurons per concept, encoders = the stored codes (codes as ON/OFF receptive fields).
3. **THRESHOLD PLACEMENT** (the key change) — a negative concept bias places the firing intercept so that off-target
   concepts (cosine ~0) emit **ZERO spikes** and the true concept (cosine ~0.31) fires. The neuron's own rectification
   nonlinearity IS the cleanup — the discretization a graded rate readout structurally cannot do (Stewart-Eliasmith:
   a rate population is a linear reconstructor `Σ a_i φ_i`, never an argmax; off-target leak was the 0.91 cap).
4. **Clean per-concept readout** — sum each concept's `n_per` neurons' firing → argmax. Clean because off-target is
   silent. No graded output divisive norm.
5. **Input-driven** (Betteti 2025) — the est is clamped throughout the readout window (the winning basin self-deepens).

**Operating point (seed-robust):** bias=-625 (or -700), w_match=120, **n_per=12**, w_in_cfs=1.0, w_in_fs=10, n_in_fs=60,
einh=-80, run_steps=400, proj_dim=800, V=320. Per-seed: bias=-625 → {42:0.978, 43:1.000, 44:1.000}.

**Why n_per mattered:** at n_per=6 the spiking threshold-crossing is noisy → seed 44 = 0.889; at n_per=12 the
population averages the noise → seed 44 = 1.000 (the synthesis's "representational error ∝ 1/N_neurons", Stewart-
Eliasmith use ~10+). The remaining knob is purely the neuron count, not a fragile tuning.

## Next: build the NEF cleanup into the composer (Task 2)
Replace `np.argmax([concepts[w]·est])` in `core_sim_composition.unbind` / `_render_filler` with the spiking NEF
cleanup: the composer holds a persistent NEF cleanup bridge built once from its codebook; each unbind drives the est
through it and reads the per-concept firing. Opt-in flag (`enable_spiking_cleanup`) so the numpy path stays the
default until the build is validated. **No-regression GATE:** the full capability matrix (flat / one-attr / two-attr
/ negation) at production D=2048 multi-seed must match the numpy path (the de-risk was at proj_dim=800, the harder
noisier regime; D=2048 is cleaner so parity is expected — the matrix validation confirms it at production). If the
matrix holds with the spiking cleanup → (A) the readout shortcut is genuinely CLEARED, fully spiking.

## RESOLVED — built into the composer + production D=2048 validation: (A) CLEARED

The NEF cleanup is now wired into the composer (commit 18352657, `research/runners/core_sim_composition.py`):
opt-in `enable_spiking_cleanup=False` flag; when True, the composer builds ONE persistent NEF cleanup bridge from its
own codebook at init (operating point `NEF_CLEANUP_OP` = the validated bias=-625/w_match=120/n_per=12) and routes
`unbind` + `_render_filler` through it (a new `_cleanup` helper; the 2-code AFFIRM/NEGATE polarity sub-codebook falls
back to numpy). The numpy default path is **byte-unchanged** (the `else` branch is the original
`argmax([concepts[w]·est])`); the 10 on-brain tests stay green; NO `sim/` edits.

**Production no-regression validation (`_nef_composer_validate.py`, D=2048, V=320, seeds 42/43/44):** the
spiking-cleanup composer answers the capability matrix (flat who/what, abstention, one-attribute, negation/yes-no,
generation) **IDENTICALLY to numpy — 27/27 match across all three seeds → GO.** The de-risk was at the harder
proj_dim=800 (noisier est); D=2048 is cleaner, and the operating point transferred cleanly (the input normalization
makes the threshold scale-invariant, as the synthesis predicted), so production parity held with no retuning.

**(A) the readout shortcut is CLEARED:** the composer's numpy `argmax` cleanup has a validated, fully-spiking,
biologically-grounded replacement (Stewart-Eliasmith NEF cleanup) at production-scale numpy parity, multi-seed,
no regression. It is opt-in (the grounded conversational agent enables it; the numpy path stays the fast default for
non-grounded uses). The deeper **(B) memory shortcut** (the numpy-held bound fact + numpy superposition/opponency) is
the remaining piece — options drafted in `docs/plans/2026-06-05-composer-B-substrate-held-memory-options.md`.

## Artifacts
- `research/findings/raw/_spiking_cleanup_nef.py`, `_nef_multiseed.json`, `_nef_nper12.json`,
  `_nef_composer_validate.py` + `_nef_composer_validate.json` (production validation, 27/27)
- Composer integration: `research/runners/core_sim_composition.py` (commit 18352657)
- Synthesis: `2026-06-05-spiking-cleanup-memory-literature-synthesis.md`
- Backend: CuPy / RTX 3090.
