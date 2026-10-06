# Amendment A0 to the long-delay credit prereg (Claude, 2026-10-04)

Amends `research/findings/2026-09-29-long-delay-credit-bg-organ-PREREG.md` (on `research/long-delay-credit-spiking`
and `research/draft-ldc-outcome-timing`; not yet on main, so the amendment is recorded here and the prereg text is
not edited). Input: the Step 4a pilot, `docs/drafts/long-delay-credit/PILOT.md` @ 4e6b19ec2 on
`research/draft-ldc-outcome-timing` (dev seeds only, not evidence).

## The rule A0 applies (fixed in the HANDOFF before the pilot ran)

"If no onset of 300 ms or less signs the outcome DA (unrewarded min <= 0.47 with |ITI offset| <= 0.005), the
striosome-GABA_B critic on this organ is banked as a finding and the next method is a timed omission route (LHb ->
RMTg GABA onto SNc); the capability stays. If DA is signed but sanity/credit_immediate stay below 0.8 on most seeds,
the next method targets the eligibility tag's action-specificity."

## Ruling: the outcome DA is NOT signed; the timed omission route is next

- One cell of 24 passes all three bars: credit_immediate, onset 150 ms, w_outc_strio 60, seed 11, unrewarded min
  0.468728 against the 0.47 bar, a margin of 0.0013. The same cell on seed 8 reads 0.471890 and fails by 0.0019. The
  margin is smaller than the spread between the two seeds in that one cell (0.0032), so the pass is a draw at the
  bar, not a signed DA. The verify-go rule applies: an effect inside the seed-to-seed spread is not an effect.
- Across the grid the striosome dip is shallow by construction. At w_outc_strio 30 every unrewarded min sits at
  0.4977 to 0.4983 against a 0.5 baseline. At 60 the dip deepens to about 0.47, and on late_immediate it also opens
  an ITI leak of -0.11 to -0.15. The only lever that deepens the dip breaks the I bar. That is the shape of a
  slow GABA_B projection (tau 150 ms, w 4.0) asked to produce a timed pause.
- So the A0 condition "no onset signs the DA" holds. The striosome-GABA_B critic on this organ is banked as a method
  finding: it learns the outcome value but cannot by itself pause the SNc in the outcome window. **The long-delay
  credit capability stays open.**
- The companion process the organ replaced with a constant is the habenular omission pathway. In the animal the
  pause on an omitted reward comes mainly through the lateral habenula and the RMTg's GABA_A projection onto the
  dopamine cells, while the striosome-GABA_B path carries a slower value signal (Matsumoto & Hikosaka 2007;
  Jhou et al. 2009; Hong & Hikosaka 2008). The project has a working spiking omission detector of exactly this
  motif, at 6/6 (`research/findings/2026-08-01-curiosity-reward-omission-veto-spiking-circuit-6seed.md`): the
  delivered reward drives a GABAergic relay that silences a tonic LHb-like pool. Step 4b times it to the outcome
  screen and routes it onto the SNc.

## Two instrument findings carried into Step 4b (not methods, not verdicts)

- **Seeds 9 and 12 do not clear memory at the default clear dose** (32.27 / 28.64 Hz after the clear). Each of
  `w_clear_fs 60`, `CLEAR_MS 500` and `clear_pA 1200` clears to 0.00 Hz with hold 1.000 on both seeds, and each
  weaker setting is worse. The clear is a fixed dose, not a per-seed pathology. Step 4b selects one stronger dose on
  dev seeds by a fixed order, before any credit cell runs.
- **Seed 14 never explores**: p(ch0|A) = p(ch0|B) = 1.000 at lr 0. The bias is ch0 over ch1, not A over B, so it is
  a motor-channel asymmetry, not a context one. Step 4b records its source (per-channel drive and first-spike
  latency at lr 0). It is not fixed here. A seed that never explores cannot show credit, and that seed is reported
  UNDEFINED for credit, never a FAIL.

## What does not change

The registered runner and organ, the prereg's bars and gate seeds, and Step 4's banked sweep. Every Step 4b run is a
draft build on dev seeds, in pilot copies. Nothing from Step 4b is evidence for the gate. A route that works enters
the registered organ only through a later Claude amendment (A1), with its own byte-identical-off test.
