# gap#5 RANK 2 (sequence replay) — the FORWARD CHAIN mechanism WORKS; the blocker is an isolated within-reactivation divergence in the driver's `_prepare_sequence` vs RANK 1's proven `_prepare`

**2026-07-22 (autonomous, coexisting with the production LM run — all CPU/numpy).** RANK 1 (single-assembly spontaneous
reactivation) is a solid 6-seed-confirming GO (see the RANK 1 finding). RANK 2 is the next SWR-replay rung: a stored
A→B→C sequence that spontaneously replays in FORWARD ORDER. Built by a subagent (`_gap5_sequence_replay_derisk.py`);
I de-risked it and reached a clean two-part result — one part GO, one part an isolated, precisely-characterized blocker.

## PART 1 — the FORWARD-CHAIN mechanism WORKS (GO on the encode side)
The directional A→B→C chain forms correctly via theta-compressed BTSP sweeps. A 3-way isolation (encode-only, seed 42):
- **forward-only sweeps (24f,0r): w_fwd=6.77, w_rev=0.50, asym=+6.27** — clean asymmetric forward chain.
- reverse-only (0f,8r): w_fwd=0.50, w_rev=6.75, asym=−6.25 — clean reverse.
- combined (24f,8r): w_fwd=5.00, w_rev=6.75, asym=−1.75 — reverse-BIASED.
Root cause of the reverse-bias: the reverse sweeps run LAST → heterosynaptically depress the forward edges (6.77→5.00)
while potentiating reverse. A **temporal-order artifact, not a mechanism failure** — `chain_rev=0` gives a clean forward
chain (verified asym=+2.53 even at the full smoke). ⇒ the traveling-wave A→B→C hand-off, the genuine open question the
research gate flagged, is realizable in the recurrent weights.

## PART 2 — the BLOCKER: the sequence assemblies don't spontaneously REACTIVATE (0 rest-phase events), isolated to a `_prepare_sequence` code-path divergence
Every corrected forward-chain smoke produced `asm_active=[0,...]`, 0 events — no assembly self-completes under weak noise.
Isolated cleanly: the RANK 2 driver at RANK 1's **EXACT single-assembly config** (n_mem=1, ca3_density=0.05,
structural_sep=1, within_events=30) STILL gives 0 reactivation — while RANK 1's OWN driver reactivates (memb~0.31,
3-6 events/seed) at that identical config.
- **Configs are byte-identical** (compared line-by-line: density, apical_kir_g=3.0, apical_gc=1.0, apical_gc_read=5.0,
  k_thresh=18, plateau_strength=120, coact_thresh=0.02, ca3_fb_inhib=20, apical_R=50, plateau_v_hold=−35,
  recall_k_thresh=40, all BTSP encode params). Same assembly draw (same rng seed*17+3). Same w_within=5.0.
- Two hypotheses REFUTED empirically: (a) ca3_density — RANK 1 reactivates at 0.05, NOT the 0.35 I wrongly assumed from
  the *two-assembly co-storage* arc (self-corrected); (b) plateau self_regen during encode — splitting it (bistable 0.15
  within-phase / transient 0 chain-phase, matching RANK 1) left w_within=5.0 UNCHANGED and still 0 reactivation, so the
  self_regen-during-encode does not affect the within weight (BTSP potentiates during the drive regardless).
⇒ the divergence is in the `_prepare_sequence` within-encode **code path** (the subagent's reimplementation), NOT any
config/weight/assembly value. Not quickly spottable by reading.

## NEXT STEP (clear, honest — per THE LAW a blocker is a method verdict, not a capability wall)
Make `_prepare_sequence` **reuse RANK 1's proven `_prepare` for the within-encode** (import + call it to build the
bistable self-completing assemblies) and layer ONLY the chain phase on top — eliminating the reimplementation divergence
by construction. Then re-verify n_mem=1 reactivation → n_mem=3 forward replay (chain_rev=0) → anti-cheats → 6-seed.
Delegated as a focused follow-on. The RANK 1 spontaneous-reactivation GO stands as the solid gap#5 deliverable this cycle;
RANK 2's forward-chain mechanism is validated and the reactivation blocker is reduced to a known encode-reuse fix.

Additive default-preserving driver improvements committed: `--ca3-density`/`--structural-sep` CLI flags; the
within-bistable/chain-transient self_regen split (conceptually correct, harmless — not the fix). NO `sim/` edit.
