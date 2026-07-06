# objrel isolation control — DEEPER-RESIDUAL: neither frequency-rebalancing NOR a class-margin recovers objrel under shared-3-way BPTT → a REACHABILITY residual that MOTIVATES the closure's per-role decomposition

**Date:** 2026-07-06
**Runner:** `research/runners/_rungB1c_objrel_freq_vs_geometry_isolation_derisk.py`
**Raw:** `research/findings/raw/_rungB1c_objrel_freq_vs_geometry_isolation.json`
**Verdict:** DEEPER-RESIDUAL (the diagnostic value: rules OUT frequency + margin, motivates the architectural fix).
**Scope:** an ISOLATION CONTROL (BPTT on the SHARED 3-way Dale-legal read), NOT the deliverable. Its biological equivalent (salience-DA) + the actual closure (per-role decomposition) are separate.

## Result (6-seed-blind, all anti-cheats held: genuinely spiking + Dale-legal + held-out + no-spike-collapse)
- **Stage A (frequency rebalancing):** objrel-slot0 by oversample factor — 1× (DANN baseline) 0.00, **7× 0.01, 14× 0.01**. Canon regressed (`canon_ok=False`). ⇒ frequency-balancing does NOT recover objrel; it hurts canon.
- **Stage B (LDAM class-margin on balanced 7×):** objrel-slot0 by margin — **{0.0: 0.014, 0.5: 0.014, 1.0: 0.0, 2.0: 0.0}**. ⇒ a decision-boundary margin does NOT recover objrel either.
- **Verdict:** the residual is NEITHER a pure frequency wall NOR a frequency+geometry (margin-fixable) wall — it is a **REACHABILITY residual**: BPTT (gradient descent) on the shared 3-way Dale-legal read cannot reach the minority signed-THEME direction even with balance + margin. The analytic Dale reference still proves the read EXISTS in weight space (adversarially verified, `2026-07-06-objrel-analytic-reference-adversarially-VERIFIED-*.md`), so this is an OPTIMIZATION/reachability residual, not a representation/substrate/Dale wall.

## Why this is diagnostic value (not a wall) — it MOTIVATES the closure's architecture
The isolation tested the SHARED 3-way NONLINEAR read + BPTT + two levers (frequency, margin). It RULES OUT both levers and localizes the residual to REACHABILITY in the shared basin (the majority-AGENT basin + the see-saw the 3-way pooled read suffers). The verdict itself points to the fix: *"the biological closure must go beyond salience + margin (e.g. staged eligibility, or an explicit minority-direction PRIOR)."*

The closure de-risk (#1) makes EXACTLY that architectural change: **per-role BINARY Dale-legal detectors** — each role gets its OWN detector (E-path + its own inhibitory relay carrying that role's negative rows), so the minority THEME direction is learned INDEPENDENTLY, not competing in a shared 3-way majority basin. That per-role decomposition IS the "explicit minority-direction prior" the isolation verdict calls for. The closure also uses a per-role LINEAR delta rule (the isolation was shared-NONLINEAR-BPTT), which for a single binary unit reaches the LMS/discriminant directly. ⇒ the isolation's DEEPER-RESIDUAL (shared-3-way-BPTT can't reach it) is CONSISTENT with + motivates the closure's per-role-linear approach; they are different levers, and the closure's dev-probe (42/43/44) suggested the per-role decomposition reaches held-out objrel.

## Note (adversarial-verify follow-through)
The isolation confirms two levers (frequency, margin) are NOT the fix for the SHARED read. The closure's claimed fix (per-role decomposition + salience-weighted delta) is a DIFFERENT lever; its own adversarial-verify (agenda recorded in AUTONOMOUS_STATE CYCLE 931e) must still settle: is salience genuinely load-bearing (or is the per-role DECOMPOSITION the whole fix, salience incidental?), is it emergent (starts at chance + rises), and is the framing honest (a salience-weighted teaching-signal delta rule on per-role Dale-legal-spiking detectors, NOT reward-RL basin-immunity). The isolation's finding sharpens that: since neither frequency nor margin works on the shared read, if the closure succeeds the credit is per-role decomposition (± salience), NOT reward-modulation-per-se.

## Files
- `research/runners/_rungB1c_objrel_freq_vs_geometry_isolation_derisk.py` — the isolation control (Stage A oversample sweep + Stage B LDAM margin; reuse-by-import; NO sim/ edit).
- `research/findings/raw/_rungB1c_objrel_freq_vs_geometry_isolation.json` — 6-seed-blind Stage A + Stage B records + anti-cheats.
