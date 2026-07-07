# GNW Rung-2 — competitive access: MUTUAL EXCLUSION is robust (6-seed GO — the shared inhibition enforces one-content-at-a-time, load-bearing), but the salience-SELECTED winner + causal-swap membership test are PHASE-ERRATIC on the synchronous limit cycle → the named next mechanism is an async attractor + adaptation-based eviction

**Date:** 2026-07-07
**Runner:** `research/runners/_gnw_rung2_competitive_access_derisk.py` (incumbency protocol; reuse-by-import of the Rung-1 assembly-loop builder + snapshot-restore reset; NO `sim/` edit).
**Verdict:** 6-seed GO (42/43/44/100/101/102) on the ROBUST property (mutual exclusion + inhibition load-bearing); the salience-graded takeover + causal-swap membership test are an honestly-characterized PARTIAL with a named next mechanism. Builds on the Rung-1 ignition GO (`2026-07-07-GNW-rung1-spiking-ignition-6seed-GO.md`).

## What is ROBUST (6-seed GO)
Two self-recurrent assemblies (A, B) in one `workspace` region, sharing one inhibitory `workspace_fs` pool. Using the **incumbency protocol** (ignite A alone → let it hold → challenge with B at a swept drive):
- **MUTUAL EXCLUSION — never co-ignition** (all 6 seeds, across the whole challenger sweep): the two assemblies are NEVER both ignited at once. Only one content can occupy the workspace at a time — the core GNW single-content-access property (Baars 1988 "one spotlight at a time"; Dehaene-Changeux 2011 metastable single ignition).
- **The shared inhibition is LOAD-BEARING** (all 6 seeds): the lesion (fs→workspace weight → 0) lets BOTH assemblies ignite simultaneously → the mutual exclusion is caused by the shared inhibition, not by anything else.
- ⇒ `mutual_exclusion ∧ inhibition_load_bearing` = GO on all 6 seeds.

## What is SCOPED-PENDING (honestly not yet achieved — the phase-erratic part)
The three refinements that would complete competitive access are phase-erratic / seed-dependent (NOT gated; reported in `scoped_pending`):
- **The incumbent stably holding a weak challenger** — mostly holds, but a weak challenger can sometimes knock the incumbent out to NONE (mutual annihilation) rather than A cleanly holding (a_holds_weak robust on only 1/6).
- **A clean salience-graded takeover** — a strong challenger CAN take over on some seeds, but the takeover threshold is non-monotonic in the challenger drive (B wins at some drives, A holds at others).
- **The CAUSAL-SWAP membership test** (the GWT paper's key test: swap the salience → flips the reported/ignited content) — clean on only 1/6 seeds.

## Root cause + the named NEXT mechanism (this is a boundary that launches the next build, NOT a wall)
The ignited state is a **synchronous period-3 limit cycle** (Rung-1's adversarial-verify finding: all assembly neurons fire in lockstep every 3rd step). A challenger pulse lands on an arbitrary phase of the incumbent's cycle → whether it captures the basin is phase-erratic. Two operating-point searches (documented below) confirmed the tension:
- **Deterministic (synchronous):** never co-ignition, but takeover is phase-erratic.
- **Async (heterogeneity + OU noise → breaks the lockstep):** the incumbent holds cleanly, BUT a strong challenger then CO-IGNITES (needs stronger inhibition); stronger inhibition (fs 48+) over-stabilizes the incumbent so it locks in forever; no clean fs window gives async + single-winner + evictable.
The missing ingredient is **ADAPTATION / FATIGUE on the incumbent** — the Dehaene-Changeux (2011) metastability mechanism by which an ignited workspace state is "destabilized" and "spontaneously replaced by another": an ignited assembly must FATIGUE over time so a more-salient challenger can displace it. My assemblies have no adaptation (homeostasis/spike-frequency-adaptation OFF for the clean Rung-1 bifurcation), so an established attractor is either un-evictable (locks in) or annihilates. **The named next mechanism (Rung-2b, folding in the Rank-4 eviction property): async rate attractor (heterogeneity + low OU noise — both already plumbed as `build_competitive_bridge` params) + spike-frequency adaptation on the workspace assemblies**, so the biased competition is smooth and an established content is displaceable by salience → the clean takeover + causal-swap membership test.

## Operating-point search (for the record — the tuning that mapped the tension)
- fs (shared-inhibition strength) sweep {8,12,16,24,32,48,64}: fs=16 is the WTA window in the incumbency protocol (weaker → incumbent never yields; stronger → over-suppression/lock-in).
- Challenger-window length {35,70,105 steps}: does NOT monotonize takeover (still phase-erratic).
- Heterogeneity + OU noise {0–120 pA}: desynchronizes (incumbent holds cleanly, A=0 stays silent), but strong challengers co-ignite unless fs is raised, and raising fs over-stabilizes → no clean window without adaptation.

## What this establishes toward the unified spiking GNW
Rung-1 gave the ignition primitive; Rung-2 shows the SECOND GNW property in its robust core — the shared inhibition enforces that **only one content occupies the workspace at a time** (single-content access), load-bearing. The salience-SELECTED winner (which content wins) is the pending refinement, precisely diagnosed to need adaptation-based eviction — the next concrete build.

## Files
`research/runners/_gnw_rung2_competitive_access_derisk.py`; `research/findings/raw/_gnw_rung2_seed{42,43,44,100,101,102}.json` (6-seed).
