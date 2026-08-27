---
type: finding
status: go
date: 2026-08-26
mechanism: ACTIVITY-SILENT WORKING MEMORY production organ (Mongillo/Barak/Tsodyks 2008) — a maintenance-mode SWAP on the anaphora referent store that holds the discourse FOCUS in short-term synaptic FACILITATION (cp_stp_u) across an intervening distractor turn (delay genuinely SILENT), then reactivates it on the next temporal-recall query by a NONSPECIFIC ping; wired behind BRAIN_SILENT_WM (default OFF)
lane: working memory / conversation (activity-silent WM — hold a discourse referent across a distractor via facilitation, beside the D6 active buffer)
verdict: WIREABLE / GO — the organ reuses-by-import the 6/6-GO de-risk `_activity_silent_wm_ping_derisk.ActivitySilentWM` (no reimplementation, NO sim/ edit). 6-seed production soak (42/43/44/100/101/102, numpy-CPU): INTACT reactivates the correct focus (rotated dog/cat/bird — specificity) with ensemble-mean ping margin +8.6..+21.9 and the delay VERIFIED SILENT (persistent-attractor path suppressed by the de-risk w_rec=60); the FAIR facilitation LESION (tau_f~5, excitability-matched) ABSTAINS on 6/6 (recovered=None, margin regresses to -3.3..+3.5); the rendered reply genuinely CHANGES intact<->lesion (correct anaphor "earlier we were talking about the dog" vs "I don't recall...") -> load-bearing not hollow; FLAG-OFF (BRAIN_SILENT_WM unset) + out-of-scope turns (no temporal-recall query, incl. a D6-style hold-query) return None -> byte-identical; the read-out abstains rather than confabulate (no-confab gate) and never manufactures a fact / flips an abstain (moat preserved). The underlying de-risk still reads 6/6 GO on these seeds (reactivation 0.72..0.90 vs FAIR control 0.15..0.40).
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_activity_silent_wm_production_soak.py
organ: research/runners/activity_silent_wm_production_organ.py
artifacts:
  - research/findings/raw/_activity_silent_wm_production_soak.json
  - research/findings/raw/_activity_silent_wm_ping.json
depends_on: 2026-08-10-parallel-push-results-activity-silent-WM-GO-NE-gain-positive-CA3-transmission-refuted-3rd.md
---

# Activity-silent WM wired to production: a nonspecific ping recovers the silently-held anaphora focus across a distractor turn (lesion-load-bearing), behind BRAIN_SILENT_WM default-OFF

## The faculty and the swap

The live anaphora referent store (`MultiTurnAgent.wm`, a `SpikingLoopContextBuffer`) holds the discourse focus in a
PERSISTENT-ACTIVITY attractor — it must keep FIRING to remember. Biology's alternative (Mongillo, Barak & Tsodyks 2008,
*Science*; Stokes activity-silent WM) holds the item in short-term synaptic FACILITATION (`cp_stp_u`) with the assembly
SILENT, reactivated by a NONSPECIFIC ping (a uniform pulse carrying no item identity). This wire-in is that alternative
as a production maintenance mode: a discourse referent is held across an intervening distractor turn in the facilitated
recurrent synapses, delay genuinely silent, and read back on the next referential turn by the ping.

## Reuse-by-import, no sim/ edit

The whole spiking mechanism is the adversarially-verified de-risk `research/runners/_activity_silent_wm_ping_derisk.py`
(the 6/6 GO `2026-08-10-parallel-push-results-activity-silent-WM-GO-...md`): K=4 isolated excitatory assemblies with
within-assembly recurrent E->E (STP ON, `stp_tau_f=1500`, `stp_tau_d=200`, `w_rec=60` sub-self-sustaining so the delay
is SILENT), `ActivitySilentWM.load(k)` -> `.delay()` -> `.ping()`. The organ
(`research/runners/activity_silent_wm_production_organ.py`) only BINDS discourse-referent strings to assemblies, drives
the silent delay per intervening turn, and surfaces the reactivation as an honest read-out. The de-risk constants
(W_REC/STP_U/TAU_F/TAU_D/DELAY_STEPS/PING_*) are imported and NOT re-tuned.

## The named RISK, tested explicitly: the "silent" claim does not leak

Two things had to hold or the claim is empty. (1) **The persistent-attractor path must be suppressed.** It is, by the
de-risk's `w_rec=60` — sub-self-sustaining, so the assembly cannot hold itself firing without drive and the delay falls
silent. The organ reports `silent_delay` on every recall; the soak asserts it, and it is **True on 6/6 seeds** (delay
firing < 0.01). (2) **The lesion must be FAIR (excitability-matched), not the crude STP-off toggle.** The lesion is
`stp_tau_f`~5 ms: STP stays ON (the u*x multiplier and thus net excitability are identical) but facilitation cannot
BRIDGE the delay. We do NOT use `enable_short_term_plasticity=False` (that jumps effective recurrence to full weight and
turns the net into a Wang-2002 PERSISTENT-FIRING attractor — the delay would no longer be silent). This is the de-risk's
own FAIR control.

## The lesion oracle IS the load-bearing test (anti-hollow)

The de-risk oracle: after an intervening distractor turn, the correct prior referent is recoverable by the ping with
the facilitated buffer but NOT with `tau_f` collapsed — and the recovered referent must CHANGE the reply. The soak
drives the ORGAN through a 3-turn discourse per seed (introduce a focus referent among a couple of earlier distractor
referents -> one intervening distractor turn = the silent delay -> a temporal-recall query) and requires ALL of:

- **INTACT** reactivates the correct focus (`recovered == focus`, ensemble-mean ping margin > MARGIN_MIN=7);
- the hold was **SILENT** (`silent_delay` True — the persistent path did not leak);
- the FAIR facilitation **LESION** (`tau_f`~5) **ABSTAINS** (`recovered is None`);
- the **RENDERED REPLY genuinely CHANGES** intact<->lesion (correct anaphor vs abstain).

6-seed result (numpy-CPU), artifact `research/findings/raw/_activity_silent_wm_production_soak.json`: **6/6 GO.**
INTACT recovers the rotated focus (dog/cat/bird — specificity: the ping recovers WHICHEVER was the focus, not a fixed
structural favorite) with margin +8.6..+21.9; LESION abstains on all six with margin regressed to -3.3..+3.5 (the focus
is no longer specially facilitated, so the ensemble-mean margin falls below the no-confab gate). The reply flips "Going
back — earlier we were talking about the dog." <-> "I don't recall what we were discussing before that." on every seed.
A chat turn's OUTPUT thus demonstrably DEPENDS on the silent hold, and the dependence VANISHES when the facilitation
coupling is lesioned. The `attributable_to` subtraction over the 6-seed-mean ping margin credits **95.6%** of the
reactivation to the facilitation manipulation (only 4.4% survives the FAIR lesion) — the silent hold OWNS the effect,
it is not a residual of the host parse. The de-risk artifact `research/findings/raw/_activity_silent_wm_ping.json`
carries the underlying 6/6 mechanism GO.

## The decision statistic (why an ensemble, why a no-confab gate)

A single spiking read is noisy; the organ reads the buffer as an ENSEMBLE of `RECALL_TRIALS=7` load->silent-delay->ping
builds (different heterogeneity micro-seeds) and gates the read-out on the ENSEMBLE-MEAN ping margin (the loaded
assembly minus the mean of the others) plus a majority argmax==focus. Ensembling regresses the FAIR-lesion's per-seed
structural-favorite coincidence toward chance (so the lesion robustly abstains), while the facilitated intact margin
stays high. `MARGIN_MIN=7` sits cleanly in the empirically-verified gap (intact per-seed ensemble-mean >= 8.6, lesion
<= 3.5). The read-out ABSTAINS rather than confabulate a referent when the ping did not decisively reactivate — the
project's no-confab moat, applied to the recall.

## The wiring (additive, default-OFF, disjoint)

`webapp/server.py brain_chat`, guarded behind `BRAIN_SILENT_WM` (default OFF), per-session organ (`_SESSION_SILENT_WM`,
cleared on reset — a process singleton would leak one conversation's focus into another's recall, mirroring
`_SESSION_MULTIREF`). Two paths: (a) READ-OUT — a DISJOINT temporal-recall query ("what did we start with / the original
topic / go back to the beginning") reactivates the silently-held focus and short-circuits with the honest read-out; the
trigger regex deliberately shares NO lexeme with the D6 hold-query (which runs first and reports the CURRENTLY-held
multi-referent set), verified disjoint both ways. (b) MAINTAIN — every other turn is a pure WRITE-ONLY side effect: a
turn naming a referent holds it silently as the focus; a turn with no new referent grows the silent delay. Neither
MAINTAIN changes the reply. LESION flag `BRAIN_SILENT_WM_LESION=1` builds the recall buffer with `tau_f`~5. The block is
wrapped in `# BEGIN/END faculty:activity-silent-wm` markers (the parent merges several faculties' blocks). DEFAULT-OFF
-> the `os.environ.get("BRAIN_SILENT_WM","0")` guard is not truthy -> the block imports nothing and returns nothing ->
byte-identical; the parent flips default-on after the pool soak.

## Honest residuals (declared)

- **Capacity** is the de-risk's K=4 assemblies (chance = 0.25). A 5th+ distinct referent collides on the last assembly (a
  binder-cap collision, not a WM-time limit) — the same vocab/capacity ceiling class D6 declares.
- **Referent EXTRACTION + the recall-query TRIGGER are a host parse** (reuse D6's `extract_referents` lexicon + a small
  temporal-recall regex) — the declared vocab-ceiling residual; a learned referent/query detector is the next rung.
- **STP-based WM is genuinely TIME-LIMITED** (`tau_f`~1.5 s): it bridges a distractor turn or two, not an arbitrary
  span (the de-risk's declared honest wall). A durable hold is the persistent store / LTM, not this mode. The soak
  models one intervening distractor; multi-turn durability is bounded by `tau_f` and is not claimed here.
- **The read is a host argmax** over the K assemblies' ping-window firing (a read-out instrument), and the ensemble
  decision statistic is host-side; the LOAD-BEARING contribution is the spiking facilitated HOLD, which the lesion
  removes.
- **CO-RESIDENT**: the buffer runs on its OWN `ActivitySilentWM` bridge alongside the recall composer (rides the
  one-brain merge), exactly as the affect/comprehension/D6 organs do.

## Reproduce

```
SIM_BACKEND=numpy python -u -m research.runners._activity_silent_wm_production_soak \
    --seeds 42 43 44 100 101 102
# underlying de-risk (unchanged):
SIM_BACKEND=numpy python -u -m research.runners._activity_silent_wm_ping_derisk --seeds 42 43 44 100 101 102
```
