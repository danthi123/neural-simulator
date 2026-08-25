---
type: finding
status: live
date: 2026-08-25
mechanism: coincidence-binding
board: 132
---

# gap#2 delta-rule spiking binder (board #132) — production-integration assessment: delta stays non-load-bearing to P=48 (12x production's real P=4), the WRITE is still host-numpy, and production's actual shortcut has a MORE mature, already-GO, unwired alternative sitting idle (SlotBinderComposer)

**2026-08-25, CPU/numpy, 6-seed (42/43/44/100/101/102) + a wider single-seed probe.** Board #132 asked whether the
composer/variable-binding path in the live pipeline uses a host binder that `_gap2_spiking_deltarule_binder_derisk.py`
should replace, and to either PORT its WRITE to the real substrate + re-de-risk, or WIRE it into the composer
additive/default-off with a load-bearing + no-regression verify. Verdict: **do neither yet** — the extended re-de-risk
below closes the open question the 2026-07-21 audit correction left standing (delta-vs-additive at scale), and a
separate discovery (a more mature, already fully-on-bridge-WRITE, already-GO alternative for the SAME production
shortcut, `SlotBinderComposer`) makes wiring *this* mechanism the wrong next rung regardless of the WRITE-port
question. Flagged as a follow-on task, not executed here (out of #132's scope).

## Part 1 — the re-de-risk: does delta ever beat additive, at production's real scale?

The 2026-07-21 finding + its same-day audit correction left two things unresolved: (a) the run used `--cap 300`
although the docstring said 788, and (b) `delta>additive` — the runner's own designed second gate clause — was FALSE
at every tested P (1..5); both arms hit the 1.000 ceiling identically, so the delta-rule's namesake advantage
(error-correction beating plain Hebbian accumulation) was never demonstrated, only assumed not to matter yet because
crosstalk at D=128 near-orthogonal roles is small (~9%) up to P=5.

This session re-ran the **unmodified, already-audited runner** (`research/runners/_gap2_spiking_deltarule_binder_derisk.py`,
no code changes) two ways:

Full results transcribed (by hand, from the raw run output) into
`research/findings/raw/_gap2_deltarule_binder_production_scale_recheck.json` — cited below for every number in this
section.

1. **6-seed, `--cap 788` (the full corpus, not 300), `--pmax 8`** (production's real per-fact role count, measured
   below, is 4 — this doubles it): `python -m research.runners._gap2_spiking_deltarule_binder_derisk --seeds 42 43 44
   100 101 102 --pmax 8 --n-facts 20 --cap 788` (argv + git SHA auto-recorded by the runner-provenance wrapper in its
   ledger under `research/findings/raw/_provenance/`, file `runs.jsonl`; raw stdout kept locally,
   gitignored like the original finding's `_gap2_spk_seed*.log` files, so the numbers below are transcribed into
   `research/findings/raw/_gap2_deltarule_binder_production_scale_recheck.json` → `six_seed_formal.by_P` —
   reproducible deterministically, `cfg.seed` seeds the substrate):

   | P | DELTA | additive | permuted-role | decorrelated-ctrl |
   |---|---|---|---|---|
   | 1..8 | 1.000 | 1.000 | 0.000 | 1.000 |

   All 6 seeds identical at every P. `additive` never collapses; `DELTA == additive` throughout.

2. **A single-seed (42) probe extending P far past anything production could ever need** (`--cap 788`, isolated
   `retrieve()` calls, n_facts=5-8 per point — a scoping probe, NOT a 6-seed artifact; numbers in
   `research/findings/raw/_gap2_deltarule_binder_production_scale_recheck.json` → `single_seed_wide_probe.by_P`):
   **P = 8, 12, 16, 24, 32, 48 — delta == additive == 1.000 at every point, permuted-role == 0.000 throughout.** At
   P=48 a single shared D=128 fast-weight matrix is holding 48 independent role-filler bindings without error, 12x
   production's actual per-fact role count, with zero daylight between the error-correcting and the plain-Hebbian
   write.

**What production's real P actually is:** the live deployed corpus — bundle `bridges/developed/scale787/day_33`, its
`facts.json` (404 facts; not part of this repo checkout, gitignored generated data, read directly from the running
deployment) — was checked directly; every one of the 404 facts has exactly **4** populated roles (agent, action,
patient, polarity); none currently use `attribute`/`attribute2`. The count is transcribed into
`research/findings/raw/_gap2_deltarule_binder_production_scale_recheck.json` → `production_role_count_check`. So the
tested range (P=1..48) covers production's actual regime with a wide margin, not just the paper's more limited
P=1..5.

**Conclusion:** the audit's open question is now closed at production-relevant scale (not just resolved-in-principle):
**delta is not shown load-bearing over additive at ANY tested P, from production's real P=4 to 12x that.** The
runner's own designed GO gate (`delta>=0.80 & delta>additive`) stays FALSE throughout; the standing re-scoped verdict
is unchanged (spiking-read reaches the fixed-FHRR ceiling; permuted-role collapses to chance) but the delta-rule's
specific selling point over plain Hebbian bundling has no empirical support at any scale tested here.

## Part 2 — does production use a host binder this mechanism should replace? Yes — but a BETTER-POSITIONED replacement already exists, unwired

Read the live pipeline (`webapp/server.py` → `_build_chat_brain` → `developed_brain_io.load_developed_brain` →
`brain_conversational_agent.BrainConversationalAgent`) and the composer code
(`research/runners/rf_phasor_composer.py`):

- Every currently-deployed developed-brain bundle (`bridges/developed/scale787/day_{9,33}/brain.json`, checked
  directly) has `"composer_kind": "rf"`. Production chat runs **`RFPhasorComposer`**, whose `_bind`/`_unbind_phases`
  install the role phasor as a **FIXED, host-designed diagonal complex synapse** (`self.roles[role]`, never learned,
  never adapts) — exactly the "FHRR exact-inverse bind" #132's docstring targets. This confirms the "host binder in
  the live path" half of the board question: yes, it exists, and #132's mechanism is architecturally a candidate
  replacement for it (a learned associative matrix in place of a fixed diagonal transform, read the same way — via
  `rf_kick`/`rf_resonate_steps`/`rf_read_phases`).

- But **`#132`'s own WRITE is still host-numpy** (`build_W` in the runner: `np.outer(...)` in plain numpy, then
  installed onto the bridge via `rf_set_complex_weights`) — the 2026-07-21 audit already flagged this and it remains
  true; only the READ is on-substrate.

- While tracing this, a **materially more mature alternative for the exact same production shortcut** turned up,
  already built and already GO, sitting unused: **`SlotBinderComposer`** (`research/runners/slotbinder_composer.py`,
  selectable via `composer_kind="slotbinder"` in `brain_conversational_agent.py`). Its `store_pair` write is a REAL
  on-bridge operation — it drives external current through a slot pool and a filler pool together and runs
  `b._run_one_simulation_step()` under a per-slot plasticity gate (genuine spike-driven Hebbian potentiation, not a
  host formula). Three findings (`2026-07-17-gap2-adversarial-verify-CONFIRMED-and-content-addressable-wire-in-GO.md`,
  `2026-07-22-gap2-attribute-slot-GO-FHRR-retirement-step1.md`, `2026-07-22-gap2-pointer-clause-GO-FHRR-fully-retirable.md`)
  show it 6-seed-GO with anti-cheats (permuted-pointer, lesion-the-second-hop, wrong-clause distractor, the no-confab
  moat) covering flat SVO + polarity + multi-hop + single-attribute + **depth-1 embedded clauses** (via pointer
  indirection) — the COMPLETE capability surface the deployed FHRR composer ships today. The last of those three
  findings' own closing line: *"Next: make the slot-binder the production DEFAULT (retire the FHRR/rf fallback) — a
  wire-in + a 320-scale GPU re-verify (gated on the fluency training)."* That wire-in was never executed — the live
  bundles checked above are still `composer_kind: "rf"`, over a month later.

- A likely reason it stalled: `SlotBinderComposer` pre-allocates ONE bridge sized `K = 5 * max_facts` slot-pools (20
  neurons each) wired by a **dense** plastic pathway to every one of `KF` filler pools (20 neurons each). At
  production's real scale (404 facts, K≈2020; a few hundred words, KF≈300-800) that dense slot→filler wiring is
  O(K·KF) pathways — a genuine, uninvestigated scale question this session did not resolve (flagged, not measured;
  see below).

## ⇒ Verdict for board #132

**Do not port #132's WRITE to the substrate, and do not wire #132's mechanism into production, in this pass.**

1. The re-de-risk closes the delta-vs-additive question at (and 12x past) production's real scale: there is no
   demonstrated capability win for the delta rule anywhere tested, so there is no capability case for spending a
   WRITE-port effort on it. A WRITE port (a genuine on-bridge complex-Hebbian/error-correcting update) would still
   only be able to reach parity with the far simpler additive rule at every P checked — and two prior on-bridge WRITE
   attempts for the general "on-substrate multi-bind write" defect (`2026-07-15-edge5-rung2-STP-store-onbridge-6seed-GO.md`,
   `2026-07-15-edge5-rung3-delta-write-PARTIAL-error-correction-refuted.md`, extended in rung-3b to KV=8/P=8) already
   found delta≈additive there too, at a genuine capacity wall (~2 binds). A third lever against the same class of
   defect, without new biological grounding, is exactly what `tools/before_you_build.sh`'s 2-lever rule exists to
   stop (confirmed: the RAG surfaces these same three findings for this exact query).
2. Even setting the WRITE-purity question aside, #132's mechanism is not the best-positioned candidate to wire into
   the live composer for this shortcut: `SlotBinderComposer` already clears the SAME production capability surface
   (including embedded clauses, which #132 was never tested against) with a genuinely on-bridge WRITE, and its own
   finding already named the wire-in as the next step — a month ago, still undone. Wiring #132 in now would install a
   mechanism that is architecturally more complex (full D×D coupling vs. a diagonal transform) than what it replaces,
   with a still-host WRITE, no demonstrated accuracy win, and no coverage of embedded clauses — while a
   strictly-better alternative for the identical gap sits unexecuted.
3. This is a verdict on the METHOD (#132's delta-rule full-matrix binder), not the capability: the capability (a
   learned, coincidence-written replacement for the FHRR's fixed exact-inverse bind) is not abandoned — it is already
   de-risked twice over by a different, more complete mechanism (`SlotBinderComposer`) whose blocker is most likely a
   scale question this session did not measure. Flagged as a follow-on task (`task_5c54ca7f`, spawned this session):
   measure `SlotBinderComposer` at the live corpus's actual scale (404 facts / full vocab) before either fixing its
   likely dense-pathway scale wall or executing its already-scoped wire-in.

## Honest scope / what this does NOT establish

- This did not build or run anything new for `SlotBinderComposer` — its scale question is reported from reading the
  code (`K = 5*max_facts` slot pools, a dense plastic pathway to every filler pool) and a back-of-envelope synapse
  count, not measured. Flagged, not verified.
- The single-seed P-extension probe (P up to 48) is a scoping check, not a 6-seed artifact — it is reported as
  supporting context for the decision, not as a new committed GO/NO-GO claim on its own; the committed 6-seed artifact
  is the `--pmax 8 --cap 788` run.
- Production facts today are 100% flat SVO+polarity (P=4, no attributes in use) — the P=1..48 sweep bounds the
  question broadly, but a corpus that starts using `attribute`/`attribute2` densely, or grows to thousands of facts
  sharing role vocabulary, is untested.

Runner (unmodified): `research/runners/_gap2_spiking_deltarule_binder_derisk.py`. Biology binding updated:
`research/biology/coincidence-binding.md` (`current_status` now notes delta is non-load-bearing to P=48, not just P=5,
and that production has never used this mechanism).
