---
type: finding
status: live
date: 2026-08-01
mechanism: agency-authorship-source-monitoring
lane: A-affect
---

# Lane A · Affect (Phase-0 self-model): AGENCY / AUTHORSHIP 1-bit source tag — a spiking corollary-discharge monitor answers "did you say that or did I?", 6-seed GO (2026-08-01)

A self-model primitive on ONE spiking substrate. Source monitoring — telling self-produced content from
externally-parsed content — is load-bearing for the affective self-model (Johnson-Hashtroudi-Lindsay 1993; its
FAILURE = misattributing self-generated inner speech to an external voice, Feinberg 1978 / Ford-Mathalon). The
biology built here: self-production emits a COROLLARY DISCHARGE / efference copy (Sperry 1950; von Holst-Mittelstaedt
1950; Crapse-Sommer 2008) available to a comparator (Frith 1992); externally-parsed content arrives via a sensory
stream WITHOUT it. A biased-competition comparator reads "was an efference copy present?" and emits a 1-bit self/other
tag. Runner: `research/runners/_agency_authorship_tag_derisk.py`; artifact:
`research/findings/raw/_agency_authorship_tag_6seed.json`.

## Mechanism (brain-based; reuse-by-import; NO `sim/` edit)
One numpy Izhikevich `SimulationBridge`, 9 regions / 370 neurons / ~5.9k synapses, all plasticity OFF (a fixed
comparator). `production` (self generator) emits an efference copy to a corollary-discharge pool `cd`
(`production -> cd` is a real synaptic projection); `parse` (external listener) drives a `sensory_marker` pool. Both
acts weakly drive a shared `content` pool, but content IDENTITY is world-set (external stimulus current to the item's
neurons) and IDENTICAL for self vs other, so content carries no source signal. The source monitor: `cd -> src_self`,
`sensory_marker -> src_other`, with Namburi-Tye cross-inhibition between the two tag pools (each drives its own FS
interneuron that inhibits the other). Read-out per utterance = sign of rate(src_self) − rate(src_other) vs the
ground-truth source. `cfg.seed` set per seed (seeds the substrate — verified: seed-42 thresholds reproducible across
two builds, seed-43 differs).

## 6-seed result {42 43 44 100 101 102} — GO (all 6 gate checks pass; gate's own verdict GO=True)
<!--derived-->
(all values below are aggregates — means/min/max over the 6 seeds — derived from the cited per-seed artifact
`research/findings/raw/_agency_authorship_tag_6seed.json` (`means` + `per_seed`); no per-seed file holds a mean.)
| metric | mean | per-seed | reads as |
|---|---|---|---|
| **authorship accuracy** | **1.000** | 1.0 ×6 | 1-bit self/other judgment vs ground truth (chance 0.5) |
| lesion → chance | 0.486 | 0.367–0.583 | cut `cd->src_self` + `sensory_marker->src_other` gates → genuine chance |
| swap → tag flips | 0.000 (rel 1.000) | 0.0 / 1.0 ×6 | rewire `cd->src_other`, `sm->src_self` → systematic FLIP |
| tag ⟂ content (min per-item acc) | 1.000 | 1.0 ×6 | identical content as self AND other → correct BOTH times |
| tag ⟂ content (content-decode) | 0.167 | 0.050–0.283 | item does not decode from tag pools (LOO; chance 1/6 = 0.167) |
| catch (content, no act) → chance | 0.508 | 0.433–0.567 | content alone carries no authorship |

- **Authorship = 1.000 on every seed** while lesion and catch sit at genuine chance (means 0.486 / 0.508, with
  real seed-to-seed spread from random tie-breaking on a no-signal tie) ⇒ the corollary-discharge / sensory-arrival
  carriers are load-bearing, not decorative.
- **Swap → 0.000 (relabelled 1.000)**: routing the SAME carriers to the OPPOSITE tag pools inverts the judgment
  systematically. The tag tracks the source WIRING, not the content.
- **tag ⟂ content is decisive**: an item uttered as self AND as other is correctly source-tagged BOTH times
  (min per-item acc 1.000; a content-encoding tag would score ~0.5 per item), and content-item identity does not
  decode from the src-tag pool rates above chance (0.167 vs 1/6). The src-margin MAGNITUDE shows weak per-item
  modulation (|r| ~0.248, seed-44 0.476) — this is decision-IRRELEVANT wiring heterogeneity (the margin SIGN, which
  sets the judgment, is 100% source-determined), reported as a diagnostic, NOT gated. It was gated in a first pass
  and mis-fired on that magnitude variance; the instrument was replaced with the well-calibrated LOO content-decoder
  (the fix: measure the actual claim — does content determine the judgment — not a magnitude proxy).

## Honest scope (the deliverable's boundary)
- **FIXED-STRUCTURE comparator** (hand-wired corollary-discharge source monitor, no learning), exactly the scope the
  affect-region GO banked for its hand-wired attractor. The named follow-ons: (1) the content-cued episodic
  SOURCE-MEMORY version (Hebbian-bind `content -> tag` at encoding, content-cue the tag at recall — answers "did you
  say X?" retrospectively for a previously-heard X); (2) self-organized wiring (the efference-copy projection learned,
  not designed).
- The carrier populations (`cd`, `sensory_marker`) are driven as the two input streams (efference copy vs sensory
  arrival) — the legit body/world boundary; the BRAIN-based computation being de-risked is the source COMPARATOR and
  its orthogonality to content. numpy is the CPU backend (real spiking Izhikevich), not a host shortcut.

## Run
```
SIM_BACKEND=numpy python -u -m research.runners._agency_authorship_tag_derisk --seeds 42 43 44 100 101 102
```
