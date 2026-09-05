---
type: finding
status: contributing
date: 2026-09-04
mechanism: da-gated-encoding
lane: integration
integration_faculty: da-gated-encoding
seeds: [42]
verdict: GO
instrument: this is a CURRENCY re-verification, not a new build. The task ("build + de-risk WAVE-0 Gap-4(a) DA-gated
  encoding") was already fully delivered on 2026-08-21/2026-08-25 (three findings, three commits, ledger row
  on_by_default:YES) -- confirmed by `git log --since=2026-08-25 -- webapp/da_encoding_drives_chat.py
  research/runners/_da_encoding_flip_verify.py research/runners/_da_encoding_wired_verify.py
  research/runners/_da_encoding_leansoak.py` returning ZERO commits (untouched for 10 days across dozens of
  intervening landings to adjacent shared integration points -- webapp/server.py, webapp/continuous_engine.py --
  from the mouth flip, affect-coupling-neural, and one-brain Stage-1 arcs). Given that adjacency risk, this session
  re-ran the existing CPU/numpy verifiers fresh on current HEAD (git_sha 64e71517b) rather than re-deriving the
  mechanism. `research/runners/_da_encoding_homeo_trigger.py` (SIM_BACKEND=numpy, no GPU) completed in full and
  reproduced the original's exact numeric result. `_da_encoding_wired_verify.py` / `_da_encoding_flip_verify.py`
  (also numpy-only, no GPU) were also launched fresh; they build multiple full ChatBrain instances through the real
  `webapp.server.brain_chat` handler and did not finish inside this session's time budget under heavy HOST-WIDE CPU
  contention (`uptime` load average ~30 on a 20-core box from other concurrent legitimate work, confirmed via `ps`
  showing the runner alone burning 1000-1200% CPU for minutes without deadlocking) -- an environment/resource
  constraint, not a result. The 6-seed decisive soak (`_da_encoding_leansoak.py --substrate-scaling`, cupy) is the
  ORIGINAL 2026-08-25 artifact, unchanged (same git-untouched file); it was not re-run here (the task's own
  instruction is numpy/CPU, not GPU, and the file is byte-unchanged since the original decisive run).
runner: research/runners/_da_encoding_homeo_trigger.py (fresh, this session) + research/runners/_da_encoding_wired_verify.py
  and research/runners/_da_encoding_leansoak.py --substrate-scaling (original 2026-08-21/2026-08-25 artifacts, cited)
artifacts:
  - research/findings/raw/_da_encoding_wired/homeo_trigger.json (FRESH this session -- run_id 1788574318-3388385,
    git_sha 64e71517b, started 2026-09-04T22:11:58, SIM_BACKEND=numpy)
  - research/findings/raw/_da_encoding_wired/homeo_trigger.json.prov.json (provenance sidecar proving the run above
    is genuinely fresh, not a stale checked-in copy)
  - research/findings/raw/_da_encoding_leansoak/soak_substrate.json (original 2026-08-25, cupy, 6-seed, unchanged)
  - research/findings/raw/_da_encoding_wired/verify.json (original 2026-08-21, numpy)
  - research/findings/raw/_da_encoding_wired/flip_verify.json (original 2026-08-25, numpy)
external: Lisman JE, Grace AA (2005) "The Hippocampal-VTA Loop: Controlling the Entry of Information into Long-Term
  Memory" Neuron 46(5):703-713 -- novelty/reward-gated DA entry into LTM; Kandel Principles of Neural Science 6e
  D.16 (dopamine gates entry into long-term memory); Turrigiano GG (2008) "The Self-Tuning Neuron" Cell
  135(3):422-435 -- homeostatic synaptic scaling. Same anchors the original findings cite; unchanged by this
  session, reconfirmed still accurate to the live code (read in full, see below).
supersedes: none -- this CONFIRMS CURRENCY of research/findings/2026-08-21-da-gated-encoding-wired-into-chat-GO.md +
  research/findings/2026-08-25-da-encoding-substrate-turrigiano-scaling-FLIP.md +
  research/findings/2026-08-25-da-encoding-faculty-default-on-flip.md. It changes no verdict and adds no new
  mechanism; it is a 10-days-later regression check plus an independent read of the honest brain-based/host-shortcut
  boundary for whoever asked for WAVE-0 Gap-4(a) next.
---
# DA-gated encoding (WAVE-0 Gap-4 coupling (a)) -- ALREADY BUILT, DE-RISKED, WIRED, and DEFAULT-ON since 2026-08-25; currency re-verified 2026-09-04

## Headline

The assigned task was to "build + de-risk WAVE-0 Gap-4 coupling (a): DA-gated ENCODING." **This capability already
exists, already meets or exceeds every requirement the task lists, and has been production default-ON for 10 days.**
No new mechanism was built. What follows is (1) the paper trail proving it is the same capability, (2) a fresh
currency check given the elapsed time and the volume of adjacent landings since, and (3) the honest brain-based
verdict the task asked for, confirmed against the live source rather than assumed from the docs.

## The paper trail (why this is not a coincidental near-match)

The task's own description -- "the dopamine (DA) signal gates memory encoding strength (bio: DA-gated hippocampal
LTP; the Lisman & Grace 2005 VTA<->hippocampus novelty/reward loop; DA sets the plasticity threshold for what gets
encoded)" -- is a paraphrase of `webapp/da_encoding_drives_chat.py`'s own module docstring almost verbatim
("dopamine gates entry into LONG-TERM memory: a fact heard while the SNc bursts ... is encoded STRONGER"; "Lisman-Grace
hippocampal-VTA loop; Kandel D.16"). This is board WAVE-0 Gap-4(a), built 2026-08-21 (commits `757833120`
integrate, `47b2a24da` finding+ledger, `d5c67f7cd` wire-in verify) and flipped to production default-ON 2026-08-25
(commits `c67096f49` UNDEFINED-flip-gate diagnosis -> `2026-08-25-da-encoding-substrate-turrigiano-scaling-FLIP`
[the Turrigiano on-substrate homeostat that resolved it] -> `2026-08-25-da-encoding-faculty-default-on-flip` [the
coordinated flip + prep rungs]). `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` row `da-gated-encoding` (lines 778-792)
carries `de_risked: YES`, `wired: YES`, `on_by_default: YES` with a `default_anchor` binding the ledger to the live
constant `_DA_ENCODING_DEFAULT_ON` in `webapp/da_encoding_drives_chat.py:123` (flipping that constant back to False
blocks the commit via `gates/production_integration` Check A). `git log --since=2026-08-25` on the four
mechanism/runner files returns **zero commits** -- nothing has touched this capability in the 10 days since.

## What this session verified fresh (currency, not novelty)

Ten days is long enough, and enough has landed to adjacent shared integration points since (the linattn mouth
flip, affect-coupling-neural, recall-gate fixes, one-brain Stage-1 -- all of which touch `webapp/server.py` and/or
`webapp/continuous_engine.py`, the exact two files this coupling's install-site and idle-tick trigger live in),
that a currency check earns its cost rather than rubber-stamping a stale doc. On current HEAD (`64e71517b`), in
this worktree, numpy/CPU only (no GPU used, per the task's own cost-routing instruction):

**`research/runners/_da_encoding_homeo_trigger.py` -- ran to completion, fresh, GO.** Provenance confirms this is a
real run, not a stale artifact (`run_id 1788574318-3388385`, `started: 2026-09-04T22:11:58`, argv pointing at this
worktree). All five checks reproduced GO, and the core numeric result **matches the original 2026-08-25 run to
every printed digit**: mean stored `|w|` `1.333333333333333 -> 1.30940917847764` over 3 DA-gated engrams (fires on
growth), a no-op with no new writes, re-fires at `n_engrams=4` after a new fact, and both `BRAIN_DA_ENCODING_LESION=1`
and `BRAIN_DA_ENCODING=0` disarm the pass. This is the prep-rung-2 consolidation trigger (`apply_substrate_homeostasis`
wired into the idle tick alongside the D5 learn-through-use pass) -- exercising exactly the code path most exposed to
the intervening `continuous_engine.py` churn, and it is byte-for-byte unregressed.

**`research/runners/_da_encoding_wired_verify.py` and `_da_encoding_flip_verify.py` -- launched fresh, inconclusive
by timeout, not by failure.** Both build several full `ChatBrain` instances through the real `brain_chat` handler
(numpy has no GPU parallelism to lean on for a ~1700-neuron network with tens of thousands of synapses). Under this
host's CURRENT load (`uptime`: load average ~30 on a 20-core machine, from other legitimate concurrent work per this
project's own multi-lane compute discipline) a run that historically completed in about a minute was still
mid-build after several minutes, burning 1000-1200% CPU with no deadlock (confirmed via `ps`/`uptime`, not just a
silent hang). I did not force this further: burning more wall-clock against host-wide contention for a
re-confirmation that homeo_trigger + the untouched-file evidence + the original decisive artifacts already support
is not a good trade. **The artifacts these two runners would refresh
(`research/findings/raw/_da_encoding_wired/verify.json`, `flip_verify.json`) are the original 2026-08-21/2026-08-25
runs** (verified NOT stale-masquerading-as-fresh: I checked their `.prov.json` sidecars, which show the original
`started` timestamps and the original scratchpad argv, before drawing any conclusion from their content) and remain
the citable evidence for load-bearing + lesion below.

**The 6-seed decisive soak was not re-run.** `_da_encoding_leansoak.py --substrate-scaling` is the flip gate
(`research/findings/raw/_da_encoding_leansoak/soak_substrate.json`, cupy, seeds 42/43/44/100/101/102) and is
untouched since 2026-08-25 (confirmed via git log above). The task instructs numpy/CPU, not GPU, for this work,
and there is no code change to re-gate.

## Load-bearing + lesionable (the task's de-risk bar), cited from the untouched original artifacts

From `research/findings/raw/_da_encoding_wired/verify.json` (2026-08-21) and reconfirmed unmodified by the
`_DA_ENCODING_DEFAULT_ON` flip logic read directly from current source (`webapp/da_encoding_drives_chat.py`,
read in full this session):

- **Load-bearing:** teaching the identical fact ("dog eat grass") under a high-engagement turn (`INDUCE=1300`, DA
  1.2386777669861797) versus a low-engagement turn (`INDUCE=100`, DA 0.04616293556102311) writes
  `g_high=2.4773555339723594 > g_low=1.0`, and that same `g` produces a measurably stronger trace on the
  magnitude-carrying store (`stored |w| ratio == g_high/g_low`, byte-exact). Varying DA changes how strongly the
  fact is encoded -- the definition of load-bearing.
- **Lesionable, non-circular control:** `BRAIN_DA_ENCODING_LESION=1` pins `g=1.0` on BOTH the high- and low-induce
  arms (`g_les_high == g_les_low == 1.0`), so the high-vs-low differential collapses to exactly `0.0` even though the
  live DA level still varies underneath -- the control isolates the DA->gain LINK, not the DA signal itself
  (`BRAIN_DA_DRIVES_LESION` is the separate, distinct lesion for the SNc read). `tools.lab.attributable_to` reports
  `attribution_to_live_DA_read == 1.0` -- the entire write-gain differential is owed to the live DA read, none of it
  a residual host `if-engaged` shortcut.
- **6-seed, target-block-attributed (original, unchanged):** the leansoak's `stress_net_genuine_violations == 0` on
  all 6 seeds; the two RAW stress-net violations (seeds 43, 102) were traced to a FOREIGN-BLOCK confabulation in the
  OFF control arm (a different engram's damaged decode coincidentally matching the cue) that ON's stronger encoding
  makes the store correctly ABSTAIN from, rather than a memory regression -- the moat working better, not worse.

## The brain-based verdict the task asked for (read from live source, not assumed)

Read `webapp/da_encoding_drives_chat.py` in full this session (290 lines) plus a live build log from the fresh
`_da_encoding_wired_verify.py` launch, which shows the actual instantiated substrate: `Region 'snc' (10 neurons):
using Izh type IZH2007_DOPAMINE` -- a real spiking dopaminergic nucleus exists in the built network, not a stub.
The honest three-part breakdown:

1. **The DA SIGNAL is brain-based.** `da_level_of(chat)` reads `chat._last_da_drives["da_level"]`, set by
   `webapp/da_mode_drives_chat.observe_turn` (board #76/#79, GO), which drives that real spiking SNc nucleus from
   the turn's engagement and reads the self-produced tonic dopamine concentration off the neuromodulator bus. This
   is neurons and synapses, not a host formula.
2. **The DA -> ENCODING-GAIN MAPPING is HOST ARITHMETIC.** `encoding_gain_for()` computes
   `g = clip(g_min, g_max, 1 + k_DA*(DA - DA_baseline))` (plus, when homeostasis is on by default, a Turrigiano-style
   EMA of the raw salience) -- a Python/numpy scalar closure with host-tuned constants (`_K_DA=2.0`, `_G_MIN=0.5`,
   `_G_MAX=3.0`, `_DA_TONIC_BASELINE=0.5`, `_EMA_BETA=0.25`, `_S_MIN/_S_MAX`). No neuron or synapse computes this
   map; it is called once per store as a plain function. **This is a documented, honest shortcut**, and the ledger
   already names it as such (row `da-gated-encoding`, `scaffold_retired: NO`, residual (3): "the gain MAP constants
   (k_DA, g_min/g_max) are host-tuned").
3. **The CONSOLIDATION pass (`apply_substrate_homeostasis` / `OneBrainComposer.apply_homeostatic_scaling`) is a
   hybrid -- genuinely neural on the SENSE side, host arithmetic on the ACTUATE side.** `_measure_block_readout`
   resonates each stored engram and reads the mean `|Z|` over that engram's real readout neurons off the bridge
   membrane -- a genuine measurement of postsynaptic activity, not a re-read of the stored DA scalar. But the
   rescaling factor it computes from that measurement is then applied to the stored synaptic weight array via a
   plain multiply (`store_conns[block] *= scale`), not by a spiking synaptic-plasticity rule executing inside the
   simulation loop. The homeostat's INPUT is brain-derived; the ACTUATION arithmetic is host-side.

**Overall: a host-computed gate reading a brain-based signal, writing to a substrate-resident synaptic store.** The
correct term per this project's own `docs/TERMS.md` is **"on-by-default (scaffold not retired)"** -- not
"integrated" / "fully spiking" (both of which require the gate arithmetic itself to be neural, which it is not).
The three residuals are exactly as the ledger already names them (unchanged by this session): (1) the upstream
message->engagement->SNc-afferent scalar is a host sensory/comprehension boundary (the #79 DA-mode read's own named
residual); (2) on the `BRAIN_COMPOSER_KIND=rf` numpy fast-path recall the store is magnitude-invariant so the gain
is a write-side reserve there ONLY -- **confirmed still irrelevant to the real default**: `webapp/server.py`'s
`_COMPOSER_KIND_DEFAULT = 'onebrain'` (read live this session, line ~3692), i.e. production actually runs the
magnitude-carrying composer where the gain is load-bearing, not the inert fast path; (3) the gain-map constants are
host-tuned. None of these are new findings -- they are the same residuals the 2026-08-25 finding named, reconfirmed
against current source rather than taken on faith.

## What this means for the WAVE-0 backlog

`GAP_CLOSURE_MISSION.md`'s own "PRE-DECIDED NEXT ACTIONS" already lists WAVE-0 Gap-4 as **(a) DA-gated encoding
[DONE, this item] -- (b) curiosity crave-threshold [ALSO already done, commit `3d855330b`, default-OFF] -- (c)
attention-gain [not yet found in this pass] -- (d) episodic-salience (affect-gated D5 store salience) [not yet
found in this pass]**. Whoever dispatched this task as open work was working from a stale read of the backlog (the
mission board's own "CURRENT STATE" header had already moved past this item to 2026-09-04d by the time this task
was issued). **No code change was made in this finding** -- there is nothing to build. The productive next step for
a fresh WAVE-0 pass is (c) or (d), not re-deriving (a).

## Honest scope of this finding

This is a confirmation, not a discovery. The 6-seed generalization evidence, the load-bearing numbers, and the
lesion numbers quoted above are the ORIGINAL 2026-08-21/2026-08-25 artifacts (byte-unchanged, verified via git log
and via each artifact's own `.prov.json` sidecar) -- this session did not reproduce the 6-seed soak or the full
wired-handler load-bearing/lesion battery end-to-end (the latter was attempted and left running past a reasonable
wall-clock budget under host contention, per the instrument section above). What this session adds is: (1) proof
the mechanism files are genuinely untouched for 10 days (git log), (2) one full fresh reproduction of the
consolidation-trigger wiring with numerically-matching output (the piece most exposed to intervening churn), and
(3) a live-source re-read of the brain-based/host-shortcut boundary rather than a restatement of the old finding's
prose.
