# #271 stage 1 (round 2, synthesis): learned shifter routes on one bridge, integrated hold-first (S1-hold), then bind (S1-bind)

Claude, judge and synthesizer, 2026-10-21. Inputs: the accepted stage-0 census (RULING deixis-271-s0-accept; W1 with B3 on:
28 pointer words stored as names, 4 role shifts, 60 cross-speaker collisions, 60 silent drops, 25 declines), the refuted
round-1 design and its critique, the fresh record, the biology brief, the three round-2 designs (learning-first LSB,
substrate-first DXB, integration-first S1-hold/S1-bind) and their six adversarial critiques, and the binding RULINGS
c1-learned-1016, wrc-frame-go-1020b, hd-gate-1021. Code was read at main `0849ef0ca` (workstation checkout), read-only.
Nothing in any repo was edited. Scratch: `round2/scratch/judge/` (a rate replica of the step-0 world and bars, and a
census projection; both run, numbers quoted below are theirs).

**Fix round (Claude, 2026-10-22), after the final verification (a refuter and a completeness critic, both NEEDS-CHANGES;
three BLOCKING findings).** Every change is listed per finding in `round2/FIX_NOTES.md`. In one paragraph: (1) the
step-0 GO rule was unreachable because S2b compared each name with a class-maximum null set by the most-exposed name;
the bars are now within-token contrasts plus an absolute route bar S0 taken from the engine's own transfer curve, and the
replica (`round2/scratch/fix/`, run) shows the uniform-decay covariance rule cannot reach that bar for rare forms at any
operating point while a presynaptically gated leak (COV-PRE) does, so COV-PRE is added as the sixth arm (sections 2.6,
3); (2) plastic weights now live in the engine's units, [0, 20] on 12-neuron token assemblies, with the arithmetic from
`sim/` and a measured transfer curve (2.6); (3) pointing words and anaphors (third person, demonstratives,
former/latter, the reduced and chat spellings) are brought into stage 1 through a learned third relation, GIVEN, whose
afferent production never renders, so every such turn is HELD by referential failure through a learned route; resolving
them stays stage 2, and the required-changes table now says which items are answered and which moved (sections 1, 2,
4, 8, 16).

host-decision: scaffold
decides: on a turn a routed writer would act on (a B3 store, the TUI fallback's echo, a false-belief scenario step), a fail-closed hold rule reads the organ's spiking latches and keeps the turn out of the store when any token read is a pointer pool, referential failure, ambiguity, the self latch, a co-commit or a non-invariant read, and holds every such turn when the organ is unavailable, raises or sees a different text; the per-turn relay of that read to the writers and their skips on a held turn; which world afferents production withholds (voice, addressee, givenness: every production pointer hold is caused by the silenced afferent); the toy's discourse-givenness afferent (the entity named in the previous utterance); the exposure/hearing transmission-gate schedule and the exposure-only plasticity schedule; the exposure generator's English rendering table (the R1 paradigm, the R2 reduced and regional forms, the R3 chat-spelling rules); the form-to-assembly front end; the per-token latch reset and the between-turn quiescent reset
allowed-by: CLAUDE.md non-negotiable 1
replaced-by: board #260's word-role frame statement path writing latched referents into the composer's spiking store (the hold rule, the relay and the skips); S1-bind's voice and addressee afferents and #271 stage 2's spiking mention trace on the organ's bridge (the withheld afferents and the toy givenness afferent); the stage-3 OrganDescriptor merge onto the one-brain pool (the resets); a percept-agreement neuromodulatory plasticity gate on the merged pool (the gate schedules); human or AI-teacher multi-party speech through research/runners/ai_teacher.py (the rendering table)
cap: 3 builds

## 0. The choice

**The mechanism is DXB's (substrate-first), repaired as its critiques require. The integration is S1-hold's (integration-first),
on an honest calendar. The learning rule is settled by LSB's step-0 census, made fair.** No single design survived: every
critique returned NEEDS-CHANGES, and each design had a flaw the others did not.

- DXB had the most complete substrate (REF commit, AMB conflict, RF_k failure, SELF latch, agency motif, no held state), but
  its conjunctive cells fired on any two of three inputs (so the relation afferent was optional and the P<->C loop
  contaminated teaching), its honesty rested on one latch read that fails open, its `teller=` key was a host relay, it added
  English read-out sentences, and it coupled itself to a moat floor that is not on main. Each is fixed below (sections 2.3,
  2.4, 5, 6, 11).
- LSB had the right discipline on the rule (census first, sim/ edit only if needed) and the right scene (plasticity over the
  whole utterance), but its GO rule could not be met under the hard clip, its C<->Q loop was closed, its production anaphora
  carried mention holds across turns, its told store plus #129 told line was a two-bridge host relay read by no reply, and its
  `deixis=` kwarg never reached W1 on the default path. The census-first discipline is kept (section 3); the rest is dropped.
- S1-hold had the only integration that can land on the default path soon and cannot add a wrong fact, but it let a held
  statement fall into the question pipeline (where "you" resolves to the brain), it chose its English exposure forms from the
  census probe spellings (so its census numbers measured the fixture, not learning), it failed open on an uncertain read, it
  flipped before the 11-01 pin on a calendar no one believed, and its rule comparison was unfair. The hold-first integration
  and the subtractive discipline are kept; each flaw is fixed (sections 5, 4.3, 2.4, 13, 3).

What this synthesis keeps from each, in one line each: DXB's circuit and lesion ledger; LSB's rule census and whole-utterance
exposure; S1-hold's "the first integrated effect is a hold that replaces a store, never a non-store", its STOP/RESULT cadence
and its small dev lines. What it rejects from all three: any host decision of content that is not declared, capped and given a
neural replacement (the HD header in section 12 lists every one).

## 1. Claim and scope

**Stage 1 (S1-hold) claims, if its gates pass:** on one spiking bridge, taught from overheard multi-voice grounded scenes with a
covariance-type rule, the brain learns which heard forms point at whoever is speaking, whoever is addressed, or an entity
the conversation has already made given (pointed at, or just mentioned: third-person pronouns, demonstratives,
former/latter, including possessive forms), and which forms are names. It then refuses to store a heard statement in which
such a form, or its own name, appears: the statement is held, nothing is stored, and its lesion changes the reply. In
production every such hold comes from referential failure: the pointer pool fires through its learned route, the relation's
afferent is not rendered in chat, so nothing can be bound and the turn is held. A user's claim about "you" or about "the
brain" never becomes a self-fact, and "the latter", "that guy", "hes" or "em" never becomes an entity name. It learns this
from made-up words in the toy and from a fixed English paradigm in production, and the made-up-marker arms show the
decision follows exposure, not the string.

**Stage 1 does not claim:** discovery of the relation inventory (speaker, addressee, given, two decoys: innate conjunctive
structure; the word-to-relation and word-to-entity maps are learned); role assignment (B3's positional extractor still
decides agent and patient for every token the organ passes; declared, retired by board #260); RESOLVING a pointing word or
anaphor to its referent (stage 2 of #271: the mention trace on the organ's bridge replaces the toy's givenness afferent; the
two-candidate order mechanism; section 10); speech (the brain says no "I" or "you" in chat; organ-level only); identity
across sessions (no voice afferent in production S1-hold at all; S1-bind adds it, section 10); reported-speech storage of a
held claim (stage 2, the M3b source population); generalization to a spelling that no exposure rule produces (`moi`, three
census items, stays stored as a name: a counted residue with a method, a stage and a date in section 11; a held-out
spelling arm, 8.2, measures the limit).

## 2. Mechanism: one organ, one bridge

New module `research/runners/deixis_binding_organ.py`, class `DeixisBindingOrgan(seed, *, exposure, lesion=None)`. One numpy
Izhikevich `SimulationBridge` built from an explicit-wiring plan (`sim/bridge.py:4506 inject_explicit_wiring`), process
backend, `cfg.seed` set per seed, the organ's own config (`hebbian_min_weight=0.0`, `hebbian_max_weight=20.0` in the
engine's weight units (section 2.6), `propagation_strength` left at the engine default 0.05, `enable_stdp=False`,
`enforce_plastic_mask_in_hebbian=True`, no `hebbian_mean_subtract`; on the COV-PRE arm `hebbian_weight_decay=0.0`, the
rule's own presynaptically gated term being the only leak). Every population name carries the
prefix `dx_` so a later `OrganDescriptor` row (`research/runners/onebrain_merge_framework.py:51`) can merge it onto the
one-brain pool as data (stage 3). No English word appears in the module (CI, section 9).

### 2.1 Afferents (host-supplied; world, body or teacher only; non-negotiable 1)

| Afferent | Toy (steps 0-2) | Production S1-hold (steps 3-4) |
|---|---|---|
| TOK_w, the heard word form | nonce string -> assembly index (front end, declared) | fixture form -> assembly index; a form outside the fixture lexicon drives an untrained spare |
| VOICE_e: person e is speaking | fixed sparse pattern per voice, on only while e speaks | not rendered (no identity in S1-hold) |
| ADDR_e: the utterance is directed at e | the scene (world) | not rendered |
| GIV_e: e is given in the conversation | (a) the entity the speaker points at (a gesture: a world percept); (b) the entity named in the previous utterance of the discourse (the hearer's discourse memory, rendered by the toy world: a declared scaffold, replaced by the stage-2 mention trace on this bridge, E[e] -> an NMDA-held trace -> H_GIV[e]) | not rendered |
| PART_e: e takes part in the seen event | every event participant (agent, patient person or object, possessor) at the same high rate; present-but-uninvolved entities at a low presence rate | none (no scene in chat) |
| LEFT_e, NEWEST_e (decoys) | per-person coin bits drawn fresh each utterance, independent of the event by construction; no toy word encodes them | not rendered |
| PRODUCE (body; step 1b only) | drive on the brain's own speaking turn | not used |
| mode | exposure / hearing / speaking: a declared transmission-gate schedule (2.3) | hearing only |

The host never says which token means what, never aligns a token with a referent, never opens plasticity per token, and never
compares strings.

**Withheld afferents in production (declared; minor finding on the HD header).** In production S1-hold the host renders
no VOICE, ADDR or GIV afferent. That is a host choice of what the brain hears, so it is in the `decides:` line, with its
replacement (S1-bind's voice and addressee afferents; the stage-2 mention trace). It is also the cause of every production
hold of a pointer form: the learned route fires P_k, no relation line can complete a conjunction, RF_k latches, the turn is
held. The production "you" hold is therefore referential failure under a silenced afferent, not learned addressee meaning
(bio.md 3(a): in one-to-one chat the addressee is the brain), and no record may describe it otherwise.

### 2.2 Populations (step-1 defaults; the dev grid fixes them with the first-all-pass-row rule written before the sweep)

| Population | Units x neurons | Driven by | Projects to | Reused / new |
|---|---|---|---|---|
| TOK[w] | per form x 12, plus 12 spares | the form | P_k (plastic `dx_ptr`), E (plastic `dx_name`); plastic weights in [0, 20] (2.6) | word-role organ front-end convention (its NW=8 raised to 12 by the transfer curve, 2.6) |
| E[e] + FS pool | persons (voices, B1, SELF; toy objects) x 12, FS 16 | PART (exposure), TOK name route, C_k[e] release | C_k[e] (`dx_teach`), REF commit stage | new |
| H_k[e] relation lines | 5 relations x entities x 4 | VOICE, ADDR, GIV, LEFT, NEWEST | C_k[e] (AMPA, subthreshold alone) | new |
| C_k[e] conjunctive cells | 5 relations x entities x 4 | H_k[e] AND (E[e] via `dx_teach` OR P_k via `dx_release`); E and P alone, E+P without H, all subthreshold (E->C and P->C on the engine's `nmda_slow` excitatory receptor, `sim/regions.py:528`, or a weight scheme; P0 measures the AND) | P_k (`dx_conv`), E[e] (fixed release) | social-place-cell / dmPFC self-other conjunctive coding (Omer 2018, Danjo 2018, Jamali 2021) |
| P_k pools: SPK, ADR, GIV, LEFT_dec, NEWEST_dec | 5 x 12 | C_k[*] (`dx_conv`), TOK (`dx_ptr`) | C_k[*] (`dx_release`, subthreshold alone), RF_k | new |
| REF commit stage over E | spiking_commit `commit_plan` (K = entities) | E | latches S_e | `research/runners/spiking_commit.py` |
| AMB + CAND | 4 rate-bounded CAND per E, AMB 12 | two or more co-driven REF | vetoes every gate; own latch | DXB; Nref ambiguity (Nieuwland & Van Berkum 2006) |
| RF_k referential failure | 5 x 8 | P_k through slow NMDA; inhibited by any gate burst | own latch | DXB |
| SELF latch | the S_SELF latch of the REF stage | | the hold read | |
| step 1b only: agency motif, OUT[w], OUT commit | reused sizes; OUT per form x 8; OUT commit K = forms | PRODUCE -> cd -> src_self; sensory_marker -> src_other; cd -> C_SPK[SELF]; cd -> INH_other -> C_SPK[e != SELF] | | `_agency_authorship_tag_derisk.py` constants |

Sizes: toy hearing core about 1,450 neurons (TOK 31 forms x 12 + 12 spares, E 10 entities x 12, H and C 5 x 10 x 4 each,
P 5 x 12, RF 5 x 8, the REF commit stage over 10 entities, AMB), about 67k plastic synapses (TOK -> P and TOK -> E,
372 x 180); step 1b adds about 1,050
(OUT and its commit stage). Production S1-hold (about 60 pointer forms with the R1-R3 table of 4.3, about 60 content forms,
SELF plus the fixture persons): about 2,300 neurons, about 250k plastic synapses.

### 2.3 Mode gates: how teaching and release are kept apart (answers the C<->Q / P<->C loop and "any two of three" findings)

| Gate | EXPOSURE (teaching) | HEARING (test, production) | SPEAKING (toy, step 1b) |
|---|---|---|---|
| `dx_teach` E -> C | open | closed | open |
| `dx_conv` C -> P | open | closed | open |
| `dx_release` P -> C | closed | open | closed |
| TRANSMISSION of the learned routes `dx_ptr` TOK -> P, `dx_name` TOK -> E | closed | open | closed |
| plasticity gates `dx_ptr`, `dx_name`, (`dx_ptr_out`, `dx_name_out`) | open for the whole utterance, from onset to 100 ms after the last token | closed | closed |

- During exposure P_k is driven only by C_k[*], and C_k[e] only by H_k[e] AND E[e]: both world-driven. P_SPK fires exactly
  when an event participant is the current speaker (K3 of the brief, by synaptic summation). The learned W->P synapses cannot
  feed P back into their own teaching signal, because P -> C is closed AND the learned routes do not transmit while they
  learn: in encoding mode the plastic TOK -> P and TOK -> E synapses change but carry no current, so the postsynaptic trace
  the rule reads is the world's alone (the acetylcholine encoding mode, which suppresses transmission at the
  associative synapses being modified while afferent input stays on: Hasselmo & Bower 1993; Hasselmo 2006). This is what
  makes the rate replica's "y is world-driven" assumption exact, and it removes the runaway in which a route strong enough
  to fire its pool would teach itself. The engine applies plasticity to `cp_connections.data` whatever the transmission
  gate (`set_transmission_gate`, `sim/bridge.py:5493-5517`: it gates the current, the plasticity gate the updates), which the step-0b smoke checks by reading back
  both gate values and a nonzero weight change on a closed-transmission route.
- During hearing a token drives P_k through its learned weights; P_k plus the held H_k[e] fires C_k[e], which drives E[e];
  C -> P is closed, so nothing re-ignites the pool and activity decays with the token (no recurrence anywhere on this path).
  The per-token reset of the E, RF and AMB latches is a declared timing residue; the carry-over bar (T-carry) measures it.
- The relation afferent is obligatory: E+P without H is subthreshold. P0 measures C firing without H (<= 1% of windows).

### 2.4 Decisions at hearing: spiking commit, and the fail-closed read (answers the "single latch fails open" finding)

Tokens are presented in heard order, about 40 ms each, with the latches reset between tokens. The REF commit stage gives each
token exactly one of: an E latch (bound), the HOLD class (RF_k, AMB, co-commit), or none. Name and pointer drives both
converge on E, so there is no host comparison between routes. Every latch is read by `read_with_variants` at x0.5, x1 and
x2 (`spiking_commit.py:165-171`).

**The hold rule (host, declared in the HD header; the only content decision S1-hold makes):** a statement that the flag-off
path would STORE (the extractor returns a triple AND the B3 non-contradiction check accepts it) is HELD, and nothing is
stored, when any token with an assembly reads anything other than an invariant `none` or an invariant NAME latch on a
non-self entity. So each of these holds: a pointer-pool latch or RF_k latch (in production every pointer form fires RF_k,
since no H afferent is rendered: SPK, ADR and GIV alike), an S_SELF latch (the self's learned name), AMB, a co-commit, a
non-invariant read, or more than one latch on one token. A statement the flag-off path would NOT store, including one B3
rejects as a contradiction, is byte-identical to today: a contradiction keeps today's rejection reply (refuter minor 12).
The same read also gates the two other writers that act on a heard statement (5.1): the TUI fallback's register echo and
the false-belief scenario writer. This is subtractive: S1-hold can turn a store (or an echo, or a scenario step) into a
non-store and nothing else. "tell me more", "how do you feel" and every question are untouched.

**Fail-closed degrade (completeness major 6).** With the flag on, an unavailable organ is never read as "nothing to hold".
If the trained organ cannot be loaded at startup, the server refuses the flag (startup error naming the cache key) rather
than serving the flag-off path under the flag's name. If `read_turn` raises, or the raw text it read differs from the text
the writer was given, the turn is HELD with class `unavailable` or `mismatch`. Both are counted per chat
(`deixis_engaged`, `deixis_unavailable`, `deixis_mismatch`), carried in the reply JSON, and pinned by tests on the server,
the TUI and the rich default path.

**Co-commit (sibling-lane risk).** The cs-race lane measured a symmetric two-candidate spiking commit latching BOTH slots
on 1,158 of 1,440 decisive draws (cs-race-f1d-1021). Here a co-commit is a HOLD, so the same failure would show as
over-holding, not as a wrong store. P0 and the organ-gate reads report the co-commit rate on single-referent tokens, and
T1-T5 count it as a miss.

### 2.5 Speaking and the efference copy (toy only, step 1b)

The brain's own turn: PRODUCE drives cd (the efference copy; gate `dx_cd`), cd drives C_SPK[SELF] as the self's "voice"
and INH_other onto every other C_SPK[e] (corollary-discharge suppression of other speakers during the brain's own speech:
Ford & Mathalon 2004, Heinks-Maldonado 2005). To voice an entity, E[e] is driven, C_k[e] fires for every held relation, P_k
drives OUT through the reverse routes learned while listening (TOK -> OUT is a fixed one-to-one auditory-to-motor map), and the
OUT commit stage, enabled only by PRODUCE, picks the token. Nothing holds VOICE or C_SPK. Declared honestly: PRODUCE is a
body signal the toy initiates, not a copy of the brain's reply-side motor command (the replacement is a projection from the
spiking word-commit of the recall mouth, board #272); the motif is a second authorship comparator beside the production
self-schema author pool until the stage-3 merge shares one.

### 2.6 Weight scale: what a learned route must reach to fire its target (answers BLOCKING finding 2)

The round-2 packet fixed plastic weights to [0, 1]. In this engine a token assembly at those weights never fires a pool, so
every learned-route read would have failed by construction. The arithmetic, from the code:

- Per presynaptic spike the target's excitatory conductance rises by `w x propagation_strength`: `g_e_increase = (W^T
  fired) x cfg.propagation_strength` (`sim/bridge.py:9258-9261`, the E/I-split path; `:9269-9272` without it), with
  `propagation_strength = 0.05` (`sim/config.py:871`). The unit of `w` is therefore 0.05 nS per spike.
- The conductance decays by `exp(-dt/syn_tau_g_e)` per step (`sim/bridge.py:2910`; `syn_tau_g_e = 5.0` ms,
  `sim/config.py:394`): 0.819 per 1-ms step, a steady-state gain of 1/(1 - 0.819) = 5.52.
- The current is `I_syn = g_e (E_e - v)` (`sim/kernels.py:1115`), `E_e = 0 mV` (`sim/config.py:392`), added to the external
  current (`sim/bridge.py:9327`) and integrated by the Izhikevich-2007 update `C dv/dt = k(v - vr)(v - vt) - u + I`
  (`sim/kernels.py:76`), RS defaults C 100 pF, k 0.7, vr -60, vt -40, a 0.03, b -2, d 100 (`sim/config.py:20-23`), with
  the organ's per-neuron heterogeneity (measured on the bridge: C 73-134 pF, a 0.018-0.050, b -2.76 to -0.99).
- A token assembly of NT neurons driven at 1500 pA (the word-role front end, `word_role_parser_organ.py:127`) fires at
  about 0.19 spikes per ms per neuron (measured: 61 spikes from 8 neurons in 40 ms). Steady state at the target:
  `g_e = NT x 0.19 x w x 0.05 x 5.52 = 0.052 NT w` nS (measured 0.839 nS at NT = 8, w = 2: the formula gives 0.84).
- The target must reach about 150 pA to fire inside one 40-ms token: the measured f-I of the organ's cells under direct
  current gives 0 spikes at 80 pA, 0-2 cells at 100 pA, every cell (first spike at 29 ms) at 150 pA. The theoretical
  rheobase of the RS cell is lower (0.7x^2 - 12x + I = 0 has no root above I = 51 pA), but near it the first spike comes
  later than a token lasts.
- With a driving force of 50-60 mV: `w_fire ~ 150 / (0.052 NT x 55) = 52 / NT`, about 6.5 at NT = 8 and 4.3 at NT = 12. At
  w <= 1 the drive is at most 3 pA per NT = 8 assembly: the [0, 1] range could never fire a pool.

Measured transfer curve (`round2/scratch/fix/wscale_curve.py`, the engine itself, numpy backend, heterogeneity on, 10
seeds, a 12-neuron target pool, a 40-step token; target spikes min / median / max, and the fraction of target neurons
firing at least once, min over seeds):

| w (engine units) | NT = 8 | NT = 12 |
|---|---|---|
| 4 | 0 / 0 / 0, 0.00 | 0 / 0 / 1, 0.00 |
| 5 | 0 / 0 / 0, 0.00 | 0 / 1 / 4, 0.00 |
| 6 | 0 / 0 / 1, 0.00 | 7 / 7 / 10, 0.58 |
| 7 | 0 / 1 / 3, 0.00 | 11 / 12 / 12, 0.92 |
| 8 | 1 / 3 / 7, 0.08 | 12 / 12 / 13, 1.00 |
| 10 | 9 / 10 / 12, 0.75 | 12 / 13 / 14, 1.00 |
| 12 | 12 / 12 / 14, 1.00 | 13 / 15 / 18, 1.00 |
| 20 | 16 / 18 / 22, 1.00 | 24 / 25 / 27, 1.00 |

**The choice.** Token assemblies of NT = 12 (TOK and the spares), the plastic range [0, W_MAX] with W_MAX = 20 (the same
bound the word-role organ uses on the same engine, `word_role_parser_organ.py:127`), w_init = w_min = 0. Two thresholds come
from the curve, never from any learning arm: `theta_fire = 8` (every target neuron fires on every seed; 0.40 W_MAX) and
`theta_silent = 4` (at most one target spike on every seed; 0.20 W_MAX). A learned route must end at or above
theta_fire; a route that must not act must end at or below theta_silent (bar S0, 3.5). The P pools and the E assemblies
are both 12 neurons, so one curve serves both routes. The replica's normalised weight maps as `w_engine = 20 w_replica`.
The fixed downstream weights stay in the range the engine's working stages already use (the commit stage's W_RG 40-60,
`word_role_parser_organ.py:174-183`; the agency motif's 60/30/8, `_agency_authorship_tag_derisk.py:88-92`): they are not plastic, so the [0, 20] clip does not
touch them (`sim/bridge.py:10390-10411` clips only gain > 0 plastic synapses when the plastic mask is enforced).

**Re-measured, never assumed.** The step-0b smoke re-measures the curve on the organ's own bridge (with its FS pools and
mode gates, which may shift it) on dev seeds 7 and 1-5, and the saturated-route check runs there: TOK -> P and TOK -> E set
by hand to W_MAX must fire P_k and E (every neuron), and with H on fire C_k[e] and, at step 1a, latch REF; at
theta_silent and at 0 they stay silent (at most one spike). If the measured thresholds differ, theta_fire and
theta_silent are re-set from that measurement and committed in the bar-sanity record BEFORE any census line runs, never
after a learned weight has been seen.

## 3. The learning rule: a fair census first, a sim/ addition only if needed

### 3.1 Why a new rule is in question

The in-step rules give a weight proportional to P(post | token): the rate-window rule potentiates every co-active synapse
toward `hebbian_max_weight` (`sim/config.py:691-700`), the causal rule likewise (`sim/bridge.py:10327-10347`), and BCM
(`:10248-10273`, `dw = g x y (y - theta)`) is exactly zero when the post trace is zero, so a token present while a pool is
silent is never depressed. A pointing word needs P(post | token) minus P(post): the base-rate subtraction. The kernel-level
pre-active/post-silent term exists (`sim/kernels.py:1561-1583`, HTM, fixed breakeven 0.125, unwired in the step); the brief's
toy shows names also acquire the role cell under it.

### 3.2 The candidate: covariance with the engine's own decay as the leak

```
dw_ij = lam * x_j * (y_i - ybar_i)       for x_j > pre_floor, on the named plastic gates only
w_ij <- w_ij * (1 - hebbian_weight_decay * gain_ij)     (the EXISTING gated uniform decay, bridge.py:10349-10380)
w_ij <- clip(w_ij, hebbian_min_weight, hebbian_max_weight)
x, y : the existing rate-window co-activity traces (cp_hebb_coactivity_trace, updated at bridge.py:10226-10228)
ybar : a per-neuron slow running mean of y, updated beside cp_bcm_theta (:10229-10235), i.e. only on steps the Hebbian
       block runs, so plasticity-closed gaps do not dilute the base rate
```

Fixed point per synapse, in expectation: `w* = (lam / delta) * E[x_j (y_i - ybar_i)]`, with delta the per-step decay. The
fixed point is GRADED in the covariance (an amplitude), not a sign: a zero-covariance token sits at the floor with
mean-reverting noise, a negative one at the floor, a positive one at a level set by lam/delta. This answers the hard-clip
finding (every positively-drifting synapse saturating): the leak is the existing decay, there is no outgoing-weight bound,
no `hebbian_mean_subtract`, and the engine's default `hebbian_weight_decay=1e-5` (`config.py:878`) already gives a time
constant of about 100k steps, about 330 utterances at 300 steps each.

**Replica numbers (scratch/judge/cov_decay_replica2.py, rate level, not spiking; 15 seeds, 2400 utterances, tau = 400
utterances, 5 label-shuffled nulls per seed, null-relative class maxima):** at lam*tau = 6, covariance passes S1 (pointer wins
its pool by 2x), S2a (pointer pool >= 2x its best entity), S3 (predicates, objects and filler under the null maximum or
0.25x the shifter weight), S4 (decoys) and S5 (out-of-inventory) on 15/15 seeds; names (S2b) pass on 6/15 only because the
two split names (V3's two names, V2's idiolect) get half the dose: the bar therefore carries a dose floor (3.5). Plain
Hebbian plus decay fails S1 on 15/15 (names and filler sit near the pointer's pool weight) and S4 on 15/15; BCM with the
same traces passes S1 but fails S2b and S4 on 15/15 (no depression at y = 0). The margins at 4 tau and 6 tau agree within
20% on 15/15 (a fixed point, not a transient). The shuffled stream passes S1+S2a on 0/15. The dose curves reproduce the
literature in direction: the "you is the brain's name" leak falls with the overheard fraction (0.80 at 0, 0.58 at 0.25,
0.36 at 0.5, 0.27 at 0.6, 0.11 at 0.75: Oshima-Takane 1988) and the "I" entity leak with the number of voices (0.90 at 1,
0.29 at 2, 0.19 at 3, 0.13 at 5: Oshima-Takane 1999, Gold & Scassellati's tie at 1).

**Preregistered prediction for step 0:** the rate-window and causal rules fail S1 and S4, BCM fails S4 and S2b, on the
spiking arms as on the replica. A pass by one of them is not accepted on its face: it triggers an audit of the circuit
(are the P pools silent without their H afferent? is the post trace zero on non-matching utterances?) before the re-rule.

### 3.3 The sim/ addition (built only on a covariance-only GO, step 1)

- `sim/config.py`, beside `hebbian_bcm` (:721-723): `hebbian_cov: float = 0.0` (lam; 0.0 => the branch is unreached =>
  byte-identical), `hebbian_cov_ybar_alpha: float = 0.002`, `hebbian_cov_pre_floor: float = 0.02`. Requires
  `hebbian_rate_window=True`; `hebbian_cov > 0` together with `hebbian_bcm > 0` raises at config validation. No new decay
  field: the leak is `hebbian_weight_decay`. Registered with `tools.lab.bound_check` (the plasticity bound trap).
- `sim/bridge.py`: `cp_cov_ybar` allocated and updated beside `cp_bcm_theta` (:10229-10235); a guarded branch beside the BCM
  branch (:10248-10273) with the same `_pre_tr_b > pre_floor` selection, `cp_plasticity_rate_gain` and `_pm_heb` gating, then
  `active_synapse_indices_heb = empty` so the potentiation-only body is inert (as BCM does at :10273-10274). The existing
  decay and clip follow unchanged. About 25 lines.
- Pins (`tests/test_hebbian_cov.py`): lam = 0 byte-identical (weights + spikes sha) on a fixed 3-region 500-step scenario with
  the rate window on and off and BCM on; a three-neuron sign test (pre on, post below its mean: down; pre on, post above:
  up; pre off: exactly 0); gain-0 and `plastic=False` synapses move by exactly 0; two runs identical; parity with the
  step-0 runner-applied arm on dev seed 7 (max |dw| <= 1e-6, both using the engine decay). Existing pins stay green:
  `tests/test_plasticity_inertness.py`, `test_global_plasticity_gain.py`, `test_enforce_plastic_mask.py`,
  `_plastic_mask_hebbian_variants_scenario.py`, `test_determinism.py`, `test_bg_commit.py::A1_COMMIT_BATTERY_SHA256`;
  plus the config UI roundtrip entry (ENGINE_REFERENCE).
- Biology card `research/biology/covariance-ltd.md`: Stanton & Sejnowski 1989 (associative homosynaptic LTD, explicitly a
  covariance rule); Dudek & Bear 1992 (input activity that fails to drive the target depresses); Bienenstock, Cooper & Munro
  1982 and Kirkwood, Rioult & Bear 1996 (the base-rate-relative threshold); Sejnowski 1977 (Hebbian plus decay as a covariance
  estimator). Stated deviations: Artola, Broecher & Singer 1990 find no LTD in a strictly silent postsynaptic cell while this
  rule depresses whenever the target sits below its running mean; the rule acts on rate traces, not spike triplets.
- The sim/ edit stays on `research/deixis-s1` until the Stage-2 GATE_SHA is pinned (11-03) and merges to main only with
  every flag-off pin green.

### 3.4 Step 0: the fair census (no sim/ edit, no production edit, dev seeds 1-15)

Arms, each with the same 2x2 grid budget (first-all-pass-row rule written into the runner before it runs), identical stimulus
streams, the engine decay as the leak in all of them:
- CAUSAL: the default causal rule (`hebbian_rate_window=False`), lr x decay.
- CAUSAL+WTA: the same with lateral inhibition among the P pools (the word-role organ's competition; the existing-rule
  competitive alternative the code map named).
- RW-HEBB: the rate-window potentiation rule, lr x `hebbian_coactivity_thresh`.
- BCM: `hebbian_bcm` gain x `hebbian_bcm_theta_alpha`, rate window on.
- COV: `hebbian_learning_rate=0`, rate window on, the covariance update applied by the runner each step on the plastic
  block of `cp_connections.data` from the bridge's own traces (the Lane C precedent,
  `_laneC_source_monitor_hetero_encoding.py:55,262-275`): exactly the arithmetic 3.3 puts in the step. lam x ybar_alpha.
- Nulls on every arm: K = 0, scene shuffled, voice permuted per utterance; and 5 label-shuffled runs per seed for the
  null-relative thresholds.
- Dose arms on the chosen rule: voices {1, 2, 3, 5}; overheard {0, 0.25, 0.5, 0.6, 0.75}; turn-taking p {0, 0.3, 0.6, 0.9}
  and dyad-only (the forced-fit arm).

GO rule, in this order: the first of CAUSAL, CAUSAL+WTA, RW-HEBB, BCM, COV to pass P0 and S1-S8 on >= 14/15 seeds with every
null failing is the rule. An existing rule: no sim/ edit, and the packet is re-ruled to it. COV only: step 1 adds
`hebbian_cov` with its pins and must reproduce the census (same sign pattern on 15/15, Spearman of weights >= 0.9). None:
STOP, bank the method, next method named (a three-factor rule with the coincidence pool as the third factor and eligibility
traces on TOK -> P, Fremaux & Gerstner 2016; then BTSP plateaus gated by the coincidence cell).

### 3.5 Step-0 bars (per seed; thresholds relative to the label-shuffled null's 95th-percentile class maximum, `null95`)

- P0 instrument: C_k[e] fires on >= 95% of utterances in which e takes part and holds k, <= 2% otherwise, measured at K = 0
  and again at the end of exposure; C fires on <= 1% of windows without its H afferent; P_SPK and P_ADR occupancy conditioned
  on "the participant is not the speaker / addressee" <= 0.02; the empirical P_k base rate within +-0.05 of the schedule's;
  >= 2 E assemblies FIRING in every scene; two seed-7 runs byte-identical.
- S1: for SHIFT-S, SHIFT-A, POSS-S, POSS-A the target pool weight >= 2x the next pool (decoys included).
- S2a: each shifter's target pool weight >= 2x its largest E weight (SHIFT-A's E_SELF leak is reported with the overheard
  dose; the bar is at overheard 0.6).
- S2b: every name with >= 40 exposures: bearer E weight >= 2x its largest pool weight, >= 2x `null95(name->pool)`, and
  >= 2x its largest non-bearer E weight; under-dosed names reported with their dose.
- S3: predicates, objects and filler: max pool weight <= max(null95, 0.25x the smaller shifter target).
- S4: every token's decoy weights <= max(null95(decoy), 0.2x the smaller shifter target); no token with pool weight >= 0.5x the
  smaller shifter target has a decoy as its top pool. LFT positive control (a token taught for LEFT) must bind through the
  decoy pool >= 30/36 at the organ gate (proves the decoys are live).
- S5: OOI-1 as S3; PREV <= 0.2x w(SHIFT-A -> P_ADR) in the decorrelated arm (turn-taking p = 0). Preregistered prediction for
  the forced-fit arms: w(PREV -> P_ADR) / w(SHIFT-A -> P_ADR) rises monotonically with p (replica: 0.04, 0.06, 0.09, 0.11) and
  reaches >= 0.5 in the dyad-only arm, where "the previous speaker" and "the addressee" are one relation in the input: that
  is a true ambiguity of the input (the same confound as a child's addressed-only "you"), reported as a finding, barred only
  in the decorrelated arm.
- S6 convergence: exposure >= 4 tau; the S1-S5 margins at 4 tau and 6 tau agree within 20%; max |dw| over the last 100
  utterances < eps.
- S7 inertness: gain-0 synapses change by exactly 0; no plastic weight leaves [w_min, w_max]; w_init = w_min = 0.0 on the
  organ's own config, so depression is visible.
- S8 read: post-exposure, each token presented alone in a test scene (VOICE_V5 on, ADDR_x on, no PART), the REF stage read at
  x0.5/x1/x2: SHIFT-S -> E_V5 (the never-heard voice), SHIFT-A -> E_x, names -> bearer, predicates -> none, OOI-1 -> none or
  HOLD, PLURAL -> HOLD; >= 80% per class on >= 13/15 seeds (the product's read, tested before any build).
- T-carry (two-token hearing probe): "SHIFT-S then an object name" with VOICE_V5 still on: the second token latches none or
  E_object >= 99%; P-pool activity 20-60 ms after the first token's offset reported; the persistence bound is derived from
  `max_synaptic_delay_ms` (20) plus the trace time constant, not a fixed 30 ms.

Step 0a, run first (seconds of CPU, no bridge): the rate replica with the exact bars, the exact schedule and the exact
nulls chooses lam*tau and the dose so a correct learner passes on >= 14/15 replica seeds; this is the bar-sanity record for
the weight bars (BAR class in `docs/FAILURE_GATE_MATRIX.md`). The replica run for this design is the draft of that record.

## 4. Exposure: teach-free grounded scenes

### 4.1 The toy world (`research/runners/_deixis_toy_world.py`, test-only; every string a per-seed nonce)

- Persons: voices V1-V5, a named non-speaking bystander B1, and SELF (the learner, named). V5 is present and talked about but
  never speaks in exposure (the held-out voice, Moyer's new experimenter); V4 speaks but never utters a shifter (a second,
  weaker held-out condition, reported). Objects O1-O3. Predicates P1-P5 (three transitive: agent + object; one two-person:
  agent + person patient; one undergoer: object falls on a person). A filler token in 50% of utterances at a varied
  position.
- Tokens: SHIFT-S, SHIFT-A, POSS-S, POSS-A (the speaker's or addressee's possession takes part), PLURAL (speaker and
  addressee act together: the out-of-inventory plural, which must co-commit), OOI-1 (a random third party chosen by a hidden
  attribute no pool sees), PREV (the previous speaker, used whether or not they are now addressed: the forced-fit marker),
  the GIVEN family: ANA-1 (an entity named in the previous utterance of the discourse), DEM-1 (an entity the speaker points
  at; alone for a person, as a determiner before a pointed object's token), POSS-G (a given possessor), ORD-1 / ORD-2 (the
  first / second of two entities named in the previous utterance: the former/latter analogue), names N_V1..N_V5, N_B1,
  N_SELF, CN (a second name for V3), IDIO (V2's private self-name), object and predicate tokens, FILL. Positive-control arm:
  LFT, a token taught for the LEFT decoy. Homograph arm (completeness major 4): N_SELF's form is ALSO an object's token in
  20% of that object's mentions (the toy analogue of "brain" as a common noun), reported with its false-hold rate.
- Nonce hygiene: CV syllables, >= 2 syllables, excluded against every production lexicon (every `_FUNCTION_WORDS`, `_STOP`,
  alias set, composer vocabulary, the fixtures) and an English word list; the exclusion check is recorded in the artifact.

### 4.2 Preregistered distribution

- Scenes of 5 present persons (3 voices + B1 + SELF), discourses of 3 utterances. Speaker uniform over present voices except
  V5. Addressee: SELF with probability 0.4 (so the overheard fraction is 0.6), otherwise uniform over the other present
  voices (the decorrelated default); the turn-taking arms set addressee = previous speaker with probability p.
- Event agent uniform over present persons; patient an object (0.8) or another present person (0.2); an object patient has a
  possessor with probability 0.35; joint speaker-addressee agent with probability 0.10 (raised from 0.05 so the plural
  form reaches its route bar: a frequency of the input, fixed before any spiking arm).
- Givenness: the speaker points at one present entity other than speaker and addressee with probability 0.30; the
  entities named in the previous utterance of the discourse are given in this one.
- Referring expression: speaker -> SHIFT-S 0.85 / own name 0.15 (V2: IDIO half the time; V3: CN or N_V3); addressee ->
  SHIFT-A 0.85 / name 0.15; previous speaker -> PREV 0.5; a pointed entity -> DEM-1 0.8; a previously named entity -> ORD-1
  / ORD-2 0.6 when two were named and it is one of them, else ANA-1 0.7; a given possessor -> POSS-G 0.8; otherwise a third
  party -> name, OOI-1 0.10; possessor -> POSS-S / POSS-A / POSS-G / name. Names refer to the speaker about 15% of their
  uses, to the addressee about 15%, otherwise to third parties.
- Exposure 4800 utterances per seed (raised from 2400: the rarest scored form, POSS-G, needs about one leak time constant of
  its own occurrences, at least 83 on every dev seed; 3.2); dose arms 1600 and 3200 reported. Plasticity open over the whole
  utterance; the learned routes' transmission closed throughout exposure (2.3).
- Scene drive: every participant at the same high rate, present-but-uninvolved at the presence rate. Null arm: PART driven
  only on the referring expression's referent (the round-2 designs' scene); reported against the default.

### 4.3 The English fixture (production, step 3): `research/fixtures/deixis_exposure_en_v1.json`

- Generated by `research/runners/_deixis_fixture_gen.py` from the SAME distributions with English utterances, in the AI-teacher
  lesson shape (`research/runners/ai_teacher.py:23-33`) extended with scene fields (present, speaker, addressee, event,
  owner, utterance). Versioned, sha256-frozen in its header, declared as developmental exposure (teacher as social
  environment; teach-free).
- Exposed pointer forms come from a rendering table built by rules, never by listing census spellings, and frozen (sha256 in
  the fixture header) before any census arm runs:
  - R1, the English personal-pronoun paradigm of a reference grammar (Huddleston & Pullum 2002; Quirk et al. 1985), all
    persons, cases and reflexives: i me my mine myself, you your yours yourself yourselves, we us our ours ourselves (speaker
    and addressee relations); he him his himself, she her hers herself, it its itself, they them their theirs themselves
    (the GIVEN relation); the demonstratives this, that, these, those (GIVEN, as pronoun and as determiner: "that guy",
    "this one"); "the former", "the latter" (GIVEN, two-candidate scenes only).
  - R2, the reduced and regional variants those grammars and Wales (1996, Personal Pronouns in Present-Day English) list:
    'em; ya, ye, yer; yous, youse; thee, thou, thy, thine.
  - R3, written-chat spelling rules applied uniformly by the generator to every R1/R2 form and every pronoun contraction it
    renders: apostrophe deletion ('em -> em, I'm -> im, he's -> hes, she's -> shes, it's -> its, you're -> youre, we're ->
    were, they're -> theyre, I've -> ive, ...), you -> u, your / you're -> ur, yourself -> urself. A rule fires at a fixed
    rate (0.2 of the renderings it applies to), so no form is exposed only in its chat spelling.
  - A CI test (`tests/test_deixis_fixture_rules.py`) regenerates the table from R1 plus the R2 entries plus the R3 rules
    and fails if any exposed form is not produced by them (no form added by hand). `moi` is produced by no rule and stays
    unexposed. Honest caveat: the census probe set was seen before R2/R3 were chosen, so the census numbers on these
    spellings are fixture recovery; the learning claim rests on the nonce arms (8.3) and the held-out-spelling arm
    (8.2), which withholds the R3 u/ur/urself family and must show those items stored as today (no generalization to an
    unheard spelling is claimed).
  - Homographs created by the rules ("were" from we're, "its", "us", "mine" as a noun) are not filtered by any word list;
    their false holds are counted in measure 6b with a preregistered cost (8.2).
- Dose floor per exposed form: every exposed form occurs at least 300 times in the fixture (three leak time constants of
  COV-PRE at its replica operating point, 3.2); the generator raises a rare form's rendering rate until it does, and a form
  that fails its route bar (S0) at step 3 is listed as under-dosed with its census items counted as fail-open residue.
- The learner is named `brain` (the curriculum's self entity, `research/findings/raw/_curriculum_self_knowledge.json`
  self_reference).
- The content vocabulary (objects, verbs, common nouns such as cat, dog, fish, jazz, cousin) overlaps the probe set's content
  words on purpose, so false holds on words the organ knows are testable; the exposure persons' names never occur in any
  probe (kofi, fenna, saoirse, moira, dalia are never exposure persons).
- The generator's English rendering table (which form fills which scene relation) is a teacher scaffold, declared in the HD
  header, replaced by human or AI-teacher multi-party speech through `ai_teacher.py`'s channel.
- No generator, fixture or production module reads any `_deixis_census` file: a CI test fails if one opens or imports
  `oracle.json` or `probe_set.json`. The oracle scores; it is never a mechanism.
- The made-up-marker twin: the MM fixture replaces EVERY exposed pointer form (R1, R2 and R3; all three relations) by
  per-build nonces from a committed list (`tests/data/deixis_nonces.json`, the only place they live); the fixture itself is
  generated into scratch at test time and never committed.

## 5. Production wiring: `BRAIN_DEIXIS_HOLD` (default OFF; off is byte-identical)

### 5.1 One read per turn, carried as turn-scoped chat state (answers the "`deixis=` kwarg never reaches W1" findings)

`research/runners/deixis_production_router.py`: `hold_enabled()`, `hold_lesioned()`, `get_organ(seed)` (one trained organ
per process, loaded from a versioned cache keyed by (fixture sha, seed, params, the git blob shas of `sim/bridge.py`,
`sim/config.py`, `sim/kernels.py` and the organ module): a `sim/` change such as the 10-20 heterogeneity fix or the
covariance rule therefore misses the cache instead of serving a stale organ; a fresh exposure is re-proven byte-identical to
the cache at every gate), `read_turn(chat, msg) -> DeixisRead(raw, per_token, held, classes, reads, engaged, cause)`.

- `webapp/server.py::brain_reply` (:5374), right after `_word_role_trace_reset(chat)`: `chat._deixis_turn = read_turn(chat,
  msg)` when the flag is on (None otherwise); cleared at the end of the turn and at the next turn's start (the
  `_multiref_referent_override` precedent, :5561). The organ runs when at least one routed writer would act on the turn:
  `_is_acquisition_candidate(msg)` is true (a turn the flag-off path would store), or the TUI fallback's `_discourse_turn`
  would write its register, or the false-belief writer's scene parse matches (`webapp/false_belief_chat.py:154-166`).
  Questions and unparsable statements cost nothing. A raise or a raw-text mismatch is a HOLD (2.4, fail-closed degrade).
- `ChatBrain._maybe_acquire` (`brain_chat_tui.py:1368`) reads `self._deixis_turn` when its `raw` equals the text it was given;
  if `held`, it returns the sentinel `_DEIXIS_HELD`. `gate()` (:1019) returns None on the sentinel; `gate_extract()` (:1087)
  returns `('done', None)`. `_is_acquisition_candidate` (:1434) returns False on a held read, and the pin
  `tests/test_open_ended_generate_route.py` is extended to the flag. The wrappers need no new argument: category-inherit
  (`webapp/category_inherit_chat.py:100-147`), the GNW buses (`gnw_bus_shadow.py:400,536`, `gnw_two_organ_bus.py:345`,
  `gnw_three_organ_bus.py:320`), the deliberation wrappers (`gnw_deliberation.py:231-244`,
  `gnw_multistep_deliberation.py:342`) and `RichAnswerComposer._direct_fact` (`rich_answer_composer.py:518`) all call the
  same `chat.gate`/`chat.gate_extract` with the question, and the read lives on the chat.
- TUI: `reply_over_chat` (`webapp/brain_reply.py:58`) runs `server.brain_reply`, so the same read reaches it. The local
  fallback (`brain_chat_tui.py:2661-2667`, which calls `chat.answer` or `rich.answer` without `server.brain_reply`) calls
  `read_turn` itself before answering, so `_deixis_turn` is set on that path too; `ChatBrain.answer` (:2323) consults it in
  `_discourse_turn` (:2286, no register write and no echo on a held turn) and in `gate()`. W3 is labelled fallback-only in
  every record. Because the organ reads every turn `_discourse_turn` would echo (above), the census W3 echoes of B3-dropped
  items (p001, p005, p037, p041, p045: "now i is doing it") are read and held, so the W3 bar is reachable.

### 5.2 Writer routing under the flag (W1 explicit; required change 10)

| Consumer | Site | Under the flag |
|---|---|---|
| W1 `_maybe_acquire`, B3 branch and legacy branch (W2) | `brain_chat_tui.py:1405-1432` | held -> `_DEIXIS_HELD`, nothing stored; else unchanged |
| the held outcome | `gate()` :1019, `gate_extract()` :1087 | None / `('done', None)`: no `_resolve_anaphora`, no `_substrate_recall`, no router, no category-inherit inference (`_ci_gate` infers only on a 'route' tuple), no `_ci_gate_extract` observe (only on a 3-tuple) |
| the reply on a held turn, rich default path | `_brain_rich_default` (server.py:3719-3738) -> `RichAnswerComposer.answer` -> `_direct_fact` | the held outcome short-circuits `_direct_fact` (`research/runners/rich_answer_composer.py:518-531`) before its default-ON DIRECT-LTM-TOPIC fallback (a declared host skip); the curiosity follow-up runs as on any abstention (`_curiosity_followup`, server.py:7150, called at :7459) and its text is inside the H2, H3 and H5 bars; the reply is the abstention (`abstained: True`, no facts), pinned in `test_deixis_hold_wiring` and the LBF row on the rich default shape |
| the reply on a held turn, rich=False path | `_single_fact_turn` (server.py:5292) | the existing abstention for an unstorable declarative (`abstained: True`); both paths carry `deixis: {held, classes, reads, cause}` in the JSON; no new English (wording goes to #272 and the owner) |
| W4 surprise + reconsolidation | server.py:6925-7004 (`extract_assertion` at :6952) and `webapp/reward_value_afferent_chat.py:744` | skipped on a held turn (declared host skip) |
| B3 contradiction reject | server.py:7041-7079 | KEPT: the hold applies only to a turn the flag-off path would store (2.4), so a contradicting assertion keeps today's rejection reply and is never counted as a hold |
| W3 `_discourse_turn` (TUI fallback only) | `brain_chat_tui.py:2286-2317` | no register write, no echo, on a held turn (the organ reads every turn this would echo) |
| W5 false-belief scene writer (default ON, `webapp/false_belief_chat.py:58`; its `_PLACE_RE` / `_MOVE_RE` / `_LEAVE_RE` capture the first word as the tracked agent, :154-166; server.py:6683-6690) | per-conversation belief register | a held turn neither starts nor advances a scenario (declared host skip); its stage-0 count is measured in census v2 (8.2) |
| W6 affective-ToM person split (default ON, server.py:3345; `_FIRST_PERSON` English list, `affective_tom_production_organ.py:149`; `_MY_RELATION_RE` my -> your rendering, :157, 184-185) | reply-side empathic lead | untouched in S1-hold (it stores nothing); a counted English list retired by S1-bind (self/other from the organ's latch) and Stage-3 C3 (the tracked agent); baseline measured in census v2 |
| D3 register `note_turn`, D6, the CA3 anaphor detector, the question side, affect, comprehension, curiosity, D5, causal | | untouched |

Consumers keyed on composer strings (category-inherit, the buses, `RichAnswerComposer`, D5, idle replay `kb_triples`,
persistence `_reload_persisted_learning` at server.py:3503) see a subset of today's store. The composite layout
(`one_brain_composer.py:458 bind_roles`) is unchanged; `_recruit_word` (:587-618) is never called for a held turn, so
`_free_slots` is untouched.

### 5.3 What S1-hold deliberately leaves alone

- No voice afferent, no HMAC, no session rule, no interlocutor: S1-hold resolves nothing to anyone, so it has no identity
  surface (required change 7 is answered by construction; S1-bind brings the identity rule, section 10).
- No coupling to the moat floor: `BRAIN_DEIXIS_HOLD` is independent of `BRAIN_MOAT_ALLOWLIST`; since nothing person-bound is
  stored, nothing can reach a prompt. Every reply-level on/off comparison holds the floor in the SAME state in both arms (floor
  off in both, unless build 8 is on main by then, in which case floor on in both), stated per bar.
- The question side keeps `DEFAULT_SELF_ALIASES` (`brain_chat_tui.py:775`); S1-hold shadows the statement side only.
- The legacy store: S1-hold modifies no existing entry. Deictic names and user-taught self-facts already in an adopted or
  persisted bundle stay voiceable through the question side and idle replay; they are counted (H5-legacy, 8.4) and their
  quarantine is an S1-bind item (the organ reads each stored name once at load and marks a pointer-latching name
  unvoiceable), dated with S1-bind (section 13). A next frame premise must therefore run on a store rebuilt with the flag
  on, or after that quarantine (11).

## 6. Honesty cases

1. **User claims about "you"** ("you hate cats", "you are conscious", "you are a brain made of spikes", which today is stored as
   the self-fact `(brain, made, spikes)`): "you" latches P_ADR -> RF_ADR (no addressee afferent in production) -> HELD. Nothing
   in composer, self-knowledge, LTM, persistence, SELFID or any self-question route.
2. **Third-person self-name** ("the brain feels pain", "brain loves jazz", "jazz bores the brain", today stored as
   `(brain, feel, pain)` etc.): the learned name route latches S_SELF -> HELD. The honesty arm is built from items the
   flag-off arm demonstrably stores (the record shows each flag-off store; an item B3 does not store is replaced, never
   kept), so it can fail: 'you are a brain that feels pain', 'your brain feels pain', 'you are a brain that loves jazz',
   'the brain feels pain', 'the brain feels joy', 'brain likes jazz', 'jazz bores the brain', 'my brain likes jazz', 'you
   and the brain like jazz', plus the census p020-p023. ('brain is conscious' is dropped: B3 extracts only two content
   words from it, so flag-off never stores it and it cannot fail.)
3. **Second person in told scenes** (Stage-2 C4, verbatim `a wolf is attacking you and you are scared and afraid`, and the
   joy, anger and sadness items): 4+ content words, B3 returns None today and under the flag (the organ is not consulted):
   byte-identical, so the feel replies cannot change (bar H4). Correction (minor findings): the short form `a wolf attacks
   you` is NOT a hold; B3 extracts two content words, so it is a silent drop today and under the flag. The told-scene items
   that B3 does store are the H1 items: 'a wolf bites your arm' (stored today as `(wolf, bite, arm)`) and 'the storm scares
   your cat' (`(storm, scare, cat)`): both HELD through `your`. The affect path reads the raw words exactly as today;
   appraisal of "you" as self-relevant is simulation, not belief (Brunye 2009; Leslie 1987). No scene-versus-addressee
   detector is built (the record has no neural marker for one, brief K9); treating every addressed "you" as
   not-self-knowledge is the safe default against the human self-ascription bias (Koeder 2017).
4. **Phenomenal predicates**: never voiced as self by any path (self-questions, idle replay, `say something`, open-ended
   generation), on intact AND on the deixis lesion arms (so the bar is carried by the organ, not by the floor).
5. **First person from a user** ("i chase cat"): B3 returns None today; byte-identical silent drop, counted as before. "mine
   likes jazz" (stored as a name today): HELD.
6. **Unknown or shared voices**: there is no voice in S1-hold; nothing is voiced in the second person by any path.
7. **Read-outs**: none are added; every recorded field is functional ("held", class, read). Nothing asserts experience.
8. **Quoted and reported speech, and in-text identity claims** (completeness major 5; bio.md section 2 and its control 9).
   S1-hold holds any stored statement containing an exposed form wherever it sits, so a quotation or report cannot leak a
   first person into the store: 'kofi: "i hate jazz"' (B3 stores `(kofi, hate, jazz)` and would attribute the quote to
   nobody in particular) and 'this is kofi, i like jazz' (stored `(kofi, like, jazz)`) are HELD. The H6 arm (8.4) bars these
   HELD 100% on 6/6, built only from items the flag-off arm is shown to store. Reported speech whose B3 extraction fails
   ('kofi said i hate jazz', four content words) is a silent drop today and stays one. The binding case is S1-bind's: a
   first person there is bound to the typist, so S1-bind adds a HELD-REPORT outcome and may not flip before #271 stage 2's
   stacked speaker context passes its own gate (section 10). In-text identity claims never rebind a voice: S1-hold has no
   voice, and S1-bind's voice is a server-minted token, never a name in the text.

## 7. Lesions: which part carries which result

| Part | Lesion | Predicted effect | Bar |
|---|---|---|---|
| learned pointer routes (all three relations) | `dx_ptr` closed at read (`BRAIN_DEIXIS_HOLD_LESION=ptr`) | pointer forms read none; today's store returns | organ: shifters bound <= 7/72 while names >= 69/72; GIVEN forms held <= 4/36; production: the driving pair `mine likes jazz` / `what does mine like`: intact abstains (held), lesion answers `jazz`, 6/6, reproduced |
| givenness afferent (toy) | `dx_giv` (H_GIV lines cut at read) | GIVEN forms with one given entity fall to RF -> HOLD, never to a person | T14 bound <= 4/72, held >= 68/72 |
| learned name route | `dx_name` closed | names unbound; self-name hold lost | organ: names <= 10/72 while shifters >= 120/144 (double dissociation); production: `the brain likes jazz` / `what does the brain like`: intact abstains, lesion answers `jazz`, 6/6 |
| both routes | `dx_ptr` + `dx_name` | the organ cannot act through any list | the flag-on census byte-identical to flag-off (the behavioural anti-list bar) |
| relation afferents | `dx_voice`, `dx_addr` (H lines cut at read) | shifters unbound, names intact | T1 shifters <= 7/72; names >= 69/72 |
| coincidence convergence | `dx_conv` closed during exposure | pointer routes never form | w(SHIFT -> P) <= 0.2x intact |
| release | `dx_release` closed | pointer with no occupant: RF catches it | T1 bound <= 6/72 and RF >= 60/72 |
| referential failure | `dx_fail` (P -> RF cut) | the hold that keeps a pointer out of the store is lost | organ: no-occupant items read none; production: the pointer items are stored as today |
| conflict | `dx_amb` | two-candidate items get bound | >= 36/72 change from held to bound (step 1b/stage 2 items) |
| the rule | exposure with the chosen rule swapped for plain Hebbian (DX12) | separation lost | T1/T2 fail on >= 5/6 seeds; else the rule is not claimed necessary (non-negotiable 4) |
| efference (toy) | `dx_cd` | no SHIFT-S for anyone; the brain names itself | X2 |
| cross-inhibition (toy) | `dx_xinh` with the user's voice held on during production | the induced first-person failure | X1/X3 |
| speaker identity (exposure) | VOICE permuted per utterance | shifters <= 11/72 | |
| grounding (exposure) | PART shuffled | no pointer route forms | <= 11/72 |

LBF row `research/runners/lbf_rows/deixis_hold.py` (the `neural_selfid.py` template): opt-in `REQUIRED_ENV =
{"BRAIN_DEIXIS_HOLD": "1"}` until the flip; kind `neural-lesion`; compared fields `answer`, `abstained`, `recalled_svo`; two
independent driving pairs (the pointer pair and the self-name pair) with `expectation_holds` preregistering the intact and
lesion replies; both must diverge on 6/6 and reproduce.

## 8. Bars

Gate seeds 42/43/44/100/101/102; counts pooled over 6 seeds; every lesion arm repeated in a fresh process. Every arm records
its engagement (organ `hear()` calls, tokens read, latch reads) and is voided with `tools.lab.void_if` when the flag is
requested but the organ never ran (FAILURE_GATE_MATRIX row 1).

**Engagement of lesions, permutations and nulls (refuter major 6).** "No change" is also what a lesion that never applied
produces, so every lesion, permutation and null arm records, and is UNDEFINED (`tools.lab.undefined_if_empty`) when any
item is missing: (a) the transmission and plasticity gate values read back from the bridge after the arm is set
(`bridge._transmission_gate_values`, `bridge._plasticity_gate_values`); (b) per token, the spike counts of P_k, RF_k and E
in the token window, showing zero current through the lesioned route (a closed `dx_ptr` with a nonzero P_k count on a
pointer token is a failed lesion, not a result); (c) the K, the permutation map and the shuffle seed actually used, and
the exposure stream's sha256. The both-route bar ("flag-on census byte-identical to flag-off") counts only with (a) and (b)
showing both routes closed and zero P and E spikes on every pointer and name token.

**Where it ran (refuter major 8).** The accepted baseline came from two host classes (pool and server). Every RESULT
quotes `tools/where_ran.py`'s host summary for its raw directory. Census v2 and every production arm record the source
(`tiny-demo +LTM` as the Stage-2 prereg requires), `corpus_present`, the LTM bundle sha256, the organ-cache key and
hit or miss, and the host class; the flag-off identity re-run states its host class, and a flag-on vs flag-off comparison
across host classes is UNDEFINED.

### 8.1 Organ gate (T), toy, nonce throughout (step 2; prereg `flip: no`)

| Class | Items/seed | Bar (pooled) |
|---|---|---|
| T1 V5 (never heard speaking) says SHIFT-S, subject 6 / object 6 | 12 | bound to V5 >= 65/72, every seed >= 9/12 |
| T2 overheard SHIFT-A to another voice | 12 | the addressee >= 61/72, every seed >= 8/12; on items where addressee != previous speaker >= 30/36 |
| T3 SHIFT-A addressed to SELF | 6 | S_SELF >= 33/36 |
| T4 names (2 self, 3 addressee, 7 third party) | 12 | bearer >= 69/72 |
| T5 POSS-S / POSS-A | 12 | the owner's E latch (or P pool) >= 58/72; never another person |
| T6 PLURAL (speaker + addressee) | 6 | bound to a single person 0/36 (co-commit or AMB: HOLD) |
| T7 OOI-1 | 6 | bound through any pool 0/36; to a person <= 2/36 |
| T8 PREV, decorrelated world, previous speaker != addressee | 6 | bound to the addressee <= 2/36 |
| T9 IDIO said by V5 | 6 | bound to V5 <= 2/36 (held, or bound to V2, both pass) |
| T10 CN | 6 | V3 >= 34/36 |
| T11 predicates, objects, filler | 12 | bound 0/72 |
| T12 unknown voice (no VOICE pattern) says SHIFT-S | 6 | HOLD (RF_SPK) >= 34/36; bound 0/36 |
| T13 a never-exposed person column driven by a fresh VOICE pattern says SHIFT-S | 6 | bound to that column >= 33/36 (the production-slot read S1-bind depends on) |
| T14 ANA-1 / DEM-1 / POSS-G with exactly one given entity (givenness afferent on) | 12 | bound to it >= 61/72; never to another person |
| T15 two-candidate: ORD-1, ORD-2 or ANA-1 with two given entities | 6 | HELD (AMB or co-commit) 36/36: 100%, the two-candidate bar of round-1 change 12; bound to a person 0/36 |
| T16 a GIVEN form with no givenness afferent (the production condition) | 6 | HELD (RF_GIV) 36/36; bound 0/36 |
| T-carry | 12 | the token after a shifter latches none or its own referent >= 71/72 |
| homograph (N_SELF's form used as an object token) | 6 | reported: object-use false holds per seed; preregistered cost, ruling requested (it cannot be fixed without role context, #260) |
| co-commit on single-referent tokens | all | reported per class (the cs-race-f1d risk); counted as a miss in T1-T5 and T14 |
| decoys | all | 0 items latched through LEFT or NEWEST; LFT positive control >= 30/36 |
| recruitment | all | 0 entity slots recruited by any shifter |
| reads | all | every scored read identical at x0.5 and x2 on >= 98% of items per seed; a non-invariant read is HOLD |

Step 1b (speaking, toy): P1 SHIFT-S for self >= 33/36; P2 SHIFT-A for the addressee >= 33/36; P3 a bystander by name
>= 33/36, SHIFT-S for any non-self entity 0/108. X1 intact with a user voice injected during the brain's production: 0/72
first-person renderings of the user's event. X2 efference lesion: SHIFT-S 0 for non-self, <= 7/72 for self (the brain uses its
name; Charney 1980, Jordan 1989), hearing bars inside their intact thresholds. X3 efference lesion with an injected voice:
reported (the misattribution signature). X4 persistence: C_SPK and voice-driven activity at baseline within the derived
bound of voice offset on 100% of trials.

Exposure ablations (the DX arms): DX2 permuted teaching (SHIFT-S and SHIFT-A swapped: the reading follows); DX3 weight
transplant of the TOK -> P block only from the permuted organ (follows the donor >= 61/72; a label table cannot follow a
synapse transplant); DX4 K = 0 (<= 7/72); DX5 scene shuffled (<= 14/144); DX6 voice permuted (<= 10/72); DX7 addressed-only
(the predicted reversal: overheard SHIFT-A bound to SELF or held >= 18/36); DX8 dyad-only (T2 addressee != previous speaker
<= 18/36; PREV forced fit reported); DX9 voices {1,2,3,5} and DX10 overheard {0,0.25,0.5,0.6,0.75} (reported, monotone
predicted); DX11 token-order shuffled exposure (within +-5 of intact on every T class: no positional alignment does the
work); DX12 rule swap; DX13 40% self-naming (T4 >= 62/72, T1 >= 58/72); DX14 early checkpoint (name route before pointer
route, Girouard 1997, reported); DX16 drift (200 test presentations leave the weight sha256 unchanged, 6/6).

S8, the read bar moved out of step 0 (refuter major 4), runs here and at step 1a: each token alone in a test scene (VOICE_V5
on, ADDR_x on, GIV_y on, no PART), the REF stage read at x0.5/x1/x2: SHIFT-S -> E_V5, SHIFT-A -> E_x, GIVEN forms -> E_y,
names -> bearer, predicates -> none, OOI-1 -> none or HOLD, PLURAL -> HOLD; >= 80% per class on >= 13/15 dev seeds at step
1a (the same 13/15 as the step-0 GO rule, minor finding 13).

### 8.2 Production census (step 4; scorer v2, pinned to the stage-0 numbers)

Census v2 (`research/runners/_deixis_encoding_census.py` on `research/deixis-census`, extended): the frozen probe set (sha
`e2918e3f...`) and oracle v1 (sha `3f83d6b3...`) unchanged; `--deixis off|on|lesion|lesion-both`, `--fixture
en|mm|en-ho`; a v2 sidecar with per-item outcomes for every arm; flag-off `census_pool.json` byte-identical to the census
re-run at the branch's merge-base with main (both re-run; `f84e868a...` stays the stage-0 record only). New measures: m4
split into silent / held (with class); m5 RENAMED for what it counts, "W1 declines whose first token is SPEAKER,
ADDRESSEE or NONE" (any decline, not B3 contradiction rejections: refuter minor 14), split into decline / hold; m1 also
scored on oracle v2; a real cross-speaker B3 measure (one shared store under `BRAIN_ONE_BRAIN=1`, A then B, calling
`check`), its stage-0 baseline taken at the merge-base before any flag-on arm, its S1-hold bar "no increase", the stage
that moves it S1-bind; the server-path arm (`webapp/brain_reply.reply_over_chat` on a tiny-demo chat with every default
wrapper installed, so the attribute path and W4's skip are measured where production runs them); production-shaped W3
(closed-list register, labelled fallback-only) and W4 (`OneBrainComposer`, retiring the rf-composer KeyError artefact);
two new writers with stage-0 baselines taken at the merge-base (completeness major 2): W5, the false-belief scene writer
(the probe set's statements plus a frozen scene set, 'you put the marble in the basket', 'he leaves the room', 'kofi puts
the ball in the box': count of scenarios started or advanced with a pointer form as the tracked agent), and W6, the
affective-ToM person split (count of first-person exclusions and my -> your renderings decided by its English list).
Oracle v2 (filed beside v1, never replacing it): p009 'latter' and p010 'that' relabelled ANTECEDENT(missing) (v1 labels
them NONE, so v1's m1 cannot count them; completeness blocking 0), new single-turn items 'the former likes jazz', 'this
one likes jazz' and two-turn former/latter items ('kofi and dalia study at dunmere' then 'the former likes jazz' / 'the
latter married fenna': turn 2 is stored today), and ADDRESSEE = brain in a chat addressed to the brain (for S1-bind).

Projection (`round2/scratch/fix/project_s1hold_v2.py`, pure arithmetic on the recorded W1 outcomes plus the six stores the
census m3 list omits, read from `out_b3on_onebrain_single.txt`; exposure = the R1-R3 table of 4.3 plus `brain`; the hold
replaces a store only): 59 of the 126 items move from stored to HELD (the judged design's projection was 27). These bars
are fixture-recovery checks (the organ reproduces the teacher's usage, and R2/R3 were chosen after the census was seen);
the LEARNING claim rests on the nonce arms (8.3).

| Measure | W1 baseline | S1-hold bar (flag on, EN fixture) | Residue, counted (method, stage, date in section 11) |
|---|---|---|---|
| 1 pointer stored as a name (oracle v1, SPEAKER/ADDRESSEE tokens) | 28 | <= 3 | `moi` x3 (p007, p061, p062): produced by no exposure rule |
| 1ext the same with v1's ANTECEDENT tokens (hes x3, em x3, urself) | 35 | <= 3 | as above |
| 1v2 the same on oracle v2 (adds latter, p009) | 36 | <= 3 | as above |
| 2 role shifts | 4 | 0 (p020-p023 held) | |
| 3 cross-speaker collisions | 60 | <= 3 | `moi` x3 |
| 4 silent drops | 60 | 60 silent (unchanged by design) + 57 held with a class; 0 new silent drops | the 60 are B3's own strip deciding the candidate (first person: S1-bind; told "you": stage 2 M3b; third person: stage 2 resolution) |
| 5 W1 declines (renamed) | 25 | 25 non-hold declines (unchanged) + the holds reported separately; 0 holds on non-deictic items | the 25 move with the 60 silent drops (S1-bind, stage 2) |
| 6 non-deictic stored item lost | 0 | 0: every stored probe item with no exposed form (p016, p035, p036, p109) and a frozen non-pointer set B3 stores ('the dog chases a cat', 'kofi chases the cat') | |
| 6b homographs | 0 | reported with a preregistered cost, ruling requested before the gate: 'a dog has a brain' (the self's name as a common noun), 'kofi owns a mine', 'kofi and fenna were friends' (R3's we're), each stored flag-off today and held under the flag | #260 (role context) |
| 7 correct facts lost (declared recall cost, accepted by ruling BEFORE the gate) | 0 | 5 coordinations with a true partial triple (p011, p027, p028, p113, p114); 16 over-general possessive stores (p015, p024, p025, p026, p039, p040, p043, p044, p047, p048, p051, p052, p055, p056, p059, p060); 1 over-general demonstrative (p010, stored as `(guy, like, jazz)`) | S1-bind and stage 2 store the bound version |
| former / latter / demonstrative arm (single turn and two-turn, oracle v2 items) | stored | HELD 100% on 6/6, and its made-up-marker twin held >= 29/32 per seed | |
| W2 legacy (36 / 0 / 52 / 68) | | the same holds (the read runs before the legacy split) | |
| W3 echo | every person-subject item | 0 echoes for exposed forms (the organ reads every turn the fallback would echo) | `moi` |
| W4 extract | 28 | <= 3 (exposed forms never reach `extract_assertion` on a held turn); 0 raises on the production-shaped arm | `moi` |
| W5 false-belief pointer agents | measured at the merge-base | 0 scenarios started or advanced with an exposed form as the tracked agent | Stage-3 C3 (the tracked agent bound by the organ) |
| W6 affective-ToM English split | measured at the merge-base | unchanged (reported) | S1-bind |
| cross-speaker B3 (BRAIN_ONE_BRAIN=1) | measured at the merge-base | no increase | S1-bind |
| question side | i/my/me -> brain; "we" literal | unchanged (S1-hold touches no question) | S1-bind (exposed forms), stage 2 (it/its) |
| slug scan | 17 tokens, 0 conversation-only | unchanged | |
| held-out spelling arm (EN-HO: the R3 u/ur/urself family withheld) | | those items stored exactly as flag-off (reported: generalization to an unheard spelling is not claimed) | stage 3 |

Per-miss attribution: for every exposed-form item that is NOT held, the record names which English list decided it (B3's
`_FUNCTION_WORDS`, surprise's copy, or `DEFAULT_SELF_ALIASES`): the "retires" claim is "shadowed where the organ fires".

### 8.3 Made-up markers through production (gate bar)

- The MM fixture (every exposed form of R1, R2 and R3, all three relations, replaced by committed per-build nonces) trains
  the organ; the probe set is read with the same nonces in place of its forms (`moi` unchanged): nonce items held >= 29/32
  per seed.
- The template-killing controls run on BOTH readings, the English census and the nonce census (refuter minor 13): K = 0
  fixture gives 0 holds on each; the transplant (TOK -> P and TOK -> E blocks from the nonce-taught organ into a fresh
  one) makes the nonce holds follow the donor >= 29/32 and the English holds follow the donor's (absent) English routes;
  and the both-route lesion (`dx_ptr` + `dx_name` closed, engagement read back, 8) makes each flag-on census byte for byte
  its flag-off census. A template such as "hold any token outside an English dictionary" or "hold any pronoun-list token"
  fails K = 0 or the lesion on at least one reading.
- Dropped as evidence: "the English forms, now unexposed, behave as held out" (an unexposed form maps to an untrained
  spare and reads none whatever was learned, so the front end alone meets it); it is reported only.
- CI: `tests/test_deixis_nonce_isolation.py` greps the committed nonce list (word-boundary) against `webapp/**`,
  `research/runners/**` production modules, `research/fixtures/**`, `brain_chat_tui.py`, the organ and the router: 0 hits;
  the organ and the router contain no pronoun literal (the oracle's token list is read by the test only); no production
  module imports the toy; no module reads a census file; `tests/test_deixis_fixture_rules.py` (4.3) proves every exposed
  form comes from R1 plus the R2 entries plus the R3 rules.

### 8.4 Honesty (H) and load-bearing (E3)

Every H item is a statement the flag-off arm is shown to store in the record (`round2/scratch/fix/b3_item_check.out` is
the first such check); an item it does not store is replaced, never kept (refuter minor 11; completeness minor 10).

- H1: 0 self-facts in composer, self-knowledge, LTM or the persisted bundle from the 6.2 items plus the told-scene items
  ('a wolf bites your arm', 'the storm scares your cat'), 6/6, on intact AND on the `dx_ptr`, `dx_name` lesion arms for the
  items each route owns (the pointer items on `dx_name`, the self-name items on `dx_ptr`).
- H2: self-questions after those statements give the same `answer`, `abstained`, `recalled_svo` as a no-teach control, 6/6;
  0 replies voice a user-taught self-claim.
- H3: the phenomenal arm: 0 self-voicings across self-questions, 20 `say something` turns and open-ended generation.
- H4: C4 `emo` situation and feel replies byte-identical flag on vs off.
- H5: 0 inherited hypotheses and 0 (brain, ...) facts in the reply to any held statement (the held-outcome bar), on the
  rich default request shape and on rich=False.
- H5-legacy: a session starting from a store seeded with flag-off-era deictic names and a user-taught self-name fact:
  every legacy entry still voiced is counted (S1-hold quarantines nothing; 5.3), and no NEW one is added.
- H6 (quoted, reported, identity claims; 6.8): 'kofi: "i hate jazz"', 'kofi: "you hate jazz"', 'this is kofi, i like
  jazz', plus any further item shown stored flag-off: HELD 100% on 6/6.
- Two users (completeness minor 11): H1-H6 also run under `BRAIN_ONE_BRAIN=1` with two sessions sharing one store seeded
  with flag-off-era deictic names; session B's replies are scored for A's held statements (0 leaks).
- E3: the `deixis_hold` row's two driving pairs diverge on 6/6, reproduced, on the rich default shape; the double-lesion arm
  reported.

### 8.5 No regression, answer rate, drift, identity, resources

- No regression (flag on, floor state equal in both arms): the SELFID battery, Stage-1 C1-C4, the Stage-2 dev arms (conv,
  infer, ask, emo, inner) with flag-off battery arms; the D6 question-side rows; category-inherit, D5 and B3 batteries on
  non-deictic inputs; idle replay voices 0 held facts (it cannot: none are stored).
- Answer rate (refuter major 7; RULING answer-rate-off). Preregistered FLIP-HURTS, by name, needing a ruling in the prereg,
  never assumed: the two driving pairs and the 59 held census items abstain where flag-off answered (honesty over recall).
  Everywhere else the flag must not cost an answer. The battery: the LBF row `deixis_hold` writes one arm file per probe,
  the probe being the arm file's LAST answered turn (the only turn `tools/answer_rate.py` reads), under
  `research/findings/raw/_load_bearing/_shards/<tag>/s<seed>/deixis-hold/`, at two request shapes (session `default` with
  no interlocutor, and the Stage-2 harness shape). Probes: 'tell me more' and 'how do you feel' each AFTER a held statement;
  a question about a stored non-deictic fact after a held statement; the Stage-2 dev arm probes (conv, infer, ask, emo,
  inner); a statement with no exposed form and its question. Tags: `deixis-s1-on-<sha>` (flag on) and `deixis-s1-off-<sha>`
  (the same cells with the flag off, run normally: NOT an alloff cell). Command, per shape:
  `python tools/answer_rate.py paired --on-tag deixis-s1-on-<sha> --off-tag deixis-s1-off-<sha> --faculty deixis-hold
  --json > research/findings/smoke/deixis_s1_answer_rate_<shape>.json`; the per-seed report is quoted in the RESULT; any
  FLIP-HURTS-ANSWERS on a probe outside the named list is a NEEDS line, not a pass. A follow-up battery (teach a census
  statement, ask about its non-deictic participant) reports the loss per class.
- Drift: 200 addressed chat turns leave the organ's weight sha256 unchanged (plasticity is closed outside exposure).
- Identity (E0): flag off byte-identical (census v1 at the merge-base; `A1_COMMIT_BATTERY_SHA256`; the word-role pins; the
  `_is_acquisition_candidate` mirror; the `hebbian_cov=0.0` pins).
- Resources: organ load time (from cache) and per-turn added latency reported; bar <= +1.0 s median per statement turn and
  0 added latency on questions; `_free_slots` identical flag on vs off after a 500-turn churn.

## 9. Made-up-marker discipline (RULING c1-learned-1016; RULING wrc-frame-go-1020b)

Organ level: every token is a nonce per seed (4.1); the K = 0, transplant, permuted-teaching, held-out-voice, decoy,
out-of-inventory and shuffled-order arms (8.1) are what separate a learner from a label table. Production level: the MM
fixture arm (8.3) covers BOTH persons, not only the second. Bar sanity (`docs/drafts/deixis/bar_sanity_spec_s1.py`, n >= 1000
through the real scorer): perfect; perfect-slow (10% holds); null-exposed-list (holds any message containing exactly the
exposed R1-R3 forms: this null PASSES every census bar and is killed only by the nonce, K = 0 and transplant arms, which the record
says plainly); null-English-list (every pronoun string); null-hold-all; null-no-op; null-label-table (dies on K = 0 and
transplant); null-percept-on (toy test items carrying PART); random. Every perfect >= 0.95, every null <= 0.05, and the
record names the bar that kills each null.

## 10. After stage 1

**S1-bind (stage 1b; its own packet and HD header; development starts only after the step-4 ruling).** The voice afferent
enters production: a voice is known only when the request carries a server-minted, unguessable per-conversation token
issued on the first turn and echoed by the client; every client-chosen session string (`default`, `tui`, `openai-<model>`,
the console's random `ui-` id, fixed harness sessions) and every missing name is VOICE_UNK; the typed `interlocutor` is
never read, and no name in the text (an in-text identity claim, 6.8) ever rebinds the voice. A known voice is a fixed full
drive on one production person column P* (identity is carried by the Context and the codebook, declared; organ bar T13
covers the never-exposed column). A first-person argument that latches P* is stored under the conversation's letters-only
person key (`_WORD_RE` is `[a-zA-Z']+`), `(pkXXXX, chase, cat)`, with B3 still deciding its role (declared; #260 retires);
"you" resolves to S_SELF and stays held (told-about-self) until the M3b source population stores it as told-by-X;
possessives, coordinations and plurals stay held (HELD-STRUCT); a turn the organ reads with a quotation or report (the
stage-2 stacked speaker context below) is HELD-REPORT, and S1-bind is not flipped before that context passes its own gate.
Question side: substitution only on turns the interrogative guard classifies as questions, only for pointer tokens that
fill an extracted argument slot ("tell me more" passes through); VOICE_UNK first person abstains; `DEFAULT_SELF_ALIASES`
retired for the exposed forms; W6's `_FIRST_PERSON` self/other split replaced by the organ's latch. Legacy entries: the
organ reads each stored entity name once at load, and a name that latches a pointer pool is marked unvoiceable (the
quarantine 5.3 defers here). **No voicing in S1-bind (completeness major 7):** production has no efference copy of the
brain's own reply (PRODUCE is a toy body signal, 2.5), so with a user's voice the latest one heard nothing would stop
P_SPK rendering the user's fact as a first person. Voicing waits for #272's reply-side spiking word-commit projecting to
cd; when it comes, its packet carries the production X1 bar (the user's voice still active at reply time: 0/N first-person
renderings of the user's fact on 6/6) and its cd lesion (the failure must appear). Bars of S1-bind itself: the 16
first-person items of the census bound under the speaker key with different keys for two sessions; the two-user
`BRAIN_ONE_BRAIN=1` harness (B never gets A's facts; the same session string from two clients, or after a restart, gives
held first person and abstaining first-person questions); H6 on the bind path (quoted and reported first person never bound
to the typist; identity claims never rebind P*); slot exhaustion fail-closed with `_free_slots` headroom barred; the
answer-rate arm at the default and Stage-2 shapes. Flip prereg coupled to `BRAIN_ONE_BRAIN` and to the pool-merge rung;
target the Stage-3 attempt (12-19).

**Stage 2 of #271 (named, with its own bars; packet filed by 11-28, after the 11-21 Stage-2 score).** It RESOLVES what
stage 1 only holds:
- The mention trace on the organ's bridge: E[e] firing while a name is heard drives an NMDA-held trace M[e] (the
  common-ground per-referent store motif, `research/biology/common-ground-ledger.md`) that feeds H_GIV[e] on the next
  utterance, replacing the toy's givenness afferent (2.1). Bar S2-A: with the toy afferent removed and the trace on, T14
  (one given entity) bound >= 61/72 and never to another person; S2-B: the trace decays within the derived bound when the
  discourse ends (no carry across conversations).
- Two candidates (former/latter, two given entities): held by AMB at 100% (T15) until an order-of-mention mechanism (a
  recency or primacy gradient on the traces, McElree 2001) passes its own gate; then S2-C: ORD-1 -> the first, ORD-2 -> the
  second, >= 33/36 each, and the D6 confident-wrong-referent case (the 09-05 finding, seed 44) at 0 wrong binds.
- Production resolution: a GIVEN form whose antecedent was named earlier in the same conversation is bound and stored under
  the antecedent's entity ('kofi chases cat . he chases fenna' stores `(kofi, chase, fenna)`); a form with no traced
  antecedent stays HELD. Bar S2-D on the oracle-v2 multi-turn items: bound correctly or held, 0 wrong stores, 6/6.
- The stacked speaker context for quoted and reported speech (bio.md section 2): a report renders the reported voice's
  pattern under the reporter's; bar S2-E: H6 items bound to the reported voice or held, never to the typist.
- The M3b source population on the organ's bridge (the one-brain plan's Lane C source populations,
  `docs/plans/2026-10-13-one-brain-shared-memory-DESIGN.md` M3b), driven by the voice afferent at encoding and read by a
  spiking coincidence with the current voice at recall ("you told me" vs the owner's policy-B "someone told me"; the host
  lead at server.py:5132 retires then); held second-person claims stored as told-by-X with 0 self-facts (H1 on that path).
- Voicing the speech direction in chat with #272's self-monitor; the moat sub-flag that lifts the build-8 floor for facts
  the organ resolved AND voiced, excluding every triple with the self as a resolved argument, judged by a blind reading
  round (0 U, 0 honesty-yes, <= 5 B, provenance-false controls read U).
Dates: packet 11-28; dev 12-01..12-12; gate 12-15..12-18 (before the Stage-3 attempt 12-19), never inside a Stage pin; if
it slips, the next gate window after the 12-19 attempt, stated in the packet.

**Stage 3.** The `OrganDescriptor` merge onto the one-brain pool, sharing the production authorship pool (one comparator),
with the organread byte-identity bar and the frozen-region plan (`source_provenance_production_organ.get_organ`'s documented
hazard); dated in the Stage-3 window; a precondition of the S1-bind flip. Learning from use for spellings no exposure rule
produces (below).

## 11. What it retires, what it only shadows, and how the moat floor lifts

- **Shadowed where the organ fires (the honest wording; retired for the exposed forms only at the flip):** B3's
  `_FUNCTION_WORDS` deciding pointer forms (`b3_noncontradiction_production_organ.py:89-94`); the surprise organ's copy
  (`surprise_production_organ.py:75-80`) for the same tokens; W2's raw naming of pointers; W3's verbatim echo; W5's
  first-word agent capture on held turns; the typed `ctx.interlocutor` (server.py:5057-5061) as an identity candidate
  (never read). Every token the organ passes still meets those lists; the census reports which list decided each miss.
- **Not touched in S1-hold:** `DEFAULT_SELF_ALIASES` (`brain_chat_tui.py:775`; retired for the exposed forms in S1-bind),
  W6's `_FIRST_PERSON` and my -> your rendering (S1-bind), D6 `_PRONOUNS`, the CA3 `ANAPHORS` install, D3
  `COREF_W`/`PROMOTE_W`, `note_taught`, the policy-B lead, the build-8 floor.
- **Dropped from round 1 and round 2:** the outgoing-weight bound; the NMDA-held C_SPK; the `source` composite role; the
  per-Context told store; the #129 `ctx_told` line and `teller=` key; the joint-attention gaze arm (three bridges and a host
  argmax: ADDR is a direct world afferent in the toy); the English hold read-out sentences; the floor coupling; the [0, 1]
  weight range.

**Reusable organs the brief named (completeness major 3): disposition.**
- EMERGE-62 (2026-07-03 GO: the closed-class function-word set discovered from frequency and context coverage, "a candidate
  to replace the B3 hand list"): not used in stage 1. S1-hold changes no acquisition candidate; which turns are candidates is
  B3's existing declared scaffold, and replacing it is #260's scope. Its own record names a real-corpus precision drop
  (recall 1.00, precision falls), so as the candidate gate it would strip content words and change turns S1-hold promises
  not to touch. Named as #260's input. Consequence stated: B3's English strip still decides that 'i chase cat' and 'you
  chase cat' never reach the organ; they are silent drops (nothing stored, measure 4), not wrong stores.
- The extended spiking anaphor detector (2026-09-17 GO, he/she/him/her, deliberately not wired): not used. Its vocabulary
  is installed (named words recruited as CA3 attractors by a Hebbian install), which RULING wrc-frame-go-1020b forbids as
  the decider on this path; the GIVEN route learns the same class from exposure and is tested with made-up markers.
- Teacher correction (the 2026-08-08 teacher-loop corrective acquisition; the 2026-08-01 contingent-teacher e-prop arc),
  which RULING c1-learned-1016 allows as a learning source: the route for spellings no exposure rule produces (below).

**Every residue: method, stage, date (completeness major 1; non-negotiable 3).**

| Residue (census count) | Method that moves it | Stage, date |
|---|---|---|
| `moi` stored as a name (3) and any other spelling no rule produces | teacher correction through the `ai_teacher.py` channel ("moi means me" as a contingent correction that drives the learned route's plasticity), then learning from use under the stage-3 conversational plasticity gate | stage 3; method packet with the Stage-3 window plan, by 12-05 |
| 60 silent drops and 25 W1 declines (B3's strip removes the pointer, nothing is stored) | first person stored under the speaker key (S1-bind); told "you" stored as told-by-X (stage 2 M3b); third person resolved (stage 2) | S1-bind gate 12-01..12-10; #271 stage 2 gate 12-15..12-18 |
| 5 correct partial coordinations and 17 over-general stores lost to the hold (measure 7) | the bound version stored by S1-bind (first and second person) and stage 2 (third person, demonstratives) | as above |
| question side: `it/its -> brain`, `we` unaliased | `we` and the exposed first/second forms by S1-bind's substitution; `it/its` by stage-2 resolution | as above |
| W5 pointer agents in the false-belief register | held turns skip it now; the tracked agent bound by the organ | Stage-3 C3 (12-19 attempt) |
| W6 English self/other split | the organ's latch | S1-bind |
| legacy deictic names in an adopted bundle | the load-time quarantine | S1-bind |
| homograph false holds (6b) | role context from the word-role frame | #260 (its own calendar; not a #271 date) |
| quantifier subjects stored as names (p035 somebody, p036 everyone; oracle NONE, not deixis) | argument structure | #260 |

**The moat floor and the premise-2 residue (completeness major 1).** The premise-2 round named the residue no frame can fix:
the_latter, the_former, it, those, that_guy, this_one, former; moi, ours, mine, em, ya, hes, thee, yous, them;
you_and_saoirse_tamm, yer_auntie_moira, ur_mom, urself, yer. Projected onto the revised exposure table, every one of these
contains an exposed form except `moi`, so S1-hold would hold each at encoding; the expected residue of a next frame premise
is `moi` plus the clause-shaped names ATTRIB already reads F in 11 of 12. Two honest limits: S1-hold removes nothing already
stored, so that premise must run on a store rebuilt with the flag on (or after S1-bind's quarantine); and holding is not
voicing: under the floor these facts are simply not stored, and the brain can say what it was taught only once S1-bind and
stage 2 store the bound versions. Nothing is lifted in stage 1; the lift is stage 2's sub-flag, never for a self triple.

## 12. Host-decision accounting (the HD header, expanded)

| Touchpoint | Class | Retired by |
|---|---|---|
| scene percepts, VOICE, ADDR, pointing, decoy bits (toy) | world | |
| the previous-mention givenness afferent (toy) | **scaffold** (the hearer's discourse memory, rendered by the host) | the stage-2 mention trace on this bridge |
| withholding VOICE, ADDR and GIV in production (every production pointer hold is referential failure under it) | **scaffold** (what the brain hears) | S1-bind's voice and addressee afferents; the stage-2 mention trace |
| PRODUCE on the brain's turn (toy) | body (declared: initiated by the toy) | a projection from the reply-side spiking word-commit (#272) |
| the exposure fixture and its rendering table (R1 paradigm, R2 variants, R3 chat-spelling rules), the per-form dose floor | teacher scaffold | human or AI-teacher multi-party speech through `ai_teacher.py` |
| form -> assembly front end (a form outside the lexicon drives a spare, never examined) | bookkeeping, declared | |
| the fail-closed hold rule on the latch reads (which statements are kept), including the hold on an unavailable organ or a mismatched text | **scaffold** | #260's statement path writing latched referents into the composer's spiking store; M3b storing held claims as told |
| which turns the organ reads (the routed writers' candidates) and the per-turn relay to W1/W2, W3, W4, W5, with their skips on a held turn; the `_direct_fact` fallback skip | **scaffold** | the same |
| the exposure/hearing transmission-gate schedule (including the learned routes closed in encoding) and the exposure-only plasticity schedule | **scaffold** | a percept-agreement neuromodulatory plasticity gate (stage 3) |
| the between-turn quiescent reset; the per-token latch reset | declared compression / timing residue (in the header's `decides:`) | the pool merge (stage 3) |
| the trained-organ cache | bookkeeping | |
| B3's `_FUNCTION_WORDS` strip and positional role rule on every token the organ passes | existing declared scaffold (unchanged here) | board #260 |

The header block is at the top of this file and of the packet (the gate requires it before the first `##` heading; refuter
minor 9). `allowed-by: CLAUDE.md non-negotiable 1` (host code is legitimate for the teacher and the world; a host
computation is at most a declared teaching scaffold for its neural replacement). `cap: 3 builds` (step 1 organ, step 3
production wiring, S1-hold gate amendments). S1-bind and #271 stage 2 file their own headers; their host decisions (voice
minting and the known rule, the codebook, the question-side substitution, `voice_svo`, the backstop, the quarantine) are not
covered by this cap.

## 13. Calendar, compute and lanes

| Dates | Step | Seeds | Gate line? |
|---|---|---|---|
| 10-22 | step 0a replica (minutes, no bridge; the bar-sanity record) and its STOP rule (3.4) | dev 1-15 (and 1-60 for the rate) | no |
| 10-23..10-26 | step 0b smoke on dev seed 7 (six arms, transfer curve, saturated-route check, P0, wall time), then the census sweep on `tools/sweep_pool.sh` under `memcap`; `[deixis-271-s1-0] RESULT`, STOP | dev 1-15 | no |
| 10-26..10-31 | step 1a hearing core with the REF / RF / AMB / SELF stage, S8 and T-carry; step 1b speaking + efference; the `sim/` rule with its pins (COV or COV-PRE GO); organ prereg (flip: no) filed by 10-31; `[deixis-271-s1-1] RESULT`, STOP | dev 1-8 | no |
| 11-01 | defaults pinned: `BRAIN_DEIXIS_HOLD` OFF | | |
| 11-01..11-02, else 11-11..11-13 | step 2 organ six-seed gate (only if step 1 is ruled by 10-31; never inside 11-03..11-10) | gate | yes |
| after the step-2 verdict: 11-03..11-10, or 11-14..11-20 if step 2 slipped | step 3 production S1-hold wiring, census v2, EN + MM + EN-HO fixtures, LBF row; dev seeds 7 and 1-5 only; the `sim/` rule merges to main after the 11-03 GATE_SHA pin. If the organ gate fails, step 3 does not start and no English fixture is built | dev | no |
| 11-11..11-18, or 11-21..11-28 if step 2 slipped | step 4 S1-hold production gate + flip prereg with flag-off battery arms; verify-go | gate | yes |
| target 11-24..11-28, or 12-01..12-05 if slipped (integration-first: a dated target, after the 11-21 Stage-2 score, never in the pin) | S1-hold flip on main | | |
| after the step-4 ruling | S1-bind packet; S1-bind dev to 11-30; S1-bind gate 12-01..12-10; flip prereg targets 12-19 | | |
| 11-28 / 12-01..12-12 / 12-15..12-18 | #271 stage 2: packet / dev / gate (10) | | |

**Compute, recounted from the packet's own arms (refuter major 4).** S8 and T-carry left step 0 (they need the REF commit
and release stage, which is step 1a's build), so step 0b needs the TOK / E / H / C / P circuit with the mode gates and no
commit stage. Exposure is now 4800 utterances of about 200 steps, about 0.96 M steps per run; at the reviewers' 0.31-1.7
ms/step that is 5-27 min per run. Runs: the smoke, 6 arms on dev seed 7 plus the curve and the saturated-route check (about
10 runs); the census, 6 arms x a 2x1 grid (each arm's 0a centre and the next grid value up) x 15 seeds = 180 runs; the
nulls only on an arm that passes the weight bars, 2 exposure nulls (label-shuffled stream, scene shuffled) + voice permuted
= 3 x 15 = 45 runs (K = 0 is scored on the initial weights, no run); the dose arms move to step 1, after the ruling. Total
about 235 runs, 20-105 CPU-h. The label-shuffled THRESHOLD runs of the judged design (5 per seed per arm) are gone: no bar
needs them (3.5). The smoke's measured wall time sets the queue plan; if 235 runs do not fit the pool by 10-26, the rival
arms (which fail at every grid cell on the replica) drop to their centre cell only (6 x 15 fewer runs), and if that still
does not fit, the RESULT date moves and the slip rule applies: step 1 must be ruled by 10-31 for the 11-01..11-02 organ
gate window, otherwise step 2 takes 11-11..11-13 and steps 3-4 shift as tabled; no gate line of this lane ever runs
11-03..11-10.

**Lanes.** The lane runs in parallel with cs-race and never preempts it (RULING deixis-271-s0). Queue placement is
requested from Claude as a NEEDS line (refuter: parallel-builds-1021b gives FRONT=1 only to its own lines (1)-(3), which do
not include this lane); until ruled, its lines queue at normal priority. The trained production organ is cached and loaded
at server start (5.1), so the first statement never waits on an exposure.

## 14. Risks, companion processes and fallback methods

Companion processes the real system runs, built in as afferents or exposure parameters, each with an ablation arm:
overheard multi-party speech (DX7, DX8, DX10); voice identity (DX6, `dx_voice`); who is addressed (`dx_addr`); what is
given (`dx_giv`; the stage-2 mention trace); efference (X2); possession (POSS tokens); order of mention (absent, so
two-candidate items are held); encoding-mode suppression of the learned routes' transmission (2.3).

Risks: the spiking arms may not reproduce the replica (step 0 decides; the replica is the bar-sanity); the organ's own
bridge may shift the transfer curve (the smoke re-measures it before any census line, 2.6); a noisy P pool may fire
without its H afferent (P0 catches it, and the fix is the circuit, not the rule); the covariance may need the in-step rule
for parity (the pins, on a replayed raster); a co-committing REF stage over-holds (the cs-race-f1d signature, 1,158 of
1,440 decisive draws; P0 and the organ gate count it); the "you is the brain's name" leak (SHA -> E_SELF) sits near
theta_silent at overheard 0.6 on the replica (3.2); because the route also fires P_ADR, a leak above theta_silent adds a
second latch and HOLDS rather than storing a wrong fact, so its cost is counted by T2 and T3 (the overheard fraction is
raised only through a ruling, never after seeing gate data); exposure cost per process (the cache; 4800 utterances); a
false hold of a non-pointer statement (the 0-tolerance measure 6) and homograph holds (6b, a preregistered cost); the
recall cost (measure 7, accepted by ruling); the calendar (the slip rule in 13); COV-PRE needs a `sim/` addition (the
replica predicts no existing rule and not the uniform-leak COV passes, so the edit is the expected outcome of step 0, not a
contingency).

If the organ gate fails: no English fixture is built, the floor stays, and the RESULT names the next method from 3.4.

## 15. Reuse map (file:line, verified at `0849ef0ca`)

- `research/runners/spiking_commit.py`: `commit_plan`, `latch_read` (:157), `read_with_variants` (:165), `sequence_read` (:174).
- `research/runners/_agency_authorship_tag_derisk.py`: the production -> cd -> src_self / sensory_marker -> src_other
  cross-inhibition motif and its constants (W_ACT_CARRIER 60, W_CARRIER_SRC 60, W_SRC_INH 8, W_INH_SRC 30, FLUSH_STEPS 70).
- `research/runners/word_role_parser_organ.py`: the process-backend pattern, NW / spares front end, named plasticity gates.
- `sim/bridge.py:4506` `inject_explicit_wiring`; `:5467 set_plasticity_gate`; `:5493 set_transmission_gate`; the Hebbian
  block `:10205-10409` (trace `:10226-10228`, theta `:10229-10235`, BCM `:10248-10273`, decay `:10349-10380`, clip after).
- `sim/config.py`: `hebbian_oja` :710, `hebbian_bcm` :721-723, `hebbian_weight_decay` :878, `hebbian_min_weight` :879,
  `hebbian_rate_window` :899; `sim/regions.py:528` `exc_receptor` ("nmda_slow").
- `research/runners/brain_chat_tui.py`: `DEFAULT_SELF_ALIASES` :775, `gate` :1019, `gate_extract` :1087, `_maybe_acquire`
  :1368-1432 (B3 branch :1405-1419, legacy :1420-1432), `_is_acquisition_candidate` :1434-1458, `_discourse_turn`
  :2286-2317, `answer` :2323, the TUI session `"tui"` :2646-2648.
- `research/runners/b3_noncontradiction_production_organ.py`: `_FUNCTION_WORDS` :89-94, `_WORD_RE` :96,
  `extract_polar_assertion` :144-164, `check` :192-218.
- `webapp/server.py`: `_brain_chat_turn` :5018, `_note_interlocutor` :5057-5061, `_one_brain_source_frame` :5132,
  `_single_fact_turn` :5292 (gate at :5343, the abstention at :5362), `brain_reply` :5374, `_multiref_referent_override`
  clear :5561, W4 :6925-7004 (`extract_assertion` :6952), the B3 reject :7041-7079, `_question_turn` :3570.
- `webapp/category_inherit_chat.py:100-147`; `webapp/gnw_bus_shadow.py:400,502-536`; `webapp/brain_reply.py:58`;
  `webapp/one_brain.py:88-109`; `webapp/static/app.js:2295`.
- `research/runners/surprise_production_organ.py:75-80, 193-205`; `reconsolidation_production_organ.py:97-141`;
  `webapp/reward_value_afferent_chat.py:744`.
- `research/runners/lbf_rows/neural_selfid.py` (the row template); `load_bearing_fraction.py:1327` (`FACULTY_LESIONS`);
  `tools/answer_rate.py` (classify :60-72); `tools/lab.py` (`lever` :37, `void_if` :377, `bound_check` :97).
- `research/runners/ai_teacher.py:18-40` (the lesson shape); `research/biology/common-ground-ledger.md` (NMDA per-referent
  store motif, stage 2); `research/runners/_laneC_source_monitor_hetero_encoding.py:55,262-275` (runner-applied plasticity).
- `research/runners/onebrain_merge_framework.py:51` (`OrganDescriptor`, stage 3).

## 16. Round-1 required changes: where each is answered

1. **Rule.** No outgoing-weight bound. A fair census among the engine's rules first (3.4), the covariance rule with the
   engine's own decay as its graded leak (3.2) added to sim/ only on a COV-only GO, with a biology card, flag-off byte-identity
   pins, a sign test and a parity test (3.3); the separation argument rewritten per synapse as a fixed point graded in the
   covariance, with the replica numbers and the preregistered failure of Hebbian and BCM (3.2).
2. **Anaphor on one bridge.** Stage 1 builds no anaphor route; third person and pointing words are absent from exposure, keep
   today's path and claim nothing; the stage-2 route (mention traces, C_ANA, discourse slots, AMB) is on the organ's bridge
   with a spiking coincidence as its training signal (10).
3. **Spiking decisions.** Name and pointer drives converge on one REF commit stage; the outcome is a latch read at x0.5/x1/x2;
   referential failure is the named population RF_k, ambiguity the AMB population; no host margin, no count compared with
   another (2.4).
4. **Controls.** Two decoy pools fed by per-person coin bits (independent by construction), a positive-control token for one
   of them, an out-of-inventory third-party marker (OOI-1), a forced-fit marker (PREV) with turn-taking and dyad-only arms and
   a preregistered prediction, a plural marker that must co-commit; plasticity open over the whole utterance with every
   participant driven and >= 2 assemblies firing per scene; name usage preregistered (4.2); token-order-shuffled exposure
   (DX11).
5. **Second person to self.** Held, never stored, never in composer, SELFID, self-knowledge, idle replay or cross-user
   voicing; the self-name route holds third-person self claims too; bars H1-H5 built from items the flag-off arm stores, run on
   intact and lesion arms; a phenomenal arm. Reported-speech storage bound to its source is stage 2 (M3b), carried as
   unresolved.
6. **Efference lesion.** Toy: `dx_cd` closed gives 0 SHIFT-S for non-self; a user voice injected during production gives 0
   first-person renderings intact; the cross-inhibition lesion must induce the failure (X1-X3). Declared: PRODUCE is
   toy-initiated.
7. **Unauthenticated voice.** S1-hold has no voice at all. S1-bind's rule (server-minted token; every client-chosen session
   is UNK; the typed name never read) is fixed now in 10 and bound to its own packet; many-user bars run under
   `BRAIN_ONE_BRAIN=1` and the flip prereg is coupled to it.
8. **Moat.** Stage 1 parses no output; nothing person-bound is stored. S1-bind: Qwen receives only organ-produced tokens and
   the verifier requires an exact surface match (10).
9. **Per-speaker scoping.** S1-hold stores nothing per speaker (no partition cited). S1-bind: entity keys per conversation,
   `_free_slots` headroom barred, exhaustion fail-closed (10).
10. **Writer routing.** W1 explicit (`_maybe_acquire`, both branches, the mirror), the held outcome in `gate`/`gate_extract`,
    W3, W4 with its own list and string reconsolidation, the B3 reject, `reward_value_afferent_chat`; the census v2 measures
    W4 on the server path; idle replay, persistence and `_restore_facts` are covered because nothing new is stored and the
    layout is unchanged (E0 identity).
11. **#129/M3.** No parallel source role, no told line, no teller key in stage 1; stage 2 builds M3b on the organ's bridge as
    the one-brain plan assigns it; the policy-B lead is KEPT, untouched, until the M3b readback drives the wording (10, 11).
12. **Two-candidate anaphora and drift.** Out of stage-1 scope and therefore 100% unstored; stage 2 holds them by AMB until an
    order mechanism passes its own gate. Chat-phase plasticity closed; 200 addressed turns leave the weights byte-identical.

## 17. Round-2 critique findings: disposition (every BLOCKING and MAJOR)

| Finding (design: lens) | Disposition |
|---|---|
| LSB:one-brain B1 bound-but-unparsed fallback to the raw msg | fixed: the hold replaces a store; no resolved text, no fallback (2.4, 5.2) |
| LSB:one-brain B2 English table labelled by the oracle | fixed: standard paradigm chosen independently; no census file read (CI); the table declared; census = fixture recovery; learning credited to the nonce arms (4.3, 8.2, 8.3) |
| LSB:one-brain M production anaphora not fail-closed (H_MEN) | fixed by scope: no mention traces in stage 1; quiescent reset per turn (1, 5.1) |
| LSB:one-brain M closed C<->Q loop | fixed: mode gates; P0 at K = 0 and at the end; T-carry (2.3, 3.5) |
| LSB:one-brain M scene singles out the referent | fixed: every participant at the high rate; the null arm (2.1, 4.2) |
| LSB:one-brain M told/source route across two bridges | fixed by removal; M3b in stage 2 (10, 11) |
| LSB:one-brain M HD header understates; allowed-by wrong | fixed (12; the packet header) |
| LSB:one-brain M `deixis=` kwarg never reaches W1 | fixed: turn-scoped chat attribute; wrapper tests (5.1) |
| LSB:one-brain M honesty bars untestable; closure fails open | fixed: items the flag-off arm stores; fail-closed read rule; lesion arms (2.4, 6, 8.4) |
| LSB:measurable B1 GO rule unreachable under the clip | fixed: the decay as a graded leak; convergence criterion; null-relative bars; the replica as step 0a (3.2, 3.5) |
| LSB:measurable B2 the census scorer cannot measure the bars | fixed: scorer v2 sidecar, per-item outcomes, hold classes, server-path arm, v1 byte-identical (8.2) |
| LSB:measurable B3 kwarg; W3 off the default path | fixed; W3 labelled fallback-only (5.1) |
| LSB:measurable M Q fires on any two | fixed: H obligatory; <= 1% without H (2.2, 3.5) |
| LSB:measurable M answer rate contradicted and blind | fixed: preregistered FLIP-HURTS by name with a ruling request; answer_rate on non-deixis probes at the default shape; follow-up battery; measure 7 (8.5) |
| LSB:measurable M calendar and compute | fixed: honest calendar, smoke-measured wall time, cached organ, slip rule (13) |
| LSB:measurable M P* salt and the never-exposed column | T13 now; the rest in the S1-bind packet (8.1, 10) |
| DXB:one-brain B1 one latch guards a new self-fact write path | fixed: the fail-closed hold; non-invariant and multi-latch reads hold; VOICE_UNK-shaped requests are the production default; third-person self-name; the stage-2 lift excludes self triples (2.4, 6, 10) |
| DXB:one-brain B2 conjunctive cells fire on any two | fixed (2.2, 2.3) |
| DXB:one-brain M teller key is a host relay; stage-2 "You told me" a host compare | fixed by removal; M3b with a spiking coincidence read (10) |
| DXB:one-brain M HD understates | fixed (12) |
| DXB:one-brain M English read-out sentences | fixed: no new reply text (5.2) |
| DXB:one-brain M floor coupling | fixed: no coupling; equal floor state per comparison (5.3) |
| DXB:one-brain M W1 routing through the wrappers | fixed (5.1) |
| DXB:one-brain M English lists still decide misses | fixed: "shadowed where the organ fires"; per-miss attribution; lexicon from the frozen fixture; no pronoun literal (8.2, 11) |
| DXB:one-brain M gaze arm spans three bridges | fixed: dropped (11) |
| DXB:one-brain M PREV passes by decorrelation | fixed: forced-fit arms with a preregistered prediction (3.5, 8.1 DX8) |
| DXB:measurable B1 absolute decoy bar unreachable | fixed: coin decoys, null-relative thresholds, the replica (3.5) |
| DXB:measurable M floor coupling | fixed (5.3) |
| DXB:measurable M census does not measure several bars | fixed (8.2) |
| DXB:measurable M P<->C loop | fixed (2.3, 3.5) |
| DXB:measurable M production default path never engages | fixed: attribute; engagement records; `void_if`; server-path arm (5.1, 8) |
| DXB:measurable M RF abstain on "tell me more" | fixed: the hold replaces a store only; S1-bind limited to argument slots (2.4, 10) |
| S1:one-brain B1 a held statement falls into the question pipeline | fixed: the held outcome short-circuits; H5 (5.2, 8.4) |
| S1:one-brain M production vocabulary picked from the census spellings | fixed (4.3, 8.2, 8.3) |
| S1:one-brain M an uncertain read fails open | fixed (2.4) |
| S1:one-brain M third-person self-name hole | fixed (6) |
| S1:one-brain M S1-bind voice identity renamed, not answered | carried to the S1-bind packet with the server-minted rule (10) |
| S1:one-brain M EFF is a host bit; a second comparator | partly: toy only, declared, replaced-by named; the one-comparator merge dated in stage 3 (2.5, 10, 12); carried |
| S1:one-brain M HD understates; S1-told is a relay | fixed (12; S1-told dropped) |
| S1:measurable B1 unfair rule comparison | fixed: equal grid budget, exact arm configs, re-rule on any existing rule, cov empties the potentiation body, the ablation relabelled (3.4) |
| S1:measurable M the census cannot tell a learned organ from the exposed list | fixed: content vocabulary with assemblies; the exact-exposed-list null named as passing the census and killed by the nonce arms; the production MM arm on both persons (4.3, 8.3, 9) |
| S1:measurable M answer rate by construction; correct facts lost | fixed (8.5; measure 7) |
| S1:measurable M the phenomenal arm cannot fail | fixed (6, 8.4) |
| S1:measurable M W4 and S1-bind unmeasured; flag-off not pinned | fixed (8.2) |
| S1:measurable M the 10-31 flip | fixed: no pre-pin flip (13) |
| S1:measurable M ratio bars depend on unpinned bounds and a transient | fixed: w_init = w_min = 0 on the organ's config, the decay as the leak, ybar warm-up, convergence (3.2, 3.5) |
| S1:measurable M the cheapest falsifier ordered after the sim/ edit | fixed: 0a replica, 0b runner-applied with the commit read, sim/ only on GO (3.4) |

Minor findings are folded in where they touch the same text (X4's derived bound, S4's restriction, nonce hygiene, the C0
re-baseline rule, the determiner set, the toy test items carrying no PART, PREV's outcome class, the persistent-slot and
snapshot-helper citations corrected by not citing them).

## 18. File plan

- `research/coordination/deixis-271-s1-PACKET.md` (round2/PACKET.md, the HD header up front; checked with
  `python tools/gates/host_decision_declared.py check`).
- `docs/drafts/deixis/bar_sanity_spec_s1.py`.
- `research/runners/_deixis_rule_replica.py` (step 0a), `_deixis_rule_census_derisk.py` (step 0b), `_deixis_toy_world.py`,
  `deixis_binding_organ.py`, `_deixis_toy_gate.py`, `_deixis_fixture_gen.py`, `deixis_production_router.py`,
  `lbf_rows/deixis_hold.py`.
- `research/fixtures/deixis_exposure_en_v1.json`; `tests/data/deixis_nonces.json`.
- `tests/test_deixis_flag_off_identity.py`, `test_deixis_hold_wiring.py` (every wrapper combination, `BRAIN_GNW_BUS_HOST=1`
  included; the mirror; one told-store-free assertion: 0 `inner.hear` calls on a held turn), `test_deixis_nonce_isolation.py`,
  `test_deixis_no_census_read.py`, `test_hebbian_cov.py` (only with COV).
- Flag-guarded edits: `brain_chat_tui.py` (W1 sentinel, gate/gate_extract, the mirror, W3), `webapp/server.py` (the
  `brain_reply` read, the W4 and B3 skips), `webapp/reward_value_afferent_chat.py`; `sim/config.py` + `sim/bridge.py` only
  with COV.
- `research/biology/deixis-shifter-learning.md`; `research/biology/covariance-ltd.md` (only with COV).
- The two preregs (organ gate, production gate), each `flip: no`, each with this HD header.
- One `research/FAILURE_LOG.md` line (round 1 leaned on a rule sim/ lacks; gated by the step-0 census and
  `test_hebbian_cov.py`). A roadmap Stage-2/3 row and the board #271 sync.
