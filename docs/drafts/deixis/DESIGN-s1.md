# #271 stage 1 (round 2, synthesis): learned shifter routes on one bridge, integrated hold-first (S1-hold), then bind (S1-bind)

Claude, judge and synthesizer, 2026-10-21. Inputs: the accepted stage-0 census (RULING deixis-271-s0-accept; W1 with B3 on:
28 pointer words stored as names, 4 role shifts, 60 cross-speaker collisions, 60 silent drops, 25 declines), the refuted
round-1 design and its critique, the fresh record, the biology brief, the three round-2 designs (learning-first LSB,
substrate-first DXB, integration-first S1-hold/S1-bind) and their six adversarial critiques, and the binding RULINGS
c1-learned-1016, wrc-frame-go-1020b, hd-gate-1021. Code was read at main `0849ef0ca` (workstation checkout), read-only.
Nothing in any repo was edited. Scratch: `docs/drafts/deixis/s1-scratch/judge/` (a rate replica of the step-0 world and bars, and a
census projection; both run, numbers quoted below are theirs).

**Fix round (Claude, 2026-10-22), after the final verification (a refuter and a completeness critic, both NEEDS-CHANGES;
three BLOCKING findings).** Every change is listed per finding in `docs/drafts/deixis/FIX_NOTES-s1.md`. In one paragraph: (1) the
step-0 GO rule was unreachable because S2b compared each name with a class-maximum null set by the most-exposed name;
the bars are now within-token contrasts plus an absolute route bar S0 taken from the engine's own transfer curve, and
the replica (`docs/drafts/deixis/s1-scratch/fix/`, run) shows the uniform-decay covariance rule cannot reach that bar for rare forms at
any operating point while a presynaptically gated leak (COV-PRE) does (15/15 at its centre, 59/60 on 60 seeds; every
rival 0/60), so COV-PRE is added as the sixth arm, every arm goes through one tuning procedure and one audit, the GO
count is 13/15 (justified binomially) and a failed step-0a record STOPs the lane before 0b (sections 2.6, 3); (2)
plastic weights now live in the engine's units, [0, 20] on 12-neuron token assemblies, with the arithmetic from `sim/`
and a measured transfer curve (2.6); (3) pointing words and anaphors (third person, demonstratives, former/latter, the
reduced and chat spellings) are brought into stage 1 through a learned third relation, GIVEN, whose afferent production
never renders, so every such turn is HELD by referential failure through a learned route; resolving them stays stage 2,
and the required-changes table now says which items are answered and which moved (sections 1, 2, 4, 8, 16).

**Round 3 (Claude, 2026-10-22, second fix round; `docs/drafts/deixis/FIX_NOTES-s1.md` section "Round 3"; scratch `docs/drafts/deixis/s1-scratch/fix2/`,
run on the workstation).** (1) The step-0b audit is now defined on ONE recorded exposure stream: the toy world writes it
to disk, the replica and the spiking run both read that file, and the audit is UNDEFINED unless their stream sha256 values
match; it keeps a per-seed bar-pattern check and a per-route same-side-of-bar check with a one-unit tolerance, and drops
Spearman to a reported number (shared stream 0.96, independent streams 0.45). A scratch run under a stated spiking-noise
model shows the correct learner passes it and every mislabelled or rival set of weights fails it (3.4). (2) The two
thresholds are measured per target, TOK -> P and TOK -> E (with its FS pool), on calibration seeds 61-70 with plasticity
off, before any learning arm; a scratch prototype gives P 8/4 and E 9/4 at the frozen FS constants and E 16/4 with a strong
FS pool of real FS cells, which would stop the lane. The organ runs preset-centred heterogeneity (2.2, 2.6). (3) The
English fixture's rendering is declared in full: SVO, a person-invariant tense frame with every auxiliary fused to its
subject, no copula, no agreement, 'the' before every undetermined object noun, and a closed contraction list. The toy
gains the matching markers (class W) and two co-varying stress markers (a hold-cost class), and step 3 checks every
non-pointer English form against class W before census v2 (4, 8.2). (4) Every rule arm records its engagement or is
UNDEFINED (3.4, 8). (5) W4 reads the turn-start latch, and H7 tests a self-fact rewrite through it (2.4, 5, 8.4). (6) The
answer-rate battery gains a frozen natural set it did not choose and the false-belief probes (8.5). The packet is
restructured so that only step 0a runs before a ruling.

host-decision: scaffold
decides: on a turn a routed writer would act on (a B3 store, the TUI fallback's echo, a false-belief scenario step, a W4 surprise read and reconsolidation), a fail-closed hold rule reads the organ's spiking latches and keeps the turn out of the store when any token read is a pointer pool, referential failure, ambiguity, the self latch, a co-commit or a non-invariant read, and holds every such turn when the organ is unavailable, raises or sees a different text; which turns the organ reads (an acquisition candidate, a TUI echo, a false-belief scene parse, a non-None W4 extract_assertion) and the per-turn relay of that read to the writers, with their skips on a held turn (W4 skips on the turn-start latch result whatever the later B3 check decides); which world afferents production withholds (voice, addressee, givenness: every production pointer hold is caused by the silenced afferent); the toy's discourse-givenness afferent (the entity named in the previous utterance); the exposure/hearing transmission-gate schedule and the exposure-only plasticity schedule; the exposure generator's English rendering table: one SVO clause per utterance, the tense frame drawn independently of the scene (simple past 0.6, present perfect 0.2, present progressive 0.2, regular verbs only), every auxiliary fused to its subject as a contraction and never rendered as a free word, no copula, no person agreement, 'the' before every object noun without a possessive or demonstrative determiner and in 'the former' / 'the latter', no 'a' or 'an', the R1 reference-grammar pronoun paradigm by role, R2 the single-word italic forms of the pinned Complete table of the Wikipedia article English personal pronouns (revision 1343514077), R3 apostrophe deletion and you -> u, your / you're -> ur, yourself -> urself at 0.2 over the closed contraction list i'm you're he's she's it's we're they're i've you've we've they've 'em and each fixture person's name's; the per-form dose floor (a form's rendering rate is raised until every exposed form occurs at least 300 times); the form-to-assembly front end; the per-token latch reset and the between-turn quiescent reset
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
from made-up words in the toy and from a fixed English paradigm in production (a declared teacher scaffold chosen after
the census was seen, 4.3), and the made-up-marker, K = 0 and transplant arms show the decision follows exposure, not the
string.

**Stage 1 does not claim:** discovery of the relation inventory (speaker, addressee, given, two decoys: innate conjunctive
structure; the word-to-relation and word-to-entity maps are learned); role assignment (B3's positional extractor still
decides agent and patient for every token the organ passes; declared, retired by board #260); RESOLVING a pointing word or
anaphor to its referent (stage 2 of #271: the mention trace on the organ's bridge replaces the toy's givenness afferent; the
two-candidate order mechanism; section 10); speech (the brain says no "I" or "you" in chat; organ-level only); identity
across sessions (no voice afferent in production S1-hold at all; S1-bind adds it, section 10); reported-speech storage of a
held claim (stage 2, the M3b source population); generalization to a spelling that no exposure rule produces (`moi`, three
census items, stays stored as a name: a counted residue with a method, a stage and a date in section 11; the held-out
spelling arm EN-HO, 8.2, is a front-end pin, not a measure of this limit).

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
| E[e] + FS pool | persons (voices, B1, SELF; toy objects) x 12; ONE shared FS pool of 16 `IZH2007_FS_CORTICAL_INTERNEURON` cells, E -> FS 8 and FS -> E 30 (GABA_A), dense (the agency motif's feedback-inhibition weights, `_agency_authorship_tag_derisk.py:88-92`: gain control under scene drive; competition between entities is the REF stage's job) | PART (exposure), TOK name route, C_k[e] release | C_k[e] (`dx_teach`), REF commit stage | new |
| H_k[e] relation lines | 5 relations x entities x 4 | VOICE, ADDR, GIV, LEFT, NEWEST | C_k[e] (AMPA, subthreshold alone) | new |
| C_k[e] conjunctive cells | 5 relations x entities x 4 | H_k[e] AND (E[e] via `dx_teach` OR P_k via `dx_release`); E and P alone, E+P without H, all subthreshold (E->C and P->C on the engine's `nmda_slow` excitatory receptor, `sim/regions.py:528`, or a weight scheme; P0 measures the AND) | P_k (`dx_conv`), E[e] (fixed release) | social-place-cell / dmPFC self-other conjunctive coding (Omer 2018, Danjo 2018, Jamali 2021) |
| P_k pools: SPK, ADR, GIV, LEFT_dec, NEWEST_dec | 5 x 12 | C_k[*] (`dx_conv`), TOK (`dx_ptr`) | C_k[*] (`dx_release`, subthreshold alone), RF_k | new |
| REF commit stage over E | spiking_commit `commit_plan` (K = entities) | E | latches S_e | `research/runners/spiking_commit.py` |
| AMB + CAND | 4 rate-bounded CAND per E, AMB 12 | two or more co-driven REF | vetoes every gate; own latch | DXB; Nref ambiguity (Nieuwland & Van Berkum 2006) |
| RF_k referential failure | 5 x 8 | P_k through slow NMDA; inhibited by any gate burst | own latch | DXB |
| SELF latch | the S_SELF latch of the REF stage | | the hold read | |
| step 1b only: agency motif, OUT[w], OUT commit | reused sizes; OUT per form x 8; OUT commit K = forms | PRODUCE -> cd -> src_self; sensory_marker -> src_other; cd -> C_SPK[SELF]; cd -> INH_other -> C_SPK[e != SELF] | | `_agency_authorship_tag_derisk.py` constants |

**Cell types and heterogeneity (the 10-20 substrate defect).** With `enable_parameter_heterogeneity=True` and
`preset_centred_parameter_heterogeneity=False` (the `sim/config.py:244` default), the global draw overwrites every
region's Izhikevich a/b/d/C with RS-centred values, so a region declared FS runs as an RS hybrid. The organ therefore sets
`enable_parameter_heterogeneity=True` AND `preset_centred_parameter_heterogeneity=True` in its own config (no default
changes): each cell is jittered around its own preset. Measured in the scratch prototype (seed 61, `docs/drafts/deixis/s1-scratch/fix2/
calib_proto*.out`): FS median a 0.206, C 19.0 pF with the flag on; a 0.031, C 94.9 pF with it off (hybrids). Both flag
values and the per-population median a/b/d/C are written into the bar-sanity record and into the organ-cache key (5.1).
The setting matters: with a strong FS pool (E -> FS 60, FS -> E 80), TOK -> E needs w = 16 to fire every E neuron with
real FS cells but only 9 with hybrids (2.6).

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
  gate (`set_transmission_gate`, `sim/bridge.py:5493-5517`: it gates the current, the plasticity gate the updates), which step 0a(i) and every 0b run check by reading back
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
The same read also gates the three other writers that act on a heard statement (5.1): the TUI fallback's register echo,
the false-belief scenario writer and W4 (surprise, then in-place reconsolidation).

**Two fields, two consumers (round-3 W4 finding).** `read_turn` returns `latched` (the turn-start latch result alone: some
token read is a pointer pool, RF_k, S_SELF, AMB, a co-commit, a non-invariant read or a second latch; or the organ is
unavailable or saw another text) and `held` (`latched` AND the flag-off path would store the turn). W1/W2 act on `held`.
W4, the TUI echo and the false-belief writer act on `latched`, because each runs whatever B3 later decides: W4 runs at
server.py:6950-7001, BEFORE the B3 reject at :7041, and its own extractor is not B3's ('the brain has no' gives W4
('brain', 'has', 'no') and B3 None). On a latched turn W4 is skipped whole: no surprise read, no reconsolidation, so a
user's 'you are a brain that has spikes' cannot rewrite the curriculum's (brain, has, neurons) in place. The one reply
text this removes is the surprise / reconsolidation notice on such a turn (counted, H7 in 8.4).

This is subtractive: S1-hold can turn a store, an echo, a scenario step or a W4 read-and-rewrite into nothing, and it can
replace the reply of a held turn by the abstention; nothing else. "tell me more", "how do you feel" and every question are
untouched as turns, but their replies after a held statement can change because nothing was stored (named in the
FLIP-HURTS list, 8.5).

**Fail-closed degrade (completeness major 6).** With the flag on, an unavailable organ is never read as "nothing to hold".
If the trained organ cannot be loaded at startup, the server refuses the flag (startup error naming the cache key) rather
than serving the flag-off path under the flag's name. If `read_turn` raises, or the raw text it read differs from the text
the writer was given, the turn is HELD with class `unavailable` or `mismatch`. Both are counted per chat
(`deixis_engaged`, `deixis_unavailable`, `deixis_mismatch`), carried in the reply JSON, and pinned by tests on the server,
the TUI and the rich default path.

**Co-commit (sibling-lane risk).** The cs-race lane measured a symmetric two-candidate spiking commit latching BOTH slots
on 1,158 of 1,440 decisive draws (cs-race-f1d-1021). Here a co-commit is a HOLD, so the same failure would show as
over-holding, not as a wrong store. The REF stage first exists at step 1a, so S8 there and the organ-gate reads report
the co-commit rate on single-referent tokens (step 0b has no commit stage to measure it), and T1-T5 and T14 count it as a
miss.

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

Measured transfer curve (`docs/drafts/deixis/s1-scratch/fix/wscale_curve.py`, the engine itself, numpy backend, heterogeneity on, 10
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
from a curve, never from any learning arm: on this bare curve `theta_fire = 8` (every target neuron fires on every seed;
0.40 W_MAX) and `theta_silent = 4` (at most one target spike on every seed; 0.20 W_MAX). A learned route must end at or
above theta_fire; a route that must not act must end at or below theta_silent (bar S0, 3.5). These are the DRAFT values:
the round-2 sentence "the P pools and the E assemblies are both 12 neurons, so one curve serves both routes" is
withdrawn (round 3), and step 0a(i) measures the pair per target on the organ's own circuit (below). The replica's
normalised weight maps as `w_engine = 20 w_replica`.
The fixed downstream weights stay in the range the engine's working stages already use (the commit stage's W_RG 40-60,
`word_role_parser_organ.py:174-183`; the agency motif's 60/30/8, `_agency_authorship_tag_derisk.py:88-92`): they are not plastic, so the [0, 20] clip does not
touch them (`sim/bridge.py:10390-10411` clips only gain > 0 plastic synapses when the plastic mask is enforced).

**Measured per target, on the organ's own circuit, before any learning (round-3 threshold finding).** The curve above
is a bare 12-neuron pool. The E assemblies are not bare (one shared FS pool, the release input, presence drive in
exposure), so the round-2 sentence "one curve serves both routes" is withdrawn. Step 0a(i) measures two curves on the
step-0 hearing core (3.4; the frozen constants of PACKET Appendix C), on calibration seeds 61-70 (outside the census seeds
1-15, the replica rate seeds 1-60 and the gate seeds), with plasticity off (learning rate 0, decay 0, the runner update not
applied; the plastic blocks' sha256 equal before and after):
- TOK -> P: P's hearing-mode wiring is TOK -> P plus C -> P (closed at hearing), so the bare curve is P's real curve; the
  CAUSAL+WTA arm's circuit, whose P pools inhibit each other, is measured as its own variant and scores that arm.
- TOK -> E: the target E assembly with the FS pool, the other present entities' assemblies, and the release projection
  C -> E open, in two hearing conditions: production (no H afferent, C silent) and toy test (the target's H lines on, C
  subthreshold). The exposure condition (presence drive on every present entity) is reported as the increment over
  w = 0 only: in exposure the learned routes carry no current (2.3, read back by P0).
- The rule, written before the measurement: theta_fire(target) is the smallest grid weight at which every target neuron
  fires at least once in a 40-step token on every calibration seed in every hearing condition; theta_silent(target) the
  largest grid weight below it at which the target emits at most one spike on every seed in every hearing condition.
  The values and the calibration JSON's sha256 are committed before any learning arm runs. If either target's pair
  differs from 8 / 4, step 0a(ii) runs the replica with the measured per-target pairs (it always reads them from the
  committed calibration file) and applies the 0a STOP rule (3.4) to that run. Nothing is re-set after a learned weight
  has been seen.

Scratch forecast (`docs/drafts/deixis/s1-scratch/fix2/calib_proto.py`, the engine itself on the workstation, numpy, seeds 61-70; NOT the
step-0a record: the prototype's circuit is a reduced copy of Appendix C, five entities, four C cells):

| FS constants (E -> FS / FS -> E) | heterogeneity | TOK -> P fire / silent | TOK -> E fire / silent |
|---|---|---|---|
| 8 / 30 (frozen, the agency motif) | preset-centred on | 8 / 4 | 9 / 4 |
| 8 / 30 | off (hybrids) | 8 / 4 | 9 / 4 |
| 20 / 20 | on | 8 / 4 | 9 / 4 |
| 60 / 80 | on | 8 / 4 | 16 / 4 |
| 60 / 80 | off (hybrids) | 8 / 4 | 9 / 4 |

At 8 and 9 every seed's E fires all 12 neurons at w = 9 and at w = 8 one neuron on one seed stays silent, also with the FS
pool silent: the shift to 9 is the E assemblies' own heterogeneity draw, not inhibition. Re-scored with P 8/4 and E 9/4,
the step-0a replica keeps COV-PRE at 15/15 (59/60 on seeds 1-60); with E 16/4 it passes 0/15, because its name routes
end at 11.1-15.0 engine units (median 13.7; its pointer pool routes 10.5 at the lowest, median 13.3) on seeds 1-15
(`replica4_markers.out`). So the FS constants are frozen now, from the motif precedent, and
named as the setting that decides whether the lane stops: the prototype saw all three settings before the choice, and
the choice rests on the pool's role (gain control in exposure, not competition), never on the forecast.

**The saturated-route check** runs in step 0a(i) on calibration seed 61 and again after the census: TOK -> P and TOK -> E
set by hand to W_MAX must fire P_k and E (every neuron), and with H on fire C_k[e] and, at step 1a, latch REF; at
theta_silent and at 0 they stay silent (at most one spike).

## 3. The learning rule: a fair census first, a sim/ addition only if needed

### 3.1 Why a new rule is in question

The in-step rules give a weight proportional to P(post | token): the rate-window rule potentiates every co-active synapse
toward `hebbian_max_weight` (`sim/config.py:691-700`), the causal rule likewise (`sim/bridge.py:10327-10347`), and BCM
(`:10248-10273`, `dw = g x y (y - theta)`) is exactly zero when the post trace is zero, so a token present while a pool is
silent is never depressed. A pointing word needs P(post | token) minus P(post): the base-rate subtraction. The kernel-level
pre-active/post-silent term exists (`sim/kernels.py:1561-1583`, HTM, fixed breakeven 0.125, unwired in the step); the brief's
toy shows names also acquire the role cell under it.

### 3.2 The candidates: covariance with a uniform leak (COV) or a presynaptically gated leak (COV-PRE)

```
COV      dw_ij = lam * x_j * (y_i - ybar_i)                                for x_j > pre_floor, named plastic gates only
         w_ij <- w_ij * (1 - hebbian_weight_decay * gain_ij)  (the EXISTING gated uniform decay, bridge.py:10349-10380)
COV-PRE  dw_ij = lam * x_j * (w_max * (y_i - ybar_i) - kap * w_ij)          same selection; hebbian_weight_decay = 0
both     w_ij <- clip(w_ij, hebbian_min_weight, hebbian_max_weight)         (bridge.py:10390-10411; [0, 20], 2.6)
x, y     the existing rate-window co-activity traces (cp_hebb_coactivity_trace, bridge.py:10226-10235)
ybar     a per-neuron slow running mean of y, updated only while a named plasticity gate is open, so hearing-mode
         activity and plasticity-closed gaps never move the base rate
```

Fixed points per synapse, in expectation (normalised to w_max; the engine weight is 20x):

- **COV:** `w* = (lam / delta) * E[x_j (y_i - ybar_i)] = (lam / delta) * P(x_j) * (E[y_i | x_j] - ybar_i)`, delta the
  decay per utterance. The weight is graded in the covariance, and the covariance carries the form's frequency P(x_j).
  Below the clip every weight scales together with lam/delta, so the ratio between the weakest route that must fire (the
  rarest pointer form or name) and the strongest route that must stay silent (a frequent form's chance covariance with a
  pool or a wrong entity) is fixed by the stimulus statistics. On the replica that ratio is at most 1 at every grid cell
  (median 0.09 / 0.12 at lam*tau = 6, 0.28 / 0.37 at 18; the table below), and the engine needs 2 (theta_fire /
  theta_silent = 8 / 4, 2.6). No lam/delta and no w_max can fix a ratio: the uniform leak is a defect of the learner, not
  of the bar.
- **Why the judged round-2 GO rule was unreachable (BLOCKING 1).** Its S2b compared every name with the label-shuffled
  null's class maximum, a null set by the most-exposed name. V3's two names and V2's idiolect name share their bearer's
  dose, so under COV their weight is about half, and they failed on 9 of 15 seeds at every lam*tau
  (`docs/drafts/deixis/s1-scratch/fix/replica2_baseline.out`, the judged replica copied unchanged: S2b 6/15 at 4, 6 and 9). Scored
  against each name's own dose-matched null, the same COV weights pass S2b on 15/15 (`final_v3.out`, columns S2b and
  S2b_classmax): the class-maximum null was a defect of the bar. Fixing only that bar would not have rescued COV: under
  the engine's physical S0 bar it passes 0/15 at every cell. Both defects are fixed: the bars are now within-token
  contrasts plus the absolute S0 route bar (3.5), and the learner gets a sixth arm.
- **COV-PRE:** `w* = w_max * (E[y_i | x_j] - ybar_i) / kap`. The leak acts only while the form is heard, so frequency
  sets how fast a route converges (time constant 1 / (lam kap) occurrences of the form), not where it ends: a form heard
  83 times and one heard 2,000 times end at the same weight if they predict their pool equally well. Biology: the outstar
  learning law, in which a weight tracks the postsynaptic signal only while its presynaptic cell is active (Grossberg
  1974); homosynaptic LTD that requires presynaptic activity (Dudek & Bear 1992); weight-dependent depression, LTD
  proportional to the weight while LTP is not (van Rossum, Bi & Turrigiano 2000). The covariance term is unchanged
  (Stanton & Sejnowski 1989; Sejnowski 1977).

**Replica numbers (`docs/drafts/deixis/s1-scratch/fix/replica3.py`, rate level, not spiking; the world of 4.2 with the GIVEN relation and
its five forms, 4800 utterances, tau = 400 utterances, the bars of 3.5; dev seeds 1-15 for the grid, 1-60 for rates;
outputs `final_v3.out`, `centres.out`, `final_v3b.out`, `s6_check_lt6.out` in the same folder).** Each cell: seeds passing
every bar / 15, then the median weakest must-fire route and the median strongest class-W route (normalised; S0 needs
>= 0.40 and <= 0.20).

| lam*tau | COV | COV-PRE | RW-HEBB (also CAUSAL) | +WTA | BCM |
|---|---|---|---|---|---|
| 1 | 0; 0.02, 0.02 | 0; 0.16, 0.06 | 0; 0.02, 0.31 | 0; 0.00, 0.31 | 0; 0.02, 0.12 |
| 2 | 0; 0.03, 0.04 | 0; 0.29, 0.09 | 0; 0.04, 0.62 | 0; 0.00, 0.62 | 0; 0.03, 0.24 |
| 3 | 0; 0.05, 0.06 | 5; 0.39, 0.10 | 0; 0.06, 0.93 | 0; 0.00, 0.91 | 0; 0.05, 0.36 |
| 4 | 0; 0.06, 0.08 | 15; 0.46, 0.11 | 0; 0.08, 1.00 | 0; 0.00, 0.99 | 0; 0.06, 0.48 |
| 6 | 0; 0.09, 0.12 | 15; 0.57, 0.13 | 0; 0.13, 1.00 | 0; 0.00, 1.00 | 0; 0.09, 0.72 |
| 9 | 0; 0.14, 0.18 | 14; 0.57, 0.15 | 0; 0.19, 1.00 | 0; 0.00, 1.00 | 0; 0.14, 0.99 |
| 12 | 0; 0.19, 0.24 | 12; 0.57, 0.16 | 0; 0.25, 1.00 | 0; 0.00, 1.00 | 0; 0.19, 1.00 |
| 18 | 0; 0.28, 0.37 | 9; 0.57, 0.19 | 0; 0.38, 1.00 | 0; 0.00, 1.00 | 0; 0.28, 1.00 |

- Why each arm fails, by mechanism (seeds 1-2 at each centre, read route by route). RW-HEBB and +WTA fail S1, S2a,
  S2b, S4 and S0 at every cell: the filler and the names potentiate onto the pools, decoys included, as strongly as the
  pointers (filler -> LEFT 0.88-0.92), because nothing subtracts the base rate; the winner-take-all also starves the rare
  forms and PLURAL's second target (weakest must-fire routes 0.00-0.01). BCM fails S0 at every cell and S4 on 13-15 of
  15 seeds: with no depression at y = 0, the filler keeps every decoy weight it gains by chance (0.11-0.13 at its
  centre, where the rare GIVEN forms reach only 0.01-0.02). COV fails only S0, by the frequency ratio above: its weakest
  must-fire routes are the rarest forms (POSS-G, ORD1, V3's two names, 0.05-0.14 at its centre) and its strongest class-W
  routes are frequent forms' chance covariances (SHIFT-S or the filler onto a decoy, 0.09-0.12). COV-PRE fails at
  lam*tau <= 3 because the rarest forms have not converged in 4800 utterances, and loses seeds from 9 up because a fast
  route tracks recent samples (its noise grows with lam); its window is 4-9 (>= 14/15), a factor of 2.25.
- Per-seed pass rate on dev seeds 1-60, each arm at its own centre (3.4): COV-PRE (lam*tau = 6) q = 0.983 (59/60; the
  one failure is S5, the forced-fit PREV marker); COV (6), RW-HEBB (3), +WTA (3) and BCM (1) all q = 0.000.
- Nulls on COV-PRE at 6, 15 seeds, every bar: label-shuffled stream 0/15, scene shuffled 0/15, voice permuted 0/15,
  K = 0 0/15 (each fails S0; K = 0 passes every contrast bar and fails only S0, which is why S0 exists).
- Convergence (S6) on COV-PRE at 6: every must-fire route non-decreasing between 2/3 of exposure and its end on 15/15
  seeds; the 74 route-seed pairs of forms heard >= 450 times within 20% on 15/15; 97.6% of all pointer routes within 20%.
- Dose curves on COV-PRE at 6, medians over 15 seeds, in the direction the literature reports. The "you is the brain's
  name" route SHIFT-A -> E_SELF falls with the overheard fraction: 0.57 at 0 and 0.43 at 0.25 (above theta_silent: the run
  fails, the addressed-only reversal of a child who hears "you" only addressed to them, Oshima-Takane 1988; predicted, not
  barred), 0.26 at 0.5 (passes 14/15), 0.18 at 0.6 (the schedule, 15/15), 0.08 at 0.75 (12/15; the failing bar at 0.75
  is reported by the step-1 dose arm, not yet isolated). The "I" route's best entity weight falls with the number of
  speaking voices: 0.44 at 1, 0.18 at 2, 0.13 at 3, 0.09 at 5 (Oshima-Takane 1999; the tie at one voice, Gold &
  Scassellati). The forced-fit ratio PREV -> ADR / SHIFT-A -> ADR rises with turn-taking: 0.11, 0.17, 0.30, 0.41 at
  p = 0, 0.3, 0.6, 0.9.

**Round 3, the toy with its morphology markers (`docs/drafts/deixis/s1-scratch/fix2/replica4_markers.py`; 4.1; the original tokens'
stream is unchanged, checked).** COV-PRE at its centre is unchanged, 15/15 and 59/60. Its EN-matching markers stay
silent: ART, ASP and IND reach at most 0.05-0.12 of W_MAX on any pool (1-2.4 engine units) and 0.05-0.09 median on any
entity, so class W holds on 15/15 and 60/60. The stress markers do what the reviewer predicted: AGR reaches 0.30-0.37 of
W_MAX on SPK or ADR (6-7.4 units, between the two thresholds) and DEF 0.66 on GIV (13 units, it fires). That is the
reason EN v1 renders neither (4.3) and the reason they are a hold-cost class here, not a GO bar. COV's AGR reaches 0.76.

**Preregistered predictions for step 0b (the spiking census):** CAUSAL, CAUSAL+WTA and RW-HEBB fail S1 and S0; BCM fails
S0 and S4; COV fails S0 at both of its cells; COV-PRE passes at its centre. Each is a falsifier of the replica. A pass by a
rival is not accepted on its face and not rejected either: it goes through the same audit as every pass (3.4) and, if the
audit holds, that rival is the rule.

### 3.3 The sim/ addition (built only on a COV or COV-PRE GO, step 1)

- `sim/config.py`, beside `hebbian_bcm` (:721-723): `hebbian_cov: float = 0.0` (lam; 0.0 means the branch is unreached,
  so byte-identical); `hebbian_cov_leak: float = 0.0` (kap; 0.0 is COV with `hebbian_weight_decay` as its leak, > 0 is
  COV-PRE); `hebbian_cov_ybar_alpha` (per open-gate step, set by the runner from step 0a(i)'s measured steps per utterance
  so the time constant is 200 utterances, the replica's 0.005 per utterance); `hebbian_cov_pre_floor: float = 0.02`;
  `hebbian_cov_ybar_gate: bool = True` (ybar moves only on steps on which a named plasticity gate is open). Requires
  `hebbian_rate_window=True`; `hebbian_cov > 0` with `hebbian_bcm > 0` raises at config validation, and so does
  `hebbian_cov_leak > 0` with `hebbian_weight_decay > 0` (one leak per arm). Registered with `tools.lab.bound_check`.
- `sim/bridge.py`: `cp_cov_ybar` allocated and updated beside `cp_bcm_theta` (:10229-10235); a guarded branch beside the
  BCM branch (:10248-10273) with the same `_pre_tr_b > pre_floor` selection, `cp_plasticity_rate_gain` and `_pm_heb`
  gating, then `active_synapse_indices_heb = empty` so the potentiation-only body is inert (as BCM does at :10273-10274).
  The existing decay and clip follow unchanged. About 30 lines.
- **Where the step's gating differs from the replica, measured, not assumed.** The engine runs the Hebbian block, and with
  it the decay and the clip, only on steps on which some neuron fired on both the previous and the current step
  (`if _prev_any and _fired_any`, `sim/bridge.py:10236`). Step 0a(i) measures, per arm, on calibration seeds 61-70 with
  plasticity off (learning rate 0, decay 0, the runner update not applied), the fraction of exposure
  steps on which the block runs, the steps per occurrence on which a form's trace exceeds pre_floor, and the trace value
  of a driven pool neuron (y_on). Every arm's rate parameter is then set so that its change per OCCURRENCE equals the
  replica's lam = lam*tau / 400, the decay arms' per-step decay so it equals 1/400 per utterance, and kap so a perfectly
  predictive form ends where the replica puts it (y_on is the unit conversion). The lam*tau grid then means the same thing
  on every arm. These are unit conversions, committed with the calibration's sha256 before any learning arm runs; no
  learned weight exists when they are set.
- Pins (`tests/test_hebbian_cov.py`): lam = 0 byte-identical (weights + spikes sha) on a fixed 3-region 500-step scenario
  with the rate window on and off and BCM on; a three-neuron sign test at kap = 0 and kap > 0 (pre on, post below its
  mean: down; pre on, post above: up; pre off: exactly 0, the leak included); ybar frozen while every named gate is
  closed; gain-0 and `plastic=False` synapses move by exactly 0; two runs identical; parity with the step-0 runner-applied
  arm on a REPLAYED raster (the same recorded spike trains drive both, so float drift in the spiking cannot hide or fake a
  rule difference; max |dw| <= 1e-6). Existing pins stay green: `tests/test_plasticity_inertness.py`,
  `test_global_plasticity_gain.py`, `test_enforce_plastic_mask.py`, `_plastic_mask_hebbian_variants_scenario.py`,
  `test_determinism.py`, `test_bg_commit.py::A1_COMMIT_BATTERY_SHA256`; plus the config UI roundtrip entry
  (ENGINE_REFERENCE).
- Biology card `research/biology/covariance-ltd.md`: Stanton & Sejnowski 1989 (associative homosynaptic LTD, explicitly a
  covariance rule); Dudek & Bear 1992 (input activity that fails to drive the target depresses; the depression needs the
  input active); Bienenstock, Cooper & Munro 1982 and Kirkwood, Rioult & Bear 1996 (the base-rate-relative threshold);
  Sejnowski 1977 (Hebbian plus decay as a covariance estimator); Grossberg 1974 (the outstar, the presynaptically gated
  leak); van Rossum, Bi & Turrigiano 2000 (weight-dependent depression). Stated deviations: Artola, Broecher & Singer 1990
  find no LTD in a strictly silent postsynaptic cell while this rule depresses whenever the target sits below its running
  mean; the rule acts on rate traces, not spike triplets; the presynaptic gate is a threshold on a rate trace, not a
  spike-timing window.
- The sim/ edit stays on `research/deixis-s1` until the Stage-2 GATE_SHA is pinned (11-03) and merges to main only with
  every flag-off pin green.

### 3.4 Step 0: the fair census (no sim/ edit, no production edit)

**Step 0a runs alone; every later step starts only on a Claude RULING (round 3).** Step 0a has three parts, all
provenance-stamped module runs (`python -m research.runners.<module> --out ...`, so `research/runners/__init__.py`
writes each `.prov.json`), all on the CPU pool (the census host class), none touching `sim/` or production; then STOP and
`[deixis-271-s1-0a] RESULT`. The modules and the exact commands are in the packet.

- **0a(i) calibration, on the organ's own circuit, plasticity off, calibration seeds 61-70.** The step-0 hearing core
  (TOK / E + FS / H / C / P with the mode gates, no commit stage; frozen constants in PACKET Appendix C) is built and
  measured: the per-target transfer curves and thresholds (2.6); the saturated-route check; the C-cell AND (the first
  row of the frozen C-weight grid that passes every P0 item at K = 0 on 10/10 calibration seeds); the unit conversions
  (the fraction of exposure steps on which the engine's Hebbian block runs, `if _prev_any and _fired_any`,
  `sim/bridge.py:10236`; the steps per occurrence on which a form's trace is above the presynaptic floor; y_on, the trace
  of a driven pool neuron; the presence current that gives 0.2 of the participant rate; the empirical P_k base rates);
  per-arm step time, each of the six arms' update code executing at learning rate 0 for 2,000 exposure steps (the
  runner-applied covariance update on the about 67k plastic synapses included, and the per-step gate read-back); the
  per-population median a/b/d/C under the heterogeneity setting of 2.2; and two runs of seed 61 byte-identical. A P0
  failure (no grid row passes) is a STOP with its own RESULT: no further tuning, never on census seeds. Committed with
  the JSON's sha256 before 0a(ii) reads it.
- **0a(ii) the replica and the audit check, on recorded streams.** The toy world (4.1, 4.2, with the morphology markers)
  writes every stream the replica needs to disk; the replica reads them and the calibration file; the audit check runs
  the audit of this section on those streams under the spiking-noise model below.
- **0a(iii) the bar-sanity record** (BAR class in `docs/FAILURE_GATE_MATRIX.md`): the thresholds and their source, the
  heterogeneity setting, the replica record against the STOP rule, the audit check, every input's sha256 and the
  `tools/where_ran.py` host summary.

**The replica part of 0a(ii).** The rate replica, with the exact bars of 3.5, the exact schedule of 4.2 and the exact
nulls, runs every arm through one tuning procedure, written before it runs:

- one grid for every arm: lam*tau in {1, 2, 3, 4, 6, 9, 12, 18}, dev seeds 1-15;
- each arm's centre: the cell with the most seeds passing every bar; ties go to the most bar-seed passes, then to the
  larger median weakest must-fire route (the cell farthest above theta_fire);
- each arm's spiking cells at 0b: its centre and the next grid value up (a 2x1 grid, the same budget for every arm).

The fix-round draft of that record (`docs/drafts/deixis/s1-scratch/fix/`) gives the centres COV-PRE 6, COV 6, RW-HEBB 3, +WTA 3, BCM 1
(CAUSAL and CAUSAL+WTA share the potentiation arms of the rate replica, which has no spike timing). The thresholds are
not taken from any arm: S0 comes from the engine's transfer curve (2.6) and the contrasts (2x, 0.25x, 0.2x) are round 1's.
The record only checks four properties of the bars: some learner can reach the GO rule (an arm with a 60-seed rate
q >= 0.9, the rate at which P(>= 13/15) >= 0.8); every null fails; K = 0 fails; and the audit is reachable and
discriminating (below). **STOP rule:** if any of the four fails, the bars are wrong, step 0b is not queued, and the
failing bar is re-derived from mechanism (never from an arm's margins) and re-ruled. The replica always scores with the
per-target thresholds of the committed calibration file, so a calibration that moves a threshold re-runs this record
by construction.

**The audit, defined on one recorded stream (round-3 BLOCKING finding).** The judged audit compared the spiking weights
with the replica's by Spearman >= 0.9, but nothing tied the two to one exposure: on two independent streams the same
correct learner scores a median of 0.46 (the reviewer's 0.464 reproduced: 0.448 here, `docs/drafts/deixis/s1-scratch/fix2/
stream_audit.out`), because most scored routes are class W near 0 and their ranks are sampling noise. The fix:
- **One stream.** `_deixis_toy_world.py` writes each seed's exposure stream (and each null's) to disk as canonical JSON
  lines (a header with the schema, seed, length, arm, world parameters, vocabulary and the per-seed nonce map, then one
  line per utterance: token ids, the world's drive per post population, speaker and addressee), and records its
  sha256. The spiking run drives the bridge FROM that file; the replica trains on THAT file. Each records the sha256 it
  read.
- **A0 stream identity:** the two sha256 values are equal on every seed, else the audit is UNDEFINED.
- **A1 bar pattern:** on each seed, the pass/fail of every GO bar (S0-S5, the class-W markers included) is the same in
  the spiking weights and in the replica's weights on that seed's stream; >= 13/15 seeds.
- **A2 route side:** on each seed, every scored S0 route (must-fire against theta_fire, class W against theta_silent, of
  its own target) is on the same side of its bar in both, a disagreement being ignored only when BOTH values lie within
  one engine unit (TOL = 0.05 of W_MAX, the transfer curve's grid step) of the bar; >= 13/15 seeds.
- Spearman of the scored routes is REPORTED, not a bar.
- **The spiking-noise model the 0a check uses (stated):** each post unit is 12 neurons whose trace is a Poisson spike
  count, Poisson(LAM x (g y + SPONT)) / LAM, LAM the count of a fully driven neuron in one 40-ms token (about 8 at the
  measured 0.19 spikes/ms), g a per-neuron gain drawn once per seed from lognormal(0, GAIN_SD), SPONT a spontaneous rate;
  each occurrence is missed by the presynaptic floor with probability MISS and its pre trace jitters by N(1, PRE_SD); the
  calibration's unit conversion is off by a factor drawn once per run, lam x U(1/CONV, CONV) and kap x U(1/KCONV, KCONV); a
  route's weight is the mean over its 12 x 12 synapses. Moderate: LAM 8, GAIN_SD 0.25, SPONT 0.01, PRE_SD 0.05, MISS 0.02,
  CONV 1.25, KCONV 1.10. Harsh: LAM 3, GAIN_SD 0.40, SPONT 0.03, PRE_SD 0.10, MISS 0.05, CONV 1.50, KCONV 1.25.
- **Scratch result (`docs/drafts/deixis/s1-scratch/fix2/stream_audit.py`, seeds 1-15, each arm at its 0a centre, the weights of rule g
  under the noise model audited against the replica of rule r on the same stream files).** The correct learner passes:
  COV-PRE against COV-PRE's replica, moderate A1 15/15 and A2 15/15 (Spearman median 0.96), harsh A1 13/15 and A2 14/15
  (0.93; its GO bars pass 13/15 there). Every mislabelled set fails: COV-PRE's bar-passing weights audited as COV,
  RW-HEBB, CAUSAL+WTA or BCM give A1 0-1/15 and A2 0/15 at both noise levels, so an arm whose configuration silently did
  not apply cannot win under another arm's name. Every rival's own weights fail the GO bars (0/15 at both levels), and even
  against its own replica no rival reaches 13/15 on both A1 and A2 at either level (COV 14/11 and 9/4, RW-HEBB 15/8 and
  14/2, CAUSAL+WTA 14/4 and 13/0, BCM 0/15 and 0/11: their routes and per-seed bar outcomes sit near the bars, where noise
  flips them), so none can carry an audit either. Independent streams: A0 UNDEFINED (sha mismatch), Spearman 0.45, A2 still 15/15 (the
  route-side check is robust; the rank statistic was the defect). An arm whose rule never ran (weights 0) fails S0 and is
  UNDEFINED by the engagement record, never a FAIL.

**Step 0b, the spiking census (gated on a Claude RULING on the 0a RESULT; dev seeds 1-15, 4800 utterances, the 2x1 grid
per arm):**

- CAUSAL: the default causal rule (`hebbian_rate_window=False`), lr per occurrence, engine decay.
- CAUSAL+WTA: the same with lateral inhibition among the P pools (the word-role organ's competition).
- RW-HEBB: the rate-window potentiation rule, engine decay.
- BCM: `hebbian_bcm` gain with `hebbian_bcm_theta_alpha`, rate window on, engine decay.
- COV: `hebbian_learning_rate=0`, rate window on, the covariance update applied by the runner each step on the plastic
  block of `cp_connections.data` from the bridge's own traces (the Lane C precedent,
  `_laneC_source_monitor_hetero_encoding.py:55,262-275`), exactly the arithmetic 3.3 puts in the step, engine decay.
- COV-PRE: the same runner-applied update with the presynaptically gated leak, engine decay 0.
- Nulls, only on an arm that passes the weight bars: label-shuffled stream, scene shuffled, voice permuted per utterance
  (3 x 15 runs); K = 0 is scored on the initial weights. No label-shuffled THRESHOLD runs: no bar needs one.
- The dose arms (voices {1, 2, 3, 5}; overheard {0, 0.25, 0.5, 0.6, 0.75}; turn-taking p {0, 0.3, 0.6, 0.9} and
  dyad-only) run at step 1 on the chosen rule, after the ruling.
- Every run reads its exposure from the stream file 0a wrote for that seed and arm (the same file the replica read), so
  the audit's A0 can hold. Thresholds, conversions and the C-weight row come from the committed 0a(i) calibration; the
  first per-arm run on dev seed 7 sets nothing, it checks the engagement record and the wall time.

**Engagement of every rule arm (round-3 finding (a)).** A rival whose rule never engaged would otherwise score as a
FAIL and hand the GO to a later arm. Every 0b run (and the 0a(i) step-time runs) records, and is UNDEFINED
(`tools.lab.undefined_if_empty`) without: (a) the plasticity and transmission gate values of every learned block read
back from the bridge during exposure (sampled once per utterance), with the learned routes' transmission at 0
throughout; (b) the number of exposure steps on which the update was applied (for the in-step arms the engine's block
condition counted from the firing states each step; for COV and COV-PRE the runner's own count); (c) sum |dw| per learned
block (`dx_ptr`, `dx_name`), accumulated per utterance, nonzero; (d) the rule configuration read back from the bridge's
`core_config` after build (`enable_hebbian_learning`, `hebbian_rate_window`, `hebbian_bcm`, `hebbian_bcm_theta_alpha`,
`hebbian_learning_rate`, `hebbian_weight_decay`, `hebbian_min_weight`, `hebbian_max_weight`,
`enforce_plastic_mask_in_hebbian`, `enable_stdp`), the runner's lam, kap, pre_floor and ybar_alpha for COV and COV-PRE,
and for CAUSAL+WTA the P-pool inhibitory weights read from `cp_connections` and the inhibitory pool's spike count per
utterance (nonzero); (e) the stream sha256. A failing arm counts as FAIL only when it fails its preregistered bars WITH
this record present; otherwise it is UNDEFINED.

**Audit, applied to every arm that passes, before the pass counts.** Instrument items (any one missing or failed:
UNDEFINED): A0; the engagement record; P0 on that arm's runs; the learned routes' transmission read back closed on every
exposure step (a route that carried current while it learned could have taught itself); the arm's own nulls each fail
(pass on <= 1/15); the saturated-route check on that arm's bridge. Replica items: A1 and A2 above, >= 13/15 each. A pass
whose instrument items hold but whose replica items fail is PASS-UNEXPLAINED: the replica is falsified for that rule,
the record shows the route-level disagreements, and the GO waits for a Claude ruling (a rival must never be rejected
only because a rate replica predicted it would fail, and never accepted without the disagreement being read).

**GO rule.** The rule is the first arm, in the order CAUSAL, CAUSAL+WTA, RW-HEBB, BCM, COV, COV-PRE (existing rules
first, then the smaller sim/ addition), that passes P0, the saturated-route check and S0-S7 all on the same seed on
>= 13/15 seeds at one of its two cells, whose nulls each pass on <= 1/15, and whose audit holds. An arm that is UNDEFINED
or PASS-UNEXPLAINED earlier in that order BLOCKS every later arm's GO until it is re-run with a complete record or
ruled. An existing rule: no sim/ edit, and the packet is re-ruled to it. COV or COV-PRE: step 1 adds it to sim/ (3.3)
and must reproduce the census, defined exactly: on the same recorded streams, the in-step sim/ rule and the 0b
runner-applied arm give the same per-seed pass/fail of every GO bar on 15/15 seeds and the same side of every scored
route under A2's tolerance on 15/15 seeds, and on a replayed raster their weights differ by at most 1e-6 (3.3). None:
STOP, bank the method, next method named (a three-factor rule with the coincidence pool as the third factor and
eligibility traces on TOK -> P, Fremaux & Gerstner 2016; then BTSP plateaus gated by the coincidence cell).

**Why 13/15: noise, sample size and effect size (BLOCKING 1).** A seed passes or fails, so the count on 15 seeds is
binomial in the arm's per-seed pass rate q. The bar must pass a correct learner and fail one that is right by chance. The
spiking arm adds noise the replica does not have (Poisson spiking, heterogeneous cells, a transfer curve with its own seed
spread, 2.6), so a correct learner is expected to lose some seeds. At q = 0.9, P(>= 13/15) = 0.82 and P(>= 14/15) = 0.55:
the judged rule of 14/15 would fail a correct learner that loses one seed in ten about half the time. At q = 0.95 the two
are 0.96 and 0.83. At q = 0.7, P(>= 13/15) = 0.13; at q = 0.5, 0.004. The replica's effect sizes put the arms far apart:
COV-PRE q = 0.983 on 60 seeds at its centre (59/60; one-sided 95% lower bound 0.92, at which P(>= 13/15) = 0.90), and
every rival q = 0 at its own centre on 60 seeds and 0/15 at all eight grid cells. The nulls get the mirror bar: a null
passing on >= 2/15 voids the GO, which a null with a true rate of 0.01 does with probability 0.01. 13/15 is also the S8
read bar (minor finding 13). The six gate seeds (42/43/44/100/101/102) are untouched by step 0; they belong to the organ
gate (8.1).

### 3.5 Step-0 bars (per seed; no label-shuffled threshold)

Weights in engine units: w_max = 20. Thresholds PER TARGET (2.6): every pool route is scored against theta_fire(P) /
theta_silent(P) and every entity route against theta_fire(E) / theta_silent(E), the values of the committed step-0a(i)
calibration (draft 8 / 4 and 8 / 4; the scratch forecast 8 / 4 and 9 / 4). They are measured before any learning arm and
never re-set after a learned weight is seen. The relative contrasts are round 1's, unchanged.

- P0 instrument: C_k[e] fires on >= 95% of utterances in which e takes part and holds k, <= 2% otherwise, measured at
  K = 0 and again at the end of exposure; C fires on <= 1% of windows without its H afferent; P_SPK and P_ADR occupancy
  conditioned on "the participant is not the speaker / addressee" <= 0.02; the empirical P_k base rate within +-0.05 of
  the schedule's; >= 2 E assemblies FIRING in every scene; the learned routes' transmission gate closed on every exposure
  step (read back); two runs byte-identical. P0 is first measured at K = 0 on the calibration seeds in step 0a(i), where
  it selects the C-weight row; a P0 failure there is a STOP with its own RESULT (3.4). At 0b it is re-read on every run.
- S0 route bar (absolute, from the engine's transfer curve). MUST FIRE, each route >= theta_fire: every pointer form's
  target pool (SHIFT-S, POSS-S -> SPK; SHIFT-A, POSS-A -> ADR; PLURAL -> SPK and ADR; the GIVEN forms ANA, DEM, POSS-G,
  ORD1, ORD2 -> GIV); the self's name -> E_SELF; every name -> its bearer. CLASS W, a route whose firing could cause one
  WRONG latch, each <= theta_silent: every pool route of a predicate, object, filler, name or OOI-1, and of PREV except
  PREV -> ADR (S5) and PREV -> GIV (a real overlap: the previous speaker is discourse-given); every non-target pool route
  of a single-target pointer form; every entity route of a predicate or filler; every non-bearer entity route of a name.
  CLASS H, reported: the entity routes of pointer forms and PLURAL's non-target routes; firing one adds a second latch to
  a token whose pool already fires, which the hold rule reads as HOLD (fail-closed), and the organ gate's recall bars
  count the cost (8.1). IDIO, the idiolect forced-fit marker, is barred by T9 and reported here. The morphology markers
  (4.1): ART, ASP and IND are CLASS W (every pool route <= theta_silent(P), every entity route <= theta_silent(E), and S3
  as filler); AGR and DEF are the HOLD-COST class, reported with the preregistered cost of 8.1 (T17), never in a GO bar. K = 0 fails S0 by
  construction.
- S1: each pointer form's target pool >= 2x its largest non-target pool, decoys included (PLURAL: each of its two
  targets).
- S2a: each non-possessive single-target pointer form (SHIFT-S, SHIFT-A, ANA-1, DEM-1, ORD-1, ORD-2): its target pool
  >= 2x its largest E route. The possessive forms and PLURAL are excluded (their entity routes are class H, reported).
  SHIFT-A -> E_SELF is also reported against its overheard dose (3.2).
- S2b: each name's bearer route >= 2x its largest pool route and >= 2x its largest non-bearer E route. Within-token, so
  no null and no dose floor; S0 already requires every name, however rare, to fire.
- S3: predicates, objects and filler: largest pool route <= max(theta_silent, 0.25x the smaller of SHIFT-S -> SPK and
  SHIFT-A -> ADR).
- S4: every token's decoy routes (PLURAL, a hold class, excepted) <= max(theta_silent, 0.2x that smaller shifter target);
  no token with a pool route >= 0.5x it has a decoy as its top pool. The LFT positive control proves the decoys are live
  at the organ gate (8.1).
- S5: OOI-1 as S3; PREV -> ADR <= 0.2x SHIFT-A -> ADR at turn-taking p = 0. Preregistered for the step-1 forced-fit arms:
  the ratio rises monotonically with p (replica 0.11, 0.17, 0.30, 0.41 at p = 0, 0.3, 0.6, 0.9) and reaches >= 0.5 in the
  dyad-only arm, where "the previous speaker" and "the addressee" are one relation in the input: a true ambiguity of the
  input (the confound behind a child's addressed-only "you"), reported as a finding and barred only at p = 0.
- S6 convergence (revised: the judged "margins at 4 tau and 6 tau within 20%" measured the transients of rare forms,
  not a defect): (i) every must-fire route non-decreasing between 2/3 of exposure and its end (tolerance 1 engine unit);
  (ii) the routes of forms heard >= 450 times (>= 300 by the checkpoint, at least 3 route time constants at lam*tau >= 4)
  within 20% between the two.
- S7 inertness: gain-0 synapses change by exactly 0; no plastic weight leaves [0, 20]; w_init = w_min = 0, so depression
  is visible.
- The saturated-route check (2.6) on the organ's circuit, in step 0a(i) on calibration seed 61 and after the census.
- S8 (the read) and T-carry (the two-token probe) need the REF commit stage, which is step 1a's build: they moved to
  step 1a and the organ gate (8.1; refuter major 4).

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
- Morphology markers (round-3 fixture finding; nonce tokens, rendered from the scene with no draw from the world's RNG,
  so every other token's stream is unchanged). Matching EN v1 (4.3), CLASS W in the GO bars: ART, the uniform article,
  before every object token with no possessive or demonstrative determiner and before ORD-1 / ORD-2 (EN 'the', 'the
  former', 'the latter'); ASP, the progressive frame marker, in 0.2 of utterances drawn from its own RNG, independent of
  the scene (EN's -ing frame; the perfect frame has no free token in EN v1). Co-varying stress markers that real teacher
  speech has and EN v1 does not render: AGR, person agreement, on the verb when the agent is the speaker, the addressee
  or both (the English present-tense base form); DEF / IND, a givenness-informative article before an object token whose
  object is given / not given. IND is CLASS W; AGR and DEF are the HOLD-COST class: their routes are reported per arm, and
  the organ gate's T17 (8.1) reports the holds they cause, with the cost preregistered there.
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
- Exposure 4800 utterances per seed (raised from 2400: the rarest scored form, POSS-G, needs at least one route time
  constant of its own occurrences, 67 at the COV-PRE centre lam*tau = 6, and is heard at least 83 times on every dev seed;
  3.2); dose arms 1600 and 3200 reported. Plasticity open over the whole
  utterance; the learned routes' transmission closed throughout exposure (2.3).
- Scene drive: every participant at the same high rate, present-but-uninvolved at the presence rate. Null arm: PART driven
  only on the referring expression's referent (the round-2 designs' scene); reported against the default.
- Marker rates on the default schedule (scratch, seed 3, 800 utterances): ART 0.49, IND 0.66, AGR 0.44, ASP 0.21, DEF
  0.02 of utterances. DEF is rare (given objects are mostly pronominalized); its route is reported with its dose.
- Every stream (the default, each null, each dose arm) is written to disk by the toy world and read by every learner
  (3.4); the nulls are streams of their own, with their own sha256.

### 4.3 The English fixture (production, step 3): `research/fixtures/deixis_exposure_en_v1.json`

- Generated by `research/runners/_deixis_fixture_gen.py` from the SAME distributions with English utterances, in the AI-teacher
  lesson shape (`research/runners/ai_teacher.py:23-33`) extended with scene fields (present, speaker, addressee, event,
  owner, utterance). Versioned, sha256-frozen in its header, declared as developmental exposure (teacher as social
  environment; teach-free).
- **The clause rendering (declared in the HD header; round-3 fixture finding).** Under COV-PRE a form's weight ends at
  (E[y|x] - ybar) / kap whatever its frequency, so ANY rendered form that partly predicts person or givenness gets a
  pointer route, and production then holds statements that contain it. The reviewer measured this on the replica
  ('am' -> SPK 13.1, 'are' -> ADR 13.4, a givenness-only 'the' -> GIV 12.3, a base-form verb -> SPK 6.3 and ADR 5.7; a
  uniformly used 'the' 0.6), and round 3 reproduced it with the toy's AGR and DEF (3.2). So the generator renders no
  free form that co-varies with person or givenness:
  - Word order: one SVO clause per utterance; a possessive determiner (R1 possessive or a fixture person's name's)
    before the object noun; a demonstrative determiner before a pointed object ("that cup"); the joint speaker+addressee
    agent as "we".
  - Tense and aspect: a frame drawn per utterance independently of the scene: simple past 0.6, present perfect 0.2,
    present progressive 0.2; regular verbs only, so the past and the participle are one form. The perfect's and the
    progressive's auxiliary is ALWAYS fused to its subject as one written form (i've, he's, kofi's style: "<subject>'ve /
    'm / 're / 's"), for pronoun and name subjects alike, so the -ed and -ing verb forms are person-invariant and no free
    auxiliary is ever rendered. No copular clause is rendered. Hence 'am', 'is', 'are', 'was', 'were' (as a verb), 'be',
    'been', 'have', 'has' and every present-tense verb form are never exposed: no agreement morphology is rendered.
  - Articles: 'the' before every object noun that has no possessive or demonstrative determiner, and in 'the former' /
    'the latter'; 'a' and 'an' are never rendered. 'the' therefore does not co-vary with givenness (the toy's ART, 4.1).
  - Consequence, stated: production's present-tense verbs ('likes', 'chases', 'like') and its copulas are unexposed
    forms; each drives an untrained spare and reads none. They can cause no hold and are not tested by the organ; the
    content nouns and names are.
- Exposed pointer forms come from a rendering table built by rules and frozen (sha256 in the fixture header) before any
  census arm runs. **Honest status (round-3 minor finding): R1-R3 are a declared teacher scaffold chosen AFTER the census
  probe set was seen.** No rule or CI test can show that a form was not chosen because of the census; the census numbers
  on these forms are fixture recovery. The anti-list evidence is the nonce arms, the K = 0 fixture and the synapse
  transplant on both readings (8.3).
  - R1, the English personal-pronoun paradigm of a reference grammar (Huddleston & Pullum 2002; Quirk et al. 1985), all
    persons, cases and reflexives, by grammatical role: i me my mine myself, you your yours yourself yourselves, we us our
    ours ourselves (speaker and addressee relations); he him his himself, she her hers herself, it its itself, they them
    their theirs themselves (the GIVEN relation); the demonstratives this, that, these, those (GIVEN, as pronoun and as
    determiner: "that guy", "this one"); "the former", "the latter" (GIVEN, two-candidate scenes only).
  - R2, frozen as a source's COMPLETE list, not a subset: every single-word form set in italics (the article's marking
    for nonstandard, informal and archaic forms) in the "Complete table" of the Wikipedia article "English personal
    pronouns", revision 1343514077 (2026-03-14T20:17:50Z), copied verbatim into the fixture header with the page's sha256,
    none dropped and none added. A multi-word entry ('you all', 'you guys') is excluded by one rule applied to every such
    entry, because its non-pronoun word ('all', 'guys') would be exposed only inside the entry, co-vary with the
    addressee and become a pointer; the excluded entries are listed in the header. Each R2 form takes the relation and
    role of the standard form the table aligns it with. The round-2 list ('em; ya, ye, yer; yous, youse; thee, thou, thy,
    thine), which matched the census residue, is withdrawn; 'em' stays as the R3 contraction below. (A read of that
    revision lists forms such as thou, thee, thy, thine, thyself, ye, y'all, youse, yeer, yeers, yeerselves, ourself,
    themself; the frozen list is the copy made at step 3, not this read.)
  - R3, the CLOSED contraction list and the chat-spelling rules. The frames above render exactly these contractions: i'm,
    you're, he's, she's, it's, we're, they're (progressive), i've, you've, we've, they've, he's, she's, it's (perfect),
    'em (them), and "<name>'s" for each fixture person (both frames, and the name possessive). No other contraction is
    rendered ('ll, 'd and n't never: so 'well', 'ill', 'hell', 'shell', 'shed', 'wed', 'id', 'cant', 'wont' are never
    produced). Rules, applied by the generator uniformly, each at 0.2 of the renderings it applies to, in this order:
    (a) you -> u, your -> ur, you're -> ur, yourself -> urself; (b) apostrophe deletion on any rendered form still
    carrying an apostrophe: im, youre, hes, shes, its, were, theyre, ive, youve, weve, theyve, em, y'all -> yall (if R2
    has it), and "<name>s". So no form is exposed only in its chat spelling.
  - A CI test (`tests/test_deixis_fixture_rules.py`) regenerates the table from R1, the committed R2 copy and the R3
    rules and fails if the fixture exposes any form they do not produce. That is a consistency check of the generator,
    nothing more: it cannot show that the rules were not chosen with the census in view. `moi` is produced by no rule.
  - Homographs the rules create or carry: 'were' (we're; the past copula in production), 'its' (it's; also R1), 'mine',
    'us', 'that' (R1; also a complementizer and a relative pronoun), 'u', 'ur', 'em', 'me' (if R2 aligns it with 'my').
    No word list filters them; their false holds are measure 6b, with a preregistered cost (8.2).
- **The class-W check on every non-pointer English form (step 3, before census v2 runs).** On the trained EN organ, dev
  seeds 7 and 1-5: every exposed form that is not an R1-R3 pointer form or a name ('the', every -ed and -ing verb form,
  every noun) has every pool route <= theta_silent(P) and every entity route other than its own object's <=
  theta_silent(E). A form that fails is named, census v2 does not run, and the lane STOPs for a ruling: that form would
  cause false holds in production.
- Dose floor per exposed form (declared in the HD header): every exposed form occurs at least 300 times in the fixture
  (at least three route time constants, 1 / (lam kap) occurrences, of COV-PRE anywhere in its window lam*tau >= 4, 3.2);
  the generator raises a rare form's rendering rate within its relation until it does, and records each raised rate in
  the header. A form that fails its route bar (S0) at step 3 is listed as under-dosed and its census items are counted
  as fail-open residue.
- The learner is named `brain` (the curriculum's self entity, `research/findings/raw/_curriculum_self_knowledge.json`
  self_reference).
- The content vocabulary (objects, common nouns such as cat, dog, fish, jazz, cousin; verbs only in their -ed and -ing
  forms) overlaps the probe set's content nouns on purpose, so false holds on words the organ knows are testable; the exposure persons' names never occur in any
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
per process, loaded from a versioned cache keyed by (fixture sha, seed, params including the FS constants and the
calibrated thresholds, the two heterogeneity flags `enable_parameter_heterogeneity` and
`preset_centred_parameter_heterogeneity` as the organ sets them, the git blob shas of `sim/bridge.py`, `sim/config.py`,
`sim/kernels.py` and the organ module): a `sim/` change such as the 10-20 heterogeneity fix or the covariance rule
therefore misses the cache instead of serving a stale organ; a fresh exposure is re-proven byte-identical to the cache at
every gate), `read_turn(chat, msg) -> DeixisRead(raw, per_token, latched, classes, reads, engaged, cause)`. `latched` is
the turn-start latch result (2.4); W1 turns it into a hold only at its own store point, so `held` (latched AND stored
flag-off) is decided where the store would happen, while W4, the echo and the scenario writer read `latched` itself.

- `webapp/server.py::brain_reply` (:5374), right after `_word_role_trace_reset(chat)`: `chat._deixis_turn = read_turn(chat,
  msg)` when the flag is on (None otherwise); cleared at the end of the turn and at the next turn's start (the
  `_multiref_referent_override` precedent, :5561). The organ runs when at least one routed writer would act on the turn:
  `_is_acquisition_candidate(msg)` is true (a turn the flag-off path would store), or the TUI fallback's `_discourse_turn`
  would write its register, or the false-belief writer's scene parse matches (`webapp/false_belief_chat.py:154-166`), or
  W4's own extractor finds an assertion (`surprise_production_organ.extract_assertion(msg) is not None`, the call
  server.py:6952 makes; round-3 W4 finding). Questions and unparsable statements cost nothing. A raise or a raw-text mismatch is a HOLD (2.4, fail-closed degrade).
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
| W4 surprise + reconsolidation | server.py:6925-7004 (`extract_assertion` at :6952) and `webapp/reward_value_afferent_chat.py:744` | skipped whole on a LATCHED turn, read from the turn-start latch result whatever the later B3 check decides (declared host skip; the block runs before the B3 reject at :7041); the record carries `w4: skipped_deixis_latched`; the surprise and reconsolidation notices of that turn are not prepended (H7, 8.4) |
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

9. **A user's claim that contradicts a self-fact the brain already holds** (round-3 W4 finding). W4 runs before B3:
   `extract_assertion('you are a brain that has spikes')` gives ('brain', 'has', 'spikes'), `what_does('brain', 'has')`
   recalls the curriculum's 'neurons', the surprise read fires, and reconsolidation REWRITES (brain, has, neurons) to
   (brain, has, spikes) in place. Flag on, the turn is latched ('you' latches RF_ADR, 'brain' latches S_SELF), W4 is
   skipped whole and the self-fact is byte-unchanged. The H7 arm (8.4) bars it.

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
showing both routes closed and zero P and E spikes on every pointer and name token. The learning-rule arms of step 0b
carry their own engagement record (gate values read back during exposure, the count of steps the update applied, sum |dw|
per learned block, the rule configuration read back from the bridge, the stream sha256; 3.4): a failing rule arm without
it is UNDEFINED, and an UNDEFINED arm earlier in the GO order blocks every later arm's GO.

**Where it ran (refuter major 8).** The accepted baseline came from two host classes (pool and server). Every RESULT
quotes `tools/where_ran.py`'s host summary for its raw directory. Census v2 and every production arm record the source
(`tiny-demo +LTM` as the Stage-2 prereg requires), `corpus_present`, the LTM bundle sha256, the organ-cache key and
hit or miss, and the host class; the flag-off identity re-run states its host class, and a flag-on vs flag-off comparison
across host classes is UNDEFINED. Every number a RESULT cites carries the where_ran class of the file it came from (one
column per cited number, not one summary per RESULT). The calibrations, the two-run byte-identity check and the census
run on one host class, the CPU pool; a calibration made on another class is repeated on the pool before it is used.
Every number in this design that comes from `docs/drafts/deixis/s1-scratch/` ran on the workstation (dant123-wk, class workstation), is
not provenance-stamped and is cited as a forecast only; step 0a regenerates each as a provenance-stamped module run.

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
| T17 marker-only statements (the HOLD-COST class): every referent named, the only marked form AGR (6 items) or DEF (6 items), production condition (no H afferent) | 12 | reported per marker, per seed: the fraction held. Preregistered cost: holds are EXPECTED (the replica puts AGR at 6-7.4 units, between the thresholds, and DEF at 13, firing GIV); they are the false holds teacher speech that renders agreement or informative articles would cause. EN v1 renders neither (4.3), so the production cost is 0 by construction and is checked by step 3's class-W check; the number travels to the AI-teacher replacement and to stage 2 (resolution binds DEF to the given entity), and is never a GO bar |
| class-W markers ART, ASP, IND | all | bound or held through them 0 items (they are in every T item that renders them) |
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

Projection (`docs/drafts/deixis/s1-scratch/fix/project_s1hold_v2.py`, pure arithmetic on the recorded W1 outcomes plus the six stores the
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
| 6 non-deictic stored item lost | 0 | 0: every stored probe item with no exposed form (p016, p035, p036, p109) and a frozen natural non-pointer set, each shown stored flag-off at the merge-base (an item B3 does not store is replaced, never kept): articles 'the dog chases a cat', 'kofi chases the cat', 'the dog chases the cat', 'the girl reads the book' and the Stage-2 gate's teach turns (8.5); 'are' 'cats are good pets', 'dogs are loyal friends', 'kofi and fenna are friends', 'the birds are eating seeds', 'the kids are playing football'; base-form plural verbs 'dogs chase cats', 'birds eat seeds', 'the boys play chess' (B3 extracts a triple from each today, checked with `extract_polar_assertion`) | |
| 6b homographs | 0 | reported with a preregistered cost, ruling requested before the gate, one item or more per homograph, each shown stored flag-off: the self's name as a common noun 'a dog has a brain'; 'mine' 'kofi owns a mine'; 'were' (R3's we're) 'kofi and fenna were friends', 'the cats were hungry animals'; 'that' as a relative pronoun 'the dog that barks bites'; 'its' (R1, and R3's it's) 'the dog wags its tail' (an intra-sentential anaphor: held by design, its cost is measure 7's kind); 'u' 'kofi typed u'; 'ur' 'kofi visited ur'; 'em' 'kofi typed em'. 'us' needs no item: B3 strips it, so no 'us' statement is a candidate | #260 (role context) |
| 7 correct facts lost (declared recall cost, accepted by ruling BEFORE the gate) | 0 | 5 coordinations with a true partial triple (p011, p027, p028, p113, p114); 16 over-general possessive stores (p015, p024, p025, p026, p039, p040, p043, p044, p047, p048, p051, p052, p055, p056, p059, p060); 1 over-general demonstrative (p010, stored as `(guy, like, jazz)`) | S1-bind and stage 2 store the bound version |
| former / latter / demonstrative arm (single turn and two-turn, oracle v2 items) | stored | HELD 100% on 6/6, and its made-up-marker twin held >= 29/32 per seed | |
| W2 legacy (36 / 0 / 52 / 68) | | the same holds (the read runs before the legacy split) | |
| W3 echo | every person-subject item | 0 echoes for exposed forms (the organ reads every turn the fallback would echo) | `moi` |
| W4 extract | 28 | <= 3 (W4 is skipped on every latched turn); 0 raises on the production-shaped arm; every W4 skip recorded with its turn | `moi` |
| W5 false-belief pointer agents | measured at the merge-base | 0 scenarios started or advanced with an exposed form as the tracked agent | Stage-3 C3 (the tracked agent bound by the organ) |
| W6 affective-ToM English split | measured at the merge-base | unchanged (reported) | S1-bind |
| cross-speaker B3 (BRAIN_ONE_BRAIN=1) | measured at the merge-base | no increase | S1-bind |
| question side | i/my/me -> brain; "we" literal | unchanged (S1-hold touches no question) | S1-bind (exposed forms), stage 2 (it/its) |
| slug scan | 17 tokens, 0 conversation-only | unchanged | |
| EN-HO, a front-end PIN, not a measure (round-3 minor): the R3 u/ur/urself family withheld | | those items stored exactly as flag-off. An unexposed form drives an untrained spare and reads none, so the front end alone meets this: it pins the front end, it measures nothing about learning, and no generalization to an unheard spelling is claimed | stage 3 |

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

Every H item is a statement the flag-off arm is shown to store in the record (`docs/drafts/deixis/s1-scratch/fix/b3_item_check.out` is
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
- H7 (W4 rewriting a held self-fact; 6.9), on a brain holding the curriculum self-facts
  (`research/findings/raw/_curriculum_self_knowledge.json` facts, e.g. (brain, has, neurons), (brain, uses, spikes),
  (brain, learns, words), (brain, remembers, facts)): contradicting claims whose W4 extract has the self as agent, both
  forms 'you are a brain that has spikes', 'you are a brain that uses magic', 'your brain learns nothing', and self-name
  only 'the brain has wires', 'brain uses magic', 'the brain remembers nothing' (each extract checked:
  `extract_assertion` returns ('brain', <verb>, <patient>)). Flag off, the record must show for each item that surprise
  engaged (`surprise.surprised` True) and the rewrite happened (`reconsolidation.action == rewrite`); an item where it
  did not is replaced, never kept (so the arm can fail). Flag on: every curriculum self-fact byte-unchanged in composer,
  self-knowledge, LTM and the persisted bundle (sha256 of each store before and after), and W4 recorded as
  `skipped_deixis_latched` on every item; 6/6, on intact and on the lesion arms for the items each route owns (`ptr`: all
  six, through the self-name; `name`: the three 'you' items). Control, reported: 'the brain uses no spikes' (W4 extracts
  nothing, four content words; B3 rejects it as a contradiction of (brain, uses, spikes)): the reply and every store
  byte-identical flag on vs off.
- H6 (quoted, reported, identity claims; 6.8): 'kofi: "i hate jazz"', 'kofi: "you hate jazz"', 'this is kofi, i like
  jazz', plus any further item shown stored flag-off: HELD 100% on 6/6.
- Two users (completeness minor 11): H1-H7 also run under `BRAIN_ONE_BRAIN=1` with two sessions sharing one store seeded
  with flag-off-era deictic names; session B's replies are scored for A's held statements (0 leaks).
- E3: the `deixis_hold` row's two driving pairs diverge on 6/6, reproduced, on the rich default shape; the double-lesion arm
  reported.

### 8.5 No regression, answer rate, drift, identity, resources

- No regression (flag on, floor state equal in both arms): the SELFID battery, Stage-1 C1-C4, the Stage-2 dev arms (conv,
  infer, ask, emo, inner) with flag-off battery arms; the D6 question-side rows; category-inherit, D5 and B3 batteries on
  non-deictic inputs; the `tom-false-belief` LBF row (`research/runners/lbf_rows/tom_false_belief_chat.py`, its
  `tom_fb` turn verbatim: "Sally puts the marble in the basket. Sally leaves the room. Anne moves the marble to the box.
  Where will Sally look for the marble?"), whose answer must be unchanged flag on vs off on 6/6 (the organ reads every
  turn the false-belief scene parse matches, and every scene sentence carries 'the': a learned 'the' -> GIV route would
  hold the scene and change the answer); idle replay voices 0 held facts (it cannot: none are stored).
- Answer rate (refuter major 7; RULING answer-rate-off). Preregistered FLIP-HURTS, by name, needing a ruling in the
  prereg, never assumed: the two driving pairs; the 59 held census items; and 'tell me more' and 'how do you feel' each
  AFTER a held statement (round-3 finding: it cannot be shown that they cannot hurt; after 'mine likes jazz' the
  flag-off brain has stored a fact and its topic, so 'tell me more' can elaborate where the flag-on brain, having stored
  nothing, may abstain). Everywhere else the flag must not cost an answer. The battery: the LBF row `deixis_hold` writes
  one arm file per probe, the probe being the arm file's LAST answered turn (the only turn `tools/answer_rate.py`
  reads), under `research/findings/raw/_load_bearing/_shards/<tag>/s<seed>/deixis-hold-ar/`, each named
  `intact_<probe>.json` (`paired` reads only `^intact` files; a directory of its own, so the row's named driving pairs,
  which abstain by design, are not in it), at two request shapes (session `default` with no interlocutor, and the
  Stage-2 harness shape), one tag pair per shape. Probes:
  - the named FLIP-HURTS probes above ('tell me more', 'how do you feel' after a held statement), reported apart;
  - a question about a stored non-deictic fact after a held statement;
  - **a frozen natural set this design did not choose:** every intact `"teach"` turn of the Stage-2 exit gate's groups
    conv, infer, ask and inner (`research/runners/stage2_exit_gate.py` on `research/stage2-exit-gate`, taken at the
    11-03 GATE_SHA pin, its sha recorded; e.g. 'the fox chases the hare', 'the mole is hunted by the owl'), each
    session's teach turns followed by that group's own probe turns, in the gate's order;
  - the false-belief probe (`tom_fb` above);
  - the Stage-2 dev arm probes (conv, infer, ask, emo, inner); a statement with no exposed form and its question.
  Tags: `deixis-s1-on-<shape>-<sha>` (flag on) and `deixis-s1-off-<shape>-<sha>` (the same cells with the flag off, run
  normally: NOT an alloff cell). Command, per shape: `python tools/answer_rate.py paired --on-tag
  deixis-s1-on-<shape>-<sha> --off-tag deixis-s1-off-<shape>-<sha> --faculty deixis-hold-ar --json >
  research/findings/smoke/deixis_s1_answer_rate_<shape>.json`; the per-seed report is quoted in the RESULT; any
  FLIP-HURTS-ANSWERS on a probe outside the named list is a NEEDS line, not a pass.
- **Held fraction of non-census statements, beside the answer-rate report:** per seed and per shape, the fraction of
  the statements of the natural set, the measure-6 set and the false-belief scene that the flag-on arm holds (with the
  class of each hold), so the flip ruling prices the real recall cost and not only the census's. A follow-up battery
  (teach a census statement, ask about its non-deictic participant) reports the loss per class.
- Drift: 200 addressed chat turns leave the organ's weight sha256 unchanged (plasticity is closed outside exposure).
- Identity (E0): flag off byte-identical (census v1 at the merge-base; `A1_COMMIT_BATTERY_SHA256`; the word-role pins; the
  `_is_acquisition_candidate` mirror; the `hebbian_cov=0.0` pins).
- Resources: organ load time (from cache) and per-turn added latency reported; bar <= +1.0 s median per statement turn and
  0 added latency on questions; `_free_slots` identical flag on vs off after a 500-turn churn.

## 9. Made-up-marker discipline (RULING c1-learned-1016; RULING wrc-frame-go-1020b)

Organ level: every token is a nonce per seed (4.1); the K = 0, transplant, permuted-teaching, held-out-voice, decoy,
out-of-inventory and shuffled-order arms (8.1) are what separate a learner from a label table. Production level: the MM
fixture arm (8.3) covers every exposed form of all three relations (both persons and the GIVEN forms), not only the second person. Bar sanity (`docs/drafts/deixis/bar_sanity_spec_s1.py`, n >= 1000
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
never read, and no name in the text (an in-text identity claim, 6.8) ever rebinds the voice. Declared residue
(completeness major 5): two people typing into one conversation share its minted token, so S1-bind would bind both first
persons to one voice; the S1-bind packet counts it with a same-token two-typist arm and names its method (a per-person
token from an authenticated client login: a world afferent, never a name in the text), dated with S1-bind. A known voice is a fixed full
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
| the exposure fixture and its rendering table: the clause rendering (SVO, the person-invariant tense frame, auxiliaries fused to the subject, no copula, no agreement, 'the' before every undetermined object noun, no 'a'/'an'), R1 by role, R2 the pinned source's complete single-word italic list, R3 over the closed contraction list; the per-form dose floor (rendering rates raised until every exposed form occurs >= 300 times) | teacher scaffold, chosen after the census was seen (4.3) | human or AI-teacher multi-party speech through `ai_teacher.py` (which will render agreement and informative articles: the toy's AGR / DEF cost, T17, travels with it) |
| the toy's morphology markers (ART, ASP, IND, AGR, DEF) | world (test-only nonce rendering of the toy scene) | |
| form -> assembly front end (a form outside the lexicon drives a spare, never examined) | bookkeeping, declared | |
| the fail-closed hold rule on the latch reads (which statements are kept), including the hold on an unavailable organ or a mismatched text | **scaffold** | #260's statement path writing latched referents into the composer's spiking store; M3b storing held claims as told |
| which turns the organ reads (an acquisition candidate, a TUI echo, a false-belief scene parse, a non-None W4 `extract_assertion`) and the per-turn relay to W1/W2, W3, W4, W5, with their skips (W1/W2 on a held turn; W3, W4 and W5 on the turn-start latch result, whatever B3 later decides); the `_direct_fact` fallback skip | **scaffold** | the same |
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
| 10-22..10-24 | step 0a: (i) the calibration on the hearing core, plasticity off (per-target curves and thresholds, the C-weight row by P0 at K = 0, the unit conversions, per-arm step time, two runs byte-identical, the cell-type dump), committed with its sha256; (ii) the streams, the replica and the audit check; (iii) the bar-sanity record; `[deixis-271-s1-0a] RESULT`, STOP. The only step that runs without a ruling | calibration 61-70; replica dev 1-15 and 1-60 (rate level) | no |
| after RULING on 0a; 10-24..10-27 | step 0b: one run per arm on dev seed 7 (engagement record and wall time; no threshold or conversion is set here any more), then the census sweep on `tools/sweep_pool.sh` under `memcap` with the queue plan from 0a's measured step times; `[deixis-271-s1-0b] RESULT`, STOP | dev 1-15 | no |
| after RULING on 0b; 10-27..10-31 | step 1a hearing core with the REF / RF / AMB / SELF stage, S8 and T-carry; step 1b speaking + efference; the `sim/` rule with its pins (COV or COV-PRE GO); organ prereg (flip: no) filed by 10-31; `[deixis-271-s1-1] RESULT`, STOP | dev 1-8 | no |
| 11-01 | defaults pinned: `BRAIN_DEIXIS_HOLD` OFF | | |
| 11-01..11-02, else 11-11..11-13 | step 2 organ six-seed gate (only if step 1 is ruled by 10-31; never inside 11-03..11-10) | gate | yes |
| after the step-2 verdict: 11-03..11-10, or 11-14..11-20 if step 2 slipped | step 3 production S1-hold wiring, census v2, EN + MM + EN-HO fixtures, LBF row; dev seeds 7 and 1-5 only; the `sim/` rule merges to main after the 11-03 GATE_SHA pin. If the organ gate fails, step 3 does not start and no English fixture is built | dev | no |
| 11-11..11-18, or 11-21..11-28 if step 2 slipped | step 4 S1-hold production gate + flip prereg with flag-off battery arms; verify-go | gate | yes |
| target 11-24..11-28, or 12-01..12-05 if slipped (integration-first: a dated target, after the 11-21 Stage-2 score, never in the pin) | S1-hold flip on main | | |
| after the step-4 ruling | S1-bind packet; S1-bind dev to 11-30; S1-bind gate 12-01..12-10; flip prereg targets 12-19 | | |
| 11-28 / 12-01..12-12 / 12-15..12-18 | #271 stage 2: packet / dev / gate (10) | | |

**Compute, recounted from the packet's own arms (refuter major 4).** S8 and T-carry left step 0 (they need the REF commit
and release stage, which is step 1a's build), so step 0b needs the TOK / E / H / C / P circuit with the mode gates and no
commit stage. Exposure is now 4800 utterances of about 200 steps, about 0.96 M steps per run; the reviewers' 0.31-1.7
ms/step (5-27 min per run) was not measured on this circuit and leaves out the runner's per-step covariance update on
about 67k plastic synapses and the per-step gate read-back, so step 0a(i) measures the per-arm step time with that update
executing (at learning rate 0) and the 0a RESULT posts the queue plan before 0b is queued. Step 0a itself: the calibration
(10 seeds x the curve grid x 2 targets x their conditions, short runs; the P0 grid at K = 0, 600 utterances per seed per
row until a row passes; the step-time runs, 6 x 2,000 steps; two identity runs) is CPU-minutes to a few CPU-hours, and the
replica and audit check are CPU-minutes. Step 0b runs: one run per arm on dev seed 7 first (6 runs); the census, 6 arms x a 2x1 grid (each arm's 0a centre and the next grid value up) x 15 seeds = 180 runs; the
nulls only on an arm that passes the weight bars, 2 exposure nulls (label-shuffled stream, scene shuffled) + voice permuted
= 3 x 15 = 45 runs (K = 0 is scored on the initial weights, no run); the dose arms move to step 1, after the ruling. Total
about 235 runs, 20-105 CPU-h. The label-shuffled THRESHOLD runs of the judged design (5 per seed per arm) are gone: no bar
needs them (3.5). Step 0a(i)'s measured step times set the queue plan; if 235 runs do not fit the pool by 10-27, the rival
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
circuit shifts the TOK -> E curve (measured per target in step 0a(i) before any learning, 2.6; the scratch forecast moves
theta_fire(E) to 9, which the replica tolerates, and a strong FS pool of real FS cells would move it to 16, which stops the
lane: the FS constants are frozen and the heterogeneity setting is in the cache key); a form of the English rendering
that co-varies with person or givenness becomes a pointer under COV-PRE (EN v1 renders none; step 3's class-W check
names any that slips through; the toy's AGR and DEF price what real teacher speech will cost, T17); a noisy P pool may fire
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

## 15. Reuse map (file:line, verified at `0849ef0ca`; round-3 additions verified at `da12c3aac`)

- Round 3: `sim/config.py:189-244` (`enable_parameter_heterogeneity`, `preset_centred_parameter_heterogeneity`, default
  OFF), `sim/bridge.py:3620-3660` (the preset-centred draw); `sim/enums.py:597-604` (the IZH2007 FS preset);
  `research/runners/_agency_authorship_tag_derisk.py:141-148` (RS / FS preset regions) and `:88-92` (the FS motif
  weights 8 / 30); `research/runners/surprise_production_organ.py:75-80` (`_FUNCTION_WORDS`) and `:193-205`
  (`extract_assertion`); `webapp/server.py:6950-7001` (W4: extract, `what_does`, `judge`, `reconsolidate`) before the B3
  reject at :7041; `research/runners/lbf_rows/tom_false_belief_chat.py` (`EXTRA_TURNS` `tom_fb`);
  `research/runners/stage2_exit_gate.py` on `research/stage2-exit-gate` (the teach turns and group probes, lines 139-241
  at `eddb1a071`); `research/runners/__init__.py` (the automatic `.prov.json` sidecar for `--out`); `tools/where_ran.py`
  (host classes).

- `research/runners/spiking_commit.py`: `commit_plan`, `latch_read` (:157), `read_with_variants` (:165), `sequence_read` (:174).
- `research/runners/_agency_authorship_tag_derisk.py`: the production -> cd -> src_self / sensory_marker -> src_other
  cross-inhibition motif and its constants (W_ACT_CARRIER 60, W_CARRIER_SRC 60, W_SRC_INH 8, W_INH_SRC 30, FLUSH_STEPS 70).
- `research/runners/word_role_parser_organ.py`: the process-backend pattern, NW / spares front end, named plasticity gates;
  the 1500 pA token drive and `W_MAX = 20` (:127); the commit stage's fixed W_RG 40-60 (:174-183).
- `sim/bridge.py:4506` `inject_explicit_wiring`; `:5467 set_plasticity_gate`; `:5493-5517 set_transmission_gate` (gates the
  current, not the update); the conductance increment `w x propagation_strength` (:9258-9261, :9269-9272 without the E/I
  split), its decay (:2910) and the external-current sum (:9327); the Hebbian block `:10205-10409` (traces `:10226-10235`,
  the co-firing condition `if _prev_any and _fired_any` :10236, BCM `:10248-10273`, decay `:10349-10380`, clip
  `:10390-10411`, gain > 0 plastic synapses only).
- `sim/config.py`: Izhikevich-2007 RS defaults :20-23, `E_e` :392, `syn_tau_g_e` :394, `hebbian_oja` :710, `hebbian_bcm`
  :721-723, `propagation_strength` :871, `hebbian_weight_decay` :878, `hebbian_min_weight` :879, `hebbian_rate_window`
  :899; `sim/kernels.py:76` (the Izhikevich update), `:1115` (`I_syn = g_e (E_e - v)`); `sim/regions.py:528`
  `exc_receptor` ("nmda_slow").
- `research/runners/_agency_authorship_tag_derisk.py:88-92` (the fixed motif weights); `research/runners/rich_answer_composer.py:518-531`
  (`_direct_fact` and its DIRECT-LTM-TOPIC fallback); `webapp/false_belief_chat.py:58, 154-166`;
  `research/runners/affective_tom_production_organ.py:149, 157, 184-185`; `webapp/server.py` `_brain_rich_default`
  :3719-3738, W6 call :3345, W5 :6683-6690, `_curiosity_followup` :7150 (called :7459); `brain_chat_tui.py:2661-2667` (the
  TUI local fallback).
- Fix-round scratch (run, `docs/drafts/deixis/s1-scratch/fix/`): `wscale_curve.py` and `fI_probe.py` (the engine's transfer curve and
  f-I), `replica3.py` with `final_v3.py`, `final_v3b.py`, `s6_check_lt6.py` (the step-0a draft), `project_s1hold_v2.py`
  (the census projection), `b3_item_check.out` (which honesty items flag-off stores).
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

## 16. Round-1 required changes: where each is answered, and which moved

Status words: ANSWERED (in stage 1, with a bar); MOVED (to a named stage with a date; not answered here).

1. **Rule. ANSWERED.** No outgoing-weight bound. A fair census among the engine's rules first, every arm through one tuning
   procedure and one audit (3.4); COV and COV-PRE as the fifth and sixth arms, added to sim/ only on their GO, with a biology
   card, flag-off byte-identity pins, a sign test and a parity test on a replayed raster (3.3); the separation argument
   written per synapse as a fixed point, with the replica numbers and the preregistered failure of the rivals (3.2).
2. **Anaphor on one bridge. Interim ANSWERED, route MOVED.** The interim condition (anaphors fail closed) is answered by a
   learned route: third person, demonstratives and former/latter are exposed through the GIVEN relation, whose afferent
   production never renders, so they are HELD by referential failure (2.1, 4.3; T16, the former/latter arm of 8.2 at 100%
   on 6/6). The resolving route (mention traces, discourse slots, AMB order) is stage 2 of #271 on the organ's bridge,
   with a spiking coincidence as its training signal; packet 11-28, gate 12-15..12-18 (10).
3. **Spiking decisions. ANSWERED.** Name and pointer drives converge on one REF commit stage; the outcome is a latch read
   at x0.5/x1/x2; referential failure is the named population RF_k, ambiguity the AMB population; no host margin (2.4).
4. **Controls. ANSWERED.** Two decoy pools fed by per-person coin bits, a positive-control token, an out-of-inventory
   marker (OOI-1), a forced-fit marker (PREV) with turn-taking and dyad-only arms and a preregistered prediction, a plural
   marker that must co-commit; plasticity open over the whole utterance with every participant driven; name usage
   preregistered (4.2); token-order-shuffled exposure (DX11).
5. **Second person to self. First half ANSWERED, second half MOVED.** Held, never stored, never in composer, SELFID,
   self-knowledge, idle replay or cross-user voicing; the self-name route holds third-person self claims; H1-H7 built from
   items the flag-off arm stores, on intact and lesion arms; a phenomenal arm (6, 8.4). Storing the claim as told-by-X
   (reported speech bound to its source) is stage 2's M3b, dated with stage 2 (10).
6. **Efference lesion. Toy half ANSWERED, production half MOVED.** Toy: `dx_cd` closed gives 0 SHIFT-S for non-self; a
   user voice injected during production gives 0 first-person renderings intact; the cross-inhibition lesion must induce
   the failure (X1-X3). Production has no efference source, so S1-bind does not voice; the production X1 bar and its cd
   lesion travel with #272's reply-side spiking word-commit (10).
7. **Unauthenticated voice. MOVED to S1-bind (answered by construction in S1-hold).** S1-hold has no voice at all. S1-bind's
   rule (server-minted token; every client-chosen session is UNK; the typed name and in-text identity claims never read) is
   fixed in 10 and bound to its own packet (gate 12-01..12-10).
8. **Moat. ANSWERED for stage 1.** Stage 1 parses no output and stores nothing person-bound; the floor lift is stage 2's
   sub-flag, never for a self triple (10, 11).
9. **Per-speaker scoping. MOVED to S1-bind.** S1-hold stores nothing per speaker. S1-bind: entity keys per conversation,
   `_free_slots` headroom barred, exhaustion fail-closed (10).
10. **Writer routing. ANSWERED.** W1 explicit (`_maybe_acquire`, both branches, the mirror), the held outcome in
    `gate`/`gate_extract` and on the rich default path, W3 with the organ reading every turn it would echo, W4 with its own
    list and string reconsolidation, the B3 reject kept, W5 and W6 added with stage-0 baselines (5.1, 5.2, 8.2); idle
    replay, persistence and `_restore_facts` are covered because nothing new is stored (E0 identity).
11. **#129/M3. ANSWERED by removal.** No parallel source role, no told line, no teller key in stage 1; stage 2 builds M3b
    on the organ's bridge; the policy-B lead is KEPT until the M3b readback drives the wording (10, 11).
12. **Two-candidate anaphora and drift. ANSWERED.** Two-candidate items are exposed (ORD-1/ORD-2, former/latter) and HELD
    by AMB or co-commit: T15 36/36 in the toy, and the production former/latter arm HELD 100% on 6/6 (8.1, 8.2). Chat-phase
    plasticity is closed; 200 addressed turns leave the weights byte-identical (DX16, 8.5).

## 17. Round-2 critique findings: disposition (every BLOCKING and MAJOR)

| Finding (design: lens) | Disposition |
|---|---|
| LSB:one-brain B1 bound-but-unparsed fallback to the raw msg | fixed: the hold replaces a store; no resolved text, no fallback (2.4, 5.2) |
| LSB:one-brain B2 English table labelled by the oracle | fixed: standard paradigm chosen independently; no census file read (CI); the table declared; census = fixture recovery; learning credited to the nonce arms (4.3, 8.2, 8.3) |
| LSB:one-brain M production anaphora not fail-closed (H_MEN) | fixed: no mention traces in stage 1; production renders no givenness afferent, so a GIVEN form fires its learned route into RF_GIV and the turn is HELD (2.1, T16, 8.2); quiescent reset per turn (5.1) |
| LSB:one-brain M closed C<->Q loop | fixed: mode gates; the learned routes closed in encoding; P0 at K = 0 and at the end; T-carry at step 1a (2.3, 3.5, 8.1) |
| LSB:one-brain M scene singles out the referent | fixed: every participant at the high rate; the null arm (2.1, 4.2) |
| LSB:one-brain M told/source route across two bridges | fixed by removal; M3b in stage 2 (10, 11) |
| LSB:one-brain M HD header understates; allowed-by wrong | fixed (12; the packet header) |
| LSB:one-brain M `deixis=` kwarg never reaches W1 | fixed: turn-scoped chat attribute; wrapper tests (5.1) |
| LSB:one-brain M honesty bars untestable; closure fails open | fixed: items the flag-off arm stores; fail-closed read rule; lesion arms (2.4, 6, 8.4) |
| LSB:measurable B1 GO rule unreachable under the clip | the round-2 fix was itself unreachable (final verification, BLOCKING 1); fixed in the fix round: within-token bars plus the absolute S0 route bar, COV-PRE as the sixth arm, GO at 13/15 with its binomial justification, the replica as step 0a with a STOP rule (3.2-3.5) |
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
| DXB:measurable B1 absolute decoy bar unreachable | fixed: coin decoys; the decoy bar is max(theta_silent, 0.2x the shifter target), theta_silent from the engine's transfer curve; the replica (2.6, 3.5) |
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
| S1:measurable M ratio bars depend on unpinned bounds and a transient | fixed: weights in engine units [0, 20] with w_init = w_min = 0; the thresholds from the transfer curve; ybar gated on the open state; S6 revised (non-decreasing, dosed routes within 20%) (2.6, 3.3, 3.5) |
| S1:measurable M the cheapest falsifier ordered after the sim/ edit | fixed: 0a replica, 0b runner-applied with the commit read, sim/ only on GO (3.4) |

Minor findings are folded in where they touch the same text (X4's derived bound, S4's restriction, nonce hygiene, the C0
re-baseline rule, the determiner set, the toy test items carrying no PART, PREV's outcome class, the persistent-slot and
snapshot-helper citations corrected by not citing them).

**Round 3 (second fix round, 2026-10-22).** The adversarial review of the fix round (1 BLOCKING, 4 MAJOR, 3 minor;
`the fix-round workflow record (not committed)`) is dispositioned finding by finding in `docs/drafts/deixis/FIX_NOTES-s1.md`, section "Round 3".

**Final verification (fix round, 2026-10-22).** The refuter (14 findings: 2 BLOCKING, 7 MAJOR, 5 minor) and the
completeness critic (16 findings: 1 BLOCKING, 8 MAJOR, 7 minor) are dispositioned one by one, with the section that
changed or the reason nothing did, in `docs/drafts/deixis/FIX_NOTES-s1.md`.

## 18. File plan

- `research/coordination/deixis-271-s1-PACKET.md` (research/coordination/deixis-271-s1-PACKET.md, the HD header up front; checked with
  `python tools/gates/host_decision_declared.py check`).
- `docs/drafts/deixis/bar_sanity_spec_s1.py`.
- Step 0a (round 3; each additive, default OFF, no `sim/` edit, run as a module with `--out`):
  `research/runners/_deixis_toy_world.py` (the toy world with its markers and the stream writer; from
  `docs/drafts/deixis/s1-scratch/fix2/replica4_markers.py::world_m` and `stream_audit.py::write_stream`), `_deixis_hearing_core.py` (the
  step-0 circuit, PACKET Appendix C), `_deixis_calibration.py` (step 0a(i)), `_deixis_rule_replica.py` (the six arms, the
  bars of 3.5 with per-target thresholds, the tuning procedure and the STOP rule, reading streams; from
  `docs/drafts/deixis/s1-scratch/fix/replica3.py` and `fix2/replica4_markers.py`), `_deixis_audit_check.py` (the audit of 3.4 under the
  spiking-noise model; from `fix2/stream_audit.py`), `_deixis_bar_sanity_0a.py` (step 0a(iii)). The scratch folders
  `docs/drafts/deixis/s1-scratch/fix/` and `fix2/` are committed beside the design as `docs/drafts/deixis/s1-scratch/` so each port can
  be diffed against its source. Later steps: `_deixis_rule_census_derisk.py` (step 0b),
  `deixis_binding_organ.py`, `_deixis_toy_gate.py`, `_deixis_fixture_gen.py`, `deixis_production_router.py`,
  `lbf_rows/deixis_hold.py`.
- `research/fixtures/deixis_exposure_en_v1.json` (the clause rendering and R1-R3 of 4.3, the committed R2 copy and the
  page sha256 in its header), its held-out-spelling twin `deixis_exposure_en_ho_v1.json` (EN-HO, a front-end pin); `tests/data/deixis_nonces.json`; oracle v2 beside v1 on `research/deixis-census` (8.2).
- `tests/test_deixis_flag_off_identity.py`, `test_deixis_hold_wiring.py` (every wrapper combination, `BRAIN_GNW_BUS_HOST=1`
  included; the mirror; one told-store-free assertion: 0 `inner.hear` calls on a held turn), `test_deixis_nonce_isolation.py`,
  `test_deixis_no_census_read.py`, `test_deixis_fixture_rules.py` (every exposed form from R1 + R2 + R3),
  `test_deixis_fail_closed.py` (organ unavailable, raising or mismatched: held, counted, on the server, TUI and rich
  default paths; the flag refused at startup without a trained organ), `test_hebbian_cov.py` (only with COV or COV-PRE).
- Flag-guarded edits: `brain_chat_tui.py` (W1 sentinel, gate/gate_extract, the mirror, W3), `webapp/server.py` (the
  `brain_reply` read, the W4 skip, the W5 skip), `webapp/reward_value_afferent_chat.py`,
  `research/runners/rich_answer_composer.py` (the `_direct_fact` short-circuit); `sim/config.py` + `sim/bridge.py` only
  with COV or COV-PRE (`hebbian_cov`, `hebbian_cov_leak`, `hebbian_cov_ybar_alpha`, `hebbian_cov_pre_floor`,
  `hebbian_cov_ybar_gate`).
- `research/biology/deixis-shifter-learning.md` (with the GIVEN relation); `research/biology/covariance-ltd.md` (only with
  COV or COV-PRE; the outstar and weight-dependent depression for COV-PRE).
- The two preregs (organ gate, production gate), each `flip: no`, each with this HD header.
- Two `research/FAILURE_LOG.md` lines: round 1 leaned on a rule sim/ lacks (gated by the step-0 census and
  `test_hebbian_cov.py`); round 2 set a GO rule its own correct learner could not meet and a weight range that could not
  fire a pool (gated by the step-0a STOP rule, the bar-sanity record, and the saturated-route check). A roadmap Stage-2/3 row and the board #271 sync.
