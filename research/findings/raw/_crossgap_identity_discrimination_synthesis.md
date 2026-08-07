# Cross-gap synthesis: discriminating overlapping attractors by IDENTITY (2026-08-07)

Unifies the shared boundary that #3 source monitoring, #4 replay, and the gap#5 DG arc all converged on this session.
Research synthesis (read-only across the cluster + DG arc). No code. Sets the ONE decisive de-risk + the named
fallback substrate + the parallelization recommendation.

## The shared boundary + the exact unsupplied quantity
Three gaps fail at the same point; the decisive evidence is a WITHIN-PAIR IDENTITY of aggregate statistics:
- **#4 replay** (`2026-08-07-replay-consolidation-v7-step0-...NO-GO`): the catastrophic probe (false=1.000, wrong
  assembly wins) and the perfect probe (false=0.000) have BYTE-ADJACENT aggregate stats (seed414 pr_eff 15.36 vs
  15.64, gini .698 vs .692, top_conc 1.0/1.0, active .333/.333). The regime is already maximally one-winner; the
  failure is WHICH one-winner.
- **#3 source monitoring**: source-blind recall over a shared core reactivates every committed subset; single-g_comp
  attractor WTA amplifies whichever basin latches first, failing `all_dominant_correct` in ALL g_comp>0 + all 9
  ratio-grid cells (not a bad-ratio artifact).
- **v8 rate-homeostat**: a firing-rate set-point EQUALIZES correct-vs-rival and pulls every margin to the middle. Its
  own theorem: "the margin and the activity set-point are ORTHOGONAL quantities; a homeostat that defends a firing
  rate cannot, in principle, defend a between-source contrast."
⇒ **The unsupplied quantity is the categorical IDENTITY of the winning attractor — a nominal label, not a magnitude.**
Every lever tried (sparsity, participation ratio, gini, top-share, E/I, rate set-point, symmetric-GABA gain)
regulates a LEVEL, and levels are provably identical between the catastrophic and perfect cases.

## The DG separation-vs-reliability tradeoff is NOT hard on this substrate (self-retracted) — the escape is UNTESTED
`2026-05-31-DG-...-FUNDAMENTAL-BOUNDARY` carries a ⚠️ REFINEMENT banner: the raw concept activity is ALREADY
ID-separable (within 0.896 > between 0.768, NN identity 16/16, no DG); the unmet bar was near-ORTHOGONALITY for clean
VSA binding — a HIGHER bar the identity-DISCRIMINATION task does not need. The within-concept collapse (0.2–0.3) was a
SPIKING-DYNAMICS artifact (stochastic near-threshold k-WTA flips winners; deterministic top-k is stable). So the
tradeoff is not a substrate law. **Un-tried escape: separation + completion at DIFFERENT LOCI, at proper CA3 SCALE.**
`2026-07-14-ca3-competitive-hebbian-formation-6seed-GO` proved competitive-Hebbian formation builds a selective
attractor (8.9× vs pure-LTP 1.01×, 6/6 GO) and ROOT-CAUSED the residual completion failure as SCALE-bounded (needs
~1000–2000 CA3 for robust <1% redundant assemblies; at 150 it's a knife-edge), RULING OUT inhibition (g∈{.7,.5,.3,0}
all held-out 0). Redundancy defeats the near-threshold winner-flip = reliability, while staying sparse = separation.
⛔ **CORRECTS the 2026-08-07 scale probe on the board:** that probe scaled the FEEDFORWARD readout (n+episode {12,24,36},
no attractor recurrence) and found no gain — but the decisive test is ATTRACTOR COMPETITION × storage-separation ×
n≥48, which is UNTESTED. NB scale threads the within-concept RELIABILITY half (#4 + #3-completion); it does NOT alone
solve #3 CO-RESIDENCY (shared core by construction) — that needs the storage-side separation knob jointly.

## THE ONE DECISIVE DE-RISK (cheapest; #3, numpy, minutes/seed)
Extend `research/runners/_laneC_source_monitor_attractor_competition.py` with the scoping doc's JOINT knob: co-tune
`g_comp` (fixed recurrent-E : lateral-I ratio) × a storage-side separation knob (larger unique fraction / sparser
core) AND raise `n_source` 12→≥48 (the CA3-GO load-bearing variable; the joint-uniq NO-GO literally reported "weakest
pinned at n12"). Decisive anti-cheat already exists: **`all_dominant_correct` on EVERY source incl. self_generated**
(a hard WTA silencing 2 pools regardless of correctness trivially maxes margin — caught here) + honesty guards
(source-afferent current=0 AND firing=0 at recall; competition param-symmetric; g_comp=0 byte-identical null).
GO (frozen v6): `min M > min L` AND `min M ≥ 0.15` AND all_dominant_correct, calib 650/651 → dev 652/653/654 → sealed
655/656/657. A GO unblocks #3 AND validates the parallel #4 test (competitive-Hebbian CA3 completion at n_ca3~1000–2000,
the CA3-GO's own pinned next). A NO-GO with all_dominant_correct=False even at scale = the decisive boundary verdict.

## Boundary verdict IF the joint-scale de-risk fails → the named next substrate
On a POINT-NEURON RATE substrate, identity of same-core co-resident sources is not recoverable by ANY within-substrate
competition: the only distinguishing signal (unique-cell subset) sums LINEARLY with the shared core into ONE soma
rate, and every downstream op on that rate is an aggregate that already discarded the label. Proof-by-contrast: the
`2026-08-01-agency-authorship-tag-corollary-discharge` 1-bit self/other source GO works precisely because efference
copy is a DISTINCT PHYSICAL CARRIER orthogonal to content. Co-residency has NO natural orthogonal carrier; injecting
one (phase-slot→source) is a LABEL LEAK — theta-gamma is disqualified twice (honesty + the 2026-05-20 5-architecture
spiking ceiling; phase multiplexing HOLDS items apart but can't say WHICH slot is the cued source without the pre-known
map). **Required different substrate = DENDRITIC COMPARTMENTS:** unique-source afferents cluster on one branch, shared
core on another; only the branch with COINCIDENT core+unique input crosses its NMDA-PLATEAU nonlinearity → the soma is
plateau-amplified ONLY when that cell's own source is cued (an identity-specific nonlinear AND). Discrimination lives
in WHICH BRANCH plateaued (a subunit event re-created fresh by input geometry each recall) — no soma homeostat can
equalize it (it is not a stored rate-level). Threads separation (branch) + completion/reliability (plateau latch) at
DIFFERENT LOCI = the DG requirement a point neuron can't provide. Primitives exist (two_comp/apical_R; D2 divisive-gain
`69e217bb4`; the BCM branch-sculpting rule, `2026-06-19-dendritic-binding-derisk-scoping`). ⛔ HONESTY CONDITION: the
branch assignment of unique-source afferents must SELF-ORGANIZE via the BCM rule, NEVER be host-wired per source
(host-assigning branch=source is the same label leak). A factorized/VSA identity code is the alternative but re-imports
the composer host-algebra shortcut — dendritic compartments are the honest next-substrate direction.

## Recommendation (grounded)
Run the ONE cheap joint-scale de-risk (§ above) next — decisive EITHER way — but do NOT open the dendritic (or any
months-long cluster) build before it reports. This cluster has consumed 9 source-monitor + 7 replay versions + the DG
arc on the same wall (far past the `before_you_build` ≥2-lever gate); the DEEPEST-LESSON reframe points at the ONE
variable never properly tested — SCALE in the attractor+separation config. **The identity boundary blocks a
SUB-CAPABILITY, not the roadmap** — affect, curiosity, perception, language are NOT blocked (the same window produced
the corollary-discharge source GO, 3 Phase-0 self-model GOs, place-field/reward wins). Per the compute-lane discipline,
those CPU-cheap faculty lanes should advance IN PARALLEL while the single GPU-free joint-scale de-risk runs — do NOT
serialize another monoculture session against this wall before the scale hypothesis is tested.
