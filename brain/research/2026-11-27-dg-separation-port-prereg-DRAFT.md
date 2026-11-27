# Pre-registration (DRAFT): port DG-CA3 pattern separation into the new core (memory-selection wall)

> **STATUS 11-27: NOT RUN (Fable design review: DO-NOT-RUN).** Superseded by a forward-only ceiling measurement (see the 11-27 ceiling preregs). Verdict on this design only, not on the capability.

Written 2026-11-27, DRAFT: no data has been read and this has not had its before-data review (do that, sonnet, then
land the bars). Capability: Memory selection (pattern completion and separation). Code: `brain/dg_separation.py`
(switch `Brain(key="dg")`, default OFF), harness `brain/experiments/dg_screen.py`, tests `brain/tests/test_dg_separation.py`.
Fetch level, offline. A PASS here earns a live DEV life, never a claim.

## 1. Record brief (what is known, and what is NOT portable)
- **The wall.** On the K-P slept 1.5B life the brain fetches the wrong told episode on 136 of 217 rows (right: 44).
  `PREREQUISITES.md` lists "separation of near-duplicate told episodes" as missing; the routing arc stopped at its timebox.
- **Rank is not the whole limit.** Ceiling (`results/2026-11-25-routing-ceiling-read.md`): S_all_live ranks the right
  episode first on 168/217 (S0 90). Screen 2 (`results/2026-11-25-routing-screen2-read.md`): that key through the live
  competition fetches right on only 49 (no-fetch 128): the right episode's WEIGHTED score does not clear the bar. So the
  defect is the score's shape and scale against a bar and a strength weight, not only which episode is first.
- **DG(S0) is already dead** (ceiling, K-5 rule): a fixed random expansion + k-WTA on the layer-0 key gains -45..+29 rows
  with the map seed. Cause named in the record, not tested: a fixed expansion keeps rank order to first order (SimHash)
  and does not decorrelate dense correlated codes (`research/findings/2026-06-11-cortex-fixed-expansion-decorrelation-NEGATIVE.md`).
  This prereg therefore does NOT re-run DG(S0). It puts the DG on the key that already ranks well (the all-layer change).
- **Old-track assets, read in full:**
  - `2026-08-19-replay-pattern-separation-DG-separator-...-6seed-GO.md`: GO 6/6, but 2 memories (Jaccard 0.60) on a LIF
    net, the separator acting on a PLASTIC replay write (sparse-expansive recoding + PV-basket gain; lesion = blur returns;
    halves false recall, does not remove it; a reliability cliff when sparser). Not an index over 98 stored deltas.
  - `2026-09-05-onebrain-fact-shard-dg-ca3-sublinear-spiking-retrieval-derisk-GO.md` (the 6/6 GO that flipped the shard
    default) and `research/biology/dg-ca3-sparse-index.md`: a SPEED mechanism (bucket routing to a shard so the cleanup is
    O(shard)); the project's own record says "built for speed, not separation" (routing cascade note 11-24). Its
    retrieval accuracy equalled the full scan (parity), by design; it never beat the scan.
  - `2026-05-31-DG-composition-NULL-...` and `2026-06-10-cortex-DG-CA3-cleanup-NEGATIVE.md`: the SPIKING DG read is not a
    reproducible code (re-reading the same input gives cosine 0.04-0.15); separation was necessary, not sufficient;
    reliability needs CA3 completion. Root cause there was noise, which a deterministic host-rate code does not have.
- **Portability, honestly.** Ported: the expansion (fixed sparse random fan-in, many more units than inputs), k-WTA
  sparsification, overlap read-out, optional CA3 completion as a softmax (modern-Hopfield) step over the stored codes.
  NOT ported: spiking dynamics, PV-basket inhibition as a circuit, Hebbian replay write, neurogenesis, any learning.
  This is a **code substitute** for DG -> CA3, to be listed in `brain/SCAFFOLDS.md` if it is ever adopted (no row added
  by this draft). The mechanism that made the 08-19 GO work (a plastic write that orthogonalises replay) has no analogue
  in a read-only key, so this port is a different and weaker claim: "does a sparse code reshape the score so the bar and
  margin can separate told from untold and right from wrong", not "separation was restored".
- **Prior (stated before data).** Expected-fail, partly. Rank of DG(S_all) ~= rank of S_all (~168/217). The live question is
  whether the overlap/k score has a chance floor (f) and a wider top-1 margin that make a bar fit by a told-blind rule
  fetch more right episodes than the dense cosine does under the same rule. I expect D9 (no gain over the refit S_all) to
  be the likely kill. A negative here is a verdict on this fixed read-only DG, not on separation: the next methods are
  recorded in section 8.

## 2. Mechanism (as built)
Input: the episode's per-layer matrix-memory change (the S_all read-out; the plain `route_delta` where one is kept). Each
layer block is L2-normalised, projected through a fixed sparse random map (E = 8192 granule units, each summing 64 input
dims with random +-1; one map per layer, seeded), the layer projections are summed (/sqrt(layers)): a linear map of the
S_all concatenation. Because it is linear, the store is projected once per episode (cached, not saved) and centred on the
store mean AFTER projection (feedforward inhibition against the mean trace). Code = top k = f*E units set to 1.
Similarity of cue to episode = overlap / k (chance floor ~ f). Optional CA3 completion: `a = softmax(beta * overlap)`,
cue code <- k-WTA(a @ stored codes), two steps, similarity = overlap of stored codes with the completed code; beta = 50 is
set a priori (overlap spans ~0-0.5), not fitted. Selection afterwards is the default Hippocampus competition (strength
weight, bar, margin 0.02). Storage: no key per episode (as S_all); a cache of E floats per episode in RAM.
`Brain(key="dg")` raises unless `Brain.DG_BAR = (familiarity, margin)` is set: overlap/k has no shipped bar.

## 3. Arms (rows of the table; all read the same cue once from base)
| arm | what |
|---|---|
| A | K-P as is (current key S0 through the live competition) |
| K | S_all, registered slept bar 0.165 / 0.02 (screen 2's arm), reference |
| K refit | S_all with the bar fit by the section 5 rule, so DG is compared with a dense key under the same rule |
| DG:f | k-WTA code, map seed 42, f in {0.05 (registered), 0.02 (reported only, never picked)} |
| DGcc:f | DG plus CA3 completion |
| DGlesion | sparsity lesion: no k-WTA, the dense centred projection, cosine (same map) |
| DGshuf:f | shuffled-code control: every stored code's units permuted (same sparsity, no content) |
| DG:0.05:s | the registered cell on map seeds 43, 44, 100, 101, 102 (robustness) |

## 4. Population, instrument, cost
- Population: the routing screens' own. KR night snapshots (`life_kw --night-snapshots`) of s319/s320/s321, nights 5-15,
  the banked intact-frozen battery (`routing_replay.row_class`: told, fire, identity), void rules exactly as screens 1 and 2
  (a night is void when arm A fails to reproduce the banked fetch or right flag; more than 2 void nights voids a seed;
  s321 night 5 is expected void).
- Paths (pool nodes, `~/brain-lane/route_repo/_scratch/route/kr/`, shipped by `results/2026-11-25-routing-ceiling/ship.sh`
  from `_scratch/brainv2/aws/20261124-{1219,1624,1704}/work/lifeKR_s{319,320,321}/trunk/`):
  `lifeKR_s<S>.json.json` (--life), `snaps_s<S>/n<N>/` (--snaps), `trunk_s<S>/` (--trunk, episode deltas). Validity files:
  `brain/experiments/results/2026-11-25-routing-screen/valid_s{319,320,321}.json`. Seeds 319/320/321 are the KR lives
  (design seeds, SEEN; this is the 11-25 population, not held-out).
- Run (per node, CPU fp32, `tools/memcap.sh 11`, one seed per node, pool40/41/42; model 1.5B, `--device cpu`):
  `python -m brain.experiments.dg_screen run --life $C/lifeKR_s$S.json.json --snaps $C/snaps_s$S --trunk $C/trunk_s$S
  --model 1.5b --device cpu --out $C/dg_s$S.json --work $C/dgwork_s$S`, then locally
  `python -m brain.experiments.dg_screen score dg_s319.json dg_s320.json dg_s321.json --validity valid_s319.json valid_s320.json valid_s321.json --out verdict.json`.
  Same `run_node.sh` / `ship.sh` pattern as `results/2026-11-25-routing-screen2/` (not written until the draft is reviewed).
- Cost: $0, pool CPU. Estimate (NOT measured; nothing real has been run): a few minutes per night per seed, dominated by
  projecting ~100 episodes x 24 layers once per map seed (6 maps); under 1 h per seed. Memory as the ceiling run (store
  deltas held in RAM). The scorer is offline and instant: bars are applied to recorded fields, never at run time.

## 5. Bar rule and verdict rule (code-ready; implemented in `dg_screen.score`, constants at the top of that file)
**Bar rule (told-blind).** For each arm separately: the smallest question bar `b` on the grid 0.000..1.000 step 0.005 at
which the arm fires on the pooled fire rows (open, greet, never-told, untold near-duplicate; live seeds) no more often than
arm A does. Statement cues use `b * 0.45/0.12` (the class ratio); margin 0.02 on the weighted score, as every routing screen.
No told row is read to set any bar. (Seen seeds: the live DEV life must refit on its own fire rows.)

**Verdict rule.** Let `A_right` = A's right fetches on told rows of the live seeds, `need = max(ceil(1.25 * A_right),
A_right + 10)`. The registered cell is `DG:0.05` with its fitted bar. It is KILLED if any holds:
- D1: pooled right < need.
- D2: right rises on fewer than 2 of 3 seeds, or falls by more than 1 on any seed.
- D3: more than 1 row per seed turns from A's right episode into a wrong episode.
- D4: fires on fire rows exceed A's by more than 1 on any seed.
- D5: told no-fetch exceeds A's by more than 5 on any seed (the D lesson: abstention is not recall).
- D6: right < DGlesion right + 5 (sparsity is not load-bearing: the lesion, same map, dense cosine, same bar rule).
- D7: DGshuf right > 0.25 * DG right (the control must collapse; else the gain is not content).
- D8: fewer than 4 of the 6 map seeds clear D1 (map-seed unstable, as DG(S0) was).
- D9: right < K-refit right + 5 (DG adds nothing over its own input key under the same bar rule).
Otherwise **PICK (fetch level only)**. CA3 completion is carried only if `DGcc` beats `DG` by >= 5 right; else plain DG.
Reported, not barred: right@1 of the raw rank, told-vs-fire AUROC of the best weighted score per arm (does the code's
score separate told from untold cues, the calibration question), per-seed counts, K registered, DG:0.02, DGcc.

## 6. Reply level (not in this screen; stated so a fetch gain is not read as a reply gain)
A fetch gain need not move replies (S_all on the slept gate: fetches 1/18 -> 14/18, replies +4/36, lesion as good). If
PICK: (1) optional `routing_replay --reinstate`-style generation on the same snapshots with the DG decision as the recall
(1.5B CPU generation, slow; its own prereg), then (2) a live DEV K-P life, `Brain(key="dg")` with `DG_BAR` refit on that
life's fire rows, six seeds 42/43/44/100/101/102, bars: told recall on the strict scorer vs A, no-fetch counted as a miss,
false fires on untold, and a lesion (`DGlesion` or key S0) that changes the REPLY. None of that is registered here.

## 7. Seeds
Stage 1: s319/s320/s321 (the 11-25 population, 3 seeds, SEEN) x 6 map seeds for the registered cell. No generalisation
claim. Six seeds (42/43/44/100/101/102) only at the live stage, which needs its own prereg.

## 8. If KILLED (named, not built)
The fixed read-only DG is a code substitute that cannot learn to decorrelate. Candidates with a real mechanism: (a) a
LEARNED separator, a competitive (anti-Hebbian lateral inhibition) layer trained on the night's own episodes, closer to
the 08-19 GO's plastic write; (b) separation on the WRITE side: orthogonalise a new episode's stored change against its
nearest stored neighbours at encoding (the 08-19 mechanism, in `hippocampus.store`), which a read-only key cannot do;
(c) the strength / hub path (self/other tag, a reachable strength signal), which the routing arc banked as missing and
which no key change touches. Do not re-run a random-expansion DG on a different key.

## 9. Open risks
- The map seeds, E, fan-in, f and beta are a priori choices, not tuned; E x fan-in vs the 98k-dim blocks is a sparse
  sample (64 of 98304 inputs per unit per layer).
- Bar fit on pooled fire rows of the SAME seeds that are scored (the told rows are untouched): D4 is near-circular pooled,
  hence the per-seed check; it tells nothing about a new store.
- K arm runs through the same function the screen 2 used (`routing_screen2.decide_K`); its CPU fp32 cue read meets fp16
  deltas, so borderline rows are unvalidated (screen 2 caveat 1).
- Runtime of 6 map seeds is unmeasured.
- The shuffled control permutes units per stored code with one generator per call: deterministic, but a different
  permutation per episode, so overlap is chance by construction; D7 can only fail if the harness leaks content.
