export const meta = {
  name: 'mdgl-spiking-offdiagonal-verify',
  description: 'Exhaustively verify the clean-directional spiking off-diagonal MDGL (population averaging N=16): 6-seed compute + adversarial skeptics + synthesis',
  phases: [
    { title: 'Multi-seed', detail: '6 seeds run the full-arm N=16 sweep (BPTT ceiling, e-prop, MDGL gains, sign-flip, zero-Γ, permuted; fixed eval_n)' },
    { title: 'Adversarial', detail: '4 skeptics each probe a distinct confound of the aggregated result' },
    { title: 'Synthesize', detail: 'verdict: is the off-diagonal MDGL clean-directional on spikes with population averaging, multi-seed?' },
  ],
}

const SEEDS = [42, 43, 44, 100, 101, 102]
const RUN = (s) =>
  `cd /e/Documents/Projects/sim && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SIM_BACKEND=numpy PYTHONPATH=/e/Documents/Projects/sim ` +
  `python -u -m research.runners._mdgl_replica_popcoded_spiking_derisk --seeds ${s} --full --n-replica 16 --epochs 60 --eval-n 32 ` +
  `--out research/findings/raw/_mdgl_full_s${s}.json`

const SEED_SCHEMA = {
  type: 'object',
  properties: {
    seed: { type: 'integer' },
    bptt_ceiling: { type: 'number' }, eprop: { type: 'number' }, permuted: { type: 'number' }, chance: { type: 'number' },
    best_clean_gain: { type: ['number', 'null'] }, GO: { type: 'boolean' },
    gains: { type: 'string', description: 'compact per-gain string: gain -> mdgl/signflip/zeroG' },
    raw_line: { type: 'string', description: 'the verbatim [full sN ...] stdout line' },
  },
  required: ['seed', 'eprop', 'best_clean_gain', 'GO', 'raw_line'],
}

phase('Multi-seed')
const seedResults = await parallel(SEEDS.map((s) => () =>
  agent(
    `Run this EXACT command and wait for it to finish (it is a ~10-15 min numpy job; do NOT background it, do NOT modify it):\n\n${RUN(s)}\n\n` +
    `It prints ONE line starting "[full s${s} ...". Then read research/findings/raw/_mdgl_full_s${s}.json (a 1-element array). ` +
    `Return the seed's result: seed, bptt_ceiling, eprop, permuted, chance, best_clean_gain (null if none), GO, ` +
    `a compact 'gains' string (each gain -> mdgl/signflip/zeroG), and raw_line (the verbatim stdout line). ` +
    `Do NOT interpret or claim anything — just report the numbers faithfully.`,
    { label: `seed:${s}`, phase: 'Multi-seed', schema: SEED_SCHEMA }
  )
))

const ok = seedResults.filter(Boolean)
const agg = ok.map((r) => r.raw_line).join('\n')
const nGO = ok.filter((r) => r.GO).length
log(`Multi-seed done: ${nGO}/${ok.length} seeds GO. Aggregated lines:\n${agg}`)

phase('Adversarial')
const LENS = [
  { key: 'eval-denoising', q: `Is the off-diagonal MDGL>e-prop signal a READ-OUT-DENOISING artifact rather than a recurrent-learning effect? The runner fixes eval_n=32 EQUAL for every arm (train-N=16), so all arms are read out with the same ensemble denoising — verify from the numbers that MDGL's edge over e-prop is NOT explained by the ensemble read-out (which is identical across arms). If the eval_n-equal design does NOT actually neutralize it, say so.` },
  { key: 'gain-cherry-pick', q: `Is "best_clean_gain" CHERRY-PICKED per seed? A real directional effect should be clean at a CONSISTENT gain across seeds, not a different lucky gain each seed. From the per-gain 'gains' strings, determine whether ANY SINGLE gain is clean-directional (MDGL>e-prop AND sign-flip<=e-prop AND zeroG~=e-prop) across a MAJORITY of seeds. If the clean gain jitters seed-to-seed, the effect is fragile/cherry-picked.` },
  { key: 'margin-vs-noise', q: `Are the MDGL>e-prop margins and the sign-flip collapses LARGER than the seed-to-seed NOISE? Compare the MDGL-eprop margin and the (e-prop - sign-flip) collapse at the clean gain to the spread of e-prop across seeds. If the margins are within the noise band, the "clean directional" calls are not robust.` },
  { key: 'ceiling-gap', q: `Does the off-diagonal actually MATTER — i.e. is there a real gap for it to close? Check: (a) BPTT ceiling >> e-prop (the task genuinely needs more than the diagonal), (b) permuted ~= chance (the pipeline isn't leaking labels), (c) e-prop is meaningfully below BPTT so MDGL's lift is closing a real gap, not noise near a saturated ceiling.` },
]
const verdicts = await parallel(LENS.map((L) => () =>
  agent(
    `You are an adversarial skeptic. Aggregated 6-seed results of the spiking off-diagonal MDGL de-risk (population averaging N=16, each line: chance / BPTT ceiling / e-prop / permuted / per-gain MDGL(M)/sign-flip(F)/zeroG(Z) / best_clean_gain / GO):\n\n${agg}\n\n` +
    `Your lens ONLY: ${L.q}\n\n` +
    `Return a verdict object: refuted (true if this lens REFUTES the clean-directional claim), confidence (0-1), and a 2-3 sentence reason grounded in the specific numbers. Default toward refuted if the numbers are ambiguous.`,
    { label: `verify:${L.key}`, phase: 'Adversarial',
      schema: { type: 'object', properties: { lens: { type: 'string' }, refuted: { type: 'boolean' }, confidence: { type: 'number' }, reason: { type: 'string' } }, required: ['refuted', 'reason'] } }
  ).then((v) => ({ ...v, lens: L.key }))
))

phase('Synthesize')
const vclean = verdicts.filter(Boolean)
const nRefuted = vclean.filter((v) => v.refuted).length
const synth = await agent(
  `Synthesize the final honest verdict on: "does the off-diagonal MDGL become a CLEAN, robust, multi-seed directional effect on spiking neurons when the credit is population-averaged (N=16 noisy replicas)?"\n\n` +
  `6-seed compute (${nGO}/${ok.length} seeds flagged GO by the runner):\n${agg}\n\n` +
  `Adversarial verdicts (${nRefuted}/${vclean.length} lenses refuted):\n` +
  vclean.map((v) => `- ${v.lens}: refuted=${v.refuted} (conf ${v.confidence ?? '?'}) — ${v.reason}`).join('\n') +
  `\n\nGive: (1) VERDICT — one of GO (clean+robust+multi-seed, survives all lenses) / PARTIAL (real but fragile/gain-sensitive/not-all-seeds) / NEGATIVE (a confound explains it or it doesn't hold). ` +
  `(2) The single most load-bearing reason. (3) The honest one-line implication for the owner's "what mechanism are we missing" question and whether the on-bridge realization is warranted. Be rigorous and concise; a PARTIAL or NEGATIVE is a first-class result.`,
  { label: 'synthesis', phase: 'Synthesize' }
)

return { nGO, nSeeds: ok.length, nRefuted, nLenses: vclean.length, seedResults: ok, verdicts: vclean, synthesis: synth }
