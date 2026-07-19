export const meta = {
  name: 'sub1pct-margin-readout-boundary-gate',
  description: 'Focused SURPASS research gate on the multiply-confirmed sub-1pct-margin spiking-read-out boundary: isolate residual, reframe via biology, rank cheap-first mechanisms, verdict build-vs-frontier',
  phases: [
    { title: 'Research', detail: 'isolate residual + biology reframe + project-family + cheap-first mechanisms' },
    { title: 'Verdict', detail: 'judge: the one mechanism to build, or the honest frontier' },
  ],
}

const REPO = 'C:/Users/dant123/Documents/Projects/neural-simulator'

const CONTEXT = [
  'THE BOUNDARY (multiply-confirmed this session). On-substrate role read-out: reproduce argmax_r of (f dot Ws)[r] as',
  'SPIKING neurons (NO host f@Ws matmul, NO host argmax; a NEURAL argmax over on-bridge ensemble FIRING is allowed). Ws',
  '(n_res+1 x 3) is a ridge read-out on a fixed-random SPIKING reservoir rate feature f; the +1 row is a per-role bias. The',
  'committed close-out (B-1c.2) uses the POSITIVE Dale-shifted read Ws_shifted = Ws - Ws.min() (purely EXCITATORY',
  'reservoir->ensemble synapses; winner = argmax over ensemble summed FIRING). GO 2/3 seeds: 42=18/18, 43=18/18, degraded',
  'draw 44=11/18.',
  '',
  'ROOT CAUSE of seed 44: the Dale OFFSET raises all 3 ensembles by a uniform pedestal ~= |min(Ws)|*sum(f) that DWARFS the',
  'winning margin, so the margin is only ~1pct of the ensemble drive (the finding own "sub-1pct post-offset margin"). A',
  'point-neuron spike-COUNT read cannot reliably resolve a sub-1pct margin, worsened by the non-monotone Izhikevich f-I',
  '(depolarization block at high drive: at floor 150, seed-44 slot2 HIGHEST-drive role fires the FEWEST spikes -> argmax inverts).',
  '',
  'EVERYTHING TRIED THIS SESSION (do NOT re-propose unless with a concrete fix to the named failure):',
  ' READ-OUT surface: (a) signed spike-COUNT opponent ens_pos-ens_neg = 44:18 but 42/43:0 (subtracting COUNTS is nonlinear',
  '   under the non-monotone f-I). (b) conductance-domain SIGNED (Wp on g_e, Wn on g_i of one ensemble via an inhibitory',
  '   reservoir follower + a synaptic bias unit) LOOKED 18/18 on 42/43/44 but the ANTI-CHEAT proved the follower+bias were',
  '   DECORATIVE (BARE = Wp exc rows only = 18/18) AND the positive read that carried it was OVERFIT (fair 6-seed test:',
  '   unseen 100/101/102 = 0/6/5 out of 18). (c) divisive normalization (Carandini-Heeger, input_divisive_norm) plateaus',
  '   ~9/18 (compresses the sub-1pct margin below resolution). (d) per-ensemble gain calibration, more integration (replay',
  '   14), rank-order first-spike latency, low-floor operating point: all overfit-or-plateaued, NONE generalized to 100/101/102.',
  ' FEATURE surface: (e) reservoir conditioning RES_N 300->600 = seed-44 8/18 (no lift). (f) on-substrate reservoir COMMITTEE',
  '   (M=3 independent reservoirs, concatenated features) = seed-44 10/18 (no lift).',
  '',
  'CRITICAL LESSON: any read-out TUNED on the 3 seeds 42/43/44 OVERFITS -- it does NOT generalize to unseen 100/101/102. The',
  'bar is a GENERALIZING mechanism validated on 6 seeds (42/43/44 + 100/101/102) at a FIXED principled operating point, NOT',
  'tuned per-subset.',
  '',
  'SUBSTRATE: Izhikevich RS/FS point neurons on a UnifiedBrainBridge (sim/bridge.py, sim/kernels.py). Conductance synapses',
  'I_syn = g_e*(0-v) + g_i*(-75-v). set_pathway_weights wires arbitrary synapses runner-side; cp_traits=1 = inhibitory. The',
  'bridge has: input_divisive_norm, slow-NMDA recurrent conductance (tau~100ms), GABA_B slow-K+, and RESONATE-AND-FIRE (RF)',
  'complex-PHASOR neurons (NeuronModel.RESONATE_AND_FIRE, phase-coded, used by the FHRR composer). The project has DOCUMENTED',
  'boundaries in the "graded-magnitude / rate-code-wall / point-neuron-limit" family.',
].join('\n')

phase('Research')

const [isolate, biology, family, cheap] = await parallel([
  () => agent('You are a spiking-circuit theorist.\n' + CONTEXT + '\n\n' +
    'YOUR MOVE -- ISOLATE + QUANTIFY the GENUINE residual. Is the sub-1pct margin FUNDAMENTAL or an ARTIFACT of the Dale-offset ' +
    'excitatory-only encoding? Read ' + REPO + '/research/runners/_rungB1c_spiking_reservoir_synaptic_readout_derisk.py (the ' +
    'Ws_shifted read + the sub-1pct-margin note ~lines 86-119) and ' + REPO + '/research/findings/2026-07-04-conductance-domain-signed-readout-SURPASS.md. ' +
    'KEY QUESTIONS: (1) The global Dale offset Ws - Ws.min() adds a uniform pedestal that dwarfs the margin. Is there an ' +
    'argmax-PRESERVING, Dale-legal (excitatory-only) encoding with a WIDER margin -- e.g. a per-COLUMN shift ' +
    'Ws[:,r]-Ws[:,r].min() (smaller pedestal), a rank/whitening transform of Ws that spreads the roles, or dropping low-weight ' +
    'rows so the surviving drive is more differentiated? (2) Is the margin sub-1pct for ALL seeds or only 44 -- is the residual ' +
    'the ENCODING (fixable by a better excitatory transform) or the seed-44 FEATURE (the reservoir genuinely under-separates ' +
    'the patient slot)? (3) Quantify: how wide must the margin be for a P=80/T=30 spike read to resolve it, and how far is the ' +
    'pedestal from that? Give a CONCRETE argmax-preserving Dale-legal margin-widening transform (exact formula) + predict ' +
    'whether it generalizes. Be quantitative.', { label: 'isolate:margin-vs-feature', phase: 'Research' }),

  () => agent('You are a systems neuroscientist.\n' + CONTEXT + '\n\n' +
    'YOUR MOVE -- REFRAME via how BIOLOGY reliably reads a TIGHT-MARGIN winner from a noisy spiking population (superior ' +
    'colliculus saccade target, LIP/MT decision, striatal action selection). Use WebSearch + grep ' + REPO + '/research/findings ' +
    'and ' + REPO + '/references for "winner-take-all", "biased competition", "line attractor", "ramping accumulator", ' +
    '"recurrent amplification", "normalization". RANK for THIS problem: (a) RECURRENT AMPLIFICATION / line-attractor WTA that ' +
    'AMPLIFIES a tiny input margin into a decisive firing difference over time (Wang-2002 attractor; the project own ' +
    'Wang-2002/Lo-Wang commit-burst used in navigation) -- does iterating a mutual-inhibition WTA turn a 1pct input margin into ' +
    'a clean winner, argmax-preserving + seed-robust? (b) divisive normalization done RIGHT (it plateaued -- applied wrong?). ' +
    '(c) TEMPORAL/first-spike-latency (Thorpe), intensity-invariant, immune to count-inversion. (d) ramping ACCUMULATOR to ' +
    'threshold. For the TOP mechanism: biology, citation, WHY it resolves a sub-1pct margin where a raw count read cannot, and ' +
    'whether it needs the ensembles OUT of depolarization block first.', { label: 'biology:tight-margin-winner', phase: 'Research' }),

  () => agent('You are the project historian.\n' + CONTEXT + '\n\n' +
    'YOUR MOVE -- is this the SAME FAMILY as a prior DOCUMENTED boundary, and what did the project conclude the missing ' +
    'mechanism is? Grep + read across ' + REPO + '/research/findings and ' + REPO + '/CLAUDE.md for: "rate-code wall", ' +
    '"graded-magnitude", "point-neuron limit", "Mikulasch-Priesemann", "whitening", "population code", "graded read", "sub-1". ' +
    'For each prior boundary in this family: what was the wall + how did the project SURPASS or defer it? Especially: (1) the ' +
    '"population code lifts the single-neuron read-out from 47pct to 100-108pct" result -- does reading the margin from a ' +
    'REDUNDANT POPULATION (many neurons per role, averaged) resolve a sub-1pct margin, and was it ever applied to THIS read-out? ' +
    '(2) the resonate-and-fire (RF) PHASOR substrate -- the composer moved to phase-coding to escape a rate-coded SNR wall; ' +
    'could the role LOGIT be read as a PHASE (graded, not a sub-1pct count difference)? (3) honest verdict: does this need the ' +
    'deferred DENDRITIC/graded substrate, or has the project already found a point-neuron escape (population/phasor/attractor)? ' +
    'Give the precedent + whether it transfers.', { label: 'family:prior-boundaries', phase: 'Research' }),

  () => agent('You are a pragmatic research engineer.\n' + CONTEXT + '\n\n' +
    'YOUR MOVE -- rank CHEAPEST-first UNTRIED mechanisms most likely to give a GENERALIZING 6-seed resolution (validated on ' +
    '42/43/44 AND unseen 100/101/102, FIXED operating point, NO per-subset tuning). Everything in EVERYTHING TRIED is OUT ' +
    'unless you name a concrete fix to its failure. Evaluate + rank: (1) POPULATION-REDUNDANCY read -- enlarge the ENSEMBLES ' +
    '(P=80 -> 240+) with the read = population mean, averaging the sub-1pct margin over more units (RES_N 600 + committee ' +
    'enlarged the RESERVOIR not the ENSEMBLES -- is enlarging the ensembles different?). (2) RECURRENT WTA AMPLIFICATION -- ' +
    'feed the near-tied ensemble drive into an iterated mutual-inhibition attractor (wire_wta_c2 + a Wang-2002 accumulator ' +
    'exist) that amplifies the winner BEFORE the neural argmax. (3) per-COLUMN Dale shift (smaller pedestal, wider margin -- ' +
    'cheap weight change). (4) PHASOR (resonate-and-fire) read-out (encode the logit as a PHASE). (5) NEF weighted SYNAPTIC ' +
    'decode (learned linear decode of ens firing -> role as fixed synapses onto 3 read neurons + WTA; fit per-seed like Ws). ' +
    'For your TOP 2: exact on-substrate recipe (wiring + operating point), why it GENERALIZES not overfits, and the 6-seed + ' +
    'anti-cheat validation. Concrete + laptop-CPU-runnable.', { label: 'cheap:untried-generalizing', phase: 'Research' }),
])

phase('Verdict')
const verdict = await agent('You are the deciding architect. A multiply-confirmed spiking-read-out boundary (sub-1pct margin ' +
  'across seeds) needs a VERDICT: surpassable-and-how-cheaply, or genuinely the graded/dendritic frontier.\n' + CONTEXT +
  '\n\n## ISOLATE:\n' + isolate + '\n\n## BIOLOGY:\n' + biology + '\n\n## FAMILY:\n' + family + '\n\n## CHEAP-FIRST:\n' + cheap +
  '\n\nDELIVER: (1) the SINGLE cheapest mechanism most likely to give a GENERALIZING 6-seed resolution (or a short ordered ' +
  'list if composed), with the exact runnable recipe + the 6-seed + anti-cheat validation plan; OR (2) if no point-neuron ' +
  'mechanism resolves a sub-1pct margin robustly and it IS the graded/dendritic frontier, say so DECISIVELY with evidence -- ' +
  'a well-mapped honest boundary is a valid deliverable. Explicitly flag any option that would OVERFIT (tuned on a subset) or ' +
  'reintroduce a host shortcut. Be decisive: I will implement your #1 immediately and validate on ALL 6 seeds before believing it.',
  { label: 'verdict:build-or-frontier', phase: 'Verdict' })

return { isolate, biology, family, cheap, verdict }
