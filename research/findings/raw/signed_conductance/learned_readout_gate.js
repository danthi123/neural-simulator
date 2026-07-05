export const meta = {
  name: 'biological-learned-readout-gate',
  description: 'Research gate: biologize the reservoir->role read-out. The Ws matrix is currently a HOST ridge fit (a non-biological shortcut) whose spiking realization is seed-fragile. Ground a biological LEARNED read-out (spiking plasticity) that generalizes across draws, against the projects own learned-read-out track record + biology.',
  phases: [
    { title: 'Research', detail: 'project track-record + biology of reservoir read-out learning + cheap-first build' },
    { title: 'Verdict', detail: 'the one learning rule to build + 6-seed validation plan' },
  ],
}

const REPO = 'C:/Users/dant123/Documents/Projects/neural-simulator'

const CTX = [
  'REFRAME (owner directive: everything BIOLOGICAL, no non-biological shortcuts, artificial life that learns+grows). The',
  'on-substrate role read-out reproduces argmax_r of (f dot Ws)[r] where f is a fixed-random SPIKING reservoir rate feature',
  'and Ws (n_res+1 x 3) is currently a HOST RIDGE FIT (np.linalg.solve in _fit_Ws_spiking). The B-1c arc removed the READ',
  'STEP shortcut (the host f@Ws matmul -> excitatory synapses; the host argmax -> a neural argmax over ensemble firing), BUT',
  'the read-out WEIGHTS Ws are still HOST-LEARNED (ridge solve) -- a residual non-biological shortcut.',
  '',
  'THE MEASURED PROBLEM (this session, rigorously): the host-fit Ws delivered as synapses is SEED-FRAGILE -- its spiking',
  'realization reproduces the host argmax reliably only on the 2-3 DEVELOPMENT reservoir draws (seeds 42/43) and FAILS on',
  'unseen draws (44:11/18, 100/101/102: 7/9/5 out of 18, near chance). Root cause (measurement-grounded, a research gate',
  'verified it): NOT a sub-1pct margin, NOT a degraded feature, NOT the dendritic frontier (DRIVE-WRONG=0/18 -- the drive',
  'delivers the right winner on every slot; isolated ens f-I MONOTONE to 450 pA). The failure is a WTA ignition-order',
  'inversion that a FIXED read-out circuit hits differently on different draws. ~20 fixed-circuit read mechanisms',
  '(signed/conductance/divnorm/latency/feedforward/committee) all OVERFIT the dev seeds or broke the unseen ones.',
  '',
  'THE HYPOTHESIS to evaluate: a read-out LEARNED ON THE SPIKING SUBSTRATE (via biological plasticity -- three-factor',
  'reward-STDP / Hebbian / delta-rule) instead of host-ridge-fit would (a) remove the residual host shortcut AND (b)',
  'generalize across draws BY CONSTRUCTION -- because the learning accounts for the spiking dynamics (it learns to make the',
  'CORRECT-ROLE ensemble FIRE MOST on THIS reservoir, not to reproduce a host matmul). This is the projects own learned-cortex',
  'thesis ("a real cortex has LEARNED, lossy, redundant read-outs that learn to read whatever messy code arrives").',
  '',
  'SUBSTRATE + PROJECT MACHINERY: Izhikevich RS/FS point neurons; conductance synapses; set_pathway_weights wires arbitrary',
  'synapses; reward-modulated STDP + Hebbian + eligibility traces + the neuromodulator subsystem exist. The project has a',
  'STRONG learned-read-out track record: bio_three_factor (Tier-1 word->motor 6/6 multi-seed), the concept-pool bidirectional',
  'binding (v14/v16), the on-bridge Hebbian co-occurrence STREAM cortex (corr(M,C) +0.686), the learned binder single-attr GO,',
  'the population-code read-out lift (47pct -> 100-108pct). BUT also a documented FAILURE: "global scalar feedback fails at',
  'biological scale" (W->A: sign-only DA 1/6, magnitude-DA 0/6; only per-region GRADIENT/supervised worked 3/3).',
].join('\n')

phase('Research')

const [track, biology, build] = await parallel([
  () => agent('You are the project historian of LEARNED read-outs.\n' + CTX + '\n\n' +
    'YOUR MOVE: catalog the projects OWN learned-read-out mechanisms and decide which applies to a reservoir->3-role classifier ' +
    'read-out, and WHY each generalized or failed. Grep + read across ' + REPO + '/research/findings and ' + REPO + '/CLAUDE.md ' +
    'for: bio_three_factor (three-factor reward-STDP, Tier-1/Tier-2.1 6/6), concept-pool binding (v14/v16, topographic prior + ' +
    'FS WTA), the on-bridge Hebbian CO-OCCURRENCE stream cortex, the learned binder, the population-code lift, AND the FAILURE ' +
    '"global scalar feedback fails at biological scale" (why sign-only/magnitude DA failed but per-region GRADIENT/supervised ' +
    'worked 3/3). KEY: the reservoir->role read-out is a SUPERVISED linear classifier on a fixed feature. (1) Which project ' +
    'mechanism is the right fit -- a per-role SUPERVISED/three-factor rule with a PER-ROLE eligibility/target (which worked) ' +
    'vs a global scalar reward (which failed)? (2) Did any project learned read-out demonstrate GENERALIZATION across random ' +
    'draws/seeds (the exact property we need)? (3) What is the concrete learning setup (teacher current on the correct-role ' +
    'ensemble, reward-STDP on reservoir->ens, freeze schedule)? Cite the finding files.', { label: 'track:project-learned-readouts', phase: 'Research' }),

  () => agent('You are a computational neuroscientist / reservoir-computing theorist.\n' + CTX + '\n\n' +
    'YOUR MOVE: how does BIOLOGY learn a read-out from a reservoir/population that GENERALIZES, and why does learning ON the ' +
    'spiking substrate make it robust to the spiking nonlinearities (the WTA ignition-order inversion a fixed read-out hits)? ' +
    'Use WebSearch. Cover + rank: (a) the classic reservoir-computing read-out = a trained LINEAR/perceptron read-out (delta ' +
    'rule / LMS), and its spiking analogue (a per-output DELTA rule with a teacher / three-factor with per-output eligibility ' +
    '-- Legenstein-Maass, Sussillo-Abbott FORCE, Gilra-Gerstner FOLLOW); (b) cerebellar Marr-Albus-Ito: the granule layer IS a ' +
    'reservoir and the PF->Purkinje read-out is LEARNED via climbing-fiber-gated LTD -- a supervised per-output rule, the ' +
    'canonical biological reservoir read-out; (c) why an on-substrate learned read-out is robust WHERE a host-fit matrix is not ' +
    '(it learns the correct SPIKING winner directly, so the f-I nonlinearity / ignition order is IN the training loop, not an ' +
    'unmodeled downstream surprise). For the TOP mechanism: the exact spiking learning rule (pre x post x teacher/error), why ' +
    'it generalizes across draws, and the citation. Explicitly say whether it needs a GLOBAL scalar signal (which the project ' +
    'showed FAILS) or a PER-ROLE local error/teacher (which should work).', { label: 'biology:reservoir-readout-learning', phase: 'Research' }),

  () => agent('You are a pragmatic research engineer.\n' + CTX + '\n\n' +
    'YOUR MOVE: design the CHEAPEST biological learned read-out to build + 6-seed validate, that GENERALIZES across draws ' +
    '(42/43/44 + unseen 100/101/102) and avoids the "global scalar feedback fails" trap. Read ' + REPO + ' + ' +
    '/research/runners/_rungB1c_spiking_reservoir_synaptic_readout_derisk.py (the reservoir + the host _fit_Ws_spiking to ' +
    'REPLACE) and ' + REPO + '/research/runners/bio_three_factor.py (the projects validated three-factor machinery). Propose: ' +
    'per training example (reservoir feature f for a sentence + its known role label per slot), drive the reservoir, apply a ' +
    'TEACHER current to the correct-role ensemble (supervised target), and let a PER-ROLE three-factor/Hebbian rule shape the ' +
    'reservoir->ens synapses so the correct ensemble learns to FIRE MOST -- so the LEARNED synapses ARE the read-out (no host ' +
    'ridge solve, no host f@Ws). Address: (1) the exact rule (teacher-gated Hebbian on reservoir->ens? reward-STDP with a ' +
    'per-role reward?) and why it is per-role-local not global-scalar; (2) does it need the ensembles OUT of the WTA ' +
    'ignition-inversion regime during training, or does training IN that regime make it robust to it?; (3) the freeze schedule; ' +
    '(4) the 6-seed GO gate + anti-cheats (syn-lesion collapses; a SCRAMBLED-label control must fail to learn; the learned ' +
    'read-out must NOT be host-fit -- source-clean). Give a concrete, laptop-CPU-runnable recipe. Flag any hidden host shortcut.',
    { label: 'build:cheap-biological-learned', phase: 'Research' }),
])

phase('Verdict')
const verdict = await agent('You are the deciding architect. Biologize the reservoir->role read-out with a LEARNED (spiking ' +
  'plasticity) read-out that generalizes across draws, removing the host-ridge-fit shortcut.\n' + CTX +
  '\n\n## PROJECT TRACK RECORD:\n' + track + '\n\n## BIOLOGY:\n' + biology + '\n\n## CHEAP BUILD:\n' + build +
  '\n\nDELIVER a DECISIVE, ordered build plan: the SINGLE biological learning rule to implement first (exact spiking rule, ' +
  'teacher/error signal, per-role-local not global-scalar, freeze schedule), WHY it will generalize across draws where the ' +
  'host-fit Ws did not, the 6-seed GO gate (>=17/18 on ALL of 42/43/44 + 100/101/102 at a FIXED protocol, no per-subset tune), ' +
  'and the anti-cheats (syn-lesion collapse; scrambled-label control fails to learn; source-clean = no host ridge/argmax). ' +
  'Flag any hidden host shortcut or overfit risk. Be concrete: I will implement your #1 and 6-seed-validate it.',
  { label: 'verdict:learned-readout-build', phase: 'Verdict' })

return { track, biology, build, verdict }
