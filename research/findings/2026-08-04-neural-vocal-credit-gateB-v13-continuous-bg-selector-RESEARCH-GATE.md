---
type: research-gate
status: selected
date: 2026-08-04
mechanism: neural-vocal-action-credit-v13-continuous-basal-ganglia-selector
---

# Gate B v13: repair the continuous selector, not its boundary

## Decision

V13 will replace the trial-scoped Gate A selector with a continuously operating
basal-ganglia selector. The work starts below the behavioral circuit: GPi/SNr
output must become autonomously active from the first scored step without a
host-applied tonic current. Only after that substrate gate passes may V13 add
the missing center-surround pathways and test repeated actions without a host
winner break, reset pulse, state clearing, or hidden burn-in.

This is a correction to the inherited architecture. V10-V12 did not primarily
fail because they lacked one more cortical stopping population. They exposed
that Gate A's successful action was defined inside a host-controlled trial:

- Python stopped the action phase at the first motor threshold crossing;
- Python drove `selector_reset` at `1200 pA` for `35 ms`;
- Python then imposed a `100 ms` washout; and
- Python supplied `1000 pA` continuously to each GPi population.

When V10-V12 retained neural state and ran the full action window, the selector
eventually admitted both motor channels. V11 and V12 then added neural boundary
circuits downstream of a selector that was not itself continuous. Their
construction failures are evidence against that decomposition.

Reward, dopamine output, eligibility updates, policy learning, convention
reversal, and capability seeds remain closed. V13 first has to make an action a
well-bounded neural event.

## Functional role in the whole brain

The selector must transform competing, noisy cortical proposals into a
temporally bounded action while remaining continuously connected to perception,
memory, affect, and learning. It must provide all of these functions in one
uninterrupted brain:

1. **Resting suppression:** GPi/SNr output tonically inhibits motor thalamus or
   an equivalent motor target before any proposal wins.
2. **Early global hold:** a fast hyperdirect route briefly strengthens output
   inhibition when a proposal arrives, preventing premature movement while
   competition is unresolved.
3. **Focused release:** the selected direct pathway transiently suppresses its
   GPi/SNr channel, releasing only the corresponding motor route.
4. **Competitor suppression:** nonselected channels remain inhibited through
   diffuse STN excitation and indirect-pathway activity.
5. **Autonomous termination:** pallidal and indirect feedback end the selected
   action and return the circuit to tonic suppression without host reset.
6. **Local credit:** only the route that actually produced the bounded action
   retains eligibility for a later consequence.

Passing a one-action trial does not establish this role. The behavioral gate
must require repeated actions in one continuously evolving state and causal
losses under direct, indirect, hyperdirect, and tonic-output lesions.

## Evidence from the current implementation

### Gate A's positive result is trial-scoped

`research/runners/_vocal_action_selector_gate.py` runs at most `600` action
steps but breaks immediately when one motor population reaches `12` spikes.
It then injects the shared reset current and washout. The Gate A v2 finding
describes this accurately: the host observed the first motor threshold crossing
and ended the decision epoch. This proves that neural activity can choose a
channel under that protocol; it does not prove autonomous action duration,
termination, or readiness for the next action.

### Continuous execution exposed the missing role

- V10 produced both first actions and local policy eligibility, but the losing
  motor crossed later in every fixed `600 ms` action window.
- V11's recurrent corollary boundary activated without motor output.
- V12's guarded feed-forward boundary had correct inhibitory signs and
  autonomous late recovery, but both motors crossed during startup and CuPy
  admitted both channels in both action windows.

The repeated pattern is selector state escaping after the host's old stopping
point, not a lack of evidence that downstream inhibition exists. V12's matched
source-on/source-off twins showed that proposal and commit/motor stopping routes
reduced their targets on both backends. Adding another boundary around the same
trial-scoped selector would optimize the wrong layer.

### The named pacemakers are silent without host current

The selector uses `IZH2007_GPI_OUTPUT`, `IZH2007_GPE_PACEMAKER`, and
`IZH2007_STN_BURST`, but initializes each neuron exactly at its resting fixed
point. A 2026-08-04 implementation audit built the unchanged selector with seed
`314159`, set the complete external-current vector to zero, and ran `1000`
uninterrupted steps on NumPy and CuPy. GPi, GPe, STN, thalamus, commit, and motor
all emitted exactly zero spikes on both backends.

The April R3.8 commit `35f1908` does not close this gap. It retuned the separate
`HH_GPI_OUTPUT` preset's NaP, Ih, and M-current values, explicitly left the
Izhikevich runner unchanged, and supplied no zero-input validation. The bridge
also defers per-region HH overrides, so that homogeneous HH preset cannot be
placed only in GPi inside the current mixed-region Izhikevich selector.

The labels and comments therefore overstate the executable physiology. V13
must treat autonomous output as missing.

## Biological constraints

The evidence supports a coordinated basal-ganglia correction rather than a
generic cortical stop pulse.

- SNr and GPi output neurons fire tonically, including when fast synaptic
  transmission is blocked. Persistent sodium and tonic cation currents provide
  depolarizing drive; calcium-coupled SK currents shape the afterhyperpolarization
  and regularity. Ih contributes in some output neurons. The local catalog
  records this in A.04 from Deniau et al., PBR-160 chapter 9. Primary recordings
  also found spontaneous repetitive SNr firing in vitro: Nakanishi, Kita, and
  Kitai (1987), [PubMed](https://pubmed.ncbi.nlm.nih.gov/3427482/).
- The hyperdirect cortex-to-STN-to-GPi/SNr route is faster and broader than the
  focused direct route. It creates an early global hold followed by selected
  disinhibition and later suppression: Nambu, Tokuno, and Takada (2002),
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/12067746/).
- In a stop task, low-latency STN excitation competes with striatal inhibition
  at SNr, supporting a fast output brake rather than a host action boundary:
  Schmidt et al. (2013), [Nature Neuroscience](https://www.nature.com/articles/nn.3456).
- Arkypallidal GPe neurons provide a slower stop signal back to striatum. This
  complements, rather than replaces, the fast STN-SNr pause: Mallet et al.
  (2016), [Neuron/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC4871723/).
- GPe also inhibits GPi/SNr directly, with strong proximal and perisomatic
  influence. The current selector omits this route; local catalog A.04 records
  its anatomical importance.

The local reference catalog already marks cortex-to-STN, GPe-to-GPi/SNr, and
pallidostriatal feedback as missing or partial. PBR-160 chapter 8 describes the
three-phase sequence: hyperdirect early inhibition, focused direct release, and
indirect late inhibition. That sequence supplies initiation, competitor
suppression, and termination as one circuit role.

## Selected staged mechanism

### Stage 0: autonomous output substrate

Add a region-scoped, immutable intrinsic-excitability facility for Izhikevich
neurons. It must be represented and applied inside the spiking substrate, set
only during construction, and require no per-step host decision. The first
implementation may use a constant effective intrinsic current as the reduced
Izhikevich representation of unresolved NaP/NALCN-like pacemaker conductance.
That reduction must be entered in `docs/SCAFFOLD-LEDGER.md`; it is not equivalent
to having implemented the underlying ion channels.

The substrate gate must require:

1. zero external current to the tested GPi/SNr population for the complete run;
2. `40-80 Hz` population firing from the first scored `100 ms`, with no silent
   startup interval hidden by burn-in;
3. asynchronous population phases rather than one synchronized population
   volley;
4. tonic firing throughout a fixed uninterrupted interval;
5. a construction-time intrinsic-drive lesion that silences or strongly
   reduces the population;
6. ordinary GABA-A input that causally suppresses tonic firing and ordinary
   release that restores it without state clearing;
7. exact NumPy/CuPy agreement on gate classification and bounded numerical
   agreement on rates;
8. checkpoint round-trip preservation when the feature is active; and
9. byte-identical default-off firing, weights, and state, with no persistent
   array or per-step performance cost when no region requests the feature.

This stage establishes an executable phenotype, not biological closure. A
later replacement gate must compare explicit NaP/NALCN-like depolarization and
calcium-coupled SK recovery if selector behavior depends sensitively on the
constant-current reduction.

### Stage 1: continuous center-surround selector

After Stage 0 passes, construct a V13 selector from the Gate A v2 populations
but remove runtime use of `selector_reset` and GPi host tonic current. Add the
minimum missing pathways in mechanism order:

1. proposal/cortical command to shared STN for the fast hyperdirect hold;
2. GPe to same-channel GPi/SNr for direct pallidal control of output; and
3. a distinct arkypallidal or pallidostriatal feedback route only if the first
   two mechanisms produce clean initiation but fail autonomous termination.

The direct, indirect, and hyperdirect paths must remain symmetric before
learning. No pathway may be opened or closed after observing a winner. The
runner may present a shared sensory or practice condition at fixed times, but
the circuit must decide when an action begins and ends.

The construction gate must begin scoring at step zero and require:

- no motor threshold crossing during baseline;
- tonic GPi/SNr output and inhibited thalamus during baseline;
- an early STN and GPi/SNr increase after shared proposal onset;
- one focused GPi/SNr pause followed by one clean motor action;
- suppression of the competing motor for the complete fixed action window;
- autonomous return to tonic GPi/SNr output and motor silence;
- at least two clean actions from the same uninterrupted brain;
- immutable complete weights and zero reset current; and
- the same configuration passing on NumPy and CuPy before capability seeds
  open.

Required structural lesions are: intrinsic output drive, direct D1-to-GPi/SNr,
indirect D2-to-GPe, hyperdirect proposal-to-STN, GPe-to-GPi/SNr, and any later
arkypallidal route. Each lesion must have a directional prediction filed before
execution. A lesion that does not change the named phase means that pathway is
not load-bearing at the tested operating point.

### Stage 2: reopen local reward credit

Only a continuous-selector GO can reopen the V10 eligibility question. The
later learning preregistration must use full fixed action windows and require
clean selected-route eligibility before delivering reward. It must preserve
contingent versus reward-count-matched yoked controls, acquisition and
expression lesions, fresh multiseed development and held-out phases, and
same-brain convention reversal.

## Rejected next actions

- **Tune V12 weights or add another boundary population.** Four inhibitory
  twins already established the local signs. This would keep the trial-scoped
  selector and optimize around its escape.
- **Lengthen warmup or discard startup.** A continuously running brain has no
  privileged startup interval. Resting suppression must hold from the first
  scored step.
- **Restore Python stop-on-winner or reset.** These are the functions the
  biological selector must perform and cannot be used as implementation aids in
  a claimed host-free gate.
- **Switch the complete brain to homogeneous HH.** The current bridge cannot
  assign HH only to GPi/SNr, and making every region HH would change unrelated
  physiology and cost before the missing selector role is isolated.
- **Treat the April HH preset edit as closure.** It is neither integrated into
  this selector nor validated at zero input.
- **Begin reward learning now.** Eligibility from an action whose endpoint is
  host-defined or whose competitor later executes is not executed-action-local
  credit.
- **Add short-term facilitation first.** Facilitation may later support burst
  routing, but the simulator lacks pathway-specific facilitating parameters and
  the more proximal defects are already measured: silent output, missing
  hyperdirect input, and missing GPe-to-output inhibition.

## Performance and consumer-hardware constraints

The intrinsic feature must allocate no array when unused. With it active, one
float per affected neuron is the initial ceiling; the implementation should
fuse its addition into the existing total-current path. It must not add a host
loop or device synchronization. The preregistration must benchmark unchanged
and active substrate paths on the RTX 3090 and report persistent memory, median
step time, and backend.

The center-surround selector should reuse existing regions and sparse regional
pathways. Any larger population or slow conductance needs a measured causal
benefit and a separate performance comparison. The target remains a selector
that can be replicated across speech and other action domains on consumer
hardware.

## Next exact action

Before simulator edits, file a Stage 0 preregistration that locks the reduced
intrinsic-drive representation, initialization, phase distribution, exact
physiology and lesion protocols, fresh construction seeds, backend order,
checkpoint and equivalence tests, performance ceiling, and stop rules. Then
implement only that default-off substrate facility and its focused tests.

Do not assign V13 behavioral or reward-learning seeds until Stage 0 has a
recorded cross-backend verdict and the whole-brain priority is reassessed.

