"""FHRR-on-bridge layer (b): a PARALLEL RF phasor composer running the conversational composition on the bridge's
resonate-and-fire neurons + complex synapses -- so the opponency (the rate-coded composer's SNR wall) is GONE (the
phasor algebra has no common mode). Same conversational API as core_sim_composition.CoreSimComposer; validated at
parity before the BrainConversationalAgent switches (layer c). Design:
docs/plans/2026-06-05-fhrr-layer-b-composer-recode-design.md.

Reuse-by-import the RF + complex-synapse substrate already on the bridge (NeuronModel.RESONATE_AND_FIRE +
rf_kick / rf_read_phases / rf_set_complex_weights, layers RF-on-bridge + layer-a). NO sim/ edits here.

Representation: each concept/role is a PHASOR vector (phases in [0,1)^D, deterministic per seed). bind = role (x)
filler via a DIAGONAL complex synapse (weight = the role phasor); bundle = unit complex synapses (the sum -- NO
opponency); unbind = conj diagonal synapse; cleanup = phase-cosine argmax. Abstention (the no-confab moat): the
relational query returns None when no stored fact's cue roles match (architecture-preserved).
"""
import os
from collections import namedtuple

import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.enums import NeuronModel
from sim.bridge import SimulationBridge
from sim.backend import to_host

ROLES = ("agent", "action", "patient", "polarity", "attribute", "attribute2")
DEFAULT_VOCAB = ["dog", "cat", "go", "run", "come", "stop", "look", "north", "south", "east", "west", "apple",
                 "river", "big", "small", "hot", "cold"]
# A recursive SVO clause that can be a filler ('dog look (cat go north)'). Mirrors core_sim_composition.Clause.
Clause = namedtuple("Clause", ["agent", "action", "patient"])


def _is_clause(x):
    """A clause-like filler: any namedtuple with (agent, action, patient) fields. Duck-typed so it recognizes BOTH
    this module's Clause AND core_sim_composition.Clause (the BrainConversationalAgent passes the latter) -- they are
    distinct namedtuple classes, so isinstance() would miss across them. A plain tuple (e.g. an ('adj', 'noun')
    attribute) has no _fields -> correctly NOT a clause."""
    return getattr(x, "_fields", None) == ("agent", "action", "patient")


def _build_rf_bridge(n, seed=42):
    cfg = CoreSimConfig()
    cfg.num_neurons = int(n)
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
              "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
              "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_brain_region_framework"):
        if hasattr(cfg, f):
            setattr(cfg, f, False)
    cfg.ou_std_current_pA = 0.0
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge._initialize_simulation_data(called_from_playback_init=False)
    bridge.core_config.neuron_model_type = NeuronModel.RESONATE_AND_FIRE.name
    return bridge


class RFPhasorComposer:
    def __init__(self, seed=42, D=64, vocab=None, period=200, enable_spiking_cleanup=False,
                 enable_substrate_store=False, grounded_codes=None, enable_rf_cudagraph=False,
                 encoding_gain_fn=None, local_reciprocal_unbind=False, trace=False,
                 enable_source_monitor=False, source_monitor_D=None, source_monitor_seed_offset=1000003,
                 enable_plastic_source_monitor=False, plastic_source_config=None,
                 plastic_source_seed_offset=0,
                 enable_sparse_index=False, sparse_index_g=3, sparse_index_G=16,
                 sparse_index_c=8, sparse_index_conf_floor=0.5,
                 enable_codebook_cache=False,
                 enable_decode_escalation=False, decode_escalate_margin=0.008,
                 decode_escalate_period=2000, spiking_recall_margin=False):
        self.seed = int(seed)
        self.D = int(D)
        self.period = int(period)
        # (B3 per-turn "brain activity", opt-in, DEFAULT-OFF = byte-identical) READ-ONLY trace of what the brain DID on
        # the LAST query: the decoded role-words + their cleanup match-confidence (per role), which stored fact-block
        # matched + how many were scanned, and a scalar RF activity gauge (the fraction of readout neurons that crossed
        # `cp_rf_fired.mean()` + the mean recovery magnitude |Z|). Populated from the ALREADY-COMPUTED `sims` /
        # decoded-word / matched-index produced by `_scan_first_match`/`_cleanup_all` + two `.mean()` reads of the
        # cached `_resonate` bridge -- NO extra resonate, NO extra GPU work, strictly observational of state the query
        # already produced. The no-confab moat is UNCHANGED (an abstain records matched_fact_index=None +
        # "scanned N, none matched" WITHOUT supplying a fallback answer). When `trace=False` (default) the dict is NOT
        # built -> byte-identical (the test-oracle + numpy-CPU paths are untouched). See
        # research/findings/raw/_b3_activity_viz_scoping.md (Option A).
        self.trace = bool(trace)
        self.last_trace = None
        self._last_resonate_n = None      # n of the most recent _resonate (for the gauge read; trace-only)
        # (Lane C source-monitor burn-down, opt-in, DEFAULT-OFF = byte-identical) store a redundant, independent
        # source-memory echo of each fact in its own FHRR/RF codebook. The production self-schema honesty hook can ask
        # this echo whether a recalled answer fits the cue, instead of reading the exact Python fact dict carried in the
        # trace. This is still a bounded engineering scaffold around a second memory trace, not the final biological
        # source-monitoring circuit, but it removes the most direct source-metadata shortcut.
        self.enable_source_monitor = bool(enable_source_monitor)
        self.source_monitor_D = int(source_monitor_D) if source_monitor_D is not None else max(int(D), 64)
        self.source_monitor_seed_offset = int(source_monitor_seed_offset)
        # (Lane C plastic-source rung, opt-in, DEFAULT-OFF) a source tag is learned only when an explicit
        # `observe_source_event` co-activates a complete proposition and an external-source population. `store` alone
        # does not teach it. Query-time evidence reads only learned synapses for the live recalled candidate. Its
        # BLAKE2 proposition namespace is independent from the composer's NumPy RF codebook, so the validated seed can
        # be reused without coupling the representations through an arbitrary unvalidated seed offset.
        self.enable_plastic_source_monitor = bool(enable_plastic_source_monitor)
        self.plastic_source_seed_offset = int(plastic_source_seed_offset)
        self._plastic_source_memory = None
        if self.enable_plastic_source_monitor:
            from research.runners.plastic_source_memory import PlasticSourceMemory
            self._plastic_source_memory = PlasticSourceMemory(
                seed=self.seed + self.plastic_source_seed_offset,
                config=plastic_source_config,
            )
        # (Tier-2 #6, opt-in, DEFAULT-OFF = byte-identical) DOPAMINE-GATED ENCODING STRENGTH (Lisman-Grace
        # hippocampal-VTA loop; Kandel D.16 -- dopamine gates the entry of information into LONG-TERM memory, making a
        # trace STABLE vs degradable). encoding_gain_fn: an optional callable () -> float read AT STORE TIME (the
        # shared `dopamine` concentration in deployment; a probe value in the de-risk). When set, the fact's composite
        # phasor written into the SUBSTRATE store weights (_store_substrate) is multiplied by this per-fact gain `g`.
        # Because the RF phase read-out has a hard MAGNITUDE FLOOR (sim/bridge.py:5589 `_rf_mag2 > _rf_floor2` -- a
        # readout neuron whose |Z| decays below the floor never spikes -> reads phase 0 = garbage), a higher-gain
        # (rewarded) fact reconstructs ABOVE the floor under common read damage where a unit-gain (neutral) fact
        # degrades BELOW it -> the rewarded fact wins the cue-match scan. NOT a vacuous global gain: the floor is the
        # nonlinearity that makes it differential. None -> g=1.0 for every fact -> the byte-identical unit-magnitude
        # write. Applies to the substrate store (enable_substrate_store=True); the numpy-kb fast path stores phases
        # (no magnitude), so the gain is recorded but only the substrate read exercises the floor.
        self.encoding_gain_fn = encoding_gain_fn
        # (Tier-2 #6 de-risk knobs, DEFAULT-PRESERVING) common READ DAMAGE applied at substrate retrieve so the
        # graceful-degradation knee can be reached (where a unit-gain fact starts to fail). _retrieve_lam: the decay
        # lambda used by _retrieve_substrate's rf_kick (more negative = faster magnitude decay over the read window =
        # the trace-degradation analogue, Kandel D.16). _retrieve_kick_mag: the trigger kick magnitude at retrieve
        # (scales every readout's magnitude in common -> directly probes the floor). _retrieve_floor: the RF floor at
        # retrieve. Defaults (lam=0.0, kick_mag=1.0, floor=1e-3) reproduce the current _retrieve_substrate EXACTLY.
        self._retrieve_lam = 0.0
        self._retrieve_kick_mag = 1.0
        self._retrieve_floor = 1.0e-3
        # _retrieve_noise (DEFAULT 0.0 = OFF = byte-identical): common, GAIN-INDEPENDENT additive complex READ noise
        # on the recovered readout phasor, with the RF magnitude floor applied to the NOISY phasor (a readout neuron
        # whose noisy |z| falls below _retrieve_read_floor reads garbage phase -- the documented RF floor). This is the
        # honest physical damage: read noise of fixed magnitude competes with the readout's SIGNAL magnitude `g*M`, so
        # a higher-gain (rewarded) fact has higher per-neuron SNR -> cleaner phase -> survives the floor, while a
        # unit-gain (neutral) fact's low-SNR neurons drop below the floor -> garbled phase -> the cleanup mis-recalls.
        # The differential is the floor x noise interaction (NOT a hand-set floor at the signal magnitude). Seeded by
        # _retrieve_noise_rng for reproducibility per query call sequence.
        self._retrieve_noise = 0.0
        self._retrieve_read_floor = 1.0e-2
        self._retrieve_noise_rng = np.random.default_rng(seed)
        # (perf, opt-in) route the per-op resonate window through the fused RF megakernel (one CUDA kernel/step)
        # instead of the ~15-kernel/step loop. Default OFF -> the validated loop path. == loop at the phase-read
        # tolerance (tests/test_rf_megakernel.py). See docs/plans/2026-06-17-resonate-cudagraph-refactor-design.md.
        self._enable_rf_cudagraph = bool(enable_rf_cudagraph)
        # (FHRR-B mechanism 1, opt-in, DEFAULT-OFF = byte-identical) DERIVE the unbind synapse from the bind synapse by
        # a one-time LOCAL reciprocal-conjugate WIRING RULE at construction, instead of the host re-computing
        # conj(role) from the role code per op. The bind synapse weight IS the role phasor zr[k] (developmental-random,
        # drawn once from rng.uniform(seed) -- a genome-style wiring rule, accepted as self-organized like
        # sim/dendritic_neuron.py:25 / catalog F.12/D.18). The unbind synapse must carry conj(zr[k]); the only
        # genuine host residual was that the substrate was never TOLD "unbind = the per-component conjugate of its
        # bind partner" -- it re-derived conj(role) host-side from self.roles[role]. With this flag ON, _unbind_phases
        # builds the BIND connectivity (the role phasor installed directly) and applies a LOCAL per-synapse rule --
        # `_reciprocal_conjugate` flips each bind synapse's quadrature (imaginary) component, a purely-local operation
        # on each single synapse (no read of self.roles, no np.conj over the role vector) = a reciprocal connection
        # with a quadrature-sign flip, the biological reciprocal/transpose motif. The values are bit-for-bit the same
        # as the host-conj path (conj per component IS the per-synapse rule), so the whole who/what matrix + the
        # no-confab abstentions are byte-identical -- but the unbind STRUCTURE now emerges from a local construction
        # rule over the bind connectivity (host-free at runtime), the property a neuromorphic hardware port needs (a
        # one-time device configuration, memristor-crossbar / Loihi-synapse-table style). See
        # research/findings/2026-06-20-FHRR-B-mechanism1-local-reciprocal-unbind.md.
        self.local_reciprocal_unbind = bool(local_reciprocal_unbind)
        # (cheat-C conversion, opt-in) hold each fact's bound composite in the SUBSTRATE (per-fact trigger->readout
        # complex weights) instead of a numpy array in self.kb; retrieve via firing. Default OFF: numpy kb fast path.
        self.enable_substrate_store = bool(enable_substrate_store)
        # (cheat-B conversion, opt-in) route _cleanup through the fully-on-bridge spiking cleanup (matched filter on
        # the complex synapse + Izhikevich WTA). Default OFF: numpy argmax stays the fast path (the rate composer's
        # NEF-cleanup opt-in pattern). Validated == numpy multi-seed.
        self.enable_spiking_cleanup = bool(enable_spiking_cleanup)
        self._izh_bank_cache = {}      # Stage-2 Izhikevich WTA banks, keyed by candidate count
        self._cleanup_drive_pA = 60.0  # input-normalized drive for the winner (sane band 20-100; >=200 over-drives)
        self._cleanup_window = 120
        # spiking_recall_margin (scaffold-retirement backlog rank 9, opt-in, DEFAULT-OFF = byte-identical):
        # `research/coordination/scaffold_retirement_backlog.md` #9 -- the metacog honesty-hedge's EVIDENCE
        # derivation (`metacog_production_organ.mean_role_confidence`) reads `margin`/`margin_norm`/`margin_snr`,
        # all HOST ARITHMETIC over the matched-filter scores ((peak-runner_up)/peak or a z-score, computed by
        # `_cleanup_all_score_stats`/`OneBrainComposer._margin` -- numpy comparisons of cosine-similarity
        # magnitudes, not a read of the recall circuit's own spiking). When ON, `_spiking_margin` (below) ALSO
        # runs the SAME cached Izhikevich concept bank (`_izh_bank`) `_spiking_cleanup`/`OneBrainComposer.
        # _spiking_select` already drive for the on-substrate winner-PICK (2026-06-05-phase1-tpam-cleanup-derisk-
        # GO.md), at a SEPARATE, higher operating point (`_margin_drive_pA`, below -- NOT `_cleanup_drive_pA`),
        # and reports the winner-vs-runner-up SPIKE-COUNT margin as an ADDITIONAL trace field `margin_spiking` --
        # never in place of the existing fields, never on the answer path (the decoded word is unchanged; this is
        # a trace-only evidence side-channel exactly like `margin` itself, see `_cleanup_all_score_stats` and
        # `OneBrainComposer._block_role_scores`). Env BRAIN_METACOG_SPIKING_MARGIN=1 flips it on without a code
        # change at any construction call site (the `enable_sparse_index` precedent).
        #
        # PRODUCTION-FLIP VERIFIED NO-GO (2026-09-05, research/findings/2026-09-05-metacog-spiking-margin-
        # prodflip-verify-NOGO.md): default-ON was tried + INTEGRATED-verified (real webapp.server.brain_chat
        # handler, true production faculty config, 6/6 mandated seeds). Genuinely load-bearing (6/6 lesion
        # collapse) and content-preserving (recalled_svo byte-identical in every condition), but the ambiguous-
        # band residual this de-risk already characterized is NOT a rare edge case at the conversational
        # surface: on 3 of 6 seeds (42/44/100), a real degraded-recall turn reads CONFIDENT under the spiking
        # evidence while the shipped host evidence correctly HEDGES the SAME turn -- 4 such instances across 42
        # natural noise-sweep opportunities, 0 instances in the reverse (safer) direction. Stays default-OFF.
        self.spiking_recall_margin = bool(spiking_recall_margin) or (
            os.environ.get("BRAIN_METACOG_SPIKING_MARGIN", "").strip().lower() in ("1", "true", "on", "yes"))
        # `_margin_drive_pA` (measured 2026-09-05, NOT `_cleanup_drive_pA`): the winner-PICK's 60pA settles this
        # population's heterogeneous Izhikevich thresholds to a SUBTHRESHOLD fixed point WITHOUT FIRING AT ALL
        # within `_cleanup_window` (measured directly: 0 spikes/120 steps at 60pA on the tiny-demo's cached bank;
        # a single-outcome argmax-over-firing PICK never needs the loser to fire, so this was never noticed) --
        # unusable as a MARGIN (a margin needs a graded spike-COUNT, not a single is-there-a-winner bit). 300pA
        # reliably crosses threshold within the SAME 120-step window and produces a genuine graded count
        # (measured on the real tiny-demo's captured role-score distributions, clean + 4 synaptic-noise levels,
        # `research/findings/raw/_metacog_spiking_recall_margin_derisk/calibrate_margin.json`: Pearson r=0.964,
        # Spearman rho=0.900 against the host `margin` across 25 role reads). `_cleanup_window` (120) is reused
        # UNCHANGED -- it already suffices at this drive; no new window constant.
        self._margin_drive_pA = 300.0
        self.words = sorted(vocab) if vocab is not None else sorted(DEFAULT_VOCAB)
        rng = np.random.default_rng(seed)
        # phasor codes: phases in [0,1)^D per concept + per role (deterministic per seed)
        self.concepts = {w: rng.uniform(0.0, 1.0, self.D) for w in self.words}
        # RUNTIME VOCABULARY GROWTH (in-loop learning): a word first heard mid-conversation gets a fresh sparse random
        # phasor code allocated on demand (a new concept -> a new cell assembly), deterministic per seed via a dedicated
        # growth RNG so store + later recall use the SAME code. This lets the brain learn a fact about a genuinely NEW
        # thing ("otter caught clam"), not only facts made of build-time vocabulary. See _filler_phases.
        self._growth_rng = np.random.default_rng(int(seed) + 777)
        # (cheat-A conversion, opt-in) SENSORY-GROUNDED codes: a {word: phases[D]} dict (e.g. real V1 Gabor responses
        # projected to phases) overrides the random codes for those words. Validated == random at parity (the
        # grounding INTERFACE works on the RF substrate). HONEST boundary: producing meaningful grounded codes (real
        # object images + abstract-concept grounding) is the open problem -- the embodied-cognition limit; this is the
        # interface, not full semantic grounding.
        if grounded_codes:
            for w, ph in grounded_codes.items():
                if w in self.concepts:
                    self.concepts[w] = np.asarray(ph, dtype=float)
        # AFFIRM/NEGATE polarity fillers (phasor codes; cleaned up only against pol_words, not the main vocab)
        self.pol_words = ["AFFIRM", "NEGATE"]
        for tag in self.pol_words:
            self.concepts[tag] = rng.uniform(0.0, 1.0, self.D)
        self.roles = {r: rng.uniform(0.0, 1.0, self.D) for r in ROLES}
        # enable_sparse_index (KNOWLEDGE-SCALE fast path, board #66, PORT of OneBrainComposer's validated DG-indexed
        # cleanup -- research/runners/one_brain_composer.py `_ensure_dg_index`/`_dg_shard_select`/`_full_host_select`,
        # itself a reuse-by-import of `_sparse_indexed_retrieval_derisk.DGSparseIndex`, 6-seed GO). ADDITIVE + DEFAULT
        # OFF = BYTE-IDENTICAL to today. WHY: `_cleanup`/`_cleanup_all` match the recovered role phasor against ALL V
        # rows of the concept codebook (a V x D matched-filter) then argmax -- LINEAR in vocabulary. This composer is
        # the SHARD engine behind `ShardedPhasorStore` (the tiered LTM's cortical store), whose shards already keep
        # the PER-SHARD FACT COUNT flat (~200/shard) via agent-hash routing -- but every shard shares ONE global
        # codebook, so the CLEANUP step a routed query still pays is O(V) in the FULL vocabulary, not the shard's own
        # ~200 facts (2026-08-28 finding: 1.37s@24k words -> 20.7s@347k -> 33.8s@581k, tracking V not fact count).
        # Same brain-grounded fix as OneBrainComposer: a DG-like sparse index (dentate-gyrus pattern separation +
        # CA3-conjunction routing) routes the recovered phasor to a SMALL candidate shard of the codebook; the SAME
        # matched-filter cleanup (`np.cos(2pi(rec-code)).sum()`, mathematically identical to this class's own
        # `_cleanup`/`_cleanup_all` cosine score up to the /D normalization constant, so argmax is unaffected) then
        # runs only over that shard's rows. NO-REGRESSION: (1) the shard is a SUBSET of the codebook, so its peak
        # score <= the full peak -- an abstain under the full scan abstains under the shard too (the moat is intact
        # by construction); (2) a per-cleanup CONFIDENCE FALLBACK escalates a non-decisive shard read (peak <
        # conf_floor*D) to the full-codebook scan, so the decoded word is IDENTICAL to the full scan whenever the
        # shard read is ambiguous. Only engages for the MAIN vocabulary cleanup (`words is None`, i.e. `self.words`);
        # the 2-word `pol_words` polarity cleanup is always the direct scan (too small to benefit, and callers pass
        # `words=self.pol_words` explicitly, which this flag never intercepts). `_dg_index_source` (opt-in, None by
        # default): when set to ANOTHER RFPhasorComposer instance sharing this one's `words`/`concepts` objects (the
        # `ShardedPhasorStore(share_codebook=True)` graft), the index is built ONCE on that source and every sharing
        # shard reuses the SAME DGSparseIndex/codebook objects instead of each of S shards redundantly building its
        # own copy over the identical global vocabulary (a real S-fold memory blow-up avoided, not merely an
        # optimization -- S=3745 shards x an independent V=347k index would multiply the RSS budget by S). Env
        # BRAIN_SHARD_SPARSE_INDEX=1 flips it on without a code change (kept a DISTINCT env var from OneBrainComposer's
        # BRAIN_SPARSE_INDEX_RETRIEVAL so the two composers' defaults are reviewed/flipped independently -- the
        # 2026-08-27 finding already GO'd-but-left-OFF the OneBrainComposer flag as answers-identical/redundant at
        # 100k-bundle scale; this shard composer is the ACTUALLY-BLOCKING path at bulk-KB vocab scale). Biology
        # binding: research/biology/dg-ca3-sparse-index.md (shared with OneBrainComposer's port of the same mechanism).
        import os as _os
        self.enable_sparse_index = bool(enable_sparse_index) or (
            _os.environ.get("BRAIN_SHARD_SPARSE_INDEX", "").strip().lower() in ("1", "true", "on", "yes"))
        self._dg_g = int(sparse_index_g); self._dg_G = int(sparse_index_G); self._dg_c = int(sparse_index_c)
        self._dg_conf_floor = float(sparse_index_conf_floor)   # shard peak < floor*D -> escalate to the full scan
        self._dg_index = None          # the DGSparseIndex over the concept codebook (lazy; reuse-by-import)
        self._dg_codebook = None       # (V, D) concept phase-matrix aligned to self.words (fractional-cycle phases)
        self._dg_built_V = -1          # len(self.words) the current index/codebook were built for (rebuild on change)
        self._dg_index_source = None   # another RFPhasorComposer to delegate index-building to (shared-codebook graft)
        # (#66 knowledge-scale, board #192, DEFAULT-OFF = byte-identical). Cache the (V,D) cleanup codebook ONCE
        # per vocab state (rebuild only when len(words) changes -- the same invalidation rule `_dg_built_V` uses)
        # and reuse it in the full-vocabulary cleanup paths instead of re-stacking the phasor matrix from the
        # `concepts` dict on every query. The cached matrices ARE exactly what the parent rebuilds, so decode is
        # byte-identical by construction (independent of seed). This is the O(V) codebook-rebuild hot loop the
        # 2026-08-30 finding identified (~40% of per-query time at V~24k, scaling with V not the shard's ~200
        # facts). The (V,D) codebook object is shared across shards via the existing `_dg_index_source` graft when a
        # store is constructed with `share_codebook=True` (one 16.4MB object for all shards vs S independent
        # copies), so RSS stays flat at scale. `enable_codebook_cache=False` (default) reproduces the current
        # rebuild-every-query path exactly.
        self.enable_codebook_cache = bool(enable_codebook_cache)
        self._cb_frac = None       # (V,D) fractional-cycle codebook (V*D floats; read-only, shared)
        self._cb_z = None          # (V,D) phasor codebook (V*D complex128; read-only, shared)
        self._cb_cache_V = -1      # len(words) the cached codebook was built for; -1 = unbuilt
        # word -> its row index in `self._cb_frac`/`self._cb_z` (built in the SAME pass as those matrices in
        # `_ensure_codebook_cache`, so it can never drift out of alignment; invalidated on the same V change).
        # Lets `_escalate_role_match` gather winner-concept rows with ONE vectorized fancy-index into the cached
        # codebook instead of a per-candidate Python-loop dict-lookup + np.stack (the cupy-backend hotspot fixed
        # 2026-09-02, board #108 -- research/findings/2026-09-02-escalation-gating-tighten-latency-correctness-
        # safe-not-the-lever.md pinpointed this exact loop as the cupy-specific ~1303ms->target<1000ms driver).
        self._concept_row = None   # {word: row_idx}, aligned to self._cb_frac / self.words
        # (#66 seed-44 recall hole, 2026-09-01, DEFAULT-OFF = byte-identical). Confidence-gated finer-period
        # re-examination of a MATCH candidate ("effortful second look"). ROOT CAUSE it closes: the RF phase
        # readout `((period - spike_step) % period)/period` quantizes the recovered phase to 1/period (= 0.005 at
        # period=200), coarser than the real inter-word cleanup margin for some facts. When a fact's stored cue
        # role decodes (argmax over the full vocab) to the WRONG word by a razor-thin margin, `_scan_first_match`
        # rejects a fact that GENUINELY encodes the cued value -> a false abstain (the seed-44 hole:
        # berkeley_county_virginia's `located_in...` role lost to `pelagonians` by 0.0022 of mean-cos, flipping
        # what_does to None + ask_yes_no to unknown). The fix: for a fact still viable on the earlier cue roles
        # whose stored `role` decoded to a different word than the cued value BUT where the cued value is a
        # near-tie runner-up (winner_score - value_score <= margin), re-unbind THAT ONE fact's role at a FINER
        # resonate period (a more faithful, longer-integrated neural readout -- speed-secondary/faithfulness-first)
        # and accept the match iff the finer decode now argmaxes to the cued value. MOAT-SAFE BY CONSTRUCTION:
        # (1) it only fires for an IN-VOCABULARY cued value (an unknown cue word is not in `self.concepts`, so an
        # unknown-agent/unknown-relation moat query never escalates -> it always abstains); (2) the finer readout
        # converges to the ideal (closed-form) representation, so a fact that does not genuinely encode the cued
        # value stays rejected (escalation can only RECOVER a truly-stored fact the coarse readout dropped, never
        # manufacture a wrong match). Biology binding: a difficulty-dependent decision time -- an uncertain /
        # near-tie readout triggers longer evidence integration before committing (speed-accuracy trade-off /
        # drift-diffusion decision-time). Latency stays at the fast common case (period unchanged) because the
        # finer re-resonate touches only the rare near-tie candidates, not every query.
        #
        # (#108 R1 gating TIGHTEN, 2026-09-02) `decode_escalate_margin` default 0.02 -> 0.008. The trigger must
        # only catch a near-tie the FINER readout can actually flip; the finding's measured seed-44 mean-cos
        # margin swing under readout refinement is coarse +0.0022 -> closed-form -0.0055 (a span of ~0.0077), so a
        # candidate decisive by MORE than ~0.0077 of mean-cos cannot be rescued by a finer period and never needed
        # the re-read. 0.008 sits just above that measured span AND 3.6x above the 0.0022 seed-44 coarse margin
        # (ample headroom to keep catching seed-44 + its unprobed thin-margin siblings), so it loses NO recovery
        # the 0.02 gate made while narrowing the trigger. **0.02 remains reachable as an explicit escape**
        # (pass `decode_escalate_margin=0.02`) for A/B or rollback. NOTE (numpy diagnosis, artifact
        # `_escalation_gating_tighten_smoke.json`): the 0.02 gate already fires on only ~4% of recall queries and
        # every observed flip is at the ~0.0022 seed-44 margin, so this tighten is correctness-HARDENING; it is NOT
        # by itself the #108 latency lever (the ~+300ms cupy median regression is a per-query cost independent of
        # the trigger margin -- the faithful 6-seed cupy re-verify is the gate on that).
        self.enable_decode_escalation = bool(enable_decode_escalation)
        self.decode_escalate_margin = float(decode_escalate_margin)
        self.decode_escalate_period = int(decode_escalate_period)
        self.kb = []  # (fact_dict, composite_phases)
        self._source_kb = []  # (roles_present, independent_source_composite_phases)
        if self.enable_source_monitor:
            srng = np.random.default_rng(self.seed + self.source_monitor_seed_offset)
            self.source_concepts = {
                w: srng.uniform(0.0, 1.0, self.source_monitor_D)
                for w in self.words
            }
            for tag in self.pol_words:
                self.source_concepts[tag] = srng.uniform(0.0, 1.0, self.source_monitor_D)
            self.source_roles = {
                r: srng.uniform(0.0, 1.0, self.source_monitor_D)
                for r in ROLES
            }
        else:
            self.source_concepts = {}
            self.source_roles = {}
        self._dlpfc = None       # dialogue-planning Control (lazy; rebuilt only when the association graph changes)
        self._dlpfc_key = None
        self._bridge_cache = {}  # (c-opt) reuse RF bridges by neuron count -> avoid _initialize_simulation_data per op

    # --- RF complex-synapse ops (each op a per-op RF bridge; reuse-by-import the substrate) ---
    def _resonate(self, n, conns, kick, period=None):
        # (c-opt) reuse a cached bridge per neuron count; zero its complex weights (rf_set_complex_weights appends)
        # and rf_kick resets the RF state -> each op is clean. Avoids _initialize_simulation_data per op.
        # `period` (default None -> self.period) lets a caller run a FINER-resolution resonate for the same op (the
        # decode-escalation "second look"); period is per-rf_kick, so the cached bridge is reused unchanged.
        per = self.period if period is None else int(period)
        b = self._bridge_cache.get(n)
        if b is None:
            b = _build_rf_bridge(n, self.seed)
            b.core_config.enable_rf_cudagraph = self._enable_rf_cudagraph   # opt-in megakernel resonate fast path
            self._bridge_cache[n] = b
        b.rf_set_complex_weights(conns)   # (c-opt) builds the sparse complex weights FRESH each op -> replaces; no reset needed
        b.rf_kick(kick, period=per, lam=0.0)
        b.rf_resonate_steps(per + 8)   # (c-opt) fast RF dynamics loop -- skips the full-step machinery
        if self.trace:
            self._last_resonate_n = n          # remember which cached bridge to read for the gauge (trace-only)
        return np.asarray(b.rf_read_phases())

    @staticmethod
    def _to_phasor(phases):
        return np.exp(2j * np.pi * np.asarray(phases))

    def _bind_conns(self, role_phases, lo=0, hi=None):
        """The BIND connectivity for a diagonal role->filler bind over neurons [lo, lo+D): the forward synapse
        (lo+D+k <- lo+k) carries the role phasor zr[k] (the developmental-random role code, installed directly).
        `hi` (= lo+D by default) names the post offset; a single 2D-block uses lo=0,hi=D. Returns a list of
        (post, pre, complex_weight) tuples -- the exact same structure the host-conj path's bind half installs."""
        D = self.D
        hi = lo + D if hi is None else hi
        zr = self._to_phasor(role_phases)
        return [(hi + k, lo + k, zr[k]) for k in range(D)]

    @staticmethod
    def _local_conj(z):
        """The LOCAL per-component phase-conjugate of a unit phasor (array or scalar): the quadrature
        (imaginary-component) sign flip `re + i*im -> re - i*im`, computed LOCALLY from the value's own re/im with NO
        np.conj. For |z|=1 this equals conj(z) = 1/z bit-for-bit. The single shared primitive both the single-block
        rule (_reciprocal_conjugate) and OneBrainComposer's unbind-structure build use, so the "no host conj, just a
        local quadrature flip" purity property is uniform across the production composers."""
        a = np.asarray(z)
        return a.real - 1j * a.imag

    def _cleanup_conj(self, z):
        """(FHRR-B cleanup-codebook residual, mechanism 1's local rule extended to the CLEANUP codebook.) The cleanup /
        matched-filter codebook installs, per concept, a synapse carrying conj(concept_phasor) so the recovered phasor
        correlates against each concept's CONJUGATE (the matched filter IS the transpose/reciprocal of the encoder). The
        only host residual was the substrate re-deriving that conjugate host-side via np.conj over the concept code each
        build. Since conj is per-component, the cleanup synapse is the per-component quadrature-flip of its concept
        synapse -- the SAME LOCAL reciprocal-conjugate wiring rule already used for the unbind (_local_conj), a purely-
        local function of each single synapse's own weight, NO np.conj over the concept vector. With local_reciprocal_
        unbind ON the cleanup codebook is derived by this local rule (bit-for-bit == conj for a unit phasor, so the
        whole who/what matrix + no-confab abstentions are byte-identical); OFF (default) = the legacy host np.conj,
        unchanged. Biologically the reciprocal/transpose of the concept-code synapse (the matched filter = encoder
        transpose). See 2026-06-20-FHRR-B-cleanup-codebook-local-conj.md."""
        return self._local_conj(z) if self.local_reciprocal_unbind else np.conj(z)

    @classmethod
    def _reciprocal_conjugate(cls, bind_conns):
        """The LOCAL reciprocal-conjugate WIRING RULE (FHRR-B mechanism 1): derive the UNBIND synapses from the BIND
        synapses by a per-synapse operation -- for each bind synapse (post, pre, w), the reciprocal/feedback synapse
        carries the phase-conjugate of w (the _local_conj quadrature flip). Computed LOCALLY from each synapse's OWN
        weight (NOT re-derived from the role code, NOT np.conj over the role vector). Biologically a reciprocal
        connection with a quadrature-sign flip (the ubiquitous cortical/thalamocortical reciprocal motif). The values
        equal conj(w) bit-for-bit, so held-out recovery is byte-identical to the host-conj path -- but the unbind
        structure now emerges from a one-time local construction rule over the bind connectivity, host-free at runtime
        (the neuromorphic-port property: a one-time device configuration, no host in the loop per op)."""
        return [(post, pre, complex(cls._local_conj(w))) for (post, pre, w) in bind_conns]

    def _bind(self, role_phases, filler_phases):
        """bound = role_phasor (x) filler_phasor, via a diagonal complex synapse (filler pre -> bound post,
        weight = the role phasor)."""
        D = self.D
        zf = self._to_phasor(filler_phases)
        zr = self._to_phasor(role_phases)
        conns = [(D + k, k, zr[k]) for k in range(D)]
        kick = np.zeros(2 * D, dtype=np.complex128)
        kick[:D] = zf
        return self._resonate(2 * D, conns, kick)[D:]

    def _bundle(self, phase_list):
        """composite[k] = sum_l phase_list[l][k] via unit complex synapses (NO opponency)."""
        L = len(phase_list)
        D = self.D
        conns = [(L * D + k, l * D + k, 1.0) for l in range(L) for k in range(D)]
        kick = np.zeros((L + 1) * D, dtype=np.complex128)
        for l in range(L):
            kick[l * D:(l + 1) * D] = self._to_phasor(phase_list[l])
        return self._resonate((L + 1) * D, conns, kick)[L * D:]

    def _filler_phases(self, filler):
        """The phasor phases to bind for a filler: a concept's code, OR (recursively) a Clause's bound composite."""
        if _is_clause(filler):
            return self._encode({"agent": filler.agent, "action": filler.action, "patient": filler.patient})
        code = self.concepts.get(filler)
        if code is None:
            # RUNTIME GROWTH: allocate a fresh code for a never-seen word (a new concept assembly). Deterministic per
            # seed; confab-safe (querying a never-stored word allocates a code but finds no matching fact -> abstain).
            code = self._growth_rng.uniform(0.0, 1.0, self.D)
            self.concepts[filler] = code
            # the matched-filter cleanup/scan codebooks are built from self.words -> the grown word must join it so a
            # recall can DECODE it (else the fact stores but the patient cannot be recovered). Kept sorted.
            if isinstance(filler, str) and filler not in self.words:
                import bisect
                bisect.insort(self.words, filler)
        return code

    def _encode(self, fact):
        bounds = [self._bind(self.roles[r], self._filler_phases(fact[r])) for r in ROLES if r in fact]
        return self._bundle(bounds) if len(bounds) > 1 else bounds[0]

    def _source_bind(self, role_phases, filler_phases):
        D = self.source_monitor_D
        zf = self._to_phasor(filler_phases)
        zr = self._to_phasor(role_phases)
        conns = [(D + k, k, zr[k]) for k in range(D)]
        kick = np.zeros(2 * D, dtype=np.complex128)
        kick[:D] = zf
        return self._resonate(2 * D, conns, kick)[D:]

    def _source_bundle(self, phase_list):
        L = len(phase_list)
        D = self.source_monitor_D
        conns = [(L * D + k, l * D + k, 1.0) for l in range(L) for k in range(D)]
        kick = np.zeros((L + 1) * D, dtype=np.complex128)
        for l in range(L):
            kick[l * D:(l + 1) * D] = self._to_phasor(phase_list[l])
        return self._resonate((L + 1) * D, conns, kick)[L * D:]

    def _source_filler_phases(self, filler):
        if _is_clause(filler):
            return self._source_encode({"agent": filler.agent, "action": filler.action, "patient": filler.patient})
        return self.source_concepts[filler]

    def _source_encode(self, fact):
        bounds = [
            self._source_bind(self.source_roles[r], self._source_filler_phases(fact[r]))
            for r in ROLES
            if r in fact
        ]
        return self._source_bundle(bounds) if len(bounds) > 1 else bounds[0]

    def _source_unbind_phases(self, composite_phases, role):
        D = self.source_monitor_D
        zc = self._to_phasor(composite_phases)
        zr_conj = np.conj(self._to_phasor(self.source_roles[role]))
        conns = [(D + k, k, zr_conj[k]) for k in range(D)]
        kick = np.zeros(2 * D, dtype=np.complex128)
        kick[:D] = zc
        return self._resonate(2 * D, conns, kick)[D:]

    def _source_cleanup_stats(self, rec, words=None):
        words = words if words is not None else self.words
        if len(rec) == 0:
            return []
        rec_z = np.exp(2j * np.pi * np.asarray(rec))
        cb = np.stack([np.exp(2j * np.pi * self.source_concepts[w]) for w in words])
        sims = (rec_z @ np.conj(cb).T).real / float(self.source_monitor_D)
        order = np.argsort(sims, axis=1)
        out = []
        for i in range(len(rec)):
            top = int(order[i, -1])
            runner = int(order[i, -2]) if len(words) > 1 else top
            top_raw = float(sims[i, top])
            runner_raw = float(sims[i, runner])
            confidence = float(np.clip(top_raw, 0.0, 1.0))
            runner_conf = float(np.clip(runner_raw, 0.0, 1.0))
            out.append({
                "word": words[top],
                "confidence": confidence,
                "winner_score_raw": top_raw,
                "runner_word": words[runner],
                "runner_confidence": runner_conf,
                "runner_score_raw": runner_raw,
                "margin": float(top_raw - runner_raw),
                "conflict": float(runner_conf / (confidence + runner_conf + 1e-9)),
            })
        return out

    def _source_decode_role(self, comp, role, words=None):
        rec = self._source_unbind_phases(comp, role)
        stats = self._source_cleanup_stats(np.asarray(rec)[None, :], words=words)
        return stats[0] if stats else {"word": None, "confidence": None}

    def _source_store_echo(self, fact):
        if not self.enable_source_monitor:
            return
        try:
            roles_present = tuple(r for r in ROLES if r in fact)
            self._source_kb.append((roles_present, self._source_encode(fact)))
        except KeyError:
            # Unknown fillers cannot be echoed through this vocabulary; leave the source monitor unavailable for them.
            self._source_kb.append(((), None))

    def _source_scan_first_match(self, cue_roles):
        cue_items = list(cue_roles.items())
        for i, (roles_present, comp) in enumerate(self._source_kb):
            if comp is None or not all(role in roles_present for role, _ in cue_items):
                continue
            cue_stats = []
            ok = True
            for role, asserted in cue_items:
                st = dict(self._source_decode_role(comp, role))
                st.update({"role": role, "cue": True, "asserted": asserted})
                cue_stats.append(st)
                if st.get("word") != asserted:
                    ok = False
                    break
            if ok:
                return i, roles_present, comp, cue_stats
        return None, (), None, []

    @staticmethod
    def _source_min_conf(stats):
        vals = [
            float(s["confidence"])
            for s in stats
            if s.get("confidence") is not None
        ]
        return float(min(vals)) if vals else None

    def source_consistency_record(self, *, kind, cue, raw_answer):
        """Independent source-memory echo check for Lane C confidence selection.

        The evidence comes from a second RF/FHRR memory trace with independent concept and role codes. It never reads the
        exact fact dict stored in `self.kb`; it asks the redundant echo to decode the cue and expected answer roles.
        """
        out = {
            "available": bool(self.enable_source_monitor and self._source_kb),
            "source": "rf_independent_source_echo",
            "source_monitor_D": int(self.source_monitor_D),
            "matched_source_index": None,
            "source_consistent": None,
            "source_expected_answer": None,
            "source_answer_matches": None,
            "source_cue_confidence": None,
            "source_answer_confidence": None,
            "source_confidence": None,
            "cue_roles": [],
            "answer_roles": [],
        }
        if not out["available"]:
            return out
        cue = tuple(cue)
        if kind == "what_does" and len(cue) == 2:
            cue_roles = {"agent": cue[0], "action": cue[1]}
            idx, roles_present, comp, cue_stats = self._source_scan_first_match(cue_roles)
            out["matched_source_index"] = None if idx is None else int(idx)
            out["cue_roles"] = cue_stats
            if comp is None:
                return out
            answer_stats = []
            attrs = []
            for role in ("attribute", "attribute2"):
                if role in roles_present:
                    st = dict(self._source_decode_role(comp, role))
                    st.update({"role": role, "cue": False})
                    answer_stats.append(st)
                    attrs.append(st.get("word"))
            st = dict(self._source_decode_role(comp, "patient"))
            st.update({"role": "patient", "cue": False})
            answer_stats.append(st)
            patient = st.get("word")
            expected = " ".join([str(x) for x in attrs + [patient] if x is not None])
            out["answer_roles"] = answer_stats
            out["source_expected_answer"] = expected
        elif kind == "yes_no" and len(cue) == 3:
            cue_roles = {"agent": cue[0], "action": cue[1], "patient": cue[2]}
            idx, _roles_present, comp, cue_stats = self._source_scan_first_match(cue_roles)
            out["matched_source_index"] = None if idx is None else int(idx)
            out["cue_roles"] = cue_stats
            if comp is None:
                return out
            st = dict(self._source_decode_role(comp, "polarity", words=self.pol_words))
            st.update({"role": "polarity", "cue": False})
            out["answer_roles"] = [st]
            pol = st.get("word")
            out["source_expected_answer"] = "no" if pol == "NEGATE" else "yes"
        else:
            return out
        out["source_answer_matches"] = bool(out["source_expected_answer"] == raw_answer)
        out["source_consistent"] = bool(out["source_answer_matches"])
        out["source_cue_confidence"] = self._source_min_conf(out["cue_roles"])
        out["source_answer_confidence"] = self._source_min_conf(out["answer_roles"])
        conf_vals = [
            v for v in (out["source_cue_confidence"], out["source_answer_confidence"])
            if v is not None
        ]
        out["source_confidence"] = float(min(conf_vals)) if conf_vals else None
        return out

    def observe_source_event(self, *, kind, cue, candidate, learning_enabled=True):
        """Teach that a proposition was externally experienced.

        This event is deliberately separate from `store`: placing a fact in the
        primary memory is not by itself evidence about where that fact came from.
        """
        if self._plastic_source_memory is None:
            return {
                "observed": False,
                "available": False,
                "source": "plastic_hebbian_proposition_source",
            }
        rec = self._plastic_source_memory.observe(
            kind=str(kind),
            cue=tuple(cue),
            candidate=str(candidate),
            learning_enabled=bool(learning_enabled),
        )
        return {
            "observed": True,
            "available": True,
            "source": "plastic_hebbian_proposition_source",
            **rec,
        }

    def plastic_source_consistency_record(self, *, kind, cue, raw_answer):
        """Read learned source support without consulting the primary fact table."""
        if self._plastic_source_memory is None:
            return {
                "available": False,
                "source": "plastic_hebbian_proposition_source",
                "source_consistent": None,
                "source_confidence": None,
            }
        return self._plastic_source_memory.support(
            kind=str(kind),
            cue=tuple(cue),
            candidate=str(raw_answer),
        )

    def _render(self, comp_phases, role, stored, order_fn=None):
        """Render `role`'s filler from a composite, FROM THE RF UNBIND. `stored` (a word or Clause) ROUTES
        flat-cleanup vs recursive clause-decode; the content is decoded from the substrate, not the stored labels.
        `order_fn` (opt-in, default None = the host f-string): when set, the inner clause's SVO word order is
        produced by the de-risked spiking serial-order generator instead of the host literal (the generation path
        passes it; the Q&A path leaves it None)."""
        rec = self._unbind_phases(comp_phases, role)
        if _is_clause(stored):
            a = self._cleanup(self._unbind_phases(rec, "agent"))
            ac = self._cleanup(self._unbind_phases(rec, "action"))
            pt = self._cleanup(self._unbind_phases(rec, "patient"))
            words = [a, ac, pt]
            if order_fn is not None:
                return " ".join(words[i] for i in order_fn(len(words)))   # neural serial-order (inner clause)
            return f"{a} {ac} {pt}"
        return self._cleanup(rec)

    def _unbind_phases(self, composite_phases, role, period=None):
        """recovered = conj(role_phasor) (x) composite, via a conj diagonal complex synapse.

        The unbind synapse weights are conj(role). DEFAULT (local_reciprocal_unbind=False): the host computes
        conj(self.roles[role]) and injects it (the legacy path -- the genuine host residual). With the flag ON: the
        unbind synapses are DERIVED from the BIND synapses by the LOCAL reciprocal-conjugate rule (_bind_conns ->
        _reciprocal_conjugate) -- the role phasor is installed as the bind synapse (a developmental wiring rule) and
        the unbind synapse is its per-component quadrature-flip, with NO host re-derivation of conj from the role
        code. Byte-identical (conj per component IS the per-synapse rule); the structure is then host-free at runtime
        (the neuromorphic-port property). See 2026-06-20-FHRR-B-mechanism1-local-reciprocal-unbind.md."""
        D = self.D
        zc = self._to_phasor(composite_phases)
        if self.local_reciprocal_unbind:
            conns = self._reciprocal_conjugate(self._bind_conns(self.roles[role]))   # local rule over bind connectivity
        else:
            zr_conj = np.conj(self._to_phasor(self.roles[role]))                     # host re-derivation (legacy)
            conns = [(D + k, k, zr_conj[k]) for k in range(D)]
        kick = np.zeros(2 * D, dtype=np.complex128)
        kick[:D] = zc
        return self._resonate(2 * D, conns, kick, period=period)[D:]

    def _izh_bank(self, V):
        """A cached Izhikevich concept bank of V neurons (no wiring; driven by external current) -- the Stage-2 WTA."""
        bank = self._izh_bank_cache.get(V)
        if bank is None:
            cfg = CoreSimConfig()
            cfg.num_neurons = int(V)
            cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
            cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
            cfg.seed = self.seed
            cfg.dt_ms = 1.0
            cfg.connections_per_neuron = 0
            cfg.num_traits = 1
            for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
                      "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
                      "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_brain_region_framework"):
                if hasattr(cfg, f):
                    setattr(cfg, f, False)
            cfg.ou_std_current_pA = 0.0
            bank = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                    runtime_state=RuntimeState(), gpu_config=GPUConfig())
            bank._initialize_simulation_data(called_from_playback_init=False)
            # snapshot the resting state so each cleanup starts clean (a cached bank's v/u persist across calls,
            # which would let a recently-fired neuron's adapted state bias the next cleanup's WTA)
            bank._cleanup_v0 = bank.cp_membrane_potential_v.copy()
            bank._cleanup_u0 = bank.cp_recovery_variable_u.copy()
            self._izh_bank_cache[V] = bank
        return bank

    def _spiking_cleanup(self, rec_phases, words):
        """Fully on-bridge cleanup (clears cheat B). Stage 1 -- the matched FILTER is the bridge's complex-synapse
        matvec (the SAME op as unbind): install conj(codebook) synapses (rec -> concept), kick rec, one matvec step,
        read each concept neuron's |c_k| = |S* rec| off the membrane (cp_membrane_potential_v / cp_recovery_variable_u
        = the RF re/im). Stage 2 -- the SELECTION is a spiking Izhikevich WTA driven by the input-normalized scores;
        winner = argmax-over-FIRING (a readout of spiking output, as the NEF cleanup's final argmax). The only numpy
        is the membrane readout + the firing-argmax readout -- NO numpy COMPUTATION of the match or the selection.
        Validated == numpy argmax multi-seed: research/findings/2026-06-05-phase1-tpam-cleanup-derisk-GO.md."""
        D = self.D
        V = len(words)
        # Stage 1: matched filter on the complex synapse (concept k = index D+k receives rec via conj(code_k)).
        conns = []
        for k in range(V):
            cc = self._cleanup_conj(self._to_phasor(self.concepts[words[k]]))   # local reciprocal rule when ON; conj when OFF
            for d in range(D):
                conns.append((D + k, d, cc[d]))
        b = self._bridge_cache.get(D + V)
        if b is None:
            b = _build_rf_bridge(D + V, self.seed)
            self._bridge_cache[D + V] = b
        b.rf_set_complex_weights(conns)
        kick = np.zeros(D + V, dtype=np.complex128)
        kick[:D] = self._to_phasor(rec_phases)
        b.rf_kick(kick, period=self.period, lam=0.0)
        b.rf_resonate_steps(1)
        # The matched-filter score is Re(c_k) = the concept neuron's membrane (re) = exactly the numpy cos score
        # (mean cos = Re(c_k)/D). Rectified so off-target concepts (Re~0 / negative) emit ZERO drive -> silent ->
        # a clean WTA (the NEF cleanup's "off-target emits zero spikes"). |c_k| would leave off-targets driven.
        re = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float)[D:D + V]
        scores = np.maximum(re, 0.0)
        peak = float(scores.max())
        if peak <= 1e-9:
            return words[int(np.argmax(scores))]
        # Stage 2: spiking WTA (input-normalized drive -> firing -> argmax-over-firing).
        drive = (scores / peak) * self._cleanup_drive_pA
        bank = self._izh_bank(V)
        bank.cp_membrane_potential_v[:] = bank._cleanup_v0   # reset to resting -> each cleanup is independent
        bank.cp_recovery_variable_u[:] = bank._cleanup_u0
        import sim.backend as _b
        xp, _ = _b.get_backend()
        bank.cp_external_input_current[:] = xp.asarray(drive, dtype=bank.cp_external_input_current.dtype)
        firing = np.zeros(V)
        for _ in range(self._cleanup_window):
            bank._run_one_simulation_step()
            firing += np.asarray(to_host(bank.cp_firing_states)).astype(float)
        bank.cp_external_input_current[:] = 0.0
        if float(firing.max()) <= 0.0:
            return words[int(np.argmax(scores))]
        return words[int(np.argmax(firing))]

    def _spiking_margin(self, scores, lesion=False):
        """WTA winner-vs-runner-up SPIKE-COUNT margin off the SAME Izhikevich cleanup bank Stage 2 that
        `_spiking_cleanup`/`OneBrainComposer._spiking_select` already drive for the on-substrate winner-PICK
        (scaffold-retirement backlog rank 9: `research/coordination/scaffold_retirement_backlog.md` -- replace
        the metacog honesty-hedge's role-decode evidence, `_margin`/`margin_norm`/`margin_snr` (all HOST
        ARITHMETIC comparisons of matched-filter score magnitudes: `(peak-runner_up)/peak` or a z-score), with a
        genuine read of the recall circuit's OWN spiking competition).

        Drives the SAME cached Izhikevich concept bank `_spiking_cleanup` uses (input-normalize `scores` to
        `_margin_drive_pA` -- a SEPARATE, higher operating point than the winner-pick's `_cleanup_drive_pA`; see
        that attribute's docstring for why -- integrate firing over `_cleanup_window`) and reads

            (firing[winner] - firing[runner_up]) / (firing[winner] + eps)

        -- the SAME normalized-decisiveness FORM `_margin` uses, off ACTUAL SPIKE COUNTS the competition produced,
        not a host comparison of membrane amplitudes. A degenerate silent competition (peak score <= 0, or no
        neuron fires within the window) reads margin 0 -- an uninformative/undecided competition, the same
        verdict the host formula gives a flat/zero score vector. `V<2` (nothing to compete against) reads 0.

        `lesion=True` (load-bearing test, mirrors `metacog_production_organ.nmda_norm_margin`'s own
        evidence-differential lesion): REMOVE the recall circuit's discrimination BEFORE the competition runs by
        replacing `scores` with a uniform (all-tied) vector of the same length -- every candidate drives the bank
        identically, so any resulting margin can only be competition NOISE, not a read of the (now-absent) score
        differential. A genuine read of the circuit's OWN discrimination must collapse toward 0 under this
        lesion regardless of the ORIGINAL scores' decisiveness; a mechanism that did NOT collapse would mean the
        margin was carrying information from somewhere other than this competition (a host leak).

        NOT a "conductance" read (docs/TERMS.md: name what is actually measured) -- this Izhikevich bank is
        driven by external CURRENT with no synapses (`connections_per_neuron=0`), so it carries no synaptic
        conductance state; the graded signal is FIRING COUNT over the cleanup window, the same quantity
        `_spiking_cleanup`'s winner-pick already reads (argmax-over-firing), here reported as a MARGIN between
        the top two instead of just the argmax. Trace-only by construction at every current call site
        (`_cleanup_all_score_stats`, `OneBrainComposer._block_role_scores`) -- never the answer-selection path,
        so a query's decoded word is byte-identical whether or not this is called."""
        scores = np.maximum(np.asarray(scores, dtype=float), 0.0)
        V = scores.size
        if V < 2:
            return 0.0
        peak = float(scores.max())
        if peak <= 1e-9:
            return 0.0
        if lesion:
            scores = np.ones(V, dtype=float)   # NO differential -> every candidate drives the bank identically
            peak = 1.0
        drive = (scores / peak) * self._margin_drive_pA
        bank = self._izh_bank(V)
        bank.cp_membrane_potential_v[:] = bank._cleanup_v0   # reset to resting -> each read is independent
        bank.cp_recovery_variable_u[:] = bank._cleanup_u0
        import sim.backend as _b
        xp, _ = _b.get_backend()
        bank.cp_external_input_current[:] = xp.asarray(drive, dtype=bank.cp_external_input_current.dtype)
        firing = np.zeros(V)
        for _ in range(self._cleanup_window):
            bank._run_one_simulation_step()
            firing += np.asarray(to_host(bank.cp_firing_states)).astype(float)
        bank.cp_external_input_current[:] = 0.0
        s = np.sort(firing)[::-1]
        if s[0] <= 0.0:
            return 0.0
        return float((s[0] - s[1]) / (s[0] + 1e-9))

    # --- (#66 knowledge-scale, PORTED from OneBrainComposer) DG-indexed cleanup fast path ---------------------
    def _ensure_dg_index(self):
        """Build (lazily, once per codebook mutation) the DG-like sparse index + the concept phase-matrix over the
        CURRENT cleanup codebook `self.words`. Reuse-by-import of the validated de-risk (research/runners/
        _sparse_indexed_retrieval_derisk.py, 6-seed GO -- the SAME class OneBrainComposer already uses; import
        deferred so a default-off composer never imports it). If `_dg_index_source` is set (the ShardedPhasorStore
        share_codebook graft), delegate entirely to that source composer's index instead of building a redundant
        copy over the identical shared `words`/`concepts` objects -- avoids an S-fold memory blow-up across shards.
        Rebuilt only when len(self.words) changed (a recruit/grow), so the moat is never served a stale index. m ~
        V^(1/g) keeps bucket occupancy O(1) -> the shard stays ~constant as V grows. Seeds from self.seed."""
        if self._dg_index_source is not None:
            self._dg_index_source._ensure_dg_index()
            self._dg_index = self._dg_index_source._dg_index
            self._dg_codebook = self._dg_index_source._dg_codebook
            self._dg_built_V = self._dg_index_source._dg_built_V
            return
        V = len(self.words)
        if self._dg_index is not None and self._dg_built_V == V:
            return
        from research.runners._sparse_indexed_retrieval_derisk import DGSparseIndex
        # concept codebook aligned to self.words, in FRACTIONAL-CYCLE phases (this composer's convention; the
        # phasor is exp(2pi i phase)). The de-risk index's feature convention is [cos(phase_rad), sin(phase_rad)],
        # so build/query in RADIANS (phase * 2pi).
        cb = np.stack([np.asarray(self.concepts[w], dtype=float) for w in self.words])   # (V, D) fractional-cycle
        m = max(2, int(np.ceil(V ** (1.0 / self._dg_g))))
        idx = DGSparseIndex(D=self.D, m=m, g=self._dg_g, G=self._dg_G, c=self._dg_c, seed=self.seed)
        idx.build(cb * (2.0 * np.pi))       # store each concept's band-winner conjunctive key -> bucket (id = word idx)
        self._dg_index = idx
        self._dg_codebook = cb
        self._dg_built_V = V

    def _dg_shard_select(self, rec_phases):
        """Route the recovered phasor (fractional-cycle phases) to its DG shard and decode it by the matched filter
        over ONLY the shard concepts. Returns (word, peak_score), or (None, peak) to signal ESCALATE-to-full when
        the shard is empty or its peak is not decisive (< conf_floor*D) -- the no-regression fallback. The score is
        this composer's own matched filter over fewer rows: score_w = sum_k cos(2pi(rec - code_w)) (== `_cleanup`'s
        mean-cos up to the /D constant, so argmax is unaffected)."""
        self._ensure_dg_index()
        rec = np.asarray(rec_phases, dtype=float)
        shard = self._dg_index.query(rec * (2.0 * np.pi))          # candidate word indices (the routed CA3 ensemble)
        if shard.size == 0:
            return None, 0.0
        cb = self._dg_codebook[shard]                              # (s, D) shard concept phases
        sc = np.cos(2.0 * np.pi * (rec[None, :] - cb)).sum(axis=1)  # (s,) matched-filter cleanup over the shard
        t = int(np.argmax(sc)); peak = float(sc[t])
        if peak < self._dg_conf_floor * self.D:
            return None, peak                                     # not decisive -> escalate to the full scan (parity)
        return self.words[int(shard[t])], peak

    def _full_host_select(self, rec_phases):
        """Full-codebook matched-filter argmax over the recovered phasor (the escalation path == the full scan,
        reusing the already-built `_dg_codebook` so escalation costs no extra codebook rebuild). Same operator as
        `_dg_shard_select`, over ALL V rows -- decode is IDENTICAL to the pre-port `_cleanup` full scan."""
        rec = np.asarray(rec_phases, dtype=float)
        sc = np.cos(2.0 * np.pi * (rec[None, :] - self._dg_codebook)).sum(axis=1)
        return self.words[int(np.argmax(sc))]

    def _ensure_codebook_cache(self):
        """(#66 knowledge-scale, board #192) build + cache the (V, D) cleanup codebook (fractional-cycle
        + phasor) ONCE per vocab state (rebuild only when len(self.words) changes -- the same invalidation
        rule `_dg_built_V` uses) and reuse it in the full-vocabulary cleanup paths instead of re-stacking
        the phasor matrix from the `concepts` dict on every query. The cached matrices are EXACTLY what the
        parent rebuilds each call, so decode is byte-identical by construction (independent of seed). This
        is the O(V) codebook-rebuild hot loop the 2026-08-30 finding identified (~40% of per-query time at
        V~24k, scaling with V not the shard's ~200 facts). `enable_codebook_cache=False` (default) leaves
        this method uncallable by the `_cleanup` cleanup path -> that path's rebuild-every-query behavior is
        preserved; `_escalate_role_match` (below) calls this UNCONDITIONALLY (independent of the
        `enable_codebook_cache` flag) so its winner-code gather always has the cache to index into.

        `self._concept_row` (word -> row index) is built in the SAME pass, over the SAME `self.words` order,
        so `self._concept_row[w] == i` iff `self._cb_frac[i] == self.concepts[w]` by construction -- it cannot
        drift out of alignment with the codebook, and both invalidate together on a vocab-length change."""
        V = len(self.words)
        if self._cb_cache_V == V and self._cb_frac is not None:
            return
        # fractional-cycle codebook (V, D) aligned to self.words (the single-cleanup convention)
        self._cb_frac = np.stack([np.asarray(self.concepts[w], dtype=float) for w in self.words])
        # phasor codebook (V, D) (the batched-cleanup convention)
        self._cb_z = np.exp(2j * np.pi * self._cb_frac)
        self._concept_row = {w: i for i, w in enumerate(self.words)}
        self._cb_cache_V = V

    def _cleanup(self, rec_phases, words=None):
        if self.enable_spiking_cleanup:
            return self._spiking_cleanup(rec_phases, words if words is not None else self.words)
        if self.enable_sparse_index and words is None:
            w, _pk = self._dg_shard_select(rec_phases)
            return w if w is not None else self._full_host_select(rec_phases)
        words = words if words is not None else self.words
        if self.enable_codebook_cache and words is self.words:
            self._ensure_codebook_cache()
            scores = np.cos(2.0 * np.pi * (np.asarray(rec_phases)[None, :] - self._cb_frac)).sum(axis=1)
            return self.words[int(np.argmax(scores))]
        sims = [float(np.mean(np.cos(2.0 * np.pi * (rec_phases - self.concepts[w])))) for w in words]
        return words[int(np.argmax(sims))]

    def unbind(self, composite_phases, role, words=None):
        return self._cleanup(self._unbind_phases(composite_phases, role), words)

    # --- batched query fast-path (the O(K) store-scan -> ONE launch; perf, answer-identical) ---
    def _can_batch_scan(self):
        """Batched scan applies on the numpy-kb fast path (no substrate-store) with the numpy matched-filter
        cleanup (no spiking-cleanup). Otherwise the per-fact loop is used (answer-identical either way)."""
        return bool(self.kb) and not self.enable_substrate_store and not self.enable_spiking_cleanup

    def _unbind_all_phases(self, comps, role):
        """Batched substrate unbind: unbind `role` from ALL K stored composites in ONE resonate over a
        block-diagonal bridge of K isolated 2D-blocks. Each block is an exact copy of the single `_unbind_phases`
        wiring (no cross-block coupling), so the result equals K separate unbinds EXACTLY -- but pays the 208-step
        launch overhead ONCE instead of K times (the "batch many tiny ops into one launch" fix for the O(K) query
        scan; the resonator-network matched-filter pattern, 2026-06-17-snn-vsa-gpu-optimization-literature.md).
        Returns (K, D) recovered phases."""
        K, D = len(comps), self.D
        if K == 0:
            return np.zeros((0, D))
        n = 2 * K * D
        if self.local_reciprocal_unbind:
            # per-block local rule: each 2D-block's unbind synapses derive from that block's bind synapses (no host
            # conj over the role vector) -- block i lives at offset i*2D, with the same wiring as the single unbind.
            conns = []
            for i in range(K):
                conns.extend(self._reciprocal_conjugate(self._bind_conns(self.roles[role], lo=i * 2 * D)))
        else:
            zr_conj = np.conj(self._to_phasor(self.roles[role]))              # [D] conj role phasor (host re-derivation)
            conns = [(i * 2 * D + D + k, i * 2 * D + k, zr_conj[k]) for i in range(K) for k in range(D)]
        kick = np.zeros(n, dtype=np.complex128)
        for i in range(K):
            kick[i * 2 * D:i * 2 * D + D] = self._to_phasor(comps[i])
        out = self._resonate(n, conns, kick)                                 # [n] phases
        return np.stack([out[i * 2 * D + D:i * 2 * D + 2 * D] for i in range(K)])   # (K, D)

    def _cleanup_all(self, rec, words=None):
        """Batched matched-filter cleanup (the resonator C·Cᵀ): nearest concept per row of (K, D). Returns K words.
        sims = Re(rec_phasor @ conj(codebook_phasor)ᵀ) (= the single `_cleanup`'s mean-cos up to the /D constant,
        so argmax is IDENTICAL). One matmul over the whole codebook instead of a per-word loop.

        (#66 knowledge-scale) When `enable_sparse_index` and `words is None` (the MAIN-vocabulary cleanup a routed
        ShardedPhasorStore shard pays on every query), each row routes through the DG shard instead of building the
        full (V, D) codebook matrix + a K x V matmul -- this is the O(V) term the 2026-08-28 vocab-latency-wall
        finding identified (every routed query still cleans up against the FULL shared codebook regardless of the
        shard's own ~200-fact size). A non-decisive row ESCALATES to the full-codebook scan, so the decoded word is
        IDENTICAL to the full scan whenever the shard read is ambiguous -- byte-identical decisions, not just
        usually-right ones. Escalated rows are batched into ONE matmul (`rec_z @ conj(cb_z).T`, the SAME BLAS op the
        non-indexed path below already uses), not a per-row Python loop over `_full_host_select` -- found DURING
        this port's own scale verify (2026-08-28): a naive per-row loop over real-Wikidata cues, where the DG shard
        is frequently non-decisive (many entities share thin margins at 347k-word scale), made escalation-heavy
        queries SLOWER than the pre-port full scan (each row paying its own broadcast-and-sum instead of one BLAS
        matmul over all escalated rows at once). Reuses the cached `_dg_codebook` (no rebuild)."""
        if self.enable_sparse_index and words is None:
            n = len(rec)
            if n == 0:
                return []
            rec_arr = np.asarray(rec)
            out = [None] * n
            escalate_idx = []
            for i in range(n):
                w, _pk = self._dg_shard_select(rec_arr[i])
                if w is not None:
                    out[i] = w
                else:
                    escalate_idx.append(i)
            if escalate_idx:
                # self._dg_codebook is guaranteed built (every _dg_shard_select call above ran _ensure_dg_index).
                esc_z = np.exp(2j * np.pi * rec_arr[escalate_idx])                    # (m, D)
                cb_z = np.exp(2j * np.pi * self._dg_codebook)                         # (V, D) -- built on demand,
                sims = (esc_z @ np.conj(cb_z).T).real                                 # not cached (RSS budget)
                widx = np.argmax(sims, axis=1)
                for j, i in enumerate(escalate_idx):
                    out[i] = self.words[int(widx[j])]
            return out
        words = words if words is not None else self.words
        if len(rec) == 0:
            return []
        rec_z = np.exp(2j * np.pi * np.asarray(rec))                         # (K, D)
        cb = np.stack([np.exp(2j * np.pi * self.concepts[w]) for w in words])  # (V, D)
        sims = (rec_z @ self._cleanup_conj(cb).T).real                       # (K, V); local reciprocal rule when ON, conj when OFF
        return [words[int(j)] for j in np.argmax(sims, axis=1)]

    def _scan_first_match(self, **cue_roles):
        """First stored-fact index whose cue roles ALL match (batched unbind+cleanup over the whole store), or None
        -- the batched equivalent of the per-fact match loop (first-match semantics preserved).

        With `enable_decode_escalation` (default OFF = byte-identical: the extra branch never runs), a fact whose
        stored role decoded to a different word than the cued value BUT for which the cued value is a near-tie
        runner-up is re-examined at a finer resonate period before being dropped -- the confidence-gated
        "effortful second look" that recovers a genuinely-stored fact the coarse phase readout mis-argmaxed. See
        the `enable_decode_escalation` note in __init__ (root cause: the 1/period phase-readout quantization)."""
        comps = [comp for _f, comp in self.kb]
        mask = np.ones(len(comps), dtype=bool)
        for role, val in cue_roles.items():
            rec = self._unbind_all_phases(comps, role)
            words = self._cleanup_all(rec)
            role_mask = np.fromiter((w == val for w in words), dtype=bool, count=len(words))
            if self.enable_decode_escalation:
                role_mask = self._escalate_role_match(rec, comps, role, val, words, role_mask, mask)
            mask &= role_mask
        idx = np.where(mask)[0]
        return int(idx[0]) if len(idx) else None

    def _escalate_role_match(self, rec, comps, role, val, words, role_mask, prior_mask):
        """Confidence-gated finer-period re-examination of near-tie MATCH candidates (see `_scan_first_match` +
        the `enable_decode_escalation` note). For a fact still viable on the earlier cue roles (`prior_mask`)
        whose stored `role` decoded (coarse argmax) to a word other than the cued `val`, but where `val` is a
        near-tie runner-up (winner mean-cos - `val` mean-cos <= `decode_escalate_margin`), re-unbind THAT fact's
        role at `decode_escalate_period` (a finer, longer-integrated readout) and set its match bit iff the finer
        decode now argmaxes to `val`. MOAT-SAFE: (1) an out-of-vocabulary `val` (an unknown-agent / unknown-relation
        moat cue) is never in `self.concepts`, so escalation is skipped entirely and the query abstains as before;
        (2) the finer readout converges to the ideal representation, so a fact that does not genuinely encode `val`
        is never promoted (recovers a truly-stored fact only, never manufactures a wrong match). Returns the
        (possibly-updated) role_mask; a byte-identical no-op when no candidate is a near-tie."""
        if not isinstance(val, str) or val not in self.concepts:
            return role_mask                                   # unknown/non-vocab cue -> abstain path unchanged (moat)
        cand = np.where(prior_mask & ~role_mask)[0]            # viable-so-far AND not-yet-matching this role
        if len(cand) == 0:
            return role_mask
        rec_arr = np.asarray(rec)
        val_code = self.concepts[val]
        s_val = np.cos(2.0 * np.pi * (rec_arr[cand] - val_code[None, :])).mean(axis=1)      # (m,) cued-value score
        # (2026-09-02, board #108 cupy latency) gather the m coarse-winner concept rows with ONE vectorized
        # fancy-index into the cached (V,D) codebook instead of a per-candidate Python-loop dict-lookup +
        # np.stack of individual backend arrays (the cupy-specific per-element host<->device-sync hotspot
        # pinpointed by research/findings/2026-09-02-escalation-gating-tighten-latency-correctness-safe-not-
        # the-lever.md). `_ensure_codebook_cache` builds/refreshes `self._cb_frac`/`self._concept_row` once
        # per vocab state (idempotent no-op otherwise); `row_idx` is a plain host int array (dict lookups are
        # host-side, no device traffic), so the only backend op below is the single indexed gather.
        self._ensure_codebook_cache()
        row_idx = np.fromiter((self._concept_row[words[i]] for i in cand), dtype=np.int64, count=len(cand))
        win_codes = self._cb_frac[row_idx]                                                   # VECTORIZED_WINCODE_GATHER
        s_win = np.cos(2.0 * np.pi * (rec_arr[cand] - win_codes)).mean(axis=1)               # (m,) winner score
        near = cand[(s_win - s_val) <= self.decode_escalate_margin]                          # near-tie candidates only
        for i in near:
            fine = self._unbind_phases(np.asarray(comps[i]), role, period=self.decode_escalate_period)
            if self._cleanup(fine) == val:                     # finer readout confirms the fact truly encodes val
                role_mask[i] = True
        return role_mask

    # --- (B3) READ-ONLY per-turn trace helpers (only invoked on the trace path; default OFF = byte-identical) ---
    def _cleanup_all_score_stats(self, rec, words=None):
        """Like `_cleanup_all` but ALSO returns each row's top normalized cleanup score (the decided concept's
        match confidence in [0,1]). sims = Re(rec_phasor @ conj(codebook)ᵀ)/D (the mean-cos the cleanup argmax uses);
        the score = max_j sims[i,j] (== mean cos in [-1,1], clipped to [0,1] for display). Read-only of the SAME
        matched-filter the cleanup already computes -- no new resonate, only the per-row max is extra arithmetic.
        Returns per-row dicts with the winner plus runner-up/margin conflict evidence for trace-only consumers.

        `margin_norm` (ADDED 2026-09-01, board #94 calibration-at-scale): a peak-NORMALIZED decisiveness read,
        `(top_r - runner_r) / (top_r + eps)` with both rectified to >=0 first -- the IDENTICAL formula
        `OneBrainComposer._margin(scores)` already uses ((peak-runner_up)/peak, 2026-06-18-emergent-graceful-
        degradation-derisk). ADDITIVE ONLY: `margin` (the raw, UNNORMALIZED `top_raw - runner_raw` cosine
        difference) is UNCHANGED -- `self_schema_honesty.py` and `tests/test_rf_phasor_composer.py` read that
        field directly and must stay byte-identical. WHY THIS FIELD EXISTS: `metacog_production_organ.
        mean_role_confidence` averages a `margin` field across composer types under ONE shared name, but
        `OneBrainComposer`'s own `margin` key is ALREADY the peak-normalized ratio while this composer's `margin`
        is the raw cosine difference -- two different formulas colliding under the same key. Measured directly
        (2026-09-02, the shipped `wikidata_core_15k` LTM, 80 real correct recalls through `ShardedPhasorStore`):
        the RAW `margin` field sits in [0.155, 0.275] (mean 0.216) for genuinely CORRECT, unambiguous recalls --
        entirely BELOW the metacog band's own LOW floor (`ROLE_CONF_LO=0.30`), so evidence saturates at 0 and
        `confident` can never read True on this store regardless of true recall quality. The SAME 80 recalls'
        `margin_norm` sits in [0.393, 0.552] (mean 0.473) -- squarely inside the EXISTING 0.30/0.50 band (already
        calibrated against `OneBrainComposer`'s normalized reads on the tiny-demo buffer, mrc 0.504-0.615), with
        NO band change needed: the fix is comparing like-with-like, not re-tuning the threshold per codebook size.
        A larger codebook DOES shrink the raw cosine margin (more candidate words inflate the runner-up's extreme
        value) but the PEAK-relative ratio is far less sensitive to that -- see
        research/findings/2026-09-01-confidence-forthcomingness-margin-scale-recalibration.md."""
        words = words if words is not None else self.words
        if len(rec) == 0:
            return []
        rec_z = np.exp(2j * np.pi * np.asarray(rec))                         # (K, D)
        cb = np.stack([np.exp(2j * np.pi * self.concepts[w]) for w in words])  # (V, D)
        sims = (rec_z @ self._cleanup_conj(cb).T).real / float(self.D)       # (K, V) mean-cos
        order = np.argsort(sims, axis=1)
        out = []
        for i in range(len(rec)):
            top = int(order[i, -1])
            runner = int(order[i, -2]) if len(words) > 1 else top
            top_raw = float(sims[i, top])
            runner_raw = float(sims[i, runner])
            confidence = float(np.clip(top_raw, 0.0, 1.0))
            runner_conf = float(np.clip(runner_raw, 0.0, 1.0))
            top_r = max(top_raw, 0.0)
            runner_r = max(runner_raw, 0.0)
            margin_norm = float((top_r - runner_r) / (top_r + 1e-9)) if top_r > 0.0 else 0.0
            # `margin_snr` (ADDED 2026-09-02, board #94/#108 R3 -- SCALE-INVARIANT decisiveness): the winner's
            # z-score above the NON-WINNER candidate BULK, `(top_raw - mean_nonwin) / std_nonwin`. WHY: `margin`
            # and `margin_norm` both key on the single RUNNER-UP, which is the max over the V-1 non-winner
            # candidates -- an ORDER STATISTIC that inflates as ~sqrt(2 ln V) with codebook size (extreme-value
            # of the noise floor), so `margin_norm` drifts DOWN at a larger vocab even for an equally-decisive
            # recall (measured: the identical clean `asimov_isaac employer university_of_boston` recall reads
            # margin_norm 0.497 at the 15k core vs 0.395 at the 100k bundle -- a false loss of confidence). The
            # winner-vs-BULK z-score is scale-INVARIANT because the non-winner mean (~0) and std (~1/sqrt(D),
            # D fixed) are STABLE estimators of the codebook noise floor, unaffected by adding more candidates:
            # the SAME recall reads winner_z 7.24 (15k) == 7.03 (100k). ADDITIVE ONLY -- `margin`/`margin_norm`/
            # `confidence` are byte-identical (existing readers: self_schema_honesty.py, the tests). Consumed by
            # metacog_production_organ.mean_role_confidence (preferred over margin_norm for an LTM-sourced trace).
            # See research/findings/raw/_confidence_100k_recalib/ (diagnose_margin_scale, arms_15k) +
            # research/findings/2026-09-02-confidence-forthcoming-100k-recalibration-*.md.
            row = sims[i]
            n_nonwin = max(1, len(words) - 1)
            nonwin_sum = float(row.sum()) - top_raw
            nonwin_sumsq = float(np.dot(row, row)) - top_raw * top_raw
            nonwin_mean = nonwin_sum / n_nonwin
            nonwin_var = max(0.0, nonwin_sumsq / n_nonwin - nonwin_mean * nonwin_mean)
            nonwin_std = float(np.sqrt(nonwin_var))
            margin_snr = float((top_raw - nonwin_mean) / (nonwin_std + 1e-9)) if nonwin_std > 0.0 else 0.0
            # `margin_spiking` (ADDED 2026-09-05, scaffold-retirement backlog rank 9, opt-in via
            # `self.spiking_recall_margin` / BRAIN_METACOG_SPIKING_MARGIN, DEFAULT None = byte-identical): the
            # SAME winner-vs-runner-up decisiveness read as `margin_norm`, but off the recall circuit's OWN
            # Izhikevich WTA spike counts (`_spiking_margin`) instead of a host comparison of the `sims` row.
            # None when the flag is off -- `mean_role_confidence` falls through to `margin_snr`/`margin_norm`/
            # `margin` unchanged. Gated (not unconditional) because it runs a real Stage-2 spiking competition per
            # row -- cheap at the tiny-demo's ~15-40 word vocab this is de-risked at, uncharacterized at LTM scale
            # (research/findings/2026-09-05-metacog-spiking-recall-margin-derisk*.md names the residual).
            margin_spiking = self._spiking_margin(row) if self.spiking_recall_margin else None
            out.append({
                "word": words[top],
                "confidence": confidence,
                "winner_score_raw": top_raw,
                "runner_word": words[runner],
                "runner_confidence": runner_conf,
                "runner_score_raw": runner_raw,
                "margin": float(top_raw - runner_raw),
                "margin_norm": margin_norm,
                "margin_snr": margin_snr,
                "margin_spiking": margin_spiking,
                "conflict": float(runner_conf / (confidence + runner_conf + 1e-9)),
            })
        return out

    def _cleanup_all_scored(self, rec, words=None):
        stats = self._cleanup_all_score_stats(rec, words=words)
        decoded = [s["word"] for s in stats]
        scores = [s["confidence"] for s in stats]
        return decoded, scores

    def _rf_gauge(self):
        """A scalar RF activity gauge read from the LAST `_resonate` bridge (cached by neuron count): the fraction of
        readout neurons that crossed (`cp_rf_fired.mean()`) + the mean recovery magnitude |Z| = mean(sqrt(re²+im²))
        over `cp_membrane_potential_v`(re) / `cp_recovery_variable_u`(im). All guarded: the rf slice / arrays may be
        absent -> the field is None. Strictly read-only of state the resonate already produced (no extra GPU work)."""
        n = self._last_resonate_n
        b = self._bridge_cache.get(n) if n is not None else None
        if b is None:
            return {"n_readout_neurons": None, "frac_fired": None, "mean_magnitude": None}
        out = {"n_readout_neurons": int(n) if n is not None else None,
               "frac_fired": None, "mean_magnitude": None}
        try:
            fired = getattr(b, "cp_rf_fired", None)
            if fired is not None:
                out["frac_fired"] = float(np.asarray(to_host(fired)).astype(float).mean())
        except Exception:
            pass
        try:
            re = getattr(b, "cp_membrane_potential_v", None)
            im = getattr(b, "cp_recovery_variable_u", None)
            if re is not None and im is not None:
                re_h = np.asarray(to_host(re)).astype(float)
                im_h = np.asarray(to_host(im)).astype(float)
                out["mean_magnitude"] = float(np.sqrt(re_h * re_h + im_h * im_h).mean())
        except Exception:
            pass
        return out

    def _trace_scan(self, cue_roles, idx, answer_roles):
        """Build self.last_trace for the LAST query (read-only). `cue_roles` = the {role: asserted_value} the scan
        matched on; `idx` = the matched fact-block index (or None = abstain); `answer_roles` = the read-out
        {role: (decoded_word, confidence)} for the roles the query DECODED (agent/action/patient, etc.). Records the
        per-role chips, which engram block matched + how many were scanned, and the post-resonate RF gauge. On an
        abstain (idx is None) it records matched_fact_index=None + scanned=N WITHOUT inventing an answer (the moat
        made visible). Stored on self.last_trace; never affects the return value."""
        if not self.trace:
            return
        comps = [comp for _f, comp in self.kb]
        n_scanned = len(comps)
        roles_out = []
        # the cue roles first (what the question asserted -> their decoded match over the matched block, if any)
        cue_decoded = {}
        if idx is not None:
            comp = comps[idx]
            for role in cue_roles:
                rec = self._unbind_phases(comp, role)
                stats = self._cleanup_all_score_stats(np.asarray(rec)[None, :])
                cue_decoded[role] = stats[0] if stats else {"word": None, "confidence": None}
        for role, asserted in cue_roles.items():
            stats = dict(cue_decoded.get(role, {"word": asserted, "confidence": None}))
            stats.update({"role": role, "cue": True, "asserted": asserted})
            roles_out.append(stats)
        for role, payload in (answer_roles or {}).items():
            if isinstance(payload, dict):
                stats = dict(payload)
            else:
                word, conf = payload
                stats = {"word": word, "confidence": conf}
            stats.update({"role": role, "cue": False})
            roles_out.append(stats)
        self.last_trace = {
            "roles": roles_out,
            "matched_fact_index": (int(idx) if idx is not None else None),
            "n_facts_scanned": int(n_scanned),
            "abstained": idx is None,
            "rf": self._rf_gauge(),
            "composer": "rf",
        }
        if idx is not None:
            self.last_trace["source_fact"] = dict(self.kb[int(idx)][0])

    # --- conversational API (mirrors CoreSimComposer; the no-confab moat preserved) ---
    def store(self, agent, action, patient, polarity=None):
        fact = {"agent": agent, "action": action}
        if _is_clause(patient):                    # a recursive clause filler (check BEFORE tuple: a Clause IS a tuple)
            fact["patient"] = patient
        elif isinstance(patient, tuple):           # (adj(s), noun) -- an attributed entity ('big apple' or 'big hot apple')
            adjs, noun = patient
            adjs = list(adjs) if isinstance(adjs, (tuple, list)) else [adjs]
            fact["patient"] = noun
            fact["attribute"] = adjs[0]
            if len(adjs) > 1:
                fact["attribute2"] = adjs[1]       # 2-attribute (the +-1 scheme's K=5 boundary -- does FHRR lift it?)
        else:
            fact["patient"] = patient
        if polarity is not None:
            fact["polarity"] = polarity      # a bound AFFIRM/NEGATE tag (extra binding -> more load)
        comp = self._encode(fact)
        self.kb.append((fact, self._store_substrate(comp) if self.enable_substrate_store else comp))
        self._source_store_echo(fact)

    # --- reconsolidation: prediction-error-gated in-place fact update (Option A; additive, store/query unchanged) ---
    def _find_cued_fact(self, agent, action):
        """Reactivation: the FIRST stored fact whose CUE roles (agent+action) match, by the substrate unbind +
        cleanup. Returns (kb_index, fact, composite) or None (no trace to reactivate -> abstain)."""
        for i, (fact, handle) in enumerate(self.kb):
            comp = self._retrieve_substrate(handle) if self.enable_substrate_store else handle
            if self.unbind(comp, "agent") == agent and self.unbind(comp, "action") == action:
                return i, fact, comp
        return None

    def _patient_prediction_error(self, comp, patient_word):
        """PE = 1 - phase-cos(recovered patient phasor, the asserted patient's code). ~0 when the asserted filler
        matches the stored one (a re-statement); ~1 on a mismatch (a correction)."""
        rec = self._unbind_phases(comp, "patient")
        return 1.0 - float(np.mean(np.cos(2.0 * np.pi * (rec - self.concepts[patient_word]))))

    def _calibrate_pe_labile(self):
        """Frozen labilization gate = the midpoint of the measured same-vs-different prediction-error distributions
        over the CURRENT facts (each fact's PE against its OWN stored patient = 'same'; against other facts'
        patients = 'different'). The data's own separation point -- NOT tuned to a downstream probe (the
        calibrate_threshold rule). 0.5 fallback when too few distinct facts exist to calibrate."""
        facts = []
        for fact, handle in self.kb:
            p = fact.get("patient")
            if isinstance(p, str):
                comp = self._retrieve_substrate(handle) if self.enable_substrate_store else handle
                facts.append((comp, p))
        same, diff = [], []
        for comp, p in facts:
            same.append(self._patient_prediction_error(comp, p))
            for _comp2, p2 in facts:
                if p2 != p:
                    diff.append(self._patient_prediction_error(comp, p2))
        if not same or not diff:
            return 0.5
        return 0.5 * (float(np.mean(same)) + float(np.mean(diff)))

    def update_on_mismatch(self, agent, action, new_patient, pe_labile=None):
        """RECONSOLIDATION: a corrective utterance ('actually, <agent> <action> <new_patient>') reactivates the
        cued fact and -- ONLY if the new filler carries a prediction error above the labilization gate -- rewrites
        that fact's patient IN PLACE (no contradictory duplicate). A fully-predicted re-statement re-stabilizes
        unchanged; a NEVER-stored cue ABSTAINS (the no-confab moat: a reactivated trace is updated, a missing one
        is not fabricated). ADDITIVE -- store()/query_*() are unchanged, so any caller that never invokes this
        keeps the append-only path byte-for-byte; the agent-level opt-in is where 'default-off' lives.

        pe_labile=None -> auto-calibrate the gate from the current facts (the validated midpoint rule); else use
        the supplied gate. Returns {action: abstain|rewrite|restabilize, wrote: bool, pe: float|None}. Nader 2000;
        Osan-Tort-Amaral 2011 mismatch-gated attractor update; Sevenster 2013 prediction-error necessity. De-risked
        6/6 multi-seed: research/findings/2026-06-17-reconsolidation-update-derisk-GO.md."""
        found = self._find_cued_fact(agent, action)
        if found is None:
            return {"action": "abstain", "wrote": False, "pe": None}     # no trace -> no update, no fabrication
        idx, fact, comp = found
        gate = self._calibrate_pe_labile() if pe_labile is None else float(pe_labile)
        pe = self._patient_prediction_error(comp, new_patient)
        if pe >= gate:
            f2 = dict(fact); f2["patient"] = new_patient
            comp2 = self._encode(f2)
            self.kb[idx] = (f2, self._store_substrate(comp2) if self.enable_substrate_store else comp2)
            return {"action": "rewrite", "wrote": True, "pe": pe}
        return {"action": "restabilize", "wrote": False, "pe": pe}        # PE below the gate -> re-stabilize

    def count_facts(self, agent, action):
        """Number of stored facts whose cue roles (agent+action) match -- 1 after a reconsolidation update, 2 if a
        correction was naively appended. Used by the reconsolidation tests + the correction-turn hook."""
        return sum(1 for fact, handle in self.kb
                   if self.unbind(self._retrieve_substrate(handle) if self.enable_substrate_store else handle,
                                  "agent") == agent
                   and self.unbind(self._retrieve_substrate(handle) if self.enable_substrate_store else handle,
                                   "action") == action)

    def _store_substrate(self, comp_phases):
        """Hold the bound composite in the SUBSTRATE: a persistent (1+D) RF bridge whose trigger(neuron 0) ->
        readout(1..D) complex weights carry the composite phasor. The composite lives in the synaptic weights
        (cp_rf_w_re/im), NOT a numpy array -- the Crawford-Eliasmith weight-store (Hebb memory-in-weights). The kb
        holds this bridge handle, not the composite. Validated == numpy store at parity (Phase-2 de-risk GO)."""
        D = self.D
        zc = self._to_phasor(comp_phases)
        # (Tier-2 #6) DA-gated encoding strength: scale the stored composite magnitude by the per-fact gain `g` read
        # from the dopamine signal at store time. g=1.0 (encoding_gain_fn=None) -> the byte-identical unit-mag write.
        g = 1.0 if self.encoding_gain_fn is None else float(self.encoding_gain_fn())
        conns = [(1 + k, 0, complex(g) * zc[k]) for k in range(D)]
        b = _build_rf_bridge(1 + D, self.seed)
        b.rf_set_complex_weights(conns)
        return b

    def _retrieve_substrate(self, b):
        """Read a substrate-held composite back: fire the trigger (unit phasor) -> the readout neurons reconstruct
        the composite IN PHASE (the magnitude-invariant RF phase readout)."""
        D = self.D
        kick = np.zeros(1 + D, dtype=np.complex128)
        kick[0] = complex(self._retrieve_kick_mag)   # common read damage: scale the trigger (default 1.0 = unchanged)
        b.rf_kick(kick, period=self.period, lam=self._retrieve_lam, floor=self._retrieve_floor)
        b.rf_resonate_steps(self.period + 8)
        phases = np.asarray(b.rf_read_phases())[1:1 + D]
        if self._retrieve_noise > 0.0:
            # The readout SIGNAL magnitude (|Z| of the readout neurons after the resonate) -- THIS is what the
            # encoding gain controls (g*M). Add common complex read noise of fixed sigma, apply the read floor to the
            # NOISY phasor; sub-floor neurons read garbage (phase 0). Higher gain -> higher SNR -> survives.
            re = np.asarray(to_host(b.cp_membrane_potential_v))[1:1 + D]
            im = np.asarray(to_host(b.cp_recovery_variable_u))[1:1 + D]
            sig_mag = np.sqrt(re * re + im * im)
            mag = float(np.median(sig_mag)) if np.any(sig_mag > 0) else 1.0   # the per-fact readout magnitude ~ g*M
            z = mag * np.exp(2j * np.pi * phases)
            eta = self._retrieve_noise * (self._retrieve_noise_rng.standard_normal(D)
                                          + 1j * self._retrieve_noise_rng.standard_normal(D))
            zn = z + eta
            phases = (np.angle(zn) / (2.0 * np.pi)) % 1.0
            phases = np.where(np.abs(zn) < self._retrieve_read_floor, 0.0, phases)   # sub-floor -> garbage phase
        return phases

    def _iter_facts(self):
        """Yield (fact_dict, composite_phases) per stored fact. With the substrate store, the composite is read back
        from its substrate weight-bridge (fire the trigger); else it's the numpy array in kb. Lazy -> an early-return
        query only retrieves the facts it actually checks."""
        for fact, handle in self.kb:
            yield fact, (self._retrieve_substrate(handle) if self.enable_substrate_store else handle)

    def query_agent(self, action, patient):
        """'who <action> <patient>?' -> the agent of the matching fact; None if no fact matches (abstention).
        Batched store scan on the fast path (answer-identical to the per-fact loop)."""
        if self.trace:
            self.last_trace = None
        if self._can_batch_scan():
            i = self._scan_first_match(action=action, patient=patient)
            ans = self.unbind(self.kb[i][1], "agent") if i is not None else None
            if self.trace:
                ans_roles = {}
                if i is not None:
                    rec = self._unbind_phases(self.kb[i][1], "agent")
                    stats = self._cleanup_all_score_stats(np.asarray(rec)[None, :])
                    ans_roles = {"agent": stats[0] if stats else {"word": ans, "confidence": None}}
                self._trace_scan({"action": action, "patient": patient}, i,
                                 ans_roles)
            return ans
        for fact, comp in self._iter_facts():
            if self.unbind(comp, "action") == action and self.unbind(comp, "patient") == patient:
                return self.unbind(comp, "agent")
        return None

    def query_patient(self, agent, action, order_fn=None):
        """'what does <agent> <action>?' -> the patient of the matching fact (an attributed entity 'big apple' if
        the fact bound an ATTRIBUTE); None if no match (abstention). The stored structure only routes the rendering;
        the words are decoded from the RF unbind. `order_fn` (opt-in, default None = host f-string): when set, an
        inner CLAUSE patient's SVO order is produced by the de-risked spiking serial-order generator. The moat is
        unaffected: abstention (return None) happens BEFORE any rendering. The store scan is BATCHED on the fast
        path (one resonate over all facts; answer-identical to the per-fact loop below)."""
        if self.trace:
            self.last_trace = None
        if self._can_batch_scan():
            i = self._scan_first_match(agent=agent, action=action)
            if i is None:
                if self.trace:
                    self._trace_scan({"agent": agent, "action": action}, None, {})
                return None
            fact, comp = self.kb[i]
            noun = self._render(comp, "patient", fact["patient"], order_fn=order_fn)
            adjs = [self.unbind(comp, r) for r in ("attribute", "attribute2") if r in fact]
            ans = " ".join(adjs + [noun]) if adjs else noun
            if self.trace:
                # the patient role's decoded word + confidence (read-only over the matched block)
                rec = self._unbind_phases(comp, "patient")
                stats = self._cleanup_all_score_stats(np.asarray(rec)[None, :])
                ans_roles = {"patient": stats[0] if stats else {"word": noun, "confidence": None}}
                for r in ("attribute", "attribute2"):
                    if r in fact:
                        rr = self._unbind_phases(comp, r)
                        st = self._cleanup_all_score_stats(np.asarray(rr)[None, :])
                        ans_roles[r] = st[0] if st else {"word": None, "confidence": None}
                self._trace_scan({"agent": agent, "action": action}, i, ans_roles)
            return ans
        for fact, comp in self._iter_facts():
            if self.unbind(comp, "agent") == agent and self.unbind(comp, "action") == action:
                noun = self._render(comp, "patient", fact["patient"], order_fn=order_fn)   # word OR recursive Clause
                adjs = [self.unbind(comp, r) for r in ("attribute", "attribute2") if r in fact]
                if adjs:
                    return " ".join(adjs + [noun])    # 'big apple' / 'big hot apple'
                return noun
        return None

    def query_chain(self, cue, actions):
        """Multi-hop relational reasoning: follow a chain of stored facts. Each hop matches the current concept as
        the AGENT under the hop's action and reads the PATIENT, which becomes the next hop's cue --
        query_chain('dog', ['eat', 'eat']) over {dog eat cat, cat eat mouse} -> 'mouse'. Returns the terminal
        concept, or None (abstain) the moment any hop has no matching fact -- so the no-confab moat holds at EVERY
        hop and a broken or over-run chain never confabulates. The cleanup re-discretizes the intermediate concept
        each hop, so retrieval error does NOT compound across hops. De-risked GO 3 seeds x 3 D (controls -- leaky
        spreading, permuted-relation, between-hop re-cue lesion -- all collapse): 2026-06-17-multihop-query-chain-GO.md."""
        x = cue
        for action in actions:
            x = self.query_patient(x, action)
            if x is None:
                return None
        return x

    # --- Tier 2.2: SELF-CUED associative chain-of-thought -------------------------------------------------------
    def _relation_assoc(self):
        """The agent's OWN learned RELATION-KEYED association strengths, derived from its stored facts:
        assoc[(agent, action)] = how strongly that (agent, relation) pair was reinforced (a co-occurrence count
        over the kb -- the same statistic the dialogue-planning `_assoc_graph` reads, but keyed by the RELATION so
        it can pick WHICH relation to chase). Built lazily from `self.kb` each call (the kb is the source of
        truth). Clause-patient facts contribute their (agent, action) too (the inner SVO is structural)."""
        assoc = {}
        for fact, _ in self.kb:
            a, act = fact.get("agent"), fact.get("action")
            if isinstance(a, str) and isinstance(act, str):
                assoc[(a, act)] = assoc.get((a, act), 0.0) + 1.0
        return assoc

    def _select_next_relation(self, x, assoc, lesion=None, lesion_rng=None):
        """SELF-CUE: among the relations available from concept `x` (as agent in some stored fact), pick the one
        with the HIGHEST learned association strength. Returns the relation, or None (no associate -> a dead end ->
        the moat abstains; no fabricated hop). lesion='zero' removes the learned signal (-> None, abstain);
        lesion='scramble' randomizes the ordering (the anti-cheat controls)."""
        cands = {rel: w for (a, rel), w in assoc.items() if a == x}
        if not cands:
            return None
        if lesion == "zero":
            return None
        if lesion == "scramble":
            cands = {rel: float(lesion_rng.random()) for rel in cands}
        # deterministic tie-break (sorted) so a given fact set yields a reproducible chain
        return max(sorted(cands), key=cands.get)

    def chain_of_thought(self, start, goal=None, max_hops=4, lesion=None, lesion_rng=None, return_path=False):
        """SELF-CUED associative chain-of-thought (Tier 2.2): the structural heart of 'thinking'. From `start`, at
        each step the AGENT itself SELECTS the next relation to chase -- by LEARNED association strength over its
        own stored facts (`_relation_assoc`), NOT a caller-supplied plan -- then chases it via the validated single
        hop (`query_patient`: match the current concept as AGENT under the chosen relation, read the PATIENT). The
        cleanup re-discretizes the intermediate concept each hop, so error does NOT compound. Stops at `goal` (if
        given and reached) or a dead end (no associate / no matching fact -> abstain).

        This is exactly `query_chain` with the agent CHOOSING each hop instead of the caller supplying the action
        list -- the single change that turns retrieval into self-generated thought (front-3 §2.6; roadmap 2.2).

        The no-confab MOAT holds at EVERY hop: a dead-end concept abstains rather than fabricating a hop; an
        unstored start returns None. De-risked GO (numpy, 3 seeds x 3 D): self-cued 2-hop 1.00 vs spreading floor
        0.08, lesion-the-association -> 0.00, permuted-graph -> 0.08, re-cue lesion -> 0.00, moat at every hop, no
        compounding to 4 hops -- 2026-06-27-tier2.2-chain-of-thought-GO.md.

        Returns the terminal concept (or None if it dead-ended before any hop / start unstored). With
        return_path=True, returns (terminal_or_None, [start, ...visited concepts]).
        `lesion` ('zero'|'scramble') + `lesion_rng` are the anti-cheat hooks (default None = the real chain)."""
        assoc = self._relation_assoc()
        x = start
        path = [x]
        terminal = None
        for _ in range(int(max_hops)):
            rel = self._select_next_relation(x, assoc, lesion=lesion, lesion_rng=lesion_rng)
            if rel is None:                                       # dead end -> abstain (no fabricated hop)
                break
            nxt = self.query_patient(x, rel)                      # the VALIDATED role-structured single hop + moat
            if nxt is None:
                break
            path.append(nxt)
            x = nxt
            terminal = x
            if goal is not None and x == goal:
                break
        return (terminal, path) if return_path else terminal

    def ask_yes_no(self, agent, action, patient):
        """'does <agent> <action> <patient>?' -> 'yes'/'no'/'unknown' via the bound AFFIRM/NEGATE polarity tag.
        Matches the full SVO; 'unknown' (abstention) when no stored fact matches. Batched scan on the fast path."""
        if self.trace:
            self.last_trace = None
        if self._can_batch_scan():
            i = self._scan_first_match(agent=agent, action=action, patient=patient)
            if i is None:
                if self.trace:
                    self._trace_scan({"agent": agent, "action": action, "patient": patient}, None, {})
                return "unknown"
            pol = self.unbind(self.kb[i][1], "polarity", self.pol_words)
            if self.trace:
                rec = self._unbind_phases(self.kb[i][1], "polarity")
                stats = self._cleanup_all_score_stats(np.asarray(rec)[None, :], words=self.pol_words)
                self._trace_scan({"agent": agent, "action": action, "patient": patient}, i,
                                 {"polarity": stats[0] if stats else {"word": pol, "confidence": None}})
            return "yes" if pol == "AFFIRM" else "no"
        for fact, comp in self._iter_facts():
            if (self.unbind(comp, "agent") == agent and self.unbind(comp, "action") == action
                    and self.unbind(comp, "patient") == patient):
                return "yes" if self.unbind(comp, "polarity", self.pol_words) == "AFFIRM" else "no"
        return "unknown"

    def render_fact(self, agent, order_fn=None):
        """Generation: render a full stored sentence whose agent matches `agent` -- e.g. 'dog go north' (an
        attributed patient 'big apple' or a nested clause renders too). The action + patient are DECODED from the
        RF unbind (not the stored labels); None if no fact's agent matches (the no-confab moat -- no invented
        sentence about an unknown subject).

        `order_fn` (opt-in, default None = the host f-string): a callable n -> a permutation of range(n) that
        produces the word ORDER. When set, the slot order comes from the de-risked spiking competitive-queuing
        serial-order generator (NeuralSerialOrderRenderer) instead of the host literal -- the cognitive ordering
        is then neural; only the final join (the body's emission) is host. The moat is unaffected: abstention
        (return None) happens BEFORE any ordering."""
        for fact, comp in self._iter_facts():
            if self.unbind(comp, "agent") == agent:
                ac = self.unbind(comp, "action")
                pt = self._render(comp, "patient", fact["patient"], order_fn=order_fn)   # inner clause neural too
                adjs = [self.unbind(comp, r) for r in ("attribute", "attribute2") if r in fact]
                if adjs:
                    pt = " ".join(adjs + [pt])
                words = [agent, ac, pt]
                if order_fn is not None:
                    return " ".join(words[i] for i in order_fn(len(words)))   # neural serial-order (outer SVO)
                return f"{agent} {ac} {pt}"
        return None

    # --- dialogue planning (the dlPFC content-selection Control; architecture-independent: operates on the graph) ---
    def _assoc_graph(self):
        """An association graph (concept -> {concept: weight}) from the stored facts (agent/action/patient co-occur;
        clause patients are skipped -- their inner concepts are structural). The graph the dlPFC spreads over."""
        graph = {}
        for fact, _ in self.kb:
            cs = [fact.get(r) for r in ("agent", "action", "patient") if isinstance(fact.get(r), str)]
            for x in cs:
                for y in cs:
                    if x != y:
                        graph.setdefault(x, {})[y] = graph.get(x, {}).get(y, 0.0) + 1.0
        return graph

    def elaborate(self, topic):
        """Dialogue planning: the next on-topic concept about `topic`, chosen by the dlPFC spiking content-selection
        Control (loop-attractor working memory + spreading activation) over the agent's own association graph -- the
        same validated SpikingSpreadingController the rate-coded agent uses (it operates on the GRAPH, so it is
        substrate-independent). None if `topic` is unconnected."""
        from research.runners.content_selection_spiking import SpikingSpreadingController
        graph = self._assoc_graph()
        if topic not in graph:
            return None
        key = tuple(sorted((k, tuple(sorted(v.items()))) for k, v in graph.items()))
        if self._dlpfc is None or self._dlpfc_key != key:
            self._dlpfc = SpikingSpreadingController(graph, seed=self.seed)
            self._dlpfc_key = key
        return self._dlpfc.turn_latency([topic])
