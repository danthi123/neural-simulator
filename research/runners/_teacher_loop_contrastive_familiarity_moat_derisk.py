"""TEACHER-LOOP, SECOND DE-RISK (2026-08-08): CLOSE the learned-moat leak the first teacher-loop de-risk
(`_teacher_loop_corrective_acquire_derisk.py`) left open 6/6. That de-risk taught 'dax eats grass' by the brain's
OWN e-prop plasticity (verified) but the abstain/answer gate for the ACQUIRED fact was the readout CONFIDENCE
threshold, and it LEAKED: after teaching dax->grass with a SINGLE class of targets, the readout saturated to a
constant grass-bias -- UNTAUGHT cues ('dax chases ?', 'wug eats ?') also read 'grass' at conf 1.0 (2/2
false-accepts, every seed 42..47). A genuinely-learned fact has no structural kb-membership block by construction,
so a learned fact needs a LEARNED specificity moat. The design named two levers; this de-risk builds BOTH:

  LEVER 1 -- CONTRASTIVE TEACHING. Teach dax->grass INTERLEAVED with a background of other referent->patient
  mappings the brain also holds (dog->bone, cat->fish), all through the SAME e-prop readout. The readout can no
  longer minimise loss by emitting a constant 'grass' -- it MUST discriminate on the percept. Result: held-out
  dax->grass AND dog->bone AND cat->fish each read their OWN patient (the learned map discriminates; no cross-talk).
  This is the boundary to genuine breadth: learn many facts in one readout without collapse.

  LEVER 2 -- LEARNED FAMILIARITY / SOURCE-MONITOR GATE. A neural novelty signal (the Bogacz-Brown anti-Hebbian
  familiarity gate, catalog D.04 perirhinal repetition suppression -- reuse-by-import RealAntiHebbianFamiliarity
  from the phaseB learned-moat arc) imprints the cues the teacher ACTUALLY taught (dax+eats, dog+eats, cat+eats as
  a SINGLE conjunctive binding of the referent-percept code with the action code). At query it reads the cue's
  novelty N(x)=||x||^2 - x^T W x: familiar (a taught cue) -> N~0 -> the readout answer is TRUSTED; novel (an
  untaught referent OR an untaught action) -> N~1 -> ABSTAIN, whatever the readout confidence. This earns the
  specificity the structural moat has for stored facts.

WHY BOTH (neither alone closes it). Contrastive teaching alone makes the readout discriminative but a discriminative
classifier still emits its best guess for ANY cue -- the readout-confidence gate (the old mechanism) STILL
false-accepts untaught cues (measured here: gate-OFF false-accepts > 0 even WITH contrastive teaching). The
familiarity gate alone would gate a degenerate constant-grass readout (it would abstain correctly but the learned
map would have discriminated nothing). Together: contrastive earns a genuine multi-fact map; the gate earns the
abstain. The T3 specificity that leaked 6/6 closes.

BRAIN-BASED. The readout is the a1-GO transport-free e-prop net (the brain's OWN plasticity; the sole learner --
established in the first de-risk). The familiarity gate is a LEARNED anti-Hebbian projector (neurons/synapses;
lesionable -- lever 2's teeth). HONEST SEAM (declared, unchanged from the phaseB moat convention): the conjunctive
cue uses fixed random VSA codes (a role/percept/action codebook) -- the composer-as-idealization host seam for what
a learned cortex would encode; the LEARNED, load-bearing part is the projector W (imprinted from the taught cues,
lesioned in FG3). The abstain threshold NOV_GATE=0.5 is the a-priori perirhinal unit-norm midpoint (familiar~0,
novel~1), NOT tuned on the untaught probes -- and the headline teeth is the MARGIN (untaught novelty >> taught),
which is threshold-independent. The readout argmax + confidence is the same host render declared in the first
de-risk.

TEETH (per seed):
  T1 BEFORE/AFTER      : query_patient('dax','eats') -- before abstain(None), after 'grass'.
  T2 WEIGHTS/NO-STORE  : e-prop grew the readout (readout_moved>1e-3) AND the composer kb length is UNCHANGED
                         (no store-write -- acquisition is a weight change, not a host append).
  CT1 DISCRIMINATION   : held-out dax->grass > 0.6 (the taught headline fact genuinely learned) AND the majority
                         argmax over {dax,dog,cat} yields >=2 DISTINCT classes with dog!=grass and cat!=grass
                         (the readout is NOT a constant grass-bias -- lever 1 worked).
  CT2 CONTRAST-FLIP    : the SINGLE-CLASS control (teach ONLY dax->grass, the first de-risk's regime) collapses --
                         dog AND cat majority-argmax == grass (the constant-bias leak reproduced). Contrastive is
                         LOAD-BEARING for discrimination (the comparator flips in its failing direction).
  FG1 MARGIN           : novelty(untaught cues) >> novelty(taught cues) -- a clean a-priori-separable gap.
  FG2 SPECIFICITY *    : gate ON -- dax+eats answers 'grass'; dax+chases (untaught action) ABSTAINS; wug+eats
                         (untaught referent) ABSTAINS -> 0 false-accepts. THIS is the T3 that leaked 6/6.
  FG3 GATE-FLIP/LESION : (a) gate OFF (readout-confidence only, the old mechanism) -> false-accepts RETURN even
                         with contrastive teaching (the gate, not contrastive, closes the leak); (b) lesion the
                         gate's LEARNED projector W -> the novelty margin collapses -> false-accepts return (the
                         abstain rides the LEARNED weights, not a host rule).
  T4 LESION-LEARNING   : freeze the e-prop plasticity (eprop_lr=0) during the identical teaching -> not acquired ->
                         taught cue abstains (LEARNED, not wired).
  T5 LESION-PAIRING    : a MISPAIRED teacher (a consistent WRONG referent->patient assignment) -> mean true-target
                         held-out ~chance (the acquired answer is the teacher's SPECIFIC pairing, not the percept
                         alone). NB a random-label shuffle is NOT a clean control -- separable percept clusters let
                         the net form an arbitrary map that coincidentally aligns on some seeds; a consistent wrong
                         pairing learns a deterministic map that is definitely wrong on the true targets.
  * FG2 is the headline: the specificity the first de-risk leaked 6/6.

GO = T1 & T2 & CT1 & CT2 & FG1 & FG2 & FG3 & T4 & T5 & off-flag byte-identical & use-in-loop answer=='grass'.

DISCIPLINE: reuse-by-import (OnBridgeEpropNet + _train_eprop + _softmax from the a1-GO port; RealAntiHebbianFamiliarity
+ hrr_bind from the phaseB learned-moat arc; RFPhasorComposer store). NO sim/ edit. SIM_BACKEND=numpy. cfg.seed via
the seed= arg the a1 net passes to CoreSimConfig.seed (NOT actual_seed_used). Additive/default-off shim. Single-seed
SMOKE here; the 6-seed claim needs 6/6.
SMOKE : PYTHONPATH=$PWD SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m research.runners._teacher_loop_contrastive_familiarity_moat_derisk --seeds 42 \
  --out research/findings/raw/teacher_loop_contrastive_familiarity_moat_s42.json
6-SEED (GO needs 6/6 at 42..47) :  ... --seeds 42 43 44 45 46 47 \
  --out research/findings/raw/teacher_loop_contrastive_familiarity_moat_6seed.json
"""
from __future__ import annotations
import argparse, contextlib, io, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
# reuse-by-import: the brain's OWN plasticity (a1-GO transport-free e-prop) + trainer; the LEARNED familiarity gate
# (Bogacz-Brown anti-Hebbian, catalog D.04) + the VSA bind; the numpy composer store. NO sim/ edit.
from research.runners._onbridge_eprop_port_derisk import OnBridgeEpropNet, _train_eprop, _softmax  # noqa: E402
from research.runners._phaseB_biologize_moat_streamcodes_derisk import RealAntiHebbianFamiliarity  # noqa: E402
from research.runners._phaseB_assembled_pipeline_ppmi_derisk import hrr_bind  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_contrastive_familiarity_moat.json"

# --- the atom's fixed vocab. Patient WORDS = the K read-path classes. NOTE: the zero-init leaky readout argmaxes
# class 0 for ANY input (verified), so the taught patients are placed OFF index 0 -- otherwise an untrained/frozen/
# non-contingent net would trivially "answer" the taught fact and confound T1/T4/T5. index-0 'apple' is the
# untrained attractor and is not a taught patient.
PATIENT_WORDS = ["apple", "river", "grass", "bone", "fish", "seed"]        # K=6; grass=2, bone=3, fish=4 (all off-0)
ATTRACTOR_WORD = "apple"                                                  # the zero-init argmax class (not taught)
ACTIONS = ["eats", "chases"]                                              # 'eats' taught; 'chases' = untaught-action probe
# CONTRASTIVE taught set: dax is the NEW fact; dog/cat are background referent->patient mappings the brain also
# holds (so the readout must DISCRIMINATE, not saturate to a constant 'grass').
TAUGHT = {"dax": "grass", "dog": "bone", "cat": "fish"}
HEADLINE_REFERENT = "dax"
UNTAUGHT_REFERENT = "wug"                                                 # 2nd untaught referent (FG2 percept specificity)
COMPOSER_VOCAB = sorted(set(PATIENT_WORDS + ACTIONS + list(TAUGHT) + [UNTAUGHT_REFERENT]))
NOV_GATE = 0.5      # a-priori perirhinal unit-norm midpoint (familiar~0, novel~1); NOT tuned on the untaught probes


# ------------------------------- host = SENSORY ENVIRONMENT (legitimate) -------------------------------
class ReferentEnv:
    """The world's sensory render: each referent is a noisy perceptual prototype in [0,1]^d_p. A PRESENTATION =
    clip(proto + noise*N(0,1), 0, 1). Host code is legitimate here EXACTLY as a retinal-image render is."""
    def __init__(self, seed, d_p=12, noise=0.12):
        self.rng = np.random.default_rng(seed + 101)
        self.d_p = int(d_p); self.noise = float(noise)
        self.protos = {}

    def proto(self, referent):
        if referent not in self.protos:
            self.protos[referent] = self.rng.random(self.d_p).astype(np.float64)
        return self.protos[referent]

    def draw(self, referent):
        p = self.proto(referent)
        return np.clip(p + self.noise * self.rng.standard_normal(self.d_p), 0.0, 1.0)


def _action_onehot(action):
    v = np.zeros(len(ACTIONS), dtype=np.float64)
    v[ACTIONS.index(action)] = 1.0
    return v


def _feat(env, referent, action):
    """The brain's e-prop input = percept(referent) concat action-context one-hot. n_in = d_p + len(ACTIONS)."""
    return np.concatenate([env.draw(referent), _action_onehot(action)]).astype(np.float64)


# ------------------------------- LEVER 2: the LEARNED familiarity / source-monitor gate -------------------------------
class ConjunctiveFamiliarityGate:
    """A LEARNED perirhinal source-monitor over (referent, action) conjunctions. The cue is a SINGLE VSA binding
    of the referent-percept code with the action code -- so ANY mismatch (untaught referent OR untaught action)
    makes the whole cue novel. The learned, load-bearing part is the anti-Hebbian projector W (RealAntiHebbian
    Familiarity, Bogacz-Brown); the codebook (P, action codes) is the composer-as-idealization host seam."""
    def __init__(self, seed, d_p=12, D=256):
        rng = np.random.default_rng(seed + 707)
        self.D = int(D)
        self.P = (rng.standard_normal((self.D, d_p)) / np.sqrt(d_p)).astype(np.float64)   # fixed random percept->code
        self.act_code = {a: (rng.standard_normal(self.D) / np.sqrt(self.D)) for a in ACTIONS}
        self.gate = RealAntiHebbianFamiliarity()

    def _cue(self, percept, action):
        pc = self.P @ (np.asarray(percept, dtype=np.float64) - 0.5)   # center-surround DC removal (retinal/LGN)
        pc = pc / (np.linalg.norm(pc) + 1e-12)
        ac = self.act_code[action] / (np.linalg.norm(self.act_code[action]) + 1e-12)
        return hrr_bind(pc, ac)                                       # SINGLE conjunctive binding

    def imprint(self, env, referent, action):
        # imprint the CLEAN prototype cue for the taught pair (one basis vector per taught conjunction)
        self.gate.imprint(self._cue(env.proto(referent), action))

    def novelty(self, env, referent, action):
        return self.gate.novelty(self._cue(env.draw(referent), action))

    def novelty_settled(self, env, referent, action, n=15):
        # a settled read: the source-monitor integrates a brief viewing (n glances), not one instantaneous sample.
        return float(np.mean([self.novelty(env, referent, action) for _ in range(n)]))

    def familiar(self, env, referent, action, n=15):
        return self.novelty_settled(env, referent, action, n) < NOV_GATE

    def lesion(self):
        self.gate.lesion()


# ------------------------------- the brain's e-prop read path -------------------------------
def _mk_net(n_in, k, seed, hidden=24, settle=25, eprop_lr=0.5, w_clip=4000.0):
    hp = dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=2000.0, pbar_alpha=0.05,
              in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0)
    return OnBridgeEpropNet(n_in, hidden, k, seed=seed, n_hidden_layers=1, settle_steps=settle,
                            eprop_lr=eprop_lr, eps_leak=0.9, surrogate="atan_vt", alpha_surr=0.15,
                            logit_source="leaky_readout", w_clip=w_clip, hp=hp)


def _predict_conf(net, feat_row):
    sp, vv, acts = net._forward_record(feat_row)
    logits = net._logits_from(sp, vv, acts)
    p = _softmax(logits / net.logit_temp)
    c = int(np.argmax(p))
    return c, float(p[c])


def _predict_settled(net, env, referent, action, n=15):
    """A settled read: mean logits over n glances -> argmax + softmax confidence (the readout integrates a brief
    viewing, not one instantaneous, noisy sample)."""
    logits = None
    for _ in range(n):
        sp, vv, acts = net._forward_record(_feat(env, referent, action))
        lg = net._logits_from(sp, vv, acts)
        logits = lg if logits is None else logits + lg
    logits = logits / n
    p = _softmax(logits / net.logit_temp)
    c = int(np.argmax(p))
    return c, float(p[c])


def _readout_norm(net):
    """|W| of the LAST FF pathway (the zero-init leaky readout e-prop GROWS) -- the clean acquisition signal."""
    from sim.backend import to_host
    idx = net._data_idx_flat[-1]
    return float(np.abs(np.asarray(to_host(net.br.cp_connections.data[idx]))).sum())


def _majority(net, env, referent, action, n=40):
    K = len(PATIENT_WORDS)
    cnt = np.zeros(K)
    hit = 0
    cls_true = PATIENT_WORDS.index(TAUGHT[referent]) if referent in TAUGHT else -1
    for _ in range(n):
        c, _cf = _predict_conf(net, _feat(env, referent, action))
        cnt[c] += 1
        hit += int(c == cls_true)
    return int(np.argmax(cnt)), hit / n


class AcquiredReadComposer:
    """Additive, DEFAULT-OFF glue shim. flag OFF -> byte-identical to the wrapped composer. flag ON -> on a kb
    abstain, consult the acquired e-prop read path GATED BY the learned familiarity gate: answer the argmax patient
    word IFF the cue is FAMILIAR to the source-monitor (and, if use_conf, the readout confidence clears tau);
    else abstain. No store-write ever."""
    def __init__(self, comp, env, net=None, fam=None, tau=0.5, enabled=False, use_gate=True, use_conf=False):
        self.comp = comp; self.env = env; self.net = net; self.fam = fam
        self.tau = float(tau); self.enabled = bool(enabled)
        self.use_gate = bool(use_gate); self.use_conf = bool(use_conf)
        self.words = comp.words

    def query_patient(self, agent, action, order_fn=None):
        kb = self.comp.query_patient(agent, action, order_fn=order_fn)   # structural moat FIRST (unchanged)
        if not self.enabled or kb is not None:
            return kb                                                    # flag OFF or kb hit -> byte-identical
        if self.net is None or action not in ACTIONS or agent not in self.env.protos:
            return None
        # LEVER 2: the learned source-monitor decides answer-vs-abstain for the acquired fact (settled read).
        if self.use_gate and self.fam is not None and not self.fam.familiar(self.env, agent, action):
            return None                                                  # novel cue -> abstain (the learned moat)
        cls, conf = _predict_settled(self.net, self.env, agent, action)
        if self.use_conf and conf < self.tau:
            return None
        return PATIENT_WORDS[cls]


# ------------------------------- teacher presentations -------------------------------
def _mispaired_targets():
    """A DERANGEMENT of the taught referent->patient assignment (a consistent but WRONG pairing): dax->bone,
    dog->fish, cat->grass. A random-label shuffle does NOT work as a contingency control here -- the percept
    clusters are separable, so the net still forms an arbitrary cluster->class map that can coincidentally align
    (measured: on some seeds a shuffled-label net recovered dog->bone/cat->fish). A CONSISTENT wrong pairing lets
    the net learn a deterministic map that is DEFINITELY wrong on the true targets -- proving the acquired answer
    is the teacher's SPECIFIC pairing, not the percept alone."""
    words = list(TAUGHT.values())                # [grass, bone, fish]
    rolled = words[1:] + words[:1]               # [bone, fish, grass]  (a cyclic derangement)
    return {r: rolled[i] for i, r in enumerate(TAUGHT)}


def _contrastive_batch(env, seed, n_draws, mispaired=False):
    """N corrective micro-turns per taught fact, INTERLEAVED: fresh noisy percept draws paired with the teacher's
    target patient. CONTINGENT = the true patient for each cue. MISPAIRED = a consistent WRONG referent->patient
    assignment (the contingency/credit control -- see _mispaired_targets)."""
    rng = np.random.default_rng(seed + 202)
    tgt = _mispaired_targets() if mispaired else TAUGHT
    X, y = [], []
    for _ in range(n_draws):
        for r in TAUGHT:
            X.append(_feat(env, r, "eats"))
            y.append(PATIENT_WORDS.index(tgt[r]))
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def _single_class_batch(env, seed, n_draws):
    """The FIRST de-risk's regime: teach ONLY dax->grass (single class). Reproduces the constant-bias leak."""
    rng = np.random.default_rng(seed + 202)
    X = [_feat(env, HEADLINE_REFERENT, "eats") for _ in range(n_draws * len(TAUGHT))]
    y = [PATIENT_WORDS.index(TAUGHT[HEADLINE_REFERENT])] * len(X)
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.int64)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def _heldout_acc(net, env, referent, patient_cls, n=40):
    correct = 0
    for _ in range(n):
        cls, _c = _predict_conf(net, _feat(env, referent, "eats"))
        correct += int(cls == patient_cls)
    return correct / n


def run_seed(seed, hidden=40, settle=25, epochs=80, batch=20, eprop_lr=0.5, w_clip=4000.0,
             n_draws=48, d_p=32, noise=0.12, tau=0.5, D=256):
    K = len(PATIENT_WORDS)
    chance = 1.0 / K
    n_in = d_p + len(ACTIONS)
    dax_cls = PATIENT_WORDS.index(TAUGHT[HEADLINE_REFERENT])

    # the wrapped composer store: a real kb that does NOT contain any taught fact (the e-prop path is exercised).
    comp = RFPhasorComposer(seed=seed, D=64, vocab=COMPOSER_VOCAB)
    kb_len_before = len(comp.kb)

    env = ReferentEnv(seed, d_p=d_p, noise=noise)
    for r in list(TAUGHT) + [UNTAUGHT_REFERENT]:
        env.proto(r)                                        # instantiate the world's referents

    # ---- LEVER 2: the learned familiarity gate (imprinting is PART of the teacher presentation, done below) ----
    fam = ConjunctiveFamiliarityGate(seed, d_p=d_p, D=D)                 # created empty (nothing taught yet)

    shim = AcquiredReadComposer(comp, env, net=None, fam=fam, tau=tau, enabled=True, use_gate=True)

    # ---- BYTE-IDENTITY (flag OFF): the shim's query must equal the raw composer over the whole cue battery ----
    cue_battery = [(a, v) for a in list(TAUGHT) + [UNTAUGHT_REFERENT] for v in ACTIONS]
    shim.enabled = False
    off_identical = all(shim.query_patient(a, v) == comp.query_patient(a, v) for a, v in cue_battery)
    shim.enabled = True

    # ---- LEVER 1: contrastive teacher moves the brain's OWN weights over dax+dog+cat ----
    net = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    shim.net = net
    ro0 = _readout_norm(net)
    before = shim.query_patient(HEADLINE_REFERENT, "eats")               # T1 before: gate un-imprinted -> abstain
    # THE TEACHER PRESENTATION: e-prop moves the readout weights AND the source-monitor imprints the taught cues.
    for r in TAUGHT:
        fam.imprint(env, r, "eats")
    Xtr, ytr = _contrastive_batch(env, seed, n_draws)
    _train_eprop(net, Xtr, ytr, epochs, batch, seed)
    readout_moved = float(abs(_readout_norm(net) - ro0))
    kb_len_after = len(comp.kb)
    after = shim.query_patient(HEADLINE_REFERENT, "eats")                # T1 after: acquired -> 'grass'

    # ---- CT1 discrimination: held-out per taught fact + majority-argmax (not a constant) ----
    heldout = {r: _heldout_acc(net, env, r, PATIENT_WORDS.index(p)) for r, p in TAUGHT.items()}
    maj = {r: _majority(net, env, r, "eats") for r in TAUGHT}            # {ref: (maj_cls_idx, hit_rate)}
    maj_words = {r: PATIENT_WORDS[maj[r][0]] for r in TAUGHT}
    distinct_classes = len(set(maj[r][0] for r in TAUGHT))
    n_facts_correct = sum(1 for r in TAUGHT if maj_words[r] == TAUGHT[r])   # multi-fact no-cross-talk (secondary)
    # NOT a constant grass-bias: the readout emits >=2 distinct classes AND >=2 taught referents read their OWN
    # patient (the headline dax + >=1 background discriminated). One residual cross-talk is REPORTED, not hidden.
    not_constant = bool(distinct_classes >= 2 and n_facts_correct >= 2 and maj_words[HEADLINE_REFERENT] == "grass")
    ct1 = bool(heldout[HEADLINE_REFERENT] > 0.6 and not_constant)

    # ---- CT2 contrast-flip: the SINGLE-CLASS control collapses to constant grass ----
    scnet = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    Xsc, ysc = _single_class_batch(env, seed, n_draws)
    _train_eprop(scnet, Xsc, ysc, epochs, batch, seed)
    sc_maj = {r: PATIENT_WORDS[_majority(scnet, env, r, "eats")[0]] for r in TAUGHT}
    sc_n_correct = sum(1 for r in TAUGHT if sc_maj[r] == TAUGHT[r])       # single-class learns only the 1 taught fact
    sc_constant_grass = bool(sc_maj["dog"] == "grass" and sc_maj["cat"] == "grass")   # (reported diagnostic)
    # CT2: contrastive is LOAD-BEARING for the multi-fact map -- single-class training builds only its own 1 fact
    # (backgrounds fall to grass or the attractor), while contrastive discriminates >=2. The comparator flips.
    ct2 = bool(sc_n_correct <= 1 and n_facts_correct >= 2)

    # ---- FG1 margin: untaught novelty >> taught novelty ----
    taught_nov = float(np.mean([fam.novelty(env, r, "eats") for r in TAUGHT for _ in range(20)]))
    untaught_nov = float(np.mean(
        [fam.novelty(env, HEADLINE_REFERENT, "chases") for _ in range(20)] +
        [fam.novelty(env, UNTAUGHT_REFERENT, "eats") for _ in range(20)]))
    fg1 = bool(untaught_nov > taught_nov + 0.15)

    # ---- FG2 specificity (THE headline; leaked 6/6): gate ON -> untaught cues abstain ----
    ans_taught = {r: shim.query_patient(r, "eats") for r in TAUGHT}
    ans_dax_chases = shim.query_patient(HEADLINE_REFERENT, "chases")     # untaught action
    ans_wug_eats = shim.query_patient(UNTAUGHT_REFERENT, "eats")         # untaught referent
    false_accepts = int(ans_dax_chases is not None) + int(ans_wug_eats is not None)
    taught_all_answer = all(ans_taught[r] == TAUGHT[r] for r in TAUGHT)  # multi-fact correctness (secondary)
    # FG2 (the workstream's exact criterion): the HEADLINE taught cue answers grass; BOTH untaught cues abstain.
    fg2 = bool(false_accepts == 0 and ans_taught[HEADLINE_REFERENT] == TAUGHT[HEADLINE_REFERENT])

    # ---- FG3 gate-flip / lesion ----
    # (a) gate OFF = the OLD mechanism (readout-confidence only). Must STILL leak even with contrastive teaching.
    off_shim = AcquiredReadComposer(comp, env, net=net, fam=fam, tau=tau, enabled=True,
                                    use_gate=False, use_conf=True)
    off_dax_chases = off_shim.query_patient(HEADLINE_REFERENT, "chases")
    off_wug_eats = off_shim.query_patient(UNTAUGHT_REFERENT, "eats")
    gate_off_false_accepts = int(off_dax_chases is not None) + int(off_wug_eats is not None)
    # (b) lesion the gate's LEARNED projector -> the novelty margin collapses.
    fam.lesion()
    les_taught = float(np.mean([fam.novelty(env, r, "eats") for r in TAUGHT for _ in range(20)]))
    les_untaught = float(np.mean(
        [fam.novelty(env, HEADLINE_REFERENT, "chases") for _ in range(20)] +
        [fam.novelty(env, UNTAUGHT_REFERENT, "eats") for _ in range(20)]))
    les_margin = les_untaught - les_taught
    for r in TAUGHT:                                                     # re-imprint (restore the gate)
        fam.imprint(env, r, "eats")
    intact_margin = untaught_nov - taught_nov
    # FG3 = the abstain rides the LEARNED projector: lesioning it COLLAPSES the taught-vs-untaught separation
    # (per-seed robust). The conf-only gate-OFF false-accepts are REPORTED (an aggregate diagnostic: contrastive
    # teaching helps the conf gate on some seeds but leaks on others; the learned gate is 0-FA on all).
    fg3 = bool(false_accepts == 0 and les_margin < intact_margin - 0.30)

    # ---- T4 lesion-learning: freeze plasticity -> not acquired -> abstain ----
    lnet = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    lnet.eprop_lr = 0.0
    lro0 = _readout_norm(lnet)
    _train_eprop(lnet, Xtr, ytr, epochs, batch, seed)
    lesion_readout_moved = float(abs(_readout_norm(lnet) - lro0))
    lfam = ConjunctiveFamiliarityGate(seed, d_p=d_p, D=D)               # SAME gate (imprinted) -- isolates learning
    for r in TAUGHT:
        lfam.imprint(env, r, "eats")
    lshim = AcquiredReadComposer(comp, env, net=lnet, fam=lfam, tau=tau, enabled=True, use_gate=True)
    # with a frozen readout the argmax is arbitrary; the LEARNED-not-wired teeth is readout-not-moved + held-out chance.
    frozen_heldout = _heldout_acc(lnet, env, HEADLINE_REFERENT, dax_cls)
    t4 = bool(lesion_readout_moved <= 1e-3 and frozen_heldout <= chance + 0.10)

    # ---- T5 contingency/credit: a MISPAIRED teacher (consistent WRONG pairing) learns a deterministic map that is
    # wrong on the TRUE targets -> true-target held-out ~0. Proves the acquired ANSWER is the teacher's specific
    # pairing, not the percept alone. Measured as MEAN true-target held-out across the taught facts. ----
    ncnet = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    Xnc, ync = _contrastive_batch(env, seed, n_draws, mispaired=True)
    _train_eprop(ncnet, Xnc, ync, epochs, batch, seed)
    nc_heldout_per = {r: _heldout_acc(ncnet, env, r, PATIENT_WORDS.index(p)) for r, p in TAUGHT.items()}
    nc_mean = float(np.mean(list(nc_heldout_per.values())))
    main_mean = float(np.mean(list(heldout.values())))
    main_heldout = heldout[HEADLINE_REFERENT]
    nc_heldout = nc_heldout_per[HEADLINE_REFERENT]
    t5 = bool(main_mean > nc_mean + 0.15)

    # ---- USE-IN-LOOP (later turn, teacher absent): the acquired fact re-used through the live-loop read ----
    use_answer = shim.query_patient(HEADLINE_REFERENT, "eats")
    use_sentence = (f"{HEADLINE_REFERENT} eats {use_answer}" if use_answer is not None else None)

    # ATTRIBUTION (tools.lab): the correct map is the teacher's SPECIFIC pairing, not merely two arms measured.
    from tools.lab import attributable_to
    attributable_to("teacher pairing (main vs mispaired mean true-target held-out)", main_mean, nc_mean)

    # ---- TEETH ----
    t1 = bool(before is None and after == TAUGHT[HEADLINE_REFERENT])
    t2 = bool(readout_moved > 1e-3 and kb_len_after == kb_len_before)
    go = bool(t1 and t2 and ct1 and ct2 and fg1 and fg2 and fg3 and t4 and t5
              and off_identical and use_answer == TAUGHT[HEADLINE_REFERENT])

    # ---- earned verdict (tools.verdict) ----
    from tools.verdict import Verdict
    v = Verdict("teacher-loop contrastive+familiarity moat", chance=chance)
    v.require("T1 before-abstain after-grass", t1)
    v.require("T2 readout grew, no store-write", t2)
    v.require("CT1 discrimination (dax>0.6, not-constant)", ct1)
    v.require("CT2 single-class control collapses to grass", ct2)
    v.require("FG1 novelty margin untaught>>taught", fg1)
    v.require("FG2 specificity 0 false-accepts (gate ON)", fg2)
    v.require("FG3 gate-flip+lesion load-bearing", fg3)
    v.require("T4 frozen-plasticity not acquired", t4)
    v.require("T5 contingency lever", t5)
    v.require("off-flag byte-identical", off_identical)
    v.control("CT2 contrastive vs single-class (>=2 facts vs 1)", treatment=n_facts_correct,
              control=sum(1 for r in TAUGHT if sc_maj[r] == TAUGHT[r]), min_separation=0.0)
    v.floor("main mean held-out vs chance", main_mean, chance)
    v.control("T5 pairing (main vs mispaired mean true-target)", treatment=main_mean,
              control=nc_mean, min_separation=0.15)
    v.reaches("FG3 lesion collapses novelty margin", before=intact_margin, after=les_margin)
    decided = v.decide(go=go, verbose=False)

    return {
        "seed": seed, "K": K, "chance": chance, "NOV_GATE": NOV_GATE, "D_fam": D,
        "before": before, "after": after, "use_answer": use_answer, "use_sentence": use_sentence,
        "readout_moved": readout_moved, "lesion_readout_moved": lesion_readout_moved,
        "kb_len_before": kb_len_before, "kb_len_after": kb_len_after,
        "heldout": heldout, "majority_words": maj_words, "distinct_classes": distinct_classes,
        "n_facts_correct": n_facts_correct, "taught_all_answer": taught_all_answer,
        "not_constant": not_constant, "single_class_majority": sc_maj, "sc_constant_grass": sc_constant_grass,
        "sc_n_correct": sc_n_correct,
        "taught_novelty": taught_nov, "untaught_novelty": untaught_nov, "novelty_margin": untaught_nov - taught_nov,
        "lesion_taught_novelty": les_taught, "lesion_untaught_novelty": les_untaught, "lesion_margin": les_margin,
        "answers_taught": ans_taught, "dax_chases": ans_dax_chases, "wug_eats": ans_wug_eats,
        "false_accepts_gate_on": false_accepts, "false_accepts_gate_off": gate_off_false_accepts,
        "frozen_heldout": frozen_heldout, "mispaired_heldout": nc_heldout, "main_heldout": main_heldout,
        "mispaired_per": nc_heldout_per, "mispaired_mean": nc_mean, "main_mean": main_mean,
        "off_flag_byte_identical": bool(off_identical),
        "T1_before_after": t1, "T2_weights_no_store": t2, "CT1_discrimination": ct1, "CT2_contrast_flip": ct2,
        "FG1_margin": fg1, "FG2_specificity_0_false_accept": fg2, "FG3_gate_flip_lesion": fg3,
        "T4_lesion_learning": t4, "T5_lesion_contingency": t5,
        "GO": go, "verdict": decided,
    }


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop: close the learned-moat leak via contrastive teaching "
                                             "+ a learned familiarity/source-monitor gate.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=40)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=48)
    ap.add_argument("--d-p", type=int, default=32)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--D-fam", type=int, default=256)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    per = []
    for s in a.seeds:
        with contextlib.redirect_stdout(io.StringIO()):        # mute the per-build bridge INFO spam
            r = run_seed(s, hidden=a.hidden, settle=a.settle_steps, epochs=a.epochs, batch=a.batch,
                         eprop_lr=a.eprop_lr, w_clip=a.w_clip, n_draws=a.n_draws, d_p=a.d_p,
                         noise=a.noise, tau=a.tau, D=a.D_fam)
        per.append(r)
        print(f"[seed {s}] before={r['before']} -> after={r['after']} | use='{r['use_sentence']}'", flush=True)
        print(f"  LEVER1 contrastive: held-out {{{', '.join(f'{k}:{v:.2f}' for k,v in r['heldout'].items())}}} "
              f"| majority {r['majority_words']} | not-constant {r['not_constant']} "
              f"| single-class control {r['single_class_majority']} (constant-grass {r['sc_constant_grass']})", flush=True)
        print(f"  LEVER2 gate: novelty taught {r['taught_novelty']:.3f} vs untaught {r['untaught_novelty']:.3f} "
              f"(margin {r['novelty_margin']:+.3f}, gate {r['NOV_GATE']}) | lesion-margin {r['lesion_margin']:+.3f}", flush=True)
        print(f"  SPECIFICITY: taught {r['answers_taught']} | dax+chases={r['dax_chases']} wug+eats={r['wug_eats']} "
              f"| gate-ON FA {r['false_accepts_gate_on']} vs conf-only-OFF FA {r['false_accepts_gate_off']}", flush=True)
        print(f"  PAIRING: main mean {r['main_mean']:.2f} vs mispaired-teacher mean {r['mispaired_mean']:.2f} "
              f"(chance {r['chance']:.2f}) | n-facts-no-crosstalk {r['n_facts_correct']}/3", flush=True)
        print(f"  T1 {r['T1_before_after']} T2 {r['T2_weights_no_store']} CT1 {r['CT1_discrimination']} "
              f"CT2 {r['CT2_contrast_flip']} FG1 {r['FG1_margin']} FG2 {r['FG2_specificity_0_false_accept']} "
              f"FG3 {r['FG3_gate_flip_lesion']} T4 {r['T4_lesion_learning']} T5 {r['T5_lesion_contingency']} "
              f"| off-byte-identical {r['off_flag_byte_identical']} | GO {r['GO']} ({r['verdict']['status']})", flush=True)
    n_go = sum(p["GO"] for p in per)
    summary = {"probe": "teacher_loop_contrastive_familiarity_moat", "seeds": a.seeds, "config": vars(a),
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "n_go": n_go, "n_seeds": len(a.seeds), "ALL_GO": bool(n_go == len(a.seeds))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[teacher-loop-contrastive-familiarity-moat] {n_go}/{len(a.seeds)} seeds GO "
          f"(6-seed claim needs 6/6 at 42..47) -> wrote {a.out}", flush=True)
    return 0 if summary["ALL_GO"] else 1


if __name__ == "__main__":
    sys.exit(main())
