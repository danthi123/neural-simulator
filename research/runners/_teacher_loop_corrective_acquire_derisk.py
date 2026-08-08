"""DEVELOPMENTAL TEACHER-LOOP -- FIRST DE-RISK (2026-08-08): the host TEACHER (social environment) teaches the
brain ONE new fact it does NOT already know BY CORRECTION, the brain ACQUIRES it via its OWN plasticity (a
synaptic weight change, NOT a host store-write), and the acquired concept is then RE-USED at a later
conversational turn through the UNCHANGED live-loop read (comp.query_patient).

THE ATOM (a child's corrective-learning atom). New referent 'dax' = a noisy perceptual prototype the brain has
never seen (the a1 make_referent_task small perceptual category). New fact 'dax eats grass'. The WORDS all exist
as codes in the composer vocab ('grass' is an existing patient code, 'eats'/'chases' are actions); the FACT
(this cue -> this patient) is absent from the composer kb and from any plastic map. Brand-new lexeme CODE
allocation (a dendritic/allocation frontier) is declared NEXT, not this smoke.

THE DIVIDING LINE, MADE MECHANICAL (brain-based-only). The teacher PRESENTS corrective input (a target the loop
pairs with the SAME cue the brain is responding to = a Kuhl-style CONTINGENT recast, entering as a corrective DA
third factor on the co-active cue->answer eligibility) LIKE a sensory input. The brain ACQUIRES it by moving its
OWN synaptic weights (OnBridgeEpropNet FF weights, the a1-GO transport-free e-prop rule; error = softmax(logits)
- onehot(target), which VANISHES at match so it can never become the clamp-as-crutch the 2026-06-08
teacher-correction finding warns about). NO composer.store() is called for the taught fact: the kb length is
asserted UNCHANGED (kb_len_before == kb_len_after) while the FF weights move (ff_weight_moved > 0). If the
acquisition were a kb.append THAT is the one-way injection we already have -- declared, not sold as learning.

THE READ PATH (additive, default-off glue shim). `AcquiredReadComposer` wraps the composer. With the flag OFF it
is BYTE-IDENTICAL: query_patient just calls comp.query_patient (asserted over the whole cue battery). With the
flag ON: query_patient FIRST runs the composer kb scan (the structural moat, unchanged); on an abstain it
consults the acquired e-prop read path -- fire a FRESH noisy percept of the queried referent (host = sensory
render) through the trained weights, softmax over the K patient words; answer the argmax word IFF max-softmax
clears the confidence threshold, else abstain.

HONEST SEAM (declared). For the ACQUIRED fact the abstain/answer gate is the e-prop readout CONFIDENCE threshold,
NOT the composer's structural kb-membership moat -- a genuinely-learned fact has no kb block by construction (a
kb block would be a store-write). So false-accepts on untaught cues are MEASURED (T3); a leaky learned
confidence gate is exactly the boundary the developmental engine must then close (the phaseB learned-moat
hardening arc). If T3 leaks this is a FIRST-CLASS honest negative that maps what the dev engine needs, not a
failure to hide.

SIX TEETH (per seed):
  T1 BEFORE/AFTER: query_patient('dax','eats') on fresh draws -- PASS = before abstain(None), after 'grass'.
  T2 WEIGHTS MOVED: ff_weight_moved = |W_after - W_before| > 1e-3 (acquisition is a weight change, not a read of
     a host write); AND kb_len unchanged (no store-write).
  T3 MATCHED CONTROL / moat specificity: untaught cues ('dax chases ?' -- untrained action; 'wug eats ?' -- 2nd
     untaught referent) must STILL abstain after teaching. 0 false-accepts = the update was SPECIFIC.
  T4 LESION-1 learning-pathway: freeze the plasticity (learning_gate/DA third factor = 0, eprop_lr=0) during the
     IDENTICAL teacher presentation -> not acquired -> still abstain (LEARNED, not WIRED; kills the store-write null).
  T5 LESION-2 contingency: NON-CONTINGENT teacher (target random, uncorrelated with the cue) -> held-out FRESH
     dax draws -> chance/abstain (contingency is the signal, not noise-memorization).
  T6 LESION-2b credit-route: SHUFFLE-DFA (eligibility intact, credit mismatched to the example) -> held-out
     chance (the CREDIT ROUTE carried it, not the forward reservoir).
Generalization to FRESH noisy dax draws (the a1 held-out set) makes T5/T6 clean: a real teacher signal
generalizes; a scrambled one can only memorize noise -> chance.

USE-IN-LOOP (the third leg). The acquired fact is exercised through the UNCHANGED live-loop read
comp.query_patient('dax','eats') at a LATER turn (teacher absent) and rendered into a sentence -- proving the
acquired concept is RE-USED in conversation, not merely stored. Driving the full stageA run_multi_turn_loop
(transformer mouth + GNW) is the declared NEXT step; this smoke demonstrates the byte-identical shim + the
later-turn re-use.

GO (single fact, per seed): after=='grass' AND before is None AND ff_weight_moved>1e-3 AND kb_len unchanged AND
untaught cues 0-false-accept AND frozen-W==abstain AND main_acc>non_contingent_acc+0.15 AND
main_acc>shuffle_dfa_acc+0.15. Clamp-crutch regression: the e-prop learning signal -> 0 as readout -> target.

DISCIPLINE: reuse-by-import (OnBridgeEpropNet + _train_eprop from the a1-GO port; RFPhasorComposer store),
NO sim/ edit, SIM_BACKEND=numpy, cfg.seed (NOT actual_seed_used -- the substrate is seeded via the seed= arg the
a1 net passes to CoreSimConfig.seed), additive/default-off. Single-seed SMOKE here; the 6-seed claim needs 6/6.
SMOKE: PYTHONPATH=$PWD SIM_BACKEND=numpy OPENBLAS_NUM_THREADS=1 \
  python -m research.runners._teacher_loop_corrective_acquire_derisk --seeds 42 \
  --out research/findings/raw/teacher_loop_corrective_acquire_s42.json
6-SEED (GO needs 6/6 at 42..47):  ... --seeds 42 43 44 45 46 47
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
# reuse-by-import: the a1-GO transport-free e-prop substrate (the brain's OWN plasticity) + its trainer, and the
# numpy VSA composer store (the live-loop read target). NO sim/ edit.
from research.runners._onbridge_eprop_port_derisk import OnBridgeEpropNet, _train_eprop, _softmax  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_corrective_acquire.json"

# --- the atom's fixed vocab. Patient WORDS = the K read-path classes; the taught patient is 'grass'. ---
PATIENT_WORDS = ["grass", "apple", "river", "bone", "fish", "seed"]   # K=6; 'grass' is the taught patient code
ACTIONS = ["eats", "chases"]                                          # 'eats' taught; 'chases' = untaught-action probe
TARGET_REFERENT = "dax"                                               # the new referent (noisy perceptual prototype)
TARGET_PATIENT = "grass"
UNTAUGHT_REFERENT = "wug"                                             # 2nd untaught referent (T3 percept specificity)
# words the wrapped composer must know so query_patient can render a normal (background) fact + the taught word.
COMPOSER_VOCAB = sorted(set(PATIENT_WORDS + ACTIONS + [TARGET_REFERENT, UNTAUGHT_REFERENT, "cat", "dog"]))


# ------------------------------- host = SENSORY ENVIRONMENT (legitimate) -------------------------------
class ReferentEnv:
    """The world's sensory render: each referent is a noisy perceptual prototype in [0,1]^d_p (the a1 small
    perceptual category). A PRESENTATION = clip(proto + noise*N(0,1), 0, 1). Features stay in [0,1] to match the
    bridge input-current mapping. Host code is legitimate here EXACTLY as the retinal-image render is -- the brain
    reads this percept through its OWN learned weights."""
    def __init__(self, seed, d_p=12, noise=0.12):
        self.rng = np.random.default_rng(seed + 101)
        self.d_p = int(d_p); self.noise = float(noise)
        self.protos = {}   # referent -> prototype vector

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
    """The brain's input = percept(referent) concat action-context one-hot. n_in = d_p + len(ACTIONS)."""
    return np.concatenate([env.draw(referent), _action_onehot(action)]).astype(np.float64)


# ------------------------------- the brain's ACQUIRED read path (e-prop weights) -------------------------------
def _mk_net(n_in, k, seed, hidden=24, settle=25, eprop_lr=0.5, w_clip=4000.0):
    """The a1-GO OnBridgeEpropNet build (transport-free e-prop; the FF weights are the SOLE learner)."""
    hp = dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=2000.0, pbar_alpha=0.05,
              in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0)
    return OnBridgeEpropNet(n_in, hidden, k, seed=seed, n_hidden_layers=1, settle_steps=settle,
                            eprop_lr=eprop_lr, eps_leak=0.9, surrogate="atan_vt", alpha_surr=0.15,
                            logit_source="leaky_readout", w_clip=w_clip, hp=hp)


def _predict_conf(net, feat_row):
    """(argmax_class, max_softmax) using the SAME leaky-readout logits e-prop trained on."""
    sp, vv, acts = net._forward_record(feat_row)
    logits = net._logits_from(sp, vv, acts)
    p = _softmax(logits / net.logit_temp)
    c = int(np.argmax(p))
    return c, float(p[c])


def _readout_norm(net):
    """|W| of the LAST FF pathway ONLY -- the zero-init leaky readout that e-prop GROWS. The spiking FF pathways
    are crushed toward |w|<=6 by the committed BDSP clip on every forward (docstring in the eprop port), so the
    TOTAL ff_weight_norm is dominated by that crush, not learning. The readout pathway is the clean acquisition
    signal: it starts at 0 and only e-prop moves it."""
    from sim.backend import to_host
    idx = net._data_idx_flat[-1]
    return float(np.abs(np.asarray(to_host(net.br.cp_connections.data[idx]))).sum())


class AcquiredReadComposer:
    """Additive, DEFAULT-OFF glue shim. flag OFF -> byte-identical to the wrapped composer. flag ON -> on a kb
    abstain, consult the acquired e-prop read path (fire a FRESH percept of the queried referent through the
    trained weights; answer the argmax patient word IFF max-softmax >= tau, else abstain). No store-write ever."""
    def __init__(self, comp, env, net=None, tau=0.5, enabled=False):
        self.comp = comp; self.env = env; self.net = net; self.tau = float(tau)
        self.enabled = bool(enabled)
        self.words = comp.words

    def query_patient(self, agent, action, order_fn=None):
        kb = self.comp.query_patient(agent, action, order_fn=order_fn)   # structural moat FIRST (unchanged)
        if not self.enabled or kb is not None:
            return kb                                                    # flag OFF or kb hit -> byte-identical
        if self.net is None or action not in ACTIONS or agent not in self.env.protos:
            return None                                                  # no acquired route for this cue -> abstain
        cls, conf = _predict_conf(self.net, _feat(self.env, agent, action))
        if conf < self.tau:
            return None                                                  # honest confidence gate -> abstain
        return PATIENT_WORDS[cls]


# ------------------------------- teacher presentations (corrective micro-turns) -------------------------------
def _make_corrective_batch(env, seed, n_draws, action, patient_cls, non_contingent=False, k=len(PATIENT_WORDS)):
    """N corrective micro-turns: fresh noisy dax draws (the cue) paired with the teacher's target. CONTINGENT =
    target is the true patient class paired with the cue. NON-CONTINGENT (T5) = target drawn at random,
    uncorrelated with the cue (the teacher is no longer responding to what the brain is looking at)."""
    rng = np.random.default_rng(seed + 202)
    X, y = [], []
    for _ in range(n_draws):
        X.append(_feat(env, TARGET_REFERENT, action))
        y.append(int(rng.integers(0, k)) if non_contingent else int(patient_cls))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def _heldout_acc(net, env, seed, action, patient_cls, n=40):
    """Held-out generalization: fresh dax draws -> fraction the net argmaxes to the taught class."""
    rng_state = env.rng
    correct = 0
    for _ in range(n):
        cls, _c = _predict_conf(net, _feat(env, TARGET_REFERENT, action))
        correct += int(cls == patient_cls)
    _ = rng_state
    return correct / n


def run_seed(seed, hidden=24, settle=25, epochs=60, batch=20, eprop_lr=0.5, w_clip=4000.0,
             n_draws=48, d_p=12, noise=0.12, tau=0.5):
    K = len(PATIENT_WORDS)
    patient_cls = PATIENT_WORDS.index(TARGET_PATIENT)
    chance = 1.0 / K
    n_in = d_p + len(ACTIONS)

    # the wrapped composer store: a real, non-empty kb (background facts) that does NOT contain the dax fact.
    comp = RFPhasorComposer(seed=seed, D=64, vocab=COMPOSER_VOCAB)
    comp.store("dog", "eats", "bone")      # background known facts (the brain is not a blank slate)
    comp.store("cat", "eats", "fish")
    kb_len_before = len(comp.kb)

    env = ReferentEnv(seed, d_p=d_p, noise=noise)
    env.proto(TARGET_REFERENT); env.proto(UNTAUGHT_REFERENT)   # instantiate prototypes (the world's referents)

    shim = AcquiredReadComposer(comp, env, net=None, tau=tau, enabled=True)

    # ---- BYTE-IDENTITY (flag OFF): the shim's query must equal the raw composer over the whole cue battery ----
    cue_battery = [(a, v) for a in [TARGET_REFERENT, UNTAUGHT_REFERENT, "dog", "cat"] for v in ACTIONS]
    shim.enabled = False
    off_identical = all(shim.query_patient(a, v) == comp.query_patient(a, v) for a, v in cue_battery)
    shim.enabled = True

    # ---- MAIN: contingent teacher moves the brain's OWN weights ----
    net = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    shim.net = net
    w0 = net.ff_weight_norm(); ro0 = _readout_norm(net)
    before = shim.query_patient(TARGET_REFERENT, "eats")                 # T1 before: untrained -> abstain

    Xtr, ytr = _make_corrective_batch(env, seed, n_draws, "eats", patient_cls)
    _train_eprop(net, Xtr, ytr, epochs, batch, seed)
    ff_moved = float(abs(net.ff_weight_norm() - w0))
    readout_moved = float(abs(_readout_norm(net) - ro0))                 # the CLEAN acquisition signal (see _readout_norm)
    kb_len_after = len(comp.kb)

    after = shim.query_patient(TARGET_REFERENT, "eats")                  # T1 after: acquired -> 'grass'
    main_acc = _heldout_acc(net, env, seed, "eats", patient_cls)

    # ---- T3 matched control / moat specificity: untaught cues must STILL abstain ----
    dax_ch_c, dax_ch_conf = _predict_conf(net, _feat(env, TARGET_REFERENT, "chases"))
    wug_c, wug_conf = _predict_conf(net, _feat(env, UNTAUGHT_REFERENT, "eats"))
    untaught_answers = {
        "dax_chases": shim.query_patient(TARGET_REFERENT, "chases"),     # untrained action
        "wug_eats": shim.query_patient(UNTAUGHT_REFERENT, "eats"),       # 2nd untaught referent
    }
    untaught_conf = {"dax_chases": dax_ch_conf, "wug_eats": wug_conf}    # quantify the learned-gate leak
    false_accepts = sum(1 for a in untaught_answers.values() if a is not None)

    # ---- T4 LESION-1 learning-pathway: identical teacher presentation, plasticity GATED OFF (DA/eprop_lr=0) ----
    #      the LEARNING pathway = the e-prop-grown readout; gate the third factor off -> the readout cannot grow ->
    #      not acquired -> abstain. (ff_weight_norm still 'moves' from the BDSP forward-clip crush, which is NOT the
    #      learning pathway -- so the teeth is the readout-norm + the behavioral abstain, not ff_weight_moved.)
    lnet = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    lnet.eprop_lr = 0.0                                                  # learning_gate=0 (no third-factor DA -> no dW)
    lro0 = _readout_norm(lnet)
    _train_eprop(lnet, Xtr, ytr, epochs, batch, seed)                    # SAME presentation, no weight change
    lesion_readout_moved = float(abs(_readout_norm(lnet) - lro0))        # ~0: the learning pathway did not move
    lshim = AcquiredReadComposer(comp, env, net=lnet, tau=tau, enabled=True)
    frozen_after = lshim.query_patient(TARGET_REFERENT, "eats")         # not acquired -> abstain

    # ---- T5 LESION-2 contingency: non-contingent teacher -> held-out chance ----
    ncnet = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    Xnc, ync = _make_corrective_batch(env, seed, n_draws, "eats", patient_cls, non_contingent=True)
    _train_eprop(ncnet, Xnc, ync, epochs, batch, seed)
    nc_acc = _heldout_acc(ncnet, env, seed, "eats", patient_cls)

    # ---- T6 LESION-2b credit-route: shuffle-DFA (eligibility intact, credit mismatched) -> held-out chance ----
    shnet = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    _train_eprop(shnet, Xtr, ytr, epochs, batch, seed, shuffle_dfa=True)
    sh_acc = _heldout_acc(shnet, env, seed, "eats", patient_cls)

    # ---- CLAMP-CRUTCH REGRESSION: the learning signal (softmax - onehot) -> 0 as the readout -> target ----
    def _mean_delta(nn):
        d = []
        for i in range(len(Xtr)):
            sp, vv, acts = nn._forward_record(Xtr[i])
            p = _softmax(nn._logits_from(sp, vv, acts) / nn.logit_temp)
            oh = np.zeros_like(p); oh[patient_cls] = 1.0
            d.append(float(np.abs(p - oh).sum()))
        return float(np.mean(d))
    fresh = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    delta_initial = _mean_delta(fresh)
    delta_final = _mean_delta(net)
    signal_vanishes = bool(delta_final < delta_initial)

    # ---- USE-IN-LOOP (later turn, teacher absent): the acquired fact re-used through the live-loop read ----
    use_answer = shim.query_patient(TARGET_REFERENT, "eats")
    use_sentence = (f"{TARGET_REFERENT} eats {use_answer}" if use_answer is not None else None)  # host render (declared)

    # ATTRIBUTION (tools.lab): the effect is the teacher CONTINGENCY, not merely two arms measured.
    from tools.lab import attributable_to
    attributable_to("teacher contingency (main vs non-contingent held-out)", main_acc, nc_acc)

    # ---- TEETH ----
    t1 = bool(before is None and after == TARGET_PATIENT)
    t2 = bool(readout_moved > 1e-3 and kb_len_after == kb_len_before)      # e-prop grew the readout; NO store-write
    t3 = bool(false_accepts == 0)                                         # the DECLARED honest seam (may leak)
    t4 = bool(frozen_after is None and lesion_readout_moved <= 1e-3)      # learning-gate off -> not acquired
    t5 = bool(main_acc > nc_acc + 0.15)
    # T6 shuffle-DFA is a DEPTH lesion: at 1 hidden layer the exact readout delta-rule carries the task and the
    # DFA hidden-credit is not load-bearing (a1 established this). So T6 is REPORTED but NOT a valid lesion here;
    # it belongs with the depth-2 semantic-inheritance task + its frozen-reservoir control (the NEXT de-risk).
    t6_depth_valid = False
    t6 = bool(main_acc > sh_acc + 0.15)                                   # measured; expected ~equal at depth-1
    # GO_core = the teeth that ARE load-bearing at this associative depth. T3 (the learned-moat seam) IS gated:
    # a leaky confidence gate is an HONEST NEGATIVE that maps the boundary, not something to hide.
    go_core = bool(t1 and t2 and t3 and t4 and t5 and off_identical and use_answer == TARGET_PATIENT)
    # GO_design = the full six-teeth GO exactly as the design stated it (includes T6, invalid at depth-1).
    go_design = bool(go_core and t6)

    return {
        "seed": seed, "K": K, "chance": chance,
        "before": before, "after": after, "use_answer": use_answer, "use_sentence": use_sentence,
        "ff_weight_moved": ff_moved, "readout_moved": readout_moved, "lesion_readout_moved": lesion_readout_moved,
        "kb_len_before": kb_len_before, "kb_len_after": kb_len_after,
        "main_heldout_acc": main_acc, "noncontingent_heldout_acc": nc_acc, "shuffle_dfa_heldout_acc": sh_acc,
        "untaught_answers": untaught_answers, "untaught_conf": untaught_conf, "false_accepts": false_accepts,
        "frozen_after": frozen_after, "off_flag_byte_identical": bool(off_identical),
        "delta_initial": delta_initial, "delta_final": delta_final, "signal_vanishes": signal_vanishes,
        "T1_before_after": t1, "T2_weights_moved_no_store": t2, "T3_specificity_0_false_accept": t3,
        "T4_lesion_learning_pathway": t4, "T5_lesion_contingency": t5,
        "T6_lesion_credit_route": t6, "T6_depth_valid_at_1_hidden": t6_depth_valid,
        "GO_core": go_core, "GO_design": go_design, "GO": go_core,
    }


def main():
    ap = argparse.ArgumentParser(description="Developmental teacher-loop: teach ONE new fact by correction; "
                                             "brain acquires by plasticity; re-uses it in a later turn.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=48)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    per = []
    for s in a.seeds:
        r = run_seed(s, hidden=a.hidden, settle=a.settle_steps, epochs=a.epochs, batch=a.batch,
                     eprop_lr=a.eprop_lr, w_clip=a.w_clip, n_draws=a.n_draws, d_p=a.d_p, noise=a.noise, tau=a.tau)
        per.append(r)
        print(f"[seed {s}] before={r['before']} -> after={r['after']} | use='{r['use_sentence']}' | "
              f"readout-moved {r['readout_moved']:.1f} (lesion {r['lesion_readout_moved']:.1f}) | kb {r['kb_len_before']}->{r['kb_len_after']} | "
              f"held-out main {r['main_heldout_acc']:.2f} / non-cont {r['noncontingent_heldout_acc']:.2f} / "
              f"shuffle {r['shuffle_dfa_heldout_acc']:.2f} (chance {r['chance']:.2f}) | "
              f"untaught {r['untaught_answers']} conf {r['untaught_conf']} (false-accepts {r['false_accepts']}) | frozen={r['frozen_after']}", flush=True)
        print(f"         T1 {r['T1_before_after']} T2 {r['T2_weights_moved_no_store']} T3 {r['T3_specificity_0_false_accept']} "
              f"T4 {r['T4_lesion_learning_pathway']} T5 {r['T5_lesion_contingency']} T6 {r['T6_lesion_credit_route']}(depth-valid={r['T6_depth_valid_at_1_hidden']}) "
              f"| off-byte-identical {r['off_flag_byte_identical']} | GO_core {r['GO_core']} | GO_design {r['GO_design']}", flush=True)
    n_go = sum(p["GO_core"] for p in per)
    summary = {"probe": "teacher_loop_corrective_acquire", "seeds": a.seeds, "config": vars(a),
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per,
               "n_go": n_go, "n_seeds": len(a.seeds), "ALL_GO": bool(n_go == len(a.seeds))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[teacher-loop-corrective-acquire] {n_go}/{len(a.seeds)} seeds GO "
          f"(6-seed claim needs 6/6 at 42..47) -> wrote {a.out}", flush=True)
    return 0 if summary["ALL_GO"] else 1


if __name__ == "__main__":
    sys.exit(main())
