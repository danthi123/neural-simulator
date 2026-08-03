"""Fluid GRADED-HEDGING over the composer cleanup match-score S -- replace the hard binary abstain moat with graded
natural hedging on MATCHED grounded facts (owner priority #3), WITHOUT weakening the moat.

The signal is the RF composer cleanup match-score **S** (NOT the bimodal familiarity novelty N). S is the phase-cos the
argmax already computes (`rf_phasor_composer.py::_cleanup_all_scored`), exposed via `last_trace` patient-chip
`confidence` when `trace=True`. The S-calibration de-risk (2026-07-21, GO 3-seed) showed S SEPARATES correct-from-wrong
grounded answers (AUC~0.85, permuted ~0.50) and is GRADED (not bimodal) -> it is the right signal for a graded band
ladder. See `research/findings/2026-07-21-fluid-abstain-graded-hedging-design-plus-adversarial-critique.md`.

ARCHITECTURE (additive, default-off, byte-identical when off):
  * `HedgeCalibrator` -- fits the S->band thresholds PER OPERATING POINT (D/M) from a held-out set of correct/wrong
    grounded-answer S values (a calibration curve; NOT hardcoded). Bands name honest accuracy floors.
  * bands over the SAME gate-first seam:
        L0 assert (S high, unchanged assertive reply)
        L1 "I think {fact}"
        L2 "I'm not certain, but {fact}"
        L3 graceful soft-abstain -> "I'm not sure, but it might be {p}"  (surfaces the grounded candidate p, no confab)
        MOAT (unchanged hard "I don't know" for GENUINE unknowns = query_patient/what_does -> None)  <-- LOAD-BEARING
  * `HedgingFluidChat(FluidChat)` -- opt-in `enable_hedging` flag on the production console. Default False =>
    BYTE-IDENTICAL (super()._answer). When True: enables the composer read-only trace, reads S for the matched fact,
    maps S->band, wraps the reply. The moat routing (what_does->None) is untouched.

HONEST SCOPE (from the de-risk): S carries signal only in the ERROR regime. At an UN-stressed op-point (high D / low
load) accuracy is 100% and S is flat -> the calibrator correctly makes EVERYTHING assert (nothing to hedge). Build +
calibrate at a STRESSED op-point (lower D / higher load) where the composer makes grounded errors so the graded middle
is populated. Grounded hedging tracks the within-fact D/M bundling capacity, NOT KB size.

Reuse-by-import; NO `sim/` edit. CPU / SIM_BACKEND=numpy.

Run:
  SIM_BACKEND=numpy python -m research.runners._fluidconv_graded_hedging --anti-cheats   # the full GO gate, 3 seeds
  SIM_BACKEND=numpy python -m research.runners._fluidconv_graded_hedging --demo          # a graded transcript
  SIM_BACKEND=numpy python -m research.runners._fluidconv_graded_hedging --console-check # console integration (stub faculty)
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.WARNING)   # silence the per-bridge init spam (read-only harness)
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402

# band ladder (S descending) + numeric hedge levels (L0 most confident=0 ... L3 soft-abstain=3)
BANDS = ["L0", "L1", "L2", "L3"]
HEDGE_LEVEL = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}


# ----------------------------------------------------------------------------- rendering
def _cap(s):
    return (s[:1].upper() + s[1:]) if s else s


def _decap(s):
    return (s[:1].lower() + s[1:]) if s else s


def hedge_render(band, assertive_reply, patient):
    """Template-prefix hedging. L0 = unchanged assertive; L1/L2 prefix; L3 = graceful soft-abstain surfacing the
    grounded candidate `patient` (no free generation -> no confab). The moat's hard 'I don't know' is rendered by the
    caller (band == 'MOAT') and never passes through here."""
    if band == "L0":
        return assertive_reply
    if band == "L1":
        return "I think " + _decap(assertive_reply)
    if band == "L2":
        return "I'm not certain, but " + _decap(assertive_reply)
    if band == "L3":
        return f"I'm not sure, but it might be {patient}."
    raise ValueError(f"unknown band {band!r}")


def read_patient_S(last_trace):
    """Extract S = the composer's patient-chip cleanup match-confidence from `composer.last_trace` (the SAME phase-cos
    the argmax used). None if the trace/chip is absent (=> the caller defaults to L0 assert; never hedges blind)."""
    if not last_trace:
        return None
    for ch in last_trace.get("roles", []):
        if ch.get("role") == "patient" and not ch.get("cue", False):
            return ch.get("confidence")
    return None


# ----------------------------------------------------------------------------- calibration
class HedgeCalibrator:
    """Fit S->band thresholds PER OPERATING POINT from held-out (S, correct) pairs.

    Method (a calibration curve, not a hardcode): using the monotone risk-coverage relation that the de-risk validated
    (raising the S bar preferentially discards wrong answers), pick the LOWEST threshold that meets each band's honest
    accuracy floor:
        t1 (L0 assert)     = min t : acc(answered | S>=t) >= floor_L0   (default 0.90)
        t2 (L1 'I think')  = min t : acc(answered | S>=t) >= floor_L1   (default 0.78)
        t3 (L2 'not sure') = min t : acc(answered | S>=t) >= floor_L2   (default 0.62)
        S < t3            -> L3 graceful soft-abstain
    Guaranteed ordered (t1>=t2>=t3, acc(S>=t) is monotone in t when S separates). Fallbacks keep the ladder non-degenerate
    when a floor is unreachable or would empty a band (quantile spread), so a very-stressed op-point still yields a ladder
    and an un-stressed 100%-accurate op-point yields t1=min(S) => assert-everything. Achieved per-band accuracy is
    recorded so each band NAMES its honest accuracy."""

    def __init__(self, t1, t2, t3, meta=None):
        assert t1 >= t2 >= t3, (t1, t2, t3)
        self.t1, self.t2, self.t3 = float(t1), float(t2), float(t3)
        self.meta = meta or {}

    def band(self, S):
        if S is None:
            return "L0"                     # no signal -> assert (never hedge blind, never downgrade a matched fact)
        if S >= self.t1:
            return "L0"
        if S >= self.t2:
            return "L1"
        if S >= self.t3:
            return "L2"
        return "L3"

    def as_dict(self):
        return {"t1": self.t1, "t2": self.t2, "t3": self.t3, **self.meta}

    @staticmethod
    def _acc_at(S, C, t):
        m = S >= t
        return (float(C[m].mean()) if m.any() else float("nan"), int(m.sum()))

    @classmethod
    def fit(cls, rows, floors=(0.90, 0.78, 0.62), min_n=3):
        """rows: list of (S, correct_bool) over ANSWERED grounded queries (held-out). Returns a fitted calibrator."""
        S = np.array([r[0] for r in rows], float)
        C = np.array([1.0 if r[1] else 0.0 for r in rows], float)
        if len(S) == 0:
            return cls(0.5, 0.4, 0.3, {"note": "no data -> default ladder"})
        # UN-STRESSED op-point: if the whole answered set already meets the top assert floor (accuracy ~100%, e.g. the
        # console's high-D op-point), assert EVERYTHING (t1=t2=t3=min(S)) -- the honest "nothing to hedge" behavior. No
        # quantile-spread of a flat, all-correct S distribution.
        if float(C.mean()) >= floors[0]:
            t = float(S.min())
            return cls(t, t, t, {"floors": list(floors), "n_cal": len(S), "acc_all": round(float(C.mean()), 3),
                                 "note": "assert-all (acc_all >= floor_L0; un-stressed op-point)"})
        lo, hi = float(S.min()), float(S.max())
        grid = np.unique(np.concatenate([S, np.linspace(lo, hi, 41)]))
        grid = np.sort(grid)

        def thr_for(floor):
            # lowest t meeting the accuracy floor with >= min_n answered; else the t maximizing acc (best available)
            best_t, best_acc = None, -1.0
            for t in grid:
                acc, n = cls._acc_at(S, C, t)
                if n >= min_n and acc == acc:
                    if acc >= floor:
                        return float(t)
                    if acc > best_acc:
                        best_acc, best_t = acc, float(t)
            return float(best_t if best_t is not None else lo)

        t1 = thr_for(floors[0])
        t2 = thr_for(floors[1])
        t3 = thr_for(floors[2])
        # enforce ordering (floors are increasing => thresholds should be non-increasing; clamp any inversion)
        t2 = min(t2, t1)
        t3 = min(t3, t2)
        # keep the ladder non-degenerate: if collapsed, spread by S quantiles so every band can be populated
        if not (t1 > t3):
            q = np.quantile(S, [0.20, 0.45, 0.70])
            t3, t2, t1 = float(q[0]), float(q[1]), float(q[2])
            t2 = min(t2, t1); t3 = min(t3, t2)
        # record achieved accuracy floors (the calibration curve) so each band names its honest accuracy
        meta = {
            "floors": list(floors), "n_cal": len(S),
            "acc_ge_t1": round(cls._acc_at(S, C, t1)[0], 3),
            "acc_ge_t2": round(cls._acc_at(S, C, t2)[0], 3),
            "acc_ge_t3": round(cls._acc_at(S, C, t3)[0], 3),
            "acc_all": round(float(C.mean()), 3),
        }
        return cls(t1, t2, t3, meta)


# ----------------------------------------------------------------------------- stressed-composer harness (calibration + anti-cheats)
_ANIMALS = ["dog", "cat", "wolf", "fox", "bird", "lion", "bear", "deer", "hawk", "mouse", "frog", "goat",
            "seal", "crow", "moth", "toad", "swan", "hare", "lynx", "mole"]
_VERBS = ["chase", "eat", "watch", "fear", "follow", "find", "like", "hunt"]
_OBJECTS = ["meat", "fish", "seed", "grass", "bone", "worm", "berry", "leaf", "root", "egg",
            "nut", "corn", "hay", "moss", "reed", "clam"]


def _v3(v):
    return v + ("es" if v.endswith(("s", "x", "z", "ch", "sh")) else "s")


def _build_stressed(seed, D, n_facts, vocab_mode="synthetic", composer_kwargs=None):
    """Build an RFPhasorComposer at a STRESSED op-point (low D) with distinct-(agent,action) SVO facts (moat-uniqueness
    preserved). trace=True so S is read from last_trace. Returns (composer, facts, unknown_cues).

    vocab_mode='synthetic' (default) = V=48 decorrelated codes with agent/action/patient drawn from the FULL pool (the
    exact setting the S-calibration de-risk validated: graded errors AND a CLEAN intrinsic moat at D=16 -> isolates the
    hedging mechanism). vocab_mode='themed' = readable animal/verb/object words with a small (8-verb) action subspace,
    used for the readable demo + a moat-ROBUSTNESS check (the constrained vocab makes the SUBSTRATE itself leak a few %
    at D=16, so it stresses whether hedging stays gate-first invariant even when the composer's own moat is imperfect)."""
    from research.runners.rf_phasor_composer import RFPhasorComposer
    rng = np.random.default_rng(1000 + seed)
    composer_kwargs = dict(composer_kwargs or {})
    if vocab_mode == "synthetic":
        V = 48
        vocab = [f"w{i:03d}" for i in range(V)]
        comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab, trace=True, **composer_kwargs)
        facts, used = [], set()
        tries = 0
        while len(facts) < n_facts and tries < n_facts * 60:
            tries += 1
            ia, iv, ip = rng.choice(V, 3, replace=False)
            a, v, p = vocab[ia], vocab[iv], vocab[ip]
            if (a, v) in used:
                continue
            used.add((a, v)); comp.store(a, v, p); facts.append((a, v, p))
        unknown = []
        while len(unknown) < 80:
            a, v = vocab[int(rng.integers(V))], vocab[int(rng.integers(V))]
            if (a, v) not in used and (a, v) not in set(unknown):
                unknown.append((a, v))
        return comp, facts, unknown
    # themed (readable)
    vocab = _ANIMALS + _VERBS + _OBJECTS
    comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab, trace=True, **composer_kwargs)
    facts, used = [], set()
    tries = 0
    while len(facts) < n_facts and tries < n_facts * 60:
        tries += 1
        a = _ANIMALS[int(rng.integers(len(_ANIMALS)))]
        v = _VERBS[int(rng.integers(len(_VERBS)))]
        p = _OBJECTS[int(rng.integers(len(_OBJECTS)))]
        if (a, v) in used:
            continue
        used.add((a, v)); comp.store(a, v, p); facts.append((a, v, p))
    unknown = []
    for _ in range(120):
        a = _ANIMALS[int(rng.integers(len(_ANIMALS)))]
        v = _VERBS[int(rng.integers(len(_VERBS)))]
        if (a, v) not in used and (a, v) not in set(unknown):
            unknown.append((a, v))
    return comp, facts, unknown


def grounded_answer(comp, agent, action, cal):
    """Mirror the console `_answer` grounded path: gate-first (query_patient -> None => hard moat), else render an
    assertive grounded reply and (if cal is not None) wrap it in the S-chosen hedge band. cal=None => hedging OFF."""
    p_full = comp.query_patient(agent, action)          # trace populated on the matched block
    if p_full is None:
        return {"matched": False, "band": "MOAT", "reply": "I don't know.", "S": None, "p": None}
    p = p_full.split()[-1]
    assertive = _cap(f"the {agent} {_v3(action)} {p}.")
    if cal is None:                                     # hedging OFF -> byte-identical assertive path
        return {"matched": True, "band": "L0", "reply": assertive, "S": None, "p": p}
    S = read_patient_S(comp.last_trace)
    band = cal.band(S)
    return {"matched": True, "band": band, "reply": hedge_render(band, assertive, p), "S": S, "p": p}


def _spearman(x, y):
    """Spearman rank correlation (Pearson on ranks). None if degenerate."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return None
    rx, ry = _rankdata(x), _rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _permuted_map_stat(levels, correct, seed, n_perm=300):
    """Spearman(band_level, accuracy) for the TRUE band assignment vs its permuted-label control mean. A genuine
    S->band map yields a NEGATIVE true correlation (higher band => lower accuracy) that PERMUTING destroys (~0).
    Returns (true_spearman, permuted_mean, ok) where ok = true<-0.05 and |perm| < 0.5*|true| (None-safe)."""
    true_bac = _spearman(levels, correct)
    prng = np.random.default_rng(999 + seed)
    perm = [_spearman(list(prng.permutation(levels)), correct) for _ in range(n_perm)]
    perm = [x for x in perm if x is not None]
    perm_mean = float(np.mean(perm)) if perm else float("nan")
    ok = (true_bac is not None and true_bac < -0.05 and perm_mean == perm_mean
          and abs(perm_mean) < abs(true_bac) * 0.5)
    return (round(true_bac, 3) if true_bac is not None else None,
            round(perm_mean, 3) if perm_mean == perm_mean else None, ok)


def _rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    sa = a[order]; i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + 1 + j + 1)
        i = j + 1
    return r


def run_anti_cheats(seed, D=16, n_facts=48, vocab_mode="synthetic"):
    """The full GO gate at a stressed op-point. The calibrator is fit on a SEPARATE held-out composer (same op-point
    D/M, a different instantiation seed+777) -- a genuine held-out set of correct/wrong grounded-answer S values, per
    the design's 'calibrate per operating point from held-out'. Every anti-cheat is then measured on the FULL fact set
    of the eval composer (max wrong answers for robust statistics) + its unknown-cue moat."""
    # --- fit the calibrator on a SEPARATE held-out composer at the SAME op-point ---
    comp_cal, facts_cal, _u = _build_stressed(seed + 777, D, n_facts, vocab_mode=vocab_mode)
    cal_rows = []
    for (a, v, p) in facts_cal:
        r = grounded_answer(comp_cal, a, v, cal=HedgeCalibrator(1, 1, 1))   # dummy cal just to force the trace read
        if r["S"] is not None:
            cal_rows.append((r["S"], r["p"] == p))
    cal = HedgeCalibrator.fit(cal_rows)

    # --- the eval composer + FULL-set evaluation (held-out from calibration) ---
    comp, facts, unknown = _build_stressed(seed, D, n_facts, vocab_mode=vocab_mode)
    ev = []            # per-eval-query record
    for (a, v, p) in facts:
        on = grounded_answer(comp, a, v, cal=cal)
        off = grounded_answer(comp, a, v, cal=None)
        correct = (on["p"] == p)
        ev.append(dict(a=a, v=v, gold=p, ret=on["p"], correct=correct,
                       S=on["S"], band=on["band"], reply_on=on["reply"], reply_off=off["reply"]))
    ev = [e for e in ev if e["S"] is not None]

    # (1) MOAT -- the LOAD-BEARING gate. The moat is gate-first (what_does/query_patient -> None => hard 'I don't
    # know'), and hedging reads S only AFTER that decision. So the correct property to VERIFY is that hedging does not
    # WEAKEN the moat, decomposed into:
    #   (1a) no abstain->answer conversion: whenever the composer abstains (OFF query_patient -> None), the ON reply is
    #        byte-identical hard 'I don't know' (hedging never turns a genuine abstain into an answer);
    #   (1b) gate-first invariance: the SET of unknown cues that yield an answer is IDENTICAL on/off (hedging adds ZERO
    #        false-accepts) -- so the composer's own false-accept count is unchanged by hedging;
    #   (1c) context: the composer's INTRINSIC false-accept rate at this op-point (a substrate property, NOT hedging).
    #        Where the substrate DOES false-match at extreme stress, hedging DEMOTES the low-S ones toward L3 soft-abstain
    #        (safer), never confidently asserts a new one.
    on_answered, off_answered = set(), set()
    abstain_leak = 0                 # (1a) violations: OFF abstains but ON did not byte-identically abstain
    intrinsic_fa = 0                 # (1c) composer's own false-accepts (query_patient -> non-None on an unknown)
    fa_demoted_low = 0               # of those, how many hedging pushed to L2/L3 (safer)
    for (a, v) in unknown:
        on = grounded_answer(comp, a, v, cal=cal)
        off = grounded_answer(comp, a, v, cal=None)
        if on["matched"]:
            on_answered.add((a, v))
        if off["matched"]:
            off_answered.add((a, v)); intrinsic_fa += 1
            if on["band"] in ("L2", "L3"):
                fa_demoted_low += 1
        else:                        # OFF abstained (genuine moat) -> ON must be byte-identical hard IDK
            if on["reply"] != "I don't know." or on["matched"]:
                abstain_leak += 1
    gatefirst_invariant = (on_answered == off_answered)     # (1b) hedging adds/removes 0 false-accepts
    moat_ok = (abstain_leak == 0) and gatefirst_invariant   # hedging does not weaken the moat

    # (2) enable_hedging OFF byte-identical: OFF reply == the plain assertive path for every matched eval query
    off_byte_id = all(
        grounded_answer(comp, e["a"], e["v"], cal=None)["reply"] == _cap(f"the {e['a']} {_v3(e['v'])} {e['ret']}.")
        for e in ev)

    # (3) hedge-rate MONOTONE in S: Spearman(-S, hedge_level) > 0 over held-out matched queries
    S_arr = [e["S"] for e in ev]
    lvl = [HEDGE_LEVEL[e["band"]] for e in ev]
    sp = _spearman([-s for s in S_arr], lvl)
    monotone_ok = (sp is not None and sp > 0.0)

    # (4) NO extra confident-wrong: among assertive (L0/L1) answers, wrong-rate <= the un-hedged baseline wrong-rate
    base_wrong = np.mean([0.0 if e["correct"] else 1.0 for e in ev]) if ev else float("nan")
    assertive = [e for e in ev if e["band"] in ("L0", "L1")]
    asrt_wrong = np.mean([0.0 if e["correct"] else 1.0 for e in assertive]) if assertive else 0.0
    no_extra_wrong = (asrt_wrong <= base_wrong + 1e-9)

    # (5) GRACEFUL soft-abstain: matched-but-low-S (band L3) surfaces the candidate p, never a flat 'I don't know';
    #     the flat-refusal rate on MATCHED queries is 0.
    matched_flat_idk = sum(1 for e in ev if e["reply_on"] == "I don't know.")
    l3 = [e for e in ev if e["band"] == "L3"]
    l3_surfaces_p = all((e["ret"] is not None and e["ret"] in e["reply_on"]) for e in l3)
    graceful_ok = (matched_flat_idk == 0) and l3_surfaces_p

    # (6) permuting the S->band map DESTROYS the accuracy-vs-band relation
    #     true: Spearman(band_level, accuracy) should be NEGATIVE (higher band => lower accuracy). permuted -> ~0.
    #     N/A (ok=None) when the eval split has <2 wrong answers (nothing to separate) -- the GATE is the POOLED
    #     across-seeds version computed in main (more wrong answers = robust). Reported per-seed for context.
    n_wrong_eval = int(sum(1 for e in ev if not e["correct"]))
    true_bac, perm_mean, permute_ok = _permuted_map_stat(lvl, [1.0 if e["correct"] else 0.0 for e in ev], seed)
    if n_wrong_eval < 2:
        permute_ok = None       # N/A this seed (degenerate); pooled version is the gate

    band_counts = {b: sum(1 for e in ev if e["band"] == b) for b in BANDS}
    band_acc = {b: round(float(np.mean([1.0 if e["correct"] else 0.0 for e in ev if e["band"] == b])), 3)
                if band_counts[b] else None for b in BANDS}

    return dict(
        seed=seed, D=D, vocab_mode=vocab_mode, n_facts=len(facts), n_eval=len(ev), n_unknown=len(unknown),
        cal=cal.as_dict(), band_counts=band_counts, band_acc=band_acc,
        moat={"abstain_leak": abstain_leak, "gatefirst_invariant": gatefirst_invariant,
              "intrinsic_composer_false_accepts": intrinsic_fa, "of_those_demoted_to_L2L3": fa_demoted_low,
              "ok": moat_ok},
        off_byte_identical=off_byte_id,
        hedge_monotone={"spearman_negS_level": round(sp, 3) if sp is not None else None, "ok": monotone_ok},
        no_extra_confident_wrong={"baseline_wrong": round(float(base_wrong), 3),
                                  "assertive_wrong": round(float(asrt_wrong), 3),
                                  "n_assertive": len(assertive), "ok": no_extra_wrong},
        graceful_soft_abstain={"matched_flat_idk": matched_flat_idk, "n_L3": len(l3),
                               "l3_surfaces_candidate": l3_surfaces_p, "ok": graceful_ok},
        permuted_map={"true_band_acc_spearman": true_bac, "permuted_mean": perm_mean,
                      "n_wrong_eval": n_wrong_eval, "ok": permute_ok},
        ev_rows=[(float(e["S"]), int(HEDGE_LEVEL[e["band"]]), bool(e["correct"])) for e in ev],
        all_ok=all([moat_ok, off_byte_id, monotone_ok, no_extra_wrong, graceful_ok,
                    (permute_ok in (True, None))]),
    )


def run_demo(seed=42, D=16, n_facts=48):
    """A graded transcript: build a STRESSED eval composer + a held-out calibration composer (same op-point), then show
    queries whose S lands in each band (assert / I think / not certain / soft-abstain) + an unknown cue -> the moat.
    For each band, prefer a representative that tells the story: L0/L1 a CORRECT answer (confident, right); L2/L3 a
    WRONG answer where possible (the value: hedging DEMOTES a wrong grounded answer to soft-abstain instead of
    asserting it), else a correct one."""
    comp_cal, facts_cal, _u = _build_stressed(seed + 777, D, n_facts, vocab_mode="themed")
    cal_rows = []
    for (a, v, p) in facts_cal:
        r = grounded_answer(comp_cal, a, v, cal=HedgeCalibrator(1, 1, 1))
        if r["S"] is not None:
            cal_rows.append((r["S"], r["p"] == p))
    cal = HedgeCalibrator.fit(cal_rows)
    comp, facts, unknown = _build_stressed(seed, D, n_facts, vocab_mode="themed")

    # bucket every eval query by band; pick a representative per band (prefer wrong for L2/L3, correct for L0/L1)
    buckets = {b: [] for b in BANDS}
    for (a, v, p) in facts:
        r = grounded_answer(comp, a, v, cal=cal)
        if r["S"] is None:
            continue
        buckets[r["band"]].append(dict(q=f"what does the {a} {v}?", **r, gold=p, correct=(r["p"] == p)))

    def _pick(b):
        cand = buckets[b]
        if not cand:
            return None
        want_wrong = b in ("L2", "L3")
        pref = [c for c in cand if (not c["correct"]) == want_wrong]
        return (pref or cand)[0]

    lines = []
    lines.append(f"[calibrated on a held-out composer @ D={D} (stressed): t1={cal.t1:.3f} t2={cal.t2:.3f} t3={cal.t3:.3f}]")
    for b in BANDS:
        r = _pick(b)
        if r is None:
            lines.append(f"  ({b}: no eval query landed here this seed)")
            continue
        tag = "correct" if r["correct"] else f"WRONG(gold={r['gold']})"
        lines.append(f"  you>   {r['q']}")
        lines.append(f"  brain> {r['reply']}    [S={r['S']:.3f} -> {b}; {tag}]")
    # moat -- pick an unknown the composer genuinely abstains on (query_patient -> None)
    for (a, v) in unknown:
        r = grounded_answer(comp, a, v, cal=cal)
        if not r["matched"]:
            lines.append(f"  you>   what does the {a} {v}?   (never taught)")
            lines.append(f"  brain> {r['reply']}    [MOAT: query_patient->None, S not read]")
            break
    return "\n".join(lines), cal


def deployment_moat_check(seed, D=256, n_facts=34):
    """The DEPLOYED op-point moat (D=256, the console's phasor dimension): the composer's INTRINSIC false-accepts on
    unknown cues should be 0 (clean moat in deployment -> the low-D leak is purely a stress artifact), matched facts
    all assert (calibrated to assert-everything at 100% accuracy), and hedging is gate-first invariant. Synthetic
    decorrelated codes (the deployment's codes are similarly decorrelated at D=256)."""
    comp, facts, unknown = _build_stressed(seed, D, n_facts, vocab_mode="synthetic")
    # calibrate from the stored facts (all correct at D=256 -> assert-everything)
    rows = []
    for (a, v, p) in facts:
        r = grounded_answer(comp, a, v, cal=HedgeCalibrator(1, 1, 1))
        if r["S"] is not None:
            rows.append((r["S"], r["p"] == p))
    cal = HedgeCalibrator.fit(rows)
    acc = float(np.mean([1.0 if c else 0.0 for _s, c in rows])) if rows else float("nan")
    intrinsic_fa = 0
    abstain_leak = 0
    on_ans, off_ans = set(), set()
    for (a, v) in unknown:
        on = grounded_answer(comp, a, v, cal=cal)
        off = grounded_answer(comp, a, v, cal=None)
        if on["matched"]:
            on_ans.add((a, v))
        if off["matched"]:
            off_ans.add((a, v)); intrinsic_fa += 1
        elif on["reply"] != "I don't know." or on["matched"]:
            abstain_leak += 1
    # matched facts all assert (band L0) at this un-stressed op-point
    all_assert = all(grounded_answer(comp, a, v, cal=cal)["band"] == "L0" for (a, v, _p) in facts)
    return dict(seed=seed, D=D, n_facts=len(facts), accuracy=round(acc, 3),
                intrinsic_composer_false_accepts=intrinsic_fa, abstain_leak=abstain_leak,
                gatefirst_invariant=(on_ans == off_ans), matched_all_assert=all_assert,
                ok=(intrinsic_fa == 0 and abstain_leak == 0 and (on_ans == off_ans) and all_assert))


# ----------------------------------------------------------------------------- console integration (subclass)
def _import_fluidchat():
    from research.runners._fluidconv_chat_repl import FluidChat
    return FluidChat


def make_hedging_console_class():
    """Build HedgingFluidChat(FluidChat) at call time (so importing this module never forces the console's heavy
    imports / the FT ckpt). enable_hedging default False => byte-identical to the shipped console."""
    FluidChat = _import_fluidchat()

    class HedgingFluidChat(FluidChat):
        """The production console + opt-in graded hedging on the grounded-fact path. Default off = byte-identical."""

        def __init__(self, *args, enable_hedging=False, hedge_cal=None, **kw):
            super().__init__(*args, **kw)
            self.enable_hedging = bool(enable_hedging)
            # a default calibrator if none supplied; at the console op-point (high D) this yields assert-everything.
            self.hedge_cal = hedge_cal or HedgeCalibrator(0.5, 0.4, 0.3, {"note": "default"})
            if self.enable_hedging:
                # enable the composer's READ-ONLY trace so S is populated on what_does (return value unchanged)
                try:
                    self.mta.agent.composer.trace = True
                except Exception:
                    pass

        def calibrate_here(self, floors=(0.90, 0.78, 0.62), n_probe_cues=None):
            """Fit the band thresholds at THIS console's operating point from its own stored facts (held-out S,
            correct/wrong). At the console's high-D op-point accuracy is ~100% and S is flat -> assert-everything."""
            comp = self.mta.agent.composer
            was = getattr(comp, "trace", False)
            comp.trace = True
            rows = []
            for f in self.store_keys:
                if len(f) != 3:
                    continue
                a, v, p = f
                ret = self.mta.agent.what_does(a, v)
                S = read_patient_S(comp.last_trace)
                if ret is None or S is None:
                    continue
                rows.append((S, ret.split()[-1] == p))
            comp.trace = was or self.enable_hedging
            self.hedge_cal = HedgeCalibrator.fit(rows, floors=floors)
            return self.hedge_cal

        def _answer(self, subj, verb):
            # parent does what_does (populates the read-only trace) + render; moat + off-path = byte-identical
            p, reply = super()._answer(subj, verb)
            if not self.enable_hedging or p is None:
                return p, reply
            S = read_patient_S(self.mta.agent.composer.last_trace)
            band = self.hedge_cal.band(S)
            pnoun = p.split()[-1] if isinstance(p, str) else p
            return p, hedge_render(band, reply, pnoun)

    return HedgingFluidChat


# ----------------------------------------------------------------------------- console-integration check (stub faculty; FT ckpt absent on this box)
def run_console_check(seed=42):
    """Verify the console wiring WITHOUT the FT generator (its ckpt is absent on this Linux migration): a deterministic
    stub faculty replaces the ~21M ANN so FluidChat builds. This isolates exactly what hedging touches -- the moat
    routing + the hedge wrap + byte-identity -- none of which depend on the generator's text. Checks:
      (a) enable_hedging OFF => _answer is byte-identical to the base console;
      (b) MOAT: an untaught cue => hard 'I don't know', on BOTH off and on;
      (c) enable_hedging ON at the console op-point => calibrates to assert-everything (grounded facts still asserted,
          nothing to hedge -- the honest un-stressed-op-point behavior); the moat is unchanged.
    """
    import research.runners._fluidconv_chat_repl as cr

    class _StubFaculty:
        device = "cpu"
        npar = 0.0

        class _P:  # a minimal parser stand-in with the attribute the console reads
            pass
        npar = 0.0

        def __init__(self):
            self.npar = 0.0
            self.device = "cpu"
            self.parser = None

        def answer(self, ctx, q):
            # deterministic: echo the grounded context (what the RA generator is asked to render) -> stable text
            return ctx.strip().rstrip(" .") + " ."

    orig = cr.FTFaculty
    cr.FTFaculty = _StubFaculty
    try:
        HedgingFluidChat = make_hedging_console_class()
        base = cr.FluidChat(seed=seed)                              # the shipped console (stub faculty)
        off = HedgingFluidChat(seed=seed, enable_hedging=False)     # hedging OFF
        on = HedgingFluidChat(seed=seed, enable_hedging=True)       # hedging ON
        cal = on.calibrate_here()

        # a set of grounded cues from the curriculum + untaught cues
        grounded = [(a, v) for (a, v, p) in [tuple(f) for f in base.store_keys if len(f) == 3]][:8]
        untaught = [("lion", "eat"), ("dragon", "chase"), ("zzz", "like")]

        # (a) OFF byte-identical to base -- at BOTH the _answer seam AND the full turn() level (the whole console), since
        # HedgingFluidChat overrides only _answer and only sets the composer trace when hedging is ON.
        off_byte_id = True
        for (a, v) in grounded:
            if base._answer(a, v) != off._answer(a, v):
                off_byte_id = False
        turn_probes = ["what does the dog eat?", "who eats meat?", "does the dog eat meat?",
                       "the wolf eats rabbit", "what does the wolf eat?", "what does the dragon eat?",
                       "tell me about the dog"]
        for t in turn_probes:
            if base.turn(t) != off.turn(t):
                off_byte_id = False
        # (b) MOAT byte-identical (hard IDK) on untaught, both off and on
        moat_ok = True
        for (a, v) in untaught:
            pb, rb = base._answer(a, v)
            po, ro = off._answer(a, v)
            pn, rn = on._answer(a, v)
            if not (pb is None and rb == "I don't know." and (po, ro) == (pb, rb) and pn is None and rn == "I don't know."):
                moat_ok = False
        # (c) ON at console op-point: grounded facts still asserted (calibrated to assert-everything); show a sample
        on_sample = []
        for (a, v) in grounded[:4]:
            p, r = on._answer(a, v)
            on_sample.append({"cue": f"{a} {v}", "reply": r})
        return dict(seed=seed, off_byte_identical=off_byte_id, moat_ok=moat_ok,
                    console_cal=cal.as_dict(), on_sample=on_sample,
                    all_ok=bool(off_byte_id and moat_ok))
    finally:
        cr.FTFaculty = orig


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anti-cheats", action="store_true", help="run the full GO gate at a stressed op-point (3 seeds)")
    ap.add_argument("--demo", action="store_true", help="print a graded transcript (assert/I think/not certain/soft-abstain/moat)")
    ap.add_argument("--console-check", action="store_true", help="verify console integration (stub faculty; off byte-identical + moat)")
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--n-facts", type=int, default=48)
    ap.add_argument("--vocab-mode", choices=["synthetic", "themed"], default="synthetic")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--out", default=str(_REPO / "research" / "findings" / "raw" / "_fluidconv_graded_hedging_report.json"))
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    report = {"D": a.D, "n_facts": a.n_facts, "seeds": seeds}

    if a.demo or not (a.anti_cheats or a.console_check):
        print("=== GRADED-HEDGING DEMO (stressed op-point; the graded middle is populated) ===")
        txt, cal = run_demo(seed=seeds[0], D=a.D, n_facts=a.n_facts)
        print(txt)
        report["demo"] = txt

    if a.anti_cheats:
        print(f"\n=== ANTI-CHEATS (held-out eval, stressed op-point D={a.D}, vocab={a.vocab_mode}) ===")
        rows = []
        for s in seeds:
            r = run_anti_cheats(s, D=a.D, n_facts=a.n_facts, vocab_mode=a.vocab_mode)
            rows.append(r)
            print(f"\n-- seed {s} (D={r['D']}, eval n={r['n_eval']}, unknown n={r['n_unknown']}) --")
            print(f"   calibrator: t1={r['cal']['t1']:.3f} t2={r['cal']['t2']:.3f} t3={r['cal']['t3']:.3f}  "
                  f"band_counts={r['band_counts']}  band_acc={r['band_acc']}")
            print(f"   [MOAT]              abstain_leak={r['moat']['abstain_leak']} gatefirst_invariant={r['moat']['gatefirst_invariant']} "
                  f"(intrinsic_composer_FA={r['moat']['intrinsic_composer_false_accepts']}, of those demoted->L2/L3={r['moat']['of_those_demoted_to_L2L3']}) -> {r['moat']['ok']}")
            print(f"   [OFF byte-ident]    {r['off_byte_identical']}")
            print(f"   [hedge monotone]    spearman(-S,level)={r['hedge_monotone']['spearman_negS_level']} -> {r['hedge_monotone']['ok']}")
            print(f"   [no extra c-wrong]  baseline_wrong={r['no_extra_confident_wrong']['baseline_wrong']} "
                  f"assertive_wrong={r['no_extra_confident_wrong']['assertive_wrong']} "
                  f"(n_assertive={r['no_extra_confident_wrong']['n_assertive']}) -> {r['no_extra_confident_wrong']['ok']}")
            print(f"   [graceful abstain]  matched_flat_idk={r['graceful_soft_abstain']['matched_flat_idk']} "
                  f"n_L3={r['graceful_soft_abstain']['n_L3']} l3_surfaces_p={r['graceful_soft_abstain']['l3_surfaces_candidate']} -> {r['graceful_soft_abstain']['ok']}")
            print(f"   [permuted-map]      true_band_acc_spearman={r['permuted_map']['true_band_acc_spearman']} "
                  f"permuted_mean={r['permuted_map']['permuted_mean']} (n_wrong_eval={r['permuted_map']['n_wrong_eval']}) -> {r['permuted_map']['ok']}")
            print(f"   ALL_OK: {r['all_ok']}")
        report["anti_cheats"] = rows
        # POOLED-across-seeds permuted-map + hedge-monotone (robust: enough wrong answers to separate). This is the
        # headline gate for the S->band structure; per-seed is context (N/A when a split has <2 wrong).
        pool = [row for r in rows for row in r["ev_rows"]]                     # (S, band_level, correct)
        p_S = [x[0] for x in pool]; p_lvl = [x[1] for x in pool]; p_cor = [1.0 if x[2] else 0.0 for x in pool]
        pooled_mono = _spearman([-s for s in p_S], p_lvl)
        p_true, p_perm, p_perm_ok = _permuted_map_stat(p_lvl, p_cor, seed=1234)
        n_wrong_pool = int(sum(1 for c in p_cor if c == 0.0))
        # pooled no-extra-confident-wrong: L0/L1 (assertive, level<=1) wrong-rate vs the un-hedged baseline
        base_wrong_pool = 1.0 - float(np.mean(p_cor)) if pool else float("nan")
        asrt = [1.0 - c for (c, l) in zip(p_cor, p_lvl) if l <= 1]
        asrt_wrong_pool = float(np.mean(asrt)) if asrt else 0.0
        no_extra_pool_ok = (asrt_wrong_pool <= base_wrong_pool + 1e-9)
        print(f"\n-- POOLED (3 seeds, n={len(pool)}, n_wrong={n_wrong_pool}) --")
        print(f"   [pooled hedge monotone]  spearman(-S,level)={round(pooled_mono,3) if pooled_mono is not None else None}")
        print(f"   [pooled permuted-map]    true_band_acc_spearman={p_true} permuted_mean={p_perm} -> {p_perm_ok}")
        print(f"   [pooled no-extra-wrong]  assertive(L0/L1)_wrong={round(asrt_wrong_pool,3)} <= baseline_wrong={round(base_wrong_pool,3)} "
              f"(n_assertive={len(asrt)}) -> {no_extra_pool_ok}")
        report["pooled"] = {"n": len(pool), "n_wrong": n_wrong_pool,
                            "hedge_monotone_spearman": round(pooled_mono, 3) if pooled_mono is not None else None,
                            "permuted_map": {"true": p_true, "permuted_mean": p_perm, "ok": p_perm_ok},
                            "no_extra_confident_wrong": {"assertive_wrong": round(asrt_wrong_pool, 3),
                                                         "baseline_wrong": round(base_wrong_pool, 3),
                                                         "n_assertive": len(asrt), "ok": no_extra_pool_ok}}
        gates = {
            # STRUCTURAL gates (guaranteed by gate-first / threshold construction) -- per-seed, all 3 seeds:
            "moat_ok": all(x["moat"]["ok"] for x in rows),
            "off_byte_identical": all(x["off_byte_identical"] for x in rows),
            "hedge_monotone": all(x["hedge_monotone"]["ok"] for x in rows) and (pooled_mono is not None and pooled_mono > 0),
            "graceful_soft_abstain": all(x["graceful_soft_abstain"]["ok"] for x in rows),
            # STATISTICAL gates (need power; the POOLED test is the sound one -- per-seed n_wrong~5-9 is underpowered at
            # the substrate's AUC~0.85, so a marginal single-answer per-seed non-monotonicity is expected + reported):
            "permuted_map": bool(p_perm_ok),
            "no_extra_confident_wrong": bool(no_extra_pool_ok),
        }
        gates["VERDICT"] = "GO" if all(gates.values()) else "PARTIAL"
        report["gates_3seed"] = gates
        print(f"\n=== 3-SEED GATES ===\n{json.dumps(gates, indent=2)}")

        # DEPLOYMENT-op-point moat (D=256, the console phasor dimension): the intrinsic moat is CLEAN there (the low-D
        # leak is a stress artifact), matched facts assert, hedging gate-first invariant.
        print("\n=== DEPLOYMENT MOAT (D=256, console op-point) ===")
        dep = [deployment_moat_check(s, D=256, n_facts=a.n_facts) for s in seeds]
        for d in dep:
            print(f"-- seed {d['seed']}: acc={d['accuracy']} intrinsic_composer_FA={d['intrinsic_composer_false_accepts']} "
                  f"abstain_leak={d['abstain_leak']} gatefirst_invariant={d['gatefirst_invariant']} "
                  f"matched_all_assert={d['matched_all_assert']} -> ok={d['ok']}")
        report["deployment_moat"] = dep
        report["deployment_moat_ok"] = all(d["ok"] for d in dep)

        # THEMED-vocab moat ROBUSTNESS (constrained 8-verb vocab at D=16 -> the SUBSTRATE itself leaks a few %): shows
        # hedging stays gate-first invariant (adds 0 FA) AND demotes the low-S substrate false-matches toward L2/L3.
        print("\n=== MOAT ROBUSTNESS under substrate leak (themed vocab, D=16) ===")
        th = [run_anti_cheats(s, D=16, n_facts=a.n_facts, vocab_mode="themed") for s in seeds]
        for r in th:
            m = r["moat"]
            print(f"-- seed {r['seed']}: intrinsic_composer_FA={m['intrinsic_composer_false_accepts']} "
                  f"(of those demoted->L2/L3={m['of_those_demoted_to_L2L3']}) abstain_leak={m['abstain_leak']} "
                  f"gatefirst_invariant={m['gatefirst_invariant']} -> moat_ok(hedging-not-weakened)={m['ok']}")
        report["themed_moat_robustness"] = th
        report["themed_moat_ok"] = all(r["moat"]["ok"] for r in th)

    if a.console_check:
        print("\n=== CONSOLE INTEGRATION CHECK (stub faculty; FT ckpt absent on this box) ===")
        rows = []
        for s in seeds:
            r = run_console_check(seed=s)
            rows.append(r)
            print(f"-- seed {s}: off_byte_identical={r['off_byte_identical']} moat_ok={r['moat_ok']} "
                  f"console_cal(t1={r['console_cal']['t1']:.3f}) -> all_ok={r['all_ok']}")
            for smp in r["on_sample"]:
                print(f"     [hedging ON] {smp['cue']:>12} -> {smp['reply']}")
        report["console_check"] = rows
        report["console_gates_3seed"] = {
            "off_byte_identical": all(x["off_byte_identical"] for x in rows),
            "moat_ok": all(x["moat_ok"] for x in rows),
        }
        print(f"\n=== CONSOLE GATES ===\n{json.dumps(report['console_gates_3seed'], indent=2)}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[saved] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
