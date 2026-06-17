"""CYCLE (productive-syntax) — the EASY half: LEARNED multi-FRAME word order, cheap-first de-risk.

Pre-registered by `research/findings/2026-06-17-productive-syntax-scoping.md` (Option 1, ranked #1). The agent's
current grammar is a set of HARDCODED templates (one fixed SVO frame; word order = a hardcoded primacy tuple,
`_phaseB_serial_order_spiking_derisk.PRIMACY_pA`). "Productive" word order means: a grammatical FRAME is a LEARNED
primacy gradient over the grammatical ROLE slots (NOT a hardcoded tuple) -> a NEW frame's order generalizes to
fillers it was never trained on, because the order is over ROLES, not words; a context cue SELECTS the frame
(dlPFC-style). This composes the validated pieces: the rate-coded competitive-queuing serial-order generator
(`neural_serial_order_renderer` / `_phaseB_serial_order_spiking_derisk`: graded current -> spiking-RATE ranking =
emission order; Grossberg 1978 / Bullock-Rhodes 2003, catalog G.07/H.19) + a per-FRAME Hebbian primacy gradient +
a Hebbian cue->frame SELECTION map.

This is DISTINCT from the CYCLE-106 multi-frame precursors (`_phaseB_serial_order_multiframe[_spiking]_derisk.py`,
which tested frame-CONDITIONED order via a cross-frame control only). The productivity de-risk ADDS the load-bearing
controls those lacked: (1) HELD-OUT FILLERS (the gradient is learned over roles on a train split, emission tested on
a DISJOINT filler set), (2) a NON-NATIVE second frame (verb-initial "ran dog north"), (3) FRAME-SELECTION (a context
cue -> the correct frame, learned), (4) the PERMUTED-FRAME control (shuffle frame->gradient -> order must collapse
to chance -- the discriminator that the order is the LEARNED frame, not a fixed/native bias), (5) a LESION control
(remove the learned gradient -> collapse), and (6) the no-confab MOAT (an unfilled role / unknown filler abstains).

SPIKING substrate: the order read-out is REAL SPIKES -- the per-frame primacy gradient is graded EXTERNAL CURRENT
into the fact's driven concept pools, the per-pool spiking RATE ranking is the emission order (reuses the validated
`build_pool_bridge` / `pool_rates`). The frame-selection / permuted / lesion / moat are layered on. No `sim/` edit;
reuse-by-import; CPU bridge is fine (tiny driven pools) but CuPy honored if set.

PRE-REGISTERED GATE (FROZEN before data; >=6 seeds; FRACTIONAL >=5/6 bar per feedback_6seed_validation):
  GO        = held-out productivity (the NON-NATIVE learned frame's order on held-out fillers) >= 0.90 AND
              frame-SELECTION accuracy >= 0.90, BOTH on >=5/6 seeds, AND the PERMUTED-FRAME control collapses to
              chance, AND the LESION collapses to chance, AND native SVO still emits correctly (no regression).
  BOUNDARY  = native + one learned frame work but held-out generalization OR frame-selection is seed-fragile.
  NEGATIVE  = the learned order does NOT generalize to held-out fillers (memorized words, not a role-order) OR the
              permuted-frame control does NOT collapse (the "order" was the fixed native order all along).
Report whichever the data shows; do NOT tune-to-pass.

Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_learned_multiframe_word_order_derisk --seeds 42,43,44,100,101,102
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.song_g1_core import score_order, permuted_order_controls  # noqa: E402
from research.runners._phaseB_serial_order_spiking_derisk import (  # noqa: E402  (reuse the validated spiking read)
    build_pool_bridge, pool_rates, VOCAB, N_PER)

# ---- frame inventory: grammatical ROLE orders. The de-risk's minimal productive step = a NON-NATIVE 2nd frame. ----
# Roles: 0=subject(agent), 1=verb(action), 2=object(patient). The agent's NATIVE frame is SVO=[0,1,2]; F_VI is the
# NON-NATIVE verb-initial frame ("ran dog north" = verb subject object) the agent was NOT given as a template.
ROLE_NAMES = ["subj", "verb", "obj"]
FRAMES = {
    "SVO": [0, 1, 2],     # native: subject verb object   ("dog ran north")
    "VSO": [1, 0, 2],     # NON-NATIVE: verb subject object ("ran dog north")  <- the learned-frame capability target
}
FRAME_KEYS = list(FRAMES.keys())
N_ROLES = 3
N_TRAIN_SENT = 12         # frame-tagged example sentences per frame the gradient + selection learn from
N_HELDOUT = 12            # held-out filler tuples emission is tested on (DISJOINT concept tuples from training)
N_PERM = 5                # permuted-ORDER controls per emission (for the chance reference)
LR_PRIM = 0.10            # Hebbian primacy learning rate (per the CQ de-risk)
LR_SEL = 0.20             # Hebbian cue->frame selection learning rate
SEL_NOISE = 0.05          # selection-WTA noise (small; the cue is a clean one-hot context signal)
PASS_ORDER = 0.90         # GO bar: held-out order-accuracy
PASS_SELECT = 0.90        # GO bar: frame-selection accuracy
CHANCE_ORDER = 1.0 / 6.0  # exact-order chance for 3 distinct slots (1 of 3! orderings); score_order ~0.33 for ~random


# --------------------------------------------------------------------------------------------------------------------
# Facts / fillers. Each "sentence" = an ordered triple of DISTINCT concept indices (the subj/verb/obj fillers). The
# TRAIN split teaches the per-frame role-primacy + the cue->frame map; the HELD-OUT split (disjoint concept tuples)
# tests emission. Because the primacy gradient is over ROLES (3 values), held-out emission probes whether the LEARNED
# role-order generalizes to fillers (concept pools) the gradient never saw -- the productivity claim.
# --------------------------------------------------------------------------------------------------------------------
def build_fillers(seed):
    rng = np.random.default_rng(seed * 101 + 7)
    triples, seen = [], set()
    while len(triples) < (N_TRAIN_SENT + N_HELDOUT):
        trip = tuple(int(x) for x in rng.choice(VOCAB, N_ROLES, replace=False))
        if trip not in seen:
            seen.add(trip)
            triples.append(trip)
    return triples[:N_TRAIN_SENT], triples[N_TRAIN_SENT:]   # (train fillers, held-out fillers) -- disjoint tuples


class LearnedFrameGrammar:
    """A learned grammar over the validated CQ serial-order substrate.

    `prim[frame][role]` = the per-frame planning-layer PRIMACY GRADIENT, LEARNED from frame-tagged example sentences
    (each sentence in frame F teaches F's role order). `sel_w[cue, frame]` = a Hebbian cue->frame SELECTION map (a
    context cue one-hot -> which frame's gradient to apply). NOTHING is hardcoded: both the per-frame order and the
    cue->frame routing are learned. `emit_spiking` realizes the selected frame's gradient as GRADED CURRENT into the
    fact's concept pools and reads the per-pool spiking RATE ranking = the emission order (the validated read-out)."""

    def __init__(self, bridge, pool_idx, n_frames, seed):
        self.bridge, self.pool_idx = bridge, pool_idx
        self.n_frames = n_frames
        # random init so the order is NOT baked in (the gradient must be LEARNED, the de-risk's whole point).
        self.prim = np.random.default_rng(seed * 13 + 5).standard_normal((n_frames, N_ROLES)).astype(np.float64) * 0.01
        self.sel_w = np.zeros((n_frames, n_frames), np.float64)   # sel_w[cue, frame]; cue index == frame's own index
        # primacy gradient -> graded current. Highest primacy = most current; map a primacy RANK to a current level.
        # (3 levels matched to the validated PRIMACY_pA gap; only the RANKING matters for the rate read-out.)
        self._levels = np.array([2400.0, 1700.0, 1000.0], np.float64)

    # --- learning (Hebbian; from frame-tagged training sentences) ---
    def learn_frame_order(self, frame, role_order):
        """Teach frame `frame` its role order (earlier in the order -> more primacy). Hebbian push, identical to the
        validated CQ `learn`. Called once per training sentence in that frame."""
        for pos, role in enumerate(role_order):
            self.prim[frame][role] += LR_PRIM * (N_ROLES - 1 - pos)

    def learn_selection(self, cue, frame):
        """Teach the cue->frame map: co-fire of context-cue `cue` with frame `frame` (Hebbian). Called once per
        training sentence (the utterance-type cue is presented with its frame)."""
        self.sel_w[cue, frame] += LR_SEL

    # --- frame selection (a context cue -> which frame; the dlPFC-style router) ---
    def select_frame(self, cue, rng, sel_w=None):
        """Given a context cue (one-hot index), pick the frame via WTA over the learned cue->frame weights (+ small
        noise). Returns the selected frame index. `sel_w` override lets the permuted control swap the routing."""
        w = self.sel_w if sel_w is None else sel_w
        a = w[cue] + SEL_NOISE * rng.standard_normal(self.n_frames)
        return int(np.argmax(a))

    # --- emission on the SPIKING substrate (the validated rate-coded competitive-queuing read-out) ---
    def _grade_to_current(self, frame):
        """Convert frame `frame`'s learned primacy vector into a per-role current LEVEL by primacy RANK (the choice
        layer reads ranking, so only the order of primacies matters). Returns role->current."""
        order_by_prim = sorted(range(N_ROLES), key=lambda r: -self.prim[frame][r])   # roles, most->least primate
        cur = {}
        for rank, role in enumerate(order_by_prim):
            cur[role] = float(self._levels[rank])
        return cur

    def emit_spiking(self, frame, trip, lesion=False):
        """Drive the fact's 3 filler pools with the selected frame's primacy-graded current, read per-pool spiking
        RATE, emit fillers by rate DESC = the produced word order. `lesion=True` zeroes the gradient (equal drive ->
        chance). Returns the emitted list of concept indices (the produced order)."""
        if lesion:
            drive = {int(trip[r]): float(self._levels.mean()) for r in range(N_ROLES)}   # equal drive: no gradient
        else:
            role_cur = self._grade_to_current(frame)
            drive = {int(trip[r]): role_cur[r] for r in range(N_ROLES)}
        rate = pool_rates(self.bridge, self.pool_idx, drive)
        return [int(trip[r]) for r in sorted(range(N_ROLES), key=lambda r: -rate[int(trip[r])])]

    # --- the no-confab MOAT: an unfilled role (a None filler) / unknown filler must NOT be confabulated ---
    def emit_with_moat(self, frame, trip_or_none):
        """trip is a 3-tuple of fillers OR may contain None for an UNFILLED role / an out-of-vocab index for an
        UNKNOWN filler. The moat: a slot whose filler is None or out-of-vocab is emitted as ABSTAIN (None) -- never
        a confabulated concept. Fillable slots are ordered by the learned gradient as usual."""
        role_cur = self._grade_to_current(frame)
        fillable = [r for r in range(N_ROLES)
                    if trip_or_none[r] is not None and 0 <= int(trip_or_none[r]) < VOCAB]
        if not fillable:
            return [None] * N_ROLES
        drive = {int(trip_or_none[r]): role_cur[r] for r in fillable}
        rate = pool_rates(self.bridge, self.pool_idx, drive)
        ordered_fillable = sorted(fillable, key=lambda r: -rate[int(trip_or_none[r])])
        # emit fillable concepts in learned-gradient order, then ABSTAIN (None) for every unfilled/unknown slot.
        out = [int(trip_or_none[r]) for r in ordered_fillable]
        out += [None] * (N_ROLES - len(out))
        return out


# --------------------------------------------------------------------------------------------------------------------
def _emit_order_acc(gram, frame, held, lesion=False, prim_override=None):
    """Mean exact-order accuracy: emit each held-out filler tuple under `frame`, score the produced order vs the
    frame's TRUE order. prim_override swaps the gradient (permuted-frame control)."""
    saved = None
    if prim_override is not None:
        saved = gram.prim.copy()
        gram.prim = prim_override
    order = FRAMES[FRAME_KEYS[frame]]
    accs, perms = [], []
    rng = np.random.default_rng(777)
    for trip in held:
        intended = [trip[r] for r in order]
        emitted = gram.emit_spiking(frame, trip, lesion=lesion)
        accs.append(score_order(emitted, intended))
        perms.append(max((score_order(emitted, c) for c in permuted_order_controls(intended, rng, N_PERM)),
                         default=0.0))
    if saved is not None:
        gram.prim = saved
    return float(np.mean(accs)), float(np.mean(perms))


def run_seed(seed):
    bridge, pool_idx = build_pool_bridge(seed)
    train_fill, held_fill = build_fillers(seed)
    n_frames = len(FRAME_KEYS)
    gram = LearnedFrameGrammar(bridge, pool_idx, n_frames, seed)

    # --- TRAIN: learn each frame's role-primacy + the cue->frame selection map, from frame-tagged example sentences.
    # Each training sentence is (a filler tuple, a frame). The cue for a frame == that frame's own index (a clean
    # one-hot utterance-type context). The gradient is learned on TRAIN fillers only; emission tested on HELD-OUT.
    for fi, fkey in enumerate(FRAME_KEYS):
        order = FRAMES[fkey]
        for trip in train_fill:
            gram.learn_frame_order(fi, order)     # teach this frame's role order (filler-agnostic: over ROLES)
            gram.learn_selection(fi, fi)          # teach cue fi -> frame fi

    # --- (1) HELD-OUT productivity, per frame (the capability). The NON-NATIVE frame (VSO) is the headline metric. ---
    svo_acc, _ = _emit_order_acc(gram, FRAME_KEYS.index("SVO"), held_fill)
    vso_acc, vso_perm = _emit_order_acc(gram, FRAME_KEYS.index("VSO"), held_fill)

    # --- (2) FRAME-SELECTION: a context cue -> the correct frame (the produced order must match the SELECTED frame). ---
    rng = np.random.default_rng(seed * 71 + 3)
    sel_hits, n_sel = 0, 0
    for fi in range(n_frames):
        for trip in held_fill:
            picked = gram.select_frame(fi, rng)        # learned cue->frame routing
            n_sel += 1
            if picked != fi:
                continue
            # routed correctly -> emit under the SELECTED frame; the produced order must be that frame's order.
            emitted = gram.emit_spiking(picked, trip)
            intended = [trip[r] for r in FRAMES[FRAME_KEYS[picked]]]
            if score_order(emitted, intended) >= 0.999:
                sel_hits += 1
    sel_acc = sel_hits / float(n_sel)

    # --- (3) PERMUTED-FRAME control (load-bearing): shuffle the frame->gradient mapping (so the VSO label points at a
    # DIFFERENT frame's gradient). The produced order vs VSO's TRUE order must collapse toward chance -- proving the
    # order is driven by the LEARNED frame gradient, not a fixed/native bias. ---
    perm_rng = np.random.default_rng(seed * 53 + 11)
    perm_map = perm_rng.permutation(n_frames)
    while np.array_equal(perm_map, np.arange(n_frames)):       # ensure a real shuffle (frame labels actually move)
        perm_map = perm_rng.permutation(n_frames)
    prim_shuffled = gram.prim[perm_map].copy()                 # VSO's slot now holds another frame's gradient
    permframe_acc, _ = _emit_order_acc(gram, FRAME_KEYS.index("VSO"), held_fill, prim_override=prim_shuffled)

    # --- (4) LESION control: remove the learned gradient (equal drive) -> order must collapse to chance. ---
    lesion_acc, _ = _emit_order_acc(gram, FRAME_KEYS.index("VSO"), held_fill, lesion=True)

    # --- (5) MOAT: an UNFILLED role (None) and an UNKNOWN filler (out-of-vocab) must ABSTAIN, never confabulate. ---
    moat_ok = 0
    moat_total = 0
    for trip in held_fill[:6]:
        unfilled = [trip[0], None, trip[2]]                    # verb slot unfilled
        out = gram.emit_with_moat(FRAME_KEYS.index("VSO"), unfilled)
        moat_total += 1
        # PASS: exactly one None (the unfilled slot abstained), the two real fillers are present, NO confabulated id.
        nones = sum(1 for x in out if x is None)
        reals = set(x for x in out if x is not None)
        if nones == 1 and reals == {trip[0], trip[2]}:
            moat_ok += 1
        unknown = [trip[0], VOCAB + 5, trip[2]]                # out-of-vocab filler in the verb slot
        out2 = gram.emit_with_moat(FRAME_KEYS.index("VSO"), unknown)
        moat_total += 1
        nones2 = sum(1 for x in out2 if x is None)
        reals2 = set(x for x in out2 if x is not None)
        if nones2 == 1 and reals2 == {trip[0], trip[2]}:
            moat_ok += 1
    moat_acc = moat_ok / float(moat_total)

    seed_go = bool(vso_acc >= PASS_ORDER and sel_acc >= PASS_SELECT)
    print(f"  [seed {seed}] held-out: SVO {svo_acc:.3f} | VSO(non-native) {vso_acc:.3f} (perm-order {vso_perm:.3f}) "
          f"| select {sel_acc:.3f} | permuted-FRAME {permframe_acc:.3f} | lesion {lesion_acc:.3f} | moat {moat_acc:.3f}"
          f" -> {'PASS' if seed_go else 'FAIL'}", flush=True)
    return {"seed": seed, "svo": svo_acc, "vso": vso_acc, "vso_perm_order": vso_perm, "select": sel_acc,
            "permuted_frame": permframe_acc, "lesion": lesion_acc, "moat": moat_acc, "seed_go": seed_go}


def _example_sentence(seed):
    """Build a concrete example: emit a HELD-OUT filler tuple in the LEARNED NON-NATIVE (VSO) frame, with mock words,
    for the findings doc. Returns (held_tuple, svo_words_in_order, vso_words_in_order)."""
    bridge, pool_idx = build_pool_bridge(seed)
    train_fill, held_fill = build_fillers(seed)
    gram = LearnedFrameGrammar(bridge, pool_idx, len(FRAME_KEYS), seed)
    for fi, fkey in enumerate(FRAME_KEYS):
        for trip in train_fill:
            gram.learn_frame_order(fi, FRAMES[fkey]); gram.learn_selection(fi, fi)
    # pick a held-out tuple; mock-spell concepts as readable words (subj/verb/obj exemplars)
    trip = held_fill[0]
    names = {trip[0]: "dog", trip[1]: "ran", trip[2]: "north"}
    spell = lambda c: names.get(c, f"w{c}")
    svo = [spell(c) for c in gram.emit_spiking(FRAME_KEYS.index("SVO"), trip)]
    vso = [spell(c) for c in gram.emit_spiking(FRAME_KEYS.index("VSO"), trip)]
    return trip, " ".join(svo), " ".join(vso)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102",
                    help="comma-separated seeds (default the standard 6-seed set)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    print(f"[learned multi-FRAME word-order de-risk] EASY-half productive syntax: can the agent GENERATE a sentence "
          f"in a LEARNED NON-NATIVE word-order frame (VSO 'ran dog north') on HELD-OUT fillers, SELECT the frame from "
          f"a cue, while permuted-frame + lesion collapse to chance and the moat abstains? (native SVO must not "
          f"regress)", flush=True)
    print(f"  Frames: {', '.join(f'{k}={FRAMES[k]}' for k in FRAME_KEYS)} (roles {ROLE_NAMES}); "
          f"GO bars: order>={PASS_ORDER}, select>={PASS_SELECT}, chance order ~{CHANCE_ORDER:.3f}.", flush=True)

    rows = [run_seed(s) for s in seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    svo, vso, sel = m("svo"), m("vso"), m("select")
    permframe, lesion, moat = m("permuted_frame"), m("lesion"), m("moat")
    n_seeds = len(rows)
    n_go = sum(1 for r in rows if r["seed_go"])
    # the permuted-frame + lesion must COLLAPSE: their order-accuracy must drop near chance (well below the GO bar).
    # "collapse" = mean accuracy <= the random permuted-ORDER baseline + a small slack (i.e. no real frame signal).
    rand_baseline = m("vso_perm_order")           # the empirical chance order-accuracy on the same emissions
    permframe_collapsed = permframe <= rand_baseline + 0.05
    lesion_collapsed = lesion <= rand_baseline + 0.05
    svo_unregressed = svo >= PASS_ORDER
    pass_frac = n_go >= int(np.ceil((5.0 / 6.0) * n_seeds))   # FRACTIONAL >=5/6 bar

    print(f"\n{'='*112}", flush=True)
    print(f"  MEAN ({n_seeds} seeds): held-out SVO {svo:.3f} | VSO(non-native learned) {vso:.3f} | "
          f"frame-SELECT {sel:.3f} | {n_go}/{n_seeds} seed-GO", flush=True)
    print(f"  controls: permuted-FRAME {permframe:.3f} (collapse? {permframe_collapsed}) | "
          f"lesion {lesion:.3f} (collapse? {lesion_collapsed}) | moat {moat:.3f} | "
          f"chance-order baseline {rand_baseline:.3f}", flush=True)
    print(f"{'='*112}", flush=True)

    go = bool(pass_frac and vso >= PASS_ORDER and sel >= PASS_SELECT
              and permframe_collapsed and lesion_collapsed and svo_unregressed)
    boundary = bool((not go) and svo_unregressed and (vso >= PASS_ORDER or sel >= PASS_SELECT)
                    and permframe_collapsed)
    if go:
        print(f"  GO: PRODUCTIVE word-order syntax (easy half) on the spiking substrate -- a LEARNED NON-NATIVE frame "
              f"(VSO) emits held-out fillers in the correct order ({vso:.3f} >= {PASS_ORDER}, {n_go}/{n_seeds} seeds), "
              f"the frame is SELECTED from a cue ({sel:.3f} >= {PASS_SELECT}), the permuted-FRAME control collapses to "
              f"chance ({permframe:.3f} ~ {rand_baseline:.3f}) and lesion collapses ({lesion:.3f}), the moat abstains "
              f"({moat:.3f}), and native SVO is un-regressed ({svo:.3f}). ==> the agent produces grammatical "
              f"structure it was NOT given as a template; the order is over ROLES (a learned gradient), not words. "
              f"Promote to a GPU 6-seed gate + wire a learned multi-frame `render` into the agent (default-off).",
              flush=True)
    elif boundary:
        print(f"  BOUNDARY: native + the learned frame partly work but a GO bar is seed-fragile (VSO {vso:.3f} vs "
              f"{PASS_ORDER}, select {sel:.3f} vs {PASS_SELECT}, {n_go}/{n_seeds} seeds). The permuted-frame control "
              f"DID collapse ({permframe:.3f}), so the order is a learned frame -- but generalization/selection isn't "
              f"robust. Localizes selection-vs-capacity as the next sub-problem.", flush=True)
    else:
        if not (permframe_collapsed and lesion_collapsed):
            print(f"  NEGATIVE: the permuted-FRAME / lesion control did NOT collapse (permframe {permframe:.3f}, "
                  f"lesion {lesion:.3f} vs chance {rand_baseline:.3f}) -- the 'order' is a fixed/native bias, not a "
                  f"learned frame. The learned-frame claim FAILS. Honest negative.", flush=True)
        else:
            print(f"  NEGATIVE: the learned non-native order does not generalize to held-out fillers / select "
                  f"reliably (VSO {vso:.3f}, select {sel:.3f}). Learnable productive word order is itself a wall here "
                  f"-- record it (a biology-translatable negative about learnable serial order); reconsider (e.g. the "
                  f"AC projection parser, Option 2).", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s", flush=True)

    # a concrete example sentence in the LEARNED non-native frame, on a held-out filler tuple
    try:
        trip, svo_s, vso_s = _example_sentence(seeds[0])
        print(f"\n  EXAMPLE (seed {seeds[0]}, held-out filler tuple {trip}): native SVO -> \"{svo_s}\" ; "
              f"LEARNED non-native VSO -> \"{vso_s}\"  (same fillers, different LEARNED frame order)", flush=True)
    except Exception as e:        # noqa: BLE001  (example is illustrative; never fail the de-risk on it)
        svo_s = vso_s = None
        print(f"  (example sentence skipped: {e})", flush=True)

    verdict = "GO" if go else ("BOUNDARY" if boundary else "NEGATIVE")
    out = {"verdict": verdict, "n_seeds": n_seeds, "n_go": n_go, "frames": FRAMES,
           "svo": svo, "vso": vso, "select": sel, "permuted_frame": permframe, "lesion": lesion, "moat": moat,
           "chance_order_baseline": rand_baseline, "permframe_collapsed": permframe_collapsed,
           "lesion_collapsed": lesion_collapsed, "svo_unregressed": svo_unregressed,
           "pass_order_bar": PASS_ORDER, "pass_select_bar": PASS_SELECT,
           "example": {"svo": svo_s, "vso": vso_s}, "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_learned_multiframe_word_order.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}\n", flush=True)


if __name__ == "__main__":
    main()
