"""RANK-13 PRODUCTION-FLIP verification: self/identity + anaphora-miss (Track-1 ship-the-validated-wins).

The mechanism (research/runners/brain_chat_tui.py's `_neural_selfid_enabled` / `_neural_anaphora_abstain_enabled`,
gating BRAIN_NEURAL_SELFID / BRAIN_NEURAL_ANAPHORA_ABSTAIN) earned a 6-seed de-risk GO with both flags default OFF
(research/findings/2026-09-05-rank13-selfid-anaphora-scaffold-derisk-GO-6of6.md). THIS runner verifies the FLIP to
default-ON (`_NEURAL_SELFID_DEFAULT_ON = True` / `_NEURAL_ANAPHORA_ABSTAIN_DEFAULT_ON = True`, already applied in
this worktree) is safe + genuinely load-bearing, per the production-flip charter's three requirements, ALL 6-seed
(42/43/44/100/101/102):

  1. NO REGRESSION: default-ON, the REAL production combiner (the installed GNW ignition bus,
     `gnw_bus_shadow.install_bus_gate` -- the SAME wrapper `webapp/server.py::brain_reply` installs unconditionally)
     still answers STORED / abstains UNSTORED / resolves anaphora-HIT IDENTICALLY to the shipped default-OFF
     baseline, on BOTH composer kinds actually reachable in production (`rf`, the numpy fast-path escape, AND
     `onebrain`, `_COMPOSER_KIND_DEFAULT` -- the TRUE production default, genuinely spiking).
  2. LOAD-BEARING, NOT HOLLOW: (a) VARY -- toggling the flag genuinely changes live behavior: the anaphora-miss
     FINAL ANSWER flips (['dog','chase','cat'] confab -> None abstain), and the self-factual/self-identity answers
     stop being authored by the host router (call-count-measured, not inferred: 0 calls to `_gate_router_combine`
     / `QuestionRouter.match_fact`). A byte-identical outcome flag-on vs flag-off on these covered classes would be
     the HOLLOW failure mode (the affect-hollow-mouth shape) -- checked explicitly below, and it does not hold.
     (b) LESION -- on the genuinely-neural sub-mechanism (class (a), the on-brain BridgeParser), monkeypatching
     `parser.role_of` to a fixed junk role (the 2026-08-12 CHOOSE-1 recipe, unchanged) collapses the self-factual
     answer back to abstain -- the flag-ON gain VANISHES under lesion -- while the parser-INDEPENDENT recall reflex
     (`composer.query_patient`) still returns the fact (dissociation: the substrate still HAS it, only the
     comprehension that would route to it is gone). This is the ONLY class here with a spiking mechanism to lesion
     (self/identity-copula and the anaphora-abstain are host comprehension-helper / control-flow, same honesty
     class as the pre-existing copula/relation-fronted routes -- see the de-risk finding's own scoping).
  3. REAL DEFAULT: both flags resolve ON with the env vars UNSET (not merely settable) -- checked directly, and
     exercised throughout by NEVER setting BRAIN_NEURAL_SELFID/BRAIN_NEURAL_ANAPHORA_ABSTAIN in the "on" arm below.

SEED-PROPAGATION FIX (methodological correction to the original de-risk's own instrument, found by reading the
source, not inferred): `webapp.server._build_chat_brain('tiny-demo', ...)` calls
`_build_tiny_demo(42, use_multiturn=True, enable_neural_render=False, composer_kind=_ck)` with a HARDCODED literal
`42` -- there is no env var or parameter that lets a caller of `_build_chat_brain` vary it. The original de-risk
runner's `_build(seed, flags_on, with_bus)` accepts a `seed` argument but never threads it anywhere, so its own
"6 seeds" loop (both the fast rf battery and the onebrain lesion battery) built the IDENTICAL substrate 6 times --
an instance of the documented cfg.seed trap (CLAUDE.md: "actual_seed_used DOES NOT SEED ANYTHING"), one call-stack
level up. This runner repairs that FOR ITS OWN VERIFICATION ONLY (webapp/server.py is untouched) by monkeypatching
`research.runners.brain_chat_tui._build_tiny_demo` for the duration of each build to thread a REAL, varying seed
through: `_build_chat_brain`'s `from research.runners.brain_chat_tui import _build_tiny_demo` is a LOCAL import
inside the function body, re-executed on every call, so it re-resolves whatever the module attribute currently
points to -- patching it before calling `_build_chat_brain` makes the ACTUAL production entry point (LTM attach,
backend pin, everything) build with the intended seed, while remaining the real production code path, not a
hand-rolled substitute. The de-risk finding's own correctness/retirement/regression/lesion CONCLUSIONS are not
contradicted by this (the logic being tested is deterministic control-flow, not a claim that hinges on
substrate randomness) -- but the "6 seeds" LABEL on that evidence was not doing the substrate-diversity job the
project's 6-seed convention exists for, and this runner's own 6-seed claim would repeat the same gap uncorrected.

SCOPE (honest, matching the de-risk finding's own scoping convention):
  - FAST (rf composer) battery: BOTH combiners (plain host `gate()` + the installed GNW-bus `gate_via_bus`), full
    class coverage (self_factual, self_identity, anaphora_miss, anaphora_hit, STORED, UNSTORED), genuinely-seeded,
    flag-ON exercised via UNSET env (the real default) vs flag-OFF via explicit "0" (the shipped baseline).
  - SLOW (onebrain composer, the TRUE production default) battery: the installed GNW-bus combiner ONLY (that is
    what `brain_reply` actually installs on every real turn -- testing `plain` there too is not what ships), flag-ON
    via UNSET env, genuinely-seeded, all 6 seeds: (i) one build/seed runs the FULL stateless panel (self_factual +
    self_identity + STORED + UNSTORED, call-count-wrapped) THEN reuses the SAME build for the BridgeParser lesion
    (the crux). The anaphora sequences are NOT re-run under onebrain (scope decision, stated plainly): the
    anaphora-abstain control flow does not branch on `parser_present` at all (only class (a)'s EXTRACTION does),
    so it is composer-agnostic by construction (verified by reading `gate()`/`gate_via_bus` -- the anaphora branch
    checks only the resolved patient / bus outcome, never `composer.parser`), and it already gets genuine 6-seed
    coverage from the fast battery above; re-running it under the slow composer would spend ~180s/build/seed
    re-confirming a claim the code structure already makes composer-independent. Likewise the onebrain "OFF"
    (explicit 0/0) arm is not separately rebuilt for regression: STORED/UNSTORED never contain a self-alias token
    and the anaphora classes are excluded from this composer's battery for the reason just given, so there is no
    onebrain-reachable code path left for the flags to perturb -- the regression claim rests on (a) that structural
    read of the source and (b) the fast battery's own genuinely-seeded regression proof, not a second slow pass.

Run (numpy-CPU throughout, no GPU):
  SIM_BACKEND=numpy python -u -m research.runners._rank13_selfid_anaphora_prodflip_verify
  SIM_BACKEND=numpy python -u -m research.runners._rank13_selfid_anaphora_prodflip_verify --skip-onebrain   (fast only)
  SIM_BACKEND=numpy python -u -m research.runners._rank13_selfid_anaphora_prodflip_verify --onebrain-only
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

from tools.lab import attributable_to, void_if   # noqa: E402
from tools.verdict import Verdict   # noqa: E402

SEEDS = [42, 43, 44, 100, 101, 102]

# ── panels (identical fixture to the de-risk finding, so results are directly comparable) ─────────────────────
SELF_FACTUAL = [
    ("what do you use?", ["brain", "use", "spikes"]),
    ("what do you learn?", ["brain", "learn", "words"]),
    ("what do you store?", ["brain", "store", "memory"]),
]
SELF_IDENTITY = [
    ("what are you", ["brain", "use", "spikes"]),
    ("who are you", ["brain", "use", "spikes"]),
]
STORED = [
    ("what does dog chase?", ["dog", "chase", "cat"]),
    ("what does cat eat?", ["cat", "eat", "fish"]),
    ("what does brain learn?", ["brain", "learn", "words"]),
]
UNSTORED = ["what does fish fly?", "what does ball roll?"]
ANAPHORA_HIT_SEQ = [("what does dog chase?", ["dog", "chase", "cat"]), ("what does it eat?", ["cat", "eat", "fish"])]
ANAPHORA_MISS_SEQ = [("what does dog chase?", ["dog", "chase", "cat"]), ("what does it fly?", None)]
# BUILD-COST MERGE (both sequences share the SAME first turn, which establishes the referent; the 2nd turn's
# recalled patient in each -- 'fish' (eat) / nothing (fly, abstain) -- is never itself a known agent, so neither
# turn re-writes the WM referent -- see `ChatBrain._note_referent`'s guard, `p in self.agents_set` -- confirmed by
# reading the source, not assumed): chaining HIT then MISS after ONE shared first turn is behaviorally identical
# to running the two 2-turn sequences in separate builds, at 2/3 the build cost. Verified against the separate-
# build de-risk fixture's own expected answers (identical panel).
ANAPHORA_CHAIN_SEQ = [("what does dog chase?", ["dog", "chase", "cat"]),
                      ("what does it eat?", ["cat", "eat", "fish"]),
                      ("what does it fly?", None)]


def _svo_eq(x, y) -> bool:
    if x is None and y is None:
        return True
    if x is None or y is None:
        return False
    return list(x) == list(y)


@contextlib.contextmanager
def _real_seed(seed: int):
    """Monkeypatch `research.runners.brain_chat_tui._build_tiny_demo` so that
    `webapp.server._build_chat_brain('tiny-demo', ...)`'s hardcoded `_build_tiny_demo(42, ...)` call actually
    builds at `seed` -- see the module docstring's SEED-PROPAGATION FIX. Restores the original on exit
    unconditionally. Does not touch webapp/server.py (which still literally passes 42 as its own local variable);
    it substitutes what that literal reaches at the one indirection point _build_chat_brain uses."""
    import research.runners.brain_chat_tui as bct
    orig = bct._build_tiny_demo

    def _patched(_ignored_seed, *a, **kw):
        return orig(seed, *a, **kw)

    bct._build_tiny_demo = _patched
    try:
        yield
    finally:
        bct._build_tiny_demo = orig


def _set_flags_explicit_off():
    os.environ["BRAIN_NEURAL_SELFID"] = "0"
    os.environ["BRAIN_NEURAL_ANAPHORA_ABSTAIN"] = "0"


def _unset_flags():
    """The REAL default: do not set the env vars at all (requirement 3 -- a real default change)."""
    os.environ.pop("BRAIN_NEURAL_SELFID", None)
    os.environ.pop("BRAIN_NEURAL_ANAPHORA_ABSTAIN", None)


def _build(seed, on: bool, with_bus: bool, composer_kind: str):
    """A FRESH production ChatBrain at a GENUINELY varied seed (`_real_seed`), REAL production composer kind, the
    installed GNW bus when `with_bus` (the actual `webapp/server.py::brain_reply` combiner). `on=True` -> flags
    UNSET (the new default); `on=False` -> flags explicit '0' (the shipped baseline)."""
    os.environ["BRAIN_COMPOSER_KIND"] = composer_kind
    os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "off"   # irrelevant + expensive here -- see de-risk _build's docstring
    if on:
        _unset_flags()
    else:
        _set_flags_explicit_off()
    from webapp.server import _build_chat_brain
    from webapp import gnw_bus_shadow as gbs
    with _real_seed(seed):
        chat, _src = _build_chat_brain("tiny-demo", "stub")
    if with_bus:
        gbs.install_bus_gate(chat)
    return chat


def _call_counts(chat):
    counts = {"_gate_router_combine": 0, "match_fact": 0}
    orig_rc = chat._gate_router_combine
    orig_mf = chat.router.match_fact

    def _rc(q, *a, **k):
        counts["_gate_router_combine"] += 1
        return orig_rc(q, *a, **k)

    def _mf(q, *a, **k):
        counts["match_fact"] += 1
        return orig_mf(q, *a, **k)

    chat._gate_router_combine = _rc
    chat.router.match_fact = _mf

    def _restore():
        chat._gate_router_combine = orig_rc
        chat.router.match_fact = orig_mf

    return counts, _restore


def _batch(chat, questions):
    counts, restore = _call_counts(chat)
    out = {}
    try:
        for q in questions:
            counts["_gate_router_combine"] = 0
            counts["match_fact"] = 0
            svo = chat.gate(q)
            out[q] = (svo, dict(counts))
    finally:
        restore()
    return out


def _stateful(chat, seq):
    counts, restore = _call_counts(chat)
    try:
        answers = [chat.gate(u) for u, _w in seq]
        return answers, dict(counts)
    finally:
        restore()


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# FAST (rf composer) battery -- both combiners, genuinely-seeded, ON-via-unset vs OFF-via-explicit-0.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

# BUILD-COST MERGE (one build per flag state, not one per class): STORED's own first question ("what does dog
# chase?") already establishes the discourse referent 'cat' (`_note_referent`, since 'cat' is itself a known
# agent); every OTHER question below recalls a patient that is NOT a known agent (spikes/words/memory/fish/None),
# so by `_note_referent`'s own guard (`p in self.agents_set`) none of them can overwrite that referent -- verified
# by reading the source, not assumed (see `_note_referent`'s two call sites, `gate()`'s inline combine and
# `_gate_router_combine`, both gated identically). This lets the WHOLE panel -- self_factual, self_identity,
# STORED, UNSTORED, AND the anaphora HIT+MISS tail -- run as ONE ordered conversation per (seed, flag-state,
# combiner), cutting `_fast_eval` from 6 brain builds to 2 with no loss of coverage (`_batch` resets call-counts
# before EACH question while keeping the SAME chat object -- i.e. the SAME conversational turn order -- so this
# is not a new code path, just a different question ORDER through the existing per-question instrumentation).
_ORDERED_PANEL = ([q for q, _w in STORED] + list(UNSTORED) + [q for q, _w in SELF_FACTUAL]
                  + [q for q, _w in SELF_IDENTITY] + ["what does it eat?", "what does it fly?"])


def _fast_eval(seed, with_bus):
    chat_off = _build(seed, False, with_bus, "rf")
    off_batch = _batch(chat_off, _ORDERED_PANEL)
    chat_on = _build(seed, True, with_bus, "rf")
    on_batch = _batch(chat_on, _ORDERED_PANEL)

    self_factual = []
    for q, want in SELF_FACTUAL:
        off_svo, _ = off_batch[q]
        on_svo, on_c = on_batch[q]
        self_factual.append({"q": q, "want": want, "off_svo": off_svo, "on_svo": on_svo,
                             "on_correct": _svo_eq(on_svo, want),
                             "retired": bool(_svo_eq(on_svo, want) and on_c["_gate_router_combine"] == 0
                                            and on_c["match_fact"] == 0)})
    self_identity = []
    for q, want in SELF_IDENTITY:
        off_svo, _ = off_batch[q]
        on_svo, on_c = on_batch[q]
        self_identity.append({"q": q, "want": want, "off_svo": off_svo, "on_svo": on_svo,
                              "on_correct": _svo_eq(on_svo, want),
                              "retired": bool(_svo_eq(on_svo, want) and on_c["_gate_router_combine"] == 0
                                             and on_c["match_fact"] == 0)})
    regression = []
    for q, want in STORED:
        off_svo, _ = off_batch[q]
        on_svo, _ = on_batch[q]
        regression.append({"cls": "stored", "q": q, "off": off_svo, "on": on_svo,
                           "identical": _svo_eq(off_svo, on_svo), "correct": _svo_eq(off_svo, want)})
    for q in UNSTORED:
        off_svo, _ = off_batch[q]
        on_svo, _ = on_batch[q]
        regression.append({"cls": "unstored", "q": q, "off": off_svo, "on": on_svo,
                           "identical": _svo_eq(off_svo, on_svo), "moat_ok": (off_svo is None and on_svo is None)})

    off_hit_svo, _ = off_batch["what does it eat?"]
    on_hit_svo, _ = on_batch["what does it eat?"]
    regression.append({"cls": "anaphora_hit_seq", "q": "dog chase cat / it eat",
                       "off": [off_batch[STORED[0][0]][0], off_hit_svo], "on": [on_batch[STORED[0][0]][0], on_hit_svo],
                       "identical": _svo_eq(off_hit_svo, on_hit_svo),
                       "correct": _svo_eq(off_hit_svo, ANAPHORA_HIT_SEQ[1][1])})

    off_miss_svo, _ = off_batch["what does it fly?"]
    on_miss_svo, on_miss_c = on_batch["what does it fly?"]
    anaphora_miss = {
        "off_confab": off_miss_svo, "on_answer": on_miss_svo,
        "off_confabulated": off_miss_svo is not None,
        "on_abstains": on_miss_svo is None,
        "first_turn_unaffected": _svo_eq(off_batch[STORED[0][0]][0], on_batch[STORED[0][0]][0]),
        "vary_changes_output": (off_miss_svo is not None) != (on_miss_svo is None) or not _svo_eq(off_miss_svo, on_miss_svo),
        "retired": bool(on_miss_svo is None and on_miss_c["_gate_router_combine"] == 0),
    }
    return {"combiner": ("bus" if with_bus else "plain"), "seed": seed,
            "self_factual": self_factual, "self_identity": self_identity,
            "anaphora_miss": anaphora_miss, "regression": regression}


def run_fast():
    rows = {"plain": [], "bus": []}
    for seed in SEEDS:
        rows["plain"].append(_fast_eval(seed, with_bus=False))
        rows["bus"].append(_fast_eval(seed, with_bus=True))
    return rows


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SLOW (onebrain composer, the TRUE production default) battery -- bus combiner only, genuinely-seeded.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

def _onebrain_seed(seed):
    # same build-cost-merge reasoning as `_ORDERED_PANEL` above: fold the anaphora HIT+MISS tail onto the SAME
    # build (STORED[0]'s "dog chase" turn already establishes the referent; nothing between it and the tail can
    # overwrite it) so this one expensive onebrain build (~180s+) covers every class, not just self_factual.
    panel_qs = ([q for q, _w in STORED] + list(UNSTORED) + [q for q, _w in SELF_FACTUAL]
               + [q for q, _w in SELF_IDENTITY] + ["what does it eat?", "what does it fly?"])
    t0 = time.time()
    chat = _build(seed, True, True, "onebrain")   # flags UNSET (real default), bus installed (real production)
    build_s = time.time() - t0
    batch = _batch(chat, panel_qs)
    anaphora = {
        "hit_svo": batch["what does it eat?"][0], "hit_correct": _svo_eq(batch["what does it eat?"][0], ["cat", "eat", "fish"]),
        "miss_svo": batch["what does it fly?"][0], "miss_abstains": batch["what does it fly?"][0] is None,
        "miss_retired": bool(batch["what does it fly?"][0] is None
                             and batch["what does it fly?"][1]["_gate_router_combine"] == 0),
    }

    self_factual = [{"q": q, "want": want, "svo": batch[q][0], "correct": _svo_eq(batch[q][0], want),
                     "retired": bool(_svo_eq(batch[q][0], want) and batch[q][1]["_gate_router_combine"] == 0
                                    and batch[q][1]["match_fact"] == 0)}
                    for q, want in SELF_FACTUAL]
    self_identity = [{"q": q, "want": want, "svo": batch[q][0], "correct": _svo_eq(batch[q][0], want),
                      "retired": bool(_svo_eq(batch[q][0], want) and batch[q][1]["_gate_router_combine"] == 0
                                     and batch[q][1]["match_fact"] == 0)}
                     for q, want in SELF_IDENTITY]
    stored = [{"q": q, "want": want, "svo": batch[q][0], "correct": _svo_eq(batch[q][0], want)} for q, want in STORED]
    unstored = [{"q": q, "svo": batch[q][0], "moat_ok": batch[q][0] is None} for q in UNSTORED]

    # LOAD-BEARING LESION (class (a), the sole genuinely-neural mechanism here): reuse THIS SAME build.
    lesion_q, lesion_want = "what do you use?", ["brain", "use", "spikes"]
    intact = batch[lesion_q][0]   # already asked above, in the SAME (unlesioned) build -- reuse, don't re-ask
    parser = chat.inner.composer.parser
    orig_role_of = parser.role_of
    parser.role_of = lambda *a, **k: "junk_role"   # the 2026-08-12 CHOOSE-1 lesion recipe, verbatim
    try:
        lesioned = chat.gate(lesion_q)
    finally:
        parser.role_of = orig_role_of
    reflex = chat.inner.composer.query_patient("brain", "use")   # parser-INDEPENDENT recall reflex
    intact_correct = _svo_eq(intact, lesion_want)
    lesion_collapses = (lesioned is None)
    attribution = attributable_to(
        "onebrain-production-flip: self-factual answer owed to the on-brain BridgeParser, not a host fallback",
        1.0 if intact_correct else 0.0, 0.0 if lesion_collapses else 1.0)

    return {"seed": seed, "build_seconds": build_s, "composer": type(chat.inner.composer).__name__,
            "self_factual": self_factual, "self_identity": self_identity, "stored": stored, "unstored": unstored,
            "anaphora": anaphora,
            "lesion": {"q": lesion_q, "want": lesion_want,
                      "intact": (list(intact) if intact is not None else None),
                      "lesioned": (list(lesioned) if lesioned is not None else None),
                      "reflex": reflex, "intact_correct": intact_correct, "lesion_collapses": lesion_collapses,
                      "reflex_survives": (reflex == "spikes"), "attribution": attribution}}


def run_onebrain():
    return [_onebrain_seed(s) for s in SEEDS]


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-onebrain", action="store_true", help="fast (rf) battery only")
    ap.add_argument("--onebrain-only", action="store_true", help="skip the fast battery (rf already covered)")
    args = ap.parse_args()

    # requirement 3, checked directly, zero-cost: the flags resolve ON with the env vars UNSET.
    _unset_flags()
    import research.runners.brain_chat_tui as bct
    default_selfid_on = bct._neural_selfid_enabled()
    default_anaphora_on = bct._neural_anaphora_abstain_enabled()

    _mode_suffix_early = "_onebrain_only" if args.onebrain_only else ("_fast_only" if args.skip_onebrain else "")
    _ckpt = f"research/findings/raw/_rank13_selfid_anaphora_prodflip/_checkpoint{_mode_suffix_early}.json"
    os.makedirs(os.path.dirname(_ckpt), exist_ok=True)

    def _save_ckpt(**kw):
        try:
            with open(_ckpt, "w") as f:
                json.dump(kw, f, indent=2, default=str)
        except Exception as e:
            print(f"[rank13-prodflip] checkpoint save failed (non-fatal): {e!r}", flush=True)

    fast_rows = None
    if not args.onebrain_only:
        print("[rank13-prodflip] running FAST (rf composer) battery, genuinely-seeded, both combiners...", flush=True)
        fast_rows = run_fast()
        _save_ckpt(fast_rows=fast_rows)   # DEFENSIVE (2026-09-05): a later onebrain crash must not lose this

    onebrain_rows = None
    if not args.skip_onebrain:
        print("[rank13-prodflip] running SLOW (onebrain composer, TRUE production default) battery, "
              "genuinely-seeded, bus combiner...", flush=True)
        onebrain_rows = run_onebrain()
        _save_ckpt(fast_rows=fast_rows, onebrain_rows=onebrain_rows)

    def _all_true(vals):
        return bool(vals) and all(bool(v) for v in vals)

    fast_summary = {}
    if fast_rows is not None:
        for combiner in ("plain", "bus"):
            rws = fast_rows[combiner]
            fast_summary[combiner] = {
                "n_seeds": len(rws),
                "self_factual_correct": _all_true([x["on_correct"] for r in rws for x in r["self_factual"]]),
                "self_factual_retired": _all_true([x["retired"] for r in rws for x in r["self_factual"]]),
                "self_identity_correct": _all_true([x["on_correct"] for r in rws for x in r["self_identity"]]),
                "self_identity_retired": _all_true([x["retired"] for r in rws for x in r["self_identity"]]),
                "anaphora_off_confabulated": _all_true([r["anaphora_miss"]["off_confabulated"] for r in rws]),
                "anaphora_on_abstains": _all_true([r["anaphora_miss"]["on_abstains"] for r in rws]),
                "anaphora_first_turn_unaffected": _all_true([r["anaphora_miss"]["first_turn_unaffected"] for r in rws]),
                "anaphora_vary_changes_output": _all_true([r["anaphora_miss"]["vary_changes_output"] for r in rws]),
                "anaphora_retired": _all_true([r["anaphora_miss"]["retired"] for r in rws]),
                "no_regression": _all_true([x["identical"] for r in rws for x in r["regression"]]),
                "n_regression_checks": sum(len(r["regression"]) for r in rws),
            }

    onebrain_summary = None
    if onebrain_rows is not None:
        onebrain_summary = {
            "n_seeds": len(onebrain_rows),
            "self_factual_correct": _all_true([x["correct"] for r in onebrain_rows for x in r["self_factual"]]),
            "self_factual_retired": _all_true([x["retired"] for r in onebrain_rows for x in r["self_factual"]]),
            "self_identity_correct": _all_true([x["correct"] for r in onebrain_rows for x in r["self_identity"]]),
            "self_identity_retired": _all_true([x["retired"] for r in onebrain_rows for x in r["self_identity"]]),
            "stored_correct": _all_true([x["correct"] for r in onebrain_rows for x in r["stored"]]),
            "unstored_moat_ok": _all_true([x["moat_ok"] for r in onebrain_rows for x in r["unstored"]]),
            "anaphora_hit_correct": _all_true([r["anaphora"]["hit_correct"] for r in onebrain_rows]),
            "anaphora_miss_abstains": _all_true([r["anaphora"]["miss_abstains"] for r in onebrain_rows]),
            "anaphora_miss_retired": _all_true([r["anaphora"]["miss_retired"] for r in onebrain_rows]),
            "lesion_intact_correct": _all_true([r["lesion"]["intact_correct"] for r in onebrain_rows]),
            "lesion_collapses": _all_true([r["lesion"]["lesion_collapses"] for r in onebrain_rows]),
            "lesion_reflex_survives": _all_true([r["lesion"]["reflex_survives"] for r in onebrain_rows]),
            "composer_kinds_seen": sorted({r["composer"] for r in onebrain_rows}),
            "build_seconds": [r["build_seconds"] for r in onebrain_rows],
        }

    fast_void = void_if(fast_rows is not None and (not fast_rows["plain"] or not fast_rows["bus"]),
                        "the fast panel produced ZERO rows -- UNDEFINED, not a GO")
    onebrain_void = void_if(onebrain_rows is not None and not onebrain_rows,
                            "the onebrain panel produced ZERO rows -- UNDEFINED, not a GO")

    fast_go = True if fast_rows is None else bool(
        not fast_void
        and fast_summary["plain"]["self_factual_correct"] and fast_summary["plain"]["self_factual_retired"]
        and fast_summary["bus"]["self_factual_correct"] and fast_summary["bus"]["self_factual_retired"]
        and fast_summary["plain"]["self_identity_correct"] and fast_summary["plain"]["self_identity_retired"]
        and fast_summary["bus"]["self_identity_correct"] and fast_summary["bus"]["self_identity_retired"]
        and fast_summary["plain"]["anaphora_off_confabulated"] and fast_summary["plain"]["anaphora_on_abstains"]
        and fast_summary["bus"]["anaphora_off_confabulated"] and fast_summary["bus"]["anaphora_on_abstains"]
        and fast_summary["plain"]["anaphora_vary_changes_output"] and fast_summary["bus"]["anaphora_vary_changes_output"]
        and fast_summary["plain"]["anaphora_first_turn_unaffected"] and fast_summary["bus"]["anaphora_first_turn_unaffected"]
        and fast_summary["plain"]["no_regression"] and fast_summary["bus"]["no_regression"]
    )
    onebrain_go = True if onebrain_rows is None else bool(
        not onebrain_void
        and onebrain_summary["self_factual_correct"] and onebrain_summary["self_factual_retired"]
        and onebrain_summary["self_identity_correct"] and onebrain_summary["self_identity_retired"]
        and onebrain_summary["stored_correct"] and onebrain_summary["unstored_moat_ok"]
        and onebrain_summary["anaphora_hit_correct"] and onebrain_summary["anaphora_miss_abstains"]
        and onebrain_summary["lesion_intact_correct"] and onebrain_summary["lesion_collapses"]
        and onebrain_summary["lesion_reflex_survives"]
    )
    go = bool(default_selfid_on and default_anaphora_on and fast_go and onebrain_go)

    v = Verdict("rank-13 PRODUCTION-FLIP verify: self/identity + anaphora-miss default-ON")
    v.require("BRAIN_NEURAL_SELFID resolves ON with the env var UNSET (a real default change)",
              default_selfid_on, expect=True)
    v.require("BRAIN_NEURAL_ANAPHORA_ABSTAIN resolves ON with the env var UNSET (a real default change)",
              default_anaphora_on, expect=True)
    if fast_rows is not None:
        p, b = fast_summary["plain"], fast_summary["bus"]
        v.require("FAST/rf: self-factual correct+retired, plain gate() and installed GNW-bus gate()",
                  p["self_factual_correct"] and p["self_factual_retired"]
                  and b["self_factual_correct"] and b["self_factual_retired"], expect=True)
        v.require("FAST/rf: self-identity correct+retired, plain gate() and installed GNW-bus gate()",
                  p["self_identity_correct"] and p["self_identity_retired"]
                  and b["self_identity_correct"] and b["self_identity_retired"], expect=True)
        v.require("FAST/rf: anaphora-miss confabulates OFF, abstains ON (both combiners) -- the defect is real",
                  p["anaphora_off_confabulated"] and p["anaphora_on_abstains"]
                  and b["anaphora_off_confabulated"] and b["anaphora_on_abstains"], expect=True)
        v.require("HOLLOW-CHECK: toggling the flag changes the anaphora-miss FINAL ANSWER "
                  "(not byte-identical on vs off -- the affect-hollow-mouth failure mode does NOT hold here)",
                  p["anaphora_vary_changes_output"] and b["anaphora_vary_changes_output"], expect=True)
        v.require("FAST/rf: legitimate anaphora-HIT recall unaffected by the flag",
                  p["anaphora_first_turn_unaffected"] and b["anaphora_first_turn_unaffected"], expect=True)
        v.require("FAST/rf: NO REGRESSION on stored/unstored/anaphora-hit, genuinely-seeded, both combiners",
                  p["no_regression"] and b["no_regression"], expect=True,
                  note=f"{p['n_regression_checks']}+{b['n_regression_checks']} checks, 6 REAL seeds")
    if onebrain_rows is not None:
        s = onebrain_summary
        v.require("SLOW/onebrain (TRUE production default): self-factual + self-identity correct+retired",
                  s["self_factual_correct"] and s["self_factual_retired"]
                  and s["self_identity_correct"] and s["self_identity_retired"], expect=True)
        v.require("SLOW/onebrain: STORED correct + UNSTORED moat-honest (no regression on the real default composer)",
                  s["stored_correct"] and s["unstored_moat_ok"], expect=True)
        v.require("SLOW/onebrain: anaphora-HIT correct + anaphora-MISS abstains honestly (real production composer)",
                  s["anaphora_hit_correct"] and s["anaphora_miss_abstains"], expect=True)
        v.require("LOAD-BEARING (crux): lesioning the on-brain BridgeParser.role_of COLLAPSES the self-factual "
                  "answer to abstain, on the TRUE production composer, 6 GENUINELY-varied seeds",
                  s["lesion_collapses"], expect=True, note=f"n_seeds={s['n_seeds']}")
        v.require("intact (non-lesioned) onebrain self-factual answer is correct",
                  s["lesion_intact_correct"], expect=True)
        v.require("the parser-INDEPENDENT recall reflex (query_patient) SURVIVES the lesion (dissociation)",
                  s["lesion_reflex_survives"], expect=True)
        v.control("lesion dissociation (production composer): self-factual answer needs the on-brain parser",
                  treatment=(1.0 if s["lesion_intact_correct"] else 0.0),
                  control=(0.0 if s["lesion_collapses"] else 1.0), min_separation=0.0)
    else:
        v.disabled("onebrain (TRUE production default) battery", why="--skip-onebrain passed this run")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    print("\n" + "=" * 112, flush=True)
    print("  RANK-13 PRODUCTION-FLIP VERIFY -- self/identity + anaphora-miss default-ON", flush=True)
    print("=" * 112, flush=True)
    print(f"  default (env UNSET): BRAIN_NEURAL_SELFID={default_selfid_on} "
          f"BRAIN_NEURAL_ANAPHORA_ABSTAIN={default_anaphora_on}", flush=True)
    if fast_rows is not None:
        for combiner in ("plain", "bus"):
            s = fast_summary[combiner]
            print(f"  [FAST/{combiner:5s}] self_factual correct={s['self_factual_correct']} "
                  f"retired={s['self_factual_retired']} | self_identity correct={s['self_identity_correct']} "
                  f"retired={s['self_identity_retired']} | anaphora vary_changes={s['anaphora_vary_changes_output']} "
                  f"on_abstains={s['anaphora_on_abstains']} | no_regression={s['no_regression']} "
                  f"(n={s['n_regression_checks']})", flush=True)
    if onebrain_summary is not None:
        s = onebrain_summary
        print(f"  [SLOW/onebrain] composer={s['composer_kinds_seen']} self_factual correct={s['self_factual_correct']} "
              f"retired={s['self_factual_retired']} | self_identity correct={s['self_identity_correct']} "
              f"retired={s['self_identity_retired']} | stored={s['stored_correct']} unstored_moat={s['unstored_moat_ok']}"
              f" | anaphora hit={s['anaphora_hit_correct']} miss_abstains={s['anaphora_miss_abstains']}"
              f" | LESION collapses={s['lesion_collapses']} reflex_survives={s['lesion_reflex_survives']}"
              f" | build_seconds={[round(x,1) for x in s['build_seconds']]}", flush=True)
        for r in onebrain_rows:
            print(f"      seed={r['seed']:4d} intact={r['lesion']['intact']} lesioned={r['lesion']['lesioned']} "
                  f"reflex={r['lesion']['reflex']!r}", flush=True)
    verdict = "GO (production flip verified: no-regression + load-bearing, both flags default ON)" if go \
        else "NO-GO / CHARACTERIZED"
    print(f"\n  VERDICT: {verdict}\n" + "=" * 112, flush=True)

    out = {"runner": "_rank13_selfid_anaphora_prodflip_verify", "go": go, "status": decided["status"],
           "seeds": SEEDS, "default_selfid_on": default_selfid_on, "default_anaphora_on": default_anaphora_on,
           "fast_summary": fast_summary, "fast_rows": fast_rows,
           "onebrain_summary": onebrain_summary, "onebrain_rows": onebrain_rows,
           "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
           "undefined_reasons": decided["undefined_reasons"]}
    # DISTINCT output path per mode (2026-09-05 fix): running --skip-onebrain and --onebrain-only as separate
    # PROCESSES (the correct pattern -- see the module docstring's pool1-cross-contamination note) previously
    # both wrote the SAME fixed path, so whichever finished LAST silently overwrote the other's artifact --
    # exactly the "an arm produced no output" class of silent failure verify-go warns about, just on disk
    # instead of in a process. `_mode_suffix` makes the collision structurally impossible.
    _mode_suffix = "_onebrain_only" if args.onebrain_only else ("_fast_only" if args.skip_onebrain else "")
    op = f"research/findings/raw/_rank13_selfid_anaphora_prodflip/result{_mode_suffix}.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [saved] {op}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
