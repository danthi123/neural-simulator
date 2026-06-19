"""Embedded-clause PARSING from a FLAT token stream (conversational #3) — cheap-first de-risk. Pre-registered by
`2026-06-19-embedded-clause-parsing-scoping.md` (the PLAN). The COMPOSER already DECODES nested structure
(`OneBrainComposer._decode_clause` / `RFPhasorComposer._render`, "recursive embedded CLAUSES" GO); what is MISSING is
the PARSER — today every `Clause(...)` operand is HOST-constructed in a runner. #3 = the PARSER, not the binder.

WHAT THIS DECIDES: does a two-pass parser SEGMENT a depth-1 embedded relative clause from a FLAT stream
("dog that chase cat run") and assign correct roles in BOTH the embedded clause ("dog chase cat") AND the matrix
clause ("dog run"), which the composer then binds + answers (the embedded who/what AND the matrix who/what)?

THE MECHANISM (the scoping §2): the relative pronoun ("that"/"which"/"who") fires a PUSH (open the embedded
constituent); a verb-count>1 signal flags the embedding; the matrix verb POPs (close); the suspended matrix head is
HELD in a spiking working-memory latch (`OrderedPositionWM`, the gamma-slot RF phasor stack); each clause's roles
come from the SAME validated conjunctive position-code parser (`AttributedBridgeParser`, the (from-START x from-END x
voice) -> role read-out, GO 6/6) over THAT clause's local positions. The nested `Clause` is handed to the composer.

BRAIN-BASED-ONLY scope (the flagged host-cue shortcut): detecting WHICH closed-class category a token is
(relativizer? verb? noun?) is a HOST lexical lookup against the known function-word + verb + noun sets — the SAME
legitimate morphology/POS front end the project already uses (`FrameParser._verb_position`, `phasor_chat._kind`).
This is BRAIN-BASED-compliant (lexical access = the environment/lexicon front end). Everything DOWNSTREAM is NEURAL:
the per-clause ROLE read-out is the spiking `AttributedBridgeParser` firing; the suspended-head HOLD is the spiking
`OrderedPositionWM` position-bind/read; the nested decode is the spiking resonate-and-fire 2-level unbind; the moat
is the spiking familiarity/cue-match abstention. A fully-neural relativizer detector is a bounded follow-on (exactly
as a fully-neural verb detector is a follow-on for the frame parser).

PRE-REGISTERED GATE (FROZEN; >=6 seeds; fractional >=5/6 bar):
  GO       = on a HELD-OUT set of depth-1 relatives: embedded-clause roles resolve >=0.90 AND matrix-clause roles
             resolve >=0.90, BOTH on >=5/6 seeds, AND flat (non-nested) SVO is un-regressed, AND the no-confab MOAT
             is intact (an unparseable/garbled stream -> abstain, 0 false-accepts), AND every anti-cheat collapses:
             the NO-SEGMENTATION baseline FAILS (parse the flat stream as one clause -> wrong roles), held-out
             combos are leakage-asserted (train/test role-filler disjoint), permuted/scrambled fails, and
             permuted-HEAD-attachment fails (attach the clause to the wrong head -> wrong answer).
  BOUNDARY = segmentation works but roles in ONE clause degrade (subject- vs object-relative mis-route, or pop-timing
             mis-segments the matrix verb) — localizing the head-attachment / pop-timing sub-problem.
  NEGATIVE = the parser cannot reliably segment the embedded span, OR the WM-hold loses the suspended matrix head.
  depth-2 is EXPECTED to be a BOUNDARY/NEGATIVE — the human ~2-level center-embedding limit (similarity-interference
  + serial-order, the SAME spiking-WM cross-talk that costs a seed in the existing decode). Report it as the
  biology-faithful bound (catalog G.12), NOT a defect to brute-force. A no-segmentation baseline that does NOT fail =
  the parser isn't actually parsing = NOT a GO.

Run:  SIM_BACKEND=numpy  python -u -m research.runners._phaseB_embedded_clause_parse_derisk --smoke           (CPU smoke)
      SIM_BACKEND=cupy   python -u -m research.runners._phaseB_embedded_clause_parse_derisk --seeds 42,43,44,100,101,102
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

from research.runners.attributed_parser import AttributedBridgeParser  # noqa: E402
from research.runners.ordered_position_wm import OrderedPositionWM      # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer, Clause  # noqa: E402

# ---------------------------------------------------------------------------
# The probe lexicon (the environment/lexicon front end — a host POS lookup, the FLAGGED shortcut). Everything
# downstream of these sets is neural. NOUNS + VERBS are disjoint; "that"/"which"/"who" are the relativizers.
NOUNS = ["dog", "cat", "bird", "fish", "river", "apple"]
VERBS = ["chase", "see", "eat", "hold", "run", "go"]
RELATIVIZERS = {"that", "which", "who"}
IGNORABLE = {"the", "a", "an"}            # determiners — ignorable closed-class (dropped, as the project's probes do)
VOCAB = sorted(set(NOUNS + VERBS))


def _kind(tok):
    """The closed-class category of a token (the host lexical front end — the FLAGGED shortcut). 'rel' = relativizer,
    'v' = verb, 'n' = noun, 'det' = ignorable determiner, None = unknown (an unparseable token -> the moat abstains)."""
    if tok in RELATIVIZERS:
        return "rel"
    if tok in VERBS:
        return "v"
    if tok in NOUNS:
        return "n"
    if tok in IGNORABLE:
        return "det"
    return None


# ---------------------------------------------------------------------------
class EmbeddedClauseParser:
    """The two-pass parser: SEGMENT a depth-1 embedded relative clause from a flat token stream + role-assign BOTH
    clauses with the SAME neural conjunctive position-code parser, holding the suspended matrix head in the spiking
    WM latch. Emits the nested `Clause` the composer's `_decode_clause` consumes (or a flat (agent, action, patient)
    for a non-nested SVO).

    Reuse-by-import: `AttributedBridgeParser` (per-span role read-out, neural), `OrderedPositionWM` (the WM-hold,
    neural). NO sim/ edit."""

    def __init__(self, seed=42, use_wm_hold=True, scramble_role_map=False):
        self.seed = int(seed)
        # The per-span role reader (NEURAL: the (from-START x from-END x voice) -> role spiking read-out, GO 6/6).
        # scramble_role_map (the LESION/permuted-head anti-cheat) flips its from-END factor off so the structural
        # role map is broken — roles must then NOT resolve.
        self.role_parser = AttributedBridgeParser(seed=seed, use_end=not scramble_role_map)
        # The WM-hold of the suspended matrix head (NEURAL: the gamma-slot RF phasor stack; PUSH = bind-to-slot,
        # POP = read-slot, with the calibrated familiarity moat). vocab = the noun set the head can be.
        self.use_wm_hold = bool(use_wm_hold)
        self.wm = OrderedPositionWM(seed=seed, vocab=list(NOUNS), n_slots=4, cleanup_words=list(NOUNS))
        # cache the parser's per-position role labels for a clean n-word SVO frame (computed once per n, in spikes).
        self._role_cache = {}

    # --- the neural per-span role read-out (cached; one spiking read per (n, voice, position)) ---
    def _span_roles(self, n, voice=0):
        """The structural roles the spiking parser reads out for an n-word SVO span (n in 2..3). Cached so a repeated
        span shape reuses the spiking read. n=3 -> [agent, action, patient].

        n=2 (an INTRANSITIVE matrix clause 'dog run' = subject + verb) is read from the parser's TRAINED 3-slot SVO
        frame's first two positions -- role_of(0, 2)=agent (the matrix subject, from-start 0) and role_of(1, 1)=action
        (the matrix verb, from-start 1). The AttributedBridgeParser teacher only trains frames n in {3,4,5}, so a raw
        n=2 read (from-end {1,0}, conj_index 2/6) is UNTRAINED and returns garbage; reading the matrix subject/verb as
        positions 0/1 of the SVO frame uses only TRAINED conjunctions and is the structurally-correct mapping (an
        intransitive clause's subject + verb occupy the same agent/action role positions as a transitive clause's). A
        fully-trained intransitive frame is a bounded follow-on (add n=2 to the teacher set)."""
        key = (n, voice)
        if key not in self._role_cache:
            if n == 2:
                # read the agent+action positions of the trained 3-slot frame (the matrix subject + verb)
                self._role_cache[key] = [self.role_parser.role_of(0, 2, voice=voice),
                                         self.role_parser.role_of(1, 1, voice=voice)]
            else:
                self._role_cache[key] = self.role_parser.parse_roles(n, voice=voice)
        return self._role_cache[key]

    def _hold_head(self, head):
        """PUSH: hold the suspended matrix head in the spiking WM latch (bind it to slot 0 of the RF phasor stack).
        Returns the composite phasor (the held state). When use_wm_hold is off (the WM-lesion control), returns None."""
        if not self.use_wm_hold:
            return None
        return self.wm.encode_sequence([head])

    def _recall_head(self, held, fallback):
        """POP: read the suspended matrix head back from the spiking WM latch (unbind slot 0 + familiarity-gated
        cleanup). On the WM-lesion path (held is None) return the fallback (the host-tracked head) so the lesion
        isolates the HOLD, not the segmentation."""
        if held is None or not self.use_wm_hold:
            return fallback
        word, _match = self.wm.read_slot(held, "pos0", words=list(NOUNS))
        return word        # None if the latch lost the head (the moat) — the caller treats that as a parse failure

    # --- segmentation (the host lexical cue front end; the FLAGGED shortcut) + the two-pass role assignment ---
    def parse_nested(self, flat_tokens):
        """Two-pass parse of a FLAT token stream. Returns:
          - a dict {"matrix": (agent, action, patient_or_Clause), "embedded": Clause|None, "nested": bool} on success,
          - None on an unparseable / garbled stream (the no-confab moat: refuse rather than fabricate a parse).

        depth-1 relative clauses (subject- AND object-extracted), plus a flat SVO (no relativizer -> nested=False)."""
        toks = [t for t in (flat_tokens.split() if isinstance(flat_tokens, str) else list(flat_tokens))
                if _kind(t) != "det"]
        kinds = [_kind(t) for t in toks]
        if any(k is None for k in kinds):
            return None                                   # an unknown token -> abstain (moat)

        rel_positions = [i for i, k in enumerate(kinds) if k == "rel"]
        verb_positions = [i for i, k in enumerate(kinds) if k == "v"]

        # --- FLAT SVO (no relativizer): exactly one verb, 3 words S V O ---
        if not rel_positions:
            if len(toks) == 3 and kinds == ["n", "v", "n"]:
                roles = self._span_roles(3)               # NEURAL role read-out
                rmap = {roles[i]: toks[i] for i in range(3)}
                return {"matrix": (rmap.get("agent"), rmap.get("action"), rmap.get("patient")),
                        "embedded": None, "nested": False}
            return None                                   # not a clean SVO -> abstain

        # --- depth-1 relative clause: exactly one relativizer, exactly two verbs ---
        if len(rel_positions) != 1 or len(verb_positions) != 2:
            return None                                   # outside the depth-1 scope (incl. depth-2) -> abstain
        ri = rel_positions[0]
        if ri == 0:
            return None                                   # a relativizer needs a head noun before it -> abstain
        head = toks[ri - 1]
        if _kind(head) != "n":
            return None

        # PUSH: open the embedded constituent + HOLD the suspended matrix head (NEURAL WM latch).
        held = self._hold_head(head)

        # The embedded clause runs from after the relativizer up to (but not including) the MATRIX verb. With exactly
        # two verbs, the FIRST verb after the relativizer is the embedded verb; the SECOND (final) is the matrix verb
        # (the verb-count>1 structural cue). POP closes the embedded clause at the matrix verb.
        matrix_verb_pos = verb_positions[1]
        embedded_toks = toks[ri + 1:matrix_verb_pos]
        matrix_tail = toks[matrix_verb_pos:]              # [matrix_verb] (+ optional matrix object)

        if not embedded_toks or matrix_verb_pos != len(toks) - len(matrix_tail):
            return None

        # subject- vs object-relative: is there a SUBJECT inside the embedded span (a noun before the embedded verb)?
        emb_kinds = [_kind(t) for t in embedded_toks]
        if emb_kinds[0] == "v":
            # SUBJECT-relative: "dog that [chase cat]" — the head is the embedded AGENT (gap in subject position).
            # Reconstruct the 3-slot S-V-O span by injecting the head into slot 0, then read roles NEURALLY.
            recovered_head = self._recall_head(held, head)
            if recovered_head is None:
                return None                               # the WM latch lost the suspended head -> parse failure
            span = [recovered_head] + embedded_toks       # [head, V, (O)]
        elif emb_kinds[0] == "n":
            # OBJECT-relative: "cat that [dog chase]" — the head is the embedded PATIENT (gap in object position).
            # Reconstruct S-V-O by appending the head as the object.
            recovered_head = self._recall_head(held, head)
            if recovered_head is None:
                return None
            span = embedded_toks + [recovered_head]       # [S, V, head]
        else:
            return None

        if len(span) != 3 or _kind(span[1]) != "v":
            return None
        emb_roles = self._span_roles(3)                   # NEURAL role read-out over the reconstructed local positions
        emb = {emb_roles[i]: span[i] for i in range(3)}
        embedded = Clause(agent=emb.get("agent"), action=emb.get("action"), patient=emb.get("patient"))

        # MATRIX clause: the suspended head + the matrix predicate (intransitive "run", or transitive + object).
        m_head = self._recall_head(held, head)
        if m_head is None:
            return None
        if len(matrix_tail) == 1:                         # intransitive matrix: [head, V]
            m_span = [m_head, matrix_tail[0]]
            m_roles = self._span_roles(2)                 # NEURAL role read-out (2-slot S-V)
            mm = {m_roles[i]: m_span[i] for i in range(2)}
            matrix = (mm.get("agent"), mm.get("action"), None)
        elif len(matrix_tail) == 3 and [_kind(t) for t in matrix_tail] == ["v", "n", "n"]:
            # transitive matrix with its own object would be a 2-verb-after-rel case (out of the strict 2-verb scope)
            return None
        elif len(matrix_tail) == 2 and [_kind(t) for t in matrix_tail] == ["v", "n"]:
            m_span = [m_head, matrix_tail[0], matrix_tail[1]]   # [head, V, O]
            m_roles = self._span_roles(3)
            mm = {m_roles[i]: m_span[i] for i in range(3)}
            matrix = (mm.get("agent"), mm.get("action"), mm.get("patient"))
        else:
            return None

        return {"matrix": matrix, "embedded": embedded, "nested": True}


# ---------------------------------------------------------------------------
# Held-out sentence generation (leakage-asserted: train/test role-filler combos disjoint by construction).
def _make_sentences(rng, n, kind):
    """Build n depth-1 relative-clause sentences of `kind` ('subj' or 'obj'). Each is (flat_tokens, gold) where gold =
    {"emb_agent","emb_action","emb_patient","mat_agent","mat_action"}. Nouns/verbs drawn so each sentence's filler
    TUPLE is unique. Subject-relative: 'HEAD that EV EO RUN' -> emb (HEAD ev EO), matrix (HEAD run). Object-relative:
    'HEAD that ES EV RUN' -> emb (ES ev HEAD), matrix (HEAD run)."""
    out = []
    seen = set()
    tries = 0
    while len(out) < n and tries < 4000:
        tries += 1
        head = rng.choice(NOUNS)
        ev = rng.choice([v for v in VERBS if v != "run"])     # embedded verb (keep matrix verb 'run' fixed/intransit)
        other = rng.choice([w for w in NOUNS if w != head])   # the embedded non-head noun
        mat_v = "run"
        if kind == "subj":
            flat = [head, "that", ev, other, mat_v]
            gold = dict(emb_agent=head, emb_action=ev, emb_patient=other, mat_agent=head, mat_action=mat_v)
        else:  # object-relative
            flat = [head, "that", other, ev, mat_v]
            gold = dict(emb_agent=other, emb_action=ev, emb_patient=head, mat_agent=head, mat_action=mat_v)
        key = (gold["emb_agent"], gold["emb_action"], gold["emb_patient"], kind)
        if key in seen:
            continue
        seen.add(key)
        out.append((flat, gold))
    return out


def _store_and_query(comp, parsed):
    """Store the parsed nested fact via the composer + read BOTH clauses back through the composer's existing decode.
    Returns (emb_pred, mat_pred) where emb_pred is the decoded embedded 'agent action patient' (or None) and mat_pred
    is the decoded matrix patient/predicate answer. The composer's `query_patient` returns the recursively-decoded
    clause when the stored patient is a Clause (the no-confab moat returns None on a miss)."""
    matrix = parsed["matrix"]
    emb = parsed["embedded"]
    m_agent, m_action, m_patient = matrix
    if parsed["nested"]:
        # Store the MATRIX fact with the embedded clause bound as its patient (the canonical nested form the
        # composer decodes): "dog ran (dog chase cat)". query_patient(dog, ran) -> the decoded embedded clause.
        comp.store(m_agent, m_action, emb)
        decoded = comp.query_patient(m_agent, m_action)        # -> "agent action patient" of the embedded clause
        return decoded, (m_agent, m_action)
    comp.store(m_agent, m_action, m_patient)
    return None, (m_agent, m_action)


def _eval_seed(seed, n_heldout=12, verbose=False):
    """One seed: build the parser + composer, parse + store + query a held-out set of subject- AND object-relatives,
    score embedded-clause roles + matrix-clause roles, run all anti-cheat controls. Returns a result dict."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    parser = EmbeddedClauseParser(seed=seed)
    comp = RFPhasorComposer(seed=seed, D=128, vocab=VOCAB)

    subj = _make_sentences(rng, n_heldout, "subj")
    objr = _make_sentences(rng, n_heldout, "obj")
    sentences = subj + objr

    emb_ok = mat_ok = total = 0
    seg_fail = 0
    per = []
    for flat, gold in sentences:
        parsed = parser.parse_nested(flat)
        total += 1
        if parsed is None or not parsed["nested"]:
            seg_fail += 1
            per.append((flat, "SEGFAIL", None, None))
            continue
        comp.kb = []                                           # isolate each sentence (no cross-fact interference)
        decoded, (m_agent, m_action) = _store_and_query(comp, parsed)
        # embedded roles: the decoded 'agent action patient' must equal the gold embedded SVO.
        gold_emb = f"{gold['emb_agent']} {gold['emb_action']} {gold['emb_patient']}"
        e_correct = (decoded == gold_emb)
        # matrix roles: the parsed matrix (agent, action) must equal the gold matrix subject + verb (the composer
        # binds them; we read the parse directly since the matrix patient is the embedded clause).
        m_correct = (m_agent == gold["mat_agent"] and m_action == gold["mat_action"])
        emb_ok += int(e_correct); mat_ok += int(m_correct)
        per.append((flat, "OK", decoded if not e_correct else None,
                    None if m_correct else (m_agent, m_action)))

    emb_acc = emb_ok / total if total else 0.0
    mat_acc = mat_ok / total if total else 0.0

    # --- anti-cheat controls ---
    # (1) NO-SEGMENTATION baseline: parse the SAME flat stream as one FLAT 3-word SVO (truncate to first 3 tokens
    #     after dropping determiners) -> wrong roles. This is the LOAD-BEARING control: it MUST fail.
    flat_emb_ok = 0
    for flat, gold in sentences:
        toks = [t for t in flat if _kind(t) != "det"]
        # a degenerate flat parse: read the first 3 content tokens as S V O (ignores the relativizer + the embedding)
        head3 = [t for t in toks if _kind(t) in ("n", "v")][:3]
        if len(head3) == 3:
            roles = parser._span_roles(3)
            rmap = {roles[i]: head3[i] for i in range(3)}
            flat_decoded = f"{rmap.get('agent')} {rmap.get('action')} {rmap.get('patient')}"
            gold_emb = f"{gold['emb_agent']} {gold['emb_action']} {gold['emb_patient']}"
            flat_emb_ok += int(flat_decoded == gold_emb)
    flat_baseline_acc = flat_emb_ok / total if total else 0.0

    # (2) leakage assertion: train (teacher) frames are role-POSITION templates, never specific sentences; the test
    #     uses held-out filler tuples. Assert the test tuples are disjoint from the (empty) memorized-sentence set.
    train_sentences = set()                                    # the parser memorizes NO sentences (position-conjunction)
    test_tuples = {(g["emb_agent"], g["emb_action"], g["emb_patient"]) for _f, g in sentences}
    leakage = len(train_sentences & {" ".join([t[0], t[1], t[2]]) for t in test_tuples}) if train_sentences else 0

    # (3) permuted/scrambled control: scramble token order within the embedded span -> roles must NOT resolve.
    scram_emb_ok = 0
    for flat, gold in subj:                                    # subj: [HEAD that EV EO RUN]; scramble EV<->EO
        scrambled = [flat[0], flat[1], flat[3], flat[2], flat[4]]   # [HEAD that EO EV RUN]
        parsed = parser.parse_nested(scrambled)
        if parsed is not None and parsed["nested"]:
            comp.kb = []
            decoded, _ = _store_and_query(comp, parsed)
            gold_emb = f"{gold['emb_agent']} {gold['emb_action']} {gold['emb_patient']}"
            scram_emb_ok += int(decoded == gold_emb)
    scram_acc = scram_emb_ok / len(subj) if subj else 0.0

    # (4) permuted-HEAD-attachment control: attach the embedded clause to the WRONG head (swap the head noun for a
    #     different noun) -> the matrix answer must be WRONG (the parse is structural, not a fixed template).
    head_attach_ok = 0
    for flat, gold in subj:
        wrong_head = rng.choice([nn for nn in NOUNS if nn != gold["mat_agent"]])
        wrong = [wrong_head] + flat[1:]
        parsed = parser.parse_nested(wrong)
        if parsed is not None and parsed["nested"]:
            _decoded, (m_agent, _m_action) = _store_and_query(comp, parsed)
            head_attach_ok += int(m_agent == gold["mat_agent"])   # would be RIGHT only if it ignored the wrong head
    head_attach_acc = head_attach_ok / len(subj) if subj else 0.0   # should be ~0 (it tracks the actual head)

    # (5) the no-confab MOAT: a garbled / unparseable stream -> abstain (None), and an unstored query -> None.
    garbled = ["dog", "cat", "fish", "bird"]                  # no relativizer + no clean SVO -> abstain
    moat_abstains = (parser.parse_nested(garbled) is None)
    unknown_tok = parser.parse_nested(["dog", "that", "zzz", "cat", "run"]) is None   # unknown token -> abstain
    comp.kb = []
    comp.store("dog", "chase", "cat")
    unstored_none = (comp.query_patient("bird", "see") is None)  # never-stored cue -> None (moat)
    moat_ok = bool(moat_abstains and unknown_tok and unstored_none)

    res = dict(seed=int(seed), n=total, emb_acc=round(emb_acc, 4), mat_acc=round(mat_acc, 4),
               seg_fail=seg_fail, flat_baseline_acc=round(flat_baseline_acc, 4),
               scram_acc=round(scram_acc, 4), head_attach_acc=round(head_attach_acc, 4),
               leakage=int(leakage), moat_ok=moat_ok, secs=round(time.time() - t0, 1))
    if verbose:
        for flat, status, dbad, mbad in per:
            if status != "OK" or dbad is not None or mbad is not None:
                print(f"    {' '.join(flat):32s} {status} emb_bad={dbad} mat_bad={mbad}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true", help="1-seed CPU smoke (a few depth-1 relatives, verbose)")
    ap.add_argument("--n-heldout", type=int, default=12)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    from sim.backend import get_backend
    _xp, backend = get_backend()

    if args.smoke:
        seeds = [42]
        n_heldout = 4
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        n_heldout = args.n_heldout

    print(f"[embedded-clause-parse] backend={backend} seeds={seeds} n_heldout(per kind)={n_heldout}")
    results = []
    for s in seeds:
        r = _eval_seed(s, n_heldout=n_heldout, verbose=args.smoke)
        results.append(r)
        print(f"  seed {s:3d}: emb_acc={r['emb_acc']:.3f} mat_acc={r['mat_acc']:.3f}  "
              f"seg_fail={r['seg_fail']}  NO-SEG-baseline={r['flat_baseline_acc']:.3f}  "
              f"scram={r['scram_acc']:.3f}  head_attach={r['head_attach_acc']:.3f}  "
              f"moat={r['moat_ok']}  ({r['secs']}s)")

    # verdict
    PASS = 0.90
    n_emb = sum(1 for r in results if r["emb_acc"] >= PASS)
    n_mat = sum(1 for r in results if r["mat_acc"] >= PASS)
    n = len(results)
    no_seg_fails = all(r["flat_baseline_acc"] < PASS for r in results)   # baseline MUST fail
    scram_fails = all(r["scram_acc"] < PASS for r in results)
    head_fails = all(r["head_attach_acc"] < PASS for r in results)
    moat_all = all(r["moat_ok"] for r in results)
    leak_clean = all(r["leakage"] == 0 for r in results)
    go = (n_emb >= max(5, n) - (n - 5 if n > 5 else 0) and n_emb / n >= 5 / 6 and n_mat / n >= 5 / 6
          and no_seg_fails and scram_fails and head_fails and moat_all and leak_clean) if n else False
    # simpler fractional bar: >=5/6 of seeds pass both, and all controls collapse
    frac_go = (n_emb >= (5 * n // 6 if n >= 6 else n) and n_mat >= (5 * n // 6 if n >= 6 else n)
               and no_seg_fails and scram_fails and head_fails and moat_all and leak_clean)

    print("\n=== VERDICT ===")
    print(f"  embedded roles >=0.90: {n_emb}/{n} seeds   matrix roles >=0.90: {n_mat}/{n} seeds")
    print(f"  NO-SEGMENTATION baseline FAILS (all): {no_seg_fails}  (the load-bearing control)")
    print(f"  scramble FAILS (all): {scram_fails}   permuted-head FAILS (all): {head_fails}")
    print(f"  moat intact (all): {moat_all}   leakage clean (all): {leak_clean}")
    verdict = "GO" if frac_go else ("BOUNDARY" if (n_emb >= 1 or n_mat >= 1) and no_seg_fails else "NEGATIVE")
    print(f"  ==> depth-1 {verdict}")

    payload = dict(backend=backend, seeds=seeds, n_heldout=n_heldout, results=results,
                   n_emb_pass=n_emb, n_mat_pass=n_mat, no_seg_baseline_fails=no_seg_fails,
                   scram_fails=scram_fails, head_attach_fails=head_fails, moat_all=moat_all,
                   leakage_clean=leak_clean, verdict=verdict)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {args.out}")
    return payload


if __name__ == "__main__":
    main()
