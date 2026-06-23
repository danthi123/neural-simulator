"""Cheap-first DE-RISK P3 for the grounded-language faculty arc (scoping
`research/findings/2026-06-22-grounded-language-faculty-scoping.md` §3d + §4 Rank-4).

THE ARCHITECTURE (§3d): the fluent faculty (P1, a converted SLM later) is confined to producing
fluent SURFACE FORM, while the BRAIN's structured store supplies + verifies the CONTENT -- so the
no-confab moat the whole conversational arc built is PRESERVED even with a hallucination-prone
generator in the loop. Three enforcement layers, cheapest-first, run PER QUERY:

  (i)  GATE      -- the composer's exact-match recall returns the stored fact OR abstains
                    (query_patient/query_agent -> None ; ask_yes_no -> "unknown"). If it abstains,
                    the faculty is given NOTHING to render -> the moat. (rf_phasor_composer.py:579/568/618)
  (ii) CONSTRAIN -- a TEMPLATE-STUB faculty (standing in for the real P1 LLM -- NO model download,
                    keeps this cheap) renders the retrieved SVO fact into a fluent sentence using ONLY
                    the fact's words/roles. Its only freedom is grammar/phrasing (determiners, verb
                    inflection, template choice); it CANNOT introduce, drop, or swap a content word.
  (iii) VERIFY   -- the faculty's asserted SVO is re-parsed by the SAME parser the agent comprehends
                    with (BrainConversationalAgent.parse, brain_conversational_agent.py:424) and the
                    re-parsed {agent, action, patient} is asserted to MATCH the stored fact -- else the
                    output is REJECTED (a drifted/confabulated assertion never reaches the user).

  Layers (i)+(iii) are the moat-preservers; (ii) is the hallucination-reducer.

THE TEMPLATE-STUB FACULTY (`TemplateStubFaculty`): given a stored fact's content tokens
(agent, action, patient), it emits BOTH
  - a fluent SURFACE FORM ("The dog eats meat.") -- the human-readable deliverable, with determiners +
    a crude present-tense inflection + a choice of template variants (its grammar/phrasing freedom), and
  - the canonical 3-token SVO `[agent, action, patient]` it is ASSERTING -- the content it commits to,
    which VERIFY re-parses. (The real LLM's "asserted SVO" is recovered by re-parsing its prose; here
    the stub exposes it directly so the smoke is cheap and deterministic. The re-parse is run on the
    canonical content tokens, NOT the inflected surface form, because the position-only BridgeParser
    keys on word POSITION -- the determiners/inflection are surface fluency the parser does not model.)

THE ADVERSARIAL INJECTION (test (c)): an `InjectingStubFaculty` deliberately CONFABULATES -- it asserts
an SVO whose patient (or agent) does NOT match the stored fact ("The dog eats fish" when the fact is
(dog, eat, meat)). VERIFY MUST catch this (the re-parsed SVO mismatches the stored fact -> REJECT). A
faculty that confabulates and is NOT caught = a VERIFY leak (the moat breached at the generate stage).

Metrics per seed:
  (a) GROUNDED   -- grounded queries -> a fluent sentence whose re-parsed SVO matches the taught fact
                    (the CONSTRAIN layer): grounded_correct/total (~1.0 bar).
  (b) UNTAUGHT   -- untaught-cue queries -> the GATE abstains -> the faculty produces NO sentence
                    ("I don't know" / None): untaught_abstain/total (the MOAT; 0 leaks is the bar).
  (c) CONFAB     -- injected-confabulation -> the VERIFY re-parse REJECTS it: confab_caught/total
                    (the bar is ALL caught; an uncaught confab = a leak).

GO = grounded->fluent-correct AND untaught->abstain (moat held) AND injected-confab->caught-by-verify,
>=3 seeds => the grounding mechanism preserves the moat while a faculty supplies fluency (the
architecture is sound; the real P1 faculty later swaps in for the stub). Or HONEST: which layer leaks
+ why.

Reuses the P2 curriculum + the P2 vocab-collection + the P2 teach path verbatim (the brain half is
already GO). NO `sim/` edit, NO LLM, CPU-feasible (`SIM_BACKEND=numpy`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

from research.runners.core_sim_composition import Clause
from research.runners.brain_conversational_agent import BrainConversationalAgent
# reuse the P2 curriculum-loading + vocab-collection + teach path verbatim (the brain half is GO)
from research.runners._grounded_lang_p2_derisk import _collect_vocab, _teach, CURRICULUM

OUT = os.path.join(os.path.dirname(__file__), "..", "findings", "raw", "_grounded_lang_p3_derisk.json")

# crude present-tense surface inflection for the FLUENT surface form (display-only; NOT used by VERIFY).
# The faculty's grammar freedom -- a real SLM would do this far better. Irregulars the curriculum uses:
_IRREGULAR = {"eat": "eats", "give": "gives", "make": "makes", "live": "lives in", "is": "is"}


def _inflect(action):
    """3rd-person-singular present of a curriculum verb, for the fluent surface form ONLY."""
    if action in _IRREGULAR:
        return _IRREGULAR[action]
    if action.endswith(("s", "sh", "ch", "x", "z")):
        return action + "es"
    return action + "s"


def _determiner(word, role):
    """A crude article for the fluent surface form ONLY (the patient of an attribute fact / a mass noun
    gets no article; everything else gets 'The'/'a'). Display fluency, NOT content."""
    return "The "


class TemplateStubFaculty:
    """The P1 FLUENT FACULTY stand-in: renders a stored SVO fact into fluent surface form, CONSTRAINED to
    the fact's own content words. It returns BOTH the fluent string (the deliverable) and the canonical
    content SVO it is asserting (what VERIFY re-parses). Its ONLY freedom is grammar/phrasing -- template
    choice + determiners + verb inflection; it can NEVER introduce/drop/swap a content word. This is the
    grounded-generation 'constrain the generator to the retrieved structure' layer (the scoping §3c
    constrained/template-decoding SOTA), here exact-template because the brain supplies the content."""

    def __init__(self, n_templates=2):
        self.n_templates = int(n_templates)

    def render_svo(self, agent, action, patient, template=0):
        """Return (surface_form, asserted_svo). `asserted_svo` is the canonical 3-token content list the
        faculty commits to; `surface_form` is the human-readable fluent rendering. The surface form's
        words are a strict superset that adds ONLY function words (determiners) + inflection -- never a
        new content word."""
        asserted = [agent, action, patient]                       # the CONTENT the faculty asserts (VERIFY checks this)
        det_a = _determiner(agent, "agent")
        verb = _inflect(action)
        if template % 2 == 0:
            surface = f"{det_a}{agent} {verb} {patient}."          # "The dog eats meat."
        else:
            surface = f"{det_a}{agent} {verb} the {patient}."      # "The dog eats the meat." (a phrasing variant)
        return surface, asserted

    def render_yesno(self, agent, action, patient, truth):
        """Fluent yes/no answer for an ask_yes_no fact. The asserted content is the SVO + the polarity
        (truth)."""
        if truth == "yes":
            surface = f"Yes, the {agent} {_inflect(action)} {patient}."
        else:
            surface = f"No, the {agent} does not {action} {patient}."
        return surface, [agent, action, patient]


class InjectingStubFaculty(TemplateStubFaculty):
    """An ADVERSARIAL faculty that CONFABULATES: it renders a DIFFERENT SVO than the one the gate
    retrieved -- it swaps the patient (or agent) for a plausible-but-WRONG content word ("The dog eats
    fish" when the stored fact is (dog, eat, meat)). The fluent surface form looks perfectly grammatical;
    only the re-parse VERIFY (content check vs the stored fact) can catch it. This is the hallucination a
    real LLM would emit -- the case the grounding mechanism must reject."""

    def __init__(self, swap_map, swap_role="patient", n_templates=2):
        super().__init__(n_templates=n_templates)
        self.swap_map = dict(swap_map)        # {true_word: confabulated_word}
        self.swap_role = swap_role

    def render_svo(self, agent, action, patient, template=0):
        # confabulate: replace the gated content word with a wrong one BEFORE asserting + rendering
        ca, cp = agent, patient
        if self.swap_role == "patient":
            cp = self.swap_map.get(patient, patient + "_X")    # wrong patient (fall back to a guaranteed-wrong token)
        elif self.swap_role == "agent":
            ca = self.swap_map.get(agent, agent + "_X")
        asserted = [ca, action, cp]                            # the faculty asserts the CONFABULATED content
        surface = f"The {ca} {_inflect(action)} {cp}."         # fluent + grammatical, but WRONG
        return surface, asserted


# ----------------------------------------------------------------------------------------------------
# the GATE -> CONSTRAIN -> VERIFY loop (one query)
# ----------------------------------------------------------------------------------------------------
def grounded_reply(agent, faculty, q):
    """Run one query through the three-layer grounding loop and return a structured record.

    Returns a dict with: gate_content (the SVO the gate retrieved, or None=abstain), surface (the fluent
    string the faculty produced, or None when the gate abstained), asserted_svo (what the faculty
    committed), verify_svo (the re-parsed content), verified (did the re-parse match the stored fact?),
    emitted (did a verified fluent reply reach the 'user'?)."""
    qtype = q["type"]; cue = q["cue"]
    truth = None

    # (i) GATE -- exact-match recall over the spiking store; abstains (None / "unknown") when no fact matches
    if qtype == "patient":
        content = agent.what_does(cue[0], cue[1])        # -> composer.query_patient -> patient word OR None
        gate_svo = [cue[0], cue[1], content] if content is not None else None
    elif qtype == "agent":
        content = agent.who_does(cue[0], cue[1])         # -> composer.query_agent -> agent word OR None
        gate_svo = [content, cue[0], cue[1]] if content is not None else None
    elif qtype == "yesno":
        truth = agent.is_it_true(cue[0], cue[1], cue[2]) # -> "yes"/"no"/"unknown"
        gate_svo = [cue[0], cue[1], cue[2]] if truth != "unknown" else None
    else:
        raise ValueError(f"unknown query type {qtype!r}")

    rec = {"cue": cue, "type": qtype, "gate_svo": gate_svo, "gate_truth": truth}

    # gate abstained -> the faculty is given NOTHING -> no sentence (the MOAT). Nothing to constrain/verify.
    if gate_svo is None:
        rec.update({"surface": None, "asserted_svo": None, "verify_svo": None,
                    "verified": None, "emitted": False, "abstained": True})
        return rec

    # (ii) CONSTRAIN -- the faculty renders the gated content into fluent surface form (content-locked)
    if qtype == "yesno":
        surface, asserted = faculty.render_yesno(gate_svo[0], gate_svo[1], gate_svo[2], truth)
    else:
        surface, asserted = faculty.render_svo(gate_svo[0], gate_svo[1], gate_svo[2])

    # (iii) VERIFY -- re-parse the faculty's ASSERTED content with the SAME parser; its SVO must match the
    # gated fact (else the assertion drifted/confabulated -> REJECT, emit nothing). Re-parse the canonical
    # content tokens (the position-only BridgeParser keys on POSITION; the determiners/inflection are
    # surface fluency the parser does not model -- the asserted_svo is exactly the content to re-parse).
    parsed = agent.parse(asserted, voice="active")       # -> {agent, action, patient} (the brain's comprehension)
    verify_svo = [parsed.get("agent"), parsed.get("action"), parsed.get("patient")]
    verified = (verify_svo == gate_svo)                  # the re-parsed content must equal the gated fact

    rec.update({"surface": surface, "asserted_svo": asserted, "verify_svo": verify_svo,
                "verified": verified, "emitted": bool(verified), "abstained": False})
    return rec


def run_seed(cur, seed, vocab):
    agent = BrainConversationalAgent(seed=seed, concepts={w: None for w in vocab}, composer_kind="rf")
    taught = _teach(agent, cur)                          # reuse the P2 teach path (AFFIRM polarity) verbatim

    faculty = TemplateStubFaculty()

    # --- (a) GROUNDED: grounded queries -> fluent correct content (tests CONSTRAIN) ---
    grounded = []
    n_grounded_ok = 0
    for q in cur.get("queries_recall", []):
        rec = grounded_reply(agent, faculty, q)
        # a grounded query SHOULD emit a verified fluent sentence (the gate has the fact; CONSTRAIN keeps it true).
        # Skip the deliberately-untaught negative recall probe (cat eat grass -> expect no_or_unknown) -- it is a
        # gate-abstain case, scored under the moat logic, not the grounded-fluency logic.
        expect = q.get("expect")
        is_negative_probe = (q["type"] == "yesno" and expect in ("no", "no_or_unknown"))
        if is_negative_probe:
            # the gate SHOULD abstain (unknown) OR answer 'no'; either way no FALSE fluent assertion -- count as
            # correct iff the loop did NOT emit a (false) affirmative sentence.
            ok = (rec["gate_svo"] is None) or (not rec["emitted"])
        else:
            ok = rec["emitted"] and rec["verified"]
        n_grounded_ok += int(ok)
        grounded.append({**rec, "expect": expect, "is_negative_probe": is_negative_probe, "ok": ok})

    # --- (b) UNTAUGHT: untaught queries -> ABSTAIN (the GATE gives the faculty nothing -> moat held) ---
    untaught = []
    n_untaught_abstain = 0
    for q in cur.get("queries_moat", []):
        rec = grounded_reply(agent, faculty, q)
        # the moat HOLDS iff the loop emitted NO fluent assertion (gate abstained, OR -- for a yes-no whose
        # only honest answers are 'no'/'unknown' -- it did not emit a false affirmative). A non-abstaining
        # patient/agent that produced a sentence is a moat LEAK.
        note = q.get("note", "")
        if q["type"] == "yesno":
            # untaught yes-no ('apple is blue' when only 'apple is red' was taught): the gate returns 'unknown'
            # (no fact matches the false SVO) -> abstain, OR 'no' (a stored negated fact) -> a NEGATIVE sentence.
            # Neither is a false AFFIRMATIVE assertion, so the moat HOLDS unless the loop emitted a sentence that
            # AFFIRMS the false fact. render_yesno's 'no' path produces "No, the apple does not ..." (a denial,
            # not a false assertion); only an emitted 'yes'-style sentence about an untaught SVO would leak.
            held = (rec.get("gate_truth") != "yes")
        else:
            held = rec["abstained"] or (not rec["emitted"])
        n_untaught_abstain += int(held)
        untaught.append({**rec, "note": note, "held": held})

    # --- (c) INJECTED-CONFABULATION: adversarial faculty asserts a WRONG SVO -> VERIFY must REJECT it ---
    # Build a swap map from the taught facts: for each (agent, action) cue, swap its TRUE patient for a
    # DIFFERENT taught patient (a plausible confabulation), so the surface form is grammatical but the
    # content is false. The re-parse VERIFY must catch the mismatch (re-parsed SVO != stored fact).
    confab_targets = []
    all_patients = sorted({f[2] for f in cur.get("facts", [])})
    for f in cur.get("facts", [])[:8]:                  # first 8 facts as adversarial probes (enough signal, cheap)
        a, ac, p = f
        wrong = next((x for x in all_patients if x != p), p + "_X")   # a different taught patient = the confabulation
        confab_targets.append((a, ac, p, wrong))
    swap_map = {p: wrong for (_, _, p, wrong) in confab_targets}
    inj_faculty = InjectingStubFaculty(swap_map, swap_role="patient")

    confab = []
    n_confab_caught = 0
    for (a, ac, p, wrong) in confab_targets:
        q = {"type": "patient", "cue": [a, ac]}
        rec = grounded_reply(agent, inj_faculty, q)
        # the gate retrieves the TRUE fact (a, ac, p); the injecting faculty asserts (a, ac, wrong). The re-parse
        # VERIFY compares the asserted re-parsed SVO against the GATED fact -> they differ on the patient -> NOT
        # verified -> NOT emitted = the confab was CAUGHT. caught iff the loop refused to emit it.
        caught = (rec["gate_svo"] is not None) and (not rec["emitted"])
        n_confab_caught += int(caught)
        confab.append({**rec, "true_patient": p, "confab_patient": wrong, "caught": caught})

    # --- ANTI-CHEAT: prove the VERIFY re-parse is GENUINELY load-bearing role-assignment, not a trivial echo. ---
    # Without this, one could object that VERIFY only checks asserted==gated content equality (true in the stub,
    # since the BridgeParser faithfully re-parses an active SVO in order). These two probes show the re-parse does
    # real work: (1) a faculty that asserts the SAME fact in PASSIVE surface order is ACCEPTED only because the
    # parser REORDERS position->thematic-role to recover the canonical SVO (an identity echo would mis-order it ->
    # reject a TRUE fact); (2) a faculty that asserts the agent/patient SWAPPED (active voice) is CAUGHT (the
    # reordered SVO mismatches the gated fact) -- a content-ORDER confabulation, not just a word swap.
    anticheat = {"passive_roundtrip": [], "swapped_role_catch": []}
    for f in cur.get("facts", [])[:4]:
        a, ac, p = f
        gate = agent.what_does(a, ac)                      # gate the TRUE fact
        if gate is None:
            continue
        gate_svo = [a, ac, gate]
        # (1) passive-frame assertion of the SAME fact: surface order [patient, action, agent]; re-parse with
        # voice='passive' must REORDER it back to {agent:a, action:ac, patient:p}=gate_svo -> ACCEPT.
        passive_tokens = [p, ac, a]
        parsed = agent.parse(passive_tokens, voice="passive")
        psvo = [parsed.get("agent"), parsed.get("action"), parsed.get("patient")]
        passive_ok = (psvo == gate_svo)                    # the reorder recovered the canonical fact
        anticheat["passive_roundtrip"].append({
            "fact": [a, ac, p], "passive_surface_tokens": passive_tokens, "reparsed_svo": psvo,
            "gate_svo": gate_svo, "accepted": passive_ok})
        # (2) agent/patient SWAPPED, active voice: assert [patient, action, agent] AS active -> the parser reads it
        # positionally as {agent:p, action:ac, patient:a} != gate_svo -> CAUGHT (a role-order confabulation).
        swapped_active = [p, ac, a]
        parsed2 = agent.parse(swapped_active, voice="active")
        ssvo = [parsed2.get("agent"), parsed2.get("action"), parsed2.get("patient")]
        swap_caught = (ssvo != gate_svo)                   # the swapped assertion does NOT match the gated fact
        anticheat["swapped_role_catch"].append({
            "fact": [a, ac, p], "swapped_active_tokens": swapped_active, "reparsed_svo": ssvo,
            "gate_svo": gate_svo, "caught": swap_caught})
    n_passive_ok = sum(x["accepted"] for x in anticheat["passive_roundtrip"])
    n_passive = len(anticheat["passive_roundtrip"])
    n_swap_caught = sum(x["caught"] for x in anticheat["swapped_role_catch"])
    n_swap = len(anticheat["swapped_role_catch"])

    n_grounded = len(grounded)
    n_untaught = len(untaught)
    n_confab = len(confab)
    return {
        "seed": seed,
        "taught": taught,
        "grounded_correct": n_grounded_ok,
        "grounded_total": n_grounded,
        "grounded_rate": (n_grounded_ok / n_grounded) if n_grounded else None,
        "untaught_abstain": n_untaught_abstain,
        "untaught_total": n_untaught,
        "untaught_abstain_rate": (n_untaught_abstain / n_untaught) if n_untaught else None,
        "confab_caught": n_confab_caught,
        "confab_total": n_confab,
        "confab_caught_rate": (n_confab_caught / n_confab) if n_confab else None,
        "anticheat_passive_roundtrip_ok": n_passive_ok,
        "anticheat_passive_roundtrip_total": n_passive,
        "anticheat_swapped_role_caught": n_swap_caught,
        "anticheat_swapped_role_total": n_swap,
        "grounded_detail": grounded,
        "untaught_detail": untaught,
        "confab_detail": confab,
        "anticheat_detail": anticheat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    with open(os.path.abspath(CURRICULUM), "r", encoding="utf-8") as fh:
        cur = json.load(fh)
    vocab = _collect_vocab(cur)
    # the adversarial confab swap uses '<token>_X' fall-backs; ensure they're encodable (add to vocab so the
    # composer can encode them, though they should never be needed since distinct taught patients exist).
    vocab = sorted(set(vocab) | {p + "_X" for p in {f[2] for f in cur.get("facts", [])}})

    print(f"[p3-derisk] backend={os.environ.get('SIM_BACKEND', 'auto')} vocab={len(vocab)} words")

    per_seed = []
    t0 = time.time()
    for seed in args.seeds:
        ts = time.time()
        try:
            r = run_seed(cur, seed, vocab)
        except Exception as e:
            r = {"seed": seed, "error": repr(e), "traceback": traceback.format_exc()}
            print(f"[p3-derisk] seed {seed} ERROR: {e!r}")
            traceback.print_exc()
        per_seed.append(r)
        dt = time.time() - ts
        if "error" not in r:
            print(f"[p3-derisk] seed {seed}: grounded {r['grounded_correct']}/{r['grounded_total']} "
                  f"(={r['grounded_rate']:.3f})  untaught-abstain {r['untaught_abstain']}/{r['untaught_total']}  "
                  f"confab-caught {r['confab_caught']}/{r['confab_total']}  "
                  f"[anti-cheat: passive-roundtrip {r['anticheat_passive_roundtrip_ok']}/{r['anticheat_passive_roundtrip_total']}, "
                  f"swapped-role-caught {r['anticheat_swapped_role_caught']}/{r['anticheat_swapped_role_total']}]  [{dt:.1f}s]")

    ok_seeds = [r for r in per_seed if "error" not in r]
    all_grounded = bool(ok_seeds) and all(r["grounded_rate"] == 1.0 for r in ok_seeds)
    all_moat = bool(ok_seeds) and all(r["untaught_abstain"] == r["untaught_total"] for r in ok_seeds)
    all_confab = bool(ok_seeds) and all(r["confab_caught"] == r["confab_total"] for r in ok_seeds)
    n_ge3 = len(ok_seeds) >= 3
    go = all_grounded and all_moat and all_confab and n_ge3

    min_grounded = min((r["grounded_rate"] for r in ok_seeds), default=None)
    min_moat = min((r["untaught_abstain_rate"] for r in ok_seeds), default=None)
    min_confab = min((r["confab_caught_rate"] for r in ok_seeds), default=None)
    # anti-cheat: the VERIFY re-parse must do REAL role-assignment (passive reorders a TRUE fact -> accept;
    # active role-swap -> catch). All-or-nothing across seeds; surfaced so a clean GO can't be a trivial echo.
    anticheat_passive_ok = bool(ok_seeds) and all(
        r["anticheat_passive_roundtrip_ok"] == r["anticheat_passive_roundtrip_total"] for r in ok_seeds)
    anticheat_swap_ok = bool(ok_seeds) and all(
        r["anticheat_swapped_role_caught"] == r["anticheat_swapped_role_total"] for r in ok_seeds)

    if go:
        verdict = (
            "GO -- grounded->fluent-correct AND untaught->abstain (moat held) AND injected-confab->caught-by-verify, "
            ">=3 seeds; the GATE->CONSTRAIN->VERIFY grounding mechanism PRESERVES the no-confab moat while a faculty "
            "supplies fluency. The architecture is sound; the real P1 faculty (a converted SLM) later swaps in for the stub."
        )
    else:
        leaks = []
        if not all_grounded:
            leaks.append(f"CONSTRAIN leaks (grounded min rate {min_grounded}) -- a grounded query did not emit a "
                         "verified fluent sentence")
        if not all_moat:
            leaks.append(f"GATE/moat leaks (untaught-abstain min rate {min_moat}) -- an untaught cue produced a sentence")
        if not all_confab:
            leaks.append(f"VERIFY leaks (confab-caught min rate {min_confab}) -- an injected confabulation was NOT "
                         "caught by the re-parse")
        if not n_ge3:
            leaks.append(f"only {len(ok_seeds)} seed(s) ran (need >=3)")
        verdict = "HONEST/NO-GO -- " + " ; ".join(leaks)

    summary = {
        "curriculum": os.path.relpath(os.path.abspath(CURRICULUM),
                                      os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))),
        "backend": os.environ.get("SIM_BACKEND", "auto"),
        "vocab_size": len(vocab),
        "seeds": args.seeds,
        "n_seeds_ok": len(ok_seeds),
        "architecture": "GATE (composer exact-match recall / abstain) -> CONSTRAIN (template-stub faculty "
                        "renders the gated SVO into fluent surface form, content-locked) -> VERIFY (re-parse the "
                        "asserted SVO with the same BridgeParser; reject on mismatch with the stored fact)",
        "layer_mapping": {
            "(a) grounded": "tests CONSTRAIN -- the rendered sentence re-parses back to the taught fact",
            "(b) untaught": "tests GATE/moat -- the gate gives the faculty nothing -> no sentence",
            "(c) injected-confab": "tests VERIFY -- a deliberately-wrong faculty SVO is caught by the re-parse",
        },
        "faculty": "TEMPLATE-STUB (standing in for the real P1 LLM -- NO model download). Emits a fluent surface "
                   "form + the canonical content SVO it asserts; the real LLM's asserted SVO is recovered by "
                   "re-parsing its prose.",
        "all_grounded_correct": all_grounded,
        "min_grounded_rate": min_grounded,
        "all_moat_held": all_moat,
        "min_untaught_abstain_rate": min_moat,
        "all_confab_caught": all_confab,
        "min_confab_caught_rate": min_confab,
        "anticheat_passive_roundtrip_all_ok": anticheat_passive_ok,
        "anticheat_swapped_role_all_caught": anticheat_swap_ok,
        "anticheat_note": "VERIFY is genuinely role-assignment (not an echo): passive-frame TRUE fact accepted "
                          "via reorder; active role-swap caught. Both all-or-nothing across seeds.",
        "GO": go,
        "verdict": verdict,
        "elapsed_seconds": round(time.time() - t0, 1),
        "per_seed": per_seed,
    }

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[p3-derisk] VERDICT: {verdict}")
    print(f"[p3-derisk] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
