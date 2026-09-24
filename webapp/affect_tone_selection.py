"""AFFECT SELECTS THE REPLY'S TONE WHILE ITS CONTENT IS HELD FIXED (D5 "feel" over the OPEN reply) — DEFAULT-OFF.

WHY THIS METHOD (research/findings/2026-09-23-affect-conditioned-mouth-live-valence-diagnosis-and-staged-derisk.md
and its amend1 data): conditioning the Qwen mouth's GENERATION on the organ's valence (a graded MOOD line, or
residual-stream steering) changed the reply's CONTENT, not only its tone. The greedy spiking-Qwen decode is path-
sensitive: any change to the prompt or the residual stream sends it down a different trajectory, so a known-topic
reply about 'Frank Lincoln Wright' became one about Franklin D. Roosevelt in the treatment AND in the shuffled-valence
control. Tone and content were produced by one act, so one could not move without the other.

THIS METHOD SEPARATES THE TWO ACTS:
  1. DRAFT (content). The mouth writes the reply ONCE from the production prompt with the MOOD line replaced by a
     fixed, affect-free line. The draft cannot depend on the organ's state.
  2. RESTYLE (proposals). The mouth rewrites that one draft in a fixed set of tones (STYLES), each at a fixed decode
     seed. The proposals do not depend on the organ's state either.
  3. CONTENT LOCK (host verification, a declared shortcut). A restyle is admissible only if it keeps every number,
     every proper name and every retrieved-fact word the draft carried, and adds no number or proper name the draft
     did not have. The draft itself is always admissible (candidate 0).
  4. THE BRAIN EVALUATES EACH PROPOSAL. Each admissible candidate is appraised and read through the SAME spiking
     affect organ that holds the mood (`AffectProductionOrgan.read_differential`, snapshot-isolated, so reading a
     candidate does not change the held mood). This is the perceptual-loop idea (Levelt 1983; Hartsuiker & Kolk
     2001, doi:10.1006/cogp.2000.0744): the speaker's comprehension system monitors a planned utterance before it
     is released, and that monitor is sensitive to emotional valence (Slevc & Ferreira 2006,
     doi:10.1016/j.jml.2005.11.002).
  5. SELECTION (mood congruence). The released reply is the candidate whose organ-evoked valence is CLOSEST to the
     organ's held valence (ties -> the lowest index, so the draft wins a tie). Held valence 0 (the BRAIN_AFFECT_LESION
     arm) -> the most neutral-reading candidate. ML analogue: Prompt-and-Rerank (Suzgun, Melas-Kyriazi & Jurafsky,
     EMNLP 2022, https://arxiv.org/abs/2205.11503), where the rerank axes here are content (a hard lock) and the
     brain's own affect match.

BRAIN-BASED BOUNDARY (declared, per CLAUDE.md):
  * brain: the held mood (spiking organ) and each candidate's evoked differential (the same spiking organ).
  * host shortcuts: (a) the appraisal lexicon that turns candidate text into the organ's input (`appraise_text`,
    the existing upstream shortcut the held mood also uses); (b) the content lock (string checks); (c) THE SELECTION
    ITSELF: argmin_i |v_i - v_held| is a host comparator + argmin. It is NOT on the substrate. The named next rung is
    to run it as the basal-ganglia race (research/runners/bg_action_selection_production_organ.py generalised to N
    channels, candidate salience = congruence read off the organ populations). A GO here is NOT credited to a
    spiking selection.
  * scaffold: the Qwen mouth (draft + restyles), the fixed STYLES list and restyle instruction (owner-ratified
    articulation scaffold, 2026-09-19).

Flag: BRAIN_OPEN_ENDED_AFFECT_TONE_SELECT in {1,true,on,yes}. Unset -> answer_turn never imports this module and
the production path runs byte-identically. HONESTY: functional read-out only; nothing here claims felt experience.
No sim/ edit.
"""
from __future__ import annotations

import hashlib
import os
import re

# ── preregistered operating point (fixed before any tone result existed) ─────────────────────────────────────────
NEUTRAL_MOOD_LINE = "MOOD: you feel even and steady."
# (key, descriptor). Descriptor words must be OUTSIDE the independent tone-scoring lexicon, so a reply that echoes the
# instruction is not scored for the echo (the runner's selftest enforces it). The set spans both signs at two
# intensities; it is fixed and identical in every arm.
STYLES = [
    ("pos2", "very warm, joyful and cheerful"),
    ("pos1", "slightly warmer and friendlier"),
    ("neg1", "slightly more subdued and wistful"),
    ("neg2", "very sad, somber and melancholy"),
]
RESTYLE_SYSTEM = ("You rewrite a reply so that it expresses a different emotional tone. Keep every name, number, "
                  "date, place and fact exactly as it is. Do not add or remove any facts. Change only the wording "
                  "that carries the tone. Output only the rewritten reply.")
RESTYLE_MAX_NEW_TOKENS = 160
STYLE_SEED_STRIDE = 7919          # decode seed of style k = seed + STYLE_SEED_STRIDE * (k + 1)

LAST_TRACE: dict = {}             # the most recent turn's selection trace (read in-process by the runner)
EVALUATOR = None                  # test seam: callable(text) -> dict(appraisal, differential, valence, n_hits)


def tone_select_enabled() -> bool:
    return os.environ.get("BRAIN_OPEN_ENDED_AFFECT_TONE_SELECT", "").strip().lower() in ("1", "true", "on", "yes")


def neutral_draft_system(system: str) -> str:
    """The production system prompt with its MOOD line replaced by a fixed affect-free line (all other lines kept)."""
    out, done = [], False
    for line in system.split("\n"):
        if line.startswith("MOOD: ") and not done:
            line, done = NEUTRAL_MOOD_LINE, True
        out.append(line)
    return "\n".join(out)


def restyle_user(descriptor: str, draft: str) -> str:
    return "Rewrite this reply in a %s tone:\n\n%s" % (descriptor, draft)


def clean_restyle(text: str) -> str:
    """Body-side tidy of a restyle: drop a leading 'Here is the rewritten reply:'-style line and wrapping quotes."""
    t = (text or "").strip()
    lines = t.split("\n")
    if len(lines) > 1 and lines[0].rstrip().endswith(":"):
        t = "\n".join(lines[1:]).strip()
    if len(t) >= 2 and t[0] in "\"'“" and t[-1] in "\"'”":
        t = t[1:-1].strip()
    return t


# ── the content lock ─────────────────────────────────────────────────────────────────────────────────────────────
_TOK = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+")


def _fold(w: str) -> str:
    w = w.lower()
    return w[:-1] if (len(w) > 3 and w.endswith("s")) else w


def numbers(text: str) -> set:
    return set(re.findall(r"\d+", text or ""))


def proper_names(text: str) -> set:
    """Capitalised word tokens that are NOT sentence-initial (and not 'I'), lower-cased + plural-folded."""
    out = set()
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        toks = [t for t in _TOK.findall(sent) if not t.isdigit()]
        for t in toks[1:]:
            if t[0].isupper() and t not in ("I", "I'm", "I've", "I'd", "I'll"):
                out.add(_fold(t))
    return out


def word_set(text: str) -> set:
    return {_fold(t) for t in _TOK.findall(text or "") if not t.isdigit()}


def fact_words_in(text: str, facts, prompt: str = "") -> set:
    """Retrieved-fact OBJECT content words (len>2, not stop, not in the prompt) that the text actually carries."""
    try:
        from research.runners._affect_distributional_tag_derisk import STOP
    except Exception:
        STOP = set()
    pw = word_set(prompt)
    fw = set()
    for f in facts or []:
        obj = f[-1] if isinstance(f, (list, tuple)) and f else str(f)
        for w in word_set(str(obj).replace("_", " ")):
            if len(w) > 2 and w not in STOP and w not in pw:
                fw.add(w)
    return fw & word_set(text)


def content_lock(draft: str, cand: str, facts=(), prompt: str = "") -> tuple:
    """(ok, detail). ok iff cand is non-empty, keeps every number / proper name / fact word the draft carries, and
    adds no number or proper name the draft lacks."""
    cw = word_set(cand)
    dw = word_set(draft)
    miss_num = sorted(numbers(draft) - numbers(cand))
    new_num = sorted(numbers(cand) - numbers(draft))
    miss_name = sorted(proper_names(draft) - cw)
    new_name = sorted(proper_names(cand) - dw)
    miss_fact = sorted(fact_words_in(draft, facts, prompt) - cw)
    ok = bool((cand or "").strip()) and not (miss_num or new_num or miss_name or new_name or miss_fact)
    return ok, {"missing_numbers": miss_num, "new_numbers": new_num, "missing_names": miss_name,
                "new_names": new_name, "missing_fact_words": miss_fact}


# ── the brain's evaluation of a proposal (the spiking organ) ────────────────────────────────────────────────────
def _organ_evaluate(text: str) -> dict:
    from research.runners import affect_production_organ as AO
    from research.runners._open_ended_state_driven_generation_derisk import _valence_from_differential
    seed = int(os.environ.get("BRAIN_CHAT_SEED", "42") or 42)
    organ = AO.get_organ(seed=seed)
    appr = AO.appraise_text(text)
    # lesion=False EXPLICITLY: BRAIN_AFFECT_LESION removes the HELD mood (the claimed edge into the selector); the
    # organ's evaluation of a planned utterance is a different use of it and is left intact (specific-edge lesion).
    read = organ.read_differential(float(appr["valence"]), lesion=False)
    d = float(read["differential"])
    return {"appraisal": float(appr["valence"]), "n_hits": int(appr.get("n_hits", 0)),
            "differential": d, "valence": _valence_from_differential(d)}


def evaluate(text: str) -> dict:
    return (EVALUATOR or _organ_evaluate)(text)


def select_by_mood(cand_valences, v_held: float) -> int:
    """HOST comparator + argmin (declared shortcut): the index whose evoked valence is closest to the held valence;
    ties -> the lowest index."""
    best, best_d = 0, None
    for i, v in enumerate(cand_valences):
        d = abs(float(v) - float(v_held))
        if best_d is None or d < best_d - 1e-12:
            best, best_d = i, d
    return best


def _sha(t: str) -> str:
    return hashlib.sha256((t or "").encode("utf-8")).hexdigest()[:16]


def generate_selected(gen, system: str, user: str, valence: float, *, facts=(), seed: int, max_new_tokens: int):
    """The entry point answer_turn calls when the flag is on and the Qwen one-shot path is taken. Returns (raw, secs)."""
    total = 0.0
    draft, s = gen.generate(neutral_draft_system(system), user, seed=seed, max_new_tokens=max_new_tokens)
    total += float(s or 0.0)
    cands = [{"style": "draft", "text": draft, "lock_ok": True, "lock": None, "decode_seed": int(seed)}]
    for k, (key, desc) in enumerate(STYLES):
        dseed = int(seed) + STYLE_SEED_STRIDE * (k + 1)
        raw, s = gen.generate(RESTYLE_SYSTEM, restyle_user(desc, draft), seed=dseed,
                              max_new_tokens=RESTYLE_MAX_NEW_TOKENS)
        total += float(s or 0.0)
        txt = clean_restyle(raw)
        ok, det = content_lock(draft, txt, facts, user)
        cands.append({"style": key, "text": txt, "lock_ok": ok, "lock": det, "decode_seed": dseed})
    adm = [i for i, c in enumerate(cands) if c["lock_ok"]]
    for i in adm:
        cands[i]["eval"] = evaluate(cands[i]["text"])
    vals = [cands[i]["eval"]["valence"] for i in adm]
    j = adm[select_by_mood(vals, valence)]
    LAST_TRACE.clear()
    LAST_TRACE.update({"mode": "select", "v_held": float(valence), "draft_sha": _sha(draft),
                       "n_admissible": len(adm), "admissible": adm, "selected": j,
                       "selected_style": cands[j]["style"],
                       "candidates": [{"style": c["style"], "sha": _sha(c["text"]), "text": c["text"],
                                       "lock_ok": c["lock_ok"], "lock": c["lock"], "decode_seed": c["decode_seed"],
                                       "eval": c.get("eval")} for c in cands]})
    return cands[j]["text"], round(total, 2)
