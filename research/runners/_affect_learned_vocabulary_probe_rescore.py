"""POST-HOC re-score of the tone-selection seed-7 restyle probe (research/findings/raw/_affect_tone_selection/
amend1_probe/restyle_probe_s7.json) with (a) the learned affect vocabulary switched on (seed-7 DEV weights) and (b) the
content lock's new content-word-recall term. No text is regenerated: the stored draft and rewrites are re-appraised
and re-locked, and the AMENDMENT 1 choice rule (`choose_variant`) is applied to the result.

Seed 7 is not an evaluation seed of either pre-registration. This answers one question: with the brain able to hear
more negative wording, and refusals rejected, how much of the probe's coverage failure is left, and what is it?
"""
from __future__ import annotations

import copy
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

SRC = os.path.join(_REPO, "research", "findings", "raw", "_affect_tone_selection", "amend1_probe", "restyle_probe_s7.json")
DST = os.path.join(_REPO, "research", "findings", "raw", "_affect_learned_vocab", "dev_s7", "tone_probe_rescore_s7.json")


def rescore(weights_path):
    from research.runners import affect_production_organ as AO
    from research.runners import affect_learned_vocabulary as A
    from research.runners import _lbf_affect_tone_selection_derisk as T
    from webapp import affect_tone_selection as ATS
    with open(SRC) as fh:
        base = json.load(fh)
    arms = {}
    for arm, flag, new_lock in (("original", False, False), ("lock_only", False, True),
                                ("vocab_only", True, False), ("vocab_and_lock", True, True)):
        os.environ.pop("BRAIN_AFFECT_LEARNED_VOCAB", None)
        if flag:
            os.environ["BRAIN_AFFECT_LEARNED_VOCAB"] = "1"
            os.environ["BRAIN_AFFECT_LEARNED_VOCAB_PATH"] = weights_path
            os.environ["BRAIN_CHAT_SEED"] = "7"
            A._READER.clear()
        d = copy.deepcopy(base)
        for r in d["rows"]:
            r["draft_appraisal"] = AO.appraise_text(r["draft"])["valence"]
            for v in d["variants"]:
                for c in r["variants"][v]["candidates"]:
                    c["appraisal"] = AO.appraise_text(c["text"])["valence"]
                    if new_lock:
                        facts = []          # the stored lock detail kept the fact-word verdict; re-apply it below
                        ok, det = ATS.content_lock(r["draft"], c["text"], facts)
                        c["lock_ok"] = bool(c["lock_ok"]) and not det["low_content_recall"]
                        c["content_recall"] = det["content_recall"]
        arms[arm] = T.summarize_probe(d)
    os.environ.pop("BRAIN_AFFECT_LEARNED_VOCAB", None)
    out = {"what": __doc__, "source": os.path.relpath(SRC, _REPO), "weights": os.path.relpath(weights_path, _REPO)
           if weights_path.startswith(_REPO) else weights_path, "arms": arms}
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(arms, indent=1))
    return out


if __name__ == "__main__":
    rescore(sys.argv[1])
