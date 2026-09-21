"""LOAD-BEARING FRACTION — the instrument for the #1 project metric (owner-ratified 2026-09-19).

THE METRIC (docs/plans/2026-09-19-roadmap-with-a-permanent-llm-mouth.md §4; memory
project_2026_09_19_permanent_llm_mouth_loadbearing_metric): the FRACTION of production faculties where LESIONING the
brain's contribution provably CHANGES the reply. Operationalises the standing bar (feedback_faculties_must_drive_not_
observe): a faculty is REAL only when varying its state changes the reply AND lesioning it makes that difference vanish.
The permanent LLM mouth is acceptable BY DEFINITION only while this fraction is high + lesion-verified — the LLM words,
the brain decides. When it falls, we have "an LLM wearing a brain", the thing the project rejects.

REUSE, NOT REINVENT. This is the SHIPPED-FACULTY REGRESSION BATTERY (research/runners/onebrain_regression_battery.py)
run with the SECOND ARM = "lesion this faculty's brain contribution" instead of "flip an unrelated flag", and the
verdict INVERTED. It imports and reuses, verbatim:
  * `_spawn_arm(env, turn_labels, out_path)` — build ONE fresh brain (seed 42, numpy/CPU, stub renderer, no LLM) in a
    subprocess with a given env and run probe turns through the REAL webapp.server.brain_chat. (Each lesion is an env
    flag the fresh build reads, so no in-process monkeypatching is needed — the battery worker is reused as-is.)
  * `compare(intact, lesioned, faculties=[row])` — the DECISION-ONLY (categorical) equality check, with the continuous
    NOISE fields (rates/levels/mood/margins/...) already excluded. The battery calls a diff a "regression"; here a diff
    is a CHANGE = the faculty is LOAD-BEARING.
  * `PROBE_TURNS` / `_TURN_BY_LABEL` / `FACULTY_PROBES` — the per-faculty (probe turn, decision fields) registry, and
    the shared-session dependency ordering (hold->held, dr_a->dr_b->dr_c, bc_a->bc_b).

THE VERDICT INVERSION (the whole point). Regression battery: PASS = the two arms decide IDENTICALLY (no unrelated flip
broke the faculty). Load-bearing: LOAD_BEARING = the two arms decide DIFFERENTLY (lesioning the brain's contribution
changed the decision). So per faculty, reusing the SAME compare():
    compare(intact, lesioned) verdict == "regressed" (a decision field differs)  -> LOAD-BEARING
    compare(...)             verdict == "pass"        (present + identical)       -> NOT load-bearing (cosmetic)
    compare(...)             verdict == "not-exercised"(fields absent both arms)  -> NOT-EXERCISED (thin probe)

THE LESION KNOB, PER FACULTY (FACULTY_LESIONS below). The metric wants to cut the BRAIN's contribution while leaving the
organ INSTALLED — not to disable the whole wiring block. Most production organs ship a dedicated `BRAIN_<X>_LESION=1`
knob that does exactly that (e.g. affect_drives_chat.py: BRAIN_AFFECT_DRIVES_LESION cuts the interoceptive->ladder
synapses so the felt mood collapses to ~0 and the affective lead VANISHES, the organ otherwise untouched). Those are the
gold-standard lesions used here. Where a faculty has no dedicated neural-cut knob we record HOW it is lesioned honestly:
  neural-lesion   a dedicated BRAIN_<X>_LESION flag cuts the neural read (the organ stays installed).  [PRIMARY]
  whether-disable no neural-cut knob; the BRAIN_<X>=0 master switch removes the organ (weaker: a byte-identical escape
                  may leave the ANSWER unchanged while only the decision-field presence swings). Noted per row.
  in-process      lesioned only by an in-process monkeypatch (parser.role_of / composer.query / _substrate_recall) —
                  ALREADY covered behaviorally by research/runners/_production_lesion_probe.py (CHOOSE/GENERATE/LEARN);
                  this env-flag instrument does not reach it. NOT a gap in coverage, a gap in THIS harness's mechanism.
  thin            a dedicated lesion flag EXISTS but the brain_chat-only probe set cannot construct the trigger (the
                  gnw-deliberation / value-driven-choice >=2-distinct-patient conflict — see the battery's own note).
  mechanism-only  the flip changes the SUBSTRATE not the ANSWER by design (one-brain-substrate: onebrain==rf answers).
  proposed        no clean lesion knob exists yet; the minimal one to add is named in the note. (NOT faked.)

RELIABILITY — "changed, and not just noise" (the metric's own qualifier). The webapp default builds the tiny-demo brain
and every organ at seed 42 (hardcoded; no per-request seed), so a given env is DETERMINISTIC: re-running an arm yields
byte-identical decisions. The anti-noise guarantee is therefore made EXECUTABLE, not assumed:
  (1) DETERMINISM SELF-CHECK: the intact arm for a turn-group is built TWICE and its decision fields must match. If they
      do not, the harness is non-deterministic and EVERY load-bearing verdict is untrustworthy -> reported loudly, the
      metric is marked UNRELIABLE. (This is the real "not just noise" proof: with the harness proven deterministic, any
      intact-vs-lesion decision diff is attributable to the lesion, not to run-to-run variation.)
  (2) LESION-REPEAT: each lesion arm is (optionally, --repeats>1) rebuilt and its decision fields must reproduce; a
      diff that does not reproduce is flagged `noisy` and EXCLUDED from the load-bearing count.
  (3) CHANGE KIND: a diff is classified `structural` (a field present intact / absent-or-null lesioned, i.e. the
      organ's output is genuinely gated off) vs `value` (both present, categorical value flips). Structural changes are
      inherently robust to any RNG-trajectory shift; value-only changes are still counted but flagged for the record.
  (4) FLAG-RESOLVES GUARD: a declared lesion flag that does NOT appear in source is reported as `lesion-knob-missing`,
      NEVER silently as "not load-bearing" — a no-op flag that reads as "no change" is the exact false-negative this
      metric must not make (the "a check that cannot fail" lesson, applied to a lesion that cannot bite).

HONEST BOUNDARY. Same as the battery's: this is a REACHABILITY + DECISION-DIFFERENCE instrument over the tiny-demo
brain_chat path, not a proof of each faculty's correctness. LOAD_BEARING_FRACTION is reported over the DENOMINATOR of
faculties this harness can actually lesion+exercise (neural-lesion / whether-disable with a driving probe); the
in-process / thin / mechanism-only / proposed faculties are reported SEPARATELY with their reason, never silently folded
into either numerator or denominator.

CPU / tiny-smoke. numpy backend, stub renderer, no LLM (inherited from the battery worker). Every arm is a full-brain
build -> RUN UNDER tools/memcap.sh (the 2026-09-18 OOM guard). Peak RSS is ONE build at a time (arms are sequential);
each arm runs only its faculty's minimal turn-group (1-3 turns), far lighter than the battery's 26-turn single process.

Run (smoke, ONE faculty, capped):
  tools/memcap.sh 20 -- .venv/bin/python -m research.runners.load_bearing_fraction --smoke curiosity-followup \
      --out research/findings/raw/_load_bearing/smoke.json
Verify the EPISODIC DRIVING fix (default-off; flips episodic-memory hollow->load-bearing; cupy strongly preferred so
the forced BTSP write is ~seconds not ~510s/store):
  SIM_BACKEND=cupy LB_EPISODIC_DRIVE_PROBE=1 tools/memcap.sh 24 -- .venv/bin/python \
      -m research.runners.load_bearing_fraction --only episodic-memory --repeats 2 \
      --out research/findings/raw/_load_bearing/episodic_drive.json     # expect load-bearing=1, null-control clean
Run (full measurement, capped; defer to a non-gaming window):
  tools/memcap.sh 24 -- .venv/bin/python -m research.runners.load_bearing_fraction \
      --out research/findings/raw/_load_bearing/load_bearing.json
Self-test (no brain build, no cap needed): .venv/bin/python -m research.runners.load_bearing_fraction --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

# ── reuse the regression-battery machinery verbatim (no edit to that module) ─────────────────────────────────────
from research.runners.onebrain_regression_battery import (
    PROBE_TURNS,
    _EXTRA_TURNS,
    _TURN_BY_LABEL,
    FACULTY_PROBES,
    _spawn_arm as _spawn_arm_raw,
    compare,
    faculty_list,
)
# The attribution discipline (tools.lab): a treatment/control PAIR must make an explicit attribution call — measuring
# both arms is not the same as asking whose the difference was (the gap#5 clamp owned 97% of a change nobody subtracted).
# Here: treatment = decision fields changed INTACT-vs-LESION; control (null) = decision fields changed INTACT-vs-INTACT-
# REBUILD (same env, rebuilt). With the harness deterministic the null is 0 -> 100% of the change is the lesion; a
# non-zero null means the "change" is partly run-to-run noise and the load-bearing verdict is NOT clean.
from tools.lab import attributable_to

# RESUME-SKIP (opt-in, env-gated; default OFF -> byte-identical to a fresh run). When LB_RESUME_SKIP_EXISTING is set,
# an arm whose output file already exists AND parses as valid JSON is LOADED instead of rebuilt -- correct because
# _spawn_arm_raw's return value IS `json.load(open(out_path))`, so a loaded arm is identical to a freshly-built one.
# This makes a killed run resumable: the expensive per-faculty brain-builds are reused off disk; only missing or
# truncated arms rebuild. A file truncated by a mid-write kill fails json.load -> falls through to a real rebuild.
_LB_RESUME = os.environ.get("LB_RESUME_SKIP_EXISTING", "").strip().lower() in ("1", "true", "yes", "on")

# ── EPISODIC DRIVING PROBE (opt-in, env-gated; default OFF -> byte-identical to the 2026-09-19 hollow baseline) ────
# WHY (diagnosis, finding 2026-09-20-hollow-episodic-drive): episodic-memory is isolated-lesion-load-bearing (the dAP
# completion collapses 0.909->0.000 under BRAIN_EPISODIC_LESION) yet reads INTEGRATED-HOLLOW here for a PROBE reason,
# not a wiring reason. Its default probe turn `episodic` ("did we discuss the dog") runs on a FRESH session with NO
# prior storage, so the INTACT recall correctly reads in_memory=False (an honest not-in-memory) -- the SAME output the
# lesion produces (an unformed-weights collapse of a memory that was never formed). The recall GATE already DRIVES the
# reply (webapp/server.py Hook A: answer/abstained/verified/episodic ALL flip with in_memory); it simply never had a
# memory to recall. Compounding it, the BTSP WRITE is cupy-gated (server._episodic_store_ok), so on the numpy probe
# backend nothing is stored even if a storing turn were added. This flag makes the instrument CONSTRUCT the driving
# condition: remap the episodic probe to the store->recall pair (battery turns `epi_store`->`epi_recall`, session
# 'epi2') and FORCE the write with BRAIN_EPISODIC_STORE=1 so it runs on ANY backend (~510s/store on numpy@2000,
# ~seconds on cupy -- the declared latency residual; speed is secondary). Then intact recalls it (in_memory=True ->
# disclosure) and the lesion collapses it (in_memory=False -> "I don't recall") -> the decision field `episodic.in_
# memory` FLIPS -> LOAD-BEARING. OFF (default) -> episodic is measured on the lone `episodic` turn exactly as the
# baseline did (hollow), and no other faculty is touched.
LB_EPISODIC_DRIVE = os.environ.get("LB_EPISODIC_DRIVE_PROBE", "").strip().lower() in ("1", "true", "yes", "on")
_EPISODIC_DRIVE_TURN = "epi_recall"      # the referential RECALL turn (its group is store->recall, same session)
_EPISODIC_DRIVE_ENV = {"BRAIN_EPISODIC_STORE": "1"}   # force the BTSP write to execute on the probe backend


def _spawn_arm(env, turn_labels, out_path):
    if _LB_RESUME and os.path.exists(out_path):
        try:
            return json.load(open(out_path))
        except Exception:
            pass  # truncated/corrupt (e.g. killed mid-write) -> rebuild
    return _spawn_arm_raw(env, turn_labels, out_path)

# A sentinel env var NOTHING reads — the null lesion. Setting it changes no brain behavior; used by the self-test to
# confirm the instrument does NOT report a change when the "lesion" is a no-op (guards against a false-positive harness).
NULL_LESION_FLAG = "BRAIN_LOAD_BEARING_NULL_LESION"

# SEED THREADING (research/seed-threading-lbf, 2026-09-20): every arm build already reads the substrate seed off
# `BRAIN_CHAT_SEED` (webapp/server.py._brain_chat_seed + each per-organ workspace's own `_DEFAULT_SEED`, all
# reading the SAME env var) — see main()/run() below, which set it once for the whole invocation. `_seed_suffix`
# namespaces every per-arm output FILENAME by seed so LB_RESUME_SKIP_EXISTING and the in-memory intact_cache never
# false-skip/collide across a multi-seed sweep sharing one --out directory. seed=42 (the pre-existing hardcoded
# value) keeps the ORIGINAL, un-suffixed filenames — BYTE-IDENTICAL to every existing on-disk artifact and to the
# pre-this-change resume behavior; only a non-42 seed adds the `_s<seed>` tag.
def _seed_suffix(seed: int) -> str:
    return "" if int(seed) == 42 else "_s%d" % int(seed)


# ── the PER-FACULTY LESION MAP ───────────────────────────────────────────────────────────────────────────────────
# faculty_key -> {flag, value, kind, note}. `kind` is one of: neural-lesion / whether-disable / in-process / thin /
# mechanism-only / proposed (see the module docstring). `value` is the env value that LESIONS ("1" for a BRAIN_*_LESION
# neural cut; "0" for a BRAIN_* master-switch disable). `flag=None` for kinds this env-flag harness cannot drive.
# Every flag below was confirmed present in source (grep webapp/ research/runners/); _flag_resolves() re-checks at run
# time so a stale/typo'd flag surfaces as `lesion-knob-missing`, never as a silent "not load-bearing".
FACULTY_LESIONS = {
    # ---- core CHOOSE/RECALL/LEARN: in-process lesions, ALREADY behaviorally covered by _production_lesion_probe ----
    "content-selection":        dict(flag=None, value=None, kind="in-process",
        note="lesioned in-process by parser.role_of (junk role -> factual questions ABSTAIN) in _production_lesion_probe "
             "CHOOSE; no env knob. Minimal env lesion to add: BRAIN_SUBSTRATE_PARSE_LESION."),
    "semantic-recall":          dict(flag=None, value=None, kind="in-process",
        note="lesioned in-process by composer.query_patient (the battery module's own spiking_recall_lesion demo / "
             "_production_lesion_probe); no env knob. Minimal env lesion to add: BRAIN_COMPOSER_RECALL_LESION."),
    "moat-verify":              dict(flag=None, value=None, kind="in-process",
        note="the no-confab moat: lesioned in-process by chat._substrate_recall (lesion -> the abstain flips to a "
             "keyword confab), _production_lesion_probe CHOOSE. No env knob; minimal add: BRAIN_SUBSTRATE_RECALL_LESION."),
    "in-loop-learning":         dict(flag=None, value=None, kind="in-process",
        note="lesioned in-process by chat._substrate_recall (a fact taught this turn disappears), _production_lesion_"
             "probe LEARN. No env knob; minimal add: BRAIN_SUBSTRATE_RECALL_LESION."),
    "one-brain-substrate":      dict(flag="BRAIN_COMPOSER_KIND", value="rf", kind="mechanism-only",
        note="onebrain vs rf give the SAME answers BY DESIGN (ledger lesion_note) — a MECHANISM claim (recall on firing "
             "neurons), not an answer change. activity.composer label flips but that is cosmetic; NOT answer-load-bearing "
             "by construction. Excluded from the fraction."),
    # ---- dedicated BRAIN_*_LESION neural-cut knobs (PRIMARY: cut the neural read, organ stays installed) ----
    "comprehension-monitor":               dict(flag="BRAIN_COMPREHENSION_LESION", value="1", kind="neural-lesion", note=""),
    "comprehension-learned-animacy-cue":   dict(flag="BRAIN_LEARNED_ANIMACY_LESION", value="1", kind="neural-lesion",
        note="cuts the learned animacy cue; on a hand-table-OOV noun ('monkey') comprehension.on swings present/absent."),
    "comprehension-learned-verb-selects":  dict(flag="BRAIN_LEARNED_VERB_SELECTS_LESION", value="1", kind="neural-lesion",
        note="cuts the learned verb-selects cue; on a hand-table-OOV verb ('clean') comprehension.on swings."),
    "noncontradiction-gate":    dict(flag="BRAIN_NONCONTRADICTION_LESION", value="1", kind="neural-lesion", note=""),
    "affect-coloring":          dict(flag="BRAIN_AFFECT_LESION", value="1", kind="neural-lesion",
        note="affect_production_organ.py:242 BRAIN_AFFECT_LESION cuts the Gate-B ladder read."),
    "affect-drives-response":   dict(flag="BRAIN_AFFECT_DRIVES_LESION", value="1", kind="neural-lesion",
        note="cuts the interoceptive->ladder synapses -> felt mood collapses -> the affective lead vanishes. NOTE the "
             "'well' turn is mood-neutral (level 0) intact too, so the acted decision may be unchanged there; the change "
             "shows on an affective turn — a lower-confidence row on the neutral probe."),
    "affect-marker-spiking-wta":dict(flag="BRAIN_AFFECT_MARKER_SPIKING_LESION", value="1", kind="neural-lesion",
        note="cuts the felt-state->marker WTA selection; measured on the strongly-affective 'emo' turn (level>0)."),
    "da-mode-drives-response":  dict(flag="BRAIN_DA_DRIVES_LESION", value="1", kind="neural-lesion", note=""),
    "da-gated-encoding":        dict(flag="BRAIN_DA_ENCODING_LESION", value="1", kind="neural-lesion", note=""),
    "source-provenance-honesty":dict(flag="BRAIN_SOURCE_PROVENANCE_HONESTY_LESION", value="1", kind="neural-lesion",
        note="server.py get_organ(lesion=source_provenance_lesioned())."),
    "common-ground-drives":     dict(flag="BRAIN_CG_DRIVES_LESION", value="1", kind="neural-lesion", note=""),
    "swap-drives-response":     dict(flag="BRAIN_SWAP_DRIVES_LESION", value="1", kind="neural-lesion", note=""),
    "wm-binding-advanced":      dict(flag="BRAIN_MULTIREF_LESION", value="1", kind="neural-lesion", note=""),
    "prospective-memory":       dict(flag="BRAIN_PMEM_LESION", value="1", kind="neural-lesion", note=""),
    "pragmatic-implicature":    dict(flag="BRAIN_PRAGMATIC_LESION", value="1", kind="neural-lesion", note=""),
    "surprise-monitor":         dict(flag="BRAIN_SURPRISE_LESION", value="1", kind="neural-lesion", note=""),
    "metacog-monitor":          dict(flag="BRAIN_METACOG_LESION", value="1", kind="neural-lesion", note=""),
    "worldmodel-forward":       dict(flag="BRAIN_WORLDMODEL_LESION", value="1", kind="neural-lesion", note=""),
    "curiosity-followup":       dict(flag="BRAIN_CURIOSITY_LESION", value="1", kind="neural-lesion",
        note="removes the spiking ASK-pool crave drive -> the novel-abstain follow-up is silenced (DR-1 GO's own lesion "
             "arm). curiosity.curious flips True->False on the 'unknown' abstain turn."),
    "reconsolidation":          dict(flag="BRAIN_RECONSOLIDATION_LESION", value="1", kind="neural-lesion", note=""),
    "episodic-memory":          dict(flag="BRAIN_EPISODIC_LESION", value="1", kind="neural-lesion", note=""),
    "discourse-register":       dict(flag="BRAIN_DISCOURSE_REGISTER_LESION", value="1", kind="neural-lesion", note=""),
    "gnw-multistep-deliberation":dict(flag="BRAIN_GNW_MULTISTEP_LESION", value="1", kind="neural-lesion", note=""),
    "self-initiated-utterance": dict(flag="BRAIN_SELF_INITIATE_LESION", value="1", kind="neural-lesion", note=""),
    "vision-identity-spiking-hmax":dict(flag="BRAIN_VISION_IDENTITY_LESION", value="1", kind="neural-lesion", note=""),
    "bg-action-selection":      dict(flag="BRAIN_BG_SELECT_LESION", value="1", kind="neural-lesion", note=""),
    "open-ended-generation":    dict(flag="BRAIN_SPIKING_DRAW_LESION", value="1", kind="neural-lesion",
        note="cuts the vocab-agnostic spiking DRAW off cp_firing_states -> the generated hypothesis collapses "
             "(plausible-frac ~0.83->~0.04 in the B1 verify). Measured on the rich_open turn."),
    # ---- dedicated lesion flag EXISTS but the brain_chat-only probe cannot construct the trigger (thin) ----
    "gnw-deliberation":         dict(flag="BRAIN_GNW_DELIBERATE_LESION", value="1", kind="thin",
        note="lesion knob exists but the >=2-distinct-patient (agent,action) conflict is unconstructible via brain_chat "
             "(reconsolidation rewrites in place); the battery marks this probe thin. Not exercised by this harness."),
    "value-driven-choice":      dict(flag="BRAIN_VALUE_CHOICE_LESION", value="1", kind="thin",
        note="same root cause as gnw-deliberation: no 2-candidate conflict is constructible via brain_chat teaching."),
    # ---- no dedicated neural-cut knob; the master switch disables the whole organ (weaker) ----
    "confidence-forthcomingness":dict(flag="BRAIN_CONFIDENCE_FORTHCOMING", value="0", kind="whether-disable",
        note="no BRAIN_*_LESION knob; BRAIN_CONFIDENCE_FORTHCOMING=0 removes the organ -> the confidence_forthcoming key "
             "swings present/absent (a structural change, but a disable not a neural cut). Minimal neural lesion to add: "
             "BRAIN_CONFIDENCE_FORTHCOMING_LESION cutting the balance-of-evidence read while leaving the granted/reason "
             "trace attached."),
    # ---- no clean env lesion yet (in-process only or none); minimal proposal named ----
    "anaphora-wm":              dict(flag=None, value=None, kind="proposed",
        note="the compared field activity.roles is the parser's role binding on a DIRECT SVO ('the wolf watches the owl'), "
             "not anaphora resolution; the closest lesion is content-selection's in-process parser.role_of. No clean env "
             "lesion drives activity.roles. Minimal add: BRAIN_SUBSTRATE_PARSE_LESION (shared with content-selection)."),
    "discourse-planner":        dict(flag=None, value=None, kind="proposed",
        note="the rich multi-sentence plan (fields rich/n_sentences) has a neural_planner master switch but no neural-cut "
             "lesion. Minimal add: BRAIN_DISCOURSE_PLANNER_LESION cutting the plan's spiking sentence-count read."),
    "selective-attention-biased-competition":dict(flag=None, value=None, kind="proposed",
        note="the host content_bias lexicon was DELETED (ledger scaffold_retired:YES; learned SpikingFeatureCompat is the "
             "sole source), lesioned in-process by SpikingFeatureCompat.lesion(). No env knob. Minimal add: "
             "BRAIN_BIASED_COMPETITION_LESION reading the weights-cleared twin."),
}


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _faculty_row(key):
    for row in FACULTY_PROBES:
        if row[0] == key:
            return row  # (key, turn_label, fields, thin)
    return None


def turn_group(label):
    """The minimal ordered turn-group needed to reach `label`: all same-session turns up to and INCLUDING it (in
    declaration = execution order). Encodes the battery's shared-session dependencies (hold->held, dr_a->dr_b->dr_c,
    bc_a->bc_b, and the label-only epi_store->epi_recall) so a lesion arm reproduces the SAME session history the intact
    arm sees — a clean per-faculty control. Iterates PROBE_TURNS + _EXTRA_TURNS so the label-only driving pair (which is
    deliberately kept OUT of the default roster) still resolves to its store->recall group."""
    target = _TURN_BY_LABEL[label]
    sess = target[2]
    grp = []
    for t in list(PROBE_TURNS) + list(_EXTRA_TURNS):
        if t[2] == sess:
            grp.append(t[0])
        if t[0] == label:
            break
    return grp


def _flag_resolves(flag):
    """True iff the lesion flag string appears in an ORGAN source file (webapp/ or research/runners/), EXCLUDING this
    runner's own file (where every flag is named in FACULTY_LESIONS). Anti-false-negative guard: a flag that does not
    resolve is a NO-OP lesion that would read as 'not load-bearing' — reported as `lesion-knob-missing` instead. (This
    is presence-based: a necessary condition against typos/stale names, not a proof the flag gates the measured field —
    that is what the measurement itself decides.)"""
    if not flag:
        return False
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (research/ is one below)
    proj = os.path.dirname(root)
    for sub in ("webapp", os.path.join("research", "runners")):
        d = os.path.join(proj, sub)
        try:
            p = subprocess.run(["grep", "-rqlE", "--exclude-dir=__pycache__", "--include=*.py",
                                "--exclude=%s" % os.path.basename(__file__), flag, d],
                               capture_output=True)
            if p.returncode == 0:
                return True
        except Exception:
            pass
    return False


def _classify_diffs(diffs):
    """structural = a field goes present<->absent/null (organ output gated off); value = both present, value flips."""
    kinds = set()
    for d in diffs:
        on, off = d.get("on"), d.get("off")
        if on is None or off is None:
            kinds.add("structural")
        else:
            kinds.add("value")
    if "structural" in kinds:
        return "structural"
    return "value" if kinds else "none"


# ── the per-faculty lesion measurement ───────────────────────────────────────────────────────────────────────────
def _n_decision_diffs(row, arm_a, arm_b):
    """The number of the faculty's DECISION fields that differ between two arms (categorical, noise excluded). Reuses
    compare() so the exact same field set / noise exclusion / present-vs-absent rules apply."""
    return len(compare(arm_a, arm_b, faculties=[row])["per_faculty"][0]["diffs"])


def measure_faculty(key, out_dir, repeats=1, intact_cache=None, seed=42):
    """Build the INTACT arm TWICE (a, cached per turn-group; b, the NULL control) and the LESION arm for `key`, then
    make the explicit attribution call: TREATMENT = decision fields changed intact-vs-lesion; CONTROL = decision fields
    changed intact-vs-intact-rebuild. load-bearing requires a treatment change (attributed to the lesion) AND a clean
    null control (0 changes intact-vs-intact) — a change that also appears in the null is run-to-run noise, not the
    lesion. Returns the per-faculty result dict.

    `seed` (default 42, byte-identical to before this param existed) is NOT passed to _spawn_arm as an env override
    — the substrate seed is threaded via the process-wide BRAIN_CHAT_SEED env var set once by main()/run() for the
    whole invocation, so every arm build (including this faculty's) already builds at that seed. `seed` here only
    namespaces the OUTPUT FILENAMES (via `_seed_suffix`) so a multi-seed sweep sharing one --out dir cannot collide
    or false-skip under LB_RESUME_SKIP_EXISTING."""
    spec = FACULTY_LESIONS.get(key)
    row = _faculty_row(key)
    res = {"faculty": key, "turn": (row[1] if row else None),
           "kind": (spec or {}).get("kind"), "flag": (spec or {}).get("flag"),
           "load_bearing": None, "verdict": None, "change_kind": None, "diffs": [],
           "treatment_diffs": None, "control_diffs": None, "attributable_fraction": None,
           "null_control_clean": None, "lesion_reproduced": None, "flag_resolves": None,
           "note": (spec or {}).get("note", "")}
    if row is None:
        res["verdict"] = "unmapped-in-battery"; return res
    if spec is None:
        res["verdict"] = "unmapped-lesion"; return res
    if spec["kind"] in ("in-process", "thin", "mechanism-only", "proposed"):
        # not drivable by this env-flag harness — report the reason, not a fake number.
        res["verdict"] = "not-covered:" + spec["kind"]
        res["flag_resolves"] = _flag_resolves(spec["flag"]) if spec["flag"] else None
        return res
    flag, val = spec["flag"], spec["value"]
    res["flag_resolves"] = _flag_resolves(flag)
    if not res["flag_resolves"]:
        res["verdict"] = "lesion-knob-missing"      # a flag that is not read anywhere -> would be a silent no-op
        res["load_bearing"] = None
        return res

    # EPISODIC DRIVING remap (default-off; see LB_EPISODIC_DRIVE). Make the episodic probe exercise its load-bearing
    # recall path: remap to the store->recall turn (its group is derived below as ['epi_store','epi_recall'] because
    # both are in session 'epi2', declared store-first) and FORCE the BTSP write so it runs on any backend. base_env is
    # applied to BOTH the intact and lesion arms (so the NULL control also stores -> both intact arms read in_memory=
    # True -> clean null; only the lesion collapses it). Every OTHER faculty keeps base_env={} -> byte-identical.
    base_env = {}
    if LB_EPISODIC_DRIVE and key == "episodic-memory":
        row = ("episodic-memory", _EPISODIC_DRIVE_TURN, ["episodic.in_memory"], False)
        base_env = dict(_EPISODIC_DRIVE_ENV)
        res["turn"] = _EPISODIC_DRIVE_TURN
        res["note"] = "LB_EPISODIC_DRIVE_PROBE: store->recall on session 'epi2' + BRAIN_EPISODIC_STORE=1. " + res["note"]

    grp = turn_group(row[1])
    # cache key includes base_env so a stored (BRAIN_EPISODIC_STORE) intact arm never aliases a plain-{} arm on a
    # shared turn-group (the driving group is unique anyway, but keep the key honest).
    env_sig = ",".join("%s=%s" % (k, v) for k, v in sorted(base_env.items()))
    grp_sig = ",".join(grp) + ("|" + env_sig if env_sig else "")
    _fname = grp_sig.replace(",", "_").replace("|", "__").replace("=", "-")
    intact_cache = intact_cache if intact_cache is not None else {}
    _sfx = _seed_suffix(seed)

    # INTACT arm (base env = all defaults on, plus any driving base_env), built TWICE: `a` (cached per turn-group,
    # shared across faculties on the same turn) and `b` the NULL control (a fresh rebuild at the same seed -> the
    # run-to-run baseline of "no change").
    if grp_sig not in intact_cache:
        # union: base_env carries the driving-probe env (episodic + hollow drives); _sfx threads the per-seed suffix.
        a = _spawn_arm(dict(base_env), grp, os.path.join(out_dir, "intact_a_%s%s.json" % (_fname, _sfx)))
        b = _spawn_arm(dict(base_env), grp, os.path.join(out_dir, "intact_b_%s%s.json" % (_fname, _sfx)))
        intact_cache[grp_sig] = (a, b)
    intact_a, intact_b = intact_cache[grp_sig]

    # LESION arm. union: _sfx threads the per-seed suffix; base_env carries the driving-probe env (both arms share it).
    les_out = os.path.join(out_dir, "lesion_%s%s.json" % (key.replace("-", "_"), _sfx))
    lesioned = _spawn_arm({**base_env, flag: val}, grp, les_out)
    if intact_a is None or intact_b is None or lesioned is None:
        res["verdict"] = "arm-build-failed"; return res

    treat_pf = compare(intact_a, lesioned, faculties=[row])["per_faculty"][0]
    res["verdict"] = treat_pf["verdict"]        # "regressed" (=changed) / "pass" (=identical) / "not-exercised"
    res["diffs"] = treat_pf["diffs"]
    res["change_kind"] = _classify_diffs(treat_pf["diffs"]) if treat_pf["diffs"] else "none"
    treatment_diffs = len(treat_pf["diffs"])
    control_diffs = _n_decision_diffs(row, intact_a, intact_b)   # the NULL: must be 0 on a deterministic harness
    res["treatment_diffs"] = treatment_diffs
    res["control_diffs"] = control_diffs
    res["null_control_clean"] = (control_diffs == 0)

    # THE ATTRIBUTION CALL (tools.lab): what fraction of the observed decision change is the lesion vs the null control.
    res["attributable_fraction"] = attributable_to("load-bearing[%s]" % key, treatment_diffs, control_diffs)

    # LESION-REPEAT (optional extra anti-noise): rebuild the lesion arm and require the SAME verdict.
    reproduced = True
    for i in range(max(0, repeats - 1)):
        les2 = _spawn_arm({**base_env, flag: val}, grp, les_out + ".rep%d" % i)
        if les2 is None or compare(intact_a, les2, faculties=[row])["per_faculty"][0]["verdict"] != treat_pf["verdict"]:
            reproduced = False
            break
    res["lesion_reproduced"] = reproduced

    # Load-bearing iff: the decision CHANGED under lesion, the change is NOT present in the null control (clean
    # attribution -> the change is the lesion's, not noise), and it reproduces.
    if treat_pf["verdict"] == "not-exercised":
        res["load_bearing"] = None
    elif treat_pf["verdict"] == "pass":
        res["load_bearing"] = False
    elif not res["null_control_clean"]:
        res["load_bearing"] = None
        res["verdict"] = "noisy-null-control"       # the intact arm itself changed run-to-run -> verdict untrustworthy
    elif not reproduced:
        res["load_bearing"] = None
        res["verdict"] = "noisy"
    else:
        res["load_bearing"] = True                  # changed, attributable to the lesion, reproduced
    return res


# ── the full measurement ─────────────────────────────────────────────────────────────────────────────────────────
def run(out_dir="research/findings/raw/_load_bearing", only=None, repeats=1, seed=42):
    """`seed` (default 42, byte-identical): the substrate seed for this WHOLE invocation. Callers (main() below) are
    responsible for setting the process-wide BRAIN_CHAT_SEED env var to this same value BEFORE calling run() — this
    function does not set it itself (it may be called directly, e.g. from a test, without the env side effect) —
    `seed` here is threaded only to `measure_faculty` for output-filename namespacing (`_seed_suffix`)."""
    os.makedirs(out_dir, exist_ok=True)
    report = {"runner": "research.runners.load_bearing_fraction",
              "metric": "load_bearing_fraction", "repeats": repeats, "seed": seed}

    keys = only or faculty_list()
    intact_cache = {}
    per = [measure_faculty(k, out_dir, repeats=repeats, intact_cache=intact_cache, seed=seed) for k in keys]
    report["per_faculty"] = per

    # DETERMINISM / NULL CONTROL, aggregated from every exercised faculty's intact-vs-intact-rebuild control. The
    # harness is deterministic iff every null control is clean (0 changes); a dirty null makes that faculty's verdict
    # untrustworthy (its "change" is partly run-to-run noise). This IS the metric's "not just noise" proof, now
    # per-turn-group rather than a single representative turn.
    checked = [p for p in per if p.get("control_diffs") is not None]
    dirty = [p["faculty"] for p in checked if not p.get("null_control_clean")]
    report["determinism"] = {"deterministic": (len(dirty) == 0) if checked else None,
                             "n_null_controls_checked": len(checked), "dirty_null_controls": dirty}
    if dirty:
        report["UNRELIABLE"] = ("intact arm is NON-DETERMINISTIC at seed 42 for %s -> those load-bearing verdicts are "
                                "untrustworthy (the change is partly run-to-run noise); seed the substrate" % dirty)

    # DENOMINATOR = faculties this harness can lesion+exercise (neural-lesion / whether-disable with a driving probe).
    coverable = [p for p in per if p["kind"] in ("neural-lesion", "whether-disable")]
    exercised = [p for p in coverable if p["verdict"] in ("regressed", "pass")]  # not-exercised/missing/noisy excluded
    load_bearing = [p for p in exercised if p["load_bearing"] is True]
    report["counts"] = {
        "n_faculties_total": len(per),
        "n_coverable_env_lesion": len(coverable),
        "n_exercised": len(exercised),
        "n_load_bearing": len(load_bearing),
        "n_not_load_bearing": len([p for p in exercised if p["load_bearing"] is False]),
        "n_not_exercised": len([p for p in coverable if p["verdict"] == "not-exercised"]),
        "n_noisy": len([p for p in coverable if p["verdict"] in ("noisy", "noisy-null-control")]),
        "n_lesion_knob_missing": len([p for p in coverable if p["verdict"] == "lesion-knob-missing"]),
        # reported separately, NOT in the denominator:
        "n_in_process": len([p for p in per if p["kind"] == "in-process"]),
        "n_thin": len([p for p in per if p["kind"] == "thin"]),
        "n_mechanism_only": len([p for p in per if p["kind"] == "mechanism-only"]),
        "n_proposed": len([p for p in per if p["kind"] == "proposed"]),
    }
    n_ex = report["counts"]["n_exercised"]
    report["load_bearing_fraction"] = (report["counts"]["n_load_bearing"] / n_ex) if n_ex else None
    report["load_bearing_faculties"] = [p["faculty"] for p in load_bearing]
    report["not_load_bearing_faculties"] = [p["faculty"] for p in exercised if p["load_bearing"] is False]
    return report


# ── self-test: prove the instrument's LOGIC without building a brain (no memcap needed) ──────────────────────────
def selftest(out_path=None):
    """Verify the verdict inversion + change classification on synthetic responses (no brain build). Mirrors the
    battery's --skip-real synthetic demo: a real load-bearing lesion is DETECTED, a no-op is NOT, absence is handled.
    With out_path, ALSO write a citable static-verification artifact (the checks + episodic-driving remap wiring)."""
    row = _faculty_row("curiosity-followup")  # fields: curiosity.curious / curiosity.on
    intact = {"unknown": {"abstained": True, "curiosity": {"curious": True, "on": True}}}
    lesioned = {"unknown": {"abstained": True, "curiosity": {"curious": False, "on": True}}}  # crave silenced
    no_op = {"unknown": {"abstained": True, "curiosity": {"curious": True, "on": True}}}       # identical
    absent = {"unknown": {"abstained": True}}                                                   # field gone both arms

    d_change = compare(intact, lesioned, faculties=[row])["per_faculty"][0]
    d_noop = compare(intact, no_op, faculties=[row])["per_faculty"][0]
    d_absent = compare(absent, absent, faculties=[row])["per_faculty"][0]

    checks = {
        "load_bearing_detected (regressed on a real lesion)": d_change["verdict"] == "regressed",
        "change_kind value": _classify_diffs(d_change["diffs"]) == "value",
        # attribution wiring: a change absent from the null control is 100% attributable to the lesion; a change equally
        # present in the null control is 0% attributable (noise) -> the load-bearing gate must reject it.
        "attribution clean-null == 1.0": attributable_to("selftest-clean", 1, 0) == 1.0,
        "attribution dirty-null == 0.0": attributable_to("selftest-dirty", 1, 1) == 0.0,
        "attribution both-null UNDEFINED": attributable_to("selftest-none", 0, 0) is None,
        "not_load_bearing (pass on a no-op lesion)": d_noop["verdict"] == "pass",
        "not_exercised (fields absent both arms)": d_absent["verdict"] == "not-exercised",
        "structural change classified": _classify_diffs(
            [{"field": "x.y", "on": "v", "off": None}]) == "structural",
        "turn_group hold->held": turn_group("held") == ["hold", "held"],
        "turn_group dr chain": turn_group("dr_c") == ["dr_a", "dr_b", "dr_c"],
        "turn_group bc pair": turn_group("bc_b") == ["bc_a", "bc_b"],
        "turn_group single": turn_group("well") == ["well"],
        "null lesion flag unread in source": not _flag_resolves(NULL_LESION_FLAG),
        "a real lesion flag resolves in source": _flag_resolves("BRAIN_CURIOSITY_LESION"),
        # episodic-driving remap (LB_EPISODIC_DRIVE_PROBE): the store->recall pair exists and its group is store-first,
        # the forced-write flag is real, and remapping episodic to `epi_recall` still reads the same in_memory field.
        "episodic-drive turns exist": all(l in _TURN_BY_LABEL for l in (_EPISODIC_DRIVE_TURN, "epi_store")),
        "episodic-drive group is store->recall": turn_group(_EPISODIC_DRIVE_TURN) == ["epi_store", _EPISODIC_DRIVE_TURN],
        "episodic-drive forces the BTSP write": _flag_resolves("BRAIN_EPISODIC_STORE") and "1" in _EPISODIC_DRIVE_ENV.values(),
        "every FACULTY_LESIONS key is a real battery faculty":
            all(k in faculty_list() for k in FACULTY_LESIONS),
        "every battery faculty is mapped": all(k in FACULTY_LESIONS for k in faculty_list()),
        # SEED THREADING (research/seed-threading-lbf, 2026-09-20): seed=42 (the pre-existing hardcoded value)
        # keeps the ORIGINAL un-suffixed filenames -- byte-identical to every on-disk artifact from before --seed
        # existed; only a non-42 seed adds the _s<seed> tag, so a 6-seed sweep sharing one --out dir never collides.
        "seed_suffix default(42) is empty (byte-identical filenames)": _seed_suffix(42) == "",
        "seed_suffix non-default namespaces the filename": _seed_suffix(43) == "_s43",
        "seed_suffix accepts the full 6-seed roster": [_seed_suffix(s) for s in (42, 43, 44, 100, 101, 102)]
            == ["", "_s43", "_s44", "_s100", "_s101", "_s102"],
    }
    ok = all(checks.values())
    print("=== LOAD-BEARING INSTRUMENT SELF-TEST ===")
    for name, passed in checks.items():
        print("  [%s] %s" % ("PASS" if passed else "FAIL", name))
    # coverage summary of the lesion map
    from collections import Counter
    kinds = Counter(v["kind"] for v in FACULTY_LESIONS.values())
    print("  lesion-map coverage:", dict(kinds), "over", len(faculty_list()), "battery faculties")
    print("VERDICT:", "PASS" if ok else "FAIL")
    if out_path:
        # A citable STATIC-verification artifact (no brain build): the instrument-logic checks + the episodic-driving
        # remap wiring + the roster-unchanged facts. Written through the runner so provenance sidecars it.
        # NB: key is `selftest_result`, NOT `verdict`/`status`/`go` — a selftest is an instrument-logic check, not a
        # scientific GO/NO-GO verdict, so it deliberately does not trip the verdict-preconditions gate.
        art = {"runner": "research.runners.load_bearing_fraction", "kind": "selftest",
               "selftest_result": "PASS" if ok else "FAIL", "checks": checks,
               "n_probe_turns_default_roster": len(PROBE_TURNS),
               "episodic_drive_group": turn_group(_EPISODIC_DRIVE_TURN),
               "episodic_drive_env": _EPISODIC_DRIVE_ENV,
               "lesion_map_coverage": dict(kinds)}
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        json.dump(art, open(out_path, "w"), indent=2, default=str)
        print("  wrote", out_path)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Load-bearing fraction instrument (lesion-verified, reuses the regression battery).")
    ap.add_argument("--out", default="research/findings/raw/_load_bearing/load_bearing.json")
    ap.add_argument("--smoke", default=None, help="measure ONE faculty (its key) intact-vs-lesion, for the tiny memcap smoke")
    ap.add_argument("--only", default=None, help="comma-separated faculty keys to restrict to")
    ap.add_argument("--repeats", type=int, default=1, help="lesion-arm rebuilds for the anti-noise reproduce check (>=1)")
    ap.add_argument("--seed", type=int, default=42,
                    help="substrate seed for every arm this invocation builds (research/seed-threading-lbf, "
                         "2026-09-20): sets BRAIN_CHAT_SEED for the whole run + namespaces every per-arm output "
                         "filename (_seed_suffix) so a multi-seed sweep sharing one --out dir cannot collide or "
                         "false-skip under LB_RESUME_SKIP_EXISTING. Default 42 is BYTE-IDENTICAL to before this "
                         "flag existed (unsuffixed filenames, BRAIN_CHAT_SEED=42 behaves exactly like unset). The "
                         "mandated 6-seed validation is 42/43/44/100/101/102, one invocation per seed.")
    ap.add_argument("--selftest", action="store_true", help="verify the instrument logic without building a brain (no cap needed)")
    ap.add_argument("--selftest-out", default=None, help="also write the selftest checks to this JSON artifact (no brain build)")
    ap.add_argument("--map", action="store_true", help="print the per-faculty lesion map and exit (no brain build)")
    args = ap.parse_args()

    if args.selftest or args.selftest_out:
        return 0 if selftest(out_path=args.selftest_out) else 1
    if args.map:
        for k in faculty_list():
            s = FACULTY_LESIONS.get(k, {})
            print("%-42s %-14s %-38s %s" % (k, s.get("kind", "UNMAPPED"), s.get("flag") or "-",
                                            (("val=%s " % s["value"]) if s.get("value") else "")))
        return 0

    # SEED THREADING: set the process-wide substrate seed ONCE, before any arm build. _spawn_arm_raw (reused
    # verbatim from onebrain_regression_battery.py) passes `env=dict(os.environ)` to every arm subprocess, so this
    # single assignment is what makes EVERY arm (intact + lesion, every faculty) build at args.seed. Unconditional
    # even at the default (--seed unset -> 42): BRAIN_CHAT_SEED=42 reads identically to unset everywhere it is
    # consumed (webapp/server.py._brain_chat_seed + every per-organ workspace's own _DEFAULT_SEED), so this is a
    # byte-identical no-op for the shipped default.
    os.environ["BRAIN_CHAT_SEED"] = str(args.seed)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    if args.smoke:
        report = run(out_dir=out_dir, only=[args.smoke], repeats=max(2, args.repeats), seed=args.seed)
    else:
        only = args.only.split(",") if args.only else None
        report = run(out_dir=out_dir, only=only, repeats=args.repeats, seed=args.seed)

    json.dump(report, open(args.out, "w"), indent=2, default=str)
    print("\n===== LOAD-BEARING FRACTION =====")
    print("  seed=%d (BRAIN_CHAT_SEED=%s)" % (args.seed, os.environ.get("BRAIN_CHAT_SEED")))
    if "determinism" in report:
        d = report["determinism"]
        print("  determinism/null-control (intact vs intact-rebuild): deterministic=%s (checked=%s, dirty=%s)"
              % (d.get("deterministic"), d.get("n_null_controls_checked"), d.get("dirty_null_controls")))
    if "UNRELIABLE" in report:
        print("  ⛔ UNRELIABLE:", report["UNRELIABLE"])
    c = report["counts"]
    print("  faculties: total=%d coverable(env-lesion)=%d exercised=%d" % (
        c["n_faculties_total"], c["n_coverable_env_lesion"], c["n_exercised"]))
    print("  LOAD-BEARING=%d  not-load-bearing=%d  not-exercised=%d  noisy=%d  knob-missing=%d"
          % (c["n_load_bearing"], c["n_not_load_bearing"], c["n_not_exercised"], c["n_noisy"], c["n_lesion_knob_missing"]))
    print("  (reported separately: in-process=%d thin=%d mechanism-only=%d proposed=%d)"
          % (c["n_in_process"], c["n_thin"], c["n_mechanism_only"], c["n_proposed"]))
    frac = report["load_bearing_fraction"]
    print("  LOAD_BEARING_FRACTION = %s  (%d/%d exercised env-lesion faculties)"
          % (("%.3f" % frac) if frac is not None else "n/a", c["n_load_bearing"], c["n_exercised"]))
    if report.get("load_bearing_faculties"):
        print("  load-bearing:", report["load_bearing_faculties"])
    if report.get("not_load_bearing_faculties"):
        print("  NOT load-bearing (cosmetic on this probe):", report["not_load_bearing_faculties"])
    print("  wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
