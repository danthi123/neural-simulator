"""CLASS BC — EVERY ARM BELOW CHANCE, reported as a NEGATIVE. The verdict is UNDEFINED, not a NO-GO.

THE FAILURE. A run whose task/label wiring is broken produces arms that are all *worse than guessing*. There is
no verdict in such a run: the comparison between arms measures the instrument, not the hypothesis. Reading a
NO-GO off it fabricates a negative — and a fabricated negative is more expensive than a fabricated positive,
because it CLOSES a line of work rather than inviting scrutiny.

THE RECORDED INSTANCE (`research/FAILURE_LOG.md`, 2026-07-31): the first smoke of
`research/runners/_gap4_credit_on_expanded_forward_derisk.py` read raw 0.0362 / expanded 0.0652 / label-shuffle
0.0399 against a nominal chance of 0.200. The cause was not the substrate: the eval set was ~90% unscoreable by
construction (249 of 276 rows carried a reserved class that never appears in training), so the CEILING on that
split was 27/276 = 0.0978 — every arm was pinned under "chance" by the scoring set. That runner has a guard and
refused the verdict. **Nothing covered the class**, which is why the log row read
`NOT-GATEABLE yet: the guard exists in ONE runner ... widening the coverage recogniser to accept a single runner
would have made this pass while covering nothing.` This module is that coverage.

WHAT IT CHECKS. In every artifact JSON, for every dict that declares a chance/floor:
    chance c  <- an explicit `chance` / `chance_level` / `chance_baseline` / `chance_rate` / `chance_floor` key
                 (float, 0 < c < 1), or DERIVED as 1/n from an unambiguous class-count key
                 (`n_classes` / `k_classes` / `num_classes` / `n_categories` / `n_supers` / `n_labels`, 2..64);
    arms      <- float values in [0,1] in that same dict and in its immediate sub-dicts, minus a deny list of
                 quantities that are not scores-against-chance (losses, correlations, norms, counts, config);
    FIRE when there are >= 2 arms, at least one is NOT a null control, and EVERY arm is strictly below c.
The artifact clears itself by DECLARING the state — a true `below_chance` / `undefined` / `instrument_failed`
flag, or a verdict string containing UNDEFINED, anywhere in the document, suppresses the finding. That escape is
the desired behaviour, not a loophole: owning the UNDEFINED verdict is exactly what the class asks for.

CALIBRATION, MEASURED BEFORE SHIPPING — every figure below is reproduced by
`python -m tools.gates.below_chance` (`corpus_rate()`), which prints the table and the full hit list. A gate
that cries wolf gets disabled, which is worse than no gate.

  REACH FIRST, because it bounds everything else: only **763 of 7151** artifacts in `research/findings/raw/`
  (10.7%) declare a floor this gate can read. **89% of the corpus is out of reach and always will be.**
  The audit (`check(None)`) totals **34 files / 35 sites** across the two artifact directories. Staged mode
  reads every staged `.json`, not just those two directories; measured repo-wide over 7944 `.json` files that
  yields the same 34 — **zero hits anywhere else** — so the broader staged rule buys coverage of future
  artifact locations at no false-positive cost.

  * `research/findings/raw/` — **16 of 7151 artifacts (0.22%), 17 sites**; that is 16 of the 763 it can see
    (2.1%). Ten were opened individually and all ten are genuine; the other six are same-runner siblings of
    inspected files. **None of the 16 is dirty in git**, so this gate blocks nothing that is staged today.
    Examples:
      - `_npwall_spiking_s42.json`: chance 0.549, np 0.460, shuffle_dl 0.446, hidden_frozen 0.468 — and it
        records `"GO": false` AND `"shuffle_collapses": true`. The shuffle "collapsed" to a number the treatment
        could not beat, under chance. A NO-GO read off that is the class exactly.
      - `_ml_stacked_s42.json` `/per_seed[0]/cooc`: chance 0.25, acc 0.13, deranged 0.14 — the DERANGED CONTROL
        BEAT THE TREATMENT, both under chance, `"verdict": "NEGATIVE"`.
      - `_lge_gpu_seed42.json` `/per_seed/42/generalization`: chance 0.25, graded 0.2375, orthogonal 0.11875,
        permuted 0.24375; `"verdict": "NEGATIVE_no_structure"`, `g2_a1_generalizes: false`.
      - `_rc_inherit_spk_smoke.json` `/aggregate`: chance 0.25, heldout 0.125, permuted 0.125, deranged 0.0,
        lesion 0.0; `"verdict": "NEGATIVE"`, `"beats_chance": false`.
  * top-level `raw/` (a live artifact dump: `_reslm_*`, `_ngram_*`, `_dend*`/`_gen*` write there) — **18 of 234
    (7.7%)**, or 18 of the 121 it can see (14.9%). Genuine in kind, e.g. `_gen3_fw32.json` `/summary`:
    fixed_heldout_mean 0.0833 = learn_heldout_mean 0.0833 against chance 0.1667, `"verdict": "BOUNDARY"`.
    **DISCLOSED WEAKNESS, quantified: 13 of those 18 have only ONE genuine accuracy in the flagged scope** —
    `fixed_heldout` — and reach the required 2 arms only because a `coupling` sub-dict contributes `B_rest` /
    `B_apical`, which are burst-coupling coefficients, not scores. Those 13 were opened: all 13 have
    `learn_heldout: null` (the second arm never ran), an empty `summary`, and `fixed_heldout` 0.0-0.133 against
    a 0.1667 floor, so flagging them is right in substance — but "EVERY arm below chance" is a degenerate claim
    over one arm. I looked for a structural rule that separates them from the true positives and **did not find
    one**: the recorded gap#4 instance ALSO carries all of its arms inside a single sub-dict (`means`), so any
    rule that discounts a lone sub-dict discards the case this gate exists for. Disclosed, not silently tuned.
  * `webapp/` — 0 of 256 (0 declare a floor).
  * **BLAST RADIUS, stated rather than discovered later:** of the 34 hit files, 7 are modified and 11 untracked,
    all in `raw/`. A wholesale `git add raw/` would therefore produce up to 18 blocks, 13 of them the degenerate
    kind above. Each clears with one JSON field. That cost is accepted deliberately: `raw/` is written by live
    runners, and excluding it to keep the number pretty would be coverage theatre.
Three narrowing decisions were each measured, not guessed, and each has a regression control in `selftest`:
  (1) DENY BY TOKEN, NOT SUBSTRING. A substring deny list matched `n_` inside `vision_co[n_c]ept_cat_acc` and
      `ge[n_m]ean`, hiding the winning arm and manufacturing 2 false positives (`_genfrontier_capstone...`,
      `_l1_spiking_oja_smoke`). Token-split fixed both.
  (2) FLOAT-ONLY ARMS. JSON ints are counts. `funcint_perception_to_memory_probe.json` reads
      `recall_correct_per_seed {42: 4}` / `lesion_correct_per_seed {42: 0}` against chance 0.25 — items
      recalled, not a rate; and `_perception_v2it_smoke.json` counts spikes per region as `0`. Both were false
      positives until arms were restricted to floats.
  (3) NO DERIVATION FROM A BARE `k`/`K`. Measured 0-for-3: `K` meant a hypervector width (1024), a branch count
      (6) and a pool size. Restricted to explicit class-count names, derivation adds 0 hits and 0 false
      positives to the 16 — it is coverage for artifacts that record only a class count, not a source of noise.

WHAT THIS GATE CANNOT CATCH — the class is NOT closed by it:
  * **THE 89% OF ARTIFACTS THAT NEVER RECORD A FLOOR AT ALL.** 763 of 7151 declare one. If the floor lives in a
    findings `.md`, in the runner source, or only in the author's head, nothing here fires. This is the largest
    hole and no amount of tuning closes it; recording chance in the artifact does.
  * **A WRONG CHANCE VALUE — which was the actual defect in the recorded instance.** The gap#4 smoke declared
    0.200 when the split's ceiling was 0.0978. This gate detects the SYMPTOM (everything under the declared
    floor); it cannot verify that the declared floor is the right one. A majority-class floor
    (`max(1/k, majority-class rate)`, as that runner now uses) is strictly harder and still not checkable here.
  * **ARM IDENTIFICATION IS NAME-BLIND BY NECESSITY.** The recorded instance's arms are called `raw` and
    `expanded` — no accuracy token anywhere — so no allow-list of metric names can work, and a non-score that
    happens to live in [0,1] beside the chance (a coupling coefficient, a firing fraction) is counted as an arm.
    Bounded consequence: a spurious arm can only ever ADD to the arm count; it can never rescue a hit, because a
    single genuine arm at or above chance suppresses the finding outright.
  * Arms more than one nesting level from their chance; single-arm runs; runs where only null controls are
    reported; metrics whose names collide with the deny list (`*_rate`, `*_ratio`, `*_margin`); accuracies
    serialised as ints; chance expressed as a percentage (25.0, not 0.25); non-JSON artifacts; anything unstaged.
  * Artifacts over `_MAX_BYTES` (2 MB), lists past `_MAX_LIST` (60 entries) and nesting past `_MAX_DEPTH` (10)
    are skipped for the registry's time budget. Measured cost of the size cap: **1 of 7385 artifacts in scope**
    (`research/findings/raw/_corpus_pos_map.json`, 2.8 MB, a position map with no arms). Cost of the list cap:
    a 61st seed in a flat results array is unread.
  * It does not read the finding text. An artifact that is clean here can still be narrated as a NO-GO in prose.

HOW THE SELFTEST WAS VERIFIED NOT TO BE VACUOUS (failure class 3 is "the check that cannot fail", 9 incidents,
and a selftest that only proves good input passes IS that class). The gate was deliberately broken TEN ways —
made unfailable; deny-by-substring restored; integer arms admitted; bare-`k` derivation re-admitted; the
not-only-controls rule dropped; `check([])` allowed to fall through to a corpus scan; `<` relaxed to `<=`; the
read pre-filter desynchronised from the recognisers; the acknowledgement escape widened to any key containing
"below"; and the verdict escalation taken from any string anywhere — and `selftest()` was required to FAIL on
each. **It initially caught 6 of 8, then 9 of 10**, and each miss was a real defect in the check, not in the
mutation:
  · **bare-`k` derivation was masked by the read pre-filter, not tested by the recogniser** — the same
    producer/consumer seam the FAILURE_LOG records for the provenance door, appearing INSIDE one file. Closed
    by generating `_PREFILTER` from `_CHANCE_NAMES + _DERIVE_NAMES`;
  · the escalation control asserted on the string `"the build fails"`, which `\\bFAILED?\\b` does not match — an
    assertion that could not fail. It now asserts its own control string matches `_NEG_STR` first.
Both are covered by the unit-level assertions at the end of `selftest()`. A decision is only tested if breaking
it makes that function fail; run the mutation loop again after editing any rule here.
"""
from __future__ import annotations

import json
import os
import re
import tempfile

NAME = "below-chance"
CLASS_ID = "BC"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Directories the AUDIT walks. Staged mode does not use this list -- it reads every staged .json, because a
# runner that starts writing artifacts somewhere new must not silently fall out of coverage, and that costs
# nothing: measured repo-wide, 7944 .json files produce 34 hits and ALL 34 are inside these two directories.
_ARTIFACT_DIRS = ("research/findings/raw", "raw")
_MAX_BYTES = 2_000_000
_MAX_REPORT = 30
_MAX_DEPTH = 10
_MAX_LIST = 60

# ONE SOURCE OF TRUTH for the key names. The recognisers AND the read-time pre-filter are both generated from
# these tuples, because a producer and a consumer that keep their own copies of a format is the logged
# INTEGRATION-SEAM class (2026-07-31: the provenance door wrote `.prov.json` while its gate looked for
# `.cmd.json`). Here the seam is internal: a pre-filter that skipped a file the recogniser would have accepted
# would silently shrink this gate's reach, and a mutation test proved it also HID a broken recogniser.
_CHANCE_NAMES = ("chance", "chance_level", "chance_baseline", "chance_rate", "chance_acc",
                 "chance_accuracy", "chance_floor")
# Derivation ONLY from names that can only mean "size of the label space". A bare `k`/`K` is excluded: measured
# 0-for-3 on this corpus (hypervector width 1024, branch count 6, pool size).
_DERIVE_NAMES = ("n_class", "n_classes", "k_class", "k_classes", "num_class", "num_classes",
                 "n_categories", "n_category", "n_cat", "n_supers", "n_labels", "n_options", "n_alternatives")

_CHANCE_KEY = re.compile(r"^(%s)$" % "|".join(_CHANCE_NAMES), re.I)
_DERIVE_KEY = re.compile(r"^(%s)$" % "|".join(_DERIVE_NAMES), re.I)
# Sub-dicts that hold settings, not results. Descending into them turns `lr: 0.01` into an "arm".
_DENY_DICT = re.compile(
    r"^(config|cfg|args|argv|params|hyper|hyperparams|meta|metadata|bars|knobs|settings|env"
    r"|provenance|prov|timing)$", re.I)
# Sidecars written by the provenance door — they carry a command line, never arms.
_SIDECARS = (".cmd.json", ".provenance.json", ".prov.json")

# A quantity that is NOT a score compared against chance: lower-is-better, a correlation, a magnitude, a count,
# or a setting. Matched against WHOLE TOKENS of the key (see calibration note 1).
_DENY_TOKENS = frozenset("""
chance baseline floor majority null bar bars thresh threshold tol tolerance target criterion
loss ce mse mae err error nll ppl perplexity dist distance drift leak cost entropy kl
var variance std sem ci pval pvalue alpha beta decay lr rate hz
sparsity sparse repro reproducibility density ratio frac fraction prop proportion share
norm weight weights gain scale cos corr correlation pearson spearman sim similarity
align alignment overlap jaccard dice margin gap delta diff prob probability p0 p
elapsed time times second seconds ms us sec count n num idx index seed dim size len
epoch epochs iter iters step steps spikes
""".split())
# A null control is EXPECTED at or below chance. A scope reporting only controls is not evidence of this class.
_CONTROL_TOKENS = frozenset("""
perm permute permuted permutation shuffle shuffled shuf scramble scrambled scram
lesion lesioned ablate ablated ablation sham null nulls deranged derangement derange
random rand randset control ctl ctrl untrained notrain chance nocredit noteaching
""".split())
_ACK_TOKENS = frozenset(("undefined", "uninterpretable", "instrument", "broken", "invalid"))
_ACK_STR = re.compile(r"UNDEFINED|INSTRUMENT FAIL|NOT[- ]INTERPRETABLE|UNINTERPRETABLE", re.I)
# Only strings under a VERDICT-shaped key escalate the message. A bare `\bFAILS?\b` anywhere in the document
# would label a passing note as a verdict, and the escalation is the sentence that tells the author their
# conclusion is fabricated -- it must not be attached to the wrong sentence.
_VERDICT_KEY = frozenset(("verdict", "verdicts", "go", "result", "results", "outcome", "conclusion",
                          "interpretation", "why", "summary", "overall"))
_NEG_STR = re.compile(r"NO[- ]?GO\b|NEGATIVE|BOUNDARY|\bKILL(?:ED)?\b|REFUTED|\bFAILED?\b", re.I)
# Cheap pre-filter so the audit does not JSON-parse the ~89% of artifacts that declare no floor at all.
# GENERATED from the two name tuples above so it can never disagree with the recognisers.
_PREFILTER = re.compile(r'"(%s)"' % "|".join(_CHANCE_NAMES + _DERIVE_NAMES), re.I)


def _tokens(key):
    return [t for t in re.split(r"[^A-Za-z0-9]+", str(key).lower()) if t]


def _denied(key):
    return any(t in _DENY_TOKENS for t in _tokens(key))


def is_control(key):
    """True if the arm name marks a null control (permuted / shuffled / lesion / deranged / untrained)."""
    return any(t in _CONTROL_TOKENS for t in _tokens(key))


def _is_arm(key, value):
    # FLOAT-ONLY: a JSON int is a count (spikes fired, items recalled), not a rate. See calibration note 2.
    return (isinstance(value, float) and value == value and 0.0 <= value <= 1.0
            and not _denied(key) and not _CHANCE_KEY.match(str(key)))


def arms_of(scope):
    """Arm name -> value, for the dict `scope` and its immediate result sub-dicts.

    One level of descent is load-bearing, not incidental: the recorded instance keeps its arms in a `means`
    sibling of the chance, and `_lge_gpu_seed42` keeps `graded.accuracy` / `permuted.accuracy` one level under
    the `generalization` dict that declares the floor.
    """
    out = {}
    for k, v in scope.items():
        if _is_arm(k, v):
            out[k] = float(v)
    for k, v in scope.items():
        if isinstance(v, dict) and not _DENY_DICT.match(str(k)):
            for k2, v2 in v.items():
                if _is_arm(k2, v2):
                    out["%s.%s" % (k, k2)] = float(v2)
    return out


def _chance_of(scope):
    """(value, how) for the floor this dict declares, or (None, None)."""
    for k, v in scope.items():
        if _CHANCE_KEY.match(str(k)) and isinstance(v, float) and 0.0 < v < 1.0:
            return float(v), str(k)
    for k, v in scope.items():
        if (_DERIVE_KEY.match(str(k)) and isinstance(v, int) and not isinstance(v, bool) and 2 <= v <= 64):
            return 1.0 / v, "1/%s(=%d)" % (k, v)
    return None, None


def _doc_flags(node, flags, depth=0):
    """Document-wide: has the artifact ALREADY declared the state, and does it record a negative verdict?"""
    if depth > _MAX_DEPTH:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            tk = _tokens(k)
            if v is True and (set(tk) & _ACK_TOKENS or ("below" in tk and "chance" in tk)):
                flags["ack"] = True
            elif isinstance(v, str) and _ACK_STR.search(v):
                flags["ack"] = True
            if isinstance(v, str) and set(tk) & _VERDICT_KEY and _NEG_STR.search(v):
                flags["neg"] = flags["neg"] or str(k)
            elif v is False and tk and tk[0] in ("go", "pass", "passed", "success"):
                flags["neg"] = flags["neg"] or str(k)
            _doc_flags(v, flags, depth + 1)
    elif isinstance(node, list):
        for v in node[:_MAX_LIST]:
            _doc_flags(v, flags, depth + 1)


def scan_doc(doc):
    """[(scope_path, chance, how, sorted_arms)] for every all-arms-below-chance scope in one parsed artifact."""
    flags = {"ack": False, "neg": ""}
    _doc_flags(doc, flags)
    if flags["ack"]:
        return [], flags
    found = []

    def walk(node, where, depth):
        if depth > _MAX_DEPTH:
            return
        if isinstance(node, dict):
            chance, how = _chance_of(node)
            if chance is not None:
                arms = arms_of(node)
                if (len(arms) >= 2 and any(not is_control(k) for k in arms)
                        and max(arms.values()) < chance):
                    found.append((where or "/", chance, how,
                                  sorted(arms.items(), key=lambda t: -t[1])))
            for k, v in node.items():
                if not _DENY_DICT.match(str(k)):
                    walk(v, "%s/%s" % (where, k), depth + 1)
        elif isinstance(node, list):
            for i, v in enumerate(node[:_MAX_LIST]):
                walk(v, "%s[%d]" % (where, i), depth + 1)

    walk(doc, "", 0)
    return found, flags


def _is_artifact_path(path):
    p = str(path).replace("\\", "/")
    return p.endswith(".json") and not any(p.endswith(s) for s in _SIDECARS)


def _audit_corpus():
    out = []
    for d in _ARTIFACT_DIRS:
        for root, _dirs, files in os.walk(os.path.join(_ROOT, d)):
            for f in sorted(files):
                if f.endswith(".json") and not any(f.endswith(s) for s in _SIDECARS):
                    out.append(os.path.join(root, f))
    return out


def _load(full):
    try:
        if os.path.getsize(full) > _MAX_BYTES:
            return None
        text = open(full, encoding="utf-8", errors="replace").read()
    except (OSError, ValueError):
        return None
    if not _PREFILTER.search(text):
        return None                      # declares no floor: out of this gate's reach, and it says so above
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None                      # a .json that is not JSON is some other gate's problem


def _problems_for(files):
    problems, total = [], 0
    for p in files:
        full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
        doc = _load(full)
        if doc is None:
            continue
        sites, flags = scan_doc(doc)
        if not sites:
            continue
        total += len(sites)
        rel = os.path.relpath(full, _ROOT)
        where, chance, how, arms = sites[0]
        shown = ", ".join("%s=%.4g" % (k, v) for k, v in arms[:5])
        if len(arms) > 5:
            shown += ", ... (%d arms)" % len(arms)
        extra = "" if len(sites) == 1 else " (+%d more scope(s) in this file)" % (len(sites) - 1)
        verdict = ("; the artifact records a NEGATIVE verdict at `%s` — that verdict is FABRICATED from an "
                   "instrument failure" % flags["neg"]) if flags["neg"] else ""
        problems.append(
            "%s @%s: EVERY arm is below the declared chance %.4g (%s) — %s%s%s. Verdict is UNDEFINED, not a "
            "NO-GO: nothing here measures the hypothesis. Fix the task/label wiring or the scored split, or "
            "declare it (`\"below_chance\": true` / an UNDEFINED verdict) so the artifact owns its own status."
            % (rel, where, chance, how, shown, extra, verdict))
    return problems, total


def check(paths):
    """paths=None -> full audit of the artifact corpus. [] -> nothing of my kind staged, pass. Else staged."""
    # An EMPTY list means "staged mode, nothing of my kind staged" -> nothing to check. Only paths=None means
    # "standalone/audit run". Without this split a gate's corpus fallback silently undoes the hook's scoping,
    # which is itself a logged failure (research/FAILURE_LOG.md, 2026-07-31).
    if paths is not None and len(paths) == 0:
        return []
    files = [p for p in paths if _is_artifact_path(p)] if paths else _audit_corpus()
    problems, total = _problems_for(files)
    if len(problems) > _MAX_REPORT:
        # Report the TRUE total, never let the cap masquerade as a count -- the mistake the lever-efficacy
        # audit had to correct after "40 hits" turned out to be _MAX_REPORT.
        head = problems[:_MAX_REPORT]
        head.append("... and %d more below-chance file(s) [%d sites total]; showing %d of %d. Run "
                    "`python -m tools.gates.below_chance` for the full list."
                    % (len(problems) - _MAX_REPORT, total, _MAX_REPORT, len(problems)))
        return head
    return problems


def corpus_rate():
    """Measured calibration, reproducible: {dir: (n_files_parsed, n_files_hit, n_sites)}."""
    out = {}
    for d in ("research/findings/raw", "raw", "webapp"):
        base = os.path.join(_ROOT, d)
        if not os.path.isdir(base):
            continue
        n = hit = sites = 0
        for root, _dirs, files in os.walk(base):
            for f in sorted(files):
                if not f.endswith(".json") or any(f.endswith(s) for s in _SIDECARS):
                    continue
                n += 1
                doc = _load(os.path.join(root, f))
                if doc is None:
                    continue
                s, _flags = scan_doc(doc)
                if s:
                    hit += 1
                    sites += len(s)
        out[d] = (n, hit, sites)
    return out


def selftest():
    """FAILING DIRECTION FIRST: the gate must CATCH the recorded instance and two live-corpus shapes.

    Every MUST-NOT-FIRE case below is a CALIBRATION CONTROL pinning one narrowing decision, and three of them
    are REGRESSION controls copied from real artifacts that a looser prototype flagged. Widening a rule fails
    here instead of on the owner's next commit.
    """
    problems = []

    must_catch = [
        # THE RECORDED INSTANCE: _gap4_credit_on_expanded_forward_derisk first smoke, chance 0.200,
        # raw 0.0362 / expanded 0.0652 / shuffle 0.0399 (module docstring of that runner).
        ("the recorded gap#4 smoke (chance 0.200, arms 0.033-0.065)",
         {"chance": 0.2, "go": False,
          "means": {"raw": 0.0362, "expanded": 0.0652, "expanded_label_shuffle": 0.0399}}),
        # LIVE CORPUS: research/findings/raw/_npwall_spiking_s42.json, verbatim.
        ("the live _npwall_spiking_s42 shape (chance 0.549, np 0.460, GO false)",
         {"results": [{"seed": 42, "chance": 0.549, "np": 0.46, "shuffle_dl": 0.446,
                       "hidden_frozen": 0.468, "np_beats_chance": False, "GO": False}]}),
        # LIVE CORPUS: research/findings/raw/_ml_stacked_s42.json -- the control BEAT the treatment, under chance.
        ("the live _ml_stacked_s42 shape (deranged 0.14 > acc 0.13, chance 0.25)",
         {"verdict": "NEGATIVE", "per_seed": [{"cooc": {"acc": 0.13, "deranged": 0.14, "chance": 0.25}}]}),
        # DERIVED floor: an artifact that records only the label-space size.
        ("a derived floor from n_classes=5",
         {"n_classes": 5, "arms": {"treated": 0.11, "control": 0.09}}),
        # The acknowledgement escape must be an EXACT declaration, not any key containing "below".
        ("an unrelated 'below' flag must not silence the gate",
         {"chance": 0.2, "delta_below_bar": True,
          "means": {"raw": 0.0362, "expanded": 0.0652, "expanded_label_shuffle": 0.0399}}),
    ]

    must_pass = [
        # THE PRIMARY SUPPRESSOR: one arm at or above chance. _rc_spk_fix_s102.json, verbatim.
        ("a live arm above chance (_rc_spk_fix_s102)",
         {"aggregate": {"heldout": 0.375, "deranged": 0.0625, "permuted_feat": 0.125,
                        "lesion": 0.0, "chance": 0.125}}),
        # REGRESSION (substring deny list): `n_` matched inside `visio[n_c]oncept...`, hiding the 0.75 arm.
        ("_genfrontier_capstone: the winning arm's name contains 'n_'",
         {"chance": 0.25, "vision_concept_cat_acc": 0.75, "flat_concept_cat_acc": 0.16666666666666666,
          "permuted_concept_cat_acc": 0.16666666666666666}),
        # REGRESSION (substring deny list): `n_` matched inside `ge[n_m]ean`, hiding the 0.854 arm.
        ("_l1_spiking_oja_smoke: the winning arm is `gen_mean`",
         {"best": {"gen_mean": 0.8541666666666666, "sat_mean": 0.0,
                   "perm_mean": 0.0001361457550779673, "chance": 0.125}}),
        # REGRESSION (float-only arms): funcint_perception_to_memory_probe -- ints are ITEMS RECALLED.
        ("integer counts beside a chance (funcint probe)",
         {"chance": 0.25, "recall_correct_per_seed": {"42": 4, "43": 4},
          "lesion_correct_per_seed": {"42": 0, "43": 0}}),
        # REGRESSION (no bare-k derivation): K meant a branch count, the "arms" are activity means.
        ("a bare K that is not a class count",
         {"K": 6, "oracle_mean": 0.0, "random_mean": 0.0}),
        ("only null controls below chance",
         {"chance": 0.25, "permuted_acc": 0.11, "shuffled_acc": 0.09, "lesion_acc": 0.0}),
        ("a single arm below chance (no comparison exists)",
         {"chance": 0.25, "heldout": 0.11}),
        ("exactly at chance is not below it",
         {"chance": 0.25, "treated": 0.25, "control": 0.2}),
        ("settings beside a chance are not arms",
         {"chance": 0.25, "config": {"lr": 0.01, "p0": 0.3, "dropout": 0.1},
          "heldout": 0.9, "train": 0.95}),
        ("loss / correlation family are not scores against chance",
         {"chance": 0.25, "val_loss": 0.02, "pearson_mean": 0.05, "weight_norm": 0.4}),
        ("an artifact that already declares the state",
         {"chance": 0.2, "below_chance": True, "go": False,
          "means": {"raw": 0.0362, "expanded": 0.0652, "expanded_label_shuffle": 0.0399}}),
        ("an artifact whose verdict already says UNDEFINED",
         {"chance": 0.2, "verdict": "UNDEFINED — every arm below chance",
          "means": {"raw": 0.0362, "expanded": 0.0652, "expanded_label_shuffle": 0.0399}}),
    ]

    with tempfile.TemporaryDirectory() as d:
        def w(name, obj):
            p = os.path.join(d, name)
            with open(p, "w") as fh:
                json.dump(obj, fh)
            return p

        for i, (what, obj) in enumerate(must_catch):
            if not check([w("catch%d.json" % i, obj)]):
                problems.append("MISSED %s — the gate cannot fail on the case it exists for" % what)
        for i, (what, obj) in enumerate(must_pass):
            hits = check([w("pass%d.json" % i, obj)])
            if hits:
                problems.append("FALSE POSITIVE on %s — the gate would cry wolf: %s" % (what, hits[0][:120]))

        # Contract: [] means nothing of my kind is staged, and must NOT trigger a corpus scan.
        if check([]) != []:
            problems.append("check([]) scanned the corpus — an empty staged list must pass")
        # And a staged non-artifact must not drag the corpus in either.
        if check([w("notes.md", {})] and [os.path.join(d, "notes.md")]) != []:
            problems.append("check() on a non-JSON path returned problems")

    # UNIT-LEVEL assertions on the rules that carry the calibration. These exist because a MUTATION TEST
    # (breaking the gate seven ways and requiring this function to fail each time) found two decisions that
    # the file-level cases did NOT actually test: bare-`k` derivation was masked by the read pre-filter rather
    # than by the recogniser, so re-admitting `k` slipped through. A decision only counts as tested if
    # breaking it makes THIS function fail.
    if _denied("vision_concept_cat_acc") or _denied("gen_mean"):
        problems.append("the deny list is matching substrings again — it must split on tokens")
    if not _denied("majority_class_rate") or not _denied("val_loss"):
        problems.append("the deny list stopped excluding a floor / a loss")
    if _is_arm("acc", 0) or not _is_arm("acc", 0.0):
        problems.append("arm typing broke: ints are counts, floats are rates")
    if not is_control("expanded_label_shuffle") or is_control("expanded"):
        problems.append("control detection broke")
    if _chance_of({"K": 6})[0] is not None or _chance_of({"k": 3})[0] is not None:
        problems.append("a bare k/K is being read as a class count — measured 0-for-3 on this corpus "
                        "(hypervector width 1024, branch count 6, pool size)")
    if _chance_of({"n_classes": 5})[0] != 0.2 or _chance_of({"chance": 0.2})[0] != 0.2:
        problems.append("floor recognition broke for an explicit chance or a class count")
    if _chance_of({"n_classes": 500})[0] is not None:
        problems.append("an out-of-range class count is being used as a floor")
    for _n in _CHANCE_NAMES + _DERIVE_NAMES:
        if not _PREFILTER.search('{"%s": 1}' % _n):
            problems.append("the read pre-filter skips %r, which the recogniser accepts — a file the gate "
                            "should judge is never parsed" % _n)
    if _PREFILTER.search('{"k": 6, "K": 6}'):
        problems.append("the pre-filter admits a bare k/K that the recogniser rejects")

    def _flags(doc):
        f = {"ack": False, "neg": ""}
        _doc_flags(doc, f)
        return f
    if not _flags({"below_chance": True})["ack"] or not _flags({"verdict": "UNDEFINED"})["ack"]:
        problems.append("the acknowledgement escape stopped working — an honest runner would be flagged anyway")
    if _flags({"delta_below_bar": True})["ack"] or _flags({"np_beats_chance": False})["ack"]:
        problems.append("the acknowledgement escape is too broad — an unrelated flag silences the gate")
    if not _flags({"verdict": "NEGATIVE"})["neg"] or not _flags({"GO": False})["neg"]:
        problems.append("a recorded negative verdict is no longer detected for the escalation sentence")
    # The string MUST be one _NEG_STR actually matches, or this assertion tests nothing -- it passed vacuously
    # once because "fails" does not match `\bFAILED?\b`. Guard that directly.
    _prose = "an earlier NO-GO on this task was retracted"
    if not _NEG_STR.search(_prose):
        problems.append("the escalation control string no longer matches _NEG_STR — the next check is vacuous")
    if _flags({"note": _prose})["neg"]:
        problems.append("prose outside a verdict key is being reported as a fabricated verdict")
    return problems


if __name__ == "__main__":
    # The audit prints the FULL list, not check()'s capped view -- a cap that reads as a count is the mistake
    # the lever-efficacy audit had to correct ("40 hits" was _MAX_REPORT).
    _all, _n = _problems_for(_audit_corpus())
    for _line in _all:
        print(_line)
    print("-" * 100)
    print("AUDIT: %d file(s), %d site(s)" % (len(_all), _n))
    for _d, (_n, _h, _s) in corpus_rate().items():
        print("%-26s %5d artifacts parsed  %3d file(s) hit (%.2f%%)  %3d site(s)"
              % (_d, _n, _h, 100.0 * _h / max(1, _n), _s))
    _st = selftest()
    print("selftest: %s" % ("PASS" if not _st else "FAIL -> %s" % _st))
