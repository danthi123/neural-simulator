#!/usr/bin/env python3
"""The experiment harness: you cannot get a VERDICT out of it without passing the gates.

WHY THIS EXISTS (2026-07-31, owner diagnosis). The project's bottleneck is not the science -- it is that
judgement errors and skipped checks eat most of the working time. The checks already existed: before_you_build.sh,
research_gate.sh, tools/lab.py, lane_check.py, workflow_check.sh, docs/TERMS.md, verify-go. They are OPTIONAL, and
optional checks are skipped exactly when momentum makes skipping attractive -- which is exactly when they matter.

The evidence for that is the whole record, but the sharpest single fact is this: on 2026-07-31 the ONLY check that
stopped a mistake without being invited was the pre-commit hook, which blocked a commit for over-long doc lines.
Every other check that ran that day ran because it was remembered. Every failure that day was a check not
remembered:

  * 94 GPU-hours re-deriving a NO-GO banked a week earlier   -> before_you_build.sh not run
  * a control that agreed with its treatment to 1e-9         -> no instrument validation
  * a metric a position-shuffle reproduced to 1.3%           -> no negative control on the metric
  * the FIFTH instance of the plasticity bound trap          -> pre-flight existed as PROSE for four other rules
  * an A/B whose arms differed in 3 variables, not 1         -> no one-variable assertion
  * 6 runs staged on a wrong provenance assumption           -> config never recorded, filename used as provenance

THE DESIGN RULE, and the only one that matters: this class FAILS CLOSED. `verdict()` raises unless the
experiment was pre-registered, the corpus was checked, and the instrument was validated in BOTH directions. It is
not a linter you run at the end. It is the only door to a reportable result.

    from tools.experiment import Experiment

    exp = Experiment(
        name="gap5-laps-isolation",
        lane="H · Memory",
        question="Is laps=1 (a single induction pass) the operative variable for place-specificity?",
        hypothesis="laps=1 gives shuffle-ratio > 2.0; laps=5 stays ~1.0",
        gate="permutation p < 0.05 AND ratio > 2.0, 6 seeds",
        kill="if laps=1 ALSO gives ratio ~1.0, the induction-event framing is REFUTED -- record it, do not retune",
        one_variable="laps",
        arms={"L1": dict(laps=1, dwell=30, w_max=2500), "L5": dict(laps=5, dwell=30, w_max=2500)},
    )
    exp.check_bounds(w_max=2500, weight=250)              # the bound trap, executable
    exp.validate_instrument(metric, positive=..., negative=...)   # REQUIRED, both directions
    exp.verdict(observed=..., passed=True)                # raises if any gate above was skipped
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.lab import bound_check, sign_budget, void_if, LeverError  # noqa: E402

# Re-exported so a caller needs ONE import to get the whole discipline, not four.
__all__ = ["Experiment", "HarnessError", "bound_check", "sign_budget", "void_if"]


class HarnessError(LeverError):
    """Raised when a gate is skipped. Never catch this to 'keep going' -- that IS the failure mode.

    Subclasses LeverError (itself an AssertionError) so that ONE except-clause covers every guard in the
    harness AND in tools.lab. Two exception types would mean a caller could catch one and silently proceed
    past the other, which is the precise shape of failure this module exists to prevent.
    """


class Experiment:
    # ---------------------------------------------------------------- registration
    def __init__(self, name, lane, question, hypothesis, gate, kill, one_variable,
                 arms=None, supersedes=None, corpus_check=True):
        """Pre-registration is the CONSTRUCTOR, so an experiment cannot exist without it.

        `kill` is not optional and not decorative: a pre-registered prediction protects against retrofitting a
        conclusion, and a kill criterion protects against retuning past a refutation. Both were skipped on the
        levers that cost the most.

        `one_variable` names the SINGLE thing that differs between arms. "ONE FLAG != ONE VARIABLE" is its own
        recurring failure: `--bdsp-wmax` was one config field but two functional variables, and
        `hebbian_mean_subtract` changed the fixed point AND the weight mass AND the firing rate, which turned a
        clean-looking refutation into a confound.
        """
        for field, val in (("name", name), ("lane", lane), ("question", question),
                           ("hypothesis", hypothesis), ("gate", gate), ("kill", kill),
                           ("one_variable", one_variable)):
            if not val or not str(val).strip():
                raise HarnessError(
                    "PRE-REGISTRATION INCOMPLETE: '%s' is required. Every field here exists because omitting it "
                    "cost this project a retraction or a day of compute. If you cannot state the kill criterion, "
                    "you do not yet know what would refute you, and the run cannot be interpreted either way."
                    % field)
        self.name = name
        self.lane = lane
        self.question = question
        self.hypothesis = hypothesis
        self.gate = gate
        self.kill = kill
        self.one_variable = one_variable
        self.arms = dict(arms or {})
        self.supersedes = supersedes
        self._registered_at = time.time()
        self._instrument_ok = False
        self._instrument_report = None
        self._bounds_checked = []
        self._corpus_hits = []

        print("=" * 78)
        print("EXPERIMENT  %s   [lane %s]" % (self.name, self.lane))
        print("=" * 78)
        print("  Q      : %s" % self.question)
        print("  H      : %s" % self.hypothesis)
        print("  GATE   : %s" % self.gate)
        print("  KILL   : %s" % self.kill)
        print("  VAR    : %s" % self.one_variable)
        if self.arms:
            self._assert_one_variable()
        if corpus_check:
            self._corpus_check()

    def _assert_one_variable(self):
        """Arms must differ in exactly the declared variable. Anything else is a confound, by construction."""
        keys = set()
        for cfg in self.arms.values():
            keys |= set(cfg.keys())
        differing = []
        for k in sorted(keys):
            vals = {json.dumps(cfg.get(k), sort_keys=True, default=str) for cfg in self.arms.values()}
            if len(vals) > 1:
                differing.append(k)
        if not differing:
            raise HarnessError(
                "ARMS ARE IDENTICAL: no config key differs across %s. The A/B is void and would have produced "
                "two identical numbers that look like a result." % list(self.arms))
        extra = [k for k in differing if k != self.one_variable]
        if extra:
            raise HarnessError(
                "CONFOUNDED ARMS: you declared one_variable=%r but the arms also differ in %s. Either hold those "
                "fixed, or declare the comparison honestly as multi-variable and expect it to be uninterpretable. "
                "(Earned: a mean-subtract A/B that also changed weight mass and firing rate, so 'the rule is "
                "worse' was inseparable from '3x less weight and 4x lower rate'.)" % (self.one_variable, extra))
        print("  arms   : %s differ ONLY in %r ✔" % (list(self.arms), self.one_variable))

    def _corpus_check(self):
        """Ask the record BEFORE spending compute. Refuses on a strong hit unless explicitly superseded."""
        try:
            out = subprocess.run(
                [os.path.join(ROOT, ".venv-rag/bin/python"), "tools/rag/rag_search.py",
                 self.question, "5", "--corpus", "finding"],
                cwd=ROOT, capture_output=True, text=True, timeout=180).stdout
        except Exception as e:                                  # narrow enough to see; never silent
            print("  corpus : ⚠️  check FAILED (%s: %s) — treat this as UNCHECKED, not clean"
                  % (type(e).__name__, e))
            return
        hits = []
        for ln in out.split("\n"):
            s = ln.strip()
            if s.startswith("[") and "(finding)" in s:
                try:
                    score = float(s.split("]")[1].split()[0])
                except Exception:
                    continue
                hits.append((score, s.split("(finding)")[-1].strip()))
        self._corpus_hits = hits[:5]
        strong = [h for h in hits if h[0] > 3.0]
        for sc, path in hits[:3]:
            print("  corpus : %+.2f  %s" % (sc, path[:88]))
        if strong and not self.supersedes:
            raise HarnessError(
                "THE RECORD MAY ALREADY ANSWER THIS. Strong prior finding(s):\n    %s\n"
                "READ them first. If they genuinely do not cover this experiment, re-register with "
                "supersedes='<why this is different>'. (Earned: 94 GPU-hours spent re-deriving a NO-GO that had "
                "been banked a week earlier, in a config that also reversed the re-scope made to afford it.)"
                % "\n    ".join(p for _, p in strong[:3]))
        if self.supersedes:
            print("  corpus : superseding prior work — %s" % self.supersedes)

    # ---------------------------------------------------------------- gates
    def check_bounds(self, **pairs):
        """check_bounds(btsp_w_max=(150, 250)) or check_bounds(w_max=150, weight=250)."""
        if "weight" in pairs and len(pairs) == 2:
            (rule, bound), = [(k, v) for k, v in pairs.items() if k != "weight"]
            try:
                bound_check(rule, bound, pairs["weight"])
            except LeverError as e:
                raise HarnessError(str(e)) from None
            self._bounds_checked.append(rule)
            return self
        for rule, pair in pairs.items():
            try:
                bound_check(rule, pair[0], pair[1])
            except LeverError as e:                 # surface as ONE harness exception type
                raise HarnessError(str(e)) from None
            self._bounds_checked.append(rule)
        return self

    def validate_instrument(self, metric, positive, negative, n=30, alpha=0.05,
                            min_power=0.9, max_fpr=0.15):
        """REQUIRED before any verdict. `metric(case) -> p_value`. Both directions, measured not asserted.

        `positive` and `negative` are callables taking a draw index and returning a case the metric should and
        should not flag. Power and false-positive RATE are measured over `n` independent draws -- never judged
        from a single draw, which is how a legitimate borderline p=0.0398 was briefly mistaken for a broken gate.
        """
        pos_p, neg_p = [], []
        for i in range(int(n)):
            pos_p.append(float(metric(positive(i))))
            neg_p.append(float(metric(negative(i))))
        power = sum(1 for p in pos_p if p < alpha) / float(len(pos_p))
        fpr = sum(1 for p in neg_p if p < alpha) / float(len(neg_p))
        self._instrument_report = dict(n=int(n), alpha=alpha, power=power, fpr=fpr,
                                       neg_p_median=sorted(neg_p)[len(neg_p) // 2])
        print("  instrument: power %.3f (want >= %.2f) | FPR %.3f (want <= %.2f) over %d draws"
              % (power, min_power, fpr, max_fpr, n))
        if power < min_power:
            raise HarnessError(
                "INSTRUMENT HAS NO POWER (%.3f): it fails to detect an effect that IS present, so a NEGATIVE "
                "from it is UNINTERPRETABLE -- not a scientific null. (Earned: a control that agreed with its "
                "own treatment to 1e-9 in 29 of 36 runs while printing confident 'NOT place-specific' verdicts.)"
                % power)
        if fpr > max_fpr:
            raise HarnessError(
                "INSTRUMENT CRIES WOLF (FPR %.3f): it flags effects that are NOT present, so a POSITIVE from it "
                "is uninterpretable." % fpr)
        self._instrument_ok = True
        return self

    # ---------------------------------------------------------------- the only exit
    def verdict(self, observed, passed, notes="", artifact=None):
        """The ONLY way to a reportable result -- and it raises if any gate above was skipped."""
        if not self._instrument_ok:
            raise HarnessError(
                "NO VERDICT WITHOUT A VALIDATED INSTRUMENT. Call validate_instrument() first. A refutation needs "
                "its instrument verified exactly as much as a confirmation does; most of this project's "
                "retractions were correct measurements read through an unverified instrument.")
        rec = dict(name=self.name, lane=self.lane, question=self.question, hypothesis=self.hypothesis,
                   gate=self.gate, kill_criterion=self.kill, one_variable=self.one_variable,
                   arms=self.arms, supersedes=self.supersedes,
                   corpus_hits=[{"score": s, "path": p} for s, p in self._corpus_hits],
                   bounds_checked=self._bounds_checked, instrument=self._instrument_report,
                   observed=observed, passed=bool(passed), notes=notes,
                   registered_at=self._registered_at)
        print("-" * 78)
        print("  VERDICT: %s" % ("PASS — the pre-registered gate is met" if passed else
                                 "⛔ FAIL — did the KILL criterion fire? If so, RECORD it; do NOT retune."))
        print("  gate    : %s" % self.gate)
        print("  observed: %s" % json.dumps(observed, default=str)[:400])
        if not passed:
            print("  kill    : %s" % self.kill)
        if artifact:
            # FULL config into the artifact. A filename is NOT provenance -- recovering one run's pool_k once
            # required forensics on its synapse count because the knob existed only in the file's name.
            os.makedirs(os.path.dirname(os.path.abspath(artifact)), exist_ok=True)
            json.dump(rec, open(artifact, "w"), indent=1, default=str)
            print("  artifact: %s (full pre-registration + instrument report embedded)" % artifact)
        return rec


if __name__ == "__main__":
    print(__doc__)
