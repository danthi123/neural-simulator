"""research/runners/c3_topic_stop_score.py: the reader of the C3 topic-stop dev battery (packet
research/coordination/c3-topic-stop-PACKET.md section 7, D1-D8) and of its gate-seed battery
(research/findings/2026-10-28-c3-topic-stop-gate-PREREGISTRATION.md, G1-G10 and DEC).

The dev tests read the 192 committed dev shards (research/findings/raw/_c3_topic_stop/dev, revision 79c6f5128) and pin the
dev verdict the battery was read at (D1-D8 PASS on 11/11 counted seeds, the attribution 29 vs 0) and the informational
numbers the prereg's caveats quote. The gate tests use the bar-sanity spec's synthetic six-gate-seed cell sets (the dev
shards relabelled): the failing direction first (a broken brain FAILs, a cell that cannot be read is UNDEFINED, never
PASS), then the perfect brain PASSes. No brain is built.
"""
import importlib.util
import os
import re
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import c3_topic_stop_score as S  # noqa: E402
from research.runners import stage2_exit_gate as G  # noqa: E402

DEV_ROOT = os.path.join(REPO, S.DEV_DIR)
HAVE_DEV = all(os.path.exists(S.cell_path(DEV_ROOT, s, t)) for s in S.DEV_SEEDS for t in S.TAGS)
SPEC = os.path.join(REPO, "docs/drafts/c3-topic-stop-gate/bar_sanity_spec_c3ts_gate.py")
LINES = os.path.join(REPO, "research/coordination/c3-topic-stop-gate-lines.txt")
needs_dev = pytest.mark.skipif(not HAVE_DEV, reason="the dev shards are not in this checkout")


@pytest.fixture(scope="module")
def dev():
    return S.score(S.load_cells(DEV_ROOT, S.DEV_SEEDS), S.DEV_SEEDS, "dev", S.DEV_PIN, counted=S.DEV_COUNTED,
                   needs={"5": 9})


@pytest.fixture(scope="module")
def spec():
    s = importlib.util.spec_from_file_location("c3ts_gate_spec", SPEC)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def _gate(cells, spec):
    return S.score(cells, S.GATE_SEEDS, "gate", spec.PIN)


# ── the dev read ────────────────────────────────────────────────────────────────────────────────────────────────────
@needs_dev
def test_dev_read_reproduces_the_dev_verdict(dev):
    assert dev["n_defined"] == 192
    for b in S.PACKET_BARS:
        t = dev["bars"][b]
        assert t["verdict"] == "PASS" and t["pass"] == 11 and t["of"] == 11, (b, t)
    assert dev["bars"]["5"]["need"] == 9 and dev["bars"]["1"]["need"] == 11
    att = dev["bars"]["4"]["attribution"]
    assert (att["treatment_intact_a"], att["control_null_toad_diffs"], att["treatment_reversed_reported"]) == (29, 0, 24)
    assert att["fraction"] == 1.0 and att["verdict"] == "PASS"
    assert dev["DEC"]["verdict"] == "PASS"
    assert "9" in dev["rows"] and 9 not in dev["counted"]          # the design seed is read, never counted


@needs_dev
def test_dev_informational_numbers_the_prereg_quotes(dev):
    rows = [dev["rows"][str(s)] for s in S.DEV_COUNTED]
    info = [r["info"] for r in rows]
    # (a) Lpinned passes every C3 clause in both orders; (b) pconst too, its vector = C on 9/11 and 11/11
    assert all(i["Lpinned_intact_a"]["all"] and i["Lpinned_reversed"]["all"] for i in info)
    assert all(i["pconst_intact_a"]["all"] and i["pconst_reversed"]["all"] for i in info)
    assert sum(i["pconst_intact_a"]["equals_C_vector"] for i in info) == 9
    assert sum(i["pconst_reversed"]["equals_C_vector"] for i in info) == 11
    # (c) the host count rule k=1 reproduces C's ask decision on 117 of 121 abstain probes
    assert sum(i["null_count"]["1"][0] for i in info) == 117 and sum(i["null_count"]["1"][1] for i in info) == 121
    # (e) C intact_b is an exact replay of intact_a (the responses compare equal as JSON)
    assert all(r["2"]["responses_identical"] for r in rows)
    # (f) every applied C probe: asked == (want_decision_hz >= threshold)
    assert all(i[t]["host_threshold_agrees"][0] == i[t]["host_threshold_agrees"][1] > 0
               for i in info for t in ("C_intact_a", "C_reversed"))
    # (g) the library answer rate: 7 of 19 replies answered, flag on and flag off, every counted seed
    assert all((r["AR"]["pairs"]["C_intact_a"]["on_answered"], r["AR"]["pairs"]["C_intact_a"]["off_answered"],
                r["AR"]["pairs"]["C_intact_a"]["n"]) == (7, 7, 19) for r in rows)
    # (a) seed 12 lesion_P intact_a still asked all three novel gaps
    s12 = dev["rows"]["12"]["info"]["lesionP_intact_a"]["asked"]
    assert s12[0] == s12[2] == s12[4] == "1"


@needs_dev
def test_gate_mode_on_dev_shards_is_undefined_without_the_prereg_argv():
    res = S.score(S.load_cells(DEV_ROOT, (7,)), (7,), "gate", S.DEV_PIN)
    assert res["n_defined"] == 0 and res["DEC"]["verdict"] == "UNDEFINED"
    assert all("--allow-gate-seed --prereg" in "; ".join(v) for v in res["cells"].values())


# ── the gate read on synthetic six-gate-seed sets ───────────────────────────────────────────────────────────────────
@needs_dev
@pytest.mark.parametrize("gen,want", [
    ("null_no_stop", "FAIL"), ("null_stop_order_carried", "FAIL"), ("null_learnable_not_asked", "FAIL"),
    ("null_moats_asked", "FAIL"), ("null_nondeterministic", "FAIL"), ("null_permuted_inert", "FAIL"),
    ("null_lesionL_inert", "FAIL"), ("null_lesionL_one_order", "FAIL"), ("null_lesionP_inert", "FAIL"),
    ("null_lesion_curiosity_asks", "FAIL"), ("null_answer_lost", "FAIL"), ("null_recalled_differs", "FAIL"),
    ("null_lesionP_not_held", "FAIL"), ("null_lesionL_not_held", "FAIL"), ("null_one_seed_no_stop", "FAIL"),
    ("null_one_seed_lesionP_inert", "FAIL"), ("null_one_seed_answer_lost", "FAIL"),
    ("null_missing_cell", "UNDEFINED"), ("null_sidecar_off_pin", "UNDEFINED"), ("null_sidecar_no_prereg", "UNDEFINED"),
    ("null_engagement_missing", "UNDEFINED"), ("null_flag_off_not_off", "UNDEFINED"),
    ("null_sidecar_env_wrong", "UNDEFINED"),
    ("perfect", "PASS"), ("perfect_goat_acquired", "PASS"), ("info_informational_cell_missing", "PASS")])
def test_gate_decision_on_synthetic_sets(spec, gen, want):
    for k in range(3):
        res = _gate(getattr(spec, gen)(np.random.default_rng(k)), spec)
        assert res["DEC"]["verdict"] == want, (gen, k, {b: v["verdict"] for b, v in res["bars"].items()})


@needs_dev
def test_one_seed_lesionP_inert_fails_g5_although_a_9_of_11_rate_would_admit_it(spec):
    res = _gate(spec.null_one_seed_lesionP_inert(np.random.default_rng(0)), spec)
    t = res["bars"]["5"]
    assert (t["pass"], t["need"], t["verdict"]) == (5, 6, "FAIL") and 5 / 6 >= 9 / 11


@needs_dev
def test_the_scorer_does_not_mutate_its_input(spec):
    cells = spec.perfect(np.random.default_rng(0))
    before = {k: id(v["shard"]["responses"]) for k, v in cells.items()}
    snap = repr(sorted((k, sorted(v["shard"]["responses"])) for k, v in cells.items()))
    _gate(cells, spec)
    assert {k: id(v["shard"]["responses"]) for k, v in cells.items()} == before
    assert repr(sorted((k, sorted(v["shard"]["responses"])) for k, v in cells.items())) == snap


# ── the pool lines ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_emitted_lines_shape():
    lines = S.emit_lines()
    assert len(lines) == 6 * 16
    for cmd, why in lines:
        m = re.search(r"--seed (\d+) --group ask --arm (\S+) --allow-gate-seed --prereg (\S+) --out (\S+)$", cmd)
        assert m, cmd
        seed, arm, prereg, out = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        tag = re.search(r"ask__(\S+)\.json$", out).group(1)
        assert seed in S.GATE_SEEDS and prereg == S.PREREG and S.TAGS[tag][0] == arm
        assert out == "%s/s%d/ask__%s.json" % (S.GATE_DIR, seed, tag)
        assert cmd.startswith("cd ~/derisk-pool/revisions/<sha> && bash tools/mem_ok.sh 9 1 && mkdir -p %s/s%d && "
                              "bash tools/memcap.sh 10 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 " % (S.GATE_DIR, seed))
        envs = dict(re.findall(r"(BRAIN_CURIOSITY[A-Z_]*)=(\S+)", cmd))
        assert envs == S.env_of(tag), (tag, envs)
        assert "prereg %s" % S.PREREG in why and why.endswith("mem_gb=9 max_age_h=240")
        assert "\t" not in cmd and "\t" not in why and "\n" not in why


def test_committed_lines_file_is_the_emitted_lines():
    with open(LINES) as f:
        got = [l.rstrip("\n") for l in f if l.strip()]
    assert got == ["%s\t%s" % cw for cw in S.emit_lines()]


def test_every_line_parses_in_the_runner(monkeypatch):
    """A dry argparse parse of every line through stage2_exit_gate.main (run_shard stubbed: no brain)."""
    seen = []

    def stub(seed, group, arm, out, backend="numpy", labels=None, allow_gate_seed=False, prereg=None):
        seen.append((seed, group, arm, out, allow_gate_seed, prereg))
        return 0
    monkeypatch.setattr(G, "run_shard", stub)
    for cmd, _why in S.emit_lines():
        argv = cmd.split("-m research.runners.stage2_exit_gate ", 1)[1].split()
        monkeypatch.setattr(sys, "argv", ["stage2_exit_gate"] + argv)
        with pytest.raises(SystemExit) as e:
            G.main()
        assert e.value.code == 0
    assert len(seen) == 96 and all(x[4] is True and x[5] == S.PREREG and x[1] == "ask" for x in seen)


def test_closure_holds_the_arms_code():
    c = set(S.import_closure())
    for f in ("research/runners/stage2_exit_gate.py", "webapp/server.py", "webapp/curiosity_topic_stop_chat.py",
              "research/runners/curiosity_topic_stop_organ.py", "research/runners/_curiosity_lp_trace_derisk.py",
              "webapp/da_curiosity_drives_chat.py", "webapp/da_mode_drives_chat.py", "webapp/one_brain.py",
              "sim/bridge.py", "sim/neuromodulators.py"):
        assert f in c, f
