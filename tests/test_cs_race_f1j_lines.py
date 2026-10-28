"""CI guard -- the cs-race f1j STEP-0 pool-lines generator (research/runners/cs_race_f1j_lines.py) and the generated file
research/coordination/cs-race-f1j-0-pool-lines.txt (packet research/coordination/cs-race-f1j-PACKET.md, "Step-0 pool
lines: generator spec", the invariants the packet says the test pins):
  1. regenerating waves 1-2 equals the committed file byte for byte, and wave 1 (90 lines) is written before wave 2 (952);
  2. per-arm counts 2 / 72 / 16 / 144 / 144 / 504 / 144 / 16 = 1042, unique --out paths, all under
     research/findings/raw/_cs_race/f1j/step0/, the f1i stems, no gate seed, seeds in packet order inside each cell;
  3. every TRACE line has a LESION twin differing only by --som-lesion and the stem; every IPSC-12 line differs from its
     TRACE twin only by --tau-gi 12 and the stem; every APV line differs from the matching f1i step-0 line only by the
     NMDA flags, the output directory and the stem; the identity lines are the f1i corner-cell lines with the stem moved;
  4. every --nmda-g0 line has --nmda-tau; W_MS 429 appears only in APV, GRID, K1-NOFAC (and NOMEM); every reason names
     the arm and ends "mem_gb=1 max_age_h=240"; `<sha>` stays a placeholder;
  5. every line parses through the runner's own argparse (cs_race_f1._build_parser + _base_from_args) and through the
     f1i scorer's line reader;
  6. wave 3 (NOMEM) from an explicit working-gN0 set: 45 lines per gN0 on the 45 f1i early records re-derived from the
     landed f1i artifacts, --nmda-tau 2, no --out repeating a wave 1-2 path; a gN0 outside the grid is refused.
"""
import os
import shlex
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from research.runners import cs_race_f1i_score as F1I  # noqa: E402
from research.runners import cs_race_f1j_lines as LN  # noqa: E402

LINES_PATH = os.path.join(REPO, LN.LINES)
F1I_LINES = os.path.join(REPO, "research/coordination/cs-race-f1i-0-pool-lines.txt")
F1I_ROOT = os.path.join(REPO, LN.F1I_DIR)
HAVE_F1I = os.path.exists(os.path.join(F1I_ROOT, "cal_sa0p125_ms215_s41.json"))
N_WAVE = {"identity": 2, "apv": 72, "k1nofac": 16, "cal": 144, "lesion": 144, "grid": 504, "ipsc12": 144,
          "ipsc12_apv": 16}


def _lines():
    return LN.wave1() + LN.wave2()


def _cmd(ln):
    return ln.split("\t")[0]


def _argv(ln):
    """The runner argv of a line (after `-m research.runners.cs_race_f1`)."""
    toks = shlex.split(_cmd(ln))
    return toks[toks.index("research.runners.cs_race_f1") + 1:]


def _out(ln):
    a = _argv(ln)
    return a[a.index("--out") + 1]


def _stem(ln):
    return os.path.basename(_out(ln))[:-len(".json")]


def _arm(ln):
    st = _stem(ln)
    if st.startswith("ipsc12_apv_"):
        return "ipsc12_apv"
    return st.split("_")[0]


def _seed(ln):
    a = _argv(ln)
    return int(a[a.index("--seeds") + 1])


def _val(ln, flag):
    a = _argv(ln)
    return a[a.index(flag) + 1] if flag in a else None


def _strip(argv, drop=(), drop_with_value=()):
    out, i = [], 0
    while i < len(argv):
        if argv[i] in drop_with_value:
            i += 2
            continue
        if argv[i] in drop:
            i += 1
            continue
        out.append(argv[i])
        i += 1
    return out


def _cmd_minus(ln, drop=(), drop_with_value=()):
    """The cmd with the --out value, the named flags and the mkdir dir removed (what the twin rules compare)."""
    toks = shlex.split(_cmd(ln))
    j = toks.index("research.runners.cs_race_f1")
    head = [t for t in toks[:j + 1] if not t.startswith("research/findings/raw/_cs_race/")]
    return head + _strip(toks[j + 1:], drop, ("--out",) + tuple(drop_with_value))


# ── 1. the committed file ──────────────────────────────────────────────────────────────────────────────────────────

def test_regenerating_equals_the_committed_file():
    with open(LINES_PATH) as f:
        committed = f.read()
    assert committed == LN.render(_lines())


def test_waves_in_packet_order():
    lines = _lines()
    assert len(LN.wave1()) == 90 and len(LN.wave2()) == 952 and len(lines) == 1042
    order = [_arm(ln) for ln in lines]
    assert set(order[:90]) == {"identity", "apv", "k1nofac"}
    assert set(order[90:]) == {"cal", "lesion", "grid", "ipsc12", "ipsc12_apv"}
    firsts = [a for i, a in enumerate(order) if i == 0 or order[i - 1] != a]
    assert firsts == ["identity", "apv", "k1nofac", "cal", "lesion", "grid", "ipsc12", "ipsc12_apv"]


# ── 2. counts, outputs, seeds ──────────────────────────────────────────────────────────────────────────────────────

def test_counts_unique_outputs_and_seeds():
    lines = _lines()
    counts = {}
    for ln in lines:
        counts[_arm(ln)] = counts.get(_arm(ln), 0) + 1
    assert counts == N_WAVE and sum(counts.values()) == 1042
    outs = [_out(ln) for ln in lines]
    assert len(set(outs)) == 1042
    assert all(o.startswith(LN.OUT_DIR + "/") and o.endswith(".json") for o in outs)
    assert all(o.count("/") == LN.OUT_DIR.count("/") + 1 for o in outs)
    assert not {_seed(ln) for ln in lines} & set(LN.GATE_SEEDS)
    assert {_seed(ln) for ln in lines} == set(LN.SEEDS)
    # seeds 41, 45-51 in that order inside each cell
    groups = {}
    for ln in lines:
        groups.setdefault(_stem(ln).rsplit("_s", 1)[0], []).append(_seed(ln))
    for stem, seeds in groups.items():
        want = list(LN.IDENTITY_SEEDS) if stem.startswith("identity") else list(LN.SEEDS)
        assert seeds == want, stem


def test_stems_follow_the_packet_tags():
    stems = {_stem(ln) for ln in _lines()}
    assert "identity_f1i_sa0p125_ms215_s41" in stems and "identity_f1i_sa0p125_ms215_s45" in stems
    assert "apv_sa0p5_ms429_s51" in stems and "k1nofac_sa0p25_ms429_s41" in stems
    assert "cal_t40_g4_sa0p125_ms215_s41" in stems and "lesion_t75_g16_sa0p125_ms304_s51" in stems
    assert "grid_t50_g8_sa0p5_ms429_s47" in stems and "ipsc12_t75_g16_sa0p125_ms304_s50" in stems
    assert "ipsc12_apv_sa0p125_ms304_s41" in stems
    assert not any("ms429" in s for s in stems if s.startswith(("cal_", "lesion_", "ipsc12_")))


# ── 3. twins ───────────────────────────────────────────────────────────────────────────────────────────────────────

def test_lesion_twin_of_every_trace_line():
    by = {_stem(ln): ln for ln in _lines()}
    n = 0
    for stem, ln in by.items():
        if not stem.startswith("cal_"):
            continue
        tw = by["lesion_" + stem[len("cal_"):]]
        assert "--som-lesion" not in _argv(ln) and "--som-lesion" in _argv(tw)
        assert _cmd_minus(tw, drop=("--som-lesion",)) == _cmd_minus(ln)
        n += 1
    assert n == 144


def test_ipsc12_twin_of_every_trace_line():
    by = {_stem(ln): ln for ln in _lines()}
    n = 0
    for stem, ln in by.items():
        if not stem.startswith("ipsc12_t"):
            continue
        tw = by["cal_" + stem[len("ipsc12_"):]]
        assert _val(ln, "--tau-gi") == "12" and _val(tw, "--tau-gi") is None
        assert _cmd_minus(ln, drop_with_value=("--tau-gi",)) == _cmd_minus(tw)
        n += 1
    assert n == 144


def _f1i_by_stem():
    with open(F1I_LINES) as f:
        return {_stem(ln): ln for ln in f.read().splitlines() if ln.strip()}


def test_apv_and_identity_lines_are_the_f1i_lines_plus_the_nmda_flags():
    f1i = _f1i_by_stem()
    n = 0
    for ln in _lines():
        st = _stem(ln)
        if _arm(ln) == "apv":
            ref = f1i["cal_" + st[len("apv_"):]]
            assert _cmd_minus(ln, drop_with_value=("--nmda-g0", "--nmda-tau")) == _cmd_minus(ref)
            assert (_val(ln, "--nmda-g0"), _val(ln, "--nmda-tau")) == ("0", "50")
            # textually: the f1i cmd with the output dir moved, the flags inserted after --som-ap and the stem changed
            want = (_cmd(ref).replace("_cs_race/f1i/step0", "_cs_race/f1j/step0")
                    .replace(" --out ", " --nmda-g0 0 --nmda-tau 50 --out ").replace("/cal_", "/apv_"))
            assert _cmd(ln) == want
            n += 1
        elif _arm(ln) == "identity":
            ref = f1i["cal_" + st[len("identity_f1i_"):]]
            assert _cmd(ln) == (_cmd(ref).replace("_cs_race/f1i/step0", "_cs_race/f1j/step0")
                                .replace("/cal_", "/identity_f1i_"))
            n += 1
    assert n == 74


def test_k1nofac_lines():
    f1i = _f1i_by_stem()
    k1 = [ln for ln in _lines() if _arm(ln) == "k1nofac"]
    assert {(float(_val(ln, "--som-ap")), int(_val(ln, "--som"))) for ln in k1} == set(LN.K1_CELLS)
    for ln in k1:
        ref = f1i["cal_" + _stem(ln)[len("k1nofac_"):]]
        assert _cmd_minus(ln, drop=("--som-nofac",), drop_with_value=("--nmda-g0", "--nmda-tau")) == _cmd_minus(ref)
        assert (_val(ln, "--nmda-g0"), _val(ln, "--nmda-tau")) == ("0", "50")


# ── 4. flags and reasons ───────────────────────────────────────────────────────────────────────────────────────────

def test_flag_rules_and_reasons():
    for ln in _lines():
        a = _argv(ln)
        arm = _arm(ln)
        if "--nmda-g0" in a:
            assert "--nmda-tau" in a
        assert ("--nmda-g0" in a) == (arm != "identity")
        if _val(ln, "--som") == "429":
            assert arm in ("apv", "grid", "k1nofac"), _stem(ln)
        if arm in ("cal", "lesion", "grid", "ipsc12"):
            t, g = _stem(ln).split("_")[1:3]
            assert (_val(ln, "--nmda-tau"), _val(ln, "--nmda-g0")) == (t[1:], g[1:])
            assert (int(t[1:]), int(g[1:])) in LN.NMDA_CELLS
        assert ("--tau-gi" in a) == arm.startswith("ipsc12")
        cmd, reason = ln.split("\t")
        assert cmd.startswith("cd ~/derisk-pool/revisions/<sha> && ")
        assert reason.startswith("cs-race f1j step-0 ")
        assert reason.endswith("mem_gb=1 max_age_h=240")
        assert "research/coordination/cs-race-f1j-PACKET.md (Step 0; seeds 41,45-51, no gate seeds)" in reason
        assert "RULING f1i-0-1028 (f1j = A-NMDA; cap-raise +1)" in reason
        assert ln.count("\t") == 1 and "\n" not in ln


# ── 5. every line parses ───────────────────────────────────────────────────────────────────────────────────────────

def test_every_line_parses_through_the_runner_argparse():
    import research.runners.cs_race_f1 as M
    ap = M._build_parser()
    seen = {}
    for ln in _lines():
        args = ap.parse_args(_argv(ln))
        base = M._base_from_args(ap, args)
        arm = _arm(ln)
        assert bool(base.get("NMDA_ON")) == (arm != "identity")
        if arm != "identity":
            assert base["NMDA_G0"] == float(_val(ln, "--nmda-g0"))
            assert base["NMDA_TAU"] == float(_val(ln, "--nmda-tau"))
        assert base["TAU_GI"] == (12.0 if arm.startswith("ipsc12") else 0.0)
        assert bool(base["SOM_LESION"]) == (arm == "lesion") and bool(base["SOM_NOFAC"]) == (arm == "k1nofac")
        seen[arm] = seen.get(arm, 0) + 1
    assert seen == N_WAVE


def test_every_line_parses_through_the_f1i_line_reader(tmp_path):
    p = tmp_path / "lines.txt"
    p.write_text(LN.render(_lines()))
    parsed = F1I.parse_lines(str(p))
    assert len(parsed) == 1042
    for ln in _lines():
        assert parsed[_out(ln)] == _argv(ln)


def test_cli_writes_the_same_file(tmp_path):
    out = tmp_path / "x.txt"
    assert LN.main(["--out", str(out)]) == 0
    assert out.read_text() == LN.render(_lines())


# ── 6. wave 3 ──────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAVE_F1I, reason="f1i step-0 artifacts not in this checkout")
def test_wave3_from_a_working_set():
    from research.runners import cs_race_f1j_score as S
    early = S.f1i_c6_records(F1I_ROOT)["early"]
    assert len(early) == 45
    w3 = LN.wave3((16, 8), F1I_ROOT)
    assert len(w3) == 90
    wave12 = {_out(ln) for ln in _lines()}
    assert not {_out(ln) for ln in w3} & wave12
    assert len({_out(ln) for ln in w3}) == 90
    for ln in w3:
        assert _arm(ln) == "nomem" and _val(ln, "--nmda-tau") == "2"
        assert _val(ln, "--nmda-g0") in ("8", "16")
        key = (float(_val(ln, "--som-ap")), int(_val(ln, "--som")), _seed(ln))
        assert key in early
        assert _stem(ln) == "nomem_g%s_%s_s%d" % (_val(ln, "--nmda-g0"), LN.cell_tag(key[0], key[1]), key[2])
        assert ln.endswith("mem_gb=1 max_age_h=240")
    assert [_val(ln, "--nmda-g0") for ln in w3[:45]] == ["8"] * 45      # sorted gN0, early-record order inside
    import research.runners.cs_race_f1 as M
    ap = M._build_parser()
    for ln in w3[:3]:
        M._base_from_args(ap, ap.parse_args(_argv(ln)))
    with pytest.raises(ValueError):
        LN.wave3((12,), F1I_ROOT)
    with pytest.raises(ValueError):
        LN.wave3((), F1I_ROOT)


def test_every_arm_moves_its_argv_against_its_twin():
    from tools.lab import LeverError
    lines = _lines()
    assert LN.twin_levers(lines) == {"--som-lesion": 144, "--tau-gi 12": 144, "--som-nofac": 16,
                                     "--tau-gi 12 (APV)": 16,
                                     "A-NMDA (--nmda-g0 G > 0) vs APV at the same cell": 144 + 504}
    # a LESION line that lost its flag is identical to its TRACE twin: the A/B would be void, so it raises
    i = next(k for k, ln in enumerate(lines) if _stem(ln).startswith("lesion_"))
    broken = list(lines)
    broken[i] = lines[i].replace(" --som-lesion", "", 1)
    with pytest.raises(LeverError):
        LN.twin_levers(broken)
    # an arm with no twin is refused, never skipped
    with pytest.raises(ValueError):
        LN.twin_levers([ln for ln in lines if not _stem(ln).startswith("apv_sa0p25_ms304")])
