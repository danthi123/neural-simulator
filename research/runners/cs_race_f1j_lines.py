"""Generator for the cs-race f1j STEP-0 pool lines (packet research/coordination/cs-race-f1j-PACKET.md, "Step-0 pool
lines: generator spec"). Deterministic, no clock, no run.

    python -m research.runners.cs_race_f1j_lines                       # waves 1-2 -> LINES (1042 lines)
    python -m research.runners.cs_race_f1j_lines --nomem-g0 8,16       # wave 3 (NOMEM) -> LINES_WAVE3

One line per run, `cmd<TAB>reason`, the f1i format. The cmd is the f1i step-0 command (research/coordination/
cs-race-f1i-0-pool-lines.txt) with the output directory moved to research/findings/raw/_cs_race/f1j/step0/, the arm
flags inserted after `--som-ap <W_SA>` and the f1j stem. `<sha>` stays a placeholder: a revision that contains the
merged E build, the runner flags and the packet.

Waves (the packet's order): wave 1 = identity (i) 2, APV 72, K1-NOFAC 16 (90 lines); wave 2 = TRACE 144, LESION 144,
GRID 504, IPSC-12 144, IPSC-12 APV 16 (952 lines); total 1042. Wave 3 (NOMEM, conditional and informational) is written
only from an explicit working-gN0 set (the f1j step-0 score names it), 45 lines per gN0 on the 45 f1i early (cell, seed)
records re-derived from the landed f1i artifacts (research/runners/cs_race_f1j_score.f1i_c6_records).
Seeds 41, 45, 46, 47, 48, 49, 50, 51 in that order inside each cell; seeds 42-44 and 100-102 are gate seeds, never run.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import shlex
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.lab import lever  # noqa: E402  (every arm's manipulation must MOVE its argv vs its twin: gates/attribution_required)

PACKET = "research/coordination/cs-race-f1j-PACKET.md"
LINES = "research/coordination/cs-race-f1j-0-pool-lines.txt"
LINES_WAVE3 = "research/coordination/cs-race-f1j-0-pool-lines-wave3.txt"
OUT_DIR = "research/findings/raw/_cs_race/f1j/step0"
F1I_DIR = "research/findings/raw/_cs_race/f1i/step0"

SEEDS = (41, 45, 46, 47, 48, 49, 50, 51)
GATE_SEEDS = (42, 43, 44, 100, 101, 102)
IDENTITY_SEEDS = (41, 45)
W_SA = (0.125, 0.25, 0.5)
W_MS = (215, 304, 429)
TAU_N = (40, 50, 75)                 # ms: Wang 2008 adult V1 L5-L5, Poleg-Polsky 2015 / replay centre, Rhodes/Doron
G_N0 = (4, 8, 16)                    # g_L: the memo's set (unanchored in this current-source model)
NMDA_CELLS = tuple((t, g) for t in TAU_N for g in G_N0)
F1I_CELLS = tuple((sa, ms) for sa in W_SA for ms in W_MS)
PRIMARY_MS = (215, 304)
GRID_CELLS = tuple(c for c in F1I_CELLS if c not in ((0.125, 215), (0.125, 304)))   # the 7 other f1i cells
K1_CELLS = ((0.125, 429), (0.25, 429))
APV_FLAGS = ("--nmda-g0", "0", "--nmda-tau", "50")
NOMEM_TAU = 2

CMD_HEAD = ("cd ~/derisk-pool/revisions/<sha> && bash tools/mem_ok.sh 1 1 && mkdir -p {out_dir} && bash tools/memcap.sh 2 "
            "-- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
            "CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m research.runners.cs_race_f1 --calibrate --seeds {seed} "
            "--grid-row 3 --w-gis 30 --w-mi 20 --w-im 40 --nim 8 --encode-words --ment-pa 150 --trace intrinsic "
            "--soma-g 40 --apical-pulse 120 --som {ms} --som-ap {sa}")

REASON_TAIL = ("packet research/coordination/cs-race-f1j-PACKET.md (Step 0; seeds 41,45-51, no gate seeds); "
               "RULING f1i-0-1028 (f1j = A-NMDA; cap-raise +1); base cell RULING f1h-0-1027 + the f1i SOM path; "
               "no apical-NMDA engine run exists (the step-1c engram NMDA arm, nmda_grid_n3, is a different "
               "mechanism); tauN from the adult-cortex literature, gN0 the memo's set; the open-loop replay of the "
               "f1i records on the calibration seeds informed tauN 100->40 and where the grid sits (unrecorded; "
               "disclosed in the packet); mem_gb=1 max_age_h=240")


def sa_tag(sa):
    return "sa" + ("%g" % sa).replace(".", "p")


def cell_tag(sa, ms):
    return "%s_ms%d" % (sa_tag(sa), int(ms))


def nmda_tag(tau, g0):
    return "t%d_g%d" % (int(tau), int(g0))


def _fmt(x):
    return "%g" % x


def line(stem, seed, sa, ms, arm_flags, reason):
    head = CMD_HEAD.format(out_dir=OUT_DIR, seed=int(seed), ms=int(ms), sa=_fmt(sa))
    flags = (" " + " ".join(arm_flags)) if arm_flags else ""
    cmd = "%s%s --out %s/%s.json" % (head, flags, OUT_DIR, stem)
    if "\t" in cmd or "\t" in reason or "\n" in reason:
        raise ValueError("tab or newline inside a pool line")
    return "%s\t%s; %s" % (cmd, reason, REASON_TAIL)


def _nm(tau, g0):
    return ("--nmda-g0", "%d" % g0, "--nmda-tau", "%d" % tau)


def wave1():
    out = []
    for s in IDENTITY_SEEDS:
        out.append(line(
            "identity_f1i_%s_s%d" % (cell_tag(0.125, 215), s), s, 0.125, 215, (),
            "cs-race f1j step-0 IDENTITY (i), STOP-ID: the f1i runner with NO f1j flag at the corner cell W_SA "
            "0.125 W_MS 215, seed %d; the record must hash equal, minus wall_s, to %s/cal_%s_s%d.json (host-class "
            "rule of identity (iii) applies)" % (s, F1I_DIR, cell_tag(0.125, 215), s)))
    for sa, ms in F1I_CELLS:
        for s in SEEDS:
            out.append(line(
                "apv_%s_s%d" % (cell_tag(sa, ms), s), s, sa, ms, APV_FLAGS,
                "cs-race f1j step-0 APV (NO-NMDA) arm (--nmda-g0 0 --nmda-tau 50: receptor blocked, gating runs), "
                "f1i cell W_SA %g W_MS %d, seed %d; identity (iii) against %s/cal_%s_s%d.json, S11 (g_N 0 at every "
                "ms, s reaches s_full on the pulsed slots), and the same-pin baseline for every rescue count"
                % (sa, ms, s, F1I_DIR, cell_tag(sa, ms), s)))
    for sa, ms in K1_CELLS:
        for s in SEEDS:
            out.append(line(
                "k1nofac_%s_s%d" % (cell_tag(sa, ms), s), s, sa, ms, ("--som-nofac",) + APV_FLAGS,
                "cs-race f1j step-0 K1-NOFAC arm (informational, the carried K1 item's falsifier: --som-nofac static "
                "slot->SOM at W_MS x 0.049, APV), W_SA %g W_MS %d, seed %d; late C6 fails against the APV line of "
                "the same cell and seed" % (sa, ms, s)))
    return out


def wave2():
    out = []
    trace = [(t, g, ms) for (t, g) in NMDA_CELLS for ms in PRIMARY_MS]
    for t, g, ms in trace:
        for s in SEEDS:
            out.append(line(
                "cal_%s_%s_s%d" % (nmda_tag(t, g), cell_tag(0.125, ms), s), s, 0.125, ms, _nm(t, g),
                "cs-race f1j step-0 TRACE arm (selectable; A-NMDA tauN %d ms, gN0 %d g_L), W_SA 0.125 W_MS %d, "
                "seed %d; items C0-C8, rescue / late / kept / eraser against the APV line of the same cell and seed, "
                "S14" % (t, g, ms, s)))
    for t, g, ms in trace:
        for s in SEEDS:
            out.append(line(
                "lesion_%s_%s_s%d" % (nmda_tag(t, g), cell_tag(0.125, ms), s), s, 0.125, ms,
                ("--som-lesion",) + _nm(t, g),
                "cs-race f1j step-0 LESION arm (--som-lesion: the cs_som gate closed the whole run, SOM cells still "
                "driven and firing), A-NMDA tauN %d gN0 %d, W_SA 0.125 W_MS %d, seed %d; scores C7(d) and identity "
                "(ii) for the TRACE line of the same cell and seed, S14" % (t, g, ms, s)))
    for t, g in NMDA_CELLS:
        for sa, ms in GRID_CELLS:
            for s in SEEDS:
                out.append(line(
                    "grid_%s_%s_s%d" % (nmda_tag(t, g), cell_tag(sa, ms), s), s, sa, ms, _nm(t, g),
                    "cs-race f1j step-0 GRID arm (never selectable; A-NMDA tauN %d gN0 %d at the f1i cell W_SA %g "
                    "W_MS %d), seed %d; re-runs the f1i grid closed loop for F1 and the prediction table (rescue / "
                    "late / kept, grid C7) against the APV line of the same cell and seed, S14" % (t, g, sa, ms, s)))
    for t, g, ms in trace:
        for s in SEEDS:
            out.append(line(
                "ipsc12_%s_%s_s%d" % (nmda_tag(t, g), cell_tag(0.125, ms), s), s, 0.125, ms,
                ("--tau-gi", "12") + _nm(t, g),
                "cs-race f1j step-0 IPSC-12 arm (--tau-gi 12: every GABA_A decay 12 ms, the f1g pool's somatic IPSC "
                "too; A-NMDA tauN %d gN0 %d), W_SA 0.125 W_MS %d, seed %d; reads F3 against the IPSC-12 APV line of "
                "the same cell and seed; S13 reads 12, S14" % (t, g, ms, s)))
    for ms in PRIMARY_MS:
        for s in SEEDS:
            out.append(line(
                "ipsc12_apv_%s_s%d" % (cell_tag(0.125, ms), s), s, 0.125, ms, ("--tau-gi", "12") + APV_FLAGS,
                "cs-race f1j step-0 IPSC-12 APV arm (--tau-gi 12, --nmda-g0 0 --nmda-tau 50), W_SA 0.125 W_MS %d, "
                "seed %d; the F3 baseline; S11, S13 reads 12" % (ms, s)))
    return out


def wave3(g0_set, f1i_root=None):
    """NOMEM lines for each working gN0 (explicit argument) on the 45 f1i early (cell, seed) records."""
    from research.runners.cs_race_f1j_score import f1i_c6_records
    root = f1i_root or os.path.join(_REPO, F1I_DIR)
    early = f1i_c6_records(root)["early"]
    if len(early) != 45:
        raise RuntimeError("f1i early records re-derived: %d, the packet's count is 45" % len(early))
    gs = sorted(set(int(g) for g in g0_set))
    if not gs or any(g not in G_N0 for g in gs):
        raise ValueError("the working gN0 set must be a non-empty subset of %r, got %r" % (G_N0, g0_set))
    out = []
    for g in gs:
        for (sa, ms, s) in early:
            out.append(line(
                "nomem_g%d_%s_s%d" % (g, cell_tag(sa, ms), s), s, sa, ms,
                ("--nmda-g0", "%d" % g, "--nmda-tau", "%d" % NOMEM_TAU),
                "cs-race f1j step-0 NOMEM arm (wave 3, informational: --nmda-tau 2 keeps the in-volley clamp and "
                "removes the post-edge memory; gN0 %d from the scored working set), f1i early record W_SA %g W_MS %d "
                "seed %d; a rescue NOMEM also produces is an in-volley clamp rescue, not the memory; S14"
                % (g, sa, ms, s)))
    return out


def _argv_by_stem(lines):
    out = {}
    for ln in lines:
        toks = shlex.split(ln.split("\t")[0])
        argv = toks[toks.index("research.runners.cs_race_f1") + 1:]
        i = argv.index("--out")
        stem = os.path.basename(argv[i + 1])[:-len(".json")]
        out[stem] = argv[:i] + argv[i + 2:]
    return out


TWINS = (  # (arm prefix, twin prefix, the manipulation the arm adds to its twin)
    ("lesion_", "cal_", "--som-lesion"),
    ("ipsc12_t", "cal_t", "--tau-gi 12"),
    ("cal_", "apv_", "A-NMDA (--nmda-g0 G > 0) vs APV at the same cell"),
    ("grid_", "apv_", "A-NMDA (--nmda-g0 G > 0) vs APV at the same cell"),
    ("k1nofac_", "apv_", "--som-nofac"),
    ("ipsc12_apv_", "apv_", "--tau-gi 12 (APV)"),
)


def twin_levers(lines):
    """Every arm's manipulation must actually move the runner argv against its twin (tools.lab.lever, required): an
    arm identical to its twin would make that A/B void. Returns {manipulation: number of twin pairs checked}.
    TRACE and GRID (<arm>_t<T>_g<G>_<cell>_s<seed>) are checked against the APV line of the same cell and seed."""
    by = _argv_by_stem(lines)
    counts = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for stem, argv in by.items():
            for arm, twin, what in TWINS:
                if not stem.startswith(arm) or (arm == "ipsc12_t" and stem.startswith("ipsc12_apv_")):
                    continue
                if twin == "apv_":
                    rest = stem[len(arm):]
                    if arm in ("cal_", "grid_"):
                        rest = rest.split("_", 2)[2]          # drop t<T>_g<G>
                    tw = "apv_" + rest
                else:
                    tw = twin + stem[len(arm):]
                if tw not in by:
                    raise ValueError("%s has no %s twin %s" % (stem, twin, tw))
                lever("%s: %s vs %s" % (what, stem, tw), " ".join(by[tw]), " ".join(argv), required=True)
                counts[what] = counts.get(what, 0) + 1
    return counts


def render(lines):
    return "".join(ln + "\n" for ln in lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="output file (default: %s, or %s with --nomem-g0)"
                    % (LINES, LINES_WAVE3))
    ap.add_argument("--nomem-g0", default=None,
                    help="wave 3 only: the comma-separated working gN0 set named by the f1j step-0 score")
    ap.add_argument("--f1i-root", default=None, help="f1i step-0 artifacts (default %s)" % F1I_DIR)
    a = ap.parse_args(argv)
    if a.nomem_g0:
        lines = wave3([int(x) for x in a.nomem_g0.split(",") if x.strip()], a.f1i_root)
        dst = a.out or LINES_WAVE3
    else:
        lines = wave1() + wave2()
        dst = a.out or LINES
        print("twin levers moved: %s" % twin_levers(lines))
    path = dst if os.path.isabs(dst) else os.path.join(_REPO, dst)
    with open(path, "w") as f:
        f.write(render(lines))
    print("%d lines -> %s" % (len(lines), dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
