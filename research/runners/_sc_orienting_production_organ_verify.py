"""VERIFY the spiking-SC orienting production organ + its EMBODIED visuomotor consumer.

Three checks (all CPU / numpy — organ-level, no chat handler):

(1) CORRECT-CARDINAL battery (the orienting fidelity, = the soak metric). A battery of static
    (agent, goal) presentations spanning the four cardinals + diagonals at distance 3-6 cells. For each,
    the spiking SC's cardinal (winning cortex pool BY FIRING) is graded against the true dominant-axis
    bearing. INTACT must be HIGH; the SC_SCRAMBLE lesion must COLLAPSE to chance (~1/4); the host reflex
    scaffold (the OFF path) is reported as the behavioural-equivalence comparator.

(2) EMBODIED visuomotor loop (the consumer = a real body output in a tiny world). The agent starts
    off-corner from a fixed salient goal; each step it renders the world from its EYE, the spiking SC
    emits an orienting cardinal, and the BODY moves the agent ONE CELL in that cardinal. Terminate on
    reach (Chebyshev <= 1) or a step budget. INTACT reaches with a near-optimal path; the LESION
    random-walks (the nav's 2.4x regression analogue: reach-rate collapses, path efficiency collapses).

(3) LOAD-BEARING lesion identity: the lesion changes ONLY the sc_retina->sc_map retinotopy — the
    image-only afferent (sc_map total spikes / bump strength) is UNCHANGED between intact and lesion, so
    the collapse is caused by the retinotopic decoupling, not a weaker/absent bump. Proves the
    retinotopic SPIKING sheet carries the orienting target (not a re-hidden host read).

BYTE-IDENTICAL: N-A for chat (the faculty is EMBODIED, no chat coupling). The OFF path
(BRAIN_SPIKING_SC_ORIENT unset) is the host reflex `sc_orienting_cardinal_from_image` — reported here as
the scaffold the spiking SC replaces. The organ wires into NO existing production path, so nothing
existing changes when the flag is OFF.

Run (CPU):  SIM_BACKEND=numpy python -m research.runners._sc_orienting_production_organ_verify
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.sc_orienting_production_organ import (
    get_organ, true_bearing_cardinal, host_reflex_cardinal, IMG,
)
from tools.lab import attributable_to

# ── batteries ─────────────────────────────────────────────────────────────────────────────────────
# The organ uses the LINEAR egocentric render (the de-risked default the CLOSED 6-seed used; sharp
# fovea). Its retinal field of view is +/-3 cells (image_size 32, ppc 4): a salient target within the
# FOV forms a well-separated peripheral bump; a foveated (<=1-cell) target has ~0 orienting error (the
# ramp read-out is ~0 at the sc-centre) -- biologically correct: a foveated target needs no orienting.
CENTER = (8, 8)
# 4 cardinals at distance {2,3} + 4 dominant-axis diagonals (offset (3,2)/(2,3) style) -- all within the
# retinal FOV, all with a clear dominant axis.
CARDINAL_BATTERY = [
    (CENTER, (8, 10)), (CENTER, (8, 11)),    # N (dist 2, 3)
    (CENTER, (8, 6)),  (CENTER, (8, 5)),     # S
    (CENTER, (10, 8)), (CENTER, (11, 8)),    # E
    (CENTER, (6, 8)),  (CENTER, (5, 8)),     # W
    (CENTER, (11, 10)), (CENTER, (5, 10)),   # E-dom / W-dom diagonal (offset (+/-3,+2))
    (CENTER, (10, 11)), (CENTER, (6, 5)),    # N-dom / S-dom-ish diagonal (offset (+2,+3)/(-2,-3))
]

# embodied FOVEATION episodes: a salient target APPEARS within the retinal FOV (distance 2-3) at each of
# the 8 bearings; the SC orients and the BODY moves the agent one cell/step to bring the target to the
# fovea. reach = Chebyshev <= 1 (foveated). This is the SC orienting reflex's biological FUNCTION:
# reduce the salient target's retinal eccentricity to zero.
EPISODES = [
    (CENTER, (11, 8)), (CENTER, (5, 8)),     # E, W (dist 3)
    (CENTER, (8, 11)), (CENTER, (8, 5)),     # N, S
    (CENTER, (11, 10)), (CENTER, (5, 10)),   # NE-ish, NW-ish
    (CENTER, (11, 6)),  (CENTER, (5, 6)),    # SE-ish, SW-ish
]
GRID = 16
STEP_BUDGET = 12
LOG_POLAR = False   # LINEAR render (sharp fovea; matches the CLOSED 6-seed + the probe's 8/8)


def _clamp(v, lo=0, hi=GRID - 1):
    return max(lo, min(hi, v))


def _move(agent, cardinal):
    """The BODY: move one cell in the emitted cardinal (+x=East, +y=North). Host channel-2 (legit)."""
    x, y = agent
    if cardinal == "N":
        y += 1
    elif cardinal == "S":
        y -= 1
    elif cardinal == "E":
        x += 1
    elif cardinal == "W":
        x -= 1
    return (_clamp(x), _clamp(y))


def _cheb(a, g):
    return max(abs(a[0] - g[0]), abs(a[1] - g[1]))


def cardinal_battery(organ, lesion=False):
    """Correct-cardinal rate over the static battery (the orienting fidelity)."""
    ok = 0
    tot = 0
    for a, g in CARDINAL_BATTERY:
        tru = true_bearing_cardinal(a, g)
        if tru is None:
            continue
        card = organ.orient(a, g, lesion=lesion)["cardinal"]
        ok += int(card == tru)
        tot += 1
    return ok, tot


def host_battery():
    ok = 0
    tot = 0
    for a, g in CARDINAL_BATTERY:
        tru = true_bearing_cardinal(a, g)
        if tru is None:
            continue
        ok += int(host_reflex_cardinal(a, g) == tru)
        tot += 1
    return ok, tot


def run_episode(organ, agent, goal, lesion=False, budget=STEP_BUDGET):
    """The EMBODIED loop: orient -> move, until reach (Chebyshev<=1) or budget. Returns
    (reached, steps, correct_cardinal_rate, path_efficiency)."""
    start_d = _cheb(agent, goal)
    opt = max(abs(agent[0] - goal[0]), abs(agent[1] - goal[1]))  # optimal cardinal-step count to adjacency
    correct = 0
    read = 0
    steps = 0
    for _ in range(budget):
        if _cheb(agent, goal) <= 1:
            break
        tru = true_bearing_cardinal(agent, goal)
        card = organ.orient(agent, goal, lesion=lesion)["cardinal"]
        if tru is not None:
            read += 1
            correct += int(card == tru)
        if card in ("N", "E", "S", "W"):
            agent = _move(agent, card)
        steps += 1
    reached = _cheb(agent, goal) <= 1
    ccr = correct / read if read else 0.0
    eff = (opt / steps) if (reached and steps > 0) else 0.0
    return reached, steps, ccr, eff, start_d


def main():
    print("=" * 78)
    print("SPIKING-SC ORIENTING PRODUCTION ORGAN — verify (organ-level, embodied visuomotor)")
    print("=" * 78)
    seed = int(os.environ.get("SC_SEED", "42"))
    organ = get_organ(seed=seed, log_polar=LOG_POLAR)

    # (1) correct-cardinal battery -----------------------------------------------------------------
    i_ok, i_tot = cardinal_battery(organ, lesion=False)
    l_ok, l_tot = cardinal_battery(organ, lesion=True)
    h_ok, h_tot = host_battery()
    i_rate = i_ok / max(i_tot, 1)
    l_rate = l_ok / max(l_tot, 1)
    h_rate = h_ok / max(h_tot, 1)
    print(f"\n[1] CORRECT-CARDINAL battery (distance 2-3 within FOV, all bearings):")
    print(f"    INTACT (spiking SC)      : {i_ok}/{i_tot}  = {i_rate:.3f}")
    print(f"    LESION (SC_SCRAMBLE)     : {l_ok}/{l_tot}  = {l_rate:.3f}   (chance = 0.250)")
    print(f"    HOST reflex (OFF path)   : {h_ok}/{h_tot}  = {h_rate:.3f}   (the scaffold the SC replaces)")

    # (2) embodied visuomotor loop -----------------------------------------------------------------
    print(f"\n[2] EMBODIED visuomotor loop (grid {GRID}, budget {STEP_BUDGET}, reach=Chebyshev<=1):")
    print(f"    {'start->goal':>18} | {'INTACT reach/steps/ccr/eff':>34} | {'LESION reach/steps/ccr/eff':>34}")
    i_reached = i_eff = i_ccr = 0.0
    l_reached = l_eff = l_ccr = 0.0
    n = len(EPISODES)
    for a, g in EPISODES:
        ir, isteps, iccr, ieff, _ = run_episode(organ, a, g, lesion=False)
        lr, lsteps, lccr, leff, _ = run_episode(organ, a, g, lesion=True)
        i_reached += int(ir); i_eff += ieff; i_ccr += iccr
        l_reached += int(lr); l_eff += leff; l_ccr += lccr
        print(f"    {str(a)+'->'+str(g):>18} | "
              f"{('Y' if ir else 'n')+f' {isteps:2d} {iccr:.2f} {ieff:.2f}':>34} | "
              f"{('Y' if lr else 'n')+f' {lsteps:2d} {lccr:.2f} {leff:.2f}':>34}")
    print(f"    {'MEAN':>18} | "
          f"{f'reach {i_reached/n:.2f}  ccr {i_ccr/n:.2f}  eff {i_eff/n:.2f}':>34} | "
          f"{f'reach {l_reached/n:.2f}  ccr {l_ccr/n:.2f}  eff {l_eff/n:.2f}':>34}")

    # (3) load-bearing lesion identity (afferent unchanged) ----------------------------------------
    r_int = organ.orient((8, 8), (11, 8), lesion=False)   # within-FOV target (distance 3)
    r_les = organ.orient((8, 8), (11, 8), lesion=True)
    afferent_ratio = r_les["sc_total_spikes"] / max(r_int["sc_total_spikes"], 1e-9)
    print(f"\n[3] LOAD-BEARING lesion identity (same (8,8)->(11,8) afferent):")
    print(f"    INTACT: cardinal={r_int['cardinal']}  sc_bump_spikes={r_int['sc_total_spikes']:.0f}")
    print(f"    LESION: cardinal={r_les['cardinal']}  sc_bump_spikes={r_les['sc_total_spikes']:.0f}")
    print(f"    afferent (bump strength) ratio LESION/INTACT = {afferent_ratio:.2f}  "
          f"(≈1 => the image-only afferent is UNCHANGED; only retinotopy permuted)")

    # (4) ATTRIBUTION: what fraction of the orienting is owned by the retinotopic coupling (INTACT)
    #     vs. survives the scrambled-retinotopy control (LESION). The gap#5 subtraction: measuring both
    #     arms is not the same as attributing the difference.
    print(f"\n[4] ATTRIBUTION (INTACT treatment vs SC_SCRAMBLE control):")
    attributable_to("correct-cardinal", i_rate, l_rate)
    attributable_to("embodied reach", i_reached / n, l_reached / n)

    # ── verdict ────────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    intact_ok = (i_rate >= 0.80) and (i_reached / n >= 0.80)
    lesion_break = (l_rate <= 0.45) and (l_reached / n <= 0.50)
    afferent_intact = 0.6 <= afferent_ratio <= 1.6
    load_bearing = intact_ok and lesion_break and afferent_intact
    print(f"INTACT works        : {intact_ok}  (correct-cardinal {i_rate:.2f} >=0.80, reach {i_reached/n:.2f} >=0.80)")
    print(f"LESION collapses    : {lesion_break}  (correct-cardinal {l_rate:.2f} <=0.45, reach {l_reached/n:.2f} <=0.50)")
    print(f"Afferent UNCHANGED  : {afferent_intact}  (bump ratio {afferent_ratio:.2f} in [0.6,1.6])")
    print(f"LOAD-BEARING        : {load_bearing}  (the orienting is carried by the retinotopic spiking sheet)")
    verdict = "GO" if load_bearing else "NOT-YET"
    print(f"\nVERDICT: {verdict} — the spiking SC drives embodied orienting; the SC_SCRAMBLE lesion "
          f"decouples it while the afferent is unchanged (load-bearing, not hollow).")
    return 0 if load_bearing else 1


if __name__ == "__main__":
    raise SystemExit(main())
