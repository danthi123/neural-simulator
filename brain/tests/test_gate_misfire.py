"""gate_misfire_probe: the gate fires iff familiarity < bar; the DEV sweep takes the lowest bar reaching the catch rate."""
from brain.experiments.gate_misfire_probe import auroc, bar_value, fires, sweep


def test_fires_strictly_below_bar():
    assert fires([0.1, 0.2, 0.3], 0.2) == 1


def test_sweep_lowest_bar_reaching_catch():
    never = [0.05, 0.10, 0.15, 0.20, 0.50]          # 80% catch needs bar > 0.20
    told = [0.12, 0.30, 0.40, 0.60]
    bar, mis, cat = sweep(told, never, 0.8)
    assert bar == 0.30 and cat == 0.8 and mis == 0.25   # bar 0.30: told 0.12 fires


def test_sweep_all_never_above_told():
    bar, mis, cat = sweep([0.1], [0.5, 0.6], 1.0)
    assert cat == 1.0 and mis == 1.0


def test_auroc_and_bars():
    assert auroc([1.0, 2.0], [0.0]) == 1.0 and auroc([0.0], [0.0]) == 0.5
    assert bar_value("0.22", 42) == 0.22 and abs(bar_value("F0.104", 42) - 0.104) < 1e-3
