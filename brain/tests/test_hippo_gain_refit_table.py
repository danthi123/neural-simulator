from brain.experiments.hippo_gain_refit_table import fit, verdict


def rows(bank_kinds, gain, n_right, n):
    out = []
    for i in range(n):
        out.append({"gain": gain, "kind": bank_kinds[i % len(bank_kinds)], "correct": i < n_right,
                    "class": "right" if i < n_right else "generic"})
    return out


DEV = ("orig", "para_a", "para_b")


def test_fit_prefers_most_right_then_smaller_gain():
    dev = rows(DEV, 0, 0, 108) + rows(DEV, 2, 30, 108) + rows(DEV, 4, 30, 108) + rows(DEV, 3, 10, 108)
    assert fit(dev) == 2


def test_verdicts():
    dev = rows(DEV, 0, 0, 108) + rows(DEV, 2, 40, 108)
    test_ok = rows(("orig",), 2, 7, 12) + rows(("para",), 2, 5, 12)
    assert verdict(dev, test_ok)["verdict"] == "LEAD"
    test_bad = rows(("orig",), 2, 2, 12) + rows(("para",), 2, 5, 12)
    assert verdict(dev, test_bad)["verdict"] == "NOT A LEAD"
    weak = rows(DEV, 0, 0, 108) + rows(DEV, 2, 10, 108)
    assert verdict(weak, test_ok)["verdict"] == "NO GAIN WORKS"
    broken = rows(DEV, 0, 9, 108) + rows(DEV, 2, 40, 108)
    assert verdict(broken, test_ok)["verdict"] == "VOID"
