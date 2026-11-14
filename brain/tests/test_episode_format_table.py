from brain.experiments.episode_format_table import verdict


def rows(fmt, g, kinds, n_right, n, n_echo=0):
    out = []
    for i in range(n):
        cl = "right" if i < n_right else ("echo" if i < n_right + n_echo else "none")
        out.append({"format": fmt, "gain": g, "kind": kinds[i % len(kinds)], "class": cl, "correct": cl == "right"})
    return out


D = ("orig", "para_a", "para_b")


def base_dev():
    return rows("tell", 0.5, D, 58, 108, 20) + rows("tell", 1.0, D, 35, 108, 20)


def base_test():
    return rows("tell", 0.5, ("orig",), 7, 12) + rows("tell", 0.5, ("para",), 7, 12)


def test_lead():
    dev = base_dev() + rows("lived", 0.5, D, 70, 108, 5)
    test = base_test() + rows("lived", 0.5, ("orig",), 9, 12) + rows("lived", 0.5, ("para",), 7, 12)
    assert verdict(dev, test)["verdict"] == "LEAD"


def test_not_a_lead_and_void():
    dev = base_dev() + rows("lived", 0.5, D, 60, 108, 5)
    test = base_test() + rows("lived", 0.5, ("orig",), 9, 12) + rows("lived", 0.5, ("para",), 7, 12)
    assert verdict(dev, test)["verdict"] == "NOT A LEAD"
    bad = rows("tell", 0.5, D, 57, 108) + rows("lived", 0.5, D, 90, 108)
    assert verdict(bad, test)["verdict"] == "VOID"
