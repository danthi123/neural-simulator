from brain.experiments.speaker_contrast_table import verdict


def cell(f, g, n_right, n_echo=0, n=108, tag="q"):
    out = []
    for i in range(n):
        cl = "right" if i < n_right else ("echo" if i < n_right + n_echo else "none")
        out.append({"format": f, "gain": g, "q": f"{tag}{i}", "kind": ("orig", "para_a", "para_b")[i % 3],
                    "class": cl, "correct": cl == "right"})
    return out


def test_pick_on_dev_and_judge_fresh():
    dev = cell("tell", 0.5, 58) + cell("contrast100", 0.5, 71) + cell("contrast200", 0.5, 75)
    fresh = cell("tell", 0.5, 50, 20) + cell("contrast100", 0.5, 70, 2) + cell("contrast200", 0.5, 60, 5)
    v = verdict(dev, fresh)
    assert v["best"] == ("contrast200", 0.5) and v["verdict"] == "LEAD"
    fresh2 = cell("tell", 0.5, 50, 20) + cell("contrast200", 0.5, 55, 2)
    assert verdict(dev, fresh2 + cell("contrast100", 0.5, 1))["verdict"] == "NOT A LEAD"
