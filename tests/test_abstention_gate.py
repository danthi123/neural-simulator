from research.runners.abstention_gate import abstain, gate

def test_above_keeps(): assert abstain(796.0) is False
def test_below_abstains(): assert abstain(584.0) is True
def test_boundary(): assert abstain(650.0) is True and abstain(650.1) is False
def test_custom_threshold(): assert abstain(700.0, threshold=800.0) is True
def test_gate_returns_answer():
    ranked = [("big", 779.0, "apple_big"), ("spoon", 410.0, "apple")]
    assert gate(ranked) == ("big", 779.0, "apple_big")
def test_gate_abstains_below():
    assert gate([("noise", 500.0, "x")]) is None
def test_gate_empty(): assert gate([]) is None
