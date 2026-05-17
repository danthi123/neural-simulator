import json
import torch
from sim.tiny_transformer import TinyGPT


def _mk(V=20, dm=16, nl=2, nh=2, bs=8):
    torch.manual_seed(0)
    m = TinyGPT(vocab_size=V, d_model=dm, n_layer=nl, n_head=nh,
                block_size=bs).eval()
    return m


def test_forward_shape_and_param_count():
    m = _mk()
    idx = torch.randint(0, 20, (3, 8))
    out = m(idx)
    assert tuple(out.shape) == (3, 8, 20)
    assert sum(p.numel() for p in m.parameters()) > 0


def test_deterministic_same_seed_same_output():
    a = _mk()
    b = _mk()
    idx = torch.randint(0, 20, (2, 8))
    assert torch.allclose(a(idx), b(idx), atol=0)


def test_causal_mask_no_future_leak():
    m = _mk(V=20, bs=8)
    torch.manual_seed(1)
    idx = torch.randint(0, 20, (1, 8))
    base = m(idx).detach().clone()
    idx2 = idx.clone()
    idx2[0, 7] = (idx[0, 7].item() + 5) % 20
    alt = m(idx2).detach()
    assert torch.equal(base[:, :7, :], alt[:, :7, :])
    assert not torch.equal(base[:, 7, :], alt[:, 7, :])
    idx3 = idx.clone()
    idx3[0, 4] = (idx[0, 4].item() + 3) % 20
    alt3 = m(idx3).detach()
    assert torch.equal(base[:, :4, :], alt3[:, :4, :])


def test_save_load_roundtrip_logit_exact(tmp_path):
    m = _mk()
    idx = torch.randint(0, 20, (2, 8))
    before = m(idx).detach().clone()
    p = str(tmp_path / "tg")
    m.save(p)
    assert json.loads(open(p + ".meta.json").read())["d_model"] == 16
    m2 = TinyGPT.load(p).eval()
    assert torch.equal(m2(idx).detach(), before)
