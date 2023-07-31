from rim.architectures import Hourglass
import torch


def test_hourglass_1d():
    H = 32
    B = 10
    C = 15 # channels
    ch_mult = (2, 2)
    dim = 1
    D = [16]*dim
    x = torch.randn(B, C, *D)
    score = torch.randn(B, C, *D)
    model = Hourglass(C, ch_mult=ch_mult, dimensions=dim)
    h = model.initalize_hidden_states(D, B)
    assert h.shape == torch.Size([B, H, *[d//4 for d in D]])
    print(h.shape)
    g, new_h = model(x, None, score, h)
    assert g.shape == torch.Size([B, C, *D])
    assert new_h.shape == h.shape

def test_hourglass_odd_input_size():
    H = 32
    B = 10
    C = 15 # channels
    ch_mult = (2, 2)
    dim = 1
    D = [15]*dim
    x = torch.randn(B, C, *D)
    score = torch.randn(B, C, *D)
    model = Hourglass(C, ch_mult=ch_mult, dimensions=dim)
    h = model.initalize_hidden_states(D, B)
    assert h.shape == torch.Size([B, H, *[d//4 for d in D]])
    print(h.shape)
    g, new_h = model(x, None, score, h)
    assert g.shape == torch.Size([B, C, *D])
    assert new_h.shape == h.shape

        
def test_hourglass_2d():
    H = 32
    B = 10
    C = 15 # channels
    ch_mult = (2, 2)
    dim = 2 
    D = [16]*dim
    x = torch.randn(B, C, *D)
    score = torch.randn(B, C, *D)
    model = Hourglass(C, ch_mult=ch_mult, dimensions=dim)
    h = model.initalize_hidden_states(D, B)
    assert h.shape == torch.Size([B, H, *[d//4 for d in D]])
    print(h.shape)
    g, new_h = model(x, None, score, h)
    assert g.shape == torch.Size([B, C, *D])
    assert new_h.shape == h.shape
        
def test_hourglass_3d():
    H = 32
    B = 10
    C = 15 # channels
    ch_mult = (2, 2)
    dim =3
    D = [16]*dim
    x = torch.randn(B, C, *D)
    score = torch.randn(B, C, *D)
    model = Hourglass(C, ch_mult=ch_mult, dimensions=dim)
    h = model.initalize_hidden_states(D, B)
    assert h.shape == torch.Size([B, H, *[d//4 for d in D]])
    print(h.shape)
    g, new_h = model(x, None, score, h)
    assert g.shape == torch.Size([B, C, *D])
    assert new_h.shape == h.shape
