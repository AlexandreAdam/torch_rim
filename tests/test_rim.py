from rim import RIM, Hourglass
import torch

def setup(dim):
    H = 32
    B = 10
    C = 15 # channels
    T = 10
    ch_mult = (2, 2)
    dim = 1
    D = [16]*dim
    x = torch.randn(B, C, *D)
    y = torch.randn_like(x)
    model = Hourglass(C, ch_mult=ch_mult, dimensions=dim)
    score_fn = lambda x, y: torch.randn_like(x)
    energy_fn = lambda x, y: torch.ones([B])
    return x, y, model, score_fn, energy_fn, H, B, C, T, D

def test_rim_hourglass_basic_1d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    rim = RIM(model, D, score_fn, T=T)
    x_series =  rim(y)
    assert len(x_series) == T
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])

def test_rim_hourglass_basic_2d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(2)
    rim = RIM(model, D, score_fn, T=T)
    x_series =  rim(y)
    assert len(x_series) == T
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])

def test_rim_hourglass_energy_1d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    rim = RIM(model, D, energy_fn=energy_fn, T=T)
    x_series =  rim(y)
    assert len(x_series) == T
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])

def test_rim_hourglass_model_init_1d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    rim = RIM(model, D, score_fn, T=T, initialization_method="model")
    x_series =  rim(y)
    assert len(x_series) == T + 1
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])


def test_rim_hourglass_link_fn_1d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    link_fn = lambda x: 2*x
    inv_link_fn = lambda x: x/2
    rim = RIM(model, D, score_fn, T=T, link_function=link_fn, inverse_link_function=inv_link_fn, initialization_method="model")
    x_series =  rim(y)
    assert len(x_series) == T + 1
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])


def test_rim_hourglass_adam_preprocessing_1d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    rim = RIM(model, D, score_fn, T=T, score_preprocessing_method="adam")
    x_series =  rim(y)
    assert len(x_series) == T 
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])

def test_rim_hourglass_arcsinh_1d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    rim = RIM(model, D, score_fn, T=T, score_preprocessing_method="arcsinh")
    x_series =  rim(y)
    assert len(x_series) == T 
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])


def test_rim_hourglass_rmsprop_preprocessing_1d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    rim = RIM(model, D, score_fn, T=T, score_preprocessing_method="rmsprop")
    x_series =  rim(y)
    assert len(x_series) == T
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])

