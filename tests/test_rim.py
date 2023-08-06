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
    rim = RIM(D, model, score_fn=score_fn, T=T)
    x_series =  rim(y)
    assert len(x_series) == T
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])

def test_rim_hourglass_basic_2d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(2)
    rim = RIM(D, model, score_fn=score_fn, T=T)
    x_series =  rim(y)
    assert len(x_series) == T
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])

def test_rim_hourglass_energy_1d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    rim = RIM(D, model, energy_fn=energy_fn, T=T)
    x_series =  rim(y)
    assert len(x_series) == T
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])

def test_rim_hourglass_model_init_1d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    rim = RIM(D, model, score_fn=score_fn, T=T, initialization_method="model")
    x_series =  rim(y)
    assert len(x_series) == T + 1
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])


def test_rim_hourglass_link_fn_1d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    link_fn = lambda x: 2*x
    inv_link_fn = lambda x: x/2
    rim = RIM(D, model, score_fn=score_fn, T=T, link_function=link_fn, inverse_link_function=inv_link_fn, initialization_method="model")
    x_series =  rim(y)
    assert len(x_series) == T + 1
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])


def test_rim_hourglass_adam_preprocessing_1d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    rim = RIM(D, model, score_fn=score_fn, T=T, score_preprocessing_method="adam")
    x_series =  rim(y)
    assert len(x_series) == T 
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])

def test_rim_hourglass_arcsinh_1d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    rim = RIM(D, model, score_fn=score_fn, T=T, score_preprocessing_method="arcsinh")
    x_series =  rim(y)
    assert len(x_series) == T 
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])


def test_rim_hourglass_rmsprop_preprocessing_1d():
    x, y, model, score_fn, energy_fn, H, B, C, T, D = setup(1)
    rim = RIM(D, model, score_fn=score_fn, T=T, score_preprocessing_method="rmsprop")
    x_series =  rim(y)
    assert len(x_series) == T
    for x in x_series:
        assert x.shape == torch.Size([B, C, *D])


def test_link_fn():
    _, y, model, _, energy_fn, H, B, C, T, D = setup(1)
    link_fn = lambda x: 2 * x
    i_link_fn = lambda x: x / 2
    def score_fn(x, y, *args):
        return x
    rim = RIM(D, model, score_fn=score_fn, T=T, link_function=link_fn, inverse_link_function=i_link_fn)
    
    x = torch.ones(B, C, *D)
    rim_score = rim.model_score_fn(x, y)
    assert torch.all(rim_score == 4*x) # Jacobian of link is 2, and score_fn return link(x), so we get 4x

    link_fn = lambda x: torch.ones_like(x) * 2
    i_link_fn = lambda x: torch.zeros_like(x) # special case to make sure loss is implemented correctly
    def score_fn(x, y, *args):
        return torch.ones_like(x)
    rim = RIM(D, model, score_fn=score_fn, T=T, link_function=link_fn, inverse_link_function=i_link_fn)
    
    # predict should equal 2
    y_hat = rim.predict(y)
    assert torch.all(y_hat == torch.ones_like(y_hat) * 2)
    
    # Make sure loss equals prediction squared (because inverse link set x_true to 0)
    x_true = torch.randn_like(x)
    loss = rim.loss_fn(y, x_true)
    x_series = rim(y)
    my_loss = sum([((x)**2).sum() for x in x_series]) / B # assumes w is 1, will have to work on that
    print(loss)
    assert loss.item() == my_loss.item()

if __name__ == "__main__":
    test_link_fn()

