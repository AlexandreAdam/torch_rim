import torch
from rim.layers import GRU, GRUBlock


def test_gru():
    B = 10
    D = 10
    H = 32
    x = torch.randn(B, D)
    h = torch.randn(B, H)
    layer = GRU(D, H)
    h, _ = layer(x, h)
    print(h)
    assert h.shape == torch.Size([B, H])
    assert layer.dimensions == 0


def test_conv_gru_2d():
    B = 10
    dimensions = 2
    D = [16]*dimensions
    C = 10 # channels
    H = 32 # hidden units
    K = 5 # kernel size
    x = torch.randn(B, C, *D)
    h = torch.randn(B, H, *D)
    layer = GRU(C, H, K, dimensions)
    h, _ = layer(x, h)
    print(h)
    assert h.shape == torch.Size([B, H, *D])
    assert layer.dimensions == 2
    

def test_conv_gru_1d():
    B = 10
    dimensions = 1
    D = [16]*dimensions
    C = 10 # channels
    H = 32 # hidden units
    K = 5 # kernel size
    x = torch.randn(B, C, *D)
    h = torch.randn(B, H, *D)
    layer = GRU(C, H, K, dimensions)
    h, _ = layer(x, h)
    print(h)
    assert layer.dimensions == 1
    assert h.shape == torch.Size([B, H, *D])
    
    
def test_conv_gru_3d():
    B = 10
    dimensions = 3
    D = [16]*dimensions
    C = 10 # channels
    H = 32 # hidden units
    K = 5 # kernel size
    x = torch.randn(B, C, *D)
    h = torch.randn(B, H, *D)
    layer = GRU(C, H, K, dimensions)
    h, _ = layer(x, h)
    print(h)
    assert layer.dimensions == 3
    assert h.shape == torch.Size([B, H, *D])
    
def test_gru_block():
    B = 10
    C = 10 # channels
    H = 32 # hidden units
    K = 5 # kernel size
    for dim in [0, 1, 2, 3]:
        D = [16]*dim
        x = torch.randn(B, C, *D)
        h = torch.randn(B, H, *D)
        layer = GRUBlock(C, H, K, dim)
        h, _ = layer(x, h)
        print(h)
        assert layer.dimensions == dim
        assert h.shape == torch.Size([B, C, *D])

def test_transposed_conv():
    # This test is done inside score_models from which we get those layers, but repeat here since we rely on its behavior
    from rim.layers import ConvTransposed1dSame, ConvTransposed2dSame, ConvTransposed3dSame
    from rim.layers import Conv1dSame, Conv2dSame, Conv3dSame
    B = 10
    D = 15
    C = 16
    K = 3
    for dim in [1, 2, 3]:
        x = torch.randn(B, C, *[D]*dim)
        layer_ = [Conv1dSame, Conv2dSame, Conv3dSame][dim-1]
        layer = layer_(C, C, K, stride=2)
        x_down= layer(x)
        print("Down", x_down.shape)
        assert x_down.shape == torch.Size([B, C, *[D//2]*dim])
        
        layer_ = [ConvTransposed1dSame, ConvTransposed2dSame, ConvTransposed3dSame][dim-1]
        layer = layer_(C, C, K, stride=2)
        y = layer(x_down)
        print("Up", x.shape)
        assert x.shape == torch.Size([B, C, *[D]*dim])
        
    

