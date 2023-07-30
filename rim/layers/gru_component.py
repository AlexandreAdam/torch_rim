import torch
from torch import nn
from torch.nn import functional as F
from .gru import GRU

class GRUBlock(nn.Module):
    """
    Abstraction for the recurrent block inside the RIM with 2 hidden states, mostly following
    the architecture described in Morningstar et al (2019), https://arxiv.org/pdf/1901.01359.pdf
    """
    def __init__(self, in_ch:int, hidden_units:int=32, kernel_size:int=5, dimensions:int=0, conv_kernel_size=11):
        super().__init__()
        self.dimensions = dimensions
        if dimensions == 0:
            layer = lambda in_ch, out_ch: nn.Linear(in_ch, out_ch, bias=False)
        elif dimensions == 1:
            layer = lambda in_ch, out_ch: nn.Conv1d(in_ch, out_ch, conv_kernel_size, padding="same", bias=False)
        elif dimensions == 2:
            layer = lambda in_ch, out_ch: nn.Conv2d(in_ch, out_ch, conv_kernel_size, padding="same", bias=False)
        elif dimensions == 3:
            layer = lambda in_ch, out_ch: nn.Conv3d(in_ch, out_ch, conv_kernel_size, padding="same", bias=False)
        else:
            raise ValueError(f"The dimensions {dimensions} is not supported. Only 0, 1, 2 and 3 dimensional input (not including channels) are.")
        
        if hidden_units % 2 != 0:
            raise ValueError(f"hidden units {hidden_units} must be divisible by 2")
        
        self.gru1 = GRU(in_ch, hidden_units//2, kernel_size, dimensions)
        self.gru2 = GRU(hidden_units//2, hidden_units//2, kernel_size, dimensions)
        self.hidden_layer = layer(hidden_units//2, hidden_units//2)
        self.reshape_layer = layer(hidden_units//2, in_ch)

    def forward(self, x, h):
        h_1, h_2 = torch.tensor_split(h, 2, dim=1)
        h1, _ = self.gru1(x, h_1)
        pre_h2 = F.tanh(self.hidden_layer(h1))
        x, h_2 = self.gru2(pre_h2, h_2)
        x = self.reshape_layer(x)
        h = torch.concat([h_1, h_2], dim=1)
        return x, h

