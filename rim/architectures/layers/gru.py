import torch
from torch import nn
from torch.nn import functional as F
from rim.definitions import DTYPE

class GRU(nn.Module):
    """
    Gated Recurrent Unit (GRU) with convolutional layers
    """
    def __init__(self, in_ch:int, hidden_units:int=32, kernel_size:int=5, dimensions:int=0):
        super().__init__()
        self.dimensions = dimensions
        self.bias_broadcast = [1, -1] + [1]*dimensions
        if dimensions == 0:
            layer = lambda in_ch, out_ch: nn.Linear(in_ch, out_ch, bias=False)
        elif dimensions == 1:
            layer = lambda in_ch, out_ch: nn.Conv1d(in_ch, out_ch, kernel_size, padding="same", bias=False)
        elif dimensions == 2:
            layer = lambda in_ch, out_ch: nn.Conv2d(in_ch, out_ch, kernel_size, padding="same", bias=False)
        elif dimensions == 3:
            layer = lambda in_ch, out_ch: nn.Conv3d(in_ch, out_ch, kernel_size, padding="same", bias=False)
        else:
            raise ValueError(f"The dimensions {dimensions} is not supported. Only 0, 1, 2 and 3 dimensional input (not including channels) are.")
        # Update gate
        self.w_z = layer(in_ch, hidden_units)
        self.u_z = layer(hidden_units, hidden_units)
        self.bias_z = nn.Parameter(torch.zeros(hidden_units, dtype=DTYPE))

        # Reset gate
        self.w_r = layer(in_ch, hidden_units)
        self.u_r = layer(hidden_units, hidden_units)
        self.bias_r = nn.Parameter(torch.zeros(hidden_units, dtype=DTYPE))
        
        # Candidate activation gate
        self.w_h = layer(in_ch, hidden_units)
        self.u_h = layer(hidden_units, hidden_units)
        self.bias_h = nn.Parameter(torch.zeros(hidden_units, dtype=DTYPE))

    def forward(self, x, h):
        """
        Compute the new state tensor h_{t} from h_{t-1} and x_{t}
        """
        z = F.sigmoid(self.w_z(x) + self.u_z(h) + self.bias_z.view(1, -1, *[1]*self.dimensions))  # update gate
        r = F.sigmoid(self.w_r(x) + self.u_r(h) + self.bias_r.view(1, -1, *[1]*self.dimensions))  # reset gate
        h_tilde = F.tanh(self.w_h(x) + self.u_h(r * h) + self.bias_h.view(1, -1, *[1]*self.dimensions))  # candidate activation
        new_state = (1 - z) * h + z * h_tilde
        return new_state, new_state  # h_{t+1}
