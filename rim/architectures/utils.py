import torch
from torch.nn import functional as F

def get_activation(activation:str):
    if activation.lower() == "elu":
        return F.elu
    elif activation.lower() == "relu":
        return F.relu
    elif activation.lower() == "tanh":
        return F.tanh
    elif activation.lower() == "swish" or activation.lower() == "silu":
        return F.silu
    elif activation.lower() == "sigmoid":
        return F.sigmoid
    else:
        raise ValueError(f"activation {activation} is not supported")
