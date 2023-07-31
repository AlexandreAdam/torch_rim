from typing import Union

import torch
from torch.nn import functional as F
from torch.nn import Module
from .definitions import DEVICE
import os, re, json
from glob import glob
import numpy as np
import contextlib

class NullEMA:
    """
    An EMA emulator that does nothing so that ema_decay=0 can be supported. 
    """
    def update(self, parameters=None):
        pass

    def copy_to(self, parameters=None):
        pass

    def store(self, parameters=None):
        pass

    def restore(self, parameters=None):
        pass

    def average_parameters(self, parameters=None):
        return contextlib.nullcontext()

    def to(self, device=None, dtype=None):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass



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


def load_architecture(
        checkpoints_directory, 
        model: Union[str, Module] = None, 
        dimensions=1, 
        hyperparameters=None, 
        device=DEVICE
        ) -> list[Module, dict]:
    if hyperparameters is None:
        hyperparameters = {}
    if model is None:
        with open(os.path.join(checkpoints_directory, "model_hparams.json"), "r") as f:
            hparams = json.load(f)
        hparams.update(hyperparameters)
        model = hparams.get("architecture", "hourglass")
        if "dimensions" not in hparams.keys():
            hparams["dimensions"] = dimensions
    if isinstance(model, str):
        if model.lower() == "hourglass":
            from rim import Hourglass
            model = Hourglass(**hparams).to(device)
        else:
            raise ValueError(f"{model} not supported")
    else:
        hparams = model.hyperparameters
    paths = glob(os.path.join(checkpoints_directory, "checkpoint*.pt"))
    checkpoints = [int(re.findall('[0-9]+', os.path.split(path)[-1])[-1]) for path in paths]
    if paths:
        try:
            model.load_state_dict(torch.load(paths[np.argmax(checkpoints)], map_location=device))
        except (KeyError, RuntimeError) as e:
            raise e # need to fix what's below
            # # Maybe the RIM instance was used when saving the weights, in which case we hack the loading process
            # from rim.rim import RIM
            # model = RIM(model, **hyperparameters)
            # model.load_state_dict(torch.load(paths[np.argmax(checkpoints)], map_location=device))
            # model = model.model # Remove the RIM wrapping to extract the nn
    return model, hparams 
