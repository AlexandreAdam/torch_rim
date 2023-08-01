import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from .base import Model
from rim.utils import get_activation
from rim.layers import *

class Hourglass(Model):
    def __init__(
            self,
            channels,
            nf=32,
            ch_mult=(2, 2),
            hidden_units=32,
            dimensions=1,
            num_layers=2,
            kernel_size=3, # in case dimensions>0
            conv_kernel_size=None, # parameter for the GRU block
            input_kernel_size=None,
            condition_on_observation=False,
            resample_with_conv=True,
            fir_kernel=(1, 3, 3, 1),
            combine_method="cat", # either cat or sum
            activation="tanh",
            **kwargs
            ):
        super().__init__()
        if dimensions == 1:
            layer = lambda in_ch, out_ch, k: nn.Conv1d(in_ch, out_ch, k, padding="same")
            if resample_with_conv:
                downsample_layer = lambda in_ch, out_ch, k: Conv1dSame(in_ch, out_ch, k, stride=2)
                upsample_layer = lambda in_ch, out_ch, k: ConvTransposed1dSame(in_ch, out_ch, k, stride=2)
        elif dimensions == 2:
            layer = lambda in_ch, out_ch, k: nn.Conv2d(in_ch, out_ch, k, padding="same")
            if resample_with_conv:
                downsample_layer = lambda in_ch, out_ch, k: Conv2dSame(in_ch, out_ch, k, stride=2)
                upsample_layer = lambda in_ch, out_ch, k: ConvTransposed2dSame(in_ch, out_ch, k, stride=2)
        elif dimensions == 3:
            layer = lambda in_ch, out_ch, k: nn.Conv3d(in_ch, out_ch, k, padding="same")
            if resample_with_conv:
                downsample_layer = lambda in_ch, out_ch, k: Conv3dSame(in_ch, out_ch, k, stride=2)
                upsample_layer = lambda in_ch, out_ch, k: ConvTransposed3dSame(in_ch, out_ch, k, stride=2)
        else:
            raise ValueError(f"The dimensions {dimensions} is not supported. Only 1, 2 and 3 dimensional input (not including channels) are.")
        if not resample_with_conv:
            # up and downsampling with FIR kernel
            downsample_layer = lambda in_ch, out_ch, k: lambda x: downsample(x, k=fir_kernel, dimensions=dimensions)
            upsample_layer = lambda in_ch, out_ch, k: lambda x: upsample(x, k=fir_kernel, dimensions=dimensions)

        if conv_kernel_size is None:
            conv_kernel_size = kernel_size
        if input_kernel_size is None:
            input_kernel_size = kernel_size
        self.hyperparameters = {
                "channels": channels,
                "architecture": "hourglass",
                "nf": nf,
                "ch_mult": ch_mult,
                "dimensions": dimensions,
                "num_layers": num_layers,
                "resample_with_conv": resample_with_conv,
                "fir_kernel": fir_kernel,
                "combine_method": combine_method,
                "condition_on_observation": condition_on_observation,
                "kernel_size": kernel_size,
                "conv_kernel_size": conv_kernel_size,
                "input_kernel_size": input_kernel_size,
                "activation": activation
                }

        self.nf = nf
        self.channels = channels
        self.act = get_activation(activation)
        self.num_resolutions = len(ch_mult)
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.condition_on_observation = condition_on_observation
        self.decoder_index = self.num_resolutions * (num_layers + 1)
        
        if condition_on_observation:
            raise ValueError("Not currently supported")
        
        # 2 times channels because we need to include the score 
        # If we include the observation, then this would change
        self.input_layer = layer(2*channels, nf, input_kernel_size) #TODO adapt if we append observation
        modules = []
        c = nf
        # Decoder
        for i_level in range(self.num_resolutions):
            for i_layer in range(self.num_layers):
                modules.append(layer(c, c, kernel_size))
            next_c = c * ch_mult[i_level]
            modules.append(downsample_layer(c, next_c, kernel_size))
            c = next_c
        self.gru_layer = GRUBlock(c, hidden_units, kernel_size, dimensions, conv_kernel_size)
        # Encoder 
        for i_level in reversed(range(self.num_resolutions)):
            next_c = c // ch_mult[i_level]
            modules.append(upsample_layer(c, next_c, kernel_size))
            c = next_c
            for i_layer in range(self.num_layers):
                modules.append(layer(c, c, kernel_size))
        self.all_modules = nn.ModuleList(modules)
        self.output_layer = layer(c, channels, 1)

    def forward(self, x, y, score, h):
        x = torch.concat([x, score], dim=1) # for now don't include the observation in here
        x = self.act(self.input_layer(x))
        # Encoder
        for i_layer, layer in enumerate(self.all_modules[:self.decoder_index]):
            x = layer(x)
            if (i_layer + 1) % (self.num_layers + 1) != 0:
                x = self.act(x)
        x, h = self.gru_layer(x, h)
        # Decoder
        for i_layer, layer in enumerate(self.all_modules[self.decoder_index:]):
            x = layer(x)
            if (i_layer + 1) % (self.num_layers + 1) != 0:
                x = self.act(x)
        x = self.output_layer(x)
        return x, h
     
    def initalize_hidden_states(self, dimensions: list[int], batch_size: int) -> Tensor:
        """
        At inference time, figure out the spatial dimensions of the hidden units
        """
        # for now assume dimensions are equal
        d = dimensions[0]
        hidden_d = d // 2**self.num_resolutions
        h = torch.zeros(batch_size, self.hidden_units, *[hidden_d]*len(dimensions))
        return h
