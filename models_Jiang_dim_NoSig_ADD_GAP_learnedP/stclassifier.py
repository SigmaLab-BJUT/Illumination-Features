import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .pse import PixelSetEncoder
from .tae import TemporalAttentionEncoder
from .decoder import get_decoder


class PseTae(nn.Module):
    """
    Pixel-Set encoder + Temporal Attention Encoder sequence classifier
    """

    def __init__(self, input_dim=44, mlp0=None, mlp1=None, mlp2=None, with_extra=True, mlp4=None,
                 extra_size=9, n_head=4, d_k=32, d_model=None, mlp3=None, dropout=0.2, T=1000, len_max_seq=90,
                 positions=None):
        super(PseTae, self).__init__()
        if mlp3 is None:
            mlp3 = [256, 64, 64]
        self.spatial_encoder = PixelSetEncoder(input_dim, mlp0=mlp0, mlp1=mlp1,  mlp2=mlp2, with_extra=with_extra,
                                               extra_size=extra_size)
        self.temporal_encoder = TemporalAttentionEncoder(in_channels=mlp2[-1], n_head=n_head, d_k=d_k, d_model=d_model,
                                                         n_neurons=mlp3, dropout=dropout,
                                                         T=T, len_max_seq=len_max_seq, positions=positions)


    def forward(self, input):
        out = self.spatial_encoder(input)

        out = self.temporal_encoder(out)

        return out
