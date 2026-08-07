import torch
import torch.nn as nn

from .decoder import get_decoder
from .stclassifier import PseTae
from .Xcep_Times_Ceit import Fusion


class CatNet(nn.Module):
    # 通道数
    def __init__(self, input_dim=44, mlp0=None, mlp1=None, mlp2=None, with_extra=True, mlp4=None,
                 extra_size=9, n_head=4, d_k=32, mlp3=None, dropout=0.2, T=1000, len_max_seq=90,
                 num_classes=2, depth=12, num_heads=8, dim=64, mlp_ratio=4., drop_rate=0., image_size=7,
                 attn_drop_rate=0.1, drop_path_rate=0., leff_local_size=3, leff_with_bn=True, n_segment=15, mlp5=None):
        super(CatNet, self).__init__()
        if mlp5 is None:
            mlp5 = [192, 2]

        self.LMN = PseTae(input_dim=input_dim, mlp0=mlp0, mlp1=mlp1, mlp2=mlp2, with_extra=with_extra,
                          extra_size=extra_size, n_head=n_head, d_k=d_k, mlp3=mlp3, dropout=dropout,
                          T=T, len_max_seq=len_max_seq)
        self.times = Fusion(num_classes=num_classes, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                            drop_rate=drop_rate, dim=dim, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                            leff_local_size=leff_local_size, image_size=image_size, leff_with_bn=leff_with_bn,
                            n_segment=n_segment)

        self.decoder = get_decoder(mlp5)

    def forward(self, input):
        img, lmns = input
        outT = self.times(img)
        outL = self.LMN(lmns)

        outTL = torch.cat([outT, outL], dim=1)
        out = self.decoder(outTL)

        return out
