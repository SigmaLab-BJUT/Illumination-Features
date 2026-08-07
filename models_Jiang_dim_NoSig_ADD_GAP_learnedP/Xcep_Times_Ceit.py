import torch
import torch.nn as nn
from .Ceit_QKV import CeIT

from .Xception import SCNet


class Fusion(nn.Module):
    # 通道数
    def __init__(self, num_classes=2, depth=12, num_heads=8, mlp_ratio=4., drop_rate=0.,
                 attn_drop_rate=0.1, drop_path_rate=0., dim=64, image_size=7,
                 leff_local_size=3, leff_with_bn=True, n_segment=15):
        super(Fusion, self).__init__()

        self.Xcep = SCNet(num_classes=num_classes, n_segment=n_segment)

        self.Times = CeIT(
            num_classes=num_classes,
            depth=depth,
            dim=dim,
            image_size=image_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            leff_local_size=leff_local_size,
            leff_with_bn=leff_with_bn,
            num_frames=n_segment
            )

    def forward(self, input):
        outT, cls = self.Xcep(input)
        out = self.Times(outT, cls)
        return out
