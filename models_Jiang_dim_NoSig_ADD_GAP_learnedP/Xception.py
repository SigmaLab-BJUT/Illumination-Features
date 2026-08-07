import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np
from einops import rearrange

model_urls = {
    'scnet50_v1d': 'https://backseason.oss-cn-beijing.aliyuncs.com/scnet/scnet50_v1d-4109d1e1.pth',
}


class TIM_Module(nn.Module):
    def __init__(self, in_channels, reduction=16, n_segment=8, return_attn=False):
        """The Temporal Inconsistency Module (TIM).

        Args:
            in_channels (int): Input channel number.
            reduction (int, optional): Channel compression ratio r in the split operation.. Defaults to 16.
            n_segment (int, optional): Number of input frames.. Defaults to 8.
            return_attn (bool, optional): Whether to return the attention part. Defaults to False.

        """
        super(TIM_Module, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.return_attn = return_attn
        self.n_segment = n_segment
        self.reduced_channels = self.in_channels // self.reduction
        self.para1 = torch.nn.Parameter(torch.tensor([0.25]))
        self.para2 = torch.nn.Parameter(torch.tensor([0.25]))
        self.para3 = torch.nn.Parameter(torch.tensor([0.25]))
        self.para4 = torch.nn.Parameter(torch.tensor([0.25]))
        self.para5 = torch.nn.Parameter(torch.tensor([1.]))
        self.para6 = torch.nn.Parameter(torch.tensor([0.33]))
        self.para7 = torch.nn.Parameter(torch.tensor([0.33]))
        self.para8 = torch.nn.Parameter(torch.tensor([0.33]))
        self.avg_pool_ht1 = nn.AvgPool2d((1, 2))
        self.avg_pool_ht2 = nn.AvgPool2d((1, 4))
        self.avg_pool_tw1 = nn.AvgPool2d((2, 1))
        self.avg_pool_tw2 = nn.AvgPool2d((4, 1))
        self.conv1 = nn.Conv2d(self.in_channels, self.reduced_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.reduced_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.conv_ht = nn.Conv2d(self.reduced_channels, self.reduced_channels,
                                 kernel_size=(3, 1), padding=(1, 0), groups=self.reduced_channels, bias=False)
        self.conv_tw = nn.Conv2d(self.reduced_channels, self.reduced_channels,
                                 kernel_size=(1, 3), padding=(0, 1), groups=self.reduced_channels, bias=False)
        # HTIE in two directions
        self.tw_conv1 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.ht_conv1 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.tw_conv2 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.ht_conv2 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.tw_conv3 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.ht_conv3 = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(self.reduced_channels),
        )
        self.ht_up_conv = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels)
        )
        self.tw_up_conv = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.in_channels)
        )

        self.sigmoid = nn.Sigmoid()

        self.GAP = nn.AdaptiveAvgPool2d((1, 1))
        self.FC1 = nn.Linear(self.n_segment, self.n_segment // 2)
        self.BN1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU()
        self.FC2 = nn.Linear(self.n_segment // 2, self.n_segment)
        self.BN2 = nn.BatchNorm1d(self.in_channels)

    def feat_ht(self, feat):
        """The H-T branch in the TIM module.

        Args:
            feat (torch.tensor): Input feature with shape [n, t, c, h, w] (c is in_channels // reduction)

        """
        n, t, c, h, w = feat.size()
        # [n, t, c, h, w] -> [n, w, c, h, t] -> [nw, c, h, t]
        feat_h = feat.permute(0, 4, 2, 3, 1).contiguous().view(-1, c, h, t)
        # [nw, c, h, t-1]
        feat_h_fwd, _ = feat_h.split([self.n_segment - 1, 1], dim=3)
        feat_h_conv = self.conv_ht(feat_h)
        _, feat_h_conv_fwd = feat_h_conv.split([1, self.n_segment - 1], dim=3)

        diff_feat_fwd = feat_h_conv_fwd - feat_h_fwd
        diff_feat_fwd = F.pad(diff_feat_fwd, [0, 1], value=0)  # [nw, c, h, t]
        # HTIE, down_up branch
        diff_feat_fwd1 = self.avg_pool_ht1(diff_feat_fwd)
        diff_feat_fwd1 = self.ht_conv1(diff_feat_fwd1)
        diff_feat_fwd1 = F.interpolate(diff_feat_fwd1, diff_feat_fwd.size()[2:])

        diff_feat_fwd2 = self.avg_pool_ht2(diff_feat_fwd)
        diff_feat_fwd2 = self.ht_conv2(diff_feat_fwd2)
        diff_feat_fwd2 = F.interpolate(diff_feat_fwd2, diff_feat_fwd.size()[2:])

        # HTIE, direct conv branch
        diff_feat_fwd3 = self.ht_conv3(diff_feat_fwd)
        diff_feat_fwd3 = F.interpolate(diff_feat_fwd3, diff_feat_fwd.size()[2:])

        feat_ht_out = self.ht_up_conv(self.para1 * diff_feat_fwd + self.para2 * diff_feat_fwd1 + self.para3 * diff_feat_fwd2 + self.para4 * diff_feat_fwd3)
        feat_ht_out = self.sigmoid(feat_ht_out) - 0.5
        feat_ht_out = feat_ht_out.view(n, w, self.in_channels, h, t).permute(0, 4, 2, 3, 1).contiguous()
        feat_ht_out = feat_ht_out.view(-1, self.in_channels, h, w)

        return feat_ht_out

    def feat_tw(self, feat):
        """The T-W branch in the TIM module.

        Args:
            feat (torch.tensor): Input feature with shape [n, t, c, h, w] (c is in_channels // reduction)
        """
        n, t, c, h, w = feat.size()

        feat_w = feat.permute(0, 3, 2, 1, 4).contiguous().view(-1, c, t, w)
        # [nh, c, t-1, w]
        feat_w_fwd, _ = feat_w.split([self.n_segment - 1, 1], dim=2)
        feat_w_conv = self.conv_tw(feat_w)
        _, feat_w_conv_fwd = feat_w_conv.split([1, self.n_segment - 1], dim=2)

        diff_feat_fwd = feat_w_conv_fwd - feat_w_fwd
        diff_feat_fwd = F.pad(diff_feat_fwd, [0, 0, 0, 1], value=0)  # [nh, c, t, w]
        # VTIE, down_up branch
        diff_feat_fwd1 = self.avg_pool_tw1(diff_feat_fwd)
        diff_feat_fwd1 = self.tw_conv1(diff_feat_fwd1)
        diff_feat_fwd1 = F.interpolate(diff_feat_fwd1, diff_feat_fwd.size()[2:])

        diff_feat_fwd2 = self.avg_pool_tw2(diff_feat_fwd)
        diff_feat_fwd2 = self.tw_conv2(diff_feat_fwd2)
        diff_feat_fwd2 = F.interpolate(diff_feat_fwd2, diff_feat_fwd.size()[2:])

        # VTIE, direct conv branch
        diff_feat_fwd3 = self.tw_conv3(diff_feat_fwd)
        diff_feat_fwd3 = F.interpolate(diff_feat_fwd3, diff_feat_fwd.size()[2:])

        feat_tw_out = self.tw_up_conv(self.para5 * diff_feat_fwd + self.para6 * diff_feat_fwd1 + self.para7 * diff_feat_fwd2 + self.para8 * diff_feat_fwd3)
        feat_tw_out = self.sigmoid(feat_tw_out) - 0.5

        feat_tw_out = feat_tw_out.view(n, h, self.in_channels, t, w).permute(0, 3, 2, 1, 4).contiguous()
        # print(feat_tw_out.shape)
        feat_tw_out = feat_tw_out.view(-1, self.in_channels, h, w)

        return feat_tw_out

    def GAP_T(self, feat):
        """The T-W branch in the TIM module.

        Args:
            feat (torch.tensor): Input feature with shape [n, t, c, h, w] (c is in_channels // reduction)
        """
        n, t, c, h, w = feat.size()
        feat = feat.permute(0, 2, 1, 3, 4).contiguous().view(-1, t, h, w)
        feat = self.GAP(feat)
        feat = feat.contiguous().view(n, c, t)

        feat = self.FC1(feat)
        feat = self.relu(feat)
        feat = self.BN1(feat)

        feat = self.FC2(feat)
        feat = self.BN2(feat)

        feat = self.sigmoid(feat) - 0.5
        feat = feat.contiguous().view(-1, c, 1, 1)
        return feat

    def forward(self, x):
        """
        Args:
            x (torch.tensor): Input with shape [nt, c, h, w]
        """
        bottleneck = self.conv1(x)
        bottleneck = self.bn1(bottleneck)
        # bottleneck = self.relu(bottleneck)
        bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])

        F_h = self.feat_ht(bottleneck)
        F_w = self.feat_tw(bottleneck)

        out = 0.5 * (F_h + F_w)
        GAP_x = x.view((-1, self.n_segment) + x.size()[1:])
        out_GAP = self.GAP_T(GAP_x)
        y2 = out_GAP * out * x + x

        return y2


class SeparableConv2d(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, **kwargs
    ) -> None:
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, groups=in_channels, bias=False, **kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
                                   bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pointwise(out)
        return out


class XceptionBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            relu_first: bool,
            grow_first: bool,
            repeat_times: int,
            n_segment: int
    ) -> None:
        super(XceptionBlock, self).__init__()
        rep = []
        self.TM = TIM_Module(in_channels=in_channels, reduction=16, n_segment=n_segment)
        self.relu = nn.ReLU(True)

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(stride, stride),
                                  padding=(0, 0), bias=False)
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        mid_channels = in_channels
        if grow_first:
            rep.append(nn.ReLU(True))
            rep.append(SeparableConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            rep.append(nn.BatchNorm2d(out_channels))
            mid_channels = out_channels

        for _ in range(repeat_times - 1):
            rep.append(nn.ReLU(True))
            rep.append(SeparableConv2d(mid_channels, mid_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            rep.append(nn.BatchNorm2d(mid_channels))

        if not grow_first:
            rep.append(nn.ReLU(True))
            rep.append(SeparableConv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            rep.append(nn.BatchNorm2d(out_channels))

        if not relu_first:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(False)

        if stride != 1:
            rep.append(nn.MaxPool2d((3, 3), (stride, stride), (1, 1)))

        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.TM(x)
        if self.skip is not None:
            identity = self.skip(out)
            identity = self.skipbn(identity)
        else:
            identity = out

        out = self.rep(out)
        out = torch.add(out, identity)

        return out


class SCNet(nn.Module):
    def __init__(self, stem_width=32, norm_layer=nn.BatchNorm2d, n_segment=8, num_classes=2, dropout=0.3, add_softmax=False):
        super(SCNet, self).__init__()
        conv_layer = nn.Conv2d
        self.dropout = dropout
        self.add_softmax = add_softmax
        self.n_segment = n_segment
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            conv_layer(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(stem_width),
            nn.ReLU(inplace=True),
            conv_layer(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(stem_width),
            nn.ReLU(inplace=True),
            conv_layer(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.inplanes = stem_width * 2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # inchannel = stem_width * 2
        self.block1 = XceptionBlock(64, 128, 2, False, True, 2, n_segment=n_segment)
        self.block2 = XceptionBlock(128, 256, 2, True, True, 2, n_segment=n_segment)
        self.block3 = XceptionBlock(256, 728, 2, True, True, 2, n_segment=n_segment)

        self.block4 = XceptionBlock(728, 728, 1, True, True, 3, n_segment=n_segment)
        self.block5 = XceptionBlock(728, 728, 1, True, True, 3, n_segment=n_segment)
        self.block6 = XceptionBlock(728, 728, 1, True, True, 3, n_segment=n_segment)
        self.block7 = XceptionBlock(728, 728, 1, True, True, 3, n_segment=n_segment)

        if self.add_softmax:
            self.softmax_layer = nn.Softmax(dim=1)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = F.dropout(x, self.dropout, training=self.training)
        # print(x.shape)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        return x

    def logits(self, features):
        x = self.avgpool(features)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x

    def forward(self, input):
        n, t, c, h, w = input.size()
        input = input.view(-1, c, h, w)
        x = self.features(input)
        out = rearrange(x, '(b f) c h w -> b f c h w', f=self.n_segment)
        # print(x.shape)
        cls = self.logits(x)
        #
        cls = cls.view(n, self.n_segment, -1)
        cls = cls.mean(1, keepdim=True)
        # if self.add_softmax:
        #     out = self.softmax_layer(out)
        return out, cls
