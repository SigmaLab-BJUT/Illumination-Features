import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class PixelSetEncoder(nn.Module):
    def __init__(self, input_dim, mlp0=None, mlp1=None, mlp2=None, with_extra=True, extra_size=9):

        super(PixelSetEncoder, self).__init__()
        if mlp2 is None:
            mlp2 = [73, 64]
        if mlp1 is None:
            mlp1 = [3, 1]
        if mlp0 is None:
            mlp0 = [44, 64, 64]
        self.input_dim = input_dim
        self.mlp0_dim = copy.deepcopy(mlp0)
        self.mlp1_dim = copy.deepcopy(mlp1)
        self.mlp2_dim = copy.deepcopy(mlp2)

        self.with_extra = with_extra
        self.extra_size = extra_size

        self.output_dim = input_dim if len(self.mlp2_dim) == 0 else self.mlp2_dim[-1]

        assert (input_dim == mlp0[0])

        layers = []
        for i in range(len(self.mlp0_dim) - 1):
            layers.append(linlayer(self.mlp0_dim[i], self.mlp0_dim[i + 1]))
        self.mlp0 = nn.Sequential(*layers)

        layers = []
        for i in range(len(self.mlp1_dim) - 1):
            layers.append(linLayer(self.mlp1_dim[i], self.mlp1_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)

        # MLP after pooling
        layers = []
        for i in range(len(self.mlp2_dim) - 1):
            layers.append(nn.Linear(self.mlp2_dim[i], self.mlp2_dim[i + 1]))
            layers.append(nn.BatchNorm1d(self.mlp2_dim[i + 1]))
            if i < len(self.mlp2_dim) - 2:
                layers.append(nn.ReLU())
        self.mlp2 = nn.Sequential(*layers)

    def forward(self, input):
        global batch, temp
        extra, out = input
        if len(out.shape) == 4:
            # Combine batch and temporal dimensions in case of sequential input
            reshape_needed = True
            batch, temp = out.shape[:2]

            out = out.view(batch * temp, *out.shape[2:])
            if self.with_extra:
                extra = extra.view(batch * temp, -1)
        else:
            reshape_needed = False

        out = self.mlp0(out)
        out = self.mlp1(out)

        out = torch.squeeze(out)

        if self.with_extra:
            out = torch.cat([out, extra], dim=1)
        out = self.mlp2(out)

        if reshape_needed:
            out = out.view(batch, temp, -1)
        return out


class linlayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(linlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, input):
        # out = input.permute((0, 2, 1))  # to channel last
        out = self.lin(input)  # BT C N
        # print(out.shape)
        out = out.permute((0, 2, 1))  # BT N C
        out = self.bn(out)
        out = F.relu(out)
        out = out.permute((0, 2, 1))  # BT C N

        return out


class linLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(linLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, input):
        out = input.permute((0, 2, 1))  # BT N C
        out = self.lin(out)
        # print(out.shape)
        out = out.permute((0, 2, 1))  # BT C N
        out = self.bn(out)
        out = F.relu(out)
        # out = out.permute((0, 2, 1))  # BT C N

        return out
