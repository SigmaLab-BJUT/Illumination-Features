import torch.nn as nn
from torch import einsum
from einops import rearrange, repeat

from .rotary import apply_rot_emb, AxialRotaryEmbedding, RotaryEmbedding

import math
import torch
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_


def exists(val):
    return val is not None


class LocallyEnhancedFeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 kernel_size=3, with_bn=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # pointwise
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        # depthwise
        self.conv2 = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=hidden_features
        )
        # pointwise
        self.conv3 = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()
        # self.drop = nn.Dropout(drop)

        self.with_bn = with_bn
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.bn3 = nn.BatchNorm2d(out_features)

    def forward(self, x, num_frames):
        tokens = rearrange(x, 'b (f n) d -> (b f) n d', f=num_frames)
        b, n, k = tokens.size()
        # print(n)
        x = tokens.reshape(b, int(math.sqrt(n)), int(math.sqrt(n)), k).permute(0, 3, 1, 2)
        if self.with_bn:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act(x)
            x = self.conv3(x)
            x = self.bn3(x)
        else:
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.act(x)
            x = self.conv3(x)

        tokens = x.flatten(2).permute(0, 2, 1) # 从第二维开始展平
        out = rearrange(tokens, '(b f) n d-> b (f n) d ', f=num_frames)
        return out


def attn(q, k, v, mask=None):
    sim = einsum('b i d, b j d -> b i j', q, k)
    if exists(mask):
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class Attention_Times(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            dropout=0.,
            num_frames=15,
            num_patches=49
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.num_frames = num_frames
        self.num_patches = num_patches
        # pointwise
        self.T_Q = nn.Sequential(
            nn.Conv2d(dim, inner_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_dim)
        )
        self.T_K = nn.Sequential(
            nn.Conv2d(dim, inner_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_dim)
        )
        self.T_V = nn.Sequential(
            nn.Conv2d(dim, inner_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_dim)
        )
        self.DWLN_S = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(dim)
        )
        self.DWLN_T = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=7, padding=0, stride=1),
            nn.BatchNorm1d(dim)
        )
        self.S_Q = nn.Sequential(
            nn.Conv2d(dim, inner_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_dim)
        )
        self.S_K = nn.Sequential(
            nn.Conv2d(dim, inner_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_dim)
        )
        self.S_V = nn.Sequential(
            nn.Conv2d(dim, inner_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inner_dim)
        )
        # pointwise

        self.T_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim)
        )
        self.S_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim)
        )

        # self.out = nn.Sequential(
        #     nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(dim)
        # )

    def Times_Att(self, x, einops_from, einops_to, rot_emb=None, **einops_dims):
        h = self.heads
        b, n, c = x.shape
        x_Q = rearrange(x, 'b (f h w) d -> (b f) d h w', f=self.num_frames, h=int(math.sqrt(self.num_patches)),
                        w=int(math.sqrt(self.num_patches)))
        T_q = self.T_Q(x_Q)

        x_KV = rearrange(x, 'b (f n) d  -> (b n) d f', f=self.num_frames)
        x_KV = self.DWLN_T(x_KV)
        x_KV = rearrange(x_KV, '(b h w) d f -> (b f) d h w', h=int(math.sqrt(self.num_patches)),
                         w=int(math.sqrt(self.num_patches)))
        T_k = self.T_K(x_KV)
        T_v = self.T_V(x_KV)

        q = rearrange(T_q, '(b f) d h w -> (b h w) f d', f=self.num_frames)
        k = rearrange(T_k, '(b f) d h w -> (b h w) f d', b=b)
        v = rearrange(T_v, '(b f) d h w -> (b h w) f d', b=b)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q = q * self.scale

        # rearrange across time or space
        # q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q, k, v))

        # add rotary embeddings, if applicable
        # if exists(rot_emb):
        #     q = apply_rot_emb(q, rot_emb)

        # attention
        out = attn(q, k, v, mask=None)

        # merge back time or space
        out = rearrange(out, '(b h) f d -> b f (h d)', h=h)
        # out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # merge back the heads

        # combine heads out
        out = rearrange(out, '(b h w) f d -> (b f) d h w', h=int(math.sqrt(self.num_patches)),
                        w=int(math.sqrt(self.num_patches)))
        out = self.T_out(out)
        out = rearrange(out, '(b f) d h w -> b (f h w) d', f=self.num_frames)
        return out

    def Space_Att(self, x, einops_from, einops_to, rot_emb=None, **einops_dims):
        h = self.heads
        b, n, c = x.shape

        # x_QKV = rearrange(x, 'b n (d h w) -> (b n) d h w', h=1, w=1)
        x_Q = rearrange(x, 'b (f h w) d -> (b f) d h w', f=self.num_frames, h=int(math.sqrt(self.num_patches)),
                        w=int(math.sqrt(self.num_patches)))
        S_q = self.S_Q(x_Q)

        x_KV = rearrange(x, 'b (f h w) d  -> (b f) d h w', f=self.num_frames, h=int(math.sqrt(self.num_patches)),
                         w=int(math.sqrt(self.num_patches)))
        x_KV = self.DWLN_S(x_KV)
        S_k = self.S_K(x_KV)
        S_v = self.S_V(x_KV)

        q = rearrange(S_q, 'b d h w -> b (h w) d')
        k = rearrange(S_k, 'b d h w -> b (h w) d')
        v = rearrange(S_v, 'b d h w -> b (h w) d')
        # v = x
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        q = q * self.scale

        # rearrange across time or space
        # q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q, k, v))

        # add rotary embeddings, if applicable
        # if exists(rot_emb):
        #     q = apply_rot_emb(q, rot_emb)

        # attention
        out = attn(q, k, v, mask=None)

        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # merge back time or space
        # out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # merge back the heads

        # combine heads out
        out = rearrange(out, 'b (h w) d -> b d h w', h=int(math.sqrt(self.num_patches)),
                        w=int(math.sqrt(self.num_patches)))
        out = self.S_out(out)
        out = rearrange(out, '(b f) d h w -> b (f h w) d', f=self.num_frames)
        return out

    def forward(self, x, rot_embT=None, rot_embS=None):

        b, n, c = x.shape
        T_att = self.Times_Att(x, 'b (f n) d', '(b n) f d', rot_emb=rot_embT,
                               n=self.num_patches)
        S_att = self.Space_Att(x, 'b (f n) d', '(b f) n d', rot_emb=rot_embS,
                               f=self.num_frames)

        out = 0.5 * (S_att + T_att)
        # out = rearrange(out, 'b n (d h w) -> (b n) d h w', h=1, w=1)
        # out = self.out(out)
        # out = rearrange(out, '(b n) d h w -> b n (d h w)', n=n)
        return out


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, kernel_size=3, with_bn=True,
                 feedforward_type='leff', dim_head=64, num_patches=16, num_frames=25, frame_rot_emb=None,
                 image_rot_emb=None):
        super().__init__()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.feedforward_type = feedforward_type
        self.frame_rot_emb = frame_rot_emb
        self.image_rot_emb = image_rot_emb

        self.num_patches = num_patches
        self.num_frames = num_frames
        if feedforward_type == 'leff':
            self.timeSpace_attn = Attention_Times(dim, dim_head=dim_head, heads=num_heads, dropout=attn_drop,
                                                  num_frames=num_frames, num_patches=num_patches)
            self.leff = LocallyEnhancedFeedForward(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                kernel_size=kernel_size, with_bn=with_bn,
            )

    def forward(self, x):
        frame_pos_emb = self.frame_rot_emb(self.num_frames, device=x.device)
        image_pos_emb = self.image_rot_emb(self.num_patches, 1, device=x.device)
        x = x + self.drop_path(self.timeSpace_attn(self.norm1(x), rot_embT=frame_pos_emb, rot_embS=image_pos_emb))
        x = x + self.drop_path(self.leff(self.norm2(x), self.num_frames))

        return x


class CeIT(nn.Module):
    def __init__(self,
                 image_size=7,
                 patch_size=1,
                 num_classes=2,
                 embed_dim=728,
                 depth=12,
                 dim=64,
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 leff_local_size=3,
                 leff_with_bn=True,
                 num_frames=15
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.num_frames = num_frames
        self.patch_size = patch_size
        dim_head = dim
        # dim_head = embed_dim // num_heads
        self.frame_rot_emb = RotaryEmbedding(dim_head)
        self.image_rot_emb = AxialRotaryEmbedding(dim_head)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                kernel_size=leff_local_size, with_bn=leff_with_bn, dim_head=dim_head, num_patches=self.num_patches,
                num_frames=num_frames,
                frame_rot_emb=self.frame_rot_emb, image_rot_emb=self.image_rot_emb
            )
            for i in range(depth)])

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward_features(self, x, cls_token):
        b, f, _, h, w, *_, device, p = *x.shape, x.device, self.patch_size
        assert h % p == 0 and w % p == 0, f'height {h} and width {w} of video must be divisible by the patch size {p}'

        # video to patch embeddings
        tokens = rearrange(x, 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1=p, p2=p)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = rearrange(tokens, 'b (f n) d -> b (f d) n', f=f)
        out = self.avgpool(tokens)
        out = out.view(b, f, -1)
        out = out.mean(1, keepdim=False)

        return out

    def forward(self, x, cls_token):
        x = self.forward_features(x, cls_token)

        return x
