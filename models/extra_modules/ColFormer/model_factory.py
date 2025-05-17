import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
try:
    from typing import List
    from efficientnet_pytorch.model import MemoryEfficientSwish
    from timm.models.layers import DropPath
except:
    pass


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class PatchEmbedding(nn.Module):

    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        self.conv1 = BasicBlock(in_channels, out_channels // 2, 3, 2)
        self.conv2 = BasicBlock(out_channels // 2, out_channels, 3, 2)
        self.conv3 = BasicBlock(out_channels, out_channels, 3, 1)
        self.conv4 = BasicBlock(out_channels, out_channels, 3, 1)
        self.conv5 = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        self.layernorm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.layernorm(x)


class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
                            nn.Conv2d(dim, dim, 1, 1, 0),
                            MemoryEfficientSwish(),
                            nn.Conv2d(dim, dim, 1, 1, 0)
                            #nn.Identity()
                         )
    def forward(self, x):
        return self.act_block(x)


class EfficientAttention(nn.Module):

    def __init__(self, dim, num_heads, group_split: List[int], kernel_sizes: List[int], window_size=7,
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        # projs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3 * self.dim_head * group_head, 3 * self.dim_head * group_head, kernel_size,
                                   1, kernel_size // 2, groups=3 * self.dim_head * group_head))
            act_blocks.append(AttnMap(self.dim_head * group_head))
            qkvs.append(nn.Conv2d(dim, 3 * group_head * self.dim_head, 1, 1, 0, bias=qkv_bias))
            # projs.append(nn.Linear(group_head*self.dim_head, group_head*self.dim_head, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1] * self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1] * self.dim_head * 2, 1, 1, 0, bias=qkv_bias)
            # self.global_proj = nn.Linear(group_split[-1]*self.dim_head, group_split[-1]*self.dim_head, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size != 1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        qkv = to_qkv(x)  # (b (3 m d) h w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()  # (3 b (m d) h w)
        q, k, v = qkv  # (b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v)  # (b (m d) h w)
        return res

    def low_fre_attention(self, x: torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()

        q = to_q(x).reshape(b, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()  # (b m (h w) d)
        kv = avgpool(x)  # (b c h w)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h * w) // (self.window_size ** 2)).permute(1, 0, 2, 4,
                                                                                                 3).contiguous()  # (2 b m (H W) d)
        k, v = kv  # (b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2)  # (b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v  # (b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))


class ConvFFN(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, stride,
                 out_channels, act_layer=nn.GELU, drop_out=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.act = act_layer()
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride,
                                kernel_size // 2, groups=hidden_channels)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)
        self.drop = nn.Dropout(drop_out)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientBlock(nn.Module):

    # def __init__(self, dim, out_dim, num_heads=3, group_split=[1,1,1], kernel_sizes=[7,5], window_size=7,
    #              mlp_kernel_size=7, mlp_ratio=4, stride=2, attn_drop=0., mlp_drop=0., qkv_bias=True,
    #              drop_path=0.):
    def __init__(self, dim, out_dim, num_heads, group_split: List[int], kernel_sizes: List[int], window_size: int,
                 mlp_kernel_size: int, mlp_ratio: int, stride: int, attn_drop=0., mlp_drop=0., qkv_bias=True,
                 drop_path=0.):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = EfficientAttention(dim, num_heads, group_split, kernel_sizes, window_size,
                                       attn_drop, mlp_drop, qkv_bias)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.GroupNorm(1, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.stride = stride
        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                                nn.Conv2d(dim, dim, mlp_kernel_size, 2, mlp_kernel_size//2),
                                nn.BatchNorm2d(dim),
                                nn.Conv2d(dim, out_dim, 1, 1, 0),
                            )
        self.mlp = ConvFFN(dim, mlp_hidden_dim, mlp_kernel_size, stride, out_dim,
                        drop_out=mlp_drop)
    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.downsample(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CloLayer(nn.Module):

    def __init__(self, depth, dim, out_dim, num_heads, group_split: List[int], kernel_sizes: List[int],
                 window_size: int, mlp_kernel_size: int, mlp_ratio: int, attn_drop=0,
                 mlp_drop=0., qkv_bias=True, drop_paths=[0., 0.], downsample=True, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
                        [
                            EfficientBlock(dim, dim, num_heads, group_split, kernel_sizes, window_size,
                                mlp_kernel_size, mlp_ratio, 1, attn_drop, mlp_drop, qkv_bias, drop_paths[i])
                                for i in range(depth-1)
                        ]
                    )
        if downsample is True:
            self.blocks.append(EfficientBlock(dim, out_dim, num_heads, group_split, kernel_sizes, window_size,
                            mlp_kernel_size, mlp_ratio, 2, attn_drop, mlp_drop, qkv_bias, drop_paths[-1]))
        else:
            self.blocks.append(EfficientBlock(dim, out_dim, num_heads, group_split, kernel_sizes, window_size,
                            mlp_kernel_size, mlp_ratio, 1, attn_drop, mlp_drop, qkv_bias, drop_paths[-1]))

    def forward(self, x: torch.Tensor):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class CloFormer(nn.Module):

    def __init__(self, in_chans, num_classes, embed_dims: List[int], depths: List[int],
                 num_heads: List[int], group_splits: List[List[int]], kernel_sizes: List[List[int]],
                 window_sizes: List[int], mlp_kernel_sizes: List[int], mlp_ratios: List[int],
                 attn_drop=0., mlp_drop=0., qkv_bias=True, drop_path_rate=0.1, use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.mlp_ratios = mlp_ratios
        self.patch_embed = PatchEmbedding(in_chans, embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer != self.num_layers-1:
                layer = CloLayer(depths[i_layer], embed_dims[i_layer], embed_dims[i_layer+1], num_heads[i_layer],
                            group_splits[i_layer], kernel_sizes[i_layer], window_sizes[i_layer],
                            mlp_kernel_sizes[i_layer], mlp_ratios[i_layer], attn_drop, mlp_drop,
                            qkv_bias, dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], True, use_checkpoint)
            else:
                layer = CloLayer(depths[i_layer], embed_dims[i_layer], embed_dims[i_layer], num_heads[i_layer],
                            group_splits[i_layer], kernel_sizes[i_layer], window_sizes[i_layer],
                            mlp_kernel_sizes[i_layer], mlp_ratios[i_layer], attn_drop, mlp_drop,
                            qkv_bias, dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], False, use_checkpoint)
            self.layers.append(layer)

        # self.norm = nn.GroupNorm(1, embed_dims[-1])
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes>0 else nn.Identity()

    def forward_feature(self, x):
        '''
        x: (b 3 h w)
        '''

        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(self.norm(x))
        return x.flatten(1)

    # def forward(self, x):
    #     x = self.forward_feature(x)
    #     return self.head(x)
    def forward(self, x):
        '''
        x: (b 3 h w)
        '''
        _, _, h, w = x.size()
        scale = [4, 8, 16, 32]
        out = []
        shape = [x.shape[2] // i for i in scale]
        x = self.patch_embed(x)
        for layer in self.layers:
            x1 = x
            x = layer(x)
            if x1.shape[2] != x.shape[2] and x1.shape[2] in shape:
                out.append(x1)
        out.append(x)

        return out


def cloformer_xxs():
    return CloFormer(3, 1000, [32, 64, 128, 256], [2, 2, 6, 2], [4, 4, 8, 16], [[3, 1], [2, 2], [4, 4], [4, 12]],
            [[3], [5], [7], [9]], [8, 4, 2, 1], [5, 5, 5, 5], [4, 4, 4, 4])

def cloformer_global():
    return CloFormer(3, 1000, [32, 64, 128, 256], [2, 2, 6, 2], [4, 4, 8, 16], [[4], [4], [8], [16]],
            [[], [], [], []], [8, 4, 2, 1], [5, 5, 5, 5], [4, 4, 4, 4])

def cloformer_xs():
    return CloFormer(3, 1000, [48, 96, 160, 352], [2, 2, 6, 2], [3, 6, 10, 22], [[2, 1], [3, 3], [5, 5], [7, 15]],
            [[3], [5], [7], [9]], [8, 4, 2, 1], [5, 5, 5, 5], [4, 4, 4, 4])

def cloformer_s():
    return CloFormer(3, 1000, [64, 128, 224, 448], [2, 2, 6, 2], [4, 8, 14, 28], [[3, 1], [4, 4], [7, 7], [7, 21]],
            [[3], [5], [7], [9]], [8, 4, 2, 1], [5, 5, 5, 5], [4, 4, 4, 4])

def cloformer_no_global():
    return CloFormer(3, 1000, [32, 64, 128, 256], [2, 2, 6, 2], [4, 4, 8, 16], [[4, 0], [4, 0], [8, 0], [16, 0]],
            [[3], [5], [7], [9]], [8, 4, 2, 1], [5, 5, 5, 5], [4, 4, 4, 4])


if __name__ == "__main__":
    import torch


    # Model
    # model = EfficientBlock(96, 192)
    model = cloformer_xs().to("cuda")
    for i in model(torch.zeros(2, 3, 32, 32).to("cuda")):
        print(i.shape)
