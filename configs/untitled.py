import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizationLayer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        num_groups = 32
        num_groups = min(num_channels, num_groups)
        num_groups = max(1, num_channels // num_groups)
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.norm(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = NormalizationLayer(in_channels)
        self.norm2 = NormalizationLayer(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.temb_proj = nn.Linear(out_channels * 4, out_channels)
        self.y_proj = nn.Linear(out_channels * 4, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb, y):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        
        # Add in timestep embedding
        h += self.temb_proj(F.silu(temb))[:, :, None, None]
        
        # Add in class embedding
        if y is not None:
            h += self.y_proj(y)[:, :, None, None]
        
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

class AttnBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = NormalizationLayer(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.permute(0, 2, 3, 1).reshape(B, H*W, C)
        qkv = self.qkv(h).reshape(B, H*W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-1, -2)) * (C // self.num_heads) ** -0.5
        attn = F.softmax(attn, dim=-1)
        h = torch.matmul(attn, v)
        h = h.transpose(1, 2).reshape(B, H, W, C)
        h = self.proj_out(h)
        return x + h.permute(0, 3, 1, 2)

class UNet(nn.Module):
    def __init__(self,
                 n_channel=3,
                 N=128,
                 n_T=1000,
                 schedule_conditioning=False,
                 s_dim=16,
                 ch=128,
                 num_classes=10,
                 ch_mult=(1, 2, 2, 2),
                 num_res_blocks=2,
                 attn_resolutions=(16,),
                 dropout=0.1,
                 num_heads=1
                ):
        super().__init__()

        if schedule_conditioning:
            in_channels = n_channel + n_channel * s_dim

            self.S_embed = nn.Embedding(n_T, s_dim)
            self.S_embed.weight.data = torch.stack(
                [torch.sin(torch.arange(n_T) * 3.1415 * 2**i) for i in range(s_dim // 2)] + 
                [torch.cos(torch.arange(n_T) * 3.1415 * 2**i) for i in range(s_dim // 2)]
            ).T
        else:
            in_channels= n_channel
        out_channels = in_channels
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_classes = num_classes
        self.N = N
        self.n_T = n_T

        # Time embedding
        time_embed_dim = ch * 4
        self.time_embed = nn.Sequential(
            nn.Linear(ch, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Class embedding
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        else:
            self.class_embed = None

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        in_ch = ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            out_ch = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
                if in_ch in attn_resolutions:
                    block.append(AttnBlock(in_ch, num_heads))
            if i_level != self.num_resolutions - 1:
                block.append(nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1))
            self.down_blocks.append(block)

        # Middle
        self.mid_block1 = ResnetBlock(in_ch, in_ch, dropout)
        self.mid_attn = AttnBlock(in_ch, num_heads)
        self.mid_block2 = ResnetBlock(in_ch, in_ch, dropout)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            out_ch = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_ch + out_ch, out_ch, dropout))
                in_ch = out_ch
                if in_ch in attn_resolutions:
                    block.append(AttnBlock(in_ch, num_heads))
            if i_level != 0:
                block.append(nn.Upsample(scale_factor=2, mode='nearest'))
            self.up_blocks.append(block)

        self.norm_out = NormalizationLayer(in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, x, t, y=None):
        x_onehot = F.one_hot(x.long(), num_classes=self.N).permute(0, 3, 1, 2).float()

        # Time embedding
        temb = get_timestep_embedding(t, self.ch)
        
        t = t.float().reshape(-1, 1) * 1000 / self.n_T
        emb_dim = self.ch//2
        temb = [torch.sin(t * 10000**(-i/(emb_dim-1))) for i in range(emb_dim)] + [
            torch.cos(t * * 10000**(-i/(emb_dim-1))) for i in range(emb_dim)
        ]
        temb = torch.cat(temb, dim=1).to(x.device)
        temb = self.time_embed(temb)

        # Class embedding
        if y is not None and self.num_classes > 0:
            yemb = self.class_embed(y)
        else:
            yemb = None
        
        # Downsampling
        h = self.conv_in(x)
        hs = [h]
        for i_level in range(self.num_resolutions):
            for block in self.down_blocks[i_level]:
                if isinstance(block, ResnetBlock):
                    h = block(h, temb, yemb)
                else:
                    h = block(h)
                hs.append(h)

        # Middle
        h = self.mid_block1(h, temb, yemb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb, yemb)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for block in self.up_blocks[i_level]:
                if isinstance(block, ResnetBlock):
                    h = block(torch.cat([h, hs.pop()], dim=1), temb, yemb)
                elif isinstance(block, AttnBlock):
                    h = block(h)
                else:
                    h = block(h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        # Reshape output and add to x_onehot
        B, C, H, W = x_onehot.shape
        h = h.view(B, C, self.N, H, W)
        return x_onehot + h
