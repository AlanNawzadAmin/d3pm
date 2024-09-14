import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalizationLayer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        num_groups = 32
        num_groups = min(num_channels, num_groups)
        assert num_channels % num_groups == 0
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.norm(x)

def pad_image(x, target_size):
    """Preprocess image to target size with padding."""
    _, _, h, w = x.shape
    # target_size = math.ceil(max([h, w]) // 2 ** sec_power) * 2 ** sec_power
    if h == target_size and w == target_size:
        return x
    
    pad_h = max(target_size - h, 0)
    pad_w = max(target_size - w, 0)
    padding = (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2)
    return F.pad(x, padding, mode='constant', value=0)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, dropout, cond=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = NormalizationLayer(in_channels)
        self.norm2 = NormalizationLayer(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.emb_dim = emb_dim
        if emb_dim>0:
            self.temb_proj = nn.Linear(emb_dim, out_channels)
        if cond:
            self.y_proj = nn.Linear(emb_dim, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, temb, y):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add in timestep embedding
        if self.emb_dim >0:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        
        # Add in class embedding
        if y is not None:
            h = h + self.y_proj(y)[:, :, None, None]
        
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)

class AttnBlock(nn.Module):
    def __init__(self, channels, width, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.height = width
        self.width = width
        self.head_dim = channels // self.num_heads
        self.norm = NormalizationLayer(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x):
        B = x.shape[0]
        h = self.norm(x).view(B, self.channels, self.height*self.width).transpose(1, 2)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, -1)
        q = q.view(B, self.height*self.width, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, self.height*self.width, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, self.height*self.width, self.num_heads, self.head_dim).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).view(B, self.height, self.width, self.channels)
        h = self.proj_out(h)
        return x + h.transpose(2, 3).transpose(1, 2)

class UNet(nn.Module):
    def __init__(self,
                 n_channel=3,
                 N=256,
                 s_lengthscale=50,
                 time_lengthscale=1,
                 schedule_conditioning=False,
                 s_dim=16,
                 ch=128,
                 time_embed_dim=512,
                 num_classes=1,
                 ch_mult=(1, 2, 2, 2),
                 num_res_blocks=2,
                 attn_resolutions=(1,),
                 dropout=0.1,
                 num_heads=1,
                 width=32,
                 logistic_pars=True,
                 **kwargs
                ):
        super().__init__()

        if schedule_conditioning:
            in_channels = n_channel + n_channel * s_dim

            s = torch.arange(1000).reshape(-1, 1) * 1000 / s_lengthscale
            emb_dim = s_dim//2
            semb = 10000**(-torch.arange(emb_dim)/(emb_dim-1))
            semb = torch.cat([torch.sin(s * semb), torch.cos(s * semb)], dim=1)
        
            self.S_embed = nn.Embedding(1000, s_dim)
            self.S_embed.weight.data = semb
        else:
            in_channels= n_channel
        self.N = N
        self.n_channel = n_channel
        out_channels = n_channel * N 
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.num_classes = num_classes
        self.time_lengthscale = time_lengthscale
        self.width = width
        self.logistic_pars = logistic_pars

        # Time embedding
        self.time_embed_dim = time_embed_dim
        if self.time_embed_dim > 0:
            self.time_embed = nn.Sequential(
                nn.Linear(ch, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        # Class embedding
        self.cond = num_classes > 1 
        if self.cond:
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
                block.append(ResnetBlock(in_ch, out_ch, time_embed_dim, dropout, self.cond))
                in_ch = out_ch
                if i_level in attn_resolutions:
                    block.append(AttnBlock(in_ch, self.width//(2**i_level), num_heads))
                else:
                    block.append(nn.Identity())
            if i_level != self.num_resolutions - 1:
                block.append(nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1))
            else:
                block.append(nn.Identity())
            self.down_blocks.append(block)

        # Middle
        self.mid_block1 = ResnetBlock(in_ch, in_ch, time_embed_dim, dropout, self.cond)
        self.mid_attn = AttnBlock(in_ch, self.width//(2**i_level), num_heads)
        self.mid_block2 = ResnetBlock(in_ch, in_ch, time_embed_dim, dropout, self.cond)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        prev_out = ch * ch_mult[-1]
        for i_level in list(reversed(range(self.num_resolutions))):
            block = nn.ModuleList()
            out_ch = ch * ch_mult[i_level]
            for j in range(self.num_res_blocks + 1):
                in_ch = ch * ((1,)+ch_mult)[i_level + 1 - (j==self.num_res_blocks)]
                block.append(ResnetBlock(in_ch + prev_out, out_ch, time_embed_dim, dropout, self.cond))
                prev_out = out_ch
                if i_level in attn_resolutions:
                    block.append(AttnBlock(prev_out, self.width//(2**i_level), num_heads))
                else:
                    block.append(nn.Identity())
            if i_level != 0:
                block.append(nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                           nn.Conv2d(prev_out, prev_out, 3, padding=1)))
            else:
                block.append(nn.Identity())
            self.up_blocks.append(block)

        self.norm_out = NormalizationLayer(prev_out)
        self.conv_out = nn.Conv2d(prev_out, out_channels, 3, padding=1)

    @torch.compile(fullgraph=True, dynamic=False)
    def unet_main(self, x, temb, yemb):
        # Downsampling
        h = self.conv_in(x)
        hs = [h]
        for blocks in self.down_blocks:
            for i in range(self.num_res_blocks):
                h = blocks[2 * i](h, temb, yemb)
                h = blocks[2 * i + 1](h)
                hs.append(h)
            h = blocks[-1](h)
            hs.append(h)

        # Middle
        h = self.mid_block1(h, temb, yemb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb, yemb)

        # Upsampling
        len_hs = (self.num_res_blocks + 1) * self.num_resolutions # skip last addition
        for j, blocks in enumerate(self.up_blocks):
            for i in range(self.num_res_blocks+1):
                index = (self.num_res_blocks+1) * j + i
                old_h = hs[len_hs-(index+1)]
                h = blocks[2 * i](torch.cat([h, old_h], dim=1), temb, yemb)
                h = blocks[2 * i + 1](h)
            h = blocks[-1](h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return h
    
    def forward(self, x, t, y=None, S=None):
        x_onehot = F.one_hot(x.long(), num_classes=self.N).float()
        x = pad_image(x, self.width)
        x = x.float()
        x[..., :self.n_channel, :, :] = (2 * x[..., :self.n_channel, :, :] / self.N) - 1.0
    
        if S is not None:
            s_embed = self.S_embed(S.permute(0,2,3,1))
            s_embed = s_embed.reshape(*s_embed.shape[:-2], -1).permute(0,3,1,2)

            x = torch.cat([x, s_embed], dim=1)

        # Time embedding   
        if self.time_embed_dim > 0:
            t = t.float().reshape(-1, 1) * 1000 / self.time_lengthscale
            emb_dim = self.ch//2
            temb = 10000**(-torch.arange(emb_dim, device=t.device)/(emb_dim-1))
            temb = torch.cat([torch.sin(t * temb), torch.cos(t * temb)], dim=1)
            temb = self.time_embed(temb)
        else:
            temb = None

        # Class embedding
        if y is not None and self.num_classes > 1:
            yemb = self.class_embed(y)
        else:
            yemb = None
        
        # Reshape output and add to x_onehot
        B, C, H, W, _ = x_onehot.shape
        h = self.unet_main(x, temb, yemb)
        h = h[:, :, :H, :W].reshape(B, C, self.N, H, W).permute((0, 1, 3, 4, 2))
        return h + self.logistic_pars * x_onehot

########################


class KingmaUNet(nn.Module):
    def __init__(self,
                 n_channel=3,
                 N=256,
                 s_lengthscale=50,
                 time_lengthscale=1,
                 schedule_conditioning=False,
                 s_dim=16,
                 ch=128,
                 time_embed_dim=128,
                 num_classes=1,
                 n_layers=32,
                 inc_attn=False,
                 dropout=0.1,
                 num_heads=1,
                 n_transformers=1,
                 width=32,
                 logistic_pars=True,
                 **kwargs
                ):
        super().__init__()

        if schedule_conditioning:
            in_channels = ch * n_channel + n_channel * s_dim

            s = torch.arange(1000).reshape(-1, 1) * 1000 / s_lengthscale
            emb_dim = s_dim//2
            semb = 10000**(-torch.arange(emb_dim)/(emb_dim-1))
            semb = torch.cat([torch.sin(s * semb), torch.cos(s * semb)], dim=1)
        
            self.S_embed = nn.Embedding(1000, s_dim)
            self.S_embed.weight.data = semb
        else:
            in_channels = ch * n_channel
        self.N = N
        self.n_channel = n_channel
        out_channels = n_channel * N 
        self.ch = ch
        self.n_layers = n_layers
        self.inc_attn = inc_attn
        self.num_classes = num_classes
        self.time_lengthscale = time_lengthscale
        self.width = width
        self.logistic_pars = logistic_pars

        self.x_embed = nn.Embedding(N, ch)
        # Time embedding
        self.time_embed_dim = time_embed_dim
        if self.time_embed_dim > 0:
            self.time_embed = nn.Sequential(
                nn.Linear(ch, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        # Class embedding
        self.cond = num_classes > 1 
        if self.cond:
            self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        else:
            self.class_embed = None

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        for i_level in range(self.n_layers):
            block = nn.ModuleList()
            block.append(ResnetBlock(ch, ch, time_embed_dim, dropout, self.cond))
            if self.inc_attn:
                block.append(AttnBlock(ch, width, num_heads))
            else:
                block.append(nn.Identity())
            self.down_blocks.append(block)

        # Middle
        self.mid_block1 = ResnetBlock(ch, ch, time_embed_dim, dropout, self.cond)
        self.mid_attn = nn.Sequential(*[AttnBlock(ch, width, num_heads)
                                        for i in range(n_transformers)])
        self.mid_block2 = ResnetBlock(ch, ch, time_embed_dim, dropout, self.cond)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i_level in range(self.n_layers+1):
            block = nn.ModuleList()
            block.append(ResnetBlock(2 * ch, ch, time_embed_dim, dropout, self.cond))
            if self.inc_attn:
                block.append(AttnBlock(ch, width, num_heads))
            else:
                block.append(nn.Identity())
            self.up_blocks.append(block)

        self.norm_out = NormalizationLayer(ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    @torch.compile(fullgraph=True, dynamic=False)
    def flat_unet(self, x, temb, yemb):
        # Downsampling
        h = self.conv_in(x)
        hs = [h]
        for blocks in self.down_blocks:
            h = blocks[0](h, temb, yemb)
            h = blocks[1](h)
            hs.append(h)

        # Middle
        h = self.mid_block1(h, temb, yemb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb, yemb)

        # Upsampling
        for i, blocks in enumerate(self.up_blocks):
            h = blocks[0](torch.cat([h, hs[self.n_layers-(i+1)]], dim=1), temb, yemb)
            h = blocks[1](h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return h
    
    def forward(self, x, t, y=None, S=None):
        x_onehot = F.one_hot(x.long(), num_classes=self.N).float()
        x = self.x_embed(x.permute(0,2,3,1))
        x = x.reshape(*x.shape[:-2], -1).permute(0,3,1,2)
    
        if S is not None:
            s_embed = self.S_embed(S.permute(0,2,3,1))
            s_embed = s_embed.reshape(*s_embed.shape[:-2], -1).permute(0,3,1,2)

            x = torch.cat([x, s_embed], dim=1)

        # Time embedding        
        if self.time_embed_dim > 0:
            t = t.float().reshape(-1, 1) * 1000 / self.time_lengthscale
            emb_dim = self.ch//2
            temb = 10000**(-torch.arange(emb_dim, device=t.device)/(emb_dim-1))
            temb = torch.cat([torch.sin(t * temb), torch.cos(t * temb)], dim=1)
            temb = self.time_embed(temb)
        else:
            temb = None
        
        # Class embedding
        if y is not None and self.num_classes > 1:
            yemb = self.class_embed(y)
        else:
            yemb = None
        
        # Reshape output and add to x_onehot
        B, C, H, W, _ = x_onehot.shape
        h = self.flat_unet(x, temb, yemb)
        h = h[:, :, :H, :W].reshape(B, C, self.N, H, W).permute((0, 1, 3, 4, 2))
        return h + self.logistic_pars * x_onehot

########################

class SimpleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.norm(self.conv2(x))
        return F.relu(x)

class SimpleUNet(nn.Module):
    def __init__(self, n_channel, N, n_T, schedule_conditioning=False, s_dim=16):
        super().__init__()

        if schedule_conditioning:
            in_channel = n_channel + n_channel * s_dim

            self.S_embed = nn.Embedding(n_T, s_dim)
            self.S_embed.weight.data = torch.stack(
                [torch.sin(torch.arange(n_T) * 3.1415 * 2**i) for i in range(s_dim // 2)] + 
                [torch.cos(torch.arange(n_T) * 3.1415 * 2**i) for i in range(s_dim // 2)]
            ).T
        else:
            in_channel = n_channel

        self.N = N
        self.n_channel = n_channel
        out_channel = n_channel * N 
        
        # Encoder (downsampling)
        self.enc1 = SimpleConvBlock(in_channel, 64)
        self.enc2 = SimpleConvBlock(64, 128)
        self.enc3 = SimpleConvBlock(128, 256)
        
        # Decoder (upsampling)
        self.dec3 = SimpleConvBlock(256 + 128, 128)  # 256 from enc3, 128 from enc2
        self.dec2 = SimpleConvBlock(128 + 64, 64)    # 128 from dec3, 64 from enc1
        self.dec1 = SimpleConvBlock(64, 32)
        
        self.final = nn.Conv2d(32, out_channel, 1)

    def forward(self, x, t, cond, S=None):
        x = x.float()
        x[..., :self.n_channel, :, :] = (2 * x[..., :self.n_channel, :, :] / self.N) - 1.0
    
        if S is not None:
            s_embed = self.S_embed(S.permute(0,2,3,1))
            s_embed = s_embed.reshape(*s_embed.shape[:-2], -1).permute(0,3,1,2)

            x = torch.cat([x, s_embed], dim=1)
        
        # Encoder
        e1 = self.enc1(x.float())
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([F.interpolate(e3, scale_factor=2, mode='nearest'), e2], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode='nearest'), e1], dim=1))
        d1 = self.dec1(d2)

        d0 = self.final(d1)
        
        return d0.reshape(d0.shape[0], -1, self.N, *x.shape[2:]).transpose(2, -1)

