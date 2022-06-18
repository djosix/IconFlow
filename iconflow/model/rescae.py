import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock


def get_residual_block(in_planes, planes, **kwargs):
    if in_planes != planes:
        downsample = nn.Conv2d(in_planes, planes, 1, bias=False)
    else:
        downsample = nn.Identity()
        
    return BasicBlock(in_planes, planes, downsample=downsample, **kwargs)


class ResCAE(nn.Module):
    def __init__(
        self,
        in_ch = 3,
        out_ch = 3,
        downs =[(16, 2), (32, 2), (64, 2), (128, 2)],
        ups = [(128, 2), (64, 2), (32, 2), (16, 2)],
        is_unet = True,
        norm_layer = None,
        nonlinearity = None,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d        
        if nonlinearity is None:
            nonlinearity = lambda: nn.ReLU(inplace=True)
        
        if ups is None:
            ups = downs[::-1]
            
        if is_unet:
            assert len(downs) == len(ups)
            
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.downs = downs.copy()
        self.ups = ups.copy()
        self.is_unet = is_unet
        
        ch = downs[0][0]
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1),
            nonlinearity()
        )
        last_ch = ch
        
        self.down_levels = nn.ModuleList()
        self.down = nn.MaxPool2d(kernel_size=2)
        
        for down_ch, n_blocks in downs:
            blocks = []
            for _ in range(n_blocks):
                block = get_residual_block(
                    last_ch, down_ch,
                    norm_layer=norm_layer,
                    nonlinearity=nonlinearity)
                blocks.append(block)
                last_ch = down_ch
            self.down_levels.append(nn.Sequential(*blocks))
            
        self.up_levels = nn.ModuleList()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        for i, (up_ch, n_blocks) in enumerate(ups):
            if is_unet and i > 0:
                last_ch += downs[-i - 1][0]
                
            blocks = []
            for _ in range(n_blocks):
                block = get_residual_block(
                    last_ch, up_ch,
                    norm_layer=norm_layer,
                    nonlinearity=nonlinearity)
                blocks.append(block)
                last_ch = up_ch
            self.up_levels.append(nn.Sequential(*blocks))
        
        self.last_layer = nn.ConvTranspose2d(last_ch, out_ch, 3, padding=1)
    
    def can_forward(self, shape):
        *b, c, h, w = shape
        if not len(b) in (0, 1) or c != self.in_ch:
            return False
        min_len = 2 ** (len(self.downs) - 1)
        if h % min_len != 0 or w % min_len != 0:
            return False
        return True
    
    def forward(self, x):
        conns = []
        
        x = self.first_layer(x)
        
        for level in self.down_levels[:-1]:
            x = level(x)
            if self.is_unet:
                conns.append(x)
            x = self.down(x)
        
        x = self.down_levels[-1](x)
        x = self.up_levels[0](x)
        
        for level in self.up_levels[1:]:
            x = self.up(x)
            if self.is_unet:
                x = torch.cat([x, conns.pop(-1)], 1)
            x = level(x)
        
        x = self.last_layer(x)
        
        return x


def get_residual_unet(in_ch, out_ch, arch):
    
    max_channels = 1024

    def get_config(base_dim, n_levels, block_depth):
        downs = []
        ups = []
        for i in range(n_levels):
            ch = base_dim * 2 ** i
            downs.append((ch, block_depth))
            if i == 0:
                ch = max(ch, out_ch)
            elif i < n_levels - 1:
                ch = max(ch, out_ch - base_dim * 2 ** (i-1))
            ch = min(ch, max_channels)
            ups.insert(0, (ch, block_depth))
        return downs, ups

    assert len(arch) == 3
    size, level, depth = arch
    level, depth = map(int, (level, depth))
    
    base = {'s': 16, 'S': 32, 'M': 64, 'L': 128}[size]
    downs, ups = get_config(base, level, depth)
    
    return ResCAE(in_ch, out_ch, downs, ups, True)
