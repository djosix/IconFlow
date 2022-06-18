import torch
import torch.nn as nn

from .rescae import get_residual_block


class Upsampler128to256(nn.Module):
    def __init__(
        self,
        image_channels = 3,
        contour_channels = 1,
        dim = 64,
    ):
        super().__init__()
        
        self.x_up = nn.Sequential(
            get_residual_block(image_channels, dim),
            nn.UpsamplingBilinear2d(256),
        )
        
        self.c_res = get_residual_block(contour_channels, dim)
        
        self.r_res = nn.Sequential(
            get_residual_block(dim + dim, dim),
            get_residual_block(dim, dim),
            nn.Conv2d(dim, image_channels, 3, padding='same', padding_mode='replicate')
        )
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.shape[1:] == (3, 128, 128)
        assert c.dim() == 4 and c.shape[1:] == (1, 256, 256)
        assert x.shape[0] == c.shape[0]
        
        return torch.tanh(self.r_res(torch.cat([self.x_up(x), self.c_res(c)], 1))) / 2


class Upsampler128to512(nn.Module):
    def __init__(
        self,
        image_channels = 3,
        contour_channels = 1,
        dim = 64,
    ):
        super().__init__()
        
        self.x_up = nn.Sequential(
            get_residual_block(image_channels, dim),
            nn.UpsamplingBilinear2d(256),
            get_residual_block(dim, dim),
            nn.UpsamplingBilinear2d(512)
        )
        
        self.c_res = get_residual_block(contour_channels, dim)
        
        self.r_res = nn.Sequential(
            get_residual_block(dim + dim, dim),
            get_residual_block(dim, dim),
            get_residual_block(dim, dim),
            get_residual_block(dim, dim),
            nn.Conv2d(dim, image_channels, 3, padding='same', padding_mode='replicate')
        )
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4 and x.shape[1:] == (3, 128, 128)
        assert c.dim() == 4 and c.shape[1:] == (1, 512, 512)
        assert x.shape[0] == c.shape[0]
        
        return torch.tanh(self.r_res(torch.cat([self.x_up(x), self.c_res(c)], 1))) / 2
