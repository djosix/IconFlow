from typing import Callable

import torch
import torch.nn as nn

import einops

from .rescae import get_residual_unet
from .resnet import get_resnet_by_depth


class NormConv2d(nn.Conv2d):
    def __init__(self, channels, kernel_size=2):
        super().__init__(1, channels, kernel_size,
                         padding='same',
                         padding_mode='replicate',
                         bias=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B = input.shape[0]
        input = einops.rearrange(input, 'B C H W -> (B C) 1 H W')
        
        weight = self.weight - einops.reduce(self.weight, 'K 1 H W -> K 1 1 1', 'mean')
        output = self._conv_forward(input, weight, self.bias)
        output = einops.rearrange(output, '(B C) K H W -> B C K H W', B=B)

        output = torch.abs(output)
        output = einops.reduce(output, 'B C K H W -> B K H W', 'mean')

        return output


class Lambda(nn.Module):
    def __init__(self, func: Callable):
        self.func = func
        super().__init__()
    
    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class ReferenceBasedColorizer(nn.Module):
        def __init__(
            self,
            
            contour_channels = 1,
            image_channels = 3,
            
            embedding_dim = 16,
            style_dim = 48,
            
            content_encoder_arch = 'M51',
            content_extractor_arch = 'S31',
            norm_conv_channels = 16,
            
            resnet_depth = 50,
            decoder_width = 32,
            decoder_depth = 4,
        ):
            self.contour_channels = contour_channels
            self.image_channels = image_channels
            self.embedding_dim = embedding_dim
            self.style_dim = style_dim
            self.content_encoder_arch = content_encoder_arch
            self.content_extractor_arch = content_extractor_arch
            self.norm_conv_channels = norm_conv_channels
            self.resnet_depth = resnet_depth
            self.decoder_width = decoder_width
            self.decoder_depth = decoder_depth

            super().__init__()
            
            self.content_encoder = nn.Sequential(
                get_residual_unet(contour_channels, embedding_dim, content_encoder_arch),
                nn.Tanh(),
            )
            
            self.content_extractor = nn.Sequential(
                NormConv2d(norm_conv_channels),
                get_residual_unet(norm_conv_channels, contour_channels, content_extractor_arch),
                nn.Tanh(),
                Lambda(lambda tensor: tensor / 2) # output [-0.5, 0.5]
            )
            
            self.style_encoder = nn.Sequential(
                get_resnet_by_depth(resnet_depth, num_classes=style_dim, in_channels=image_channels),
                nn.Tanh()
            )
            
            def _decoder_layers():
                last_width = embedding_dim + style_dim
                for _ in range(decoder_depth - 1):
                    yield nn.Linear(last_width, decoder_width)
                    yield nn.LayerNorm(decoder_width)
                    yield nn.ReLU()
                    last_width = decoder_width
                yield nn.Linear(last_width, image_channels)
                yield nn.Tanh()
                yield Lambda(lambda tensor: tensor / 2) # output [-0.5, 0.5]
                
            self.decoder = nn.Sequential(*_decoder_layers())
        
        def extract_content(self, x: torch.Tensor) -> torch.Tensor:
            '''
            U-Net: image (B 3 H W) -> contour (B 1 H W)
            '''
            return self.content_extractor(x)
        
        def encode_content(self, c: torch.Tensor) -> torch.Tensor:
            '''
            U-Net: contour (B 1 H W) -> embeddings (B E H W)
            '''
            return self.content_encoder(c)
        
        def encode_style(self, x: torch.Tensor) -> torch.Tensor:
            '''
            ResNet: image (B 3 H W) -> style (B S)
            '''
            return self.style_encoder(x)
        
        def decode(self, e: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            '''
            PixelwiseDecoder: embeddings (B E H W) + style (B S) -> output (B 3 H W)
            '''
            B, _, H, W = e.shape
            
            s = einops.repeat(s, 'B S -> B S H W', B=B, H=H, W=W)
            h = torch.cat([e, s], 1)
            h = einops.rearrange(h, 'B D H W -> (B H W) D')
            
            r = self.decoder(h)
            r = einops.rearrange(r, '(B H W) C -> B C H W', B=B, H=H, W=W)
            
            return r
        
        def forward(self, c, x):
            '''
            ReferenceBasedColorizer: contour (B 1 H W) + image (B 3 H W) -> output (B 3 H W)
            '''
            e = self.encode_content(c)
            s = self.encode_style(x)
            r = self.decode(e, s)
            
            return r
