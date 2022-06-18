'''
Copyright (c) 2022 Yuankui Lee
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import torch
import torchvision.transforms.functional as T
from PIL import Image
from typing import List, Tuple
from sklearn.cluster import KMeans

from iconflow.model import (
    ReferenceBasedColorizer,
    get_flow,
    Upsampler128to512,
)
from iconflow.model.styleflow.cnf import SequentialFlow


class IconFlow:
    colorizer: ReferenceBasedColorizer
    flow: SequentialFlow
    up: Upsampler128to512

    def __init__(self, device='cuda', output_dir='output'):
        self.device = torch.device(device)
        print('device:', self.device)

        self.output_dir = output_dir
        
        self.net = ReferenceBasedColorizer().to(self.device).eval()
        self.flow = get_flow().to(self.device).eval()
        self.up = Upsampler128to512().to(self.device).eval()

        checkpoint_path = os.path.join(self.output_dir, 'checkpoint.pt')
        print('loading weights from', checkpoint_path)
        self.net.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['net'])

        checkpoint_path = os.path.join(self.output_dir, 'flow', 'checkpoint.pt')
        print('loading weights from', checkpoint_path)
        self.flow.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['flow'])

        checkpoint_path = os.path.join(self.output_dir, 'up_512', 'checkpoint.pt')
        print('loading weights from', checkpoint_path)
        self.up.load_state_dict(torch.load(checkpoint_path, map_location=self.device)['up'])


    @torch.no_grad()
    def sample_noises(self, n: int, temperture: float):
        
        return (
            torch.randn(512, self.net.style_dim).to(self.device) * temperture,
            n
        )

    @torch.no_grad()
    def get_styles(self, noises: torch.FloatTensor, location: Tuple[int, int]):

        noises, n = noises
        
        locs = torch.stack([torch.FloatTensor(location)] * noises.shape[0]).to(self.device)
        styles = self.flow.forward(noises, locs, reverse=True)

        cluster_centers = KMeans(n).fit(styles.cpu().numpy()).cluster_centers_
        cluster_centers = torch.FloatTensor(cluster_centers).to(styles.device)
        styles = styles[((styles[:, None] - cluster_centers[None]) ** 2).sum(-1).argmin(0)]
        
        return styles

    @torch.no_grad()
    def get_embeddings(self, *sketches: List[Image.Image]):
        
        sketches = torch.stack([T.to_tensor(sketch) - 0.5 for sketch in sketches]).to(self.device)
        embeddings = self.net.encode_content(sketches)
        
        return embeddings

    @torch.no_grad()
    def decode(self, embeddings: torch.FloatTensor, styles: torch.FloatTensor):
        
        # Expand to same batch size
        if embeddings.shape[0] == 1 and styles.shape[0] > 1:
            embeddings = embeddings.expand(styles.shape[0], *([-1] * (embeddings.dim() - 1)))
        if styles.shape[0] == 1 and embeddings.shape[0] > 1:
            styles = styles.expand(embeddings.shape[0], *([-1] * (styles.dim() - 1)))
            
        images = self.net.decode(embeddings, styles)
        images = torch.clamp(images + 0.5, 0.0, 1.0)
        
        return [T.to_pil_image(image) for image in images]

    @torch.no_grad()
    def upsample(self, lowResResult: Image.Image, highResSketch: Image.Image):
        lowResResult = torch.stack([T.to_tensor(lowResResult)]).to(self.device) - 0.5
        highResSketch = torch.stack([T.to_tensor(highResSketch)]).to(self.device) - 0.5
        highResResult = self.up(lowResResult, highResSketch)[0] + 0.5
        return T.to_pil_image(highResResult.cpu())

