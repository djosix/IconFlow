import random
import colorsys
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as T


class BatchRandomHue:
    def __init__(self):
        self._rgb_to_hxx = np.vectorize(colorsys.rgb_to_hsv)
        self._hxx_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

    def __call__(self, imgs):
        shift = random.random()
        outs = []
        for img in imgs:
            assert img.mode == 'RGB'
            r, g, b = np.rollaxis(np.array(img), -1)
            h, *xx = self._rgb_to_hxx(r, g, b)
            r, g, b = self._hxx_to_rgb(h + shift, *xx)
            data = np.dstack((r, g, b)).clip(0, 255).astype(np.uint8)
            outs.append(Image.fromarray(data))
        return outs


def image_transpose(img, index):
    assert index in range(8)
    if index > 0:
        img = img.transpose(index - 1)
    return img


class RandomTranspose:
    def __call__(self, img):
        return image_transpose(img, random.randrange(8))


class BatchRandomTranspose:
    def __call__(self, imgs):
        method_index = random.randrange(8)
        out_imgs = tuple(image_transpose(img, method_index) for img in imgs)
        return out_imgs

class BatchRandomPerspective(transforms.RandomPerspective):
    def forward(self, imgs):
        if torch.rand(1) < self.p:
            width, height = T._get_image_size(imgs[0])
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)

            if isinstance(self.interpolation, (list, tuple)):
                interpolations = list(self.interpolation)
            else:
                interpolations = [self.interpolation]

            interpolations = interpolations + [interpolations[-1]] * (len(imgs) - len(interpolations))

            if isinstance(self.fill, (list, tuple)):
                fills = list(self.fill)
            else:
                fills = [self.fill]

            fills = fills + [fills[-1]] * (len(imgs) - len(fills))

        
            return [
				T.perspective(img, startpoints, endpoints, interpolation, fill)
				for img, interpolation, fill in zip(imgs, interpolations, fills)
			]
        return imgs


class BatchRandomResizedCrop:
    def __init__(self, size, scale, ratio, interpolations):
        self.size = size  # (int, int)
        self.scale = scale  # (float, float)
        self.ratio = ratio  # (float, float)
        # (interpolation,) or (interpolation, ...)
        self.interpolations = interpolations

    def __call__(self, imgs):
        if len(self.interpolations) == 1:
            interpolations = (self.interpolations,) * len(imgs)
        else:
            interpolations = self.interpolations

        i, j, h, w = transforms.RandomResizedCrop.get_params(
            imgs[0], self.scale, self.ratio)
        out_imgs = []
        for img, interpolation in zip(imgs, interpolations):
            out_imgs.append(T.resized_crop(
                img, i, j, h, w, self.size, interpolation))

        return out_imgs


class TupleTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        out_imgs = []
        for img, transform in zip(imgs, self.transforms):
            out_imgs.append(transform(img))
        return out_imgs
