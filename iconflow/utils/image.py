import torch
import torchvision.transforms.functional as T

def from_image(image): # [0, 1]
    return T.to_tensor(image)

def to_image(tensor): # [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    return T.to_pil_image(tensor)

def from_image1(image): # [-0.5, +0.5]
    return from_image(image) - 0.5

def to_image1(tensor): # [-0.5, +0.5]
    return to_image(tensor + 0.5)
