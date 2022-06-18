import numpy as np
from PIL import Image
from skimage.color import rgb2lab

LAB_HIST_BINS = (8, 8, 8)
LAB_HIST_RANGE = [(0, 100), (-127, 128), (-128, 127)]

def get_image_hist(image):
    pixels = np.array(image)
    non_white_pixels = pixels[np.any(pixels < [255, 255, 255], axis=-1), :]
    non_white_pixels = rgb2lab(non_white_pixels[None])[0]
    return np.histogramdd(non_white_pixels, bins=LAB_HIST_BINS, range=LAB_HIST_RANGE)[0]

def get_colors_hist(colors, ratio=None):
    colors = rgb2lab(colors[None]).reshape((-1, 3))
    if ratio is not None:
        colors = np.stack([color for c, r in zip(colors, ratio) for color in [c] * r])
    return np.histogramdd(colors, bins=LAB_HIST_BINS, range=LAB_HIST_RANGE)[0]

def draw_white(size):
    return Image.new('RGB', size, (255, 255, 255))

def draw_style(colors, size=40):
    bg = Image.new('RGB', (size * 3, size), (255, 255, 255))
    for i, color in enumerate(map(tuple, colors)):
        bg.paste(Image.new('RGB', (size, size), color), (i * size, 0))
    return bg
