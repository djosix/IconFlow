import os
import glob
import cv2
import json
import shutil
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from PIL import Image, ImageOps
from skimage.feature import canny as get_canny_feature
from scipy.ndimage.filters import gaussian_filter
from concurrent.futures import ProcessPoolExecutor
from .hist import get_image_hist, get_colors_hist


def _load_image(path):
    raw = Image.open(path)
    if raw.mode in ('P', 'LA'):
        raw = raw.convert('RGBA')
    assert raw.mode in ('RGB', 'RGBA')
    assert raw.size == (512, 512)
    return raw

def _pad_image(img):
    n = int(img.size[0] / 8)
    if img.mode == 'RGBA':
        return ImageOps.expand(img, n, (255, 255, 255, 0))
    elif img.mode == 'RGB':
        return ImageOps.expand(img, n, (255, 255, 255))
    assert False, repr(img.mode)
    
def _get_contour(img):
    x = np.array(img)
    
    canny = 0
    for layer in np.rollaxis(x, -1):
        canny |= get_canny_feature(layer, 0)
    canny = canny.astype(np.uint8) * 255
    
    kernel = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ], dtype=np.uint8)
    
    canny = cv2.dilate(canny, kernel)
    canny = Image.fromarray(canny)
    
    return canny

def _get_pair(path):
    img = _load_image(path)
    img = _pad_image(img)
    contour = _get_contour(img)
    return img.convert('RGB'), contour

def _preprocess_image(args):
    raw_path, resolutions, output_sizes, name = args
    full_img, full_contour = _get_pair(raw_path)
    
    for resolution, output_size in zip(resolutions, output_sizes):
        (full_img
            .resize((output_size, output_size), Image.BICUBIC)
            .save(f'data/{resolution}/img/{name}.png'))
        (full_contour
            .resize((output_size, output_size), Image.BICUBIC)
            .save(f'data/{resolution}/contour/{name}.png'))

    return raw_path

def preprocess_images(
    dataset_dir='./dataset',
    resolutions=[64, 128, 256, 512],
    num_workers=24
):
    assert os.path.isdir(dataset_dir)
    os.chdir(dataset_dir)
    
    assert os.path.isdir('raw')
    assert len(glob.glob(os.path.join('raw', '*.png'))) > 0
    
    resolutions = sorted(list(set(resolutions)))
    assert set(resolutions) <= {64, 128, 256, 512}
    assert len(resolutions) > 0
    print('resolutions:', resolutions)

    assert num_workers > 0 and num_workers <= 64
    print('workers:', num_workers)
    
    for resolution in resolutions:
        shutil.rmtree(f'data/{resolution}/', ignore_errors=True)
        os.makedirs(f'data/{resolution}/img', exist_ok=True)
        os.makedirs(f'data/{resolution}/contour', exist_ok=True)
    
    output_sizes = [
        int(resolution + 2 * resolution / 8)
        for resolution in resolutions
    ]
    
    raw_paths = glob.glob('raw/*.png')
    raw_paths.sort()
    print('number of images:', len(raw_paths))

    with mp.Pool(num_workers) as pool:
        args_list = [
            (raw_path, resolutions, output_sizes, '{:06d}'.format(i))
            for i, raw_path in enumerate(raw_paths)
        ]
        paths_done = pool.imap_unordered(_preprocess_image, args_list)

        key_to_raw = {
            name: os.path.basename(raw_path)
            for raw_path, _, _, name in args_list
        }
        for resolution in resolutions:
            with open(f'data/{resolution}/key_to_raw.json', 'w') as f:
                json.dump(key_to_raw, f)

        with tqdm(paths_done, desc='Preprocessing', total=len(raw_paths)) as pbar:
            for raw_path in pbar:
                pbar.set_postfix_str(raw_path)


def compute_style_image_distance_matrix(style_info, image_paths, num_workers):
    
    print('compute style color histograms')
    ratios = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2), (2, 1, 2), (2, 2, 1)]
    style_hists = np.array([
        [get_colors_hist(colors, ratio) for ratio in ratios]
        for colors in style_info['cmb_list']
    ])
    
    print('compute image color histograms')
    image_names = []
    image_hists = []
    global _get_image_hist_task
    def _get_image_hist_task(path):
        return get_image_hist(Image.open(path).resize((64, 64), Image.NEAREST).convert('RGB'))
    with ProcessPoolExecutor(num_workers) as executor:
        futures = []
        for path in image_paths:
            image_names.append(os.path.basename(path))
            futures.append(executor.submit(_get_image_hist_task, path))
        for future in tqdm(futures):
            image_hists.append(future.result())
    
    def _normalize_hist(hist):
        hist = hist / hist.sum()
        hist = gaussian_filter(hist, 1.0)
        return hist.ravel()
    
    print('normalize histograms for style color')
    normalized_style_hists = np.stack([[_normalize_hist(hist) for hist in hists] for hists in tqdm(style_hists)])
    
    print('normalize histograms for image color')
    normalized_image_hists = np.stack(list(map(_normalize_hist, tqdm(image_hists, desc='images'))))
    
    global _normalized_style_hists
    _normalized_style_hists = normalized_style_hists
    
    global _compute_hist_dis
    def _compute_hist_dis(hist):
        dis = hist[None, None] - normalized_style_hists
        dis = np.sqrt((dis ** 2).sum(-1)).min(-1)
        return dis
    
    print('compute distance matrix')
    dismat = []
    with ProcessPoolExecutor(num_workers) as executor:
        futures = []
        for hist in normalized_image_hists:
            futures.append(executor.submit(_compute_hist_dis, hist))
        for future in tqdm(futures):
            dismat.append(future.result())
    dismat = np.array(dismat)
    
    return dismat

def dump_style_icons(dataset_dir, output_dir, max_dumps_per_style=8, max_samples=1000, num_workers=8):
    from .style import get_style_image
    from ..dataset import StylePaletteDataset
    
    data_dir = os.path.join(dataset_dir, 'data')
    dataset = StylePaletteDataset(data_dir, 128, dataset_dir, max_samples, num_workers=num_workers)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for style_name, icon_indices in dataset.style_refs.items():
        dir_path = os.path.join(output_dir, style_name.replace('/', '-'))
        os.makedirs(dir_path, exist_ok=True)
        
        style_image = get_style_image(dataset.style_to_cmb[style_name], style_name)
        style_image.save(os.path.join(dir_path, '_style.png'))
        
        icon_indices = [index for index in icon_indices if index is not None][:max_dumps_per_style]
        print('dump {} icons to {}'.format(len(icon_indices), dir_path))
        for i, index in enumerate(icon_indices):
            path = os.path.join(dir_path, f'{i:02d}.png')
            dataset.dataset.get_icon(index).save(path)
        
if __name__ == '__main__':
    import fire
    fire.Fire({
        'preprocess_images': preprocess_images,
        # 'preprocess_color_image_scale': preprocess_color_image_scale,
        'dump_style_icons': dump_style_icons,
    })
