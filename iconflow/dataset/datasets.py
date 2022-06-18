import os
import glob
import random
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as T

from ..utils.style import load_style_lists
from ..utils.dataset import compute_style_image_distance_matrix

from .transforms import (
    BatchRandomResizedCrop,
    BatchRandomHue,
    BatchRandomTranspose,
    RandomTranspose,
    TupleTransform
)


class IconContourDataset(Dataset):
    def __init__(
        self,
        root,
        image_size,
        random_crop=False,
        random_transpose=False,
        random_color=False,
        split=(0, 1),
        normalize=True,
        as_pil_image=False,
    ):

        root = os.path.expanduser(root)
        img_dir = os.path.join(root, str(image_size), 'img')
        contour_dir = os.path.join(root, str(image_size), 'contour')

        def get_key(path):
            return os.path.splitext(os.path.basename(path))[0]
        img_paths = {get_key(path): path for path in glob.glob(
            os.path.join(img_dir, '*.png'))}
        contour_paths = {get_key(path): path for path in glob.glob(
            os.path.join(contour_dir, '*.png'))}

        assert set(img_paths.keys()) == set(contour_paths.keys())
        keys = list(img_paths.keys())
        assert len(keys) > 0
        keys.sort()
        random.Random(1337).shuffle(keys)
        keys = keys[int(len(keys)*split[0]):int(len(keys)*split[1])]
        self.keys = keys

        self.img_paths = {key: img_paths[key] for key in keys}
        self.contour_paths = {key: contour_paths[key] for key in keys}

        self.batch_resized_crop = BatchRandomResizedCrop(
            (image_size, image_size), (0.8, 1.0), (1.0, 1.0),
            (T.InterpolationMode.BICUBIC, T.InterpolationMode.BICUBIC)
        ) if random_crop else TupleTransform([
            transforms.Resize((image_size, image_size),
                              T.InterpolationMode.BICUBIC),
            transforms.Resize((image_size, image_size),
                              T.InterpolationMode.BICUBIC),
        ])
        self.batch_transpose = random_transpose and BatchRandomTranspose()
        self.batch_shift_hue = random_color and BatchRandomHue()
        self.img_or_contour_to_tensor = transforms.ToTensor()
        self.img_or_contour_normalize = normalize and transforms.Normalize(0.5, 1.0)
        self.as_pil_image = as_pil_image

    def get_icon(self, index):
        return Image.open(self.img_paths[self.keys[index]]).copy()
    
    def get_contour(self, index):
        return ImageOps.invert(Image.open(self.contour_paths[self.keys[index]])).copy()

    def __getitem__(self, index):
        img = self.get_icon(index)
        contour = self.get_contour(index)

        img, contour = self.batch_resized_crop((img, contour))
        if self.batch_transpose:
            img, contour = self.batch_transpose((img, contour))
        if self.batch_shift_hue:
            img = self.batch_shift_hue([img])[0]
        
        if self.as_pil_image:
            return img, contour

        img, contour = map(self.img_or_contour_to_tensor, (img, contour))
        if self.img_or_contour_normalize:
            img, contour = map(self.img_or_contour_normalize, (img, contour))

        return img, contour

    def __len__(self):
        return len(self.keys)


class StylePaletteDataset(Dataset):
    def __init__(self,
                 root,
                 image_size,
                 style_info_dir,
                 max_samples=1000,
                 normalize=True,
                 num_workers=32):
        self.image_size = image_size
        self.dataset = IconContourDataset(root, image_size, normalize=normalize)
        self.max_samples = max_samples

        cis_pkl_path = os.path.join(style_info_dir, 'ColorImageScale.pkl')
        assert os.path.exists(cis_pkl_path)
        

        style_info = load_style_lists(cis_pkl_path)
        self.style_names = style_info['name_list']
        self.style_to_cmb = style_info['name_to_cmb']
        self.style_to_pos = style_info['name_to_pos']
        
        # cis_dm_npz_path = os.path.join(style_info_dir, 'ColorImageScale_DistanceMatrix.npz')
        # assert os.path.exists(cis_dm_npz_path)
        # dismat = np.load(cis_dm_npz_path)['dismat'].T
        image_paths = [self.dataset.img_paths[key] for key in self.dataset.keys]
        dismat = compute_style_image_distance_matrix(style_info, image_paths, num_workers).T
        
        assert dismat.shape[0] == len(self.style_names), repr(dismat.shape)
        assert dismat.shape[1] == len(image_paths), repr(dismat.shape)
        
        sorted_dismat = np.sort(dismat, 1)
        threshold = sorted_dismat[:, max_samples].min()
        self.style_refs = {}
        for style_name, dis in zip(self.style_names, dismat):
            labels = np.where(dis < threshold)[0].tolist()
            labels = [labels[i] for i in dis[labels].argsort()]
            self.style_refs[style_name] = labels[:max_samples]
            
        print('reference counts:', list(map(len, self.style_refs.values())))
        print('minimum reference count:', min(map(len, self.style_refs.values())))
        for style_name in self.style_refs:
            remaining_count = max_samples - len(self.style_refs[style_name])
            self.style_refs[style_name] = self.style_refs[style_name] + [None] * remaining_count
        
        self.random_resized_crop = transforms.RandomResizedCrop(
            (image_size, image_size), (0.8, 1.0), (1.0, 1.0), T.InterpolationMode.BICUBIC)
        self.random_transpose = RandomTranspose()
        self.to_tensor = transforms.ToTensor()
        self.normalize = normalize and transforms.Normalize(0.5, 1.0)
    
    @property
    def condition_size(self):
        return 2
    
    def position_to_condition(self, position, perturb=0.0):
        # map (-3, +3) to (-1, +1)
        position = torch.FloatTensor(position) / 3
        # add noise to make it more dense
        if perturb > 0.0:
            scale = torch.FloatTensor([0.1, 0.05]) * perturb
            noise = torch.randn_like(position) * scale
            position = position + noise
        return position

    def __len__(self):
        return len(self.style_names)
    
    def random_style_ref(self, style_name):
        refs = self.style_refs[style_name]
        idx = random.choice(refs)
        if idx is None:
            ref = self.random_pseudo_ref(style_name)
        else:
            ref = self.dataset.get_icon(idx)
            ref = self.random_resized_crop(ref)
            ref = self.random_transpose(ref)
        ref = self.to_tensor(ref)
        if self.normalize:
            ref = self.normalize(ref)
        return ref
    
    def random_pseudo_ref(self, style_name):
        cmb = self.style_to_cmb[style_name]
        image_size = self.image_size
        
        img = Image.new('RGB', (image_size, image_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        for i in range(1):
            size = int(image_size * 0.9)
            p = int(image_size * 0.1)

            colors = list(map(tuple, cmb))
            random.shuffle(colors)

            for color in colors:
                size = random.randint(int(size * 0.6), int(size * 0.8))
                free = image_size - size - 2 * p
                x, y = map(random.randrange, (free, free))
                x, y = x + p, y + p

                ex, ey = x + size, y + size

                r = random.randrange(4)
                if r < 2:
                    p = random.randrange(int(size * 0.1))
                    if r == 0:
                        x, ex = x + p, ex - p
                    elif r == 1:
                        y, ey = y + p, ey - p
                if r % 2 == 0:
                    draw.ellipse([(x, y), (ex, ey)], fill=color)
                else:
                    draw.rectangle((x, y, ex, ey), fill=color)
        
        return img
    
    def __getitem__(self, index):
        style_name = self.style_names[index]
        condition = self.position_to_condition(self.style_to_pos[style_name], 0.1)
        reference = self.random_style_ref(style_name)
        return reference, condition


class IconContourDownscaleDataset(Dataset):
    def __init__(self, root, image_size, down_size,
                 random_crop=False, random_transpose=False, random_color=False,
                 split=(0, 1), normalize=True):
        root = os.path.expanduser(root)
        self.down_size = down_size

        img_dir = os.path.join(root, str(image_size), 'img')
        contour_dir = os.path.join(root, str(image_size), 'contour')

        def get_key(path): return os.path.splitext(os.path.basename(path))[0]

        img_paths = {get_key(path): path for path in glob.glob(
            os.path.join(img_dir, '*.png'))}
        contour_paths = {get_key(path): path for path in glob.glob(
            os.path.join(contour_dir, '*.png'))}

        assert set(img_paths.keys()) == set(contour_paths.keys())
        keys = list(img_paths.keys())
        assert len(keys) > 0
        keys.sort()
        random.Random(1337).shuffle(keys)
        keys = keys[int(len(keys)*split[0]):int(len(keys)*split[1])]
        self.keys = keys

        self.img_paths = {key: img_paths[key] for key in keys}
        self.contour_paths = {key: contour_paths[key] for key in keys}

        self.batch_resized_crop = BatchRandomResizedCrop(
            (image_size, image_size), (0.8, 1.0), (1.0, 1.0),
            (T.InterpolationMode.BICUBIC, T.InterpolationMode.BICUBIC)
        ) if random_crop else TupleTransform([
            transforms.Resize((image_size, image_size),
                              T.InterpolationMode.BICUBIC),
            transforms.Resize((image_size, image_size),
                              T.InterpolationMode.BICUBIC),
        ])
        self.batch_transpose = random_transpose and BatchRandomTranspose()
        self.batch_shift_hue = random_color and BatchRandomHue()
        self.img_or_contour_to_tensor = transforms.ToTensor()
        self.img_or_contour_normalize = normalize and transforms.Normalize(0.5, 1.0)

    def __getitem__(self, index):
        key = self.keys[index]

        img_path = self.img_paths[key]
        contour_path = self.contour_paths[key]

        img: Image.Image = Image.open(img_path).copy()
        contour: Image.Image = ImageOps.invert(Image.open(contour_path).copy())

        img, contour = self.batch_resized_crop((img, contour))
        if self.batch_transpose:
            img, contour = self.batch_transpose((img, contour))
        if self.batch_shift_hue:
            img = self.batch_shift_hue([img])[0]
        
        original_img: Image.Image = img
        original_contour: Image.Image = contour
        img = img.resize((self.down_size, self.down_size), Image.BICUBIC)
        contour = contour.resize((self.down_size, self.down_size), Image.BICUBIC)

        img, contour, original_img, original_contour = \
            map(self.img_or_contour_to_tensor, (img, contour, original_img, original_contour))
        
        if self.img_or_contour_normalize:
            img, contour, original_img, original_contour = \
                map(self.img_or_contour_normalize, (img, contour, original_img, original_contour))
            
        return img, contour, original_img, original_contour


    def __len__(self):
        return len(self.keys)
