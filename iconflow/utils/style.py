import pickle
from PIL import Image, ImageDraw

def load_style_lists(path):
    with open(path, 'rb') as f:
        scale = pickle.load(f)

    style_list = []
    for group, styles in scale.items():
        group = group.replace(' ', '_')
        for style, pos, cmb in styles:
            style = style.replace(' ', '_')
            name = f'{group}/{style}'
            style_list.append((name, (pos, cmb)))
    style_list.sort(key=lambda t: t[0])

    style_to_pos = {}
    style_to_cmb = {}

    style_name_list = []
    style_pos_list = []
    style_cmb_list = []

    for name, (pos, cmb) in style_list:
        style_name_list.append(name)
        style_pos_list.append(pos)
        style_cmb_list.append(cmb)
        style_to_pos[name] = pos
        style_to_cmb[name] = cmb
    
    return dict(name_list=style_name_list,
                pos_list=style_pos_list, cmb_list=style_cmb_list,
                name_to_pos=style_to_pos, name_to_cmb=style_to_cmb)

def _add_text(image, text, pos=(4, 4), color=None):
    if color is None:
        color = {'L': (0,), 'RGB': (0, 0, 0)}[image.mode]
    image = image.copy()
    ImageDraw.Draw(image).text(pos, text, color)
    return image

def _draw_style(colors, size=40):
    bg = Image.new('RGB', (size * 3, size), (255, 255, 255))
    for i, color in enumerate(map(tuple, colors)):
        bg.paste(Image.new('RGB', (size, size), color), (i * size, 0))
    return bg

def get_style_image(combination, text=None) -> Image.Image:
    img = Image.new('RGB', (128,)*2, (255,)*3)
    img.paste(_draw_style(combination).resize((128, 128-40)), (0, 40))
    if text is not None:
        img = _add_text(img, text)
    return img
