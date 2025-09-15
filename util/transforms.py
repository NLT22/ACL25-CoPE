from PIL import Image
import numpy as np
import random
import torchvision.transforms.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class Cutout:
    def __init__(self, max_masksize=50, mask_color=(0, 0, 0)):
        self.max_masksize = max_masksize
        self.mask_color = mask_color

    def apply(self, image):
        np_image = np.array(image)
        h, w = np_image.shape[0:2]
        mask_size = random.randint(0, self.max_masksize)
        y = random.randint(0, max(0, h - mask_size))
        x = random.randint(0, max(0, w - mask_size))

        if len(np_image.shape) == 2:
            mask_color = int(np.mean(self.mask_color))
        else:
            mask_color = self.mask_color

        np_image[y:y + mask_size, x:x + mask_size] = mask_color
        return Image.fromarray(np_image)


class ModifyHSV:
    def __init__(self, max_hgain=0.1, max_sgain=0.7, max_vgain=0.7):
        self.max_hgain = max_hgain
        self.max_sgain = max_sgain
        self.max_vgain = max_vgain

    def apply(self, image):
        hgain = random.uniform(0, self.max_hgain)
        sgain = random.uniform(0, self.max_sgain)
        vgain = random.uniform(0, self.max_vgain)

        image_hsv = image.convert('HSV')
        np_img_hsv = np.array(image_hsv, dtype=np.float32)
        h, s, v = np_img_hsv[..., 0], np_img_hsv[..., 1], np_img_hsv[..., 2]

        h = (h + hgain * 255) % 255
        s = np.clip(s * (1 + sgain), 0, 255)
        v = np.clip(v * (1 + vgain), 0, 255)

        np_img_hsv[..., 0], np_img_hsv[..., 1], np_img_hsv[..., 2] = h, s, v
        np_img_hsv = np_img_hsv.astype(np.uint8)

        image_hsv_modified = Image.fromarray(np_img_hsv, mode='HSV')
        return image_hsv_modified.convert('RGB')


class RotateImage:
    def __init__(self, max_angle=90, expand=True):
        self.max_angle = max_angle
        self.expand = expand

    def apply(self, image):
        angle = random.uniform(-self.max_angle, self.max_angle)
        return image.rotate(angle, expand=self.expand)


class RandomScaling:
    def __init__(self, scale_range=(0.5, 1.5)):
        self.scale_range = scale_range

    def apply(self, image):
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        original_width, original_height = image.size
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


class AddGaussianNoise:
    def __init__(self, mean=0, std=25):
        self.mean = mean
        self.std = std

    def apply(self, image):
        np_image = np.array(image)
        noise = np.random.normal(self.mean, self.std, np_image.shape).astype(np.float32)
        noisy_image = np_image + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)


class DataAugmentation:
    def __init__(self, methods=None, p=0.2, **kwargs):
        available_methods = {
            'cutout': Cutout(kwargs.get('max_masksize', 50), tuple(kwargs.get('mask_color', (0, 0, 0)))),
            'modify_hsv': ModifyHSV(kwargs.get('max_hgain', 0.1), kwargs.get('max_sgain', 0.7), kwargs.get('max_vgain', 0.7)),
            'rotate': RotateImage(kwargs.get('max_angle', 90), kwargs.get('expand', True)),
            'scale': RandomScaling(kwargs.get('scale_range', (0.5, 1.5))),
            'noise': AddGaussianNoise(kwargs.get('mean', 0), kwargs.get('std', 25))
        }
        if methods == 'all':
            self.methods = list(available_methods.values())
        elif methods:
            self.methods = [available_methods[method] for method in methods if method in available_methods]
        else:
            self.methods = []
        self.p = p

    def apply(self, image):
        for method in self.methods:
            if random.random() < self.p:
                image = method.apply(image)
        return image


def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float = 1.25, dim: int = 224):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
