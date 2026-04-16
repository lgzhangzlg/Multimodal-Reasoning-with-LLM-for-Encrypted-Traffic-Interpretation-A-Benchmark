import os

from PIL import Image, ImageFile
import torch
import ast
import numpy as np

from ..utils.data_utils import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImagePreprocess:
    def __init__(self, image_processor, data_args={}):
        self.image_aspect_ratio = getattr(data_args, 'image_aspect_ratio', None)
        self.image_processor = image_processor
        self.image_grid_pinpoints = getattr(data_args, 'image_grid_pinpoints', None)

    def __call__(self, image):
        # ★★★ 如果已经是 tensor 或 np.ndarray，直接返回（只做基本整理）★★★
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if isinstance(image, torch.Tensor):
            # HWC -> CHW
            if image.ndim == 3 and image.shape[0] not in (1, 3):
                image = image.permute(2, 0, 1)
            if image.ndim == 2:
                image = image.unsqueeze(0)
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
            return image

        # 以下分支仍然服务于“真图像”的情况（PIL）
        if self.image_aspect_ratio == 'pad':
            image = self.expand2square(
                image,
                tuple(int(x * 255) for x in self.image_processor.image_mean)
            )
        elif self.image_aspect_ratio == "anyres":
            image = self.process_anyres_image(
                image, self.image_processor, self.image_grid_pinpoints
            )
            return image

        image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
        return image

    @classmethod
    def expand2square(cls, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    @classmethod
    def process_anyres_image(cls, image, processor, grid_pinpoints):
        if type(grid_pinpoints) is list:
            possible_resolutions = grid_pinpoints
        else:
            possible_resolutions = ast.literal_eval(grid_pinpoints)
        best_resolution = select_best_resolution(image.size, possible_resolutions)
        image_padded = resize_and_pad_image(image, best_resolution)

        patches = divide_to_patches(image_padded, processor.crop_size['height'])

        image_original_resize = image.resize(
            (processor.size['shortest_edge'], processor.size['shortest_edge'])
        )

        image_patches = [image_original_resize] + patches
        image_patches = [
            processor(image_patch, return_tensors='pt')['pixel_values'][0]
            for image_patch in image_patches
        ]
        return torch.stack(image_patches, dim=0)
