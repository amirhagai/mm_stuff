from numbers import Number
from typing import List, Optional, Union
import random
import cv2
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmdet.structures.bbox import BaseBoxes, get_box_type
from mmdet.structures.mask import PolygonMasks
from mmengine.utils import is_list_of
import torch
import os
import re
import torch.nn as nn
from PIL import Image

from torchvision.transforms import Compose, ToTensor, ColorJitter, ToPILImage
from mmrotate.registry import TRANSFORMS
from pathlib import Path
# from .builder import TRANSFORM_PRIVATE


# use the registry to manage the module
@TRANSFORMS.register_module()
class AddFourierChannels(BaseTransform):
    """
    Assumes that PackedDetInputs allready been applied 
    following https://arxiv.org/pdf/2107.00630 appendix C

    """

    def __init__(self, min_n=7, max_n=8) -> None:
        self.min_n = min_n
        self.max_n = max_n
        

    def create_fourier_channels(self, img_tensor):
        if img_tensor.max().item() > 1:
            img_tensor = img_tensor / 255

        with torch.no_grad():
            img_tensor = img_tensor * 2 * np.pi  # Scale pixel values for trigonometric transformations
            
            # Create empty lists to collect new channels
            sin_channels, cos_channels = [], []

            # Apply transformations
            for n in range(self.min_n, self.max_n + 1):
                sin_channel = torch.sin(2 ** n * img_tensor)
                cos_channel = torch.cos(2 ** n * img_tensor)
                sin_channels.append(sin_channel)
                cos_channels.append(cos_channel)

            # Concatenate original and new channels
            new_tensor = torch.cat([*sin_channels, *cos_channels], dim=1 if len(img_tensor.shape) == 4 else 0)
            return new_tensor

    def transform(self, packed_results: dict) -> dict:

        packed_results['inputs'] = \
        torch.cat([packed_results['inputs'],
        self.create_fourier_channels(packed_results['inputs'])],
         dim=1 if len(packed_results['inputs'].shape)==4 else 0)
 
        return packed_results


@TRANSFORMS.register_module()
class AddGradAndLaplacianFast(BaseTransform):
    """
    happens over all the cahnnels toghther

    """

    def __init__(self) -> None:
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_x = nn.Conv2d(3, 1, kernel_size=(3, 3), padding=1, bias=False)
        sobel_x_kernel = sobel_x_kernel.repeat(3, 1, 1, 1)
        self.sobel_x.weight.data = sobel_x_kernel.reshape((1, 3, 3, 3))

        # Sobel filter for detecting vertical gradients
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_y = nn.Conv2d(3, 1, kernel_size=(3, 3), padding=1, bias=False)
        sobel_y_kernel = sobel_y_kernel.repeat(3, 1, 1, 1)
        self.sobel_y.weight.data = sobel_y_kernel.reshape((1, 3, 3, 3))

        # Laplacian kernel for detecting edges
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.laplacian = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, bias=False)
        laplacian_kernel = laplacian_kernel.repeat(3, 1, 1, 1)
        self.laplacian.weight.data = laplacian_kernel.reshape((1, 3, 3, 3))

        

    def create_grad_channels(self, img_tensor):
        with torch.no_grad():
            return torch.cat([self.sobel_x(img_tensor.to(torch.float32)) , self.sobel_y(img_tensor.to(torch.float32)), self.laplacian(img_tensor.to(torch.float32))], dim=1 if len(img_tensor.shape) == 4 else 0)

    def transform(self, packed_results: dict) -> dict:

        packed_results['inputs'] = \
        torch.cat([packed_results['inputs'],
        self.create_grad_channels(packed_results['inputs'])],
          dim=1 if len(packed_results['inputs'].shape) == 4 else 0)
 
        return packed_results


@TRANSFORMS.register_module()
class AddGradAndLaplacian(BaseTransform):
    """
    

    """

    def __init__(self, min_n=7, max_n=8) -> None:
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_x_kernel = sobel_x_kernel.repeat(3, 1, 1, 1)
        self.sobel_x = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, groups=3, bias=False)
        self.sobel_x.weight.data = sobel_x_kernel

        # Sobel filter for detecting vertical gradients
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y_kernel = sobel_y_kernel.repeat(3, 1, 1, 1)
        self.sobel_y = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, groups=3, bias=False)
        self.sobel_y.weight.data = sobel_y_kernel

        # Laplacian kernel for detecting edges
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        laplacian_kernel = laplacian_kernel.repeat(3, 1, 1, 1)
        self.laplacian = nn.Conv2d(3, 3, kernel_size=(3, 3), padding=1, groups=3, bias=False)
        self.laplacian.weight.data = laplacian_kernel


    def create_grad_channels(self, img_tensor):
        with torch.no_grad():
            return torch.cat([self.sobel_x(img_tensor.to(torch.float32)) , self.sobel_y(img_tensor.to(torch.float32)), self.laplacian(img_tensor.to(torch.float32))], dim=1 if len(img_tensor.shape) == 4 else 0)

    def transform(self, packed_results: dict) -> dict:

        packed_results['inputs'] = \
        torch.cat([packed_results['inputs'],
        self.create_grad_channels(packed_results['inputs'])],
          dim=1 if len(packed_results['inputs'].shape) == 4 else 0)
 
        return packed_results