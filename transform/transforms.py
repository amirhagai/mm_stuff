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
from PIL import Image

from torchvision.transforms import Compose, ToTensor, ColorJitter, ToPILImage
from mmrotate.registry import TRANSFORMS
from pathlib import Path

# use the registry to manage the module
@TRANSFORMS.register_module()
class BboxColorJitter2(BaseTransform):
    """Color jitter only over bboxes

    """

    def __init__(self, prob, brightness=0, contrast=0, saturation=0, hue=0.5, class_num=5 ) -> None:
        self.prob = prob
        # self.path = path
        self.class_num = class_num 
        self.brightness, self.contrast, self.saturation, self.hue = \
            brightness, contrast, saturation, hue
        self.transform_im = \
                Compose([
                ColorJitter(
                brightness=self.brightness, contrast=self.contrast,\
                      saturation=self.saturation, hue=self.hue),

            ])
        self.transform_done_im = \
            ToPILImage()
    
    def ann_to_bbox(anno_list):
        """"
        this is how the boxes that i get looks like if anno_list is the bbox given by the annotation file
        """
        cnt = np.int0(np.array(anno_list).reshape(4, 2))
        cnt = cnt.reshape((4, 2))
        rect = cv2.minAreaRect(cnt)
        return torch.tensor([rect[0][0], rect[0][1], rect[1][0], rect[1][1], np.pi * rect[2] / 180])


    def transform(self, results: dict) -> dict:
    
        recs = results['gt_bboxes'].tensor
        labels = results['gt_bboxes_labels']
        # print(labels, results["img_path"])
        im_shape = results['img'].shape
        boxes = []
        for i, rec in enumerate(recs):
            if labels[i] == self.class_num and random.random() < self.prob:
                rec = (
                    (rec[0].item(), rec[1].item()),
                    (rec[2].item(), rec[3].item()),
                        (rec[4].item() * 180 / np.pi)
                    )         
                box = cv2.boxPoints(rec)
                box = np.int0(box)
                boxes.append(box)
        if len(boxes) > 0:
            mask = np.ones(im_shape).astype(np.uint8)
            cv2.drawContours(mask, boxes, -1, (255, 0, 0), thickness=cv2.FILLED)
            indecis = np.where(mask == 255)
            bbox_to_transform = results['img'][indecis[0], indecis[1], :].astype(np.float32) / 255.
            transformed_tensor = self.transform_im(torch.tensor(bbox_to_transform.T[:, :, None]))
            results['img'][indecis[0], indecis[1], :] = (transformed_tensor[:, :, 0].T.numpy() * 255).astype(np.uint8)
 
        return results