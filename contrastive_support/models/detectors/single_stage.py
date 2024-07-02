# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from torch import Tensor
import torch
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetectorContrastive
from ...visualiztions import *
import torch.nn as nn
import mmengine

@MODELS.register_module()
class SingleStageDetectorContrastive(BaseDetectorContrastive):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_head_also_over_backbone: bool = False) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_head_also_over_backbone = use_head_also_over_backbone
        if self.use_head_also_over_backbone:
            self.conv1 = nn.Conv2d(kernel_size=3, in_channels=neck['in_channels'][1], out_channels=neck['out_channels'], padding=1)
            self.conv2 = nn.Conv2d(kernel_size=3, in_channels=neck['in_channels'][2], out_channels=neck['out_channels'], padding=1)


    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """

        # def use_running_stats(m):
        #     if isinstance(m, mmengine.model.utils._BatchNormXd):
        #         m.track_running_stats = True

        # # Function to revert BatchNorm layers back to learning mode
        # def train_running_stats(m):
        #     if isinstance(m, mmengine.model.utils._BatchNormXd):
        #         m.track_running_stats = False


        # # Apply to all BatchNorm layers
        # self.backbone.apply(use_running_stats)
        # self.neck.apply(use_running_stats)

        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)

        # self.backbone.apply(train_running_stats)
        # self.neck.apply(train_running_stats)
        return x

    def extract_fine_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """

        # def use_running_stats(m):
        #     if isinstance(m, mmengine.model.utils._BatchNormXd):
        #         m.track_running_stats = True

        # # Function to revert BatchNorm layers back to learning mode
        # def train_running_stats(m):
        #     if isinstance(m, mmengine.model.utils._BatchNormXd):
        #         m.track_running_stats = False


        # # Apply to all BatchNorm layers
        # self.backbone.apply(use_running_stats)
        # self.neck.apply(use_running_stats)

        x_backbone = self.backbone(batch_inputs)
        if self.with_neck:
            x_neck = self.neck(x_backbone)

        # self.backbone.apply(train_running_stats)
        # self.neck.apply(train_running_stats)
        return x_backbone, x_neck

    def loss(self, batch_inputs: Tensor,
             batch_inputs_transformed: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """

        if not self.use_head_also_over_backbone:
            x = self.extract_feat(batch_inputs)
            with torch.no_grad():
                transformed_falg = (batch_inputs != batch_inputs_transformed).sum() > 0
            if transformed_falg :
                x_transformed = self.extract_feat(batch_inputs_transformed)
                # for i in range(len(batch_inputs)):
                #     if (batch_inputs[i] != batch_inputs_transformed[i]).sum() > 0:
                #         pca_diff(batch_inputs[i][None,...,], batch_inputs_transformed[i][None,...,], x[0][i][None,...,], x_transformed[0][i][None,...,], f'/mm_stuff/vis0_{i}.png')
                #         pca_diff(batch_inputs[i][None,...,], batch_inputs_transformed[i][None,...,], x[1][i][None,...,], x_transformed[1][i][None,...,], f'/mm_stuff/vis1_{i}.png')
                #         pca_diff(batch_inputs[i][None,...,], batch_inputs_transformed[i][None,...,], x[2][i][None,...,], x_transformed[2][i][None,...,], f'/mm_stuff/vis2_{i}.png')
                # print()
            else:
                x_transformed = None
            losses = self.bbox_head.loss(x, x_transformed, batch_data_samples)
        else:
            x_backbone, x_neck = self.extract_fine_feat(batch_inputs)
            with torch.no_grad():
                transformed_falg = (batch_inputs != batch_inputs_transformed).sum() > 0
            if transformed_falg :
                x_backbone_transformed, x_neck_transformed = self.extract_fine_feat(batch_inputs_transformed)
                new_x = [x_backbone_transformed[0]]
                new_x.append(self.conv1(x_backbone_transformed[1]))
                new_x.append(self.conv2(x_backbone_transformed[2]))
                x_backbone_transformed = tuple(new_x)

            else:
                x_backbone_transformed, x_neck_transformed = None, None

            new_x = [x_backbone[0]]
            new_x.append(self.conv1(x_backbone[1]))
            new_x.append(self.conv2(x_backbone[2]))
            x_backbone = tuple(new_x)


            losses_backbone = self.bbox_head.loss(x_backbone, x_backbone_transformed, batch_data_samples)
            losses_neck = self.bbox_head.loss(x_neck , x_neck_transformed, batch_data_samples)
            losses = dict()
            for key in losses_backbone.keys():
                vals = []
                if key == "contrastive":
                    vals.append(0.5 * losses_backbone[key] + 0.5 * losses_neck[key])
                else:
                    for i in range(len(losses_backbone[key])):
                        vals.append(0.2 * losses_backbone[key][i] + 0.8 * losses_neck[key][i])
                losses[key] = vals
        return losses