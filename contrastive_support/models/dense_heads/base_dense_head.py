import torch
from mmdet.structures import SampleList
from typing import Tuple
from torch import Tensor
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

from mmdet.models.utils import unpack_gt_instances


class BaseDenseHeadContrastive(BaseDenseHead):
    """Custom implementation of DenseHead with modified loss function.
    """
    def loss(self, x: Tuple[Tensor], x_transformed: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            x_transformed (tuple[Tensor]): Features from the upstream network of the transformed image, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        mse_loss = torch.nn.MSELoss()
        losses["contrastive"] = (mse_loss(x[i], x_transformed[i]) for i in range(len(x))).sum()
        return losses

