import torch
from mmdet.structures import SampleList
from typing import Tuple
from torch import Tensor
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
import torch.nn.functional as F
from mmdet.models.utils import unpack_gt_instances


class BaseDenseHeadContrastive(BaseDenseHead):


    def contrastive_loss(self, x, x_transformed, margin=0.5):
        # Flatten the feature dimensions to make x and x_transformed of shape (bs, c*w*h)
        x_flat = x.view(x.size(0), -1)
        x_transformed_flat = x_transformed.view(x_transformed.size(0), -1)
        
        # Normalize the flattened inputs to use cosine similarity
        x_norm = F.normalize(x_flat, p=2, dim=1)
        x_transformed_norm = F.normalize(x_transformed_flat, p=2, dim=1)
        
        # Cosine similarity between all pairs
        similarities = torch.mm(x_norm, x_transformed_norm.t())
        
        # Create masks for positive and negative pairs
        batch_size = x.size(0)
        positive_mask = torch.eye(batch_size).bool().to(x.device)  # Ensure the mask is on the same device as x
        negative_mask = ~positive_mask
        
        # Extract positive and negative similarities
        positive_similarities = similarities[positive_mask]
        negative_similarities = similarities[negative_mask].view(batch_size, -1)
        
        # Loss for positive pairs: Maximize similarity (minimize -similarity)
        positive_loss = -positive_similarities.mean()
        
        # Loss for negative pairs: Minimize similarity (push similarities below a margin)
        negative_loss = F.relu(negative_similarities - margin).mean()
        
        # Combine losses
        total_loss = positive_loss + negative_loss
        
        return total_loss
    
    
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

        # mse_loss = torch.nn.MSELoss()

        if x_transformed is not None:
            total_loss = torch.tensor(0.0, requires_grad=True)

            # Compute MSE loss for each pair of corresponding elements in x and x_transformed
            for i in range(len(x)):
                # mse_loss can be calculated using F.mse_loss which expects tensor inputs
                # loss = F.mse_loss(x[i], x_transformed[i], reduction='mean')
                loss = self.contrastive_loss(x[i], x_transformed[i])
                total_loss = total_loss + loss  # Accumulate the loss
            losses["contrastive"] = total_loss
        return losses

