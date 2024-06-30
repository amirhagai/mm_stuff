import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS


class MultiHeadSpatialAttention(nn.Module):
    def __init__(self, channel, reduction=8, num_heads=4):
        super(MultiHeadSpatialAttention, self).__init__()
        self.num_heads = num_heads
        self.channel_per_head = channel // num_heads

        self.query_convs = nn.ModuleList([
            nn.Conv2d(self.channel_per_head, self.channel_per_head // reduction, kernel_size=1)
            for _ in range(num_heads)])
        self.key_convs = nn.ModuleList([
            nn.Conv2d(self.channel_per_head, self.channel_per_head // reduction, kernel_size=1)
            for _ in range(num_heads)])
        self.value_convs = nn.ModuleList([
            nn.Conv2d(self.channel_per_head, self.channel_per_head, kernel_size=1)
            for _ in range(num_heads)])

        self.softmax = nn.Softmax(dim=-1)  # Apply softmax to the spatial dimension

    def forward(self, x1, x2):
        batch, _, height, width = x1.size()

        # Split input feature maps into multiple heads
        x1_splits = torch.split(x1, self.channel_per_head, dim=1)
        x2_splits = torch.split(x2, self.channel_per_head, dim=1)

        output_heads = []

        for i in range(self.num_heads):
            query = self.query_convs[i](x1_splits[i]).view(batch, -1, height * width).permute(0, 2, 1)
            key = self.key_convs[i](x2_splits[i]).view(batch, -1, height * width)

            attention = torch.bmm(query, key)
            attention = self.softmax(attention)

            value = self.value_convs[i](x2_splits[i]).view(batch, -1, height * width)
            out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch, self.channel_per_head, height, width)

            output_heads.append(out)

        # Concatenate the output from each head
        out = torch.cat(output_heads, dim=1)
        return out


@MODELS.register_module()
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, channels, num_heads=4):
        super(MultiScaleFeatureFusion, self).__init__()
        # Initialize multi-head attention modules for each scale
        self.attentions = nn.ModuleList([
            MultiHeadSpatialAttention(ch, num_heads=num_heads) for ch in channels])
        self.gates = nn.ModuleList([
            nn.Conv2d(2 * ch, ch, kernel_size=1) for ch in channels])

    def forward(self, x1, x2):
        fused_outputs = []
        for i, (features1, features2) in enumerate(zip(x1, x2)):
            attention_out = self.attentions[i](features1, features2)
            concat_out = torch.cat([features1, attention_out], dim=1)
            fused_out = self.gates[i](concat_out)
            fused_outputs.append(fused_out)
        return fused_outputs