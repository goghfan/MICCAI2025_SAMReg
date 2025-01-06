# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

# from .common import LayerNorm2d


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): 通道维度
          transformer (nn.Module): 用于预测掩码的转换器
          num_multimask_outputs (int): 消除掩码歧义时要预测的掩码数量(对于模糊意义的掩码，模型会预测多个掩码，供后续处理或用户选择。)
          activation (nn.Module): 是否能用Gelu激活函数
          iou_head_depth (int): 用于预测掩码质量的 MLP 深度
          iou_head_hidden_dim (int): 用于预测掩码质量的 MLP 隐藏维度
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs
        '''
        这个 IoU token 会被输入到 Transformer 中，与其他 token（如 mask tokens 和 prompt embeddings）一起进行处理。
        Transformer 会根据输入的特征（如图像嵌入、提示嵌入等）和 IoU token，生成与 IoU 相关的特征表示。
        最终，这些特征表示会被传递给 iou_prediction_head，用于预测每个掩码的 IoU 分数。
        '''
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.
        预测给定图像和提示嵌入的掩码。

        Arguments:
          image_embeddings (torch.Tensor): 图像编码器的嵌入
          image_pe (torch.Tensor): 基于图像嵌入形状的位置编码
          sparse_prompt_embeddings (torch.Tensor): 点和框的嵌入
          dense_prompt_embeddings (torch.Tensor): 掩码输入的嵌入
          multimask_output (bool): 是返回多个掩码还是返回单个掩码。

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        # 可学习的缩放参数
        self.weight = nn.Parameter(torch.ones(num_channels))
        # 可学习的偏移参数
        self.bias = nn.Parameter(torch.zeros(num_channels))
        # 用于数值稳定性的小常数
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算每个通道的均值
        u = x.mean(1, keepdim=True)
        # 计算每个通道的方差
        s = (x - u).pow(2).mean(1, keepdim=True)
        # 归一化：减去均值，除以标准差
        x = (x - u) / torch.sqrt(s + self.eps)
        # 应用可学习的缩放和偏移
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x

class DeformationFieldDecoder(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        # 类似于 MaskDecoder 的 IoU token 和 mask tokens
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = 1  # 可以根据需要调整
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 3D 上采样模块
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm3d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose3d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # 变形场预测头
        self.deformation_field_head = nn.Conv3d(transformer_dim // 8, 3, kernel_size=1)

        # 类似于 MaskDecoder 的超网络 MLP
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

    def forward(
        self,
        fixed_image_embeddings: torch.Tensor,
        moving_image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # 拼接固定图像和移动图像的特征
        src = torch.cat([fixed_image_embeddings, moving_image_embeddings], dim=1)

        # 拼接 IoU token 和 mask tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 扩展图像特征以匹配 token 的数量
        if src.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(src, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        # 通过 Transformer
        hs, src = self.transformer(src, pos_src, tokens)

        # 提取 mask tokens 的输出
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # 上采样特征图
        src = src.transpose(1, 2).view(src.shape[0], src.shape[2], src.shape[3], src.shape[4], src.shape[5])
        upscaled_embedding = self.output_upscaling(src)

        # 使用超网络 MLP 生成形变场
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        # 预测形变场
        b, c, d, h, w = upscaled_embedding.shape
        deformation_field = (hyper_in @ upscaled_embedding.view(b, c, d * h * w)).view(b, -1, d, h, w)
        deformation_field = self.deformation_field_head(deformation_field)

        return deformation_field
    
# if __name__ == "__main__":
# # 假设你已经定义了一个 Transformer 模块
#     transformer = (transformer_dim=256)

#     # 初始化 DeformationFieldDecoder
#     decoder = DeformationFieldDecoder(transformer_dim=256, transformer=transformer)

#     # 创建输入数据
#     fixed_image_embeddings = torch.randn(1, 256, 32, 32, 32)  # 固定图像的特征嵌入
#     moving_image_embeddings = torch.randn(1, 256, 32, 32, 32)  # 移动图像的特征嵌入
#     image_pe = torch.randn(1, 256, 32, 32, 32)  # 位置编码

#     # 前向传播
#     deformation_field = decoder(fixed_image_embeddings, moving_image_embeddings, image_pe)