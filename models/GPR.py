import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import TwoWayTransformer

class ImplicitMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        构造隐式MLP网络
        :param input_dim: 输入维度（特征值维度 + 位置维度）
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出维度（目标特征维度）
        """
        super(ImplicitMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class GeometricPerceptionReconstruction(nn.Module):
    def __init__(self, input_dim=256, low_rank_dim=32):
        super(GeometricPerceptionReconstruction, self).__init__()
        
        # Mapping layers to project features into low-rank space
        self.mlp_2d = nn.Sequential(
            nn.Conv2d(input_dim, low_rank_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(low_rank_dim, low_rank_dim, kernel_size=1)
        )
        self.mlp_3d = nn.Sequential(
            nn.Conv3d(input_dim, low_rank_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(low_rank_dim, low_rank_dim, kernel_size=1)
        )

        # MLP for relative position embedding fusion


        # MLP for alpha weight calculation
        self.mlp_alpha = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

        self.mlp_up = nn.Sequential(
            nn.Conv3d(low_rank_dim, low_rank_dim*2, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(low_rank_dim*2, input_dim, kernel_size=1)
        )

        self.input_dim = input_dim
        self.transformer = TwoWayTransformer(depth=1,embedding_dim=self.input_dim,mlp_dim=2048, num_heads=8)
        self.low_rank_dim = low_rank_dim

  
    def encode_project_2d_to_3d(self,input_matrix, original_size, compressed_size, positions, hidden_dim=128):
        """
        将2D矩阵编码到3D空间
        :param input_matrix: 输入2D矩阵，形状为 [B, C, X, Y]
        :param original_size: 原始3D空间大小，例如 (100, 100, 100)
        :param compressed_size: 压缩后3D空间大小，例如 (50, 50, 50)
        :param positions: 位置矩阵，形状为 [3]（例如中心位置 z = 10）
        :param hidden_dim: 隐藏层维度（MLP）
        :return: 编码后的3D矩阵，形状为 [B, C, O, P, Q]
        """
        B, C, X, Y = input_matrix.shape

        # 计算相对位置
        original_size_tensor = torch.tensor(original_size, dtype=torch.float32)
        relative_positions = (positions / original_size_tensor)  # 相对位置 [3]

        # Flatten 输入矩阵
        input_flat = input_matrix.view(B, C, -1)  # [B, C, X*Y]

        # 广播相对位置到每个特征点
        relative_positions_expanded = relative_positions.unsqueeze(0).unsqueeze(0).expand(B, C, -1)  # [B, C, 3]

        # 拼接 2D 特征矩阵与相对位置
        mlp_input = torch.cat([input_flat, relative_positions_expanded], dim=-1)  # [B, C, X*Y+3]

        # 调整形状为 [B * C, X*Y+3]
        mlp_input = mlp_input.view(B * C, -1)

        # 初始化MLP模型
        input_dim = X * Y + 3  # 特征维度 (X*Y) + 位置维度 (3)
        output_dim = compressed_size[0] * compressed_size[1] * compressed_size[2]  # 输出维度
        mlp = ImplicitMLP(input_dim, hidden_dim, output_dim)

        # 投影到3D空间
        mlp_output = mlp(mlp_input)  # [B * C, compressed_size[0]*compressed_size[1]*compressed_size[2]]

        # 恢复形状为 [B, C, compressed_size[0]*compressed_size[1]*compressed_size[2]]
        mlp_output = mlp_output.view(B, C, -1)

        # 恢复到目标3D形状
        output_matrix = mlp_output.view(B, C, *compressed_size)  # [B, C, O, P, Q]

        return output_matrix

    def forward(self, features_2d, pos_2d, sparse_code_2d, dense_code_2d, 
                features_3d, slice_positions, original_shape):
        """
        Args:
            features_2d: Tensor of shape (B, 256, N, N)
            pos_2d: Tensor of shape (1, 256, N, N)
            sparse_code_2d: Tensor of shape (B, 2, 256)
            dense_code_2d: Tensor of shape (B, 256, N, N)
            features_3d: Tensor of shape (B, 256, X, N, N)
            slice_positions: Tensor of shape (B, 3) representing slice X, Y, Z
            original_shape: Tuple (A, B, C) of the original 3D shape
        """
        # Step 1: Fuse 2D image features
        self.iou_token = nn.Embedding(1, self.input_dim)
        self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.input_dim)
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_code_2d.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_code_2d), dim=1)
        # Expand per-image data in batch direction to be per-mask
        if features_2d.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(features_2d, tokens.shape[0], dim=0)
        else:
            src = features_2d
        src = src + dense_code_2d
        pos_src =  torch.repeat_interleave(pos_2d, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        _,fused_2d_features =self.transformer(src, pos_src, tokens)
        fused_2d_features = fused_2d_features.transpose(1, 2).view(b, c, h, w)

        # fused_2d_features = torch.cat((src, pos_src), dim=1)       # Shape: (B, 256+256+2, N, N)
        low_rank_2d = self.mlp_2d(fused_2d_features)  # Shape: (B, low_rank_dim, N, N)
        # #
        # fused_2d_features = features_2d + pos_2d + dense_code_2d
        # low_rank_2d = self.mlp_2d(fused_2d_features)  # Shape: (B, low_rank_dim, N, N)

        # Map 3D features to low-rank space
        low_rank_3d = self.mlp_3d(features_3d)  # Shape: (B, low_rank_dim, X, N, N)

        # Step 2: Compute relative position embeddings and fuse with 2D features
        target_shape = features_3d.shape
        rel_pos_embeddings = self.encode_project_2d_to_3d(input_matrix=low_rank_2d,original_size=original_shape, compressed_size=target_shape[2:],positions=slice_positions,hidden_dim=self.low_rank_dim)  # Shape: (B, low_rank_dim, X, N, N)
        
        projection_3d = low_rank_2d.unsqueeze(2) + rel_pos_embeddings  # Shape: (B, low_rank_dim, X, N, N)

        # Step 3: Combine 3D projection with original 3D features
        cosine_similarity = F.cosine_similarity(projection_3d.unsqueeze(dim=0), low_rank_3d.unsqueeze(dim=0), dim=0)
        # Compute Gaussian similarity for the entire matrix
        positions =projection_3d.unsqueeze(dim=0)  # Shape: (B, 1, 3)
        diff = positions - low_rank_3d.unsqueeze(dim=0) # Shape: (B, B, 3)
        distances = torch.norm(diff,dim=0)  # Shape: (B, B)
        sigma = 1.0  # You can adjust sigma as needed
        
        gaussian_similarity = torch.exp(-distances / (2 * sigma ** 2))  # Shape: (B, B)

        # 将 cosine_similarity 和 gaussian_similarity 沿着最后一个维度拼接
        similarity_input = torch.stack([cosine_similarity, gaussian_similarity], dim=-1)  # 形状: (B, B, 2)
        # 对每个样本计算 alpha 值
        alpha = self.mlp_alpha(similarity_input)  # 形状: (B, B, 1)
        # 将 alpha 压缩为标量值（取均值或其他操作）
        alpha = alpha.mean(dim=-1)  # 形状: (B, 1, 1)

        result = alpha * projection_3d + (1 - alpha) * low_rank_3d

        result = self.mlp_up(result)

        return result

if __name__ == "__main__":
    # Example usage
    model = GeometricPerceptionReconstruction()
    features_2d = torch.randn(2, 256, 64, 64)
    pos_2d = torch.randn(1, 256, 64, 64)
    sparse_code_2d = torch.randn(2, 2, 256)
    dense_code_2d = torch.randn(2, 256, 64, 64)
    features_3d = torch.randn(2, 256, 16, 64, 64)
    slice_positions = torch.tensor([64,1024,1024])
    original_shape = (128, 1024, 1024)


    output = model(features_2d, pos_2d, sparse_code_2d, dense_code_2d, features_3d, slice_positions, original_shape)
    print(output.shape)
