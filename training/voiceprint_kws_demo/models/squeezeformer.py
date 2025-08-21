"""
SqueezeFormer模型实现
通过时间维度压缩减少计算复杂度的高效Transformer
适合处理长序列音频数据
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeReduction(nn.Module):
    """
    时间维度压缩模块
    通过卷积降采样减少序列长度，提高计算效率
    """

    def __init__(self, d_model: int, reduction_factor: int = 4):
        super().__init__()
        self.reduction_factor = reduction_factor

        # 时间压缩卷积：stride=reduction_factor进行降采样
        self.time_reduction_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=reduction_factor,
            stride=reduction_factor,
            padding=0
        )

        # 批归一化和激活
        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            压缩后的特征 [batch_size, seq_len//reduction_factor, d_model]
        """
        # [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)

        # 时间维度压缩
        x = self.time_reduction_conv(x)
        x = self.norm(x)
        x = self.activation(x)

        # [B, D, T'] -> [B, T', D]
        return x.transpose(1, 2)


class TimeRecovery(nn.Module):
    """
    时间维度恢复模块
    通过转置卷积恢复原始序列长度
    """

    def __init__(self, d_model: int, recovery_factor: int = 4):
        super().__init__()
        self.recovery_factor = recovery_factor

        # 时间恢复转置卷积
        self.time_recovery_conv = nn.ConvTranspose1d(
            d_model, d_model,
            kernel_size=recovery_factor,
            stride=recovery_factor,
            padding=0
        )

        self.norm = nn.BatchNorm1d(d_model)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Args:
            x: 压缩后的特征 [batch_size, seq_len', d_model]
            target_length: 目标序列长度
        Returns:
            恢复后的特征 [batch_size, target_length, d_model]
        """
        # [B, T', D] -> [B, D, T']
        x = x.transpose(1, 2)

        # 时间维度恢复
        x = self.time_recovery_conv(x)
        x = self.norm(x)
        x = self.activation(x)

        # [B, D, T] -> [B, T, D]
        x = x.transpose(1, 2)

        # 调整到目标长度（如果需要）
        current_length = x.size(1)
        if current_length != target_length:
            if current_length > target_length:
                x = x[:, :target_length, :]
            else:
                # 零填充
                pad_length = target_length - current_length
                x = F.pad(x, (0, 0, 0, pad_length))

        return x


class EfficientAttention(nn.Module):
    """
    高效注意力机制
    在压缩的时间维度上进行注意力计算
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.d_k)

        # 层归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        高效多头自注意力
        Args:
            x: [batch_size, compressed_seq_len, d_model]
        Returns:
            注意力输出 [batch_size, compressed_seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        residual = x

        # 层归一化
        x = self.norm(x)

        # 线性投影
        Q = self.w_q(x).view(batch_size, seq_len,
                             self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len,
                             self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len,
                             self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        output = self.w_o(context)

        # 残差连接
        return residual + self.dropout(output)


class SqueezeformerBlock(nn.Module):
    """
    SqueezeFormer块
    流程：时间压缩 -> 高效注意力 -> 前馈网络 -> 时间恢复
    """

    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 d_ff: int = 1024,
                 reduction_factor: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.reduction_factor = reduction_factor

        # 时间压缩
        self.time_reduction = TimeReduction(d_model, reduction_factor)

        # 高效注意力
        self.attention = EfficientAttention(d_model, num_heads, dropout)

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # 时间恢复
        self.time_recovery = TimeRecovery(d_model, reduction_factor)

        # 残差连接的投影层
        self.residual_proj = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SqueezeFormer块前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, d_model]
        Returns:
            输出特征 [batch_size, seq_len, d_model]
        """
        original_length = x.size(1)
        residual = x

        # 1. 时间维度压缩
        x_compressed = self.time_reduction(x)

        # 2. 在压缩空间进行高效注意力计算
        x_attended = self.attention(x_compressed)

        # 3. 前馈网络处理
        x_ff = x_attended + self.feed_forward(x_attended)

        # 4. 时间维度恢复
        x_recovered = self.time_recovery(x_ff, original_length)

        # 5. 残差连接
        residual_proj = self.residual_proj(
            residual.transpose(1, 2)).transpose(1, 2)

        return x_recovered + residual_proj


class SqueezeformerEncoder(nn.Module):
    """
    完整的SqueezeFormer编码器
    包含多层SqueezeFormer块
    """

    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 num_layers: int = 8,
                 num_heads: int = 8,
                 d_ff: int = 1024,
                 reduction_factor: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # SqueezeFormer层
        self.layers = nn.ModuleList([
            SqueezeformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                reduction_factor=reduction_factor,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # 输出层归一化
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码器前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
        Returns:
            编码后的特征 [batch_size, seq_len, d_model]
        """
        # 输入投影
        x = self.input_projection(x)

        # 通过SqueezeFormer层
        for layer in self.layers:
            x = layer(x)

        return self.output_norm(x)


class AdaptiveSqueezeformer(nn.Module):
    """
    自适应SqueezeFormer
    根据序列长度动态调整压缩率
    """

    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 max_reduction: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.max_reduction = max_reduction

        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model)
        )

        # 多种压缩率的SqueezeFormer块
        self.squeeze_blocks = nn.ModuleDict({
            f'reduction_{r}': SqueezeformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                reduction_factor=r,
                dropout=dropout
            ) for r in [2, 4, 8]
        })

        # 序列长度分类器（决定使用哪种压缩率）
        self.length_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3种压缩率
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        自适应前向传播
        根据序列长度选择最优的压缩策略
        """
        batch_size, seq_len, _ = x.shape

        # 输入投影
        x = self.input_projection(x)

        # 根据序列长度选择压缩率
        if seq_len < 50:
            reduction_key = 'reduction_2'
        elif seq_len < 200:
            reduction_key = 'reduction_4'
        else:
            reduction_key = 'reduction_8'

        # 使用选定的SqueezeFormer块
        x = self.squeeze_blocks[reduction_key](x)

        return x


# 测试代码
if __name__ == "__main__":
    print("=== SqueezeFormer模型测试 ===")

    # 创建测试数据
    batch_size, seq_len, input_dim = 4, 200, 80
    x = torch.randn(batch_size, seq_len, input_dim)

    print(f"输入形状: {x.shape}")

    # 1. 测试基础SqueezeFormer编码器
    print("\n1. 基础SqueezeFormer编码器:")
    encoder = SqueezeformerEncoder(
        input_dim=input_dim,
        d_model=256,
        num_layers=4,
        reduction_factor=4
    )

    output = encoder(x)
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in encoder.parameters()):,}")

    # 2. 测试自适应SqueezeFormer
    print("\n2. 自适应SqueezeFormer:")
    adaptive_encoder = AdaptiveSqueezeformer(
        input_dim=input_dim,
        d_model=256,
        num_layers=3
    )

    output_adaptive = adaptive_encoder(x)
    print(f"自适应输出形状: {output_adaptive.shape}")
    print(f"自适应参数量: {sum(p.numel() for p in adaptive_encoder.parameters()):,}")

    # 3. 计算效率对比
    print("\n3. 效率对比:")

    # 标准Transformer（全长度注意力）
    standard_ops = seq_len ** 2  # O(T^2)

    # SqueezeFormer（压缩后注意力）
    compressed_len = seq_len // 4
    squeeze_ops = compressed_len ** 2  # O((T/4)^2)

    efficiency_gain = standard_ops / squeeze_ops
    print(f"标准注意力操作数: {standard_ops:,}")
    print(f"SqueezeFormer操作数: {squeeze_ops:,}")
    print(f"效率提升: {efficiency_gain:.1f}x")
