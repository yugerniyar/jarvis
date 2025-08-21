"""
Conformer模型实现
结合卷积和自注意力的语音识别模型
论文: Conformer: Convolution-augmented Transformer for Speech Recognition
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码
    相比绝对位置编码，更适合音频等序列数据
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # 创建相对位置编码表
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # 使用sin/cos编码位置信息
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """添加位置编码"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class MultiHeadedSelfAttention(nn.Module):
    """
    多头自注意力机制
    支持相对位置编码
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] 注意力掩码
        """
        batch_size, seq_len, _ = x.shape

        # 线性投影并重塑为多头
        Q = self.w_q(x).view(batch_size, seq_len,
                             self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len,
                             self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len,
                             self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        context = torch.matmul(attn_weights, V)

        # 重塑输出
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)

        return self.w_o(context)


class ConvolutionModule(nn.Module):
    """
    Conformer的卷积模块
    使用门控线性单元(GLU)和深度可分离卷积
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()

        # 逐点卷积扩展
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, 1)

        # GLU激活
        self.glu = nn.GLU(dim=1)

        # 深度卷积
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model  # 深度可分离卷积
        )

        # 批归一化
        self.batch_norm = nn.BatchNorm1d(d_model)

        # Swish激活函数
        self.activation = nn.SiLU()

        # 逐点卷积压缩
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            卷积处理后的特征 [batch_size, seq_len, d_model]
        """
        # 转换维度: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)

        # 逐点卷积扩展
        x = self.pointwise_conv1(x)

        # GLU门控
        x = self.glu(x)

        # 深度卷积
        x = self.depthwise_conv(x)

        # 批归一化和激活
        x = self.batch_norm(x)
        x = self.activation(x)

        # 逐点卷积压缩
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # 转换回原维度: [B, D, T] -> [B, T, D]
        return x.transpose(1, 2)


class FeedForwardModule(nn.Module):
    """
    前馈网络模块
    使用Swish激活函数和Dropout
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()  # Swish激活函数
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(self.layer_norm(x)))))


class ConformerBlock(nn.Module):
    """
    完整的Conformer块
    架构: FFN(1/2) -> MHSA -> Conv -> FFN(1/2)
    每个子模块都有残差连接和层归一化
    """

    def __init__(self,
                 d_model: int = 256,
                 num_heads: int = 8,
                 d_ff: int = 1024,
                 conv_kernel_size: int = 31,
                 dropout: float = 0.1):
        super().__init__()

        # 第一个前馈网络（1/2权重）
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 多头自注意力
        self.mhsa = MultiHeadedSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # 卷积模块
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # 第二个前馈网络（1/2权重）
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.norm4 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Conformer块的前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]

        Returns:
            处理后的特征 [batch_size, seq_len, d_model]
        """
        # 1. 第一个FFN（权重为1/2）
        residual = x
        x = self.norm1(x)
        x = residual + 0.5 * self.ff1(x)

        # 2. 多头自注意力
        residual = x
        x = self.norm2(x)
        x = residual + self.mhsa(x, mask)

        # 3. 卷积模块
        residual = x
        x = self.norm3(x)
        x = residual + self.conv(x)

        # 4. 第二个FFN（权重为1/2）
        residual = x
        x = self.norm4(x)
        x = residual + 0.5 * self.ff2(x)

        return self.dropout(x)


class ConformerEncoder(nn.Module):
    """
    完整的Conformer编码器
    包含多层Conformer块和位置编码
    """

    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 num_layers: int = 16,
                 num_heads: int = 8,
                 d_ff: int = 1024,
                 conv_kernel_size: int = 31,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()

        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # 位置编码
        self.pos_encoding = RelativePositionalEncoding(d_model, max_len)

        # Conformer层
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, num_heads, d_ff, conv_kernel_size, dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码器前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
            mask: 注意力掩码

        Returns:
            编码后的特征 [batch_size, seq_len, d_model]
        """
        # 输入投影
        x = self.input_projection(x)

        # 添加位置编码
        x = self.pos_encoding(x)

        # 通过Conformer层
        for layer in self.layers:
            x = layer(x, mask)

        return self.layer_norm(x)


# 测试代码
if __name__ == "__main__":
    # 创建示例输入
    batch_size, seq_len, input_dim = 4, 100, 80
    x = torch.randn(batch_size, seq_len, input_dim)

    # 创建Conformer编码器
    encoder = ConformerEncoder(
        input_dim=input_dim,
        d_model=256,
        num_layers=6,  # 使用较少层数进行测试
        num_heads=8
    )

    # 前向传播
    output = encoder(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in encoder.parameters()):,}")
