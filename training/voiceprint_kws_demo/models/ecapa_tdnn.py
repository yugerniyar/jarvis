"""
ECAPA-TDNN模型实现
专门用于声纹识别的时延神经网络
论文: ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Res2NetBlock(nn.Module):
    """
    Res2Net残差块
    通过多尺度特征提取增强表征能力
    """

    def __init__(self, in_channels: int, out_channels: int, scale: int = 8, dilation: int = 1):
        super().__init__()

        assert in_channels % scale == 0
        assert out_channels % scale == 0

        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 分组卷积的每组通道数
        group_channels = out_channels // scale

        # 1x1卷积降维
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        # 多尺度卷积组
        self.convs = nn.ModuleList([
            nn.Conv1d(group_channels, group_channels, 3,
                      padding=dilation, dilation=dilation)
            for _ in range(scale - 1)
        ])

        self.bns = nn.ModuleList([
            nn.BatchNorm1d(group_channels) for _ in range(scale - 1)
        ])

        # 1x1卷积升维
        self.conv3 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm1d(out_channels)

        # 残差连接投影
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, time_steps]
        Returns:
            多尺度特征 [batch_size, out_channels, time_steps]
        """
        residual = x

        # 1x1卷积降维
        out = self.relu(self.bn1(self.conv1(x)))

        # 分组处理
        group_size = out.size(1) // self.scale
        groups = torch.split(out, group_size, dim=1)

        outputs = [groups[0]]  # 第一组直接使用

        # 后续组进行卷积处理
        for i in range(1, self.scale):
            if i == 1:
                sp = self.convs[i-1](groups[i])
            else:
                sp = self.convs[i-1](groups[i] + outputs[i-1])
            sp = self.relu(self.bns[i-1](sp))
            outputs.append(sp)

        # 拼接所有组
        out = torch.cat(outputs, dim=1)

        # 1x1卷积升维
        out = self.bn3(self.conv3(out))

        # 残差连接
        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation注意力块
    自动学习通道重要性权重
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, time_steps]
        Returns:
            注意力加权后的特征
        """
        b, c, t = x.size()

        # 全局平均池化
        y = self.global_pool(x).view(b, c)

        # 计算通道注意力权重
        y = self.fc(y).view(b, c, 1)

        # 应用注意力权重
        return x * y


class ECAPA_TDNN_Layer(nn.Module):
    """
    ECAPA-TDNN层
    结合Res2Net和SE注意力机制
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 scale: int = 8):
        super().__init__()

        # Res2Net块
        self.res2net = Res2NetBlock(
            in_channels, out_channels, scale, dilation)

        # SE注意力
        self.se_block = SEBlock(out_channels)

        # 额外的1x1卷积
        self.conv1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ECAPA-TDNN层前向传播"""
        # Res2Net处理
        x = self.res2net(x)

        # SE注意力加权
        x = self.se_block(x)

        # 额外的1x1卷积
        x = self.relu(self.bn(self.conv1x1(x)))

        return x


class AttentiveStatisticsPooling(nn.Module):
    """
    注意力统计池化
    自动学习重要时间段的权重
    """

    def __init__(self, channels: int, attention_channels: int = 128):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_channels, 1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_channels),
            nn.Conv1d(attention_channels, channels, 1),
            nn.Softmax(dim=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, time_steps]
        Returns:
            统计特征 [batch_size, channels * 2] (mean + std)
        """
        # 计算注意力权重
        attention_weights = self.attention(x)

        # 加权平均
        weighted_mean = torch.sum(x * attention_weights, dim=2)

        # 加权标准差
        weighted_var = torch.sum(
            ((x - weighted_mean.unsqueeze(2)) ** 2) * attention_weights, dim=2)
        weighted_std = torch.sqrt(weighted_var + 1e-8)

        # 拼接均值和标准差
        statistics = torch.cat([weighted_mean, weighted_std], dim=1)

        return statistics


class ECAPA_TDNN(nn.Module):
    """
    完整的ECAPA-TDNN模型
    专门用于声纹识别的深度网络
    """

    def __init__(self,
                 input_size: int = 80,
                 emb_dim: int = 192,
                 channels: int = 512,
                 attention_channels: int = 128):
        super().__init__()

        # 输入帧级别特征提取
        self.frame_layer = nn.Conv1d(input_size, channels, 5, padding=2)
        self.frame_bn = nn.BatchNorm1d(channels)

        # ECAPA-TDNN层
        self.layer1 = ECAPA_TDNN_Layer(channels, channels, dilation=1)
        self.layer2 = ECAPA_TDNN_Layer(channels, channels, dilation=2)
        self.layer3 = ECAPA_TDNN_Layer(channels, channels, dilation=3)
        self.layer4 = ECAPA_TDNN_Layer(channels, channels, dilation=1)

        # 特征聚合层
        self.cat_layer = nn.Conv1d(channels * 3, channels * 2, 1)
        self.cat_bn = nn.BatchNorm1d(channels * 2)

        # 注意力统计池化
        self.pooling = AttentiveStatisticsPooling(
            channels * 2, attention_channels)

        # 嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(channels * 4, channels),  # *4因为有mean和std
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, emb_dim)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ECAPA-TDNN前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, input_size]
        Returns:
            声纹嵌入 [batch_size, emb_dim]
        """
        # 维度转换: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)

        # 帧级别特征提取
        x = self.relu(self.frame_bn(self.frame_layer(x)))

        # ECAPA-TDNN层处理
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # 多层特征聚合
        cat_features = torch.cat([x2, x3, x4], dim=1)
        cat_features = self.relu(self.cat_bn(self.cat_layer(cat_features)))

        # 注意力统计池化
        pooled = self.pooling(cat_features)

        # 生成最终嵌入
        embedding = self.embedding(pooled)

        # L2归一化
        return F.normalize(embedding, p=2, dim=1)


class ECAPA_TDNN_Large(nn.Module):
    """
    大型ECAPA-TDNN模型
    更深的网络结构，适合大规模数据训练
    """

    def __init__(self,
                 input_size: int = 80,
                 emb_dim: int = 256,
                 channels: int = 1024,
                 num_layers: int = 8):
        super().__init__()

        # 输入层
        self.input_layer = nn.Sequential(
            nn.Conv1d(input_size, channels, 5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

        # 多个ECAPA-TDNN层
        self.layers = nn.ModuleList()
        dilations = [1, 2, 3, 4, 1, 2, 3, 4]  # 膨胀卷积模式

        for i in range(num_layers):
            self.layers.append(
                ECAPA_TDNN_Layer(
                    channels, channels,
                    dilation=dilations[i % len(dilations)]
                )
            )

        # 特征聚合
        self.aggregation = nn.Sequential(
            nn.Conv1d(channels * num_layers // 2, channels * 2, 1),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU()
        )

        # 注意力池化
        self.pooling = AttentiveStatisticsPooling(channels * 2)

        # 嵌入网络
        self.embedding_net = nn.Sequential(
            nn.Linear(channels * 4, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(channels, channels // 2),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """大型ECAPA-TDNN前向传播"""
        # 维度转换
        x = x.transpose(1, 2)

        # 输入处理
        x = self.input_layer(x)

        # 多层ECAPA-TDNN处理
        layer_outputs = []
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)

        # 聚合中间层特征
        aggregated = torch.cat(layer_outputs[len(layer_outputs)//2:], dim=1)
        aggregated = self.aggregation(aggregated)

        # 注意力池化
        pooled = self.pooling(aggregated)

        # 生成嵌入
        embedding = self.embedding_net(pooled)

        return F.normalize(embedding, p=2, dim=1)


# 测试代码
if __name__ == "__main__":
    print("=== ECAPA-TDNN模型测试 ===")

    # 创建测试数据
    batch_size, seq_len, input_dim = 4, 200, 80
    x = torch.randn(batch_size, seq_len, input_dim)

    print(f"输入形状: {x.shape}")

    # 1. 测试标准ECAPA-TDNN
    print("\n1. 标准ECAPA-TDNN:")
    model = ECAPA_TDNN(input_size=input_dim, emb_dim=192)

    embedding = model(x)
    print(f"声纹嵌入形状: {embedding.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 验证嵌入是否归一化
    embedding_norm = torch.norm(embedding, p=2, dim=1)
    print(f"嵌入L2范数: {embedding_norm.mean().item():.4f} (应该接近1.0)")

    # 2. 测试大型ECAPA-TDNN
    print("\n2. 大型ECAPA-TDNN:")
    large_model = ECAPA_TDNN_Large(input_size=input_dim, emb_dim=256)

    large_embedding = large_model(x)
    print(f"大型模型嵌入形状: {large_embedding.shape}")
    print(f"大型模型参数量: {sum(p.numel() for p in large_model.parameters()):,}")

    # 3. 计算相似度矩阵
    print("\n3. 声纹相似度测试:")
    similarity_matrix = torch.mm(embedding, embedding.t())
    print(f"相似度矩阵形状: {similarity_matrix.shape}")
    print(f"对角线元素(自相似度): {torch.diag(similarity_matrix).mean().item():.4f}")
    print(f"非对角线元素(互相似度): {(similarity_matrix.sum() - torch.diag(similarity_matrix).sum()) / (similarity_matrix.numel() - similarity_matrix.size(0)):.4f}")
