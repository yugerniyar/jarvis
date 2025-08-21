"""
Whisper模型适配器
将OpenAI Whisper的强大特征提取能力集成到声纹+唤醒词框架中

学习要点：
1. 预训练模型的适配：如何利用大规模预训练的语音模型
2. 特征提取vs微调：什么时候冻结预训练权重，什么时候微调
3. 多尺度特征：如何从不同层提取不同级别的特征
4. 计算效率：如何在保持性能的同时减少计算开销
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 注意：实际使用时需要安装: pip install openai-whisper
# 这里我们实现一个简化的Whisper架构用于演示


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 - Whisper风格实现

    与标准注意力的区别：
    1. 使用了不同的初始化策略
    2. 支持因果掩码（用于自回归生成）
    3. 针对语音信号优化了缩放策略
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head

        # Whisper使用单一线性层然后分割，而不是分开的Q/K/V投影
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Whisper风格的多头注意力

        Args:
            x: 输入张量 [batch, seq_len, d_model]
            mask: 注意力掩码
            kv_cache: 键值缓存（用于推理加速）
        """
        B, T, C = x.size()

        # 计算Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 重塑为多头形式
        q = q.view(B, T, self.n_head, self.d_k).transpose(
            1, 2)  # [B, n_head, T, d_k]
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # 使用缓存的K/V（推理时的优化）
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # 计算注意力分数
        att = (q @ k.transpose(-2, -1)) * self.scale

        # 应用掩码
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))

        # 注意力权重和输出
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # [B, n_head, T, d_k]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # [B, T, C]

        return self.c_proj(y)


class WhisperBlock(nn.Module):
    """
    Whisper Transformer块

    架构特点：
    1. 残差连接在层归一化之前（Pre-LN）
    2. 使用GELU激活函数
    3. 前馈网络的隐藏层是4倍模型维度
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()

        # 自注意力层
        self.attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ln_1 = nn.LayerNorm(d_model)

        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),  # Whisper使用GELU而不是ReLU
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Whisper块的前向传播
        注意：使用Pre-LayerNorm结构
        """
        # 自注意力 + 残差连接
        x = x + self.attn(self.ln_1(x), mask)

        # 前馈网络 + 残差连接
        x = x + self.mlp(self.ln_2(x))

        return x


class WhisperEncoder(nn.Module):
    """
    Whisper编码器
    专门用于音频特征提取

    技术特点：
    1. 卷积预处理：将原始音频转换为token序列
    2. 位置编码：学习到的位置嵌入
    3. 多层Transformer：深度特征提取
    """

    def __init__(self,
                 n_mels: int = 80,          # Mel频谱图通道数
                 n_ctx: int = 1500,         # 最大上下文长度
                 d_model: int = 512,        # 模型维度
                 n_head: int = 8,           # 注意力头数
                 n_layer: int = 6,          # 层数
                 dropout: float = 0.0):
        super().__init__()

        self.n_mels = n_mels
        self.n_ctx = n_ctx
        self.d_model = d_model

        # 卷积预处理：模拟Whisper的音频预处理
        # 这里简化为线性投影，实际Whisper使用2层卷积
        self.conv1 = nn.Conv1d(n_mels, d_model, 3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, 3, stride=2, padding=1)

        # 位置编码：Whisper使用学习到的位置嵌入
        self.positional_embedding = nn.Parameter(torch.randn(n_ctx, d_model))

        # Transformer层
        self.blocks = nn.ModuleList([
            WhisperBlock(d_model, n_head, dropout)
            for _ in range(n_layer)
        ])

        # 最终层归一化
        self.ln_post = nn.LayerNorm(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Whisper编码器前向传播

        Args:
            mel: Mel频谱图 [batch, n_mels, time] 或 [batch, time, n_mels]
        Returns:
            编码后的特征 [batch, seq_len, d_model]
        """
        # 确保输入格式正确
        if mel.dim() == 3 and mel.size(-1) == self.n_mels:
            mel = mel.transpose(1, 2)  # [batch, n_mels, time]

        # 卷积预处理
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x))

        # 转换为序列格式
        x = x.permute(0, 2, 1)  # [batch, seq_len, d_model]

        # 添加位置编码
        seq_len = x.size(1)
        if seq_len <= self.n_ctx:
            x = x + self.positional_embedding[:seq_len]
        else:
            # 如果序列太长，截断或插值位置编码
            pos_emb = F.interpolate(
                self.positional_embedding.unsqueeze(0).transpose(1, 2),
                size=seq_len,
                mode='linear'
            ).transpose(1, 2).squeeze(0)
            x = x + pos_emb

        # 通过Transformer层
        for block in self.blocks:
            x = block(x)

        return self.ln_post(x)


class WhisperFeatureExtractor(nn.Module):
    """
    Whisper特征提取器

    功能：
    1. 从预训练Whisper模型提取特征
    2. 支持多层特征融合
    3. 提供冻结/微调选项
    4. 适配下游任务

    使用场景：
    1. 特征提取：冻结Whisper权重，只用作特征提取器
    2. 微调：允许更新Whisper权重以适应特定任务
    3. 渐进式微调：逐层解冻进行微调
    """

    def __init__(self,
                 # Whisper配置
                 model_size: str = 'base',      # tiny, base, small, medium, large
                 freeze_encoder: bool = True,    # 是否冻结编码器

                 # 特征提取配置
                 extract_layers: List[int] = [-1, -2, -3],  # 提取哪些层的特征
                 feature_fusion: str = 'concat',             # 特征融合方式

                 # 适配层配置
                 output_dim: int = 256,         # 输出特征维度
                 use_adapter: bool = True):     # 是否使用适配层
        super().__init__()

        # Whisper模型配置
        whisper_configs = {
            'tiny': {'d_model': 384, 'n_head': 6, 'n_layer': 4},
            'base': {'d_model': 512, 'n_head': 8, 'n_layer': 6},
            'small': {'d_model': 768, 'n_head': 12, 'n_layer': 12},
            'medium': {'d_model': 1024, 'n_head': 16, 'n_layer': 24},
            'large': {'d_model': 1280, 'n_head': 20, 'n_layer': 32}
        }

        config = whisper_configs[model_size]
        self.d_model = config['d_model']
        self.extract_layers = extract_layers
        self.feature_fusion = feature_fusion

        # 创建Whisper编码器
        self.encoder = WhisperEncoder(
            d_model=config['d_model'],
            n_head=config['n_head'],
            n_layer=config['n_layer']
        )

        # 冻结编码器权重（如果需要）
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # 特征融合层
        if feature_fusion == 'concat':
            fusion_input_dim = self.d_model * len(extract_layers)
        elif feature_fusion == 'weighted_sum':
            fusion_input_dim = self.d_model
            # 学习层权重
            self.layer_weights = nn.Parameter(torch.ones(len(extract_layers)))
        else:
            fusion_input_dim = self.d_model

        # 适配层：将Whisper特征适配到目标维度
        if use_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_input_dim // 2),
                nn.LayerNorm(fusion_input_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_input_dim // 2, output_dim),
                nn.LayerNorm(output_dim)
            )
        else:
            # 简单的维度调整
            self.adapter = nn.Linear(fusion_input_dim, output_dim)

    def extract_layer_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        提取指定层的特征

        这是特征提取的核心：不同层捕捉不同级别的信息
        - 浅层：音素、音调等低级特征
        - 深层：语义、说话人等高级特征
        """
        layer_features = []

        # 卷积预处理
        if x.dim() == 3 and x.size(-1) == self.encoder.n_mels:
            x = x.transpose(1, 2)

        x = F.gelu(self.encoder.conv1(x))
        x = F.gelu(self.encoder.conv2(x))
        x = x.permute(0, 2, 1)

        # 添加位置编码
        seq_len = x.size(1)
        if seq_len <= self.encoder.n_ctx:
            x = x + self.encoder.positional_embedding[:seq_len]

        # 逐层前向传播并收集特征
        all_layer_outputs = [x]  # 包含输入（第0层）

        for block in self.encoder.blocks:
            x = block(x)
            all_layer_outputs.append(x)

        # 应用最终层归一化
        final_output = self.encoder.ln_post(x)
        all_layer_outputs[-1] = final_output

        # 提取指定层的特征
        for layer_idx in self.extract_layers:
            if layer_idx < 0:
                # 负索引：从末尾开始计算
                actual_idx = len(all_layer_outputs) + layer_idx
            else:
                actual_idx = layer_idx

            if 0 <= actual_idx < len(all_layer_outputs):
                layer_features.append(all_layer_outputs[actual_idx])

        return layer_features

    def fuse_features(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """
        融合多层特征

        不同的融合策略有不同的效果：
        1. concat: 保留所有信息，但维度较高
        2. weighted_sum: 学习最优的层权重组合
        3. attention: 动态决定关注哪些层
        """
        if self.feature_fusion == 'concat':
            # 拼接所有层特征
            return torch.cat(layer_features, dim=-1)

        elif self.feature_fusion == 'weighted_sum':
            # 加权求和
            weighted_features = []
            weights = F.softmax(self.layer_weights, dim=0)

            for i, feat in enumerate(layer_features):
                weighted_features.append(weights[i] * feat)

            return sum(weighted_features)

        elif self.feature_fusion == 'mean':
            # 简单平均
            return sum(layer_features) / len(layer_features)

        else:
            # 默认返回最后一层
            return layer_features[-1]

    def forward(self, mel: torch.Tensor,
                return_layer_features: bool = False) -> torch.Tensor:
        """
        Whisper特征提取前向传播

        Args:
            mel: Mel频谱图输入
            return_layer_features: 是否返回各层特征（用于分析）
        """
        # 提取多层特征
        layer_features = self.extract_layer_features(mel)

        # 融合特征
        fused_features = self.fuse_features(layer_features)

        # 通过适配层
        adapted_features = self.adapter(fused_features)

        if return_layer_features:
            return adapted_features, layer_features
        else:
            return adapted_features

    def progressive_unfreeze(self, num_layers: int):
        """
        渐进式解冻

        在微调时，逐步解冻更多层：
        1. 先微调适配层
        2. 再解冻顶层Transformer块
        3. 逐步解冻更深的层
        """
        # 解冻适配层
        for param in self.adapter.parameters():
            param.requires_grad = True

        # 解冻指定数量的顶层
        total_layers = len(self.encoder.blocks)
        unfreeze_start = max(0, total_layers - num_layers)

        for i in range(unfreeze_start, total_layers):
            for param in self.encoder.blocks[i].parameters():
                param.requires_grad = True

        print(f"解冻了顶部 {num_layers} 层")

    def get_feature_info(self) -> Dict[str, any]:
        """获取特征提取器信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        return {
            'model_size': f"{self.d_model}d",
            'extract_layers': self.extract_layers,
            'feature_fusion': self.feature_fusion,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_ratio': 1 - (trainable_params / total_params)
        }


# 测试代码
if __name__ == "__main__":
    print("=== Whisper特征提取器测试 ===")

    # 创建特征提取器
    feature_extractor = WhisperFeatureExtractor(
        model_size='base',
        freeze_encoder=True,
        extract_layers=[-1, -2, -3],
        feature_fusion='weighted_sum',
        output_dim=256
    )

    # 模型信息
    info = feature_extractor.get_feature_info()
    print(f"\n特征提取器信息:")
    print(f"模型大小: {info['model_size']}")
    print(f"提取层: {info['extract_layers']}")
    print(f"融合方式: {info['feature_fusion']}")
    print(f"总参数: {info['total_parameters']:,}")
    print(f"可训练参数: {info['trainable_parameters']:,}")
    print(f"冻结比例: {info['frozen_ratio']:.1%}")

    # 测试前向传播
    batch_size, n_mels, time_steps = 4, 80, 3000
    mel_input = torch.randn(batch_size, time_steps, n_mels)

    print(f"\n输入形状: {mel_input.shape}")

    # 特征提取
    with torch.no_grad():
        features, layer_feats = feature_extractor(
            mel_input, return_layer_features=True)

    print(f"输出特征形状: {features.shape}")
    print(f"提取的层数: {len(layer_feats)}")

    for i, feat in enumerate(layer_feats):
        print(f"  层 {feature_extractor.extract_layers[i]}: {feat.shape}")

    # 测试渐进式解冻
    print(f"\n=== 渐进式微调演示 ===")
    print("初始状态 - 可训练参数:",
          sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad))

    # 解冻顶部2层
    feature_extractor.progressive_unfreeze(2)
    print("解冻2层后 - 可训练参数:",
          sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad))

    print(f"\n=== Whisper在声纹+唤醒词中的作用 ===")
    print("1. 强大的音频理解：利用大规模预训练的语音表示")
    print("2. 多层特征：从不同抽象层次捕捉音频信息")
    print("3. 迁移学习：将通用语音知识迁移到特定任务")
    print("4. 计算效率：通过冻结大部分权重减少训练开销")
