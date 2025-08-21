"""
Wav2Vec2模型适配器 - 完整实现
利用Facebook的Wav2Vec2自监督预训练模型进行特征提取

学习要点：
1. 自监督学习：如何利用无标签数据学习通用音频表示
2. 对比学习：Wav2Vec2的核心训练机制
3. 量化编码：离散表示学习的概念
4. 预训练-微调范式：如何有效利用预训练模型

Wav2Vec2 vs Whisper:
- Wav2Vec2: 专注于音频的自监督表示学习，更适合声纹识别
- Whisper: 有监督的语音转文本训练，更适合语义理解
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalConvEmbedding(nn.Module):
    """
    位置卷积嵌入

    Wav2Vec2的创新：用卷积而不是正弦位置编码
    优势：
    1. 更适合音频的连续性特点
    2. 能够捕捉局部时序模式
    3. 参数共享减少过拟合
    """

    def __init__(self, d_model: int, kernel_size: int = 128, groups: int = 16):
        super().__init__()

        # 使用分组卷积减少参数量
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            位置编码后的特征
        """
        # [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)

        # 位置卷积
        x = self.conv(x)

        # 移除填充导致的多余长度
        x = x[:, :, :-1] if x.size(2) % 2 == 1 else x

        # [B, D, T] -> [B, T, D]
        x = x.transpose(1, 2)

        return self.dropout(F.gelu(x))


class Wav2Vec2FeatureEncoder(nn.Module):
    """
    Wav2Vec2特征编码器

    功能：将原始音频波形转换为特征序列
    架构：多层卷积 + 批归一化 + GELU激活

    设计原理：
    1. 逐步降采样：模拟人类听觉系统的频率分析
    2. 特征抽象：从原始波形到高级音频特征
    3. 时序压缩：减少序列长度提高效率
    """

    def __init__(self,
                 # (out_channels, kernel_size, stride)
                 conv_layers: List[Tuple[int, int, int]] = None,
                 dropout: float = 0.0):
        super().__init__()

        # 默认卷积层配置（类似Wav2Vec2-Base）
        if conv_layers is None:
            conv_layers = [
                (512, 10, 5),    # 第1层：大幅降采样
                (512, 3, 2),     # 第2层：继续降采样
                (512, 3, 2),     # 第3层
                (512, 3, 2),     # 第4层
                (512, 3, 2),     # 第5层
                (512, 2, 2),     # 第6层
                (512, 2, 2),     # 第7层：最终特征维度
            ]

        self.conv_layers = nn.ModuleList()
        in_channels = 1  # 输入是单声道音频

        for out_channels, kernel_size, stride in conv_layers:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        bias=False
                    ),
                    nn.Dropout(dropout),
                    nn.GroupNorm(
                        num_groups=out_channels,
                        num_channels=out_channels,
                        affine=True
                    ),
                    nn.GELU()
                )
            )
            in_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        特征编码前向传播

        Args:
            x: 原始音频波形 [batch, time] 或预计算特征 [batch, time, feat]
        Returns:
            编码后的特征 [batch, seq_len, d_model]
        """
        # 处理不同输入格式
        if x.dim() == 2:
            # 原始音频：[batch, time] -> [batch, 1, time]
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            # 特征输入：[batch, time, feat] -> [batch, feat, time]
            x = x.transpose(1, 2)

        # 通过卷积层
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # 转换为序列格式：[batch, d_model, seq_len] -> [batch, seq_len, d_model]
        return x.transpose(1, 2)


class Wav2Vec2TransformerLayer(nn.Module):
    """
    Wav2Vec2 Transformer层

    与标准Transformer的区别：
    1. 使用位置卷积嵌入
    2. 特定的归一化策略
    3. 针对音频优化的注意力模式
    """

    def __init__(self,
                 d_model: int = 768,
                 n_head: int = 12,
                 d_ff: int = 3072,
                 dropout: float = 0.1,
                 activation: str = 'gelu'):
        super().__init__()

        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True)

        # 前馈网络
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'relu':
            act_fn = nn.ReLU()
        else:
            act_fn = nn.SiLU()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Transformer层前向传播
        使用Post-LayerNorm结构（与BERT类似）
        """
        # 自注意力 + 残差连接
        residual = x
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(residual + self.dropout(attn_out))

        # 前馈网络 + 残差连接
        residual = x
        ff_out = self.feed_forward(x)
        x = self.norm2(residual + ff_out)

        return x


class Wav2Vec2Encoder(nn.Module):
    """
    完整的Wav2Vec2编码器

    架构组成：
    1. 特征编码器：原始音频 -> 特征序列
    2. 位置编码：添加时序信息
    3. Transformer层：深度特征提取
    4. 层归一化：稳定输出
    """

    def __init__(self,
                 # 特征编码器参数
                 conv_layers: List[Tuple[int, int, int]] = None,

                 # Transformer参数
                 d_model: int = 768,
                 n_head: int = 12,
                 n_layer: int = 12,
                 d_ff: int = 3072,

                 # 其他参数
                 dropout: float = 0.1,
                 max_positions: int = 1024):
        super().__init__()

        self.d_model = d_model

        # 特征编码器
        self.feature_encoder = Wav2Vec2FeatureEncoder(conv_layers, dropout)

        # 特征投影（如果编码器输出维度与模型维度不匹配）
        encoder_output_dim = 512  # 来自conv_layers的最后一层
        if encoder_output_dim != d_model:
            self.feature_projection = nn.Linear(encoder_output_dim, d_model)
        else:
            self.feature_projection = nn.Identity()

        # 位置编码
        self.pos_conv_embed = PositionalConvEmbedding(d_model)

        # Transformer层
        self.layers = nn.ModuleList([
            Wav2Vec2TransformerLayer(d_model, n_head, d_ff, dropout)
            for _ in range(n_layer)
        ])

        # 输出层归一化
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码器前向传播

        Args:
            x: 输入音频 [batch, time] 或特征 [batch, time, feat]
            attention_mask: 注意力掩码
        Returns:
            编码后的特征 [batch, seq_len, d_model]
        """
        # 1. 特征编码
        x = self.feature_encoder(x)  # [batch, seq_len, encoder_dim]

        # 2. 特征投影
        x = self.feature_projection(x)  # [batch, seq_len, d_model]

        # 3. 位置编码
        x = x + self.pos_conv_embed(x)
        x = self.dropout(x)

        # 4. 通过Transformer层
        for layer in self.layers:
            x = layer(x, key_padding_mask=attention_mask)

        # 5. 输出归一化
        return self.layer_norm(x)


class Wav2Vec2FeatureExtractor(nn.Module):
    """
    Wav2Vec2特征提取器 - 主接口类

    功能：
    1. 从预训练Wav2Vec2模型提取特征
    2. 支持多层特征融合
    3. 提供冻结/微调选项
    4. 适配下游任务

    使用模式：
    1. 特征提取模式：冻结所有权重，仅用作特征提取器
    2. 微调模式：允许更新部分或全部权重
    3. 渐进微调：逐层解冻进行训练
    """

    def __init__(self,
                 # 模型配置
                 model_size: str = 'base',           # base, large, xlsr
                 freeze_feature_encoder: bool = True,  # 冻结特征编码器
                 freeze_transformer: bool = True,     # 冻结Transformer

                 # 特征提取配置
                 extract_layers: List[int] = [-4, -3, -2, -1],  # 提取层
                 pooling_mode: str = 'mean',          # 池化方式

                 # 适配层配置
                 output_dim: int = 256,               # 输出维度
                 use_weighted_layer_sum: bool = True,  # 加权层融合
                 ):
        super().__init__()

        # 模型配置字典
        model_configs = {
            'base': {
                'd_model': 768,
                'n_head': 12,
                'n_layer': 12,
                'd_ff': 3072
            },
            'large': {
                'd_model': 1024,
                'n_head': 16,
                'n_layer': 24,
                'd_ff': 4096
            },
            'xlsr': {
                'd_model': 1024,
                'n_head': 16,
                'n_layer': 24,
                'd_ff': 4096
            }
        }

        config = model_configs[model_size]
        self.d_model = config['d_model']
        self.extract_layers = extract_layers
        self.pooling_mode = pooling_mode

        # 创建Wav2Vec2编码器
        self.wav2vec2 = Wav2Vec2Encoder(
            d_model=config['d_model'],
            n_head=config['n_head'],
            n_layer=config['n_layer'],
            d_ff=config['d_ff']
        )

        # 冻结权重
        if freeze_feature_encoder:
            for param in self.wav2vec2.feature_encoder.parameters():
                param.requires_grad = False
            for param in self.wav2vec2.feature_projection.parameters():
                param.requires_grad = False

        if freeze_transformer:
            for param in self.wav2vec2.layers.parameters():
                param.requires_grad = False
            for param in self.wav2vec2.layer_norm.parameters():
                param.requires_grad = False

        # 加权层融合
        if use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(
                torch.ones(len(extract_layers)) / len(extract_layers)
            )
        else:
            self.register_parameter('layer_weights', None)

        # 适配层
        self.adapter = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model // 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def extract_hidden_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        提取所有隐藏层状态

        这是理解模型表示的关键：不同层学习不同级别的特征
        - 浅层：音素级别的特征
        - 中层：音节、词级别的特征  
        - 深层：语义、说话人级别的特征
        """
        # 特征编码
        x = self.wav2vec2.feature_encoder(x)
        x = self.wav2vec2.feature_projection(x)

        # 位置编码
        x = x + self.wav2vec2.pos_conv_embed(x)
        x = self.wav2vec2.dropout(x)

        # 收集所有层的输出
        hidden_states = [x]  # 第0层（输入层）

        for layer in self.wav2vec2.layers:
            x = layer(x)
            hidden_states.append(x)

        # 最终层归一化
        final_hidden = self.wav2vec2.layer_norm(x)
        hidden_states[-1] = final_hidden

        return hidden_states

    def weighted_layer_sum(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        加权层融合

        学习最优的层权重组合，自动发现哪些层对当前任务最重要
        """
        if self.layer_weights is None:
            # 简单平均
            return torch.stack(hidden_states).mean(dim=0)

        # 归一化权重
        stacked_hidden_states = torch.stack(hidden_states, dim=0)
        norm_weights = F.softmax(self.layer_weights, dim=0)

        # 加权求和
        weighted_hidden = (norm_weights.unsqueeze(1).unsqueeze(1).unsqueeze(1) *
                           stacked_hidden_states).sum(dim=0)

        return weighted_hidden

    def pool_features(self, hidden_states: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        特征池化

        将变长序列转换为固定长度表示
        支持多种池化策略
        """
        if self.pooling_mode == 'mean':
            if attention_mask is not None:
                # 掩码平均池化
                mask_expanded = attention_mask.unsqueeze(-1).float()
                hidden_states = hidden_states * mask_expanded
                pooled = hidden_states.sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                pooled = hidden_states.mean(dim=1)

        elif self.pooling_mode == 'max':
            pooled = hidden_states.max(dim=1)[0]

        elif self.pooling_mode == 'cls':
            # 使用第一个token（如果有特殊CLS token）
            pooled = hidden_states[:, 0]

        elif self.pooling_mode == 'last':
            # 使用最后一个token
            if attention_mask is not None:
                # 找到每个序列的实际最后位置
                last_indices = attention_mask.sum(dim=1) - 1
                pooled = hidden_states[torch.arange(
                    hidden_states.size(0)), last_indices]
            else:
                pooled = hidden_states[:, -1]
        else:
            # 默认均值池化
            pooled = hidden_states.mean(dim=1)

        return pooled

    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_all_hiddens: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        特征提取前向传播

        Args:
            x: 输入音频 [batch, time] 或特征 [batch, time, feat]
            attention_mask: 注意力掩码
            return_all_hiddens: 是否返回所有隐藏状态
        """
        # 提取所有隐藏状态
        all_hidden_states = self.extract_hidden_states(x)

        # 选择指定层
        selected_hiddens = []
        for layer_idx in self.extract_layers:
            if layer_idx < 0:
                actual_idx = len(all_hidden_states) + layer_idx
            else:
                actual_idx = layer_idx

            if 0 <= actual_idx < len(all_hidden_states):
                selected_hiddens.append(all_hidden_states[actual_idx])

        # 加权融合选中的层
        if len(selected_hiddens) > 1:
            fused_hidden = self.weighted_layer_sum(selected_hiddens)
        else:
            fused_hidden = selected_hiddens[0]

        # 池化
        pooled_features = self.pool_features(fused_hidden, attention_mask)

        # 适配层
        adapted_features = self.adapter(pooled_features)

        if return_all_hiddens:
            return adapted_features, all_hidden_states
        else:
            return adapted_features

    def progressive_unfreeze(self, num_layers: int):
        """
        渐进式解冻

        微调策略：
        1. 阶段1：只训练适配层
        2. 阶段2：解冻顶层Transformer
        3. 阶段3：逐步解冻更多层
        4. 阶段4：全模型微调
        """
        # 解冻适配层
        for param in self.adapter.parameters():
            param.requires_grad = True

        # 解冻层权重（如果使用）
        if self.layer_weights is not None:
            self.layer_weights.requires_grad = True

        # 解冻指定数量的顶层Transformer
        total_layers = len(self.wav2vec2.layers)
        unfreeze_start = max(0, total_layers - num_layers)

        for i in range(unfreeze_start, total_layers):
            for param in self.wav2vec2.layers[i].parameters():
                param.requires_grad = True

        # 如果解冻了Transformer层，也解冻最终的层归一化
        if num_layers > 0:
            for param in self.wav2vec2.layer_norm.parameters():
                param.requires_grad = True

        print(f"解冻了 {num_layers} 个Transformer层")

    def get_layer_weights(self) -> Optional[torch.Tensor]:
        """获取学习到的层权重"""
        if self.layer_weights is not None:
            return F.softmax(self.layer_weights, dim=0)
        return None

    def get_model_info(self) -> Dict[str, any]:
        """获取模型详细信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        # 各组件参数统计
        feature_encoder_params = sum(
            p.numel() for p in self.wav2vec2.feature_encoder.parameters())
        transformer_params = sum(p.numel()
                                 for p in self.wav2vec2.layers.parameters())
        adapter_params = sum(p.numel() for p in self.adapter.parameters())

        return {
            'model_size': f"{self.d_model}d",
            'extract_layers': self.extract_layers,
            'pooling_mode': self.pooling_mode,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_ratio': 1 - (trainable_params / total_params),
            'component_parameters': {
                'feature_encoder': feature_encoder_params,
                'transformer': transformer_params,
                'adapter': adapter_params
            },
            'layer_weights': self.get_layer_weights()
        }


# 测试和演示代码
if __name__ == "__main__":
    print("=== Wav2Vec2特征提取器测试 ===")

    # 创建特征提取器
    feature_extractor = Wav2Vec2FeatureExtractor(
        model_size='base',
        freeze_feature_encoder=True,
        freeze_transformer=True,
        extract_layers=[-4, -3, -2, -1],
        pooling_mode='mean',
        output_dim=256,
        use_weighted_layer_sum=True
    )

    # 模型信息
    info = feature_extractor.get_model_info()
    print(f"\n=== 模型信息 ===")
    print(f"模型大小: {info['model_size']}")
    print(f"提取层: {info['extract_layers']}")
    print(f"池化方式: {info['pooling_mode']}")
    print(f"总参数: {info['total_parameters']:,}")
    print(f"可训练参数: {info['trainable_parameters']:,}")
    print(f"冻结比例: {info['frozen_ratio']:.1%}")

    print(f"\n各组件参数分布:")
    for component, params in info['component_parameters'].items():
        print(f"  {component}: {params:,}")

    if info['layer_weights'] is not None:
        print(f"\n层权重: {info['layer_weights'].detach().numpy()}")

    # 测试前向传播
    print(f"\n=== 前向传播测试 ===")

    # 模拟音频输入
    batch_size, seq_len, feat_dim = 4, 1000, 80  # 假设输入是预计算的特征
    audio_input = torch.randn(batch_size, seq_len, feat_dim)

    print(f"输入形状: {audio_input.shape}")

    # 特征提取
    with torch.no_grad():
        features, all_hiddens = feature_extractor(
            audio_input, return_all_hiddens=True)

    print(f"输出特征形状: {features.shape}")
    print(f"隐藏状态层数: {len(all_hiddens)}")

    for i, hidden in enumerate(all_hiddens):
        print(f"  层 {i}: {hidden.shape}")

    # 测试渐进式微调
    print(f"\n=== 渐进式微调演示 ===")

    print("初始可训练参数:",
          sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad))

    # 解冻顶部3层
    feature_extractor.progressive_unfreeze(3)
    print("解冻3层后可训练参数:",
          sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad))

    # 测试不同池化方式
    print(f"\n=== 不同池化方式对比 ===")

    pooling_modes = ['mean', 'max', 'last']
    for mode in pooling_modes:
        test_extractor = Wav2Vec2FeatureExtractor(
            pooling_mode=mode,
            output_dim=256
        )

        with torch.no_grad():
            pooled_feat = test_extractor(audio_input)

        print(f"{mode} 池化输出形状: {pooled_feat.shape}")

    print(f"\n=== Wav2Vec2的优势和应用 ===")
    print("1. 自监督预训练：无需标注数据即可学习通用音频表示")
    print("2. 对比学习：通过掩码预测任务学习上下文相关表示")
    print("3. 多语言支持：XLSR模型支持100+语言")
    print("4. 声纹识别：预训练表示非常适合说话人相关任务")
    print("5. 低资源场景：在少量标注数据下也能取得好效果")

    print(f"\n=== 与其他模型的对比 ===")
    print("Wav2Vec2 vs Whisper:")
    print("- Wav2Vec2: 自监督学习，更适合声纹、情感等任务")
    print("- Whisper: 有监督学习，更适合语音识别和翻译")
    print("\nWav2Vec2 vs ECAPA-TDNN:")
    print("- Wav2Vec2: 通用预训练模型，需要适配")
    print("- ECAPA-TDNN: 专门设计的声纹网络，针对性强")
