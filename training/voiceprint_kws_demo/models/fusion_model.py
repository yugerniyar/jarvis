"""
声纹+唤醒词融合模型 - 完整实现
这是整个框架的核心模型，展示了如何设计多模态融合架构

学习要点：
1. 多分支架构设计：如何组合不同的特征提取器
2. 注意力机制：跨模态特征融合的关键技术
3. 多任务学习：同时处理多个相关任务的策略
4. 特征对齐：不同维度特征的统一表示方法
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer import ConformerEncoder
from .ecapa_tdnn import ECAPA_TDNN
from .mobilenet import MobileNetV4
from .squeezeformer import SqueezeformerEncoder


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制

    作用：让不同分支的特征能够相互"交流"，学习彼此的互补信息

    原理解析：
    1. Query来自一个模态，Key/Value来自另一个模态
    2. 通过注意力权重，让模态A知道模态B的哪些部分对当前任务重要
    3. 这种交互能够发现单一模态无法捕捉的复杂模式

    实际应用：
    - 让声纹特征关注到唤醒词的关键时刻
    - 让唤醒词检测利用说话人身份信息
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 为什么需要三个不同的投影矩阵？
        # Q(查询)：决定"我想要什么信息"
        # K(键)：决定"我有什么信息"
        # V(值)：决定"具体的信息内容是什么"
        self.w_q = nn.Linear(d_model, d_model)  # Query投影
        self.w_k = nn.Linear(d_model, d_model)  # Key投影
        self.w_v = nn.Linear(d_model, d_model)  # Value投影
        self.w_o = nn.Linear(d_model, d_model)  # 输出投影

        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.d_k)  # 缩放因子，防止注意力分数过大

    def forward(self,
                query: torch.Tensor,      # 查询模态 [B, T1, D]
                key: torch.Tensor,        # 键模态 [B, T2, D]
                value: torch.Tensor,      # 值模态 [B, T2, D]
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        跨模态注意力计算过程详解

        步骤说明：
        1. 线性投影：将输入映射到查询、键、值空间
        2. 多头分割：增加模型的表达能力
        3. 注意力计算：计算query和key的相似度
        4. 权重应用：用注意力权重对value加权求和
        5. 输出投影：将结果映射回原始空间
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_kv = key.shape[1]

        # 步骤1: 线性投影并重塑为多头形式
        Q = self.w_q(query).view(batch_size, seq_len_q,
                                 self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_kv,
                               self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_kv,
                                 self.num_heads, self.d_k).transpose(1, 2)

        # 步骤2: 计算注意力分数
        # 这里计算的是query中每个位置与key中每个位置的相似度
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 步骤3: 应用掩码（如果有的话）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 步骤4: 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 步骤5: 应用注意力权重到value
        context = torch.matmul(attn_weights, V)

        # 步骤6: 重塑并输出投影
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model)

        return self.w_o(context)


class AdaptiveFusion(nn.Module):
    """
    自适应特征融合模块

    为什么需要自适应融合？
    1. 不同任务对各分支特征的依赖程度不同
    2. 不同样本可能需要不同的特征组合策略
    3. 固定权重的融合无法适应动态变化的输入

    技术创新点：
    1. 动态权重生成：根据输入内容自动调整各分支权重
    2. 特征对齐：将不同维度的特征映射到统一空间
    3. 门控机制：控制信息流动，防止不重要特征的干扰
    """

    def __init__(self, input_dims: List[int], output_dim: int, fusion_strategy: str = 'attention'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.fusion_strategy = fusion_strategy

        # 特征对齐层：将不同维度的特征映射到统一维度
        # 为什么需要对齐？因为不同网络输出的特征维度可能不同
        self.alignment_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, output_dim),
                nn.LayerNorm(output_dim),  # 层归一化稳定训练
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for dim in input_dims
        ])

        if fusion_strategy == 'attention':
            # 注意力融合：学习每个分支的重要性权重
            self.attention_fusion = nn.Sequential(
                nn.Linear(sum(input_dims), output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, len(input_dims)),
                nn.Softmax(dim=-1)
            )
        elif fusion_strategy == 'gated':
            # 门控融合：更复杂的特征交互
            self.gate_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(output_dim * len(input_dims), output_dim),
                    nn.Sigmoid()
                ) for _ in input_dims
            ])

        # 融合后的后处理网络
        self.post_fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        自适应融合过程详解

        输入：来自不同分支的特征列表
        输出：融合后的统一特征表示

        关键思想：不是简单相加，而是智能组合
        """
        # 步骤1: 特征对齐 - 将所有特征映射到相同维度空间
        aligned_features = []
        for i, feat in enumerate(features):
            aligned = self.alignment_layers[i](feat)
            aligned_features.append(aligned)

        if self.fusion_strategy == 'attention':
            # 注意力融合策略
            # 根据原始特征计算注意力权重
            concat_features = torch.cat(features, dim=-1)
            weights = self.attention_fusion(
                concat_features)  # [B, num_branches]

            # 加权融合对齐后的特征
            fused = torch.zeros_like(aligned_features[0])
            for i, feat in enumerate(aligned_features):
                fused += weights[:, i:i+1] * feat

        elif self.fusion_strategy == 'gated':
            # 门控融合策略
            all_aligned = torch.cat(aligned_features, dim=-1)
            gated_features = []

            for i, gate_net in enumerate(self.gate_networks):
                gate = gate_net(all_aligned)
                gated_feat = gate * aligned_features[i]
                gated_features.append(gated_feat)

            fused = sum(gated_features) / len(gated_features)

        else:
            # 简单平均融合
            fused = sum(aligned_features) / len(aligned_features)

        # 步骤2: 后处理 - 进一步优化融合特征
        return self.post_fusion(fused)


class MultiTaskHead(nn.Module):
    """
    多任务输出头

    多任务学习的优势：
    1. 共享表示：相关任务可以共享底层特征
    2. 正则化效果：一个任务的学习可以帮助其他任务
    3. 数据效率：有限的数据可以训练多个任务

    设计原则：
    1. 共享主干网络，分离任务特定层
    2. 任务间的损失平衡
    3. 梯度冲突的处理
    """

    def __init__(self,
                 input_dim: int,
                 num_wake_classes: int = 2,      # 唤醒词分类数
                 num_speakers: int = 1000,       # 说话人数量
                 embedding_dim: int = 192):      # 嵌入维度
        super().__init__()

        # 共享的特征处理层
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 任务特定的输出头
        # 1. 唤醒词检测头
        self.wake_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_wake_classes)
        )

        # 2. 说话人识别头
        self.speaker_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_speakers)
        )

        # 3. 声纹验证头（输出相似度分数）
        self.verification_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.Dropout(0.1)
        )

        # 4. 质量评估头（评估音频质量）
        self.quality_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # 输出质量分数
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """多任务前向传播"""
        # 共享特征处理
        shared_feat = self.shared_layers(x)

        # 各任务的输出
        outputs = {
            'wake_logits': self.wake_head(shared_feat),
            'speaker_logits': self.speaker_head(shared_feat),
            'verification_embedding': F.normalize(
                self.verification_head(shared_feat), p=2, dim=1),
            'quality_score': self.quality_head(shared_feat)
        }

        return outputs


class VoiceprintWakeWordModel(nn.Module):
    """
    声纹+唤醒词融合模型 - 完整架构

    整体设计思路：
    1. 多分支特征提取：不同网络捕捉不同层面的音频特征
    2. 跨模态交互：让各分支特征相互学习
    3. 自适应融合：智能组合各分支特征
    4. 多任务输出：同时处理相关的多个任务

    技术亮点：
    1. 端到端训练：所有组件联合优化
    2. 注意力机制：提升关键信息的利用
    3. 残差连接：保证梯度流动
    4. 动态融合：适应不同输入的特点
    """

    def __init__(self,
                 # 基本配置
                 input_dim: int = 80,              # 输入特征维度
                 d_model: int = 256,               # 模型隐藏维度
                 emb_dim: int = 192,               # 嵌入维度
                 num_classes: int = 2,             # 唤醒词类别数
                 num_speakers: int = 1000,         # 说话人数量

                 # 网络架构配置
                 conformer_layers: int = 6,        # Conformer层数
                 conformer_heads: int = 8,         # 注意力头数
                 squeeze_layers: int = 4,          # SqueezeFormer层数
                 squeeze_reduction: int = 4,       # 时间压缩倍数

                 # 训练配置
                 dropout: float = 0.1,             # Dropout概率
                 use_cross_attention: bool = True,  # 是否使用跨模态注意力
                 fusion_strategy: str = 'attention'  # 融合策略
                 ):

        super().__init__()

        self.d_model = d_model
        self.use_cross_attention = use_cross_attention

        # ===================== 多分支特征提取器 =====================

        # 分支1: Conformer - 擅长捕捉全局上下文和时序依赖
        self.conformer = ConformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=conformer_layers,
            num_heads=conformer_heads,
            dropout=dropout
        )

        # 分支2: SqueezeFormer - 高效处理长序列
        self.squeezeformer = SqueezeformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=squeeze_layers,
            reduction_factor=squeeze_reduction,
            dropout=dropout
        )

        # 分支3: ECAPA-TDNN - 专门的声纹特征提取
        self.ecapa_tdnn = ECAPA_TDNN(
            input_size=input_dim,
            emb_dim=emb_dim
        )

        # 分支4: MobileNet - 轻量级特征提取
        self.mobilenet = MobileNetV4(
            input_channels=input_dim,
            num_classes=d_model
        )

        # ===================== 跨模态交互模块 =====================

        if use_cross_attention:
            # 让不同分支的特征能够相互学习
            self.cross_attention_layers = nn.ModuleList([
                CrossModalAttention(d_model, num_heads=8, dropout=dropout)
                for _ in range(2)  # 两层跨模态注意力
            ])

        # ===================== 特征融合模块 =====================

        # 计算融合输入维度
        fusion_input_dims = [d_model, d_model, emb_dim, d_model]

        self.adaptive_fusion = AdaptiveFusion(
            input_dims=fusion_input_dims,
            output_dim=d_model,
            fusion_strategy=fusion_strategy
        )

        # ===================== 多任务输出头 =====================

        self.multi_task_head = MultiTaskHead(
            input_dim=d_model,
            num_wake_classes=num_classes,
            num_speakers=num_speakers,
            embedding_dim=emb_dim
        )

        # ===================== 辅助模块 =====================

        # 特征增强层：提升融合特征质量
        self.feature_enhancement = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),  # GELU激活函数，效果优于ReLU
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        权重初始化策略

        为什么需要好的初始化？
        1. 避免梯度消失/爆炸
        2. 加速训练收敛
        3. 提升最终性能
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化：适用于大多数全连接层
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                # Kaiming初始化：适用于ReLU激活的卷积层
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                # 归一化层的标准初始化
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self,
                x: torch.Tensor,
                return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        """
        模型前向传播 - 完整流程解析

        Args:
            x: 输入音频特征 [batch_size, seq_len, feat_dim]
            return_intermediates: 是否返回中间特征（用于分析和调试）

        Returns:
            包含所有任务输出的字典

        前向传播流程：
        1. 多分支特征提取
        2. 跨模态交互（可选）
        3. 特征池化和对齐
        4. 自适应融合
        5. 特征增强
        6. 多任务输出
        """
        batch_size, seq_len, feat_dim = x.shape

        # ==================== 阶段1: 多分支特征提取 ====================

        # Conformer分支：全局上下文建模
        conformer_out = self.conformer(x)  # [B, T, d_model]

        # SqueezeFormer分支：高效时序建模
        squeeze_out = self.squeezeformer(x)  # [B, T, d_model]

        # ECAPA-TDNN分支：声纹特征提取
        voice_embedding = self.ecapa_tdnn(x)  # [B, emb_dim]

        # MobileNet分支：轻量级特征
        # 需要调整输入维度格式
        mobile_out = self.mobilenet(x)  # [B, d_model]

        # ==================== 阶段2: 跨模态交互 ====================

        if self.use_cross_attention:
            # 让Conformer和SqueezeFormer特征相互学习
            for cross_attn in self.cross_attention_layers:
                # Conformer特征作为查询，SqueezeFormer特征作为键值
                conformer_enhanced = cross_attn(
                    query=conformer_out,
                    key=squeeze_out,
                    value=squeeze_out
                )

                # SqueezeFormer特征作为查询，Conformer特征作为键值
                squeeze_enhanced = cross_attn(
                    query=squeeze_out,
                    key=conformer_out,
                    value=conformer_out
                )

                # 残差连接：保留原始信息
                conformer_out = conformer_out + conformer_enhanced
                squeeze_out = squeeze_out + squeeze_enhanced

        # ==================== 阶段3: 特征池化和对齐 ====================

        # 时序特征池化：将变长序列转为固定长度
        conformer_pooled = torch.mean(conformer_out, dim=1)    # 平均池化
        squeeze_pooled = torch.mean(squeeze_out, dim=1)        # 平均池化

        # 也可以尝试其他池化策略：
        # conformer_pooled = torch.max(conformer_out, dim=1)[0]  # 最大池化
        # conformer_pooled = conformer_out[:, -1, :]             # 最后时刻

        # ==================== 阶段4: 自适应融合 ====================

        # 将所有分支特征收集到列表中
        branch_features = [
            conformer_pooled,   # [B, d_model]
            squeeze_pooled,     # [B, d_model]
            voice_embedding,    # [B, emb_dim]
            mobile_out          # [B, d_model]
        ]

        # 自适应融合：智能组合各分支特征
        fused_features = self.adaptive_fusion(branch_features)  # [B, d_model]

        # ==================== 阶段5: 特征增强 ====================

        # 进一步优化融合后的特征
        enhanced_features = self.feature_enhancement(fused_features)

        # 残差连接：防止信息丢失
        final_features = fused_features + enhanced_features

        # ==================== 阶段6: 多任务输出 ====================

        # 生成所有任务的输出
        task_outputs = self.multi_task_head(final_features)

        # 添加原始声纹嵌入到输出中
        task_outputs['voice_embedding'] = voice_embedding

        # ==================== 可选: 返回中间特征 ====================

        if return_intermediates:
            task_outputs.update({
                'conformer_features': conformer_pooled,
                'squeeze_features': squeeze_pooled,
                'mobile_features': mobile_out,
                'fused_features': final_features,
                'branch_features': branch_features
            })

        return task_outputs

    def get_model_info(self) -> Dict[str, any]:
        """
        获取模型信息
        用于模型分析和优化
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel()
                               for p in self.parameters() if p.requires_grad)

        # 各分支参数统计
        conformer_params = sum(p.numel() for p in self.conformer.parameters())
        squeeze_params = sum(p.numel()
                             for p in self.squeezeformer.parameters())
        ecapa_params = sum(p.numel() for p in self.ecapa_tdnn.parameters())
        mobile_params = sum(p.numel() for p in self.mobilenet.parameters())

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
            'branch_parameters': {
                'conformer': conformer_params,
                'squeezeformer': squeeze_params,
                'ecapa_tdnn': ecapa_params,
                'mobilenet': mobile_params
            }
        }


# 测试和演示代码
if __name__ == "__main__":
    print("=== 声纹+唤醒词融合模型测试 ===")

    # 创建模型
    model = VoiceprintWakeWordModel(
        input_dim=80,
        d_model=256,
        conformer_layers=4,  # 减少层数加快测试
        squeeze_layers=2,
        use_cross_attention=True,
        fusion_strategy='attention'
    )

    # 模型信息
    model_info = model.get_model_info()
    print(f"\n模型信息：")
    print(f"总参数量: {model_info['total_parameters']:,}")
    print(f"模型大小: {model_info['model_size_mb']:.2f} MB")
    print(f"各分支参数量:")
    for branch, params in model_info['branch_parameters'].items():
        print(f"  {branch}: {params:,}")

    # 测试前向传播
    batch_size, seq_len, feat_dim = 4, 100, 80
    x = torch.randn(batch_size, seq_len, feat_dim)

    print(f"\n输入形状: {x.shape}")

    # 前向传播
    with torch.no_grad():
        outputs = model(x, return_intermediates=True)

    print(f"\n输出信息:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        elif isinstance(value, list):
            print(f"{key}: list of {len(value)} tensors")

    print(f"\n模型架构解析:")
    print("1. 多分支设计：Conformer(全局) + SqueezeFormer(高效) + ECAPA-TDNN(声纹) + MobileNet(轻量)")
    print("2. 跨模态交互：通过注意力机制让不同分支相互学习")
    print("3. 自适应融合：根据输入动态调整各分支权重")
    print("4. 多任务输出：唤醒词检测 + 说话人识别 + 声纹验证 + 质量评估")
