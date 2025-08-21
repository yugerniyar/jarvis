"""
声纹+自定义唤醒词模型训练DEMO - 完整版
技术栈：Conformer、SqueezeFormer、ECAPA-TDNN、MobileNetV4、Whisper、Wav2Vec2
作者：AI Assistant
功能：结合声纹识别和关键词检测的端到端模型

学习重点：
1. 多模态融合：如何有效结合不同类型的音频特征
2. 多任务学习：同时优化多个相关任务的策略
3. 预训练模型集成：如何利用Whisper和Wav2Vec2等预训练模型
4. 渐进式训练：从简单到复杂的训练策略
"""
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# 导入我们的模型组件
try:
    from models.fusion_model import VoiceprintWakeWordModel
    from models.wav2vec2_adapter import Wav2Vec2FeatureExtractor
    from models.whisper_adapter import WhisperFeatureExtractor
except ImportError:
    # 如果模块导入失败，使用当前文件中的实现
    logger = logging.getLogger(__name__)
    logger.warning("无法导入模型模块，使用内置实现")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================= 核心模型架构 =================================


class ConformerBlock(nn.Module):
    """
    Conformer块：结合CNN和Transformer的优势
    - 前馈网络(FFN) + 多头自注意力(MHSA) + 卷积模块 + 后馈网络
    - 使用残差连接和层归一化
    """

    def __init__(self, d_model=256, nhead=8, kernel_size=31, dropout=0.1):
        super().__init__()
        # 第一个前馈网络：扩展特征维度，使用SiLU激活
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_model),  # 层归一化，稳定训练
            nn.Linear(d_model, d_model * 4),  # 扩展到4倍维度
            nn.SiLU(),  # SiLU激活函数，效果优于ReLU
            nn.Dropout(dropout),  # 防止过拟合
            nn.Linear(d_model * 4, d_model),  # 压缩回原维度
        )

        # 多头自注意力机制：捕捉长距离依赖
        self.mhsa = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_mhsa = nn.LayerNorm(d_model)

        # 卷积模块：捕捉局部特征模式
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size,
                      padding=kernel_size//2, groups=d_model),  # 深度卷积
            nn.BatchNorm1d(d_model),  # 批归一化
            nn.SiLU(),  # 激活函数
            nn.Conv1d(d_model, d_model, 1),  # 点卷积
        )
        self.norm_conv = nn.LayerNorm(d_model)

        # 第二个前馈网络
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            输出张量 [batch_size, seq_len, d_model]
        """
        # 1. 第一个FFN with 残差连接
        x = x + 0.5 * self.ff1(x)

        # 2. 多头自注意力 with 残差连接
        attn_out, _ = self.mhsa(x, x, x)
        x = x + attn_out
        x = self.norm_mhsa(x)

        # 3. 卷积模块 with 残差连接
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        x = self.norm_conv(x)

        # 4. 第二个FFN with 残差连接
        x = x + 0.5 * self.ff2(x)

        return self.final_norm(x)


class SqueezeformerBlock(nn.Module):
    """
    SqueezeFormer块：通过时间维度压缩降低计算复杂度
    - 先压缩时间维度 -> 自注意力 -> 恢复时间维度
    - 适合处理长序列音频
    """

    def __init__(self, d_model=256, reduction=4, nhead=8):
        super().__init__()
        # 时间维度压缩：减少计算量
        self.reduce = nn.Conv1d(d_model, d_model, reduction, stride=reduction)

        # 压缩后的自注意力：在压缩空间中进行注意力计算
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

        # 时间维度恢复：转置卷积恢复原始长度
        self.expand = nn.ConvTranspose1d(
            d_model, d_model, reduction, stride=reduction)

        # 残差连接的投影层
        self.residual_proj = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            压缩-注意力-恢复后的特征 [batch_size, seq_len, d_model]
        """
        residual = x  # 保存残差

        # 1. 时间维度压缩 [B, T, D] -> [B, T/4, D]
        x_reduced = self.reduce(x.transpose(1, 2)).transpose(1, 2)

        # 2. 在压缩空间进行自注意力计算
        attn_out, _ = self.attn(x_reduced, x_reduced, x_reduced)
        attn_out = self.norm(attn_out)

        # 3. 恢复原始时间维度 [B, T/4, D] -> [B, T, D]
        x_expanded = self.expand(attn_out.transpose(1, 2)).transpose(1, 2)

        # 4. 残差连接
        residual_proj = self.residual_proj(
            residual.transpose(1, 2)).transpose(1, 2)

        return x_expanded + residual_proj


class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN：专门用于声纹识别的时延神经网络
    - TDNN：时延神经网络，专门处理语音的时序信息
    - ECAPA：增强版，添加了注意力机制和残差连接
    """

    def __init__(self, input_size=80, emb_dim=192, channels=512):
        super().__init__()

        # TDNN层：逐步提取更高层的特征
        self.tdnn1 = nn.Sequential(
            nn.Conv1d(input_size, channels, 5, dilation=1),  # 膨胀卷积
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

        self.tdnn2 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, dilation=2),  # 更大的感受野
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

        self.tdnn3 = nn.Sequential(
            nn.Conv1d(channels, channels, 3, dilation=3),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

        self.tdnn4 = nn.Sequential(
            nn.Conv1d(channels, channels, 1),  # 点卷积
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )

        # 注意力池化：关注重要的时间段
        self.attention = nn.Sequential(
            nn.Conv1d(channels, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, channels, 1),
            nn.Softmax(dim=2)
        )

        # 统计池化：计算均值和标准差
        self.stat_pooling = nn.AdaptiveAvgPool1d(1)

        # 最终嵌入层
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels),  # *2因为拼接了均值和标准差
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(channels, emb_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [batch_size, seq_len, feat_dim]
        Returns:
            声纹嵌入向量 [batch_size, emb_dim]
        """
        # 转换维度：[B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)

        # TDNN特征提取
        x1 = self.tdnn1(x)
        x2 = self.tdnn2(x1)
        x3 = self.tdnn3(x2)
        x4 = self.tdnn4(x3)

        # 注意力加权
        attention_weights = self.attention(x4)
        x_attended = x4 * attention_weights

        # 统计池化：计算加权的均值和标准差
        mean = torch.mean(x_attended, dim=2)
        std = torch.std(x_attended, dim=2)

        # 拼接统计特征
        x_pooled = torch.cat([mean, std], dim=1)

        # 生成最终嵌入并L2归一化
        embedding = self.fc(x_pooled)
        return F.normalize(embedding, p=2, dim=1)


class MobileNetV4Block(nn.Module):
    """
    MobileNetV4块：轻量级CNN，适合移动端部署
    - 深度可分离卷积：大幅减少参数量
    - 倒残差结构：先扩展再压缩
    """

    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio

        # 点卷积扩展通道
        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        ) if expand_ratio != 1 else nn.Identity()

        # 深度卷积：每个通道独立卷积
        self.depthwise = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride=stride,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        # 点卷积压缩通道
        self.project = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)

        if self.use_residual:
            return out + identity
        return out

# ================================= 融合模型 =================================


class VoiceprintWakeWordModel(nn.Module):
    """
    声纹+唤醒词融合模型
    架构设计：
    1. 输入投影：将音频特征映射到统一维度
    2. Conformer分支：处理唤醒词检测任务
    3. SqueezeFormer分支：轻量化的特征提取
    4. ECAPA-TDNN分支：声纹识别
    5. MobileNet分支：移动端优化的特征提取
    6. 多模态融合：结合所有分支的输出
    """

    def __init__(self,
                 input_dim=80,      # 输入特征维度（如Mel频谱图）
                 d_model=256,       # Transformer隐藏层维度
                 emb_dim=192,       # 声纹嵌入维度
                 num_classes=2,     # 分类数（唤醒/非唤醒）
                 num_speakers=1000  # 说话人数量
                 ):
        super().__init__()

        # 输入特征投影：统一不同输入的维度
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Conformer分支：主要用于唤醒词检测
        self.conformer = nn.ModuleList([
            ConformerBlock(d_model) for _ in range(4)  # 4层Conformer
        ])

        # SqueezeFormer分支：轻量化特征提取
        self.squeezeformer = nn.ModuleList([
            SqueezeformerBlock(d_model) for _ in range(2)  # 2层SqueezeFormer
        ])

        # 声纹识别分支
        self.voiceprint = ECAPA_TDNN(input_dim, emb_dim)

        # MobileNet分支：移动端优化
        self.mobile_features = nn.Sequential(
            nn.Conv1d(input_dim, 32, 3, padding=1),
            MobileNetV4Block(32, 64),
            MobileNetV4Block(64, 128),
            nn.AdaptiveAvgPool1d(1)
        )

        # 特征融合层
        fusion_dim = d_model + emb_dim + 128  # Conformer + 声纹 + MobileNet
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 多任务输出头
        self.wake_classifier = nn.Linear(256, num_classes)      # 唤醒词检测
        self.speaker_classifier = nn.Linear(256, num_speakers)  # 说话人识别

        # 声纹验证头（用于开集识别）
        self.verification_head = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入音频特征 [batch_size, seq_len, feat_dim]
            return_embeddings: 是否返回中间嵌入
        Returns:
            包含各种输出的字典
        """
        batch_size = x.size(0)

        # 1. 输入投影
        x_proj = self.input_proj(x)  # [B, T, d_model]

        # 2. Conformer分支处理
        conformer_out = x_proj
        for layer in self.conformer:
            conformer_out = layer(conformer_out)
        conformer_feat = conformer_out.mean(dim=1)  # 时间平均池化

        # 3. SqueezeFormer分支处理
        squeeze_out = x_proj
        for layer in self.squeezeformer:
            squeeze_out = layer(squeeze_out)
        squeeze_feat = squeeze_out.mean(dim=1)

        # 4. 声纹识别分支
        voice_emb = self.voiceprint(x)  # [B, emb_dim]

        # 5. MobileNet分支
        mobile_out = self.mobile_features(x.transpose(1, 2))  # [B, 128, 1]
        mobile_feat = mobile_out.squeeze(-1)  # [B, 128]

        # 6. 特征融合：将所有分支的特征拼接
        # 使用Conformer和Squeeze的加权平均
        context_feat = 0.7 * conformer_feat + 0.3 * squeeze_feat
        fusion_feat = torch.cat([context_feat, voice_emb, mobile_feat], dim=1)

        # 7. 融合层处理
        fused = self.fusion(fusion_feat)

        # 8. 多任务输出
        wake_logits = self.wake_classifier(fused)           # 唤醒词检测
        speaker_logits = self.speaker_classifier(fused)     # 说话人分类
        verification_score = self.verification_head(fused)   # 声纹验证分数

        outputs = {
            'wake_logits': wake_logits,           # 唤醒词分类logits
            'speaker_logits': speaker_logits,     # 说话人分类logits
            'verification_score': verification_score,  # 声纹验证分数
            'voice_embedding': voice_emb,         # 声纹嵌入向量
        }

        if return_embeddings:
            outputs.update({
                'conformer_feat': conformer_feat,
                'squeeze_feat': squeeze_feat,
                'mobile_feat': mobile_feat,
                'fused_feat': fused
            })

        return outputs

# ================================= 数据处理 =================================


class AudioDataset(Dataset):
    """
    音频数据集类
    处理声纹+唤醒词的训练数据
    """

    def __init__(self, data_list, sr=16000, n_mels=80, max_len=160):
        # 数据列表：[(audio_path, wake_label, speaker_id)]
        self.data_list = data_list
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len

        # Mel频谱图提取器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            n_fft=512,
            hop_length=160,
            win_length=400
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        audio_path, wake_label, speaker_id = self.data_list[idx]

        # 加载音频
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            waveform = resampler(waveform)

        # 提取Mel频谱图
        mel_spec = self.mel_transform(waveform).squeeze(0).T  # [T, n_mels]

        # 长度标准化
        if mel_spec.size(0) > self.max_len:
            mel_spec = mel_spec[:self.max_len]
        else:
            # 零填充
            pad_len = self.max_len - mel_spec.size(0)
            mel_spec = F.pad(mel_spec, (0, 0, 0, pad_len))

        return {
            'features': mel_spec.float(),
            'wake_label': torch.tensor(wake_label, dtype=torch.long),
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long)
        }

# ================================= 增强数据处理 =================================


class AdvancedAudioDataset(Dataset):
    """
    高级音频数据集类
    支持多种音频格式和数据增强技术

    数据增强技术：
    1. 时域增强：加噪声、时间拉伸、音量调节
    2. 频域增强：SpecAugment、频率掩码
    3. 混合增强：Mixup、CutMix
    """

    def __init__(self,
                 data_manifest: str,           # 数据清单文件
                 sr: int = 16000,              # 采样率
                 n_mels: int = 80,             # Mel频谱图通道数
                 max_len: int = 160,           # 最大序列长度
                 augment: bool = True,         # 是否使用数据增强
                 noise_prob: float = 0.3,      # 噪声添加概率
                 time_stretch_prob: float = 0.2  # 时间拉伸概率
                 ):
        super().__init__()

        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len
        self.augment = augment
        self.noise_prob = noise_prob
        self.time_stretch_prob = time_stretch_prob

        # 加载数据清单
        self.data_list = self._load_manifest(data_manifest)

        # 音频变换器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_mels=n_mels,
            n_fft=512,
            hop_length=160,
            win_length=400,
            power=2.0
        )

        # 数据增强变换
        if augment:
            self._setup_augmentations()

    def _load_manifest(self, manifest_path: str) -> List[Dict]:
        """
        加载数据清单文件

        清单格式示例：
        {
            "audio_filepath": "/path/to/audio.wav",
            "duration": 2.5,
            "wake_label": 1,
            "speaker_id": "speaker001",
            "text": "你好小爱",
            "noise_level": "clean"
        }
        """
        if not os.path.exists(manifest_path):
            logger.warning(f"清单文件不存在: {manifest_path}")
            return self._create_dummy_data()

        data_list = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line.strip()))

        logger.info(f"加载了 {len(data_list)} 条数据")
        return data_list

    def _create_dummy_data(self) -> List[Dict]:
        """创建模拟数据用于演示"""
        logger.info("创建模拟数据用于演示")
        dummy_data = []

        # 创建模拟数据目录
        os.makedirs("data/dummy_audio", exist_ok=True)

        for i in range(1000):
            # 生成模拟音频数据
            duration = np.random.uniform(1.0, 3.0)
            samples = int(duration * self.sr)

            # 模拟不同类型的音频信号
            if np.random.rand() > 0.5:
                # 模拟唤醒词：正弦波 + 噪声
                t = np.linspace(0, duration, samples)
                freq = np.random.uniform(200, 800)  # 随机频率
                audio = 0.3 * np.sin(2 * np.pi * freq * t) + \
                    0.1 * np.random.randn(samples)
                wake_label = 1
            else:
                # 模拟背景音频：随机噪声
                audio = 0.1 * np.random.randn(samples)
                wake_label = 0

            # 保存音频文件
            audio_path = f"data/dummy_audio/audio_{i:04d}.wav"
            torchaudio.save(audio_path, torch.from_numpy(
                audio).unsqueeze(0), self.sr)

            dummy_data.append({
                "audio_filepath": audio_path,
                "duration": duration,
                "wake_label": wake_label,
                "speaker_id": f"speaker_{np.random.randint(0, 100):03d}",
                "text": "你好小爱" if wake_label else "背景音频",
                "noise_level": np.random.choice(["clean", "noisy", "very_noisy"])
            })

        return dummy_data

    def _setup_augmentations(self):
        """设置数据增强"""
        # SpecAugment参数
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)

    def _apply_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """添加噪声增强"""
        if np.random.rand() < self.noise_prob:
            noise_type = np.random.choice(['gaussian', 'uniform'])
            noise_level = np.random.uniform(0.001, 0.01)

            if noise_type == 'gaussian':
                noise = torch.randn_like(waveform) * noise_level
            else:
                noise = (torch.rand_like(waveform) - 0.5) * 2 * noise_level

            waveform = waveform + noise
        return waveform

    def _apply_volume_change(self, waveform: torch.Tensor) -> torch.Tensor:
        """音量变化增强"""
        if np.random.rand() < 0.3:
            volume_factor = np.random.uniform(0.7, 1.3)
            waveform = waveform * volume_factor
        return waveform

    def _apply_spec_augment(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """应用SpecAugment频域增强"""
        if self.augment and np.random.rand() < 0.5:
            # 随机应用频率掩码和时间掩码
            if np.random.rand() < 0.5:
                mel_spec = self.freq_mask(mel_spec)
            if np.random.rand() < 0.5:
                mel_spec = self.time_mask(mel_spec)
        return mel_spec

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_info = self.data_list[idx]

        # 加载音频
        try:
            waveform, sr = torchaudio.load(data_info["audio_filepath"])
            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(sr, self.sr)
                waveform = resampler(waveform)
        except Exception as e:
            logger.warning(f"加载音频失败 {data_info['audio_filepath']}: {e}")
            # 如果文件不存在，创建模拟数据
            duration = 2.0
            waveform = torch.randn(1, int(duration * self.sr)) * 0.1

        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 应用音频增强
        if self.augment:
            waveform = self._apply_noise(waveform)
            waveform = self._apply_volume_change(waveform)

        # 提取Mel频谱图
        mel_spec = self.mel_transform(waveform).squeeze(0).T  # [T, n_mels]

        # 应用频谱增强
        mel_spec = self._apply_spec_augment(mel_spec.T).T

        # 长度标准化
        if mel_spec.size(0) > self.max_len:
            # 随机截取（训练时）或从中间截取（测试时）
            if self.augment:
                start_idx = np.random.randint(
                    0, mel_spec.size(0) - self.max_len + 1)
            else:
                start_idx = (mel_spec.size(0) - self.max_len) // 2
            mel_spec = mel_spec[start_idx:start_idx + self.max_len]
        else:
            # 零填充
            pad_len = self.max_len - mel_spec.size(0)
            mel_spec = F.pad(mel_spec, (0, 0, 0, pad_len))

        # 处理标签
        wake_label = data_info.get("wake_label", 0)
        speaker_id = data_info.get("speaker_id", "unknown")

        # 如果speaker_id是字符串，转换为数字ID
        if isinstance(speaker_id, str):
            speaker_id = hash(speaker_id) % 1000  # 简单映射到0-999

        return {
            'features': mel_spec.float(),
            'raw_audio': waveform.squeeze(0),  # 原始音频用于Wav2Vec2
            'wake_label': torch.tensor(wake_label, dtype=torch.long),
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),
            'duration': torch.tensor(data_info.get("duration", 0.0), dtype=torch.float),
            'text': data_info.get("text", ""),
            'audio_path': data_info["audio_filepath"]
        }

# ================================= 损失函数 =================================


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    结合唤醒词检测、说话人识别和声纹验证任务
    """

    def __init__(self,
                 wake_weight=1.0,      # 唤醒词检测权重
                 speaker_weight=0.5,   # 说话人识别权重
                 verify_weight=0.3,    # 声纹验证权重
                 margin=0.5):          # AAM-Softmax边界
        super().__init__()
        self.wake_weight = wake_weight
        self.speaker_weight = speaker_weight
        self.verify_weight = verify_weight

        # 各任务的损失函数
        self.wake_loss = nn.CrossEntropyLoss()          # 唤醒词检测
        self.speaker_loss = nn.CrossEntropyLoss()       # 说话人分类
        self.verify_loss = nn.BCEWithLogitsLoss()       # 声纹验证
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=margin)  # 余弦距离损失

    def forward(self, outputs, targets):
        """
        计算多任务损失
        Args:
            outputs: 模型输出字典
            targets: 标签字典
        Returns:
            总损失和各分项损失
        """
        # 唤醒词检测损失
        wake_loss = self.wake_loss(
            outputs['wake_logits'], targets['wake_label'])

        # 说话人识别损失
        speaker_loss = self.speaker_loss(
            outputs['speaker_logits'], targets['speaker_id'])

        # 声纹验证损失（构造正负样本对）
        voice_emb = outputs['voice_embedding']
        batch_size = voice_emb.size(0)

        # 构造相似度矩阵
        similarity_matrix = torch.mm(voice_emb, voice_emb.t())

        # 构造标签矩阵（相同说话人为1，不同为0）
        speaker_labels = targets['speaker_id'].unsqueeze(1)
        same_speaker_mask = (speaker_labels == speaker_labels.t()).float()

        # 声纹验证损失
        verify_targets = same_speaker_mask.view(-1)
        verify_preds = similarity_matrix.view(-1)
        verify_loss = self.verify_loss(verify_preds, verify_targets)

        # 总损失
        total_loss = (self.wake_weight * wake_loss +
                      self.speaker_weight * speaker_loss +
                      self.verify_weight * verify_loss)

        return {
            'total_loss': total_loss,
            'wake_loss': wake_loss,
            'speaker_loss': speaker_loss,
            'verify_loss': verify_loss
        }

# ================================= 高级损失函数 =================================


class AdvancedMultiTaskLoss(nn.Module):
    """
    高级多任务损失函数

    改进点：
    1. 自适应权重调节：根据任务难度自动调整权重
    2. 焦点损失：处理类别不平衡问题
    3. 中心损失：增强声纹嵌入的区分性
    4. 对比损失：改善声纹表示学习
    """

    def __init__(self,
                 # 基础权重
                 wake_weight: float = 1.0,
                 speaker_weight: float = 0.5,
                 verify_weight: float = 0.3,

                 # 焦点损失参数
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,

                 # 中心损失参数
                 center_loss_weight: float = 0.1,
                 num_classes: int = 1000,
                 feat_dim: int = 192,

                 # 自适应权重
                 use_adaptive_weights: bool = True):
        super().__init__()

        self.wake_weight = wake_weight
        self.speaker_weight = speaker_weight
        self.verify_weight = verify_weight
        self.use_adaptive_weights = use_adaptive_weights

        # 基础损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        # 焦点损失参数
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # 中心损失
        self.center_loss_weight = center_loss_weight
        if center_loss_weight > 0:
            self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
            nn.init.kaiming_uniform_(self.centers)

        # 自适应权重（如果启用）
        if use_adaptive_weights:
            self.weight_params = nn.Parameter(torch.ones(3))  # 3个任务的权重参数

    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        焦点损失：解决类别不平衡问题

        公式：FL(p_t) = -α(1-p_t)^γ * log(p_t)
        其中 p_t 是正确类别的预测概率

        适用场景：唤醒词检测中，负样本（非唤醒）远多于正样本（唤醒）
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()

    def center_loss(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        中心损失：增强类内紧凑性

        原理：拉近同类样本的特征，推远不同类样本的特征
        对于声纹识别特别有效，可以让同一说话人的特征更聚集
        """
        batch_size = features.size(0)

        # 获取每个样本对应的中心
        centers_batch = self.centers[targets]  # [batch_size, feat_dim]

        # 计算特征与中心的L2距离
        center_loss = F.mse_loss(features, centers_batch)

        return center_loss

    def contrastive_loss(self, embeddings: torch.Tensor,
                         labels: torch.Tensor,
                         margin: float = 1.0) -> torch.Tensor:
        """
        对比损失：改善嵌入空间的结构

        目标：相同类别的样本应该靠近，不同类别的样本应该远离
        这对声纹识别的嵌入学习非常重要
        """
        batch_size = embeddings.size(0)

        # 计算所有样本对的欧氏距离
        distances = torch.cdist(embeddings, embeddings, p=2)

        # 构造标签矩阵：1表示同类，0表示异类
        labels_expanded = labels.unsqueeze(1)
        same_label_mask = (labels_expanded == labels_expanded.T).float()
        diff_label_mask = 1 - same_label_mask

        # 移除对角线（自己与自己的距离）
        mask = 1 - torch.eye(batch_size, device=embeddings.device)
        same_label_mask *= mask
        diff_label_mask *= mask

        # 正样本损失：相同标签的样本应该靠近
        pos_loss = (distances * same_label_mask).sum() / \
            (same_label_mask.sum() + 1e-8)

        # 负样本损失：不同标签的样本应该远离
        neg_loss = F.relu(margin - distances) * diff_label_mask
        neg_loss = neg_loss.sum() / (diff_label_mask.sum() + 1e-8)

        return pos_loss + neg_loss

    def forward(self, outputs: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失

        设计思路：
        1. 主任务损失：唤醒词检测、说话人识别
        2. 辅助损失：声纹验证、中心损失、对比损失
        3. 权重平衡：自适应或固定权重
        """
        losses = {}

        # 1. 唤醒词检测损失（使用焦点损失处理不平衡）
        wake_loss = self.focal_loss(
            outputs['wake_logits'], targets['wake_label'])
        losses['wake_loss'] = wake_loss

        # 2. 说话人识别损失
        speaker_loss = self.ce_loss(
            outputs['speaker_logits'], targets['speaker_id'])
        losses['speaker_loss'] = speaker_loss

        # 3. 声纹验证损失
        voice_emb = outputs['voice_embedding']
        batch_size = voice_emb.size(0)

        # 构造相似度矩阵和标签
        similarity_matrix = torch.mm(voice_emb, voice_emb.t())
        speaker_labels = targets['speaker_id'].unsqueeze(1)
        same_speaker_mask = (speaker_labels == speaker_labels.T).float()

        verify_targets = same_speaker_mask.view(-1)
        verify_preds = similarity_matrix.view(-1)
        verify_loss = self.bce_loss(verify_preds, verify_targets)
        losses['verify_loss'] = verify_loss

        # 4. 中心损失（如果启用）
        if self.center_loss_weight > 0:
            center_loss = self.center_loss(voice_emb, targets['speaker_id'])
            losses['center_loss'] = center_loss
        else:
            losses['center_loss'] = torch.tensor(0.0, device=voice_emb.device)

        # 5. 对比损失
        contrastive_loss = self.contrastive_loss(
            voice_emb, targets['speaker_id'])
        losses['contrastive_loss'] = contrastive_loss

        # 6. 计算总损失
        if self.use_adaptive_weights:
            # 自适应权重：让模型自动学习任务重要性
            weights = F.softmax(self.weight_params, dim=0)
            total_loss = (weights[0] * wake_loss +
                          weights[1] * speaker_loss +
                          weights[2] * verify_loss +
                          self.center_loss_weight * losses['center_loss'] +
                          0.1 * contrastive_loss)

            # 记录当前权重
            losses['adaptive_weights'] = {
                'wake': weights[0].item(),
                'speaker': weights[1].item(),
                'verify': weights[2].item()
            }
        else:
            # 固定权重
            total_loss = (self.wake_weight * wake_loss +
                          self.speaker_weight * speaker_loss +
                          self.verify_weight * verify_loss +
                          self.center_loss_weight * losses['center_loss'] +
                          0.1 * contrastive_loss)

        losses['total_loss'] = total_loss

        return losses

# ================================= 训练器 =================================


class Trainer:
    """
    模型训练器
    包含完整的训练、验证和保存逻辑
    """

    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )

        # 损失函数
        self.criterion = MultiTaskLoss()

        # 最佳模型指标
        self.best_acc = 0.0

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct_wake = 0
        correct_speaker = 0
        total_samples = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # 数据准备
            features = batch['features']  # [B, T, F]
            wake_labels = batch['wake_label']
            speaker_ids = batch['speaker_id']

            # 前向传播
            outputs = self.model(features)

            # 计算损失
            targets = {
                'wake_label': wake_labels,
                'speaker_id': speaker_ids
            }
            losses = self.criterion(outputs, targets)

            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 统计指标
            total_loss += losses['total_loss'].item()

            # 唤醒词准确率
            wake_pred = outputs['wake_logits'].argmax(dim=1)
            correct_wake += (wake_pred == wake_labels).sum().item()

            # 说话人识别准确率
            speaker_pred = outputs['speaker_logits'].argmax(dim=1)
            correct_speaker += (speaker_pred == speaker_ids).sum().item()

            total_samples += features.size(0)

            # 打印进度
            if batch_idx % 100 == 0:
                logger.info(
                    f'Batch {batch_idx}, Loss: {losses["total_loss"].item():.4f}')

        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        wake_acc = correct_wake / total_samples
        speaker_acc = correct_speaker / total_samples

        return {
            'loss': avg_loss,
            'wake_acc': wake_acc,
            'speaker_acc': speaker_acc
        }

    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct_wake = 0
        correct_speaker = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features']
                wake_labels = batch['wake_label']
                speaker_ids = batch['speaker_id']

                outputs = self.model(features)

                targets = {
                    'wake_label': wake_labels,
                    'speaker_id': speaker_ids
                }
                losses = self.criterion(outputs, targets)

                total_loss += losses['total_loss'].item()

                wake_pred = outputs['wake_logits'].argmax(dim=1)
                correct_wake += (wake_pred == wake_labels).sum().item()

                speaker_pred = outputs['speaker_logits'].argmax(dim=1)
                correct_speaker += (speaker_pred == speaker_ids).sum().item()

                total_samples += features.size(0)

        avg_loss = total_loss / len(self.val_loader)
        wake_acc = correct_wake / total_samples
        speaker_acc = correct_speaker / total_samples

        return {
            'loss': avg_loss,
            'wake_acc': wake_acc,
            'speaker_acc': speaker_acc
        }

# ================================= 主训练流程 =================================


def create_sample_data():
    """创建示例数据用于演示"""
    # 模拟训练数据
    train_data = []
    for i in range(1000):
        # 模拟数据：(音频路径, 唤醒标签, 说话人ID)
        wake_label = np.random.randint(0, 2)  # 0: 非唤醒, 1: 唤醒
        speaker_id = np.random.randint(0, 100)  # 100个说话人
        train_data.append((f"audio_{i}.wav", wake_label, speaker_id))

    return train_data


def main():
    """主训练函数"""
    # 配置参数
    config = {
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 32,
        'epochs': 100,
        'input_dim': 80,
        'd_model': 256,
        'emb_dim': 192,
        'num_classes': 2,
        'num_speakers': 100
    }

    # 创建模型
    model = VoiceprintWakeWordModel(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        emb_dim=config['emb_dim'],
        num_classes=config['num_classes'],
        num_speakers=config['num_speakers']
    )

    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 如果有实际数据，替换这里的示例数据
    # train_data = create_sample_data()
    # train_dataset = AudioDataset(train_data)
    # train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # 简化的训练演示
    batch_size, seq_len, feat_dim = 8, 100, 80
    x = torch.randn(batch_size, seq_len, feat_dim)

    # 创建示例标签
    wake_labels = torch.randint(0, 2, (batch_size,))
    speaker_ids = torch.randint(0, 100, (batch_size,))

    # 前向传播测试
    model.eval()
    with torch.no_grad():
        outputs = model(x, return_embeddings=True)

    logger.info("=== 模型输出信息 ===")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"{key}: {value.shape}")

    # 训练一步演示
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = MultiTaskLoss()

    outputs = model(x)
    targets = {'wake_label': wake_labels, 'speaker_id': speaker_ids}
    losses = criterion(outputs, targets)

    optimizer.zero_grad()
    losses['total_loss'].backward()
    optimizer.step()

    logger.info("=== 训练损失信息 ===")
    for key, value in losses.items():
        logger.info(f"{key}: {value.item():.4f}")

    logger.info("\n=== 模型架构说明 ===")
    logger.info("1. Conformer: 结合CNN和Transformer，擅长捕捉音频的局部和全局特征")
    logger.info("2. SqueezeFormer: 通过时间压缩降低计算复杂度，适合长音频")
    logger.info("3. ECAPA-TDNN: 专门的声纹识别网络，提取说话人特征")
    logger.info("4. MobileNetV4: 轻量级CNN，优化移动端部署")
    logger.info("5. 多任务学习: 同时进行唤醒词检测、说话人识别和声纹验证")


# ================================= 高级训练器 =================================

class AdvancedTrainer:
    """
    高级训练器

    功能特色：
    1. 混合精度训练：加速训练并节省显存
    2. 梯度累积：模拟大批次训练
    3. 早停机制：防止过拟合
    4. 检查点管理：自动保存最佳模型
    5. 学习率调度：智能调整学习率
    6. TensorBoard日志：可视化训练过程
    7. 渐进式训练：逐步增加任务复杂度
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict,
                 device: torch.device = None):

        self.config = config
        self.device = device if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # 模型和数据
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # 损失函数
        self.criterion = AdvancedMultiTaskLoss(
            wake_weight=config.get('wake_weight', 1.0),
            speaker_weight=config.get('speaker_weight', 0.5),
            verify_weight=config.get('verify_weight', 0.3),
            use_adaptive_weights=config.get('use_adaptive_weights', True)
        ).to(self.device)

        # 混合精度训练
        self.use_amp = config.get(
            'use_amp', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        # 梯度累积
        self.accumulate_grad_batches = config.get('accumulate_grad_batches', 1)

        # 训练状态
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_step = 0
        self.patience_counter = 0
        self.max_patience = config.get('patience', 10)

        # 日志和检查点
        self._setup_logging()
        self._setup_checkpoint_dir()

        logger.info(f"训练器初始化完成")
        logger.info(f"设备: {self.device}")
        logger.info(f"混合精度: {self.use_amp}")
        logger.info(f"梯度累积步数: {self.accumulate_grad_batches}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-4)

        if optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")

        logger.info(f"使用优化器: {optimizer_name}, 学习率: {lr}")
        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """创建学习率调度器"""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()

        if scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.get('T_0', 10),
                T_mult=self.config.get('T_mult', 2),
                eta_min=self.config.get('eta_min', 1e-6)
            )
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config.get('gamma', 0.1)
            )
        elif scheduler_name == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        elif scheduler_name == 'warmup':
            # 预热 + 余弦退火
            from torch.optim.lr_scheduler import LambdaLR
            warmup_epochs = self.config.get('warmup_epochs', 5)
            total_epochs = self.config.get('epochs', 100)

            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return epoch / warmup_epochs
                else:
                    return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

            scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            scheduler = None

        if scheduler:
            logger.info(f"使用学习率调度器: {scheduler_name}")
        return scheduler

    def _setup_logging(self):
        """设置日志系统"""
        # TensorBoard日志
        log_dir = os.path.join(
            self.config.get('log_dir', 'logs'),
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # 保存配置到日志目录
        config_path = os.path.join(log_dir, 'config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False,
                      allow_unicode=True)

        # 训练历史记录
        self.train_history = {
            'train_loss': [],
            'train_wake_acc': [],
            'train_speaker_acc': [],
            'val_loss': [],
            'val_wake_acc': [],
            'val_speaker_acc': [],
            'learning_rates': []
        }

        logger.info(f"日志目录: {log_dir}")

    def _setup_checkpoint_dir(self):
        """设置检查点目录"""
        self.checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"检查点目录: {self.checkpoint_dir}")

    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        # 统计指标
        epoch_losses = []
        epoch_wake_acc = []
        epoch_speaker_acc = []

        # 进度条
        try:
            from tqdm import tqdm
            pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        except ImportError:
            pbar = self.train_loader
            logger.info(f"开始训练 Epoch {self.current_epoch}")

        # 梯度累积计数器
        grad_accumulation_count = 0

        for batch_idx, batch in enumerate(pbar):
            # 数据移至设备
            features = batch['features'].to(self.device)
            wake_labels = batch['wake_label'].to(self.device)
            speaker_ids = batch['speaker_id'].to(self.device)

            # 前向传播（使用混合精度）
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(features)

                targets = {
                    'wake_label': wake_labels,
                    'speaker_id': speaker_ids
                }
                losses = self.criterion(outputs, targets)

                # 梯度累积：平均损失
                loss = losses['total_loss'] / self.accumulate_grad_batches

            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            grad_accumulation_count += 1

            # 梯度累积完成后更新参数
            if grad_accumulation_count % self.accumulate_grad_batches == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # 统计指标
            epoch_losses.append(losses['total_loss'].item())

            # 计算准确率
            with torch.no_grad():
                wake_pred = outputs['wake_logits'].argmax(dim=1)
                wake_acc = (wake_pred == wake_labels).float().mean().item()
                epoch_wake_acc.append(wake_acc)

                speaker_pred = outputs['speaker_logits'].argmax(dim=1)
                speaker_acc = (
                    speaker_pred == speaker_ids).float().mean().item()
                epoch_speaker_acc.append(speaker_acc)

            # 更新进度条
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'Loss': f'{losses["total_loss"].item():.4f}',
                    'Wake': f'{wake_acc:.3f}',
                    'Speaker': f'{speaker_acc:.3f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })

            # 记录到TensorBoard
            if batch_idx % 50 == 0:
                self._log_training_step(losses, wake_acc, speaker_acc)

            self.train_step += 1

        # 学习率调度
        if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step()

        # 计算epoch平均指标
        epoch_metrics = {
            'loss': np.mean(epoch_losses),
            'wake_acc': np.mean(epoch_wake_acc),
            'speaker_acc': np.mean(epoch_speaker_acc),
            'lr': self.optimizer.param_groups[0]['lr']
        }

        # 记录到历史
        self.train_history['train_loss'].append(epoch_metrics['loss'])
        self.train_history['train_wake_acc'].append(epoch_metrics['wake_acc'])
        self.train_history['train_speaker_acc'].append(
            epoch_metrics['speaker_acc'])
        self.train_history['learning_rates'].append(epoch_metrics['lr'])

        return epoch_metrics

    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()

        val_losses = []
        val_wake_acc = []
        val_speaker_acc = []
        all_wake_preds = []
        all_wake_labels = []

        with torch.no_grad():
            try:
                from tqdm import tqdm
                val_pbar = tqdm(self.val_loader, desc='Validation')
            except ImportError:
                val_pbar = self.val_loader
                logger.info("开始验证")

            for batch in val_pbar:
                features = batch['features'].to(self.device)
                wake_labels = batch['wake_label'].to(self.device)
                speaker_ids = batch['speaker_id'].to(self.device)

                # 前向传播
                outputs = self.model(features)

                targets = {
                    'wake_label': wake_labels,
                    'speaker_id': speaker_ids
                }
                losses = self.criterion(outputs, targets)

                val_losses.append(losses['total_loss'].item())

                # 计算准确率
                wake_pred = outputs['wake_logits'].argmax(dim=1)
                wake_acc = (wake_pred == wake_labels).float().mean().item()
                val_wake_acc.append(wake_acc)

                speaker_pred = outputs['speaker_logits'].argmax(dim=1)
                speaker_acc = (
                    speaker_pred == speaker_ids).float().mean().item()
                val_speaker_acc.append(speaker_acc)

                # 收集预测结果用于详细分析
                all_wake_preds.extend(wake_pred.cpu().numpy())
                all_wake_labels.extend(wake_labels.cpu().numpy())

        # 计算详细指标
        val_metrics = {
            'loss': np.mean(val_losses),
            'wake_acc': np.mean(val_wake_acc),
            'speaker_acc': np.mean(val_speaker_acc),
            'avg_acc': (np.mean(val_wake_acc) + np.mean(val_speaker_acc)) / 2
        }

        # 计算混淆矩阵和F1分数
        try:
            from sklearn.metrics import classification_report, confusion_matrix
            wake_report = classification_report(
                all_wake_labels, all_wake_preds,
                target_names=['Non-Wake', 'Wake'],
                output_dict=True
            )
            val_metrics['wake_f1'] = wake_report['Wake']['f1-score']
            val_metrics['wake_precision'] = wake_report['Wake']['precision']
            val_metrics['wake_recall'] = wake_report['Wake']['recall']
        except ImportError:
            logger.warning("sklearn未安装，跳过详细指标计算")

        # 学习率调度（如果使用ReduceLROnPlateau）
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_metrics['avg_acc'])

        # 记录到历史
        self.train_history['val_loss'].append(val_metrics['loss'])
        self.train_history['val_wake_acc'].append(val_metrics['wake_acc'])
        self.train_history['val_speaker_acc'].append(
            val_metrics['speaker_acc'])

        return val_metrics

    def _log_training_step(self, losses: Dict, wake_acc: float, speaker_acc: float):
        """记录训练步骤到TensorBoard"""
        # 损失曲线
        self.writer.add_scalar(
            'Train/Total_Loss', losses['total_loss'].item(), self.train_step)
        self.writer.add_scalar(
            'Train/Wake_Loss', losses['wake_loss'].item(), self.train_step)
        self.writer.add_scalar('Train/Speaker_Loss',
                               losses['speaker_loss'].item(), self.train_step)
        self.writer.add_scalar('Train/Verify_Loss',
                               losses['verify_loss'].item(), self.train_step)

        # 准确率曲线
        self.writer.add_scalar('Train/Wake_Accuracy',
                               wake_acc, self.train_step)
        self.writer.add_scalar('Train/Speaker_Accuracy',
                               speaker_acc, self.train_step)

        # 学习率
        self.writer.add_scalar('Train/Learning_Rate',
                               self.optimizer.param_groups[0]['lr'], self.train_step)

        # 自适应权重（如果启用）
        if 'adaptive_weights' in losses:
            for task, weight in losses['adaptive_weights'].items():
                self.writer.add_scalar(
                    f'Train/Adaptive_Weight_{task}', weight, self.train_step)

    def save_checkpoint(self, val_metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'train_history': self.train_history,
            'config': self.config,
            'val_metrics': val_metrics
        }

        # 保存最新检查点
        latest_path = os.path.join(
            self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)

        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(
                self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")

        # 定期保存带时间戳的检查点
        if self.current_epoch % 10 == 0:
            epoch_path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_epoch_{self.current_epoch}.pth'
            )
            torch.save(checkpoint, epoch_path)

    def train(self, num_epochs: int):
        """完整训练流程"""
        logger.info(f"开始训练，共 {num_epochs} 个epoch")
        logger.info(f"训练集大小: {len(self.train_loader)}")
        logger.info(f"验证集大小: {len(self.val_loader)}")

        start_time = time.time()

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # 训练一个epoch
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            # 记录epoch级别的日志
            self._log_epoch_metrics(train_metrics, val_metrics)

            # 检查是否为最佳模型
            current_val_acc = val_metrics['avg_acc']
            is_best = current_val_acc > self.best_val_acc

            if is_best:
                self.best_val_acc = current_val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # 保存检查点
            self.save_checkpoint(val_metrics, is_best)

            # 早停检查
            if self.patience_counter >= self.max_patience:
                logger.info(f"验证准确率连续 {self.max_patience} 个epoch未提升，触发早停")
                break

            # 打印epoch总结
            elapsed_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['avg_acc']:.4f}, "
                f"Best Acc: {self.best_val_acc:.4f}, "
                f"Time: {elapsed_time/60:.1f}min"
            )

        total_time = time.time() - start_time
        logger.info(f"训练完成！总耗时: {total_time/3600:.2f}小时")
        logger.info(f"最佳验证准确率: {self.best_val_acc:.4f}")

        # 关闭TensorBoard writer
        self.writer.close()

    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """记录epoch级别的指标"""
        epoch = self.current_epoch

        # TensorBoard记录
        self.writer.add_scalars('Loss', {
            'Train': train_metrics['loss'],
            'Val': val_metrics['loss']
        }, epoch)

        self.writer.add_scalars('Wake_Accuracy', {
            'Train': train_metrics['wake_acc'],
            'Val': val_metrics['wake_acc']
        }, epoch)

        self.writer.add_scalars('Speaker_Accuracy', {
            'Train': train_metrics['speaker_acc'],
            'Val': val_metrics['speaker_acc']
        }, epoch)

# ================================= 数据加载器创建 =================================


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器"""

    # 训练数据集
    train_dataset = AdvancedAudioDataset(
        data_manifest=config.get('train_manifest', 'data/train_manifest.json'),
        sr=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80),
        max_len=config.get('max_len', 160),
        augment=True,
        noise_prob=config.get('noise_prob', 0.3),
        time_stretch_prob=config.get('time_stretch_prob', 0.2)
    )

    # 验证数据集（不使用增强）
    val_dataset = AdvancedAudioDataset(
        data_manifest=config.get('val_manifest', 'data/val_manifest.json'),
        sr=config.get('sample_rate', 16000),
        n_mels=config.get('n_mels', 80),
        max_len=config.get('max_len', 160),
        augment=False
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )

    logger.info(f"训练数据: {len(train_dataset)} 个样本")
    logger.info(f"验证数据: {len(val_dataset)} 个样本")

    return train_loader, val_loader


def load_config(config_path: str = None) -> Dict:
    """加载配置文件"""
    default_config = {
        # 模型参数
        'input_dim': 80,
        'd_model': 256,
        'emb_dim': 192,
        'num_classes': 2,
        'num_speakers': 1000,

        # 训练参数
        'batch_size': 32,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 100,
        'patience': 10,

        # 优化器和调度器
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'use_amp': True,
        'accumulate_grad_batches': 1,

        # 损失函数权重
        'wake_weight': 1.0,
        'speaker_weight': 0.5,
        'verify_weight': 0.3,
        'use_adaptive_weights': True,

        # 数据参数
        'sample_rate': 16000,
        'n_mels': 80,
        'max_len': 160,
        'noise_prob': 0.3,
        'time_stretch_prob': 0.2,
        'num_workers': 4,

        # 路径配置
        'train_manifest': 'data/train_manifest.json',
        'val_manifest': 'data/val_manifest.json',
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs'
    }

    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
        default_config.update(user_config)
        logger.info(f"加载配置文件: {config_path}")
    else:
        logger.info("使用默认配置")

    return default_config

# ================================= 主训练函数重构 =================================


def main():
    """主训练函数"""
    print("=== 声纹+自唤醒词训练系统 ===")

    # 加载配置
    config = load_config('config.yaml')

    # 创建模型
    model = VoiceprintWakeWordModel(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        emb_dim=config['emb_dim'],
        num_classes=config['num_classes'],
        num_speakers=config['num_speakers']
    )

    # 模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")
    logger.info(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

    # 创建数据加载器
    try:
        train_loader, val_loader = create_data_loaders(config)
    except Exception as e:
        logger.warning(f"创建数据加载器失败: {e}")
        logger.info("使用演示数据进行训练")

        # 创建演示数据
        from torch.utils.data import TensorDataset

        # 模拟数据
        num_samples = 1000
        features = torch.randn(
            num_samples, config['max_len'], config['input_dim'])
        wake_labels = torch.randint(0, 2, (num_samples,))
        speaker_ids = torch.randint(0, 100, (num_samples,))

        # 创建数据集
        train_size = int(0.8 * num_samples)
        train_dataset = TensorDataset(
            features[:train_size],
            wake_labels[:train_size],
            speaker_ids[:train_size]
        )
        val_dataset = TensorDataset(
            features[train_size:],
            wake_labels[train_size:],
            speaker_ids[train_size:]
        )

        def collate_fn(batch):
            features, wake_labels, speaker_ids = zip(*batch)
            return {
                'features': torch.stack(features),
                'wake_label': torch.stack(wake_labels),
                'speaker_id': torch.stack(speaker_ids)
            }

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                                shuffle=False, collate_fn=collate_fn)

    # 创建训练器
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # 开始训练
    try:
        trainer.train(config['epochs'])
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise

    logger.info("训练完成！")
