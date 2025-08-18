# -*- coding: utf-8 -*-
"""
自定义声纹识别模型训练框架
支持 ECAPA-TDNN、ResNetSE、WavLM 等多种架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import yaml
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils.logger import setup_logger

logger = setup_logger(__name__)

class ECAPA_TDNN(nn.Module):
    """
    ECAPA-TDNN 声纹识别模型
    
    Paper: ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN
    """
    
    def __init__(self, 
                 input_dim: int = 80,  # Mel频谱特征维度
                 channels: int = 512,   # 中间层通道数
                 embedding_dim: int = 192,  # 嵌入向量维度
                 num_speakers: Optional[int] = None):  # 说话人数量（分类任务）
        super(ECAPA_TDNN, self).__init__()
        
        self.input_dim = input_dim
        self.channels = channels
        self.embedding_dim = embedding_dim
        self.num_speakers = num_speakers
        
        # Frame-level 特征提取层
        self.frame_conv = nn.Conv1d(input_dim, channels, kernel_size=5, dilation=1, padding=2)
        
        # ECAPA-TDNN 块
        self.ecapa_blocks = nn.ModuleList([
            ECAPA_Block(channels, channels, kernel_size=3, dilation=2),
            ECAPA_Block(channels, channels, kernel_size=3, dilation=3),
            ECAPA_Block(channels, channels, kernel_size=3, dilation=4),
            ECAPA_Block(channels, channels, kernel_size=1, dilation=1),
        ])
        
        # 通道注意力聚合
        self.channel_aggregation = nn.Conv1d(channels * 4, channels * 2, kernel_size=1)
        
        # 统计池化层
        self.stat_pooling = AttentiveStatisticsPooling(channels * 2, 128)
        
        # 嵌入层
        self.embedding = nn.Sequential(
            nn.Linear(channels * 4, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
        )
        
        # 分类层（如果需要分类任务）
        if num_speakers:
            self.classifier = nn.Linear(embedding_dim, num_speakers)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim, time_steps]
            
        Returns:
            embedding: 声纹嵌入向量
            logits: 分类输出（如果有分类层）
        """
        # Frame-level 特征提取
        x = self.frame_conv(x)
        x = F.relu(x)
        
        # ECAPA-TDNN 块处理
        residual_outputs = []
        for block in self.ecapa_blocks:
            x = block(x)
            residual_outputs.append(x)
        
        # 特征聚合
        x = torch.cat(residual_outputs, dim=1)  # 通道维度拼接
        x = self.channel_aggregation(x)
        
        # 统计池化
        x = self.stat_pooling(x)
        
        # 生成嵌入向量
        embedding = self.embedding(x)
        
        if self.num_speakers and hasattr(self, 'classifier'):
            logits = self.classifier(embedding)
            return embedding, logits
        else:
            return embedding

class ECAPA_Block(nn.Module):
    """ECAPA-TDNN 基础块"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ECAPA_Block, self).__init__()
        
        # 1D 卷积
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 
                              kernel_size=kernel_size, 
                              dilation=dilation,
                              padding=dilation)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        # SE 注意力模块
        self.se_block = SEBlock(out_channels)
        
        # 残差连接
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        
        # 1x1 卷积
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 带扩张的卷积
        out = F.relu(self.bn2(self.conv2(out)))
        
        # 1x1 卷积
        out = self.bn3(self.conv3(out))
        
        # SE 注意力
        out = self.se_block(out)
        
        # 残差连接
        if self.shortcut:
            residual = self.shortcut(residual)
        
        out += residual
        out = F.relu(out)
        
        return out

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力模块"""
    
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        
        # 全局平均池化
        y = self.global_pool(x).view(b, c)
        
        # FC 层 + 激活
        y = self.fc(y).view(b, c, 1)
        
        # 通道注意力
        return x * y.expand_as(x)

class AttentiveStatisticsPooling(nn.Module):
    """注意力统计池化层"""
    
    def __init__(self, input_dim, attention_dim):
        super(AttentiveStatisticsPooling, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, channels, time]
        Returns:
            pooled: [batch, channels * 2]  # mean + std
        """
        # 转置到 [batch, time, channels]
        x = x.transpose(1, 2)
        
        # 计算注意力权重
        attn_weights = self.attention(x)  # [batch, time, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加权平均
        weighted_mean = torch.sum(x * attn_weights, dim=1)  # [batch, channels]
        
        # 加权标准差
        weighted_var = torch.sum((x - weighted_mean.unsqueeze(1))**2 * attn_weights, dim=1)
        weighted_std = torch.sqrt(weighted_var + 1e-8)
        
        # 拼接统计特征
        pooled = torch.cat([weighted_mean, weighted_std], dim=1)
        
        return pooled

class ResNetSE(nn.Module):
    """
    ResNet + SE 声纹识别模型
    更轻量级的选择
    """
    
    def __init__(self, 
                 input_dim: int = 80,
                 embedding_dim: int = 256,
                 num_speakers: Optional[int] = None):
        super(ResNetSE, self).__init__()
        
        # 输入卷积
        self.input_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # ResNet 块
        self.resnet_blocks = nn.ModuleList([
            ResNetBlock(32, 64),
            ResNetBlock(64, 128),
            ResNetBlock(128, 256),
            ResNetBlock(256, 512),
        ])
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 嵌入层
        self.embedding = nn.Linear(512, embedding_dim)
        
        # 分类层
        if num_speakers:
            self.classifier = nn.Linear(embedding_dim, num_speakers)
    
    def forward(self, x):
        # 添加通道维度
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # [batch, 1, freq, time]
        
        # 输入卷积
        x = F.relu(self.input_conv(x))
        
        # ResNet 块
        for block in self.resnet_blocks:
            x = block(x)
        
        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 嵌入
        embedding = self.embedding(x)
        
        if hasattr(self, 'classifier'):
            logits = self.classifier(embedding)
            return embedding, logits
        else:
            return embedding

class ResNetBlock(nn.Module):
    """ResNet 基础块"""
    
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE 模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )
        
        # 跳跃连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # SE 注意力
        se_weight = self.se(out)
        out = out * se_weight
        
        out += residual
        out = F.relu(out)
        
        return out

# 数据集类
class SpeakerDataset(Dataset):
    """声纹识别数据集"""
    
    def __init__(self, 
                 data_dir: str,
                 speaker_list: List[str],
                 max_length: int = 16000 * 4,  # 4秒
                 sample_rate: int = 16000):
        
        self.data_dir = Path(data_dir)
        self.speaker_list = speaker_list
        self.speaker_to_id = {spk: i for i, spk in enumerate(speaker_list)}
        self.max_length = max_length
        self.sample_rate = sample_rate
        
        # 扫描所有音频文件
        self.audio_files = []
        self._scan_files()
        
    def _scan_files(self):
        """扫描音频文件"""
        for speaker in self.speaker_list:
            speaker_dir = self.data_dir / speaker
            if speaker_dir.exists():
                for audio_file in speaker_dir.glob("*.wav"):
                    self.audio_files.append({
                        'path': audio_file,
                        'speaker': speaker,
                        'speaker_id': self.speaker_to_id[speaker]
                    })
        
        logger.info(f"找到 {len(self.audio_files)} 个音频文件")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        item = self.audio_files[idx]
        
        # 加载音频
        audio, sr = librosa.load(item['path'], sr=self.sample_rate)
        
        # 长度调整
        if len(audio) > self.max_length:
            # 随机裁剪
            start = np.random.randint(0, len(audio) - self.max_length)
            audio = audio[start:start + self.max_length]
        else:
            # 填充到固定长度
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        
        # 提取Mel频谱特征
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sample_rate,
            n_mels=80,
            n_fft=512,
            hop_length=160
        )
        mel_spec = librosa.power_to_db(mel_spec)
        
        return {
            'features': torch.FloatTensor(mel_spec),
            'speaker_id': item['speaker_id'],
            'speaker': item['speaker'],
            'audio_path': str(item['path'])
        }

# 训练器类
class SpeakerTrainer:
    """声纹识别模型训练器"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cuda',
                 learning_rate: float = 0.001):
        
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc="Training"):
            features = batch['features'].to(self.device)
            labels = batch['speaker_id'].to(self.device)
            
            # 前向传播
            embedding, logits = self.model(features)
            loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                features = batch['features'].to(self.device)
                labels = batch['speaker_id'].to(self.device)
                
                embedding, logits = self.model(features)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                # 收集预测结果
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 50,
              save_dir: str = "models"):
        """完整训练流程"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, val_accuracy = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # 学习率调度
            self.scheduler.step()
            
            logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'accuracy': val_accuracy,
                    'loss': val_loss
                }, save_dir / 'best_model.pth')
                logger.info(f"保存最佳模型，准确率: {val_accuracy:.4f}")
        
        # 保存训练历史
        self.save_training_history(save_dir)
        
        return best_accuracy
    
    def save_training_history(self, save_dir: Path):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        
        with open(save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # 绘制训练曲线
        self.plot_training_curves(save_dir)
    
    def plot_training_curves(self, save_dir: Path):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.val_accuracies, label='Val Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_model(model_type: str = "ecapa_tdnn", **kwargs):
    """创建模型工厂函数"""
    if model_type.lower() == "ecapa_tdnn":
        return ECAPA_TDNN(**kwargs)
    elif model_type.lower() == "resnetse":
        return ResNetSE(**kwargs)
    else:
        raise ValueError(f"未支持的模型类型: {model_type}")

# 配置文件示例
TRAINING_CONFIG = {
    "model": {
        "type": "ecapa_tdnn",  # 或 "resnetse"
        "input_dim": 80,
        "channels": 512,
        "embedding_dim": 192,
        "num_speakers": None  # 根据数据集自动设置
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 100,
        "device": "cuda"
    },
    "data": {
        "train_dir": "data/train",
        "val_dir": "data/val",
        "sample_rate": 16000,
        "max_length": 64000  # 4秒
    }
}
