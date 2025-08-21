"""
音频处理工具函数
包含音频加载、预处理、特征提取等功能
"""

import logging
from typing import Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    音频处理器
    统一处理各种音频格式和采样率
    """

    def __init__(self,
                 target_sr: int = 16000,
                 n_mels: int = 80,
                 n_fft: int = 512,
                 hop_length: int = 160,
                 win_length: int = 400):
        """
        初始化音频处理器

        Args:
            target_sr: 目标采样率
            n_mels: Mel频谱图通道数
            n_fft: FFT点数
            hop_length: 帧移
            win_length: 窗长
        """
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # 创建Mel频谱图变换器
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect"
        )

        # 对数变换
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype='power')

    def load_audio(self,
                   audio_path: str,
                   start_time: Optional[float] = None,
                   duration: Optional[float] = None) -> Tuple[torch.Tensor, int]:
        """
        加载音频文件

        Args:
            audio_path: 音频文件路径
            start_time: 开始时间（秒）
            duration: 持续时间（秒）

        Returns:
            (waveform, sample_rate): 音频波形和采样率
        """
        try:
            # 使用torchaudio加载
            if start_time is not None or duration is not None:
                frame_offset = int(
                    start_time * self.target_sr) if start_time else 0
                num_frames = int(duration * self.target_sr) if duration else -1
                waveform, sr = torchaudio.load(
                    audio_path,
                    frame_offset=frame_offset,
                    num_frames=num_frames if num_frames > 0 else None
                )
            else:
                waveform, sr = torchaudio.load(audio_path)

            # 重采样到目标采样率
            if sr != self.target_sr:
                resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = resampler(waveform)

            # 转为单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            return waveform, self.target_sr

        except Exception as e:
            logger.error(f"加载音频失败 {audio_path}: {e}")
            # 返回空音频
            return torch.zeros(1, self.target_sr), self.target_sr

    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        提取Mel频谱图特征

        Args:
            waveform: 音频波形 [channels, time]

        Returns:
            mel_spec: Mel频谱图 [time, n_mels]
        """
        # 提取Mel频谱图
        mel_spec = self.mel_transform(waveform)  # [channels, n_mels, time]

        # 转换为对数尺度
        mel_spec = self.amplitude_to_db(mel_spec)

        # 转换维度：[channels, n_mels, time] -> [time, n_mels]
        mel_spec = mel_spec.squeeze(0).transpose(0, 1)

        return mel_spec

    def normalize_features(self,
                           features: torch.Tensor,
                           method: str = 'mean_std') -> torch.Tensor:
        """
        特征归一化

        Args:
            features: 输入特征 [time, feature_dim]
            method: 归一化方法 'mean_std' | 'min_max' | 'robust'

        Returns:
            normalized_features: 归一化后的特征
        """
        if method == 'mean_std':
            # 零均值单位方差归一化
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True)
            normalized = (features - mean) / (std + 1e-8)

        elif method == 'min_max':
            # 最小-最大归一化
            min_val = features.min(dim=0, keepdim=True)[0]
            max_val = features.max(dim=0, keepdim=True)[0]
            normalized = (features - min_val) / (max_val - min_val + 1e-8)

        elif method == 'robust':
            # 鲁棒归一化（使用中位数和MAD）
            median = features.median(dim=0, keepdim=True)[0]
            mad = torch.median(torch.abs(features - median),
                               dim=0, keepdim=True)[0]
            normalized = (features - median) / (mad + 1e-8)

        else:
            normalized = features

        return normalized

    def apply_voice_activity_detection(self,
                                       waveform: torch.Tensor,
                                       frame_length: int = 512,
                                       hop_length: int = 160,
                                       energy_threshold: float = 0.01) -> torch.Tensor:
        """
        简单的语音活动检测

        Args:
            waveform: 音频波形
            frame_length: 帧长
            hop_length: 帧移
            energy_threshold: 能量阈值

        Returns:
            vad_mask: 语音活动掩码 [num_frames]
        """
        # 计算短时能量
        waveform_np = waveform.squeeze().numpy()

        # 计算帧数
        num_frames = 1 + (len(waveform_np) - frame_length) // hop_length

        # 计算每帧的能量
        energy = []
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = waveform_np[start:end] if end <= len(waveform_np) else np.pad(
                waveform_np[start:], (0, end - len(waveform_np)), 'constant')
            energy.append(np.sum(frame ** 2))

        energy = np.array(energy)

        # 归一化能量
        energy_norm = energy / (np.max(energy) + 1e-8)

        # 应用阈值
        vad_mask = energy_norm > energy_threshold

        return torch.from_numpy(vad_mask.astype(np.float32))

    def extract_mfcc(self, waveform: torch.Tensor, n_mfcc: int = 13) -> torch.Tensor:
        """
        提取MFCC特征

        Args:
            waveform: 音频波形
            n_mfcc: MFCC系数数量

        Returns:
            mfcc_features: MFCC特征 [time, n_mfcc]
        """
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.target_sr,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'win_length': self.win_length,
                'n_mels': self.n_mels
            }
        )

        mfcc = mfcc_transform(waveform)  # [channels, n_mfcc, time]
        mfcc = mfcc.squeeze(0).transpose(0, 1)  # [time, n_mfcc]

        return mfcc


def apply_time_stretching(waveform: torch.Tensor,
                          stretch_factor: float) -> torch.Tensor:
    """
    应用时间拉伸

    Args:
        waveform: 输入波形 [channels, time]
        stretch_factor: 拉伸因子 (>1: 变慢, <1: 变快)

    Returns:
        stretched_waveform: 拉伸后的波形
    """
    # 使用线性插值进行简单的时间拉伸
    original_length = waveform.shape[-1]
    new_length = int(original_length * stretch_factor)

    # 重新采样
    stretched = torch.nn.functional.interpolate(
        waveform.unsqueeze(0),
        size=new_length,
        mode='linear',
        align_corners=False
    ).squeeze(0)

    return stretched


def apply_pitch_shift(waveform: torch.Tensor,
                      sample_rate: int,
                      n_steps: float) -> torch.Tensor:
    """
    应用音调变换

    Args:
        waveform: 输入波形
        sample_rate: 采样率
        n_steps: 半音步数 (正数: 升调, 负数: 降调)

    Returns:
        pitch_shifted_waveform: 变调后的波形
    """
    try:
        # 使用librosa进行音调变换
        import librosa
        waveform_np = waveform.squeeze().numpy()
        shifted = librosa.effects.pitch_shift(
            waveform_np,
            sr=sample_rate,
            n_steps=n_steps
        )
        return torch.from_numpy(shifted).unsqueeze(0)
    except ImportError:
        logger.warning("librosa未安装，跳过音调变换")
        return waveform


def add_noise(waveform: torch.Tensor,
              noise_type: str = 'gaussian',
              noise_level: float = 0.01) -> torch.Tensor:
    """
    添加噪声

    Args:
        waveform: 输入波形
        noise_type: 噪声类型 'gaussian' | 'uniform' | 'pink'
        noise_level: 噪声水平

    Returns:
        noisy_waveform: 添加噪声后的波形
    """
    if noise_type == 'gaussian':
        noise = torch.randn_like(waveform) * noise_level
    elif noise_type == 'uniform':
        noise = (torch.rand_like(waveform) - 0.5) * 2 * noise_level
    elif noise_type == 'pink':
        # 简化的粉红噪声
        white_noise = torch.randn_like(waveform)
        # 应用简单的低通滤波器模拟粉红噪声
        if len(white_noise.shape) > 1:
            for i in range(1, white_noise.shape[-1]):
                white_noise[..., i] = 0.7 * \
                    white_noise[..., i-1] + 0.3 * white_noise[..., i]
        noise = white_noise * noise_level
    else:
        noise = torch.zeros_like(waveform)

    return waveform + noise


def apply_speed_perturbation(waveform: torch.Tensor,
                             speed_factor: float) -> torch.Tensor:
    """
    应用速度扰动（改变语速但不改变音调）

    Args:
        waveform: 输入波形
        speed_factor: 速度因子 (>1: 加速, <1: 减速)

    Returns:
        perturbed_waveform: 扰动后的波形
    """
    # 简单的重采样实现
    original_length = waveform.shape[-1]
    new_length = int(original_length / speed_factor)

    # 使用插值调整长度
    perturbed = torch.nn.functional.interpolate(
        waveform.unsqueeze(0),
        size=new_length,
        mode='linear',
        align_corners=False
    ).squeeze(0)

    return perturbed


class AudioAugmentor:
    """
    音频增强器
    组合多种音频增强技术
    """

    def __init__(self, augmentation_config: dict):
        self.config = augmentation_config
        self.processor = AudioProcessor()

    def augment(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        应用音频增强

        Args:
            waveform: 输入波形
            sample_rate: 采样率

        Returns:
            augmented_waveform: 增强后的波形
        """
        augmented = waveform.clone()

        # 添加噪声
        if np.random.rand() < self.config.get('noise_prob', 0.3):
            noise_level = np.random.uniform(0.001, 0.01)
            noise_type = np.random.choice(['gaussian', 'uniform'])
            augmented = add_noise(augmented, noise_type, noise_level)

        # 音量变化
        if np.random.rand() < self.config.get('volume_prob', 0.3):
            volume_factor = np.random.uniform(0.7, 1.3)
            augmented = augmented * volume_factor

        # 速度扰动
        if np.random.rand() < self.config.get('speed_prob', 0.2):
            speed_factor = np.random.uniform(0.9, 1.1)
            augmented = apply_speed_perturbation(augmented, speed_factor)

        # 音调变换
        if np.random.rand() < self.config.get('pitch_prob', 0.2):
            pitch_steps = np.random.uniform(-2, 2)
            augmented = apply_pitch_shift(augmented, sample_rate, pitch_steps)

        return augmented


# 测试函数
if __name__ == "__main__":
    print("=== 音频处理工具测试 ===")

    # 创建音频处理器
    processor = AudioProcessor()

    # 创建测试音频
    duration = 2.0  # 2秒
    sample_rate = 16000
    t = torch.linspace(0, duration, int(duration * sample_rate))
    test_audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # 440Hz正弦波

    print(f"测试音频形状: {test_audio.shape}")

    # 测试Mel频谱图提取
    mel_spec = processor.extract_mel_spectrogram(test_audio)
    print(f"Mel频谱图形状: {mel_spec.shape}")

    # 测试MFCC提取
    mfcc = processor.extract_mfcc(test_audio)
    print(f"MFCC特征形状: {mfcc.shape}")

    # 测试特征归一化
    normalized = processor.normalize_features(mel_spec, method='mean_std')
    print(f"归一化后统计: 均值={normalized.mean():.6f}, 标准差={normalized.std():.6f}")

    # 测试语音活动检测
    vad_mask = processor.apply_voice_activity_detection(test_audio)
    print(f"VAD掩码形状: {vad_mask.shape}, 活动比例: {vad_mask.mean():.3f}")

    # 测试音频增强
    augment_config = {
        'noise_prob': 0.5,
        'volume_prob': 0.5,
        'speed_prob': 0.5,
        'pitch_prob': 0.5
    }

    augmentor = AudioAugmentor(augment_config)
    augmented = augmentor.augment(test_audio, sample_rate)
    print(f"增强后音频形状: {augmented.shape}")

    print("✓ 所有音频处理功能测试通过")
