"""
可视化工具
用于训练过程监控、模型分析和结果展示
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

logger = logging.getLogger(__name__)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrainingVisualizer:
    """训练过程可视化器"""

    def __init__(self, save_dir: str = "plots"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_training_curves(self,
                             train_history: Dict[str, List],
                             save_name: str = "training_curves.png"):
        """
        绘制训练曲线

        Args:
            train_history: 训练历史数据
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练过程监控', fontsize=16, fontweight='bold')

        epochs = range(1, len(train_history['train_loss']) + 1)

        # 损失曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs, train_history['train_loss'],
                 'b-', label='训练损失', linewidth=2)
        if 'val_loss' in train_history and train_history['val_loss']:
            ax1.plot(epochs, train_history['val_loss'],
                     'r-', label='验证损失', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 唤醒词准确率
        ax2 = axes[0, 1]
        ax2.plot(epochs, train_history['train_wake_acc'],
                 'g-', label='训练准确率', linewidth=2)
        if 'val_wake_acc' in train_history and train_history['val_wake_acc']:
            ax2.plot(epochs, train_history['val_wake_acc'],
                     'orange', label='验证准确率', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('唤醒词检测准确率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 说话人识别准确率
        ax3 = axes[1, 0]
        ax3.plot(epochs, train_history['train_speaker_acc'],
                 'purple', label='训练准确率', linewidth=2)
        if 'val_speaker_acc' in train_history and train_history['val_speaker_acc']:
            ax3.plot(
                epochs, train_history['val_speaker_acc'], 'brown', label='验证准确率', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('说话人识别准确率')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 学习率曲线
        ax4 = axes[1, 1]
        if 'learning_rates' in train_history and train_history['learning_rates']:
            ax4.plot(
                epochs, train_history['learning_rates'], 'cyan', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('学习率变化')
            ax4.set_yscale('log')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"训练曲线已保存: {save_path}")

    def plot_confusion_matrix(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              class_names: List[str],
                              save_name: str = "confusion_matrix.png"):
        """
        绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            save_name: 保存文件名
        """
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"混淆矩阵已保存: {save_path}")

    def plot_feature_distribution(self,
                                  features: torch.Tensor,
                                  labels: torch.Tensor,
                                  save_name: str = "feature_distribution.png"):
        """
        绘制特征分布

        Args:
            features: 特征数据 [N, D]
            labels: 标签 [N]
            save_name: 保存文件名
        """
        # 使用PCA降维到2D
        from sklearn.decomposition import PCA

        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_np)

        plt.figure(figsize=(10, 8))

        # 为不同类别使用不同颜色
        unique_labels = np.unique(labels_np)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels_np == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                        c=[colors[i]], label=f'类别 {label}', alpha=0.7)

        plt.xlabel(f'PC1 (方差解释比: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 (方差解释比: {pca.explained_variance_ratio_[1]:.2%})')
        plt.title('特征分布可视化 (PCA降维)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"特征分布图已保存: {save_path}")

    def plot_audio_features(self,
                            mel_spec: torch.Tensor,
                            save_name: str = "audio_features.png"):
        """
        绘制音频特征可视化

        Args:
            mel_spec: Mel频谱图 [time, freq]
            save_name: 保存文件名
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Mel频谱图
        mel_np = mel_spec.detach().cpu().numpy()

        im1 = axes[0].imshow(mel_np.T,
                             aspect='auto',
                             origin='lower',
                             cmap='viridis')
        axes[0].set_title('Mel频谱图')
        axes[0].set_xlabel('时间帧')
        axes[0].set_ylabel('Mel频率')
        plt.colorbar(im1, ax=axes[0])

        # 能量分布
        energy = mel_np.mean(axis=1)
        axes[1].plot(energy, linewidth=2)
        axes[1].set_title('平均能量分布')
        axes[1].set_xlabel('时间帧')
        axes[1].set_ylabel('能量')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"音频特征图已保存: {save_path}")


class ModelAnalyzer:
    """模型分析器"""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def analyze_parameters(self) -> Dict:
        """分析模型参数"""
        total_params = 0
        trainable_params = 0
        param_details = {}

        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count

            if param.requires_grad:
                trainable_params += param_count

            param_details[name] = {
                'shape': list(param.shape),
                'params': param_count,
                'trainable': param.requires_grad
            }

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_details': param_details,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设float32
        }

    def plot_parameter_distribution(self, save_dir: str = "plots"):
        """绘制参数分布图"""
        param_counts = []
        param_names = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_counts.append(param.numel())
                # 简化参数名称
                short_name = name.split('.')[-2] + '.' + name.split('.')[-1]
                param_names.append(short_name)

        # 选择参数量最大的前20个
        if len(param_counts) > 20:
            sorted_indices = np.argsort(param_counts)[-20:]
            param_counts = [param_counts[i] for i in sorted_indices]
            param_names = [param_names[i] for i in sorted_indices]

        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(param_names)), param_counts)
        plt.yticks(range(len(param_names)), param_names)
        plt.xlabel('参数数量')
        plt.title('模型参数分布')
        plt.grid(True, alpha=0.3)

        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                     f'{int(width):,}', ha='left', va='center')

        plt.tight_layout()
        save_path = os.path.join(save_dir, "parameter_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"参数分布图已保存: {save_path}")


def create_training_report(checkpoint_path: str,
                           output_dir: str = "reports"):
    """
    创建训练报告

    Args:
        checkpoint_path: 检查点文件路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 创建可视化器
    visualizer = TrainingVisualizer(save_dir=output_dir)

    # 绘制训练曲线
    if 'train_history' in checkpoint:
        visualizer.plot_training_curves(checkpoint['train_history'])

    # 生成HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>训练报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                     background-color: #e8f4fd; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>声纹+唤醒词模型训练报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>模型配置</h2>
            <pre>{checkpoint.get('config', {})}</pre>
        </div>
        
        <div class="section">
            <h2>训练结果</h2>
            <div class="metric">
                <strong>最佳验证准确率:</strong> {checkpoint.get('best_val_acc', 0):.4f}
            </div>
            <div class="metric">
                <strong>训练轮数:</strong> {checkpoint.get('epoch', 0)}
            </div>
        </div>
        
        <div class="section">
            <h2>训练曲线</h2>
            <img src="training_curves.png" alt="训练曲线">
        </div>
        
    </body>
    </html>
    """

    # 保存HTML报告
    report_path = os.path.join(output_dir, "training_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"训练报告已生成: {report_path}")


# 测试函数
if __name__ == "__main__":
    print("=== 可视化工具测试 ===")

    # 创建模拟训练历史数据
    epochs = 20
    train_history = {
        'train_loss': [1.0 - 0.8 * (1 - np.exp(-i/5)) + 0.1*np.random.random() for i in range(epochs)],
        'val_loss': [1.0 - 0.7 * (1 - np.exp(-i/5)) + 0.15*np.random.random() for i in range(epochs)],
        'train_wake_acc': [0.5 + 0.45 * (1 - np.exp(-i/5)) + 0.05*np.random.random() for i in range(epochs)],
        'val_wake_acc': [0.5 + 0.4 * (1 - np.exp(-i/5)) + 0.08*np.random.random() for i in range(epochs)],
        'train_speaker_acc': [0.1 + 0.8 * (1 - np.exp(-i/3)) + 0.05*np.random.random() for i in range(epochs)],
        'val_speaker_acc': [0.1 + 0.75 * (1 - np.exp(-i/3)) + 0.08*np.random.random() for i in range(epochs)],
        'learning_rates': [0.001 * (0.95 ** i) for i in range(epochs)]
    }

    # 测试可视化
    visualizer = TrainingVisualizer()
    visualizer.plot_training_curves(train_history)

    # 测试特征分布可视化
    features = torch.randn(200, 64)
    labels = torch.randint(0, 5, (200,))
    visualizer.plot_feature_distribution(features, labels)

    # 测试音频特征可视化
    mel_spec = torch.randn(100, 80)
    visualizer.plot_audio_features(mel_spec)

    print("✓ 所有可视化功能测试通过")
