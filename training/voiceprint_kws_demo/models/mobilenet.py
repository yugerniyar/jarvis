"""
MobileNetV4模型实现
轻量级卷积神经网络，适合移动端部署
针对音频处理进行了1D卷积适配
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HardSwish(nn.Module):
    """Hard Swish激活函数，计算效率更高的Swish近似"""

    def forward(self, x):
        return x * F.relu6(x + 3) / 6


class SqueezeExcitation1D(nn.Module):
    """
    1D Squeeze-and-Excitation模块
    为音频序列优化的通道注意力机制
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用SE注意力"""
        return x * self.se(x)


class InvertedResidual1D(nn.Module):
    """
    1D倒残差块(Inverted Residual Block)
    MobileNet的核心组件，适配音频序列处理
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 expand_ratio: int = 6,
                 use_se: bool = False,
                 activation: str = 'relu'):
        super().__init__()

        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_channels = in_channels * expand_ratio

        # 选择激活函数
        if activation == 'relu':
            act_fn = nn.ReLU6
        elif activation == 'hswish':
            act_fn = HardSwish
        else:
            act_fn = nn.ReLU6

        layers = []

        # 1. 扩展阶段（如果expand_ratio != 1）
        if expand_ratio != 1:
            layers.extend([
                nn.Conv1d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm1d(hidden_channels),
                act_fn(inplace=True)
            ])

        # 2. 深度卷积阶段
        layers.extend([
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                      stride=stride, padding=kernel_size//2,
                      groups=hidden_channels, bias=False),
            nn.BatchNorm1d(hidden_channels),
            act_fn(inplace=True)
        ])

        # 3. SE注意力（可选）
        if use_se:
            layers.append(SqueezeExcitation1D(hidden_channels))

        # 4. 投影阶段
        layers.extend([
            nn.Conv1d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """倒残差块前向传播"""
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV4Block(nn.Module):
    """
    MobileNetV4块
    结合了最新的优化技术
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 expand_ratio: int = 4,
                 use_se: bool = True):
        super().__init__()

        # 主分支：倒残差块
        self.main_branch = InvertedResidual1D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            expand_ratio=expand_ratio,
            use_se=use_se,
            activation='hswish'
        )

        # 辅助分支（用于特征融合）
        if stride == 1 and in_channels == out_channels:
            self.aux_branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            self.use_aux = True
        else:
            self.use_aux = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MobileNetV4块前向传播"""
        main_out = self.main_branch(x)

        if self.use_aux:
            aux_out = self.aux_branch(x)
            return main_out + 0.2 * aux_out  # 加权融合
        else:
            return main_out


class MobileNetV4(nn.Module):
    """
    完整的MobileNetV4模型
    针对音频处理优化的轻量级网络
    """

    def __init__(self,
                 input_channels: int = 80,
                 num_classes: int = 256,
                 width_multiplier: float = 1.0,
                 dropout: float = 0.2):
        super().__init__()

        # 根据宽度倍数调整通道数
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)

        # 初始卷积
        init_channels = make_divisible(32 * width_multiplier)
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, init_channels, 3,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(init_channels),
            HardSwish()
        )

        # MobileNetV4块配置
        # [input_channels, output_channels, kernel_size, stride, expand_ratio, use_se]
        block_configs = [
            [init_channels, 16, 3, 1, 1, False],
            [16, 24, 3, 2, 4, False],
            [24, 24, 3, 1, 3, False],
            [24, 40, 5, 2, 3, True],
            [40, 40, 5, 1, 3, True],
            [40, 40, 5, 1, 3, True],
            [40, 80, 3, 2, 6, False],
            [80, 80, 3, 1, 2.5, False],
            [80, 80, 3, 1, 2.3, False],
            [80, 80, 3, 1, 2.3, False],
            [80, 112, 3, 1, 6, True],
            [112, 112, 3, 1, 6, True],
            [112, 160, 5, 2, 6, True],
            [160, 160, 5, 1, 6, True],
            [160, 160, 5, 1, 6, True]
        ]

        # 构建MobileNet块
        features = []
        for config in block_configs:
            in_c, out_c, k, s, e, se = config
            in_c = make_divisible(in_c * width_multiplier)
            out_c = make_divisible(out_c * width_multiplier)

            features.append(
                MobileNetV4Block(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=s,
                    expand_ratio=e,
                    use_se=se
                )
            )

        self.features.extend(features)

        # 最终卷积层
        last_channels = make_divisible(160 * width_multiplier)
        final_channels = make_divisible(960 * width_multiplier)

        self.features.extend([
            nn.Conv1d(last_channels, final_channels, 1, bias=False),
            nn.BatchNorm1d(final_channels),
            HardSwish()
        ])

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(final_channels, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MobileNetV4前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, channels] 或 [batch_size, channels, seq_len]
        Returns:
            特征向量 [batch_size, num_classes]
        """
        # 确保输入维度正确
        if x.dim() == 3 and x.size(2) > x.size(1):
            # [B, T, C] -> [B, C, T]
            x = x.transpose(1, 2)

        # 特征提取
        x = self.features(x)

        # 分类
        x = self.classifier(x)

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征而不进行分类"""
        if x.dim() == 3 and x.size(2) > x.size(1):
            x = x.transpose(1, 2)

        x = self.features(x)
        x = F.adaptive_avg_pool1d(x, 1).flatten(1)

        return x


class EfficientMobileNet(nn.Module):
    """
    高效版MobileNet
    进一步优化计算和内存效率
    """

    def __init__(self,
                 input_channels: int = 80,
                 output_dim: int = 256,
                 stages: int = 4):
        super().__init__()

        # 渐进式通道扩展
        channels = [32, 64, 128, 256][:stages]

        # 构建网络层
        layers = []
        in_ch = input_channels

        for i, out_ch in enumerate(channels):
            stride = 2 if i > 0 else 1

            layers.append(
                MobileNetV4Block(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    stride=stride,
                    expand_ratio=2 if i == 0 else 4,
                    use_se=i >= 2  # 后期阶段使用SE
                )
            )
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """高效MobileNet前向传播"""
        if x.dim() == 3 and x.size(2) > x.size(1):
            x = x.transpose(1, 2)

        features = self.backbone(x)
        output = self.output_layer(features)

        return output


# 测试代码
if __name__ == "__main__":
    print("=== MobileNetV4模型测试 ===")

    # 创建测试数据
    batch_size, seq_len, input_dim = 4, 200, 80
    x = torch.randn(batch_size, seq_len, input_dim)

    print(f"输入形状: {x.shape}")

    # 1. 测试标准MobileNetV4
    print("\n1. 标准MobileNetV4:")
    model = MobileNetV4(input_channels=input_dim, num_classes=256)

    output = model(x)
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 计算模型大小（MB）
    param_size = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
    print(f"模型大小: {param_size:.2f} MB")

    # 2. 测试轻量版本
    print("\n2. 轻量版MobileNetV4:")
    light_model = MobileNetV4(
        input_channels=input_dim,
        num_classes=128,
        width_multiplier=0.5
    )

    light_output = light_model(x)
    print(f"轻量版输出形状: {light_output.shape}")
    print(f"轻量版参数量: {sum(p.numel() for p in light_model.parameters()):,}")

    light_param_size = sum(
        p.numel() * 4 for p in light_model.parameters()) / (1024 * 1024)
    print(f"轻量版模型大小: {light_param_size:.2f} MB")

    # 3. 测试高效版本
    print("\n3. 高效MobileNet:")
    efficient_model = EfficientMobileNet(input_channels=input_dim, stages=3)

    efficient_output = efficient_model(x)
    print(f"高效版输出形状: {efficient_output.shape}")
    print(f"高效版参数量: {sum(p.numel() for p in efficient_model.parameters()):,}")

    # 4. 计算FLOPs（简化估计）
    print("\n4. 计算复杂度对比:")

    def estimate_flops(model, input_shape):
        """简化的FLOPs估计"""
        total_flops = 0
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                out_h = input_shape[-1] // getattr(module, 'stride', [1])[0]
                kernel_flops = module.kernel_size[0] * module.in_channels
                output_elements = module.out_channels * out_h
                total_flops += kernel_flops * output_elements
        return total_flops

    standard_flops = estimate_flops(model, x.shape)
    light_flops = estimate_flops(light_model, x.shape)
    efficient_flops = estimate_flops(efficient_model, x.shape)

    print(f"标准版FLOPs: {standard_flops:,}")
    print(f"轻量版FLOPs: {light_flops:,}")
    print(f"高效版FLOPs: {efficient_flops:,}")
    print(f"轻量版效率提升: {standard_flops/light_flops:.1f}x")
    print(f"高效版效率提升: {standard_flops/efficient_flops:.1f}x")
