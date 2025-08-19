"""
语音唤醒模型架构
基于深度学习的关键词检测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class WakeWordCNN(nn.Module):
    """
    基于CNN的语音唤醒模型
    适用于关键词检测任务
    """
    
    def __init__(
        self,
        num_classes: int = 2,  # 唤醒词/非唤醒词
        input_channels: int = 1,  # 音频通道数
        dropout_rate: float = 0.3
    ):
        super(WakeWordCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # 特征提取层
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # 根据输入尺寸调整
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # 批归一化
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, channels, height, width]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        # 第一个卷积块
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 第二个卷积块
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 第三个卷积块
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class WakeWordRNN(nn.Module):
    """
    基于RNN的语音唤醒模型
    适用于时序特征建模
    """
    
    def __init__(
        self,
        input_size: int = 13,  # MFCC特征维度
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        bidirectional: bool = True
    ):
        super(WakeWordRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 全连接层
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_size]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        if self.bidirectional:
            # 双向LSTM，拼接前向和后向的最后输出
            forward_out = lstm_out[:, -1, :self.hidden_size]
            backward_out = lstm_out[:, 0, self.hidden_size:]
            lstm_out = torch.cat([forward_out, backward_out], dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]
        
        # Dropout和全连接
        output = self.dropout(lstm_out)
        output = self.fc(output)
        
        return output


class WakeWordTransformer(nn.Module):
    """
    基于Transformer的语音唤醒模型
    使用自注意力机制
    """
    
    def __init__(
        self,
        input_size: int = 13,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0.1
    ):
        super(WakeWordTransformer, self).__init__()
        
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, dropout_rate)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_size]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        # 输入投影
        x = self.input_projection(x) * (self.d_model ** 0.5)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 分类
        output = self.classifier(x)
        
        return output


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


def get_wake_word_model(model_type: str = "cnn", **kwargs):
    """
    获取语音唤醒模型
    
    Args:
        model_type: 模型类型 ('cnn', 'rnn', 'transformer')
        **kwargs: 模型参数
        
    Returns:
        模型实例
    """
    if model_type.lower() == "cnn":
        return WakeWordCNN(**kwargs)
    elif model_type.lower() == "rnn":
        return WakeWordRNN(**kwargs)
    elif model_type.lower() == "transformer":
        return WakeWordTransformer(**kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


if __name__ == "__main__":
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试CNN模型
    cnn_model = get_wake_word_model("cnn", num_classes=2)
    print(f"CNN模型参数数量: {sum(p.numel() for p in cnn_model.parameters())}")
    
    # 测试RNN模型
    rnn_model = get_wake_word_model("rnn", input_size=13, num_classes=2)
    print(f"RNN模型参数数量: {sum(p.numel() for p in rnn_model.parameters())}")
    
    # 测试Transformer模型
    transformer_model = get_wake_word_model("transformer", input_size=13, num_classes=2)
    print(f"Transformer模型参数数量: {sum(p.numel() for p in transformer_model.parameters())}")
