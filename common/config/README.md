# 配置管理系统

统一的配置管理解决方案，支持训练和推理模块的配置需求。

## 🎯 **设计目标**

- ✅ **统一管理**: 所有配置集中管理
- ✅ **类型安全**: 使用Pydantic进行数据验证
- ✅ **环境变量支持**: 支持环境变量覆盖
- ✅ **多格式支持**: 支持YAML/JSON配置文件
- ✅ **微服务就绪**: 为未来微服务架构做准备
- ✅ **热更新**: 支持运行时配置更新

## 📁 **目录结构**

```
common/config/
├── __init__.py           # 模块导出
├── base.py              # 基础配置类和管理器
├── training.py          # 训练相关配置
├── inference.py         # 推理相关配置  
├── wake_word.py         # 语音唤醒配置
├── audio.py             # 音频处理配置
├── development.yaml     # 开发环境配置示例
└── example.py           # 使用示例
```

## 🚀 **快速开始**

### 1. 基本使用

```python
from common.config import get_config, TrainingConfig

# 获取训练配置
config = get_config(TrainingConfig)
print(f"训练轮数: {config.epochs}")
print(f"学习率: {config.optimizer.lr}")
```

### 2. 从文件加载

```python
from common.config import TrainingConfig

# 从YAML文件加载
config = TrainingConfig.from_file("config/development.yaml")
```

### 3. 环境变量覆盖

```bash
# 设置环境变量
export TRAINING_EPOCHS=200
export TRAINING_BATCH_SIZE=64
```

```python
# 配置会自动应用环境变量覆盖
config = get_config(TrainingConfig)
print(config.epochs)  # 输出: 200
```

## 📋 **配置类型**

### TrainingConfig - 训练配置
- 模型配置 (ModelConfig)
- 数据配置 (DataConfig)  
- 优化器配置 (OptimizerConfig)
- 调度器配置 (SchedulerConfig)
- 数据增强配置 (AugmentationConfig)

### InferenceConfig - 推理配置
- API配置 (APIConfig)
- 音频处理配置 (AudioProcessingConfig)
- 各模块推理配置 (WakeWord, ASR, TTS, VC)

### WakeWordConfig - 语音唤醒配置
- VAD配置 (VADConfig)
- 检测配置 (WakeWordDetectionConfig)
- 说话人验证配置 (SpeakerVerificationConfig)
- 误唤醒抑制配置 (FalseWakeupSuppressionConfig)

### AudioConfig - 音频配置
- IO配置 (AudioIOConfig)
- 预处理配置 (PreprocessingConfig)
- 特征提取配置 (FeatureExtractionConfig)
- 后处理配置 (PostprocessingConfig)

## 🔧 **高级功能**

### 配置验证

所有配置类都使用Pydantic进行数据验证：

```python
try:
    config = TrainingConfig(epochs=-1)  # 无效值
except ValidationError as e:
    print(f"配置错误: {e}")
```

### 配置导出

```python
# 导出为YAML
config.to_yaml("output.yaml")

# 导出为JSON  
config.to_json("output.json")
```

### 动态配置更新

```python
from common.config import get_config_manager

manager = get_config_manager()
config = manager.get_config(TrainingConfig)

# 更新配置
config.optimizer.lr = 0.0001
manager.set_config("training", config)
```

## 🏗️ **微服务支持**

配置系统设计时考虑了微服务架构：

### 服务独立配置

每个服务可以有自己的配置文件：

```yaml
# wake-word-service.yaml
wake_word:
  detection:
    wake_words: ["xiaoming"]
    threshold: 0.5
```

### 配置中心集成

未来可以轻松集成Consul、etcd等配置中心：

```python
# 扩展ConfigManager支持配置中心
class ConsulConfigManager(ConfigManager):
    def _load_from_consul(self):
        # 从Consul加载配置
        pass
```

## 📝 **配置文件示例**

参考 `development.yaml` 文件，包含完整的配置示例。

## 🔍 **环境变量命名规则**

环境变量命名格式：`{CONFIG_KEY}_{FIELD_NAME}`

例如：
- `TRAINING_EPOCHS=100`
- `WAKE_WORD_DETECTION_THRESHOLD=0.7`
- `AUDIO_IO_SAMPLE_RATE=16000`

## ⚡ **性能优化**

- 配置缓存：避免重复加载
- 懒加载：按需加载配置模块
- 配置验证：启动时验证，运行时跳过

## 🧪 **测试建议**

```python
def test_config_loading():
    config = TrainingConfig.from_file("test_config.yaml")
    assert config.epochs == 50
    assert config.data.batch_size == 32

def test_env_override():
    os.environ["TRAINING_EPOCHS"] = "200"
    config = get_config(TrainingConfig)
    assert config.epochs == 200
```

## 🔄 **迁移指南**

从现有配置迁移到新系统：

1. 创建对应的配置类
2. 将现有配置转换为YAML/JSON格式
3. 使用 `from_file()` 方法加载
4. 逐步替换硬编码配置

这个配置系统为当前开发提供便利，同时为未来的微服务架构做好准备！
