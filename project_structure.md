# VoiceWakenAI 项目结构设计

## 推荐的完整目录结构

```
VoiceWakenAI/
│
├── main.py                    # 主入口
├── requirements.txt           # 依赖清单
├── config/                    # 全局配置
│   ├── __init__.py
│   ├── settings.py           # 全局设置
│   └── logging.conf          # 日志配置
│
├── common/                    # 通用模块
│   ├── __init__.py
│   ├── exceptions.py         # 自定义异常
│   ├── constants.py          # 常量定义
│   └── base_models.py        # 基础数据模型
│
├── utils/                     # 全局工具
│   ├── __init__.py
│   ├── audio_utils.py        # 音频工具
│   ├── file_utils.py         # 文件工具
│   └── logger.py             # 日志工具
│
├── speaker_recognition/       # 声纹识别模块
│   ├── __init__.py
│   ├── controller/
│   │   ├── __init__.py
│   │   └── speaker_api.py    # 声纹识别API
│   ├── service/
│   │   ├── __init__.py
│   │   └── speaker_service.py # 声纹识别业务逻辑
│   ├── inference/
│   │   ├── __init__.py
│   │   └── speaker_model.py  # 声纹模型推理
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── feature_extractor.py # 特征提取
│   ├── models/               # 模型文件存放
│   ├── entity/
│   │   ├── __init__.py
│   │   └── speaker_schema.py # 声纹数据结构
│   ├── utils/
│   │   ├── __init__.py
│   │   └── speaker_utils.py  # 声纹专用工具
│   └── config/
│       ├── __init__.py
│       └── speaker_config.py # 声纹模块配置
│
├── asr/                       # 语音识别模块
│   ├── __init__.py
│   ├── controller/
│   │   ├── __init__.py
│   │   └── asr_api.py
│   ├── service/
│   │   ├── __init__.py
│   │   └── asr_service.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── whisper_model.py  # Whisper推理
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── audio_preprocessor.py
│   ├── models/
│   ├── entity/
│   │   ├── __init__.py
│   │   └── asr_schema.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── asr_utils.py
│   └── config/
│       ├── __init__.py
│       └── asr_config.py
│
├── llm/                       # 大语言模型模块
│   ├── __init__.py
│   ├── controller/
│   │   ├── __init__.py
│   │   └── llm_api.py
│   ├── service/
│   │   ├── __init__.py
│   │   └── llm_service.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── local_llm.py      # 本地LLM推理
│   │   └── api_llm.py        # API LLM调用
│   ├── entity/
│   │   ├── __init__.py
│   │   └── llm_schema.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── llm_utils.py
│   └── config/
│       ├── __init__.py
│       └── llm_config.py
│
├── tts/                       # 语音合成模块
│   ├── __init__.py
│   ├── controller/
│   │   ├── __init__.py
│   │   └── tts_api.py
│   ├── service/
│   │   ├── __init__.py
│   │   └── tts_service.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── tts_model.py      # TTS模型推理
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── text_preprocessor.py
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   └── audio_postprocessor.py
│   ├── models/
│   ├── entity/
│   │   ├── __init__.py
│   │   └── tts_schema.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── tts_utils.py
│   └── config/
│       ├── __init__.py
│       └── tts_config.py
│
├── vc/                        # 声线变换模块
│   ├── __init__.py
│   ├── controller/
│   │   ├── __init__.py
│   │   └── vc_api.py
│   ├── service/
│   │   ├── __init__.py
│   │   └── vc_service.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── vc_model.py       # 声线变换推理
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── audio_preprocessor.py
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   └── voice_postprocessor.py
│   ├── models/
│   ├── entity/
│   │   ├── __init__.py
│   │   └── vc_schema.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── vc_utils.py
│   └── config/
│       ├── __init__.py
│       └── vc_config.py
│
├── data/                      # 数据存储
│   ├── speaker_embeddings/    # 声纹特征
│   ├── audio_samples/         # 音频样本
│   ├── models/               # 训练好的模型
│   └── cache/                # 缓存文件
│
└── tests/                     # 测试模块
    ├── __init__.py
    ├── test_speaker_recognition/
    ├── test_asr/
    ├── test_llm/
    ├── test_tts/
    └── test_vc/
```

## AI模块设计要点

### 1. AI特有的分层
- **inference/**: 模型推理逻辑，封装模型加载和预测
- **preprocessing/**: 数据预处理，如音频特征提取、文本清洗
- **postprocessing/**: 后处理逻辑，如音频增强、结果优化
- **models/**: 存放模型文件(.pth, .onnx, .bin等)

### 2. 与传统Web项目的区别
- **entity/**: 不只是数据库实体，更多是数据结构定义(Pydantic模型)
- **service/**: 业务逻辑 + 模型调度逻辑
- **utils/**: 包含大量音频、图像、文本处理工具

### 3. 配置管理
- 每个模块有独立配置文件
- 支持模型路径、超参数等配置
- 支持开发/生产环境切换

### 4. 数据流设计
```python
# 典型的AI模块数据流
输入数据 -> preprocessing -> inference -> postprocessing -> 输出数据
```
