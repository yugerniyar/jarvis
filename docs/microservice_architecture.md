# 语音识别微服务架构文档

## 🏗️ 架构概述

本项目采用模块化微服务架构设计，支持服务的动态加载和卸载，实现真正的解耦合。

## 📦 模块化设计原则

### 1. 服务自识别机制
- 每个服务模块包含自检功能
- 前端根据可用服务动态渲染界面
- 服务注册/注销机制

### 2. 依赖隔离
- 训练服务与推理服务完全分离
- 每个模块有独立的依赖文件
- 可选择性安装模块

## 🚀 服务模块列表

### 核心服务
- **语音识别服务** (`speech_recognition_service`)
- **声纹识别服务** (`speaker_recognition_service`) 
- **语音合成服务** (`tts_service`)
- **音频处理服务** (`audio_processing_service`)

### 扩展服务  
- **语音训练服务** (`training_service`)
- **模型优化服务** (`optimization_service`)
- **实时语音服务** (`realtime_service`)

## 📋 版本库分离策略

### 训练环境 (`requirements_training.txt`)
```python
# 重量级训练依赖
pytorch-lightning==2.1.3
transformers==4.36.0
accelerate==1.9.0
speechbrain==0.5.16
nemo-toolkit[asr]==1.22.0
wandb==0.16.1
tensorboard==2.15.1
```

### 推理环境 (`requirements_inference.txt`)
```python
# 轻量级推理依赖
torch==2.1.2+cu121
onnx==1.15.0
onnxruntime-gpu==1.16.3
faster-whisper==0.10.0
soundfile==0.12.1
librosa==0.10.1
```

### 基础环境 (`requirements_base.txt`)
```python
# 核心Web框架
fastapi==0.104.1
gradio==4.8.0
numpy>=1.22,<1.24
pydantic==2.5.0
```

## 🔧 服务自识别实现

### 服务注册机制
```python
# services/base_service.py
class BaseService:
    def __init__(self, name: str, dependencies: List[str]):
        self.name = name
        self.dependencies = dependencies
        self.available = self.check_dependencies()
        
    def check_dependencies(self) -> bool:
        """检查依赖是否满足"""
        try:
            for dep in self.dependencies:
                __import__(dep)
            return True
        except ImportError:
            return False
```

### 服务发现机制
```python
# services/service_registry.py
class ServiceRegistry:
    _services = {}
    
    @classmethod
    def register(cls, service: BaseService):
        if service.available:
            cls._services[service.name] = service
            
    @classmethod
    def get_available_services(cls) -> Dict[str, BaseService]:
        return cls._services
```

## 🎯 前端动态渲染

### Gradio界面动态生成
```python
# web/dynamic_interface.py
def create_dynamic_interface():
    available_services = ServiceRegistry.get_available_services()
    tabs = []
    
    if "speech_recognition" in available_services:
        tabs.append(create_speech_recognition_tab())
        
    if "speaker_recognition" in available_services:
        tabs.append(create_speaker_recognition_tab())
        
    return gr.TabbedInterface(tabs, tab_names=[s.name for s in tabs])
```

## 📁 推荐目录结构

```
speech_recognition_project/
├── requirements/
│   ├── base.txt              # 基础依赖
│   ├── training.txt          # 训练依赖
│   ├── inference.txt         # 推理依赖
│   └── development.txt       # 开发依赖
├── services/
│   ├── base_service.py       # 服务基类
│   ├── service_registry.py   # 服务注册
│   ├── speech_recognition/   # 语音识别服务
│   ├── speaker_recognition/  # 声纹识别服务
│   ├── tts_service/         # 语音合成服务
│   └── training_service/    # 训练服务
├── web/
│   ├── dynamic_interface.py  # 动态界面
│   └── api_router.py        # API路由
├── configs/
│   └── service_configs/     # 服务配置
└── scripts/
    ├── install_base.sh      # 基础安装
    ├── install_training.sh  # 训练环境安装
    └── install_inference.sh # 推理环境安装
```

## 🚀 部署策略

### 1. 基础部署
```bash
pip install -r requirements/base.txt
python main.py --mode=inference
```

### 2. 训练环境部署  
```bash
pip install -r requirements/base.txt
pip install -r requirements/training.txt
python main.py --mode=training
```

### 3. 完整部署
```bash
pip install -r requirements/base.txt
pip install -r requirements/training.txt
pip install -r requirements/inference.txt
python main.py --mode=full
```

## 🔄 服务热插拔

### 添加新服务
1. 创建服务模块
2. 实现BaseService接口
3. 注册到ServiceRegistry
4. 前端自动检测并显示

### 移除服务
1. 删除服务模块
2. 重启应用
3. 前端自动隐藏相关功能

## 🎯 优势

1. **模块化**: 每个服务独立开发和部署
2. **轻量化**: 按需加载依赖
3. **可扩展**: 易于添加新服务
4. **容错性**: 单个服务故障不影响整体
5. **动态性**: 支持运行时服务发现
