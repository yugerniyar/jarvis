#!/usr/bin/env python3
"""
VoiceWakenAI 项目结构快速生成脚本
运行此脚本将自动创建完整的项目目录结构和基础文件
"""

import os
from pathlib import Path

def create_directory_structure():
    """创建项目目录结构"""
    
    # 项目根目录
    project_root = Path(".")
    
    # 目录结构定义
    directories = [
        # 全局目录
        "config",
        "common", 
        "utils",
        "data/speaker_embeddings",
        "data/audio_samples", 
        "data/models",
        "data/cache",
        "tests",
        "cpp",
        
        # 声纹识别模块
        "speaker_recognition/controller",
        "speaker_recognition/service", 
        "speaker_recognition/inference",
        "speaker_recognition/preprocessing",
        "speaker_recognition/entity",
        "speaker_recognition/utils",
        "speaker_recognition/models",
        "speaker_recognition/config",
        
        # ASR模块
        "asr/controller",
        "asr/service",
        "asr/inference", 
        "asr/preprocessing",
        "asr/entity",
        "asr/utils",
        "asr/models",
        "asr/config",
        
        # LLM模块
        "llm/controller",
        "llm/service",
        "llm/inference",
        "llm/entity", 
        "llm/utils",
        "llm/config",
        
        # TTS模块
        "tts/controller",
        "tts/service",
        "tts/inference",
        "tts/preprocessing",
        "tts/postprocessing",
        "tts/entity",
        "tts/utils", 
        "tts/models",
        "tts/config",
        
        # VC模块
        "vc/controller",
        "vc/service",
        "vc/inference",
        "vc/preprocessing", 
        "vc/postprocessing",
        "vc/entity",
        "vc/utils",
        "vc/models", 
        "vc/config",
    ]
    
    # 创建目录
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ 创建目录: {directory}")
        
        # 为每个Python包目录创建__init__.py
        if not directory.startswith(("data/", "cpp", "tests")):
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# -*- coding: utf-8 -*-\n")

def create_basic_files():
    """创建基础文件"""
    
    files_content = {
        # 主入口文件
        "main.py": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VoiceWakenAI 主入口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from speaker_recognition.controller import speaker_api
from asr.controller import asr_api  
from llm.controller import llm_api
from tts.controller import tts_api
from vc.controller import vc_api

from utils.logger import setup_logger
from config.settings import settings

# 初始化日志
logger = setup_logger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="VoiceWakenAI", 
    description="语音交互AI系统",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# 注册路由
app.include_router(speaker_api.router, prefix="/api/speaker", tags=["声纹识别"])
app.include_router(asr_api.router, prefix="/api/asr", tags=["语音识别"])
app.include_router(llm_api.router, prefix="/api/llm", tags=["大语言模型"])
app.include_router(tts_api.router, prefix="/api/tts", tags=["语音合成"])
app.include_router(vc_api.router, prefix="/api/vc", tags=["声线变换"])

@app.get("/")
async def root():
    return {"message": "VoiceWakenAI API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("启动 VoiceWakenAI 服务...")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT, 
        reload=settings.DEBUG,
        log_level="info"
    )
''',

        # 全局配置
        "config/settings.py": '''# -*- coding: utf-8 -*-
"""
全局配置文件
"""

from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """应用配置"""
    
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # 数据目录
    DATA_DIR: str = "./data"
    MODEL_DIR: str = "./data/models"
    CACHE_DIR: str = "./data/cache"
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    # AI模型配置
    DEVICE: str = "cpu"  # cpu, cuda, mps
    
    class Config:
        env_file = ".env"
        
settings = Settings()
''',

        # 通用异常定义
        "common/exceptions.py": '''# -*- coding: utf-8 -*-
"""
自定义异常类
"""

class VoiceWakenAIException(Exception):
    """基础异常类"""
    pass

class ModelLoadError(VoiceWakenAIException):
    """模型加载异常"""
    pass

class AudioProcessingError(VoiceWakenAIException):
    """音频处理异常"""
    pass

class SpeakerRecognitionError(VoiceWakenAIException):
    """声纹识别异常"""
    pass

class ASRError(VoiceWakenAIException):
    """语音识别异常"""
    pass

class LLMError(VoiceWakenAIException):
    """大语言模型异常"""
    pass

class TTSError(VoiceWakenAIException):
    """语音合成异常"""
    pass

class VCError(VoiceWakenAIException):
    """声线变换异常"""
    pass
''',

        # 日志工具
        "utils/logger.py": '''# -*- coding: utf-8 -*-
"""
日志工具
"""

import logging
import sys
from pathlib import Path
from config.settings import settings

def setup_logger(name: str) -> logging.Logger:
    """设置日志器"""
    
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果配置了日志文件）
    if settings.LOG_FILE:
        log_file = Path(settings.LOG_FILE)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
''',

        # requirements.txt
        "requirements.txt": '''# Web框架
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# 音频处理
librosa==0.10.1
soundfile==0.12.1
sounddevice==0.4.6
pyaudio==0.2.13

# AI/ML库
torch==2.1.1
transformers==4.36.0
numpy==1.25.2
scipy==1.11.4

# 语音相关
faster-whisper==0.10.0
coqui-TTS==0.20.6
webrtcvad==2.0.10

# 工具库
pyyaml==6.0.1
requests==2.31.0
tqdm==4.66.1

# 开发工具
pytest==7.4.3
black==23.11.0
isort==5.12.0
''',

        # Git忽略文件（更新）
        ".gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/ 
.env
*.egg-info/
dist/
build/

# AI/ML
*.pth
*.onnx
*.bin
*.safetensors
data/models/
data/cache/
*.wav
*.mp3
*.flac

# IDE
.vscode/
.idea/
*.swp
*.swo

# 系统文件
.DS_Store
Thumbs.db

# 日志
*.log
logs/

# 测试
.pytest_cache/
.coverage
htmlcov/
''',
    }
    
    # 创建文件
    for file_path, content in files_content.items():
        file_obj = Path(file_path)
        if not file_obj.exists():
            file_obj.write_text(content, encoding='utf-8')
            print(f"✓ 创建文件: {file_path}")

def create_module_templates():
    """创建各模块的API模板文件"""
    
    modules = ["speaker_recognition", "asr", "llm", "tts", "vc"]
    
    for module in modules:
        # Controller API模板
        api_content = f'''# -*- coding: utf-8 -*-
"""
{module.upper()} 模块 API
"""

from fastapi import APIRouter, HTTPException
from {module}.service.{module}_service import {module.title().replace('_', '')}Service
from {module}.entity.{module}_schema import *

router = APIRouter()
service = {module.title().replace('_', '')}Service()

@router.get("/health")
async def health_check():
    """健康检查"""
    return {{"status": "healthy", "module": "{module}"}}

# TODO: 添加具体的API端点
'''
        
        api_file = Path(f"{module}/controller/{module}_api.py")
        if not api_file.exists():
            api_file.write_text(api_content, encoding='utf-8')
            print(f"✓ 创建API文件: {api_file}")
        
        # Service服务模板
        service_content = f'''# -*- coding: utf-8 -*-
"""
{module.upper()} 服务层
"""

from utils.logger import setup_logger

logger = setup_logger(__name__)

class {module.title().replace('_', '')}Service:
    """
    {module.upper()} 服务类
    """
    
    def __init__(self):
        logger.info(f"初始化 {module.upper()} 服务")
        # TODO: 初始化模型和资源
        
    async def process(self, input_data):
        """
        处理输入数据
        """
        logger.info(f"{module.upper()} 开始处理数据")
        # TODO: 实现具体的业务逻辑
        return {{"result": "处理完成", "module": "{module}"}}
'''
        
        service_file = Path(f"{module}/service/{module}_service.py")
        if not service_file.exists():
            service_file.write_text(service_content, encoding='utf-8')
            print(f"✓ 创建服务文件: {service_file}")
        
        # Schema实体模板
        schema_content = f'''# -*- coding: utf-8 -*-
"""
{module.upper()} 数据模型
"""

from pydantic import BaseModel
from typing import Optional, Any

class {module.title().replace('_', '')}Request(BaseModel):
    """
    {module.upper()} 请求模型
    """
    # TODO: 定义请求字段
    pass

class {module.title().replace('_', '')}Response(BaseModel):
    """
    {module.upper()} 响应模型
    """
    # TODO: 定义响应字段
    success: bool = True
    message: str = "处理成功"
    data: Optional[Any] = None
'''
        
        schema_file = Path(f"{module}/entity/{module}_schema.py")
        if not schema_file.exists():
            schema_file.write_text(schema_content, encoding='utf-8')
            print(f"✓ 创建实体文件: {schema_file}")

def main():
    """主函数"""
    print("🚀 开始创建 VoiceWakenAI 项目结构...")
    
    # 创建目录结构
    create_directory_structure()
    
    # 创建基础文件
    create_basic_files()
    
    # 创建模块模板
    create_module_templates()
    
    print("\n✅ 项目结构创建完成!")
    print("\n📋 接下来的步骤:")
    print("1. cd VoiceWakenAI")
    print("2. python -m venv venv")
    print("3. venv\\Scripts\\activate  (Windows) 或 source venv/bin/activate (Linux/Mac)")
    print("4. pip install -r requirements.txt")
    print("5. python main.py")
    print("\n🎯 开始开发您的AI语音交互系统吧!")

if __name__ == "__main__":
    main()
