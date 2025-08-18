#!/usr/bin/env python3
"""
快速部署脚本 - 根据需求自动安装相应依赖并启动服务
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

def run_command(command, description=""):
    """执行命令并处理输出"""
    print(f"\n{'='*50}")
    print(f"执行: {description}")
    print(f"命令: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ 成功!")
        if result.stdout:
            print("输出:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 失败!")
        print("错误:", e.stderr)
        return False

def install_dependencies(mode="base"):
    """安装依赖包"""
    requirements_files = {
        "base": "requirements/base.txt",
        "inference": "requirements/inference.txt", 
        "training": "requirements/training.txt",
        "full": "requirements.txt"
    }
    
    if mode not in requirements_files:
        print(f"❌ 未知模式: {mode}")
        return False
    
    req_file = requirements_files[mode]
    
    if not os.path.exists(req_file):
        print(f"❌ 依赖文件不存在: {req_file}")
        return False
    
    print(f"\n🚀 安装 {mode} 模式依赖...")
    
    # 升级pip
    run_command("pip install --upgrade pip", "升级pip")
    
    # 安装依赖
    if mode == "training":
        # 训练模式需要分批安装避免冲突
        print("📦 训练模式 - 分批安装依赖...")
        
        # 先安装基础框架
        basic_deps = [
            "torch==2.1.2+cu121",
            "torchvision==0.16.2+cu121", 
            "torchaudio==2.1.2+cu121",
            "accelerate==1.9.0"
        ]
        
        for dep in basic_deps:
            if not run_command(f"pip install {dep}", f"安装 {dep}"):
                return False
        
        # 再安装其他依赖
        run_command(f"pip install -r {req_file}", f"安装训练环境依赖")
        
    else:
        run_command(f"pip install -r {req_file}", f"安装 {mode} 模式依赖")
    
    return True

def check_gpu_support():
    """检查GPU支持"""
    print("\n🔍 检查GPU支持...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU可用: {gpu_name} (数量: {gpu_count})")
            return True
        else:
            print("⚠️  GPU不可用，将使用CPU模式")
            return False
    except ImportError:
        print("❌ PyTorch未安装，无法检查GPU")
        return False

def start_service(service_type="gradio", port=7860):
    """启动服务"""
    print(f"\n🚀 启动 {service_type} 服务 (端口: {port})...")
    
    if service_type == "gradio":
        command = f"python demo_fastapi_gradio.py --port {port}"
    elif service_type == "fastapi":
        command = f"uvicorn demo_fastapi_gradio:app --host 0.0.0.0 --port {port}"
    elif service_type == "microservice":
        command = f"python speaker_recognition/core/service_registry.py"
    else:
        print(f"❌ 未知服务类型: {service_type}")
        return False
    
    print(f"启动命令: {command}")
    try:
        subprocess.run(command, shell=True)
    except KeyboardInterrupt:
        print("\n⏹️  服务已停止")

def create_config_file(mode, gpu_available):
    """创建配置文件"""
    config = {
        "deployment_mode": mode,
        "gpu_available": gpu_available,
        "services": {
            "speaker_recognition": True,
            "speech_synthesis": False,  # 默认关闭
            "voice_conversion": False,
            "noise_reduction": False,
            "audio_enhancement": False
        },
        "ports": {
            "gradio": 7860,
            "fastapi": 8000,
            "microservice_registry": 8001
        }
    }
    
    with open("deployment_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 配置文件已创建: deployment_config.json")

def main():
    parser = argparse.ArgumentParser(description="语音识别微服务部署脚本")
    
    parser.add_argument("--mode", choices=["base", "inference", "training", "full"],
                       default="base", help="部署模式")
    
    parser.add_argument("--service", choices=["gradio", "fastapi", "microservice"],
                       default="gradio", help="启动的服务类型")
    
    parser.add_argument("--port", type=int, default=7860, help="服务端口")
    
    parser.add_argument("--skip-deps", action="store_true", 
                       help="跳过依赖安装")
    
    parser.add_argument("--gpu-check", action="store_true",
                       help="仅检查GPU支持")
    
    args = parser.parse_args()
    
    print("🎯 语音识别微服务部署脚本")
    print("=" * 50)
    
    if args.gpu_check:
        check_gpu_support()
        return
    
    # 安装依赖
    if not args.skip_deps:
        if not install_dependencies(args.mode):
            print("❌ 依赖安装失败")
            return
    
    # 检查GPU支持
    gpu_available = check_gpu_support()
    
    # 创建配置文件
    create_config_file(args.mode, gpu_available)
    
    # 启动服务
    print(f"\n🎉 准备启动 {args.service} 服务...")
    print("提示: 使用 Ctrl+C 停止服务")
    
    start_service(args.service, args.port)

if __name__ == "__main__":
    main()
