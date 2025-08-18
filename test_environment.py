#!/usr/bin/env python3
"""
环境测试脚本 - 验证VoiceWakenAI项目的依赖环境
"""
import sys
import importlib
import traceback

def test_import(module_name, package_name=None):
    """测试模块导入"""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name or module_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: Import failed - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {package_name or module_name}: {e}")
        return False

def test_pytorch_gpu():
    """测试PyTorch GPU支持"""
    try:
        import torch
        print(f"🔥 PyTorch版本: {torch.__version__}")
        print(f"🔥 CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"🔥 CUDA版本: {torch.version.cuda}")
            print(f"🔥 GPU数量: {torch.cuda.device_count()}")
            print(f"🔥 GPU名称: {torch.cuda.get_device_name(0)}")
            
            # 简单GPU测试
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = torch.mm(x, y)
            print(f"🔥 GPU计算测试: 成功 (结果形状: {z.shape})")
        return True
    except Exception as e:
        print(f"❌ PyTorch GPU测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("🚀 VoiceWakenAI 环境测试")
    print("="*60)
    
    # 基本信息
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print("-"*60)
    
    # 核心依赖测试
    core_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("torchaudio", "TorchAudio"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("datasets", "Datasets"),
    ]
    
    # Web框架测试
    web_packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("gradio", "Gradio"),
        ("pydantic", "Pydantic"),
    ]
    
    # 音频处理测试
    audio_packages = [
        ("librosa", "Librosa"),
        ("soundfile", "SoundFile"),
        ("sounddevice", "SoundDevice"),
        ("webrtcvad", "WebRTC VAD"),
    ]
    
    # 机器学习测试
    ml_packages = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "Scikit-learn"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
    ]
    
    success_count = 0
    total_count = 0
    
    print("🔍 核心深度学习框架:")
    for module, name in core_packages:
        if test_import(module, name):
            success_count += 1
        total_count += 1
    
    print("\n🌐 Web框架:")
    for module, name in web_packages:
        if test_import(module, name):
            success_count += 1
        total_count += 1
    
    print("\n🎵 音频处理:")
    for module, name in audio_packages:
        if test_import(module, name):
            success_count += 1
        total_count += 1
    
    print("\n📊 机器学习库:")
    for module, name in ml_packages:
        if test_import(module, name):
            success_count += 1
        total_count += 1
    
    print("\n" + "="*60)
    print("🔥 PyTorch GPU 测试:")
    test_pytorch_gpu()
    
    print("\n" + "="*60)
    print(f"📊 测试结果: {success_count}/{total_count} 包导入成功")
    
    if success_count == total_count:
        print("🎉 所有核心依赖都已正确安装！")
        return True
    else:
        print("⚠️  部分依赖缺失，但核心功能应该可用")
        return False

if __name__ == "__main__":
    main()
