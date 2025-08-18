#!/usr/bin/env python3
"""
微服务注册中心和自动发现系统
实现模块化部署，自动检测可用服务，动态生成前端界面
"""

import os
import sys
import importlib
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ServiceInfo:
    """服务信息数据类"""
    name: str
    version: str
    description: str
    endpoints: List[str]
    dependencies: List[str]
    health_check_url: str
    status: str = "unknown"
    port: Optional[int] = None

class ServiceInterface(ABC):
    """服务基础接口 - 所有微服务必须实现"""
    
    @abstractmethod
    def get_service_info(self) -> ServiceInfo:
        """返回服务基本信息"""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """健康检查"""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """服务初始化"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """服务清理"""
        pass

class ServiceRegistry:
    """服务注册中心 - 自动发现和管理微服务"""
    
    def __init__(self):
        self.services: Dict[str, ServiceInterface] = {}
        self.service_info: Dict[str, ServiceInfo] = {}
        self.service_modules = [
            'speaker_recognition_service',
            'speech_synthesis_service', 
            'voice_conversion_service',
            'noise_reduction_service',
            'audio_enhancement_service'
        ]
    
    def auto_discover_services(self) -> List[str]:
        """自动发现可用的服务模块"""
        available_services = []
        
        for service_name in self.service_modules:
            try:
                # 检查模块文件是否存在
                service_path = f"speaker_recognition.service.{service_name}"
                
                # 尝试导入模块
                module = importlib.import_module(service_path)
                
                # 检查模块是否有服务类
                service_class_name = ''.join([word.capitalize() for word in service_name.split('_')])
                if hasattr(module, service_class_name):
                    service_class = getattr(module, service_class_name)
                    
                    # 验证是否实现了ServiceInterface
                    if issubclass(service_class, ServiceInterface):
                        available_services.append(service_name)
                        logger.info(f"发现服务: {service_name}")
                    else:
                        logger.warning(f"服务 {service_name} 未实现 ServiceInterface")
                else:
                    logger.warning(f"模块 {service_name} 中未找到服务类 {service_class_name}")
                    
            except ImportError as e:
                logger.info(f"服务 {service_name} 不可用: {e}")
            except Exception as e:
                logger.error(f"检查服务 {service_name} 时出错: {e}")
        
        return available_services
    
    def register_service(self, service_name: str) -> bool:
        """注册单个服务"""
        try:
            service_path = f"speaker_recognition.service.{service_name}"
            module = importlib.import_module(service_path)
            
            service_class_name = ''.join([word.capitalize() for word in service_name.split('_')])
            service_class = getattr(module, service_class_name)
            
            # 实例化服务
            service_instance = service_class()
            
            # 初始化服务
            if service_instance.initialize():
                # 获取服务信息
                service_info = service_instance.get_service_info()
                
                # 健康检查
                if service_instance.health_check():
                    service_info.status = "healthy"
                else:
                    service_info.status = "unhealthy"
                
                # 注册服务
                self.services[service_name] = service_instance
                self.service_info[service_name] = service_info
                
                logger.info(f"服务 {service_name} 注册成功")
                return True
            else:
                logger.error(f"服务 {service_name} 初始化失败")
                return False
                
        except Exception as e:
            logger.error(f"注册服务 {service_name} 时出错: {e}")
            return False
    
    def unregister_service(self, service_name: str) -> None:
        """注销服务"""
        if service_name in self.services:
            try:
                self.services[service_name].cleanup()
                del self.services[service_name]
                del self.service_info[service_name]
                logger.info(f"服务 {service_name} 已注销")
            except Exception as e:
                logger.error(f"注销服务 {service_name} 时出错: {e}")
    
    def get_healthy_services(self) -> List[str]:
        """获取所有健康的服务"""
        healthy_services = []
        
        for name, service in self.services.items():
            try:
                if service.health_check():
                    self.service_info[name].status = "healthy"
                    healthy_services.append(name)
                else:
                    self.service_info[name].status = "unhealthy"
            except Exception as e:
                logger.error(f"检查服务 {name} 健康状态时出错: {e}")
                self.service_info[name].status = "error"
        
        return healthy_services
    
    def get_service_registry_json(self) -> str:
        """获取服务注册信息的JSON格式"""
        registry_data = {
            "total_services": len(self.services),
            "healthy_services": len(self.get_healthy_services()),
            "services": {
                name: asdict(info) for name, info in self.service_info.items()
            }
        }
        return json.dumps(registry_data, indent=2, ensure_ascii=False)
    
    def start_all_services(self) -> Dict[str, bool]:
        """启动所有发现的服务"""
        available_services = self.auto_discover_services()
        results = {}
        
        for service_name in available_services:
            results[service_name] = self.register_service(service_name)
        
        return results
    
    def stop_all_services(self) -> None:
        """停止所有服务"""
        service_names = list(self.services.keys())
        for service_name in service_names:
            self.unregister_service(service_name)

class DynamicUIGenerator:
    """动态UI生成器 - 根据可用服务生成前端界面"""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.registry = service_registry
    
    def generate_gradio_interface(self) -> str:
        """生成Gradio界面代码"""
        healthy_services = self.registry.get_healthy_services()
        
        interface_code = '''
import gradio as gr
from typing import Dict, Any

def create_dynamic_interface():
    """动态创建包含可用服务的界面"""
    
    # 获取可用服务
    available_services = {}
'''.format()
        
        # 为每个健康服务生成接口
        for service_name in healthy_services:
            service_info = self.registry.service_info[service_name]
            interface_code += f'''
    # {service_info.description}
    def {service_name}_interface():
        service = registry.services["{service_name}"]
        # 这里根据具体服务实现接口逻辑
        return "服务 {service_info.name} 可用"
    
    available_services["{service_name}"] = {service_name}_interface
'''
        
        interface_code += '''
    # 创建动态标签页
    tabs = []
    for service_name, service_func in available_services.items():
        with gr.Tab(service_name):
            gr.Interface(
                fn=service_func,
                inputs=gr.Textbox(label="输入"),
                outputs=gr.Textbox(label="输出"),
                title=f"{service_name} 服务"
            )
    
    return gr.TabbedInterface(tabs, tab_names=list(available_services.keys()))

if __name__ == "__main__":
    interface = create_dynamic_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)
'''
        
        return interface_code
    
    def generate_fastapi_routes(self) -> str:
        """生成FastAPI路由代码"""
        healthy_services = self.registry.get_healthy_services()
        
        routes_code = '''
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import json

app = FastAPI(title="语音识别微服务集群", version="1.0.0")

# 服务状态端点
@app.get("/services/status")
async def get_services_status():
    """获取所有服务状态"""
    return json.loads(registry.get_service_registry_json())

@app.get("/services/health")
async def health_check():
    """集群健康检查"""
    healthy_services = registry.get_healthy_services()
    return {
        "status": "healthy" if healthy_services else "unhealthy",
        "healthy_services": healthy_services,
        "total_services": len(registry.services)
    }
'''
        
        # 为每个服务生成API端点
        for service_name in healthy_services:
            service_info = self.registry.service_info[service_name]
            
            routes_code += f'''
# {service_info.description} API
@app.post("/{service_name}/process")
async def {service_name}_process(data: Dict[str, Any]):
    """处理 {service_info.name} 请求"""
    try:
        service = registry.services["{service_name}"]
        if not service.health_check():
            raise HTTPException(status_code=503, detail="服务不可用")
        
        # 这里调用具体的服务方法
        result = await service.process(data)
        return {{"status": "success", "result": result}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/{service_name}/info")
async def {service_name}_info():
    """获取 {service_info.name} 服务信息"""
    return registry.service_info["{service_name}"]
'''
        
        return routes_code

# 全局服务注册中心实例
registry = ServiceRegistry()

def initialize_microservice_system():
    """初始化微服务系统"""
    logger.info("初始化微服务系统...")
    
    # 启动所有可用服务
    results = registry.start_all_services()
    
    # 输出启动结果
    logger.info("服务启动结果:")
    for service_name, success in results.items():
        status = "成功" if success else "失败"
        logger.info(f"  {service_name}: {status}")
    
    # 输出服务注册信息
    logger.info("当前服务注册状态:")
    logger.info(registry.get_service_registry_json())
    
    return registry

if __name__ == "__main__":
    # 初始化系统
    registry = initialize_microservice_system()
    
    # 生成动态界面
    ui_generator = DynamicUIGenerator(registry)
    
    # 保存生成的代码
    with open("generated_gradio_interface.py", "w", encoding="utf-8") as f:
        f.write(ui_generator.generate_gradio_interface())
    
    with open("generated_fastapi_routes.py", "w", encoding="utf-8") as f:
        f.write(ui_generator.generate_fastapi_routes())
    
    logger.info("动态界面代码已生成")
