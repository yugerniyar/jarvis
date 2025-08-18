#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI + Gradio 集成演示脚本
运行此脚本可以看到完整的架构效果
"""

import gradio as gr
from fastapi import FastAPI
import uvicorn
import threading
import time

# 创建FastAPI应用
app = FastAPI(title="VoiceWakenAI API")

@app.get("/")
async def root():
    return {"message": "VoiceWakenAI API is running!"}

@app.post("/api/process_voice")
async def process_voice(text: str):
    """语音处理API演示"""
    return {
        "success": True,
        "original_text": text,
        "response": f"AI回复: 我收到了您的消息'{text}'"
    }

# 创建Gradio界面
def voice_chat_demo(text_input):
    """演示语音对话功能"""
    if not text_input:
        return "请输入文本"
    
    # 模拟处理过程
    response = f"🤖 AI回复: 我收到了您的消息 '{text_input}'\n\n"
    response += "📋 处理流程:\n"
    response += "✅ 1. 声纹识别 - 用户验证通过\n"
    response += "✅ 2. 语音识别 - 文本提取完成\n" 
    response += "✅ 3. 语言模型 - 对话生成完成\n"
    response += "✅ 4. 语音合成 - 音频生成完成\n"
    response += "✅ 5. 声线变换 - 声线调整完成\n"
    
    return response

# 创建Gradio界面
with gr.Blocks(title="VoiceWakenAI Demo") as demo:
    gr.Markdown("# 🎤 VoiceWakenAI - 演示版")
    gr.Markdown("### FastAPI + Gradio 架构演示")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="💬 输入文本（模拟语音输入）",
                placeholder="请输入您想说的话..."
            )
            submit_btn = gr.Button("🚀 处理", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="📤 处理结果",
                lines=10
            )
    
    submit_btn.click(
        fn=voice_chat_demo,
        inputs=text_input,
        outputs=output
    )
    
    gr.Markdown("""
    ### 🏗️ 架构说明
    - **FastAPI**: 提供RESTful API服务 (http://localhost:8000)
    - **Gradio**: 提供Web界面 (http://localhost:7860) 
    - **双模式**: 可以通过API或界面使用
    
    ### 🔧 技术栈
    - 后端: FastAPI + Python AI库
    - 前端: Gradio (轻量级Web界面)
    - 部署: 本地启动，支持并发访问
    """)

def run_fastapi():
    """在后台运行FastAPI"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    print("🚀 启动 VoiceWakenAI 演示...")
    
    # 在后台线程启动FastAPI
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    
    # 等待API启动
    time.sleep(2)
    
    print("✅ FastAPI 服务已启动: http://localhost:8000")
    print("✅ 正在启动 Gradio 界面: http://localhost:7860")
    
    # 启动Gradio界面
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
