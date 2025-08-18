# -*- coding: utf-8 -*-
"""
声纹识别模块 API 路由
"""

from fastapi import APIRouter
from speaker_recognition.controller.speaker_recognition_api import router as speaker_router

# 创建主路由
router = APIRouter()

# 包含声纹识别的所有API端点
router.include_router(speaker_router, tags=["声纹识别"])

@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "module": "speaker_recognition"}
