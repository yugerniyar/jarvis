# -*- coding: utf-8 -*-
"""
声纹识别数据模型
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class SpeakerRegistrationResponse(BaseModel):
    """声纹注册响应模型"""
    success: bool = True
    message: str = "注册成功"
    speaker_id: str
    embedding_shape: Optional[List[int]] = None

class SpeakerVerificationResponse(BaseModel):
    """声纹验证响应模型"""
    success: bool = True
    authorized: bool = False
    message: str = "验证完成"
    speaker_id: str
    similarity: float = Field(..., ge=0.0, le=1.0, description="相似度分数")
    threshold: float = Field(0.6, ge=0.0, le=1.0, description="阈值")

class SpeakerIdentificationResponse(BaseModel):
    """声纹识别响应模型"""
    success: bool = True
    identified_speaker: Optional[str] = None
    similarity: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(0.6, ge=0.0, le=1.0)
    all_similarities: Dict[str, float] = {}
    message: str = "识别完成"

class SpeakerListResponse(BaseModel):
    """说话人列表响应模型"""
    success: bool = True
    speakers: List[str] = []
    count: int = 0
    message: str = "获取成功"

class SpeakerStreamRequest(BaseModel):
    """实时音频流请求模型"""
    speaker_id: str = Field(..., description="说话人ID")
    audio_data: str = Field(..., description="音频数据（base64编码）")
    sample_rate: int = Field(16000, description="采样率")
    threshold: float = Field(0.6, ge=0.0, le=1.0, description="相似度阈值")

class SpeakerEmbeddingResponse(BaseModel):
    """声纹特征提取响应模型"""
    success: bool = True
    embedding: List[float] = []
    embedding_shape: List[int] = []
    message: str = "特征提取成功"

# 用于内部处理的请求模型
class SpeakerRecognitionRequest(BaseModel):
    """声纹识别请求模型（内部使用）"""
    action: str = Field(..., description="操作类型: register/verify/identify")
    speaker_id: Optional[str] = None
    audio_path: Optional[str] = None
    threshold: float = 0.6

class SpeakerRecognitionResponse(BaseModel):
    """声纹识别响应模型（通用）"""
    success: bool = True
    message: str = "处理成功"
    data: Optional[Dict] = None
