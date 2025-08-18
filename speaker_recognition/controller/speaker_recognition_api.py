# -*- coding: utf-8 -*-
"""
声纹识别模块 API
"""

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import librosa
import io

from speaker_recognition.service.speaker_recognition_service import SpeakerRecognitionService
from speaker_recognition.entity.speaker_recognition_schema import *

router = APIRouter()
service = SpeakerRecognitionService()

@router.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "module": "speaker_recognition"}

@router.post("/register", response_model=SpeakerRegistrationResponse)
async def register_speaker(
    speaker_id: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """
    注册新说话人
    
    Args:
        speaker_id: 说话人ID
        audio_file: 音频文件（WAV格式推荐）
    """
    try:
        # 读取音频文件
        audio_content = await audio_file.read()
        
        # 使用 librosa 加载音频
        audio_data, sample_rate = librosa.load(io.BytesIO(audio_content), sr=16000)
        
        # 调用服务层注册
        result = service.register_speaker(speaker_id, audio_data, sample_rate)
        
        if result['success']:
            return SpeakerRegistrationResponse(
                success=True,
                message=result['message'],
                speaker_id=speaker_id,
                embedding_shape=list(result['embedding_shape'])
            )
        else:
            raise HTTPException(status_code=400, detail=result['message'])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注册失败: {str(e)}")

@router.post("/verify", response_model=SpeakerVerificationResponse)
async def verify_speaker(
    speaker_id: str = Form(...),
    audio_file: UploadFile = File(...),
    threshold: float = Form(0.6)
):
    """
    验证说话人身份
    
    Args:
        speaker_id: 待验证的说话人ID
        audio_file: 音频文件
        threshold: 相似度阈值
    """
    try:
        # 读取音频文件
        audio_content = await audio_file.read()
        audio_data, sample_rate = librosa.load(io.BytesIO(audio_content), sr=16000)
        
        # 调用服务层验证
        result = service.verify_speaker(speaker_id, audio_data, sample_rate, threshold)
        
        return SpeakerVerificationResponse(
            success=result['success'],
            authorized=result['authorized'],
            message=result['message'],
            speaker_id=speaker_id,
            similarity=result['similarity'],
            threshold=threshold
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"验证失败: {str(e)}")

@router.post("/identify", response_model=SpeakerIdentificationResponse)
async def identify_speaker(
    audio_file: UploadFile = File(...),
    threshold: float = Form(0.6)
):
    """
    识别说话人（从所有注册用户中找最匹配的）
    
    Args:
        audio_file: 音频文件
        threshold: 相似度阈值
    """
    try:
        # 读取音频文件
        audio_content = await audio_file.read()
        audio_data, sample_rate = librosa.load(io.BytesIO(audio_content), sr=16000)
        
        # 调用服务层识别
        result = service.identify_speaker(audio_data, sample_rate, threshold)
        
        return SpeakerIdentificationResponse(
            success=result['success'],
            identified_speaker=result['identified_speaker'],
            similarity=result['similarity'],
            threshold=threshold,
            all_similarities=result.get('all_similarities', {}),
            message=result['message']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")

@router.get("/speakers", response_model=SpeakerListResponse)
async def list_speakers():
    """获取所有注册的说话人列表"""
    try:
        speakers = service.list_speakers()
        return SpeakerListResponse(
            success=True,
            speakers=speakers,
            count=len(speakers),
            message=f"共有 {len(speakers)} 个注册用户"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取说话人列表失败: {str(e)}")

@router.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str):
    """删除注册的说话人"""
    try:
        result = service.delete_speaker(speaker_id)
        
        if result['success']:
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=404, detail=result['message'])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

# 实时音频流处理接口（高级功能）
@router.post("/verify-stream")
async def verify_speaker_stream(request: SpeakerStreamRequest):
    """
    实时音频流声纹验证（接收音频字节流）
    
    Args:
        request: 包含说话人ID、音频数据、采样率等信息
    """
    try:
        # 解码音频数据（假设是base64编码）
        import base64
        audio_bytes = base64.b64decode(request.audio_data)
        
        # 转换为numpy数组
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # 调用服务层验证
        result = service.verify_speaker(
            request.speaker_id, 
            audio_data, 
            request.sample_rate, 
            request.threshold
        )
        
        return SpeakerVerificationResponse(
            success=result['success'],
            authorized=result['authorized'],
            message=result['message'],
            speaker_id=request.speaker_id,
            similarity=result['similarity'],
            threshold=request.threshold
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"实时验证失败: {str(e)}")
