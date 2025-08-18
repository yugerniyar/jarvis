# -*- coding: utf-8 -*-
"""
声纹识别服务层 - 基于 SpeechBrain 实现
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import logging

# SpeechBrain 相关导入
try:
    from speechbrain.pretrained import SpeakerRecognition
    from speechbrain.pretrained import EncoderClassifier
except ImportError:
    print("Warning: SpeechBrain not installed. Please run: pip install speechbrain")
    SpeakerRecognition = None
    EncoderClassifier = None

# Resemblyzer 作为备选方案
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
except ImportError:
    print("Warning: Resemblyzer not installed. Please run: pip install resemblyzer")
    VoiceEncoder = None

from utils.logger import setup_logger
from config.settings import settings

logger = setup_logger(__name__)

class SpeakerRecognitionService:
    """
    高级声纹识别服务类
    支持多种后端：SpeechBrain（主推）、Resemblyzer（备选）
    """
    
    def __init__(self, backend='speechbrain'):
        """
        初始化声纹识别服务
        
        Args:
            backend: 选择后端 'speechbrain' 或 'resemblyzer'
        """
        self.backend = backend
        self.device = torch.device(settings.DEVICE if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.encoder = None
        self.speaker_embeddings = {}  # 存储注册用户的声纹特征
        self.embeddings_file = Path(settings.DATA_DIR) / "speaker_embeddings" / "embeddings.pkl"
        
        logger.info(f"初始化声纹识别服务 - 后端: {backend}, 设备: {self.device}")
        
        # 初始化模型
        self._load_model()
        
        # 加载已有的声纹数据库
        self._load_speaker_database()
    
    def _load_model(self):
        """加载声纹识别模型"""
        try:
            if self.backend == 'speechbrain' and SpeakerRecognition is not None:
                # 使用 SpeechBrain 预训练模型
                logger.info("加载 SpeechBrain ECAPA-TDNN 模型...")
                self.model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="data/models/speaker_recognition",
                    run_opts={"device": str(self.device)}
                )
                logger.info("SpeechBrain 模型加载成功")
                
            elif self.backend == 'resemblyzer' and VoiceEncoder is not None:
                # 使用 Resemblyzer
                logger.info("加载 Resemblyzer 模型...")
                self.encoder = VoiceEncoder(device=str(self.device))
                logger.info("Resemblyzer 模型加载成功")
                
            else:
                raise ImportError(f"Backend {self.backend} not available")
                
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            # 尝试备选方案
            if self.backend == 'speechbrain' and VoiceEncoder is not None:
                logger.info("回退到 Resemblyzer...")
                self.backend = 'resemblyzer'
                self.encoder = VoiceEncoder(device=str(self.device))
            else:
                raise e
    
    def _load_speaker_database(self):
        """加载声纹数据库"""
        try:
            if self.embeddings_file.exists():
                with open(self.embeddings_file, 'rb') as f:
                    self.speaker_embeddings = pickle.load(f)
                logger.info(f"加载声纹数据库成功，包含 {len(self.speaker_embeddings)} 个说话人")
            else:
                logger.info("声纹数据库不存在，将创建新的数据库")
                self.embeddings_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"加载声纹数据库失败: {str(e)}")
            self.speaker_embeddings = {}
    
    def _save_speaker_database(self):
        """保存声纹数据库"""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.speaker_embeddings, f)
            logger.info("声纹数据库保存成功")
        except Exception as e:
            logger.error(f"保存声纹数据库失败: {str(e)}")
    
    def extract_embedding(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        提取声纹特征向量
        
        Args:
            audio_data: 音频数据 (numpy array)
            sample_rate: 采样率
            
        Returns:
            声纹特征向量
        """
        try:
            # 预处理音频
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # 转单声道
            
            # 重采样到16kHz
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            if self.backend == 'speechbrain':
                # SpeechBrain 提取特征
                with torch.no_grad():
                    # 转换为 torch tensor
                    audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0).to(self.device)
                    # 提取嵌入
                    embedding = self.model.encode_batch(audio_tensor)
                    return embedding.squeeze().cpu().numpy()
                    
            elif self.backend == 'resemblyzer':
                # Resemblyzer 提取特征
                # 预处理音频
                processed_audio = preprocess_wav(audio_data, 16000)
                # 提取嵌入
                embedding = self.encoder.embed_utterance(processed_audio)
                return embedding
                
        except Exception as e:
            logger.error(f"特征提取失败: {str(e)}")
            raise e
    
    def register_speaker(self, speaker_id: str, audio_data: np.ndarray, 
                        sample_rate: int = 16000) -> Dict:
        """
        注册新说话人
        
        Args:
            speaker_id: 说话人ID
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            注册结果
        """
        try:
            logger.info(f"注册说话人: {speaker_id}")
            
            # 提取声纹特征
            embedding = self.extract_embedding(audio_data, sample_rate)
            
            # 存储到数据库
            self.speaker_embeddings[speaker_id] = {
                'embedding': embedding,
                'registered_time': np.datetime64('now').astype(str)
            }
            
            # 保存数据库
            self._save_speaker_database()
            
            logger.info(f"说话人 {speaker_id} 注册成功")
            return {
                'success': True,
                'message': f'说话人 {speaker_id} 注册成功',
                'speaker_id': speaker_id,
                'embedding_shape': embedding.shape
            }
            
        except Exception as e:
            logger.error(f"说话人注册失败: {str(e)}")
            return {
                'success': False,
                'message': f'注册失败: {str(e)}',
                'speaker_id': speaker_id
            }
    
    def verify_speaker(self, speaker_id: str, audio_data: np.ndarray, 
                      sample_rate: int = 16000, threshold: float = 0.6) -> Dict:
        """
        验证说话人身份
        
        Args:
            speaker_id: 待验证的说话人ID
            audio_data: 音频数据
            sample_rate: 采样率
            threshold: 相似度阈值
            
        Returns:
            验证结果
        """
        try:
            if speaker_id not in self.speaker_embeddings:
                return {
                    'success': False,
                    'authorized': False,
                    'message': f'说话人 {speaker_id} 未注册',
                    'speaker_id': speaker_id,
                    'similarity': 0.0
                }
            
            # 提取当前音频的声纹特征
            current_embedding = self.extract_embedding(audio_data, sample_rate)
            
            # 获取注册时的特征
            registered_embedding = self.speaker_embeddings[speaker_id]['embedding']
            
            # 计算相似度（余弦相似度）
            similarity = self._cosine_similarity(current_embedding, registered_embedding)
            
            # 判断是否通过验证
            authorized = similarity >= threshold
            
            logger.info(f"说话人验证 - ID: {speaker_id}, 相似度: {similarity:.3f}, 通过: {authorized}")
            
            return {
                'success': True,
                'authorized': authorized,
                'message': '验证通过' if authorized else '验证失败',
                'speaker_id': speaker_id,
                'similarity': float(similarity),
                'threshold': threshold
            }
            
        except Exception as e:
            logger.error(f"说话人验证失败: {str(e)}")
            return {
                'success': False,
                'authorized': False,
                'message': f'验证失败: {str(e)}',
                'speaker_id': speaker_id,
                'similarity': 0.0
            }
    
    def identify_speaker(self, audio_data: np.ndarray, sample_rate: int = 16000, 
                        threshold: float = 0.6) -> Dict:
        """
        识别说话人（从所有注册用户中找最匹配的）
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            threshold: 相似度阈值
            
        Returns:
            识别结果
        """
        try:
            if not self.speaker_embeddings:
                return {
                    'success': False,
                    'message': '没有注册的说话人',
                    'identified_speaker': None,
                    'similarity': 0.0
                }
            
            # 提取当前音频的声纹特征
            current_embedding = self.extract_embedding(audio_data, sample_rate)
            
            # 与所有注册用户比较
            best_match = None
            best_similarity = 0.0
            
            similarities = {}
            for speaker_id, data in self.speaker_embeddings.items():
                similarity = self._cosine_similarity(current_embedding, data['embedding'])
                similarities[speaker_id] = similarity
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = speaker_id
            
            # 判断是否超过阈值
            identified = best_similarity >= threshold
            
            logger.info(f"说话人识别 - 最佳匹配: {best_match}, 相似度: {best_similarity:.3f}")
            
            return {
                'success': True,
                'identified_speaker': best_match if identified else None,
                'similarity': float(best_similarity),
                'threshold': threshold,
                'all_similarities': {k: float(v) for k, v in similarities.items()},
                'message': f'识别为 {best_match}' if identified else '未识别到匹配的说话人'
            }
            
        except Exception as e:
            logger.error(f"说话人识别失败: {str(e)}")
            return {
                'success': False,
                'message': f'识别失败: {str(e)}',
                'identified_speaker': None,
                'similarity': 0.0
            }
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """计算两个特征向量的余弦相似度"""
        # 归一化向量
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        # 计算余弦相似度
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def list_speakers(self) -> List[str]:
        """获取所有注册的说话人列表"""
        return list(self.speaker_embeddings.keys())
    
    def delete_speaker(self, speaker_id: str) -> Dict:
        """删除注册的说话人"""
        try:
            if speaker_id in self.speaker_embeddings:
                del self.speaker_embeddings[speaker_id]
                self._save_speaker_database()
                logger.info(f"删除说话人 {speaker_id} 成功")
                return {
                    'success': True,
                    'message': f'说话人 {speaker_id} 删除成功'
                }
            else:
                return {
                    'success': False,
                    'message': f'说话人 {speaker_id} 不存在'
                }
        except Exception as e:
            logger.error(f"删除说话人失败: {str(e)}")
            return {
                'success': False,
                'message': f'删除失败: {str(e)}'
            }

    async def process(self, input_data):
        """
        处理输入数据（兼容基类接口）
        """
        logger.info("SPEAKER_RECOGNITION 开始处理数据")
        # TODO: 根据具体需求实现
        return {"result": "处理完成", "module": "speaker_recognition"}
