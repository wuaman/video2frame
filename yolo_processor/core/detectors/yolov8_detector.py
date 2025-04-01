import numpy as np
from ultralytics import YOLO
from .base_detector import BaseDetector
from typing import Tuple, List
import cv2

class YOLOv8Detector(BaseDetector):
    """YOLOv8检测器实现"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.model = YOLO(model_path)
        
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        return frame
        
    def batch_preprocess(self, frames: List[np.ndarray], is_gray: bool = True) -> List[np.ndarray]:
        """批量预处理图像
        
        Args:
            frames: 输入图像列表
            is_gray: 是否转换为灰度图
            
        Returns:
            List[np.ndarray]: 预处理后的图像列表
        """
        processed_frames = []
        for frame in frames:
            if is_gray:
                frame = self.convert_to_gray(frame)
                # 转换回三通道，因为YOLO模型需要三通道输入
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            processed_frames.append(self.preprocess(frame))
        return processed_frames
        
    def batch_detect(self, frames: List[np.ndarray], is_gray: bool = True) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """批量检测
        
        Args:
            frames: 输入图像列表
            is_gray: 是否转换为灰度图
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray, np.ndarray]]: 每张图片的检测结果
            - boxes: 边界框坐标 (N, 4)
            - scores: 置信度 (N,)
            - classes: 类别ID (N,)
        """
        processed_frames = self.batch_preprocess(frames, is_gray)
        results = self.model(processed_frames, verbose=False)

        batch_results = []
        for result in results:
            # 使用OBB格式的检测结果
            obb = result.obb
            boxes = obb.xyxy.cpu().numpy()  # 转换为numpy数组
            scores = obb.conf.cpu().numpy()
            classes = obb.cls.cpu().numpy()
            batch_results.append((boxes, scores, classes))
            
        return batch_results

