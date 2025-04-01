from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Optional
import cv2
import torch
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

class BaseDetector(ABC):
    """检测器基类"""
    
    @abstractmethod
    def __init__(self, model_path: str):
        """初始化检测器
        
        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.preprocess_queue = Queue(maxsize=2)  # 双缓冲队列
        self.result_queue = Queue(maxsize=2)
        self.thread_pool = None

    def convert_to_gray(self, frame: np.ndarray) -> np.ndarray:
        """转换为灰度图像
        
        Args:
            frame: 输入图像
            
        Returns:
            np.ndarray: 灰度图像
        """
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    # @abstractmethod
    # def detect(self, frame: np.ndarray, is_gray: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """执行检测
        
    #     Args:
    #         frame: 输入图像
    #         is_gray: 是否转换为灰度图
            
    #     Returns:
    #         Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    #         - boxes: 边界框坐标 shape=(N, 4)
    #         - scores: 置信度 shape=(N,)
    #         - classes: 类别ID shape=(N,)
    #     """
    #     pass

    @abstractmethod
    def preprocess(self, frame: np.ndarray, is_gray: bool = True) -> np.ndarray:
        """图像预处理
        
        Args:
            frame: 输入图像
            is_gray: 是否转换为灰度图
            
        Returns:
            np.ndarray: 预处理后的图像
        """
        pass

    @abstractmethod
    def batch_preprocess(self, frames: List[np.ndarray], is_gray: bool = True) -> torch.Tensor:
        """批量预处理
        
        Args:
            frames: 输入图像列表
            is_gray: 是否转换为灰度图
            
        Returns:
            torch.Tensor: 预处理后的批量图像
        """
        pass

    @abstractmethod
    def batch_detect(self, frames: List[np.ndarray], is_gray: bool = True) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """批量检测接口
        
        Args:
            frames: 输入图像列表
            is_gray: 是否转换为灰度图
            
        Returns:
            List[Tuple[np.ndarray, np.ndarray, np.ndarray]]: 每张图片的检测结果
        """
        pass

    def start_workers(self, num_workers: int):
        """启动工作线程"""
        if self.thread_pool is None:
            self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)

    def stop_workers(self):
        """停止工作线程"""
        if self.thread_pool is not None:
            self.thread_pool.shutdown()
            self.thread_pool = None 