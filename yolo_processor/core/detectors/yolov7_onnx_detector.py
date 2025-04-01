import cv2
import numpy as np
import onnxruntime
from torch2trt import TRTModule
import tensorrt as trt
import time
import torch
from typing import Tuple
from .base_detector import BaseDetector

class YOLOv7ONNXDetector(BaseDetector):
    """YOLOv7 ONNX检测器实现"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]


    def preprocess(self, frame: np.ndarray, is_gray: bool = True) -> np.ndarray:
        """YOLOv7的图像预处理"""
        if is_gray:
            frame = self.convert_to_gray(frame)
            # 转换回三通道
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        input_height, input_width = self.input_shape[2:]
        
        # 调整图像大小
        image = cv2.resize(frame, (input_width, input_height))
        
        # BGR到RGB转换
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # HWC到NCHW
        image = image.transpose(2, 0, 1)
        
        # 添加batch维度
        image = np.expand_dims(image, axis=0)
        
        return image
        
    def detect(self, frame: np.ndarray, is_gray: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 预处理
        input_tensor = self.preprocess(frame, is_gray)
        
        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        num_dets, det_boxes, det_scores, det_classes = outputs
        
        # 后处理
        # 只取第一个batch的结果
        valid_dets = int(num_dets[0])
        boxes = det_boxes[0][:valid_dets]
        scores = det_scores[0][:valid_dets]
        classes = det_classes[0][:valid_dets]
        
        # 转换坐标到原图尺寸
        orig_h, orig_w = frame.shape[:2]
        input_h, input_w = self.input_shape[2:]
        
        scale_x = orig_w / input_w
        scale_y = orig_h / input_h
        
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        return boxes, scores, classes 