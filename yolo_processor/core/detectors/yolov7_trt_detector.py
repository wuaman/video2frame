import cv2
import numpy as np
import torch
import tensorrt as trt
from torch2trt import TRTModule
from typing import Tuple, List
from .base_detector import BaseDetector

class YOLOv7TRTDetector(BaseDetector):
    """YOLOv7 TensorRT检测器实现（使用torch2trt）"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.device = torch.device('cuda:0')
        self._model = self._loadModel(model_path)
        self.input_shape = self._model.input_shape
        self._warmup = True
        self.stream = torch.cuda.Stream()
        self.max_batch_size = 32  # 设置最大batch size

    def _loadModel(self, modelPath: str) -> TRTModule:
        # 加载TRT引擎
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, namespace="")
        
        with open(modelPath, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        
        # 创建TRTModule并获取输入尺寸
        model = TRTModule(engine, input_names=['images'], output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes'])
        model.input_shape = engine.get_binding_shape(0)
        
        # 处理动态batch size
        if model.input_shape[0] == -1:
            model.input_shape = list(model.input_shape)
            model.input_shape[0] = 1  # 预热时使用batch_size=1
            
        print(f"模型输入形状: {model.input_shape}")
        
        # Warmup模型
        if True:
            input_tensor = torch.zeros(tuple(model.input_shape), dtype=torch.float32).to(self.device)
            try:
                for _ in range(10):  # 减少预热次数
                    _ = model(input_tensor)
                print("模型预热完成")
            except Exception as e:
                print(f"模型预热失败: {str(e)}")
        
        return model.to(self.device)

    def preprocess(self, frame: np.ndarray, is_gray: bool = True) -> torch.Tensor:
        """预处理流程（返回PyTorch Tensor）"""
        if is_gray:
            frame = self.convert_to_gray(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        input_h, input_w = self.input_shape[2:]
        image = cv2.resize(frame, (input_w, input_h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image).unsqueeze(0).to(self.device)

    def batch_preprocess(self, frames: List[np.ndarray], is_gray: bool = True) -> torch.Tensor:
        """批量预处理，使用线程池加速"""
        def process_single(frame):
            if is_gray:
                frame = self.convert_to_gray(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
            input_h, input_w = self.input_shape[2:]
            image = cv2.resize(frame, (input_w, input_h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            image = image.transpose(2, 0, 1)
            return torch.from_numpy(image)

        if self.thread_pool is not None:
            # 使用线程池并行处理
            batch = list(self.thread_pool.map(process_single, frames))
        else:
            # 串行处理
            batch = [process_single(frame) for frame in frames]
            
        # 合并batch并移到GPU
        batch_tensor = torch.stack(batch).to(self.device)
        
        # 确保batch size不超过模型的限制
        if batch_tensor.shape[0] > self.max_batch_size:
            raise ValueError(f"Batch size {batch_tensor.shape[0]} 超过了模型最大限制 {self.max_batch_size}")
            
        return batch_tensor

    # def detect(self, frame: np.ndarray, is_gray: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     # 预处理并转换为Tensor
    #     input_tensor = self.preprocess(frame, is_gray)
        
    #     # 执行推理
    #     with torch.no_grad():
    #         num_dets, det_boxes, det_scores, det_classes = self._model(input_tensor)
        
    #     # 转换为numpy数组
    #     num_dets = num_dets.cpu().numpy()
    #     det_boxes = det_boxes.cpu().numpy()
    #     det_scores = det_scores.cpu().numpy()
    #     det_classes = det_classes.cpu().numpy()
        
    #     # 后处理
    #     valid_dets = int(num_dets[0][0])
    #     boxes = det_boxes[0][:valid_dets]
    #     scores = det_scores[0][:valid_dets]
    #     classes = det_classes[0][:valid_dets]
        
    #     # 坐标转换
    #     orig_h, orig_w = frame.shape[:2]
    #     input_h, input_w = self.input_shape[2:]
    #     boxes[:, [0, 2]] *= orig_w / input_w
    #     boxes[:, [1, 3]] *= orig_h / input_h
        
    #     return boxes, scores, classes

    def batch_detect(self, frames: List[np.ndarray], is_gray: bool = True) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """批量检测实现"""
        if not frames:
            return []
            
        try:
            # 批量预处理
            with torch.cuda.stream(self.stream):
                batch_tensor = self.batch_preprocess(frames, is_gray)
                
                # 批量推理
                with torch.no_grad():
                    num_dets, det_boxes, det_scores, det_classes = self._model(batch_tensor)

                # 同步GPU流
                self.stream.synchronize()

                # 转移到CPU
                num_dets = num_dets.cpu().numpy()
                det_boxes = det_boxes.cpu().numpy()
                det_scores = det_scores.cpu().numpy()
                det_classes = det_classes.cpu().numpy()

            # 批量后处理
            results = []
            batch_size = len(frames)
            for i in range(batch_size):
                try:
                    # 获取当前帧的检测结果
                    valid_dets = int(num_dets[i][0])
                    boxes = det_boxes[i][:valid_dets].copy()
                    scores = det_scores[i][:valid_dets].copy()
                    classes = det_classes[i][:valid_dets].copy()
                    
                    # 坐标转换
                    orig_h, orig_w = frames[i].shape[:2]
                    input_h, input_w = self.input_shape[2:]
                    boxes[:, [0, 2]] *= orig_w / input_w
                    boxes[:, [1, 3]] *= orig_h / input_h
                    
                    results.append((boxes, scores, classes))
                except Exception as e:
                    print(f"处理第 {i} 帧时出错: {str(e)}")
                    results.append((np.array([]), np.array([]), np.array([])))
                    
            return results
            
        except Exception as e:
            print(f"批量检测出错: {str(e)}")
            # 返回空结果
            return [(np.array([]), np.array([]), np.array([]))] * len(frames)