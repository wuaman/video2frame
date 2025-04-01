from enum import Enum
from typing import Optional, Callable, List
import numpy as np
import cv2
import os
from tqdm import tqdm
from queue import Queue
from threading import Event
import threading

class ProcessMode(Enum):
    CROP = "crop"
    FILTER = "filter"

class BaseProcessor:
    """图像处理器基类"""
    
    def __init__(self, detector, output_folder: str = "./output", 
                 batch_size: int = 1, num_workers: int = 2):
        self.detector = detector
        self.output_folder = output_folder
        self.batch_size = batch_size
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        # 初始化批处理相关的队列和事件
        self.frame_queue = Queue(maxsize=batch_size * 2)  # 原始帧队列
        self.result_queue = Queue(maxsize=batch_size * 2)  # 检测结果队列
        self.stop_event = Event()
        self.detector_thread = None
        self.processor_thread = None
        
        # 启动检测器的工作线程
        if batch_size > 1:
            self.detector.start_workers(num_workers)
            
    def _crop_objects(self, frame: np.ndarray, boxes: np.ndarray, 
                     classes: np.ndarray, save_name: str) -> None:
        """裁切并保存目标图像
        
        Args:
            frame: 输入图像
            boxes: 边界框坐标 (N, 4)
            classes: 类别ID (N,)
            save_name: 保存文件名
        """
        for idx, (box, cls) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            cropped_image = frame[y1:y2, x1:x2]
            save_path = os.path.join(
                self.output_folder,
                f"{os.path.splitext(save_name)[0]}_object_{idx}_class_{int(cls)}.jpg"
            )
            cv2.imwrite(save_path, cropped_image)
            
    def _process_batch(self, frames: List[np.ndarray], 
                      save_names: List[str], **kwargs) -> None:
        """处理一个批次的图像"""
        if not frames:
            return
            
        try:
            # 执行批量检测
            results = self.detector.batch_detect(frames, is_gray=kwargs.get('is_gray', True))
            
            # 处理每个结果
            for frame, (boxes, scores, classes), save_name in zip(frames, results, save_names):
                try:
                    # 类别过滤
                    if kwargs.get('target_classes') is not None:
                        mask = np.isin(classes, kwargs['target_classes'])
                        boxes = boxes[mask]
                        scores = scores[mask]
                        classes = classes[mask]
                        
                    if len(boxes) == 0:
                        continue
                        
                    # 根据模式处理
                    if kwargs.get('mode') == ProcessMode.FILTER:
                        if kwargs.get('filter_rule') is None:
                            raise ValueError("过滤模式下必须提供filter_rule")
                        if kwargs['filter_rule'](classes, scores):
                            # 删除图片
                            original_path = kwargs.get('img_path')
                            if original_path is not None:
                                try:
                                    os.remove(original_path)
                                except Exception as e:
                                    print(f"删除文件失败 {original_path}: {str(e)}")
                    else:
                        self._crop_objects(frame, boxes, classes, save_name)
                except Exception as e:
                    print(f"处理图像 {save_name} 时出错: {str(e)}")
                    
        except Exception as e:
            print(f"批处理出错: {str(e)}")
                
    def _detector_thread_func(self, **kwargs):
        """检测线程：从frame_queue获取帧，进行检测，结果放入result_queue"""
        frames = []
        save_names = []
        
        while not self.stop_event.is_set() or not self.frame_queue.empty():
            try:
                frame, save_name = self.frame_queue.get(timeout=1.0)
                frames.append(frame)
                save_names.append(save_name)
                
                # 当累积足够的帧或队列为空时处理batch
                if len(frames) >= self.batch_size or (self.frame_queue.empty() and frames):
                    try:
                        # 执行批量检测
                        results = self.detector.batch_detect(frames, is_gray=kwargs.get('is_gray', True))
                        # 将结果和对应的保存名称放入结果队列
                        self.result_queue.put((frames, results, save_names))
                    except Exception as e:
                        print(f"检测出错: {str(e)}")
                    frames = []
                    save_names = []
            except:
                if frames:  # 处理剩余的帧
                    try:
                        results = self.detector.batch_detect(frames, is_gray=kwargs.get('is_gray', True))
                        self.result_queue.put((frames, results, save_names))
                    except Exception as e:
                        print(f"处理剩余帧时出错: {str(e)}")
                break
                
    def _processor_thread_func(self, **kwargs):
        """处理线程：从result_queue获取检测结果，进行后处理"""
        while not self.stop_event.is_set() or not self.result_queue.empty():
            try:
                frames, results, save_names = self.result_queue.get(timeout=1.0)
                
                # 处理每个结果
                for frame, (boxes, scores, classes), save_name in zip(frames, results, save_names):
                    try:
                        # 类别过滤
                        if kwargs.get('target_classes') is not None:
                            mask = np.isin(classes, kwargs['target_classes'])
                            boxes = boxes[mask]
                            scores = scores[mask]
                            classes = classes[mask]
                            
                        if len(boxes) == 0:
                            continue
                            
                        # 根据模式处理
                        if kwargs.get('mode') == ProcessMode.FILTER:
                            if kwargs.get('filter_rule') is None:
                                raise ValueError("过滤模式下必须提供filter_rule")
                            if kwargs['filter_rule'](classes, scores):
                                # 删除图片
                                original_path = kwargs.get('img_path')
                                if original_path is not None:
                                    try:
                                        os.remove(original_path)
                                    except Exception as e:
                                        print(f"删除文件失败 {original_path}: {str(e)}")
                        else:
                            self._crop_objects(frame, boxes, classes, save_name)
                    except Exception as e:
                        print(f"处理图像 {save_name} 时出错: {str(e)}")
            except:
                break
                
    def process_frame(self, frame: np.ndarray, save_name: str, **kwargs) -> None:
        """处理单帧图像"""
        if self.batch_size > 1:
            # 首次处理时启动线程
            if self.detector_thread is None:
                self.detector_thread = threading.Thread(
                    target=self._detector_thread_func,
                    kwargs=kwargs
                )
                self.processor_thread = threading.Thread(
                    target=self._processor_thread_func,
                    kwargs=kwargs
                )
                self.detector_thread.start()
                self.processor_thread.start()
                
            self.frame_queue.put((frame, save_name))
        else:
            self._process_batch([frame], [save_name], **kwargs)
            
    def process_image(self, img_path: str, **kwargs) -> None:
        """处理单张图片
        
        Args:
            img_path: 图片路径
            **kwargs: 其他参数
        """
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            return
            
        self.process_frame(img, save_name=os.path.basename(img_path), 
                         img_path=img_path, **kwargs)
        
    def process_video(self, video_path: str, frame_interval: int = 1, **kwargs) -> None:
        """处理视频文件
        
        Args:
            video_path: 视频路径
            frame_interval: 处理帧间隔
            **kwargs: 其他参数
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        video_name = os.path.basename(video_path)
        
        try:
            with tqdm(total=total_frames, desc="处理视频帧") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if frame_count % frame_interval == 0:
                        frame_name = f"{os.path.splitext(video_name)[0]}_frame_{frame_count}.jpg"
                        self.process_frame(frame, save_name=frame_name, **kwargs)
                        
                    frame_count += 1
                    pbar.update(1)
        finally:
            cap.release()
            if self.batch_size > 1:
                self.stop_event.set()
                if self.detector_thread:
                    self.detector_thread.join()
                if self.processor_thread:
                    self.processor_thread.join()
                # 重置线程，以便下次使用
                self.detector_thread = None
                self.processor_thread = None
                self.stop_event.clear()
                
    def process_folder(self, folder_path: str, frame_interval: int = 1, **kwargs) -> None:
        """处理文件夹中的所有媒体文件"""
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(
            ('.png', '.jpg', '.jpeg', '.mp4', '.avi', '.mov'))]
            
        try:
            for file_name in tqdm(files, desc="处理文件夹"):
                file_path = os.path.join(folder_path, file_name)
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.process_image(file_path, **kwargs)
                elif file_name.lower().endswith(('.mp4', '.avi', '.mov')):
                    self.process_video(file_path, frame_interval, **kwargs)
        finally:
            if self.batch_size > 1:
                self.stop_event.set()
                if self.detector_thread:
                    self.detector_thread.join()
                if self.processor_thread:
                    self.processor_thread.join()
                # 重置线程，以便下次使用
                self.detector_thread = None
                self.processor_thread = None
                self.stop_event.clear()
                
    def process_txt(self, txt_path: str, frame_interval: int = 1, **kwargs) -> None:
        """处理txt文件中列出的所有路径
        
        Args:
            txt_path: txt文件路径
            frame_interval: 视频处理的帧间隔
            **kwargs: 其他参数
        """
        if not os.path.exists(txt_path):
            print(f"找不到txt文件: {txt_path}")
            return
            
        with open(txt_path, 'r', encoding='utf-8') as f:
            paths = [line.strip() for line in f if line.strip()]
            
        try:
            for path in tqdm(paths, desc="处理文件列表"):
                if not os.path.exists(path):
                    print(f"无效路径: {path}")
                    continue
                    
                if os.path.isdir(path):
                    self.process_folder(path, frame_interval, **kwargs)
                elif path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.process_image(path, **kwargs)
                elif path.lower().endswith(('.mp4', '.avi', '.mov')):
                    self.process_video(path, frame_interval, **kwargs)
                else:
                    print(f"不支持的文件类型: {path}")
        finally:
            if self.batch_size > 1:
                self.stop_event.set()
                if self.detector_thread:
                    self.detector_thread.join()
                if self.processor_thread:
                    self.processor_thread.join()
                # 重置线程，以便下次使用
                self.detector_thread = None
                self.processor_thread = None
                self.stop_event.clear()
                
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'detector'):
            self.detector.stop_workers()
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
            if hasattr(self, 'detector_thread') and self.detector_thread:
                self.detector_thread.join()
            if hasattr(self, 'processor_thread') and self.processor_thread:
                self.processor_thread.join() 