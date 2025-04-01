import os
import argparse
import yaml
import numpy as np
from typing import Dict, Any
from yolo_processor.core.detectors.yolov8_detector import YOLOv8Detector
from yolo_processor.core.detectors.yolov7_onnx_detector import YOLOv7ONNXDetector
from yolo_processor.core.detectors.yolov7_trt_detector import YOLOv7TRTDetector
from yolo_processor.core.processors.base_processor import BaseProcessor, ProcessMode

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict[str, Any]: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_detector(model_type: str, model_path: str):
    """根据模型类型创建检测器
    
    Args:
        model_type: 模型类型 ('yolov8' 或 'yolov7_onnx')
        model_path: 模型文件路径
        
    Returns:
        BaseDetector: 检测器实例
    """
    if model_type == "yolov8":
        return YOLOv8Detector(model_path)
    elif model_type == "yolov7_trt":
        return YOLOv7TRTDetector(model_path)
    elif model_type == "yolov7_onnx":
        return YOLOv7ONNXDetector(model_path)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def get_input_mode(input_path: str) -> str:
    """根据输入路径判断处理模式
    
    Args:
        input_path: 输入路径
        
    Returns:
        str: 'image', 'video', 'folder' 或 'txt'
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    
    if os.path.isdir(input_path):
        return 'folder'
    
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        return 'image'
    elif ext in ['.mp4', '.avi', '.mov']:
        return 'video'
    elif ext == '.txt':
        return 'txt'
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

def example_filter_rule(classes, scores):
    """示例过滤规则"""
    # 规则1: 同一类别不能出现超过2次
    unique_cls, counts = np.unique(classes, return_counts=True)
    if any(count > 2 for count in counts):
        return False
    
    # 规则2: 置信度必须大于0.5
    if any(score < 0.5 for score in scores):
        return False
    
    return True

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='目标检测和裁剪/过滤工具')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='配置文件路径')
    parser.add_argument('--input', type=str, required=False, default= 'data/midu-dushu-yichang.mp4',
                      help='输入路径(图片/视频文件、文件夹或txt文件)')
    parser.add_argument('--process-mode', type=str, choices=['crop', 'filter'],
                      default='crop',
                      help='处理模式：crop(裁切)或filter(过滤)，覆盖配置文件设置')
    parser.add_argument('--classes', type=int, nargs='+',
                      help='指定要检测的类别ID列表')
    parser.add_argument('--batch-size', type=int,
                      help='批处理大小，默认从配置文件读取')
    parser.add_argument('--num-workers', type=int,
                      help='数据加载线程数，默认从配置文件读取')
    return parser.parse_args()

def update_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """使用命令行参数更新配置"""
    if args.process_mode is not None:
        config['processing']['mode'] = args.process_mode
    if args.classes is not None:
        config['processing']['target_classes'] = args.classes
    if args.batch_size is not None:
        config['processing']['batch_size'] = args.batch_size
    if args.num_workers is not None:
        config['processing']['num_workers'] = args.num_workers
    return config

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置并用命令行参数更新
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    
    # 创建检测器
    detector = get_detector(
        config['model']['type'],
        config['model']['path']
    )
    
    # 创建处理器
    processor = BaseProcessor(
        detector,
        output_folder=config['processing']['output_folder'],
        batch_size=config['processing']['batch_size'],
        num_workers=config['processing']['num_workers']
    )
    
    # 获取输入类型
    input_mode = get_input_mode(args.input)
    
    # 准备处理参数
    process_params = {
        'target_classes': config['processing']['target_classes'],
        'mode': ProcessMode.FILTER if config['processing']['mode'] == 'filter' else ProcessMode.CROP,
        'filter_rule': example_filter_rule if config['processing']['mode'] == 'filter' else None,
        'is_gray': config['processing']['is_gray'],
        'frame_interval': config['processing']['frame_interval']
    }
    
    # 根据输入类型处理
    if input_mode == 'image':
        processor.process_image(args.input, **process_params)
    elif input_mode == 'video':
        processor.process_video(args.input, **process_params)
    elif input_mode == 'folder':
        processor.process_folder(args.input, **process_params)
    elif input_mode == 'txt':
        processor.process_txt(args.input, **process_params)
    
    print(f"处理完成: {args.input}")

if __name__ == '__main__':
    main()
