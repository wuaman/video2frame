# YOLO处理器

这是一个基于YOLO系列模型的目标检测和图像处理工具，支持多种YOLO模型（YOLOv8、YOLOv7 ONNX和YOLOv7 TensorRT），提供批量处理、多线程加速和灵活的处理模式。

## 功能特点

- **多模型支持**：
  - YOLOv8（使用Ultralytics库）
  - YOLOv7 ONNX（使用ONNXRuntime）
  - YOLOv7 TensorRT（使用TensorRT加速）

- **处理模式**：
  - 裁剪模式：自动裁剪并保存检测到的目标
  - 过滤模式：根据自定义规则过滤图像

- **批量处理**：
  - 支持批量处理图像和视频
  - 多线程加速
  - 支持处理文件夹和文本文件列表

- **性能优化**：
  - GPU加速
  - 批处理优化
  - 线程池并行处理

## 安装

### 环境要求

- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+（推荐用于GPU加速）

### 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 基本用法

```bash
python main.py --config configs/default.yaml --input data/example.mp4
```

### 命令行参数

- `--config`: 配置文件路径
- `--input`: 输入路径（图片、视频、文件夹或文本文件）
- `--process-mode`: 处理模式（crop或filter）
- `--classes`: 指定要检测的类别ID
- `--batch-size`: 批处理大小
- `--num-workers`: 数据加载线程数

### 配置文件示例

```yaml
model:
  type: "yolov8"  # yolov8、yolov7_onnx或yolov7_trt
  path: "./models/model.pt"
  
processing:
  output_folder: "./output"
  frame_interval: 2
  is_gray: false
  target_classes: [0, 1, 2]
  mode: "crop"
  batch_size: 32
  num_workers: 16
  
logging:
  level: "INFO"
  file: "processor.log"
```

## 处理模式

### 裁剪模式

裁剪模式会自动裁剪并保存检测到的目标：

```bash
python main.py --config configs/default.yaml --input data/images --process-mode crop
```

### 过滤模式

过滤模式根据自定义规则过滤图像：

```bash
python main.py --config configs/default.yaml --input data/images --process-mode filter
```

## 自定义过滤规则

可以在代码中自定义过滤规则，例如：

```python
def custom_filter_rule(classes, scores):
    # 规则1: 同一类别不能出现超过2次
    unique_cls, counts = np.unique(classes, return_counts=True)
    if any(count > 2 for count in counts):
        return False
    
    # 规则2: 置信度必须大于0.5
    if any(score < 0.5 for score in scores):
        return False
    
    return True
```

## 项目结构

```
yolo_processor/
├── core/
│   ├── detectors/
│   │   ├── base_detector.py
│   │   ├── yolov8_detector.py
│   │   ├── yolov7_onnx_detector.py
│   │   └── yolov7_trt_detector.py
│   └── processors/
│       └── base_processor.py
├── configs/
│   └── default.yaml
├── main.py
└── requirements.txt
```

## 高级用法

### 处理视频

```bash
python main.py --config configs/default.yaml --input data/video.mp4 --frame-interval 5
```

### 处理文件夹

```bash
python main.py --config configs/default.yaml --input data/images/
```

### 处理文本文件列表

```bash
python main.py --config configs/default.yaml --input data/file_list.txt
```

文本文件格式示例：
```
/path/to/image1.jpg
/path/to/image2.jpg
/path/to/video1.mp4
/path/to/folder
```

## 性能优化

- 增加批处理大小可以提高GPU利用率
- 调整工作线程数可以优化CPU利用率
- 使用TensorRT模型可以获得最佳推理性能

## 许可证

MIT

## 贡献

欢迎提交问题和拉取请求！
