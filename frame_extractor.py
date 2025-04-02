import os
import cv2
import multiprocessing as mp
from typing import List, Optional

def extract_frames_from_video(video_path: str, output_folder: str, frame_interval: int, num_workers: int = 1):
    """从视频中抽取帧
    
    Args:
        video_path: 视频文件路径
        output_folder: 输出文件夹
        frame_interval: 帧间隔
        num_workers: 处理线程数
    """
    # 创建输出目录
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(output_folder, video_name)
    os.makedirs(frames_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"处理视频: {video_path}")
    print(f"总帧数: {total_frames}, FPS: {fps}, 抽帧间隔: {frame_interval}")
    
    # 抽取帧
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    print(f"视频 {video_path} 处理完成，共保存 {saved_count} 帧到 {frames_dir}")
    return saved_count

def process_videos_from_txt(txt_path: str, output_folder: str, frame_interval: int, num_workers: int = 1):
    """从txt文件中读取视频路径并处理
    
    Args:
        txt_path: txt文件路径
        output_folder: 输出文件夹
        frame_interval: 帧间隔
        num_workers: 处理线程数
    """
    # 读取视频路径列表
    with open(txt_path, 'r', encoding='utf-8') as f:
        video_paths = [line.strip() for line in f if line.strip()]
    
    print(f"从 {txt_path} 中读取到 {len(video_paths)} 个视频路径")
    
    return process_videos(video_paths, output_folder, frame_interval, num_workers)

def process_videos(video_paths: List[str], output_folder: str, frame_interval: int, num_workers: int = 1):
    """处理多个视频文件
    
    Args:
        video_paths: 视频文件路径列表
        output_folder: 输出文件夹
        frame_interval: 帧间隔
        num_workers: 处理线程数
        
    Returns:
        int: 处理的视频数量
    """
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    processed_count = 0
    
    # 单线程处理
    if num_workers <= 1:
        for video_path in video_paths:
            if os.path.exists(video_path):
                extract_frames_from_video(video_path, output_folder, frame_interval)
                processed_count += 1
            else:
                print(f"视频不存在: {video_path}")
    # 多线程处理
    else:
        pool = mp.Pool(processes=num_workers)
        tasks = []
        
        for video_path in video_paths:
            if os.path.exists(video_path):
                tasks.append(pool.apply_async(extract_frames_from_video, 
                                             (video_path, output_folder, frame_interval)))
                processed_count += 1
            else:
                print(f"视频不存在: {video_path}")
        
        # 等待所有任务完成
        for task in tasks:
            task.get()
        
        pool.close()
        pool.join()
    
    return processed_count

def get_video_files_from_directory(directory: str) -> List[str]:
    """获取目录下所有视频文件的路径
    
    Args:
        directory: 目录路径
        
    Returns:
        List[str]: 视频文件路径列表
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in video_extensions:
                video_files.append(os.path.join(root, file))
    
    return video_files

def main(input_path: str, output_folder: str, frame_interval: int, num_workers: int = 1):
    """抽帧主函数
    
    Args:
        input_path: 输入路径(视频文件、txt文件或目录)
        output_folder: 输出文件夹
        frame_interval: 帧间隔
        num_workers: 处理线程数
        
    Returns:
        bool: 是否成功处理
    """
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 判断输入类型
    if not os.path.exists(input_path):
        print(f"输入路径不存在: {input_path}")
        return False
    
    # 处理单个视频文件
    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        
        # 视频文件
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
            extract_frames_from_video(input_path, output_folder, frame_interval)
            return True
        # txt文件(包含视频路径列表)
        elif ext == '.txt':
            process_videos_from_txt(input_path, output_folder, frame_interval, num_workers)
            return True
        else:
            print(f"不支持的文件类型: {ext}")
            return False
    # 处理目录
    elif os.path.isdir(input_path):
        video_files = get_video_files_from_directory(input_path)
        if not video_files:
            print(f"目录中没有找到视频文件: {input_path}")
            return False
        
        print(f"在目录 {input_path} 中找到 {len(video_files)} 个视频文件")
        process_videos(video_files, output_folder, frame_interval, num_workers)
        return True
    else:
        print(f"不支持的输入类型: {input_path}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='视频抽帧工具')
    parser.add_argument('--input', type=str, required=True,
                      help='输入路径(视频文件或txt文件)')
    parser.add_argument('--output', type=str, required=True,
                      help='输出文件夹')
    parser.add_argument('--interval', type=int, default=10,
                      help='抽帧间隔')
    parser.add_argument('--workers', type=int, default=1,
                      help='处理线程数')
    
    args = parser.parse_args()
    
    main(args.input, args.output, args.interval, args.workers) 