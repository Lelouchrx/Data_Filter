# 批量视频模糊检测测试脚本

这个脚本用于批量测试 `process.py` 对视频文件夹的处理能力。

## 功能特点

- **批量处理**: 自动扫描文件夹中的所有视频文件
- **并行处理**: 支持多线程并行处理，提高效率
- **详细日志**: 记录处理过程和结果
- **结果统计**: 生成JSON格式的处理结果报告
- **错误处理**: 完善的异常处理和错误记录

## 使用方法

### 基本用法

```bash
python3 batch_test.py -d "/media/cwr/新加卷/Detection_Data/vedio/original_data/test"
```

### 完整参数

```bash
python3 batch_test.py \
    -d "/media/cwr/新加卷/Detection_Data/vedio/original_data/test" \
    -o "batch_output" \
    -w 2 \
    -l "batch_test.log" \
    -r "results.json" \
    --sample-rate 5.0 \
    -t 50.0 \
    --motion-method farneback \
    --motion-skip-frames 5
```

### 参数说明

- `-d, --test-dir`: 测试视频文件夹路径（必需）
- `-o, --output-dir`: 输出目录，用于保存处理后的帧图像（可选）
- `-w, --max-workers`: 最大并行处理数（默认: 2）
- `-l, --log-file`: 日志文件路径（可选，默认自动生成）
- `-r, --result-file`: 结果文件路径（可选，默认自动生成）
- `--sample-rate`: 采样率（默认: 5.0）
- `-t, --threshold`: 模糊阈值（默认: 50.0）
- `--motion-method`: 运动检测方法，可选 'farneback' 或 'lk'（默认: farneback）
- `--motion-skip-frames`: 运动检测跳帧数（默认: 5）

## 输出文件

脚本会生成以下文件：

1. **日志文件**: 记录详细的处理过程
2. **结果文件**: JSON格式的处理结果和统计信息

### 结果文件格式

```json
{
  "timestamp": "2024-01-13T02:30:00",
  "total_videos": 13,
  "successful": 13,
  "failed": 0,
  "success_rate": 100.0,
  "results": [
    {
      "video_path": "/path/to/video.mp4",
      "success": true,
      "processing_time": 45.67
    }
  ]
}
```

## 示例

### 测试指定的视频文件夹

```bash
cd /media/cwr/新加卷/AI/blur
python3 batch_test.py -d "/media/cwr/新加卷/Detection_Data/vedio/original_data/test" --sample-rate 5.0 -t 50.0 --motion-method farneback --motion-skip-frames 5
```

### 使用更多并行处理

```bash
python3 batch_test.py -d "/media/cwr/新加卷/Detection_Data/vedio/original_data/test" -w 4 -o "test_output"
```

## 注意事项

- 确保测试文件夹路径正确且包含视频文件
- 根据系统性能调整 `--max-workers` 参数
- 处理大文件夹时建议设置 `--output-dir` 以保存中间结果
- 脚本会自动检测视频文件（支持 .mp4, .avi, .mov, .mkv 等格式）

