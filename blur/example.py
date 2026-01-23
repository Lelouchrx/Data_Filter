#!/usr/bin/env python3
import sys
import pathlib
from process import process_media, load_config

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_path>")
        sys.exit(1)

    config = load_config()

    inputs = sys.argv[1:]

    results = process_media(
        inputs,
        threshold=config['blur_thresh'],
        fix_size=config['fix_size'],
        output_dir=config['output_dir'],
        video_blur_ratio=config['blur_ratio'],
        sample_rate=config['sample'],
        motion_method=config['motion'],
        verbose=config['verbose'],
        display=config['display'],
        record=False,
        enable_blur_test=config['blur_test'],
        enable_jitter_test=config['jitter_test'],
        enable_exposure_test=config['exposure_test'],
        show_jitter_plot=False
    )

    print(f"处理完成，共处理 {len(results)} 个文件")

if __name__ == '__main__':
    main()
# To run this script, use the command:
# python blur/example.py <input_path1> <input_path2> ...