#!/usr/bin/env python3
"""
脚本：从 JPEG 图像中提取压缩质量因子
用于生成类似 MSCOCO_train2017.json 的质量映射文件
"""

import os
import json
import argparse
from PIL import Image, JpegImagePlugin
from tqdm import tqdm


def get_jpeg_quality(image_path):
    """
    从 JPEG 文件中提取压缩质量因子

    原理：
    JPEG 压缩使用量化表（Quantization Tables），根据质量因子（1-100）决定
    PIL 的 JPEG 图像对象包含量化表信息，可以通过估算量化表与标准表的距离来推断质量因子

    注意：
    - 如果图像不是 JPEG 格式，返回 None
    - 如果图像被重新压缩过，只能获取最后压缩的质量
    """
    try:
        img = Image.open(image_path)

        # 如果不是 JPEG，返回默认值
        if img.format != 'JPEG':
            return None

        # 尝试从图像的量化表获取质量
        # JPEG 使用两个量化表（亮度和色度）
        if hasattr(img, 'quantization'):
            # 获取量化表
            qtables = img.quantization
            if qtables:
                # 简单估计：使用第一个量化表（亮度）
                qtable = qtables[0]

                # 标准量化表（质量=95）
                std_qtable = [
                    2, 2, 2, 2, 3, 3, 4, 4,
                    3, 3, 3, 3, 4, 4, 5, 5,
                    4, 4, 4, 4, 5, 5, 6, 6,
                    4, 4, 4, 5, 5, 6, 7, 7,
                    5, 5, 5, 6, 6, 7, 8, 8,
                    6, 6, 6, 7, 7, 8, 9, 9,
                    7, 7, 7, 8, 8, 9, 10, 10,
                    8, 8, 9, 9, 10, 10, 11, 11,
                ]

                # 估算质量因子
                # 简化方法：比较量化表的平均值与标准表的比值
                q_avg = sum(qtable) / len(qtable)
                std_avg = sum(std_qtable) / len(std_qtable)

                # 质量因子与量化值成反比
                estimated_quality = int(95 * std_avg / q_avg)
                estimated_quality = max(1, min(100, estimated_quality))

                return estimated_quality

        # 如果无法获取量化表，返回默认值
        return 95

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def generate_quality_json(image_dir, output_json, recursive=True, default_quality=95):
    """
    遍历图像目录，生成质量因子映射 JSON 文件

    Args:
        image_dir: 图像目录路径
        output_json: 输出的 JSON 文件路径
        recursive: 是否递归遍历子目录
        default_quality: 默认质量因子（用于非 JPEG 图像或无法获取质量的图像）
    """
    quality_mapping = {}

    # 支持的图像扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # 收集所有图像文件
    image_files = []
    if recursive:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(image_dir):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(os.path.join(image_dir, file))

    print(f"Found {len(image_files)} images")

    # 提取质量因子
    for image_path in tqdm(image_files, desc="Extracting quality factors"):
        basename = os.path.basename(image_path)

        quality = get_jpeg_quality(image_path)
        if quality is None:
            quality = default_quality

        quality_mapping[basename] = float(quality)

    # 保存 JSON 文件
    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(quality_mapping, f, indent=2)

    print(f"\nQuality mapping saved to: {output_json}")
    print(f"Total images: {len(quality_mapping)}")

    # 统计质量分布
    quality_stats = {}
    for quality in quality_mapping.values():
        quality_stats[quality] = quality_stats.get(quality, 0) + 1

    print("\nQuality distribution:")
    for quality in sorted(quality_stats.keys()):
        count = quality_stats[quality]
        percentage = count / len(quality_mapping) * 100
        print(f"  Quality {quality}: {count} images ({percentage:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract JPEG compression quality factors from images"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively traverse subdirectories"
    )
    parser.add_argument(
        "--default_quality",
        type=int,
        default=95,
        help="Default quality factor for non-JPEG images (default: 95)"
    )

    args = parser.parse_args()

    generate_quality_json(
        image_dir=args.image_dir,
        output_json=args.output_json,
        recursive=args.recursive,
        default_quality=args.default_quality
    )
