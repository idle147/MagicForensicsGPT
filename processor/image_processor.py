import base64
from io import BytesIO
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from IPython.display import HTML, display


class ImageProcessor:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, input_size=128, max_num=12, min_num=1):
        self.input_size = input_size
        self.max_num = max_num
        self.min_num = min_num
        self.transform = self._build_transform()
        self.target_ratios = self._generate_target_ratios()  # 预先计算目标宽高比

    def _build_transform(self):
        """创建图片预处理管道"""
        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(self.input_size, interpolation=InterpolationMode.BICUBIC),
            ]
        )

    def _generate_target_ratios(self):
        """生成目标宽高比集合"""
        return sorted(
            {
                (i, j)
                for n in range(self.min_num, self.max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if self.min_num <= i * j <= self.max_num
            },
            key=lambda x: x[0] * x[1],
        )

    def find_closest_aspect_ratio(self, aspect_ratio, width, height):
        """
        找到与给定目标相近的宽高比
        """
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        half_area_threshold = 0.5 * self.input_size * self.input_size

        for ratio in self.target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)

            # 更新最佳比率
            if ratio_diff < best_ratio_diff or (ratio_diff == best_ratio_diff and area > half_area_threshold * ratio[0] * ratio[1]):
                best_ratio_diff = ratio_diff
                best_ratio = ratio

        return best_ratio

    def dynamic_preprocess(self, image, use_thumbnail=False):
        """
        动态处理图片，分割成多个块
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # 找到最接近的宽高比
        target_aspect_ratio = self.find_closest_aspect_ratio(aspect_ratio, orig_width, orig_height)

        # 计算目标宽和高
        target_width = self.input_size * target_aspect_ratio[0]
        target_height = self.input_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # 调整图片大小
        resized_img = image.resize((target_width, target_height))
        processed_images = []

        # 计算每个块的起始坐标，减少重复计算
        for i in range(blocks):
            row, col = divmod(i, target_width // self.input_size)
            box = (
                col * self.input_size,
                row * self.input_size,
                (col + 1) * self.input_size,
                (row + 1) * self.input_size,
            )
            # 分割图片
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((self.input_size, self.input_size))
            processed_images.append(thumbnail_img)

        return processed_images

    @staticmethod
    def is_url(path: str):
        """判断路径是否为 URL"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def load_image(self, image_file: Path):
        """
        加载并预处理图片
        """
        if self.is_url(str(image_file)):
            # 将图片下载到本地, 后进行处理
            image = Image.open(requests.get(image_file, stream=True).raw).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")

        image = self.transform(image)

        # 获取图像的 base64 字符串
        buffered = BytesIO()
        image.save(buffered, format="WEBP")
        return base64.b64encode(buffered.getvalue()).decode()
