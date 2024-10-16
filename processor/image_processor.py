import base64
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image
from torchvision import transforms as T
import cv2


class ImageProcessor:
    def __init__(self, max_width=None, max_height=None):
        """
        初始化ImageProcessor类。
        """
        self.max_width = max_width
        self.max_height = max_height

    def resize_image(self, image):
        if self.max_width is None and self.max_height is None:
            return image, 1
        # 获取原始图像的宽度和高度
        original_width, original_height = image.size

        # 计算宽度和高度的缩放比例
        width_ratio = self.max_width / original_width
        height_ratio = self.max_height / original_height

        # 使用较小的比例进行缩放，保证宽度和高度都不超过最大值
        scale_ratio = min(width_ratio, height_ratio)

        # 计算新的宽度和高度
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)

        # 使用LANCZOS进行高质量缩放
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return resized_image, scale_ratio

    @staticmethod
    def is_url(path: str):
        """判断路径是否为 URL"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    @staticmethod
    def get_webp_base64(image):
        with BytesIO() as buffered:
            image.save(buffered, format="WEBP")
            trans_image_webp = base64.b64encode(buffered.getvalue()).decode()
        return trans_image_webp

    @staticmethod
    def get_png_base64(image):
        with BytesIO() as buffered:
            image.save(buffered, format="png")
            trans_image_png = base64.b64encode(buffered.getvalue()).decode()
        return trans_image_png

    def load_image(self, image_file: Path, image_type="RGB"):
        """
        加载并预处理图片
        """
        if self.is_url(str(image_file)):
            # 将图片下载到本地, 后进行处理
            src_image = Image.open(requests.get(image_file, stream=True).raw).convert(image_type)
        else:
            src_image = Image.open(image_file)
        trans_image, scale_ratio = self.resize_image(src_image)
        return src_image, trans_image, scale_ratio

    def process_mask(self, src_mask, target_mask, scale_ratio):
        def read_mask(mask):
            if isinstance(mask, (str, Path)):
                return cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
            return mask

        # 读取mask图像
        src_mask_img = read_mask(src_mask)
        target_mask_img = read_mask(target_mask)

        # 确保图像大小相同
        if src_mask_img.shape != target_mask_img.shape:
            raise ValueError("Source and target masks must have the same dimensions.")

        # 合并target_mask和target_mask_img, 要求相同的像素保留, 不同的像素取大值
        combined_mask = np.maximum(src_mask_img, target_mask_img)

        # 如果需要缩放比例，可以在此处应用缩放
        if scale_ratio != 1.0:
            new_size = (int(combined_mask.shape[1] * scale_ratio), int(combined_mask.shape[0] * scale_ratio))
            combined_mask = cv2.resize(combined_mask, new_size, interpolation=cv2.INTER_NEAREST)

        # 转为PIL图像
        combined_mask = Image.fromarray(combined_mask)
        return combined_mask

    @staticmethod
    def combine_images(src_img, mask_img):
        # mask_img是灰度图像, src_img是RGB图像
        # mask_img中为白色区域的保留src_img内容,黑色区域的去除src_img内容
        src_img = src_img.convert("RGBA")
        mask_img = mask_img.convert("L")

        # 创建一个新的图像，白色区域的alpha值为255，黑色区域的alpha值为0
        mask_rgba = Image.new("L", mask_img.size)
        mask_rgba.putdata([255 if pixel == 255 else 0 for pixel in mask_img.getdata()])

        # 使用mask_rgba作为掩码合成src_img和透明背景
        combined_image = Image.composite(src_img, Image.new("RGBA", src_img.size, (0, 0, 0, 0)), mask_rgba)

        return combined_image


if __name__ == "__main__":
    # 示例使用
    processor = ImageProcessor()
    src_img, trans_img, base64_img = processor.load_image(Path("./tests/data/000000000009.jpg"))

    scaled_x, scaled_y = 12, 12  # 缩放后坐标点
    original_x, original_y = processor.get_original_coordinates(scaled_x, scaled_y, src_img.size, trans_img.size)
    print(f"Original coordinates: ({original_x}, {original_y})")
