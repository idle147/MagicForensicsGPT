import base64
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image
from torchvision import transforms as T


class ImageProcessor:
    def __init__(self, target_size=(256, 256)):
        """
        初始化ImageProcessor类。
        """
        self.target_size = target_size
        self.transform = T.Compose(
            [
                T.Lambda(lambda img: self._resize_with_aspect_ratio(img.copy())),
            ]
        )

    def _resize_with_aspect_ratio(self, img: Image.Image):
        """
        根据目标尺寸等比缩放图片，不改变长宽比。

        参数:
        img (PIL.Image): 输入图片。

        返回:
        PIL.Image: 缩放后的图片。
        """
        img.thumbnail(self.target_size)
        return img

    @staticmethod
    def get_original_coordinates(position, original_size, scaled_size):
        """
        获取缩放前的坐标点。

        参数:
        position (list): 缩放后的坐标点列表，长度应为偶数。
        original_size (tuple): 原始图片尺寸 (宽, 高)。
        scaled_size (tuple): 缩放后图片尺寸 (宽, 高)。

        返回:
        list: 缩放前的坐标点列表。
        """
        if len(position) % 2 != 0:
            raise ValueError("Position must have an even number of elements")

        original_width, original_height = original_size
        scaled_width, scaled_height = scaled_size

        width_ratio = scaled_width / original_width
        height_ratio = scaled_height / original_height

        return [coord / ratio for coord, ratio in zip(position, [width_ratio, height_ratio] * (len(position) // 2))]

    @staticmethod
    def get_scaled_coordinates(position, original_size, scaled_size):
        """
        获取缩放后的坐标点。

        参数:
        position (list): 原始坐标点列表，长度应为偶数。
        original_size (tuple): 原始图片尺寸 (宽, 高)。
        scaled_size (tuple): 缩放后图片尺寸 (宽, 高)。

        返回:
        list: 缩放后的坐标点列表。
        """
        if len(position) % 2 != 0:
            raise ValueError("Position must have an even number of elements")

        original_width, original_height = original_size
        scaled_width, scaled_height = scaled_size

        width_ratio = scaled_width / original_width
        height_ratio = scaled_height / original_height

        return [coord * ratio for coord, ratio in zip(position, [width_ratio, height_ratio] * (len(position) // 2))]

    @staticmethod
    def resize_back_to_original(scaled_img, original_size):
        """
        将缩放后的图片缩放回原来的大小。

        参数:
        scaled_img (PIL.Image): 缩放后的图片。
        original_size (tuple): 原始图片尺寸 (宽, 高)。

        返回:
        PIL.Image: 缩放回原来大小的图片。
        """
        return scaled_img.resize(original_size, Image.BICUBIC)

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
        buffered = BytesIO()
        image.save(buffered, format="WEBP")
        trans_image_webp = base64.b64encode(buffered.getvalue()).decode()
        buffered.close()
        return trans_image_webp

    def load_image(self, image_file: Path, image_type="RGB"):
        """
        加载并预处理图片
        """
        if self.is_url(str(image_file)):
            # 将图片下载到本地, 后进行处理
            src_image = Image.open(requests.get(image_file, stream=True).raw).convert(image_type)
        else:
            src_image = Image.open(image_file).convert(image_type)
        trans_image = self.transform(src_image)
        return src_image, trans_image

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
