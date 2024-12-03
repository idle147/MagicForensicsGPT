import base64
from io import BytesIO
from pathlib import Path
from tkinter import Image
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import numpy as np
import traceback
from processor.image_processor import ImageProcessor
from config import load_config
from prompts import FindDiffDescription
from PIL import Image, ImageDraw
import numpy as np
from langchain.globals import set_debug

# 启用调试模式
set_debug(True)


class GeiDiff:
    def __init__(self):
        self.image_processor = ImageProcessor()

        self.llm = ChatOpenAI(**load_config(), timeout=300)
        self.diff_prompt = FindDiffDescription(self.llm)

    def run(self, src_image_path: Path, target_image_path: Path, src_mask, ref_mask):
        src_image, _, scale_ratio = self.image_processor.load_image(src_image_path)
        target_image, _, _ = self.image_processor.load_image(target_image_path)
        mask = self.image_processor.process_mask(src_mask, ref_mask, scale_ratio)

        image_info = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{self.image_processor.get_webp_base64(src_image)}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{self.image_processor.get_webp_base64(target_image)}"},
                },
            ]
        )

        diff_info = self.diff_prompt.run(image_info)
        self.add_transparent_mask(target_image, diff_info.extract_polygons())
        print(diff_info)
        print(src_image_path)
        print(target_image_path)

    def add_transparent_mask(self, target_image: Image, points, output_image_path: str = "./tmp.png"):
        """
        将一系列点围成一个形状作为mask，以75%透明度填充到目标图片中，并保存结果。

        :param target_image: Image, 输入图片对象
        :param points: list of list, 多边形的顶点坐标
        :param output_image_path: str, 输出图片路径
        """
        # 打开目标图片并转换为RGBA模式
        image_size = target_image.size

        # Ensure target image is in RGBA mode
        if target_image.mode != "RGBA":
            target_image = target_image.convert("RGBA")

        # 创建一个空白的RGBA图像用于叠加mask
        combined = target_image.copy()

        for polygon in points:
            # points = [(float(x), float(y)) for x, y in polygon]
            # 创建一个黑色的图像作为mask
            mask = Image.new("L", image_size, 0)
            # 使用ImageDraw在mask上绘制多边形
            ImageDraw.Draw(mask).polygon(polygon, outline=1, fill=1)

            # 将mask转换为numpy数组，并设置alpha通道为192（75%透明度）
            mask_np = np.array(mask) * 192

            # 创建一个RGBA的mask
            mask_rgba = np.zeros((image_size[1], image_size[0], 4), dtype=np.uint8)
            mask_rgba[..., 1] = mask_np * 255  # Set the green channel
            mask_rgba[..., 3] = mask_np  # 只设置alpha通道

            # 将mask转换为PIL图像
            mask_image = Image.fromarray(mask_rgba, mode="RGBA")

            # 将mask应用到目标图像
            combined = Image.alpha_composite(combined, mask_image)

        # 保存结果
        combined.save(output_image_path)


if __name__ == "__main__":
    target_dir = Path("/home/yuyangxin/data/experiment/examples/moving object/images")
    src_img_dir = Path("/home/yuyangxin/data/experiment/examples/images")
    mask_img_dir = Path("/home/yuyangxin/data/experiment/examples/masks")
    ref_mask_img_dir = Path("/home/yuyangxin/data/experiment/examples/moving object/masks")
    count = 2
    for target_image_path in target_dir.glob("*.png"):
        if count != 0:
            count -= 1
            continue

        file_name = target_image_path.stem.split("_")[-1]
        src_image_path = src_img_dir / f"{file_name}.jpg"
        mask_img_path = mask_img_dir / f"mask_{file_name}.jpg"
        ref_mask_img_path = ref_mask_img_dir / f"mask_{file_name}.png"

        try:
            gei_diff = GeiDiff()
            gei_diff.run(src_image_path, target_image_path, mask_img_path, ref_mask_img_path)
        except Exception as e:
            print(traceback.format_exc())
        break
    print("Done!")
