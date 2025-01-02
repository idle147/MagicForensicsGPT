import json
import logging
import random
import shutil
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

from facenet_pytorch import MTCNN


class FaceDetector:
    """使用 MTCNN 进行人脸检测的类。"""

    def __init__(self, device: torch.device = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

    def detect_faces(self, image: Union[str, Path, np.ndarray]) -> bool:
        """
        检测输入图像中是否存在人脸。

        Args:
            image: 图像的路径或 numpy 数组。

        Returns:
            bool: 是否检测到人脸。
        """
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                logging.warning(f"无法读取图像：{image}")
                return False
        else:
            # 拷贝图像以避免修改原始数据
            image = image.copy()

        # 确保图像为 RGB 格式
        if image.ndim == 2:
            # 灰度图像，转换为 RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA 转 RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            # BGR 转 RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转换为 PIL 图像
        pil_image = Image.fromarray(image)

        # 检测人脸
        boxes, _ = self.mtcnn.detect(pil_image)

        return boxes is not None and len(boxes) > 0


class COCODataProcessor:
    """处理 COCO 数据集的类。"""

    def __init__(
        self,
        data_dir: Union[str, Path],
        data_type: str,
        save_path: Union[str, Path],
        max_images: int = 500,
        min_size: int = 64,
        max_size: int = 128,
        clear_existing: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.data_type = data_type
        self.save_path = Path(save_path)
        self.max_images = max_images
        self.min_size = min_size
        self.max_size = max_size
        self.count = 0
        self.output_dict = {}

        # 初始化 COCO API
        ann_file = self.data_dir / "annotations" / f"instances_{self.data_type}.json"
        caption_file = self.data_dir / "annotations" / f"captions_{self.data_type}.json"
        self.coco = COCO(ann_file)
        self.coco_caps = COCO(caption_file)

        # 获取所有图像 ID 并打乱顺序
        self.img_ids = self.coco.getImgIds()
        random.shuffle(self.img_ids)

        # 初始化人脸检测器
        self.face_detector = FaceDetector()

        # 初始化保存路径
        self.target_mask_dir = self.save_path / "masks"
        self.target_img_dir = self.save_path / "images"

        # 准备目录
        self._prepare_directories(clear_existing)

    def _prepare_directories(self, clear_existing: bool):
        """准备保存目录。

        如果 `clear_existing` 为 True，若存在则删除并新建；否则仅创建不存在的目录。
        """
        if clear_existing and self.save_path.exists():
            shutil.rmtree(self.save_path)
        self.target_mask_dir.mkdir(parents=True, exist_ok=True)
        self.target_img_dir.mkdir(parents=True, exist_ok=True)

    def process(self):
        """处理图像并保存结果。"""
        for img_id in self.img_ids:
            if self.count >= self.max_images:
                break
            self._process_image(img_id)

        # 保存结果为 JSON 文件
        output_file = self.save_path / "result.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(self.output_dict, f, indent=4, ensure_ascii=False)

    def _process_image(self, img_id: int):
        """
        处理单个图像。

        Args:
            img_id: 图像的 ID。
        """
        try:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            if not anns:
                logging.warning(f"图像 {img_id} 无标注，跳过")
                return

            # 打乱标注顺序
            random.shuffle(anns)

            # 加载图像信息
            img_info = self.coco.loadImgs(img_id)[0]
            image_path = self.data_dir / self.data_type / img_info["file_name"]

            # 检测人脸，若存在则跳过
            if self.face_detector.detect_faces(image_path):
                logging.info(f"{image_path}: 发现人脸，跳过")
                return

            # 获取图像的标题
            cap_ids = self.coco_caps.getAnnIds(imgIds=img_id)
            caps = self.coco_caps.loadAnns(cap_ids)
            captions = [cap["caption"] for cap in caps] if caps else []

            for ann in anns:
                bbox = ann.get("bbox")
                segmentation = ann.get("segmentation")

                if not bbox or not segmentation:
                    continue

                x, y, width, height = bbox

                if not (self.min_size <= min(width, height) <= self.max_size):
                    continue

                mask = self._get_mask(ann, img_info)
                if mask is None:
                    continue

                # 保存掩码图像
                mask_image_path = self.target_mask_dir / f"mask_{img_info['file_name']}"
                mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
                mask_image.save(mask_image_path)

                # 复制真实图像
                target_img_path = self.target_img_dir / img_info["file_name"]
                if not target_img_path.exists():
                    shutil.copy(image_path, target_img_path)

                # 获取类别信息
                category_id = ann["category_id"]
                category_info = self.coco.loadCats([category_id])
                category_name = category_info[0]["name"] if category_info else "unknown"

                # 添加到输出字典
                self.output_dict[str(image_path)] = {
                    "mask_path": str(mask_image_path),
                    "category": category_name,
                    "ann": ann,
                    "captions": captions,
                }

                self.count += 1
                break  # 已处理一个合适的标注，跳出循环

        except Exception as e:
            logging.exception(f"处理图像 {img_id} 时出错: {e}")

    def _get_mask(self, ann: dict, img_info: dict) -> Union[np.ndarray, None]:
        """
        从标注中获取掩码。

        Args:
            ann: 标注信息。
            img_info: 图像信息。

        Returns:
            np.ndarray: 掩码数组，若失败则返回 None。
        """
        try:
            rle = maskUtils.frPyObjects(ann["segmentation"], img_info["height"], img_info["width"])
            mask = maskUtils.decode(rle)

            # 确保掩码是二维的
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            return mask
        except Exception as e:
            logging.exception(f"获取掩码时出错: {e}")
            return None


if __name__ == "__main__":
    data_dir = "/home/yuyangxin/data/dataset/MSCOCO"
    data_type = "train2017"
    save_path = "/home/yuyangxin/data/experiment/examples"

    processor = COCODataProcessor(data_dir, data_type, save_path)  # 设置为 True 以清除已有数据
    processor.process()
