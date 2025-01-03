import json
import logging
import random
import shutil
from pathlib import Path
from typing import Union

import concurrent.futures
import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from collections import Counter

import cv2
from retinaface.pre_trained_models import get_model
from tqdm import tqdm


class FaceDetector:
    """使用 RetinaFace 进行人脸检测的类。"""

    def __init__(self, device: torch.device = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model("resnet50_2020-07-20", max_size=2048, device=self.device)
        self.model.eval()

    def detect_faces(self, image_path: str) -> bool:
        """
        检测输入图像中是否存在人脸。

        Args:
            image_path (str): 图像文件的路径。

        Returns:
            bool: 如果未检测到人脸，则返回 True，反之返回 False。
        """
        # 读取图片文件并转换为 numpy 数组
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像文件: {image_path}")

        faces = self.model.predict_jsons(img)
        detected_faces = 0
        for face in faces:
            if face["score"] > 0.5:
                detected_faces += 1
        return detected_faces > 0


class COCODataProcessor:
    """处理 COCO 数据集的类。"""

    def __init__(
        self,
        data_dir: Union[str, Path],
        data_type: str,
        save_path: Union[str, Path],
        max_img_count: int = -1,  # -1表示无限制
        clear_existing: bool = False,
        small_threshold: float = 0.25,
        medium_threshold: float = 0.50,
    ):
        self.data_dir = Path(data_dir)
        self.data_type = data_type
        self.save_path = Path(save_path)
        self.max_img_count = max_img_count
        self.count = 0

        # 图片阈值输出结果
        self.output_dict = {"small": {}, "medium": {}, "large": {}}
        self.small_threshold = small_threshold
        self.medium_threshold = medium_threshold

        ann_file = self.data_dir / "annotations" / f"instances_{self.data_type}.json"
        caption_file = self.data_dir / "annotations" / f"captions_{self.data_type}.json"
        self.coco = COCO(ann_file)
        self.coco_caps = COCO(caption_file)

        self.img_ids = self.coco.getImgIds()
        random.shuffle(self.img_ids)

        self.face_detector = FaceDetector()

        self.target_mask_dir = {
            "small": self.save_path / "gt_masks" / "small",
            "medium": self.save_path / "gt_masks" / "medium",
            "large": self.save_path / "gt_masks" / "large",
        }
        self.target_img_dir = self.save_path / "0_real"

        self._prepare_directories(clear_existing)

    def _prepare_directories(self, clear_existing: bool):
        if clear_existing and self.save_path.exists():
            # 添加二次确认提示
            user_input = input(f"目标数据集文件夹 '{self.save_path}' 已存在，是否继续删除？(y/n): ")
            if user_input.lower() != "y":
                print("已取消删除操作。")
            else:
                print("发现已经存在目标数据集文件, 正在执行删除")
                shutil.rmtree(self.save_path)

        for dir in self.target_mask_dir.values():
            dir.mkdir(parents=True, exist_ok=True)
        self.target_img_dir.mkdir(parents=True, exist_ok=True)

    def process(self):
        # 设置进度条，进度条的总长度使用 self.max_img_count
        with tqdm(total=self.max_img_count, desc="Processing images", ncols=100) as pbar:
            # 使用ThreadPoolExecutor创建线程池
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for img_id in self.img_ids:
                    if self.max_img_count != -1 and len(futures) >= self.max_img_count:
                        break
                    futures.append(executor.submit(self._process_image, img_id))

                # 更新进度条，每完成一个任务就更新一次
                for future in concurrent.futures.as_completed(futures):
                    self.count += 1  # 每完成一个任务，count 增加 1
                    pbar.update(1)  # 每完成一个任务就更新一次进度条

        #  保存结果
        output_file = self.save_path / "result.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(self.output_dict, f, indent=4, ensure_ascii=False)

    def _process_image(self, img_id: int):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        if not anns:
            # logging.warning(f"图像 {img_id} 无标注，跳过")
            return False

        random.shuffle(anns)

        img_info = self.coco.loadImgs(img_id)[0]
        image_path = self.data_dir / self.data_type / img_info["file_name"]
        if not image_path.exists():
            logging.error(f"图像不存在：{image_path}")
            return False

        if self.face_detector.detect_faces(image_path):
            logging.info(f"{image_path}: 发现人脸，跳过")
            return False

        captions = self._get_captions(img_id)

        selected_masks = Counter()
        for ann in anns:
            bbox = ann.get("bbox")
            segmentation = ann.get("segmentation")
            if not bbox or not segmentation:
                continue

            img_area = img_info["width"] * img_info["height"]
            mask_ratio = ann["area"] / img_area

            size_category = self._get_mask_category(mask_ratio)
            if size_category and selected_masks[size_category] < 1:
                selected_masks = self._process_mask(ann, img_info, captions, size_category, selected_masks, image_path)

        return any(count != 0 for count in selected_masks.values())

    def _get_captions(self, img_id: int):
        cap_ids = self.coco_caps.getAnnIds(imgIds=img_id)
        caps = self.coco_caps.loadAnns(cap_ids)
        return [cap["caption"] for cap in caps] if caps else []

    def _get_mask_category(self, mask_ratio: float):
        if mask_ratio < self.small_threshold:
            return "small"
        elif mask_ratio < self.medium_threshold:
            return "medium"
        else:
            return "large"

    def _process_mask(self, ann, img_info, captions, size_category, selected_masks, image_path):
        mask_image_path = self.target_mask_dir[size_category] / f"mask_{img_info['file_name']}"
        self.output_dict[size_category][str(img_info["file_name"])] = self._create_output_dict(mask_image_path, ann, captions)

        mask = self._get_mask(ann, img_info)
        if mask is not None:
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
            mask_image.save(mask_image_path)

            target_img_path = self.target_img_dir / img_info["file_name"]
            if not target_img_path.exists():
                shutil.copy(image_path, target_img_path)

        selected_masks[size_category] += 1
        return selected_masks

    def _create_output_dict(self, mask_image_path, ann, captions):
        category_id = ann["category_id"]
        category_info = self.coco.loadCats([category_id])
        category_name = category_info[0]["name"] if category_info else "unknown"
        return {
            "mask_path": str(mask_image_path),
            "category": category_name,
            "ann": ann,
            "captions": captions,
        }

    def _get_mask(self, ann: dict, img_info: dict) -> Union[np.ndarray, None]:
        try:
            rle = maskUtils.frPyObjects(ann["segmentation"], img_info["height"], img_info["width"])
            mask = maskUtils.decode(rle)

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
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 只显示错误信息

    processor = COCODataProcessor(data_dir, data_type, save_path, clear_existing=True, max_img_count=30000)
    processor.process()
