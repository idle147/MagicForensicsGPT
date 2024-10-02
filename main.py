import json
import os
from pathlib import Path
from tkinter import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import numpy as np
import traceback
from processor.image_processor import ImageProcessor
from config import load_config
from prompts import FullDescription, ModifyDesc
import concurrent.futures
from prompts.model.modify import ModifyType
from processor import utils


class ImageProcessorApp:
    def __init__(self):
        self.image_processor = ImageProcessor()

        self.llm = ChatOpenAI(**load_config(), timeout=300)
        self.full_desc = FullDescription(self.llm)
        self.modify_desc = ModifyDesc(self.llm)
        # self.modify_types = [ModifyType.OBJECT_MOVING, ModifyType.OBJECT_PASTING]
        self.modify_types = [ModifyType.OBJECT_MOVING]

        self.modify_mask_path = Path(os.getenv("MODIFY_MASK_PATH", target_path / "modify_mask"))
        if not self.modify_mask_path.exists():
            self.modify_mask_path.mkdir(parents=True, exist_ok=True)

    def run(self, image_path: Path, detail_info):
        # 加载图像信息
        if isinstance(image_path, str):
            image_path = Path(image_path)
        assert image_path.exists(), f"Image file not found: {image_path}"

        mask_path = Path(detail_info["mask_path"])

        # 保存结果
        src_img, trans_img = self.image_processor.load_image(image_path)
        src_mask, trans_mask = self.image_processor.load_image(mask_path, "1")
        trans_combine_img = self.image_processor.combine_images(trans_img, trans_mask)

        image_info = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{self.image_processor.get_webp_base64(trans_img)}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{self.image_processor.get_webp_base64(trans_mask)}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{self.image_processor.get_webp_base64(trans_combine_img)}"},
                },
            ]
        )

        # 获取图像描述
        full_description_res = self.full_desc.run(image_info, detail_info["captions"])
        desc_info = full_description_res.model_dump()
        # 保存结果
        result = {
            "origin": {
                "image": image_path.as_posix(),
                "mask": mask_path.as_posix(),
                "desc": desc_info,
            }
        }

        src_img_size, trans_img_size = src_img.size, trans_img.size

        # 根据描述修改图像信息
        for modify_type in self.modify_types:
            start_point = utils.calculate_bbox_center(detail_info["ann"]["bbox"])
            target_object = {
                "object": desc_info["mask_object_info"]["object"],
                "referring": desc_info["mask_object_info"]["object_referring"],
                "start_point": self.image_processor.get_scaled_coordinates(start_point, src_img_size, trans_img_size),
                "bbox": self.image_processor.get_scaled_coordinates(detail_info["ann"]["bbox"], src_img_size, trans_img_size),
            }

            modify_detail = self.modify_desc.run(image_info, target_object, modify_type)

            # 缩放像素点
            need_scales = modify_detail.need_scales()
            for item in need_scales:
                old_pos = getattr(modify_detail, item)
                new_pos = self.image_processor.get_original_coordinates(old_pos, src_img_size, trans_img_size)
                setattr(modify_detail, item, new_pos)

            # 保存处理后的MASK图
            modify_mask = utils.mask_change(src_mask, start_point, modify_detail.end_point, "copying")
            mask_save_dir = self.modify_mask_path / modify_type.value
            mask_save_dir.mkdir(parents=True, exist_ok=True)
            modify_mask.save(mask_save_dir / f"mask_{image_path.stem}.png")

            # 保存result
            tmp = modify_detail.model_dump()
            tmp["start_point"] = start_point
            result[modify_type.value] = tmp

        return result


def save_json(save_path: Path, result: dict):
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)


def process_images(image_path, detail_info, is_replace=True):
    image_path = Path(image_path)
    save_path = save_dir / f"{image_path.stem}.json"
    if save_path.is_file() and is_replace is False:
        raise FileExistsError(f"JSON file already exists: {save_path}")

    result = app.run(image_path, detail_info)
    save_json(save_path, result)


if __name__ == "__main__":
    target_path = Path("./examples")
    save_dir: Path = target_path / "result"
    save_dir.mkdir(parents=True, exist_ok=True)

    app = ImageProcessorApp()
    with open("/home/yuyangxin/data/experiment/result.json", "r", encoding="utf-8") as file:
        image_info = json.load(file)

    error_info = []
    # 线程池操作
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 创建一个future到image_info的映射
        future_to_image: dict = {}
        for path, detail_info in image_info.items():
            future_to_image[executor.submit(process_images, path, detail_info)] = path
            if len(future_to_image) >= 10:
                break

        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                future.result()  # 获取结果
            except Exception as e:
                print(f"Error processing {image_path}: {traceback.format_exc()}")
                error_info.append({"image_path": image_path, "error_info": str(traceback.format_exc())})
            else:
                print(f"Processed {image_path}")

    # 保存错误信息
    save_json("error_path.json", error_info)
