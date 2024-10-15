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


class EntranceApp:
    def __init__(self):
        self.image_processor = ImageProcessor()

        self.llm = ChatOpenAI(**load_config(), timeout=300)
        self.full_desc = FullDescription(self.llm)
        self.modify_desc = ModifyDesc(self.llm)
        # self.modify_types = [ModifyType.OBJECT_MOVING, ModifyType.OBJECT_PASTING]
        self.modify_types = [ModifyType.OBJECT_MOVING]

        self.modify_mask_path = MODIFY_PATH
        if not self.modify_mask_path.exists():
            self.modify_mask_path.mkdir(parents=True, exist_ok=True)

        self.moving_mask_dir = self.modify_mask_path / ModifyType.OBJECT_MOVING.value / "masks"
        self.moving_mask_dir.mkdir(parents=True, exist_ok=True)

        self.desc_save_path = MODIFY_PATH / "description"
        self.desc_save_path.mkdir(parents=True, exist_ok=True)

    def run(self, image_path: Path, detail_info):
        image_path = Path(image_path)
        save_path = self.desc_save_path / f"{image_path.stem}.json"
        if save_path.is_file():
            print(f"[{image_path.stem}]:Json文件已存在，跳过")
            return

        # 加载图像信息
        if isinstance(image_path, str):
            image_path = Path(image_path)
        assert image_path.exists(), f"Image file not found: {image_path}"

        mask_path = Path(detail_info["mask_path"])

        # 保存结果
        src_img, trans_img, scale_factor = self.image_processor.load_image(image_path)
        src_mask, trans_mask, _ = self.image_processor.load_image(mask_path, "1")
        assert src_img.size == src_mask.size, "图像和MASK的尺寸大小不一致"

        trans_combine_img = self.image_processor.combine_images(trans_img, trans_mask)
        assert trans_img.size == trans_mask.size == trans_combine_img.size, "三者mask的尺寸不一致"

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
        segmentation = utils.get_scaled_coordinates(detail_info["ann"]["segmentation"][0], scale_factor)
        full_description_res = self.full_desc.run(image_info, segmentation, detail_info["captions"])
        desc_info = full_description_res.model_dump()

        # 保存结果
        result = {
            "origin": {
                "image": image_path.as_posix(),
                "mask": mask_path.as_posix(),
                "desc": desc_info,
            }
        }

        # 根据描述修改图像信息
        for modify_type in self.modify_types:
            result[modify_type.value] = self.moving(image_path, detail_info, desc_info, segmentation, image_info, scale_factor, src_mask)

        self.save_json(save_path, result)
        return result

    def moving(self, image_path: Path, detail_info, desc_info, segmentation, image_info, scale_factor, src_mask):
        start_point = utils.calculate_bbox_center(detail_info["ann"]["bbox"])
        target_object = {
            "object": desc_info["mask_object_info"]["object"],
            "referring": desc_info["mask_object_info"]["object_referring"],
            "segmentation_position": segmentation,
            "start_point": utils.get_scaled_coordinates(start_point, scale_factor),
        }

        modify_detail = self.modify_desc.run(image_info, target_object, ModifyType.OBJECT_MOVING)

        # 缩放像素点
        need_scales = modify_detail.need_scales()
        for item in need_scales:
            old_pos = getattr(modify_detail, item)
            new_pos = utils.get_original_coordinates(old_pos, scale_factor)
            # 缩放后的点不能超过图像边界, 如果超出边界,则取边界点
            setattr(modify_detail, item, [max(0, min(new_pos[0], src_mask[0])), max(0, min(new_pos[1], src_mask[1]))])

        # 保存目标位置的MASK图
        modify_mask = utils.mask_change(src_mask, start_point, modify_detail.end_point, is_moving=True)
        save_path = (self.moving_mask_dir / f"mask_{image_path.stem}.png").absolute().as_posix()
        modify_mask.save(save_path)

        # 保存result
        tmp = modify_detail.model_dump()
        tmp["start_point"] = start_point
        tmp["mask"] = save_path
        return tmp

    @staticmethod
    def save_json(save_path: Path, result: dict):
        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    TARGET_PATH = Path("./examples")
    MODIFY_PATH = TARGET_PATH
    MODIFY_PATH.mkdir(parents=True, exist_ok=True)

    app = EntranceApp()
    with open("/home/yuyangxin/data/experiment/result.json", "r", encoding="utf-8") as file:
        image_info = json.load(file)

    error_info = []
    # 线程池操作
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 创建一个future到image_info的映射
        future_to_image: dict = {}
        for path, detail_info in image_info.items():
            future_to_image[executor.submit(app.run, path, detail_info)] = path
            if len(future_to_image) >= 200:
                break

        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                future.result()  # 获取结果
            except Exception as e:
                print(f"Error processing {image_path}: {traceback.format_exc()}")
                error_info.append({"image_path": image_path, "error_info": str(traceback.format_exc())})

    # 保存错误信息
    if error_info:
        EntranceApp.save_json("error_path.json", error_info)
