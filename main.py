import json
from pathlib import Path
from tkinter import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import numpy as np
from processor.image_processor import ImageProcessor
from config import load_config
from prompts import FullDescription, ModifyDesc, TargetDesc
import concurrent.futures
from prompts.model.modify import ModifyType


class ImageProcessorApp:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.llm = ChatOpenAI(**load_config(), timeout=300)
        self.full_desc = FullDescription(self.llm)
        self.modify_desc = ModifyDesc(self.llm, ModifyType.MOVE)
        self.target_desc = TargetDesc(self.llm)

    def run(self, image_path: Path, detail_info):
        # 加载图像信息
        if isinstance(image_path, str):
            image_path = Path(image_path)
        assert image_path.exists(), f"Image file not found: {image_path}"

        mask_path = detail_info["mask_path"]
        captions = detail_info["captions"]

        src_img, trans_img = self.image_processor.load_image(image_path)
        mask_img, mask_trans_img = self.image_processor.load_image(mask_path, "L")
        object_img = self.image_processor.combine_images(trans_img, mask_trans_img)

        image_info = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{self.image_processor.get_webp_base64(trans_img)}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{self.image_processor.get_webp_base64(mask_trans_img)}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{self.image_processor.get_webp_base64(object_img)}"},
                },
            ]
        )

        # 获取图像描述
        full_description_res = self.full_desc.run(image_info, captions)
        desc_info = full_description_res.model_dump()

        # 根据描述修改图像信息
        modify = self.modify_desc.run(image_info, str(desc_info))
        modify = modify.model_dump()
        # 保存结果
        result = {
            "origin_img_path": image_path.as_posix(),
            "nature": desc_info,
            "modification": modify.model_dump(),
        }

        return result


def save_json(save_path: Path, result: dict):
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=4)


def process_images(image_path, detail_info):
    image_path = Path(image_path)
    save_path = save_dir / f"{image_path.stem}.json"
    assert not save_path.is_file(), f"JSON file already exists: {save_path}"
    result = app.run(image_path, detail_info)
    save_json(save_path, result)


if __name__ == "__main__":
    app = ImageProcessorApp()
    target_path = Path("./examples")
    valid_extensions = {".jpg", ".png"}  # 预定义有效文件后缀
    save_dir: Path = Path("./test")
    save_dir.mkdir(parents=True, exist_ok=True)

    with open("/home/yuyangxin/data/experiment/result.json", "r", encoding="utf-8") as file:
        image_info = json.load(file)

    error_info = []
    # 线程池操作
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 创建一个future到image_info的映射
        future_to_image: dict = {}
        for path, detail_info in image_info.items():
            future_to_image[executor.submit(process_images, path, detail_info)] = path
            if len(future_to_image) >= 1:
                break

        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                future.result()  # 获取结果
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                error_info.append({"image_path": image_path, "error_info": str(e)})
            else:
                print(f"Processed {image_path}")

    # 保存错误信息
    save_json("error_path.json", error_info)
