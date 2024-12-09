from pathlib import Path
import shutil
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import numpy as np
from processor.image_processor import ImageProcessor
from config import load_config
from prompts.difficulty_analysis import FakeImageDifficultyAnalysisDescription, RealImageDifficultyAnalysisDescription
import os
import json
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class DifficultyAssessment:
    def __init__(self, output_dir):
        self.llm = ChatOpenAI(**load_config(), timeout=300)
        self.fake_diff_analysis = FakeImageDifficultyAnalysisDescription(self.llm)
        self.real_diff_analysis = RealImageDifficultyAnalysisDescription(self.llm)

        self.image_processor = ImageProcessor()
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        self.fake_out = self.output_dir / "fake"
        self.real_out = self.output_dir / "real"
        if not self.fake_out.exists():
            self.fake_out.mkdir(parents=True)
        if not self.real_out.exists():
            self.real_out.mkdir(parents=True)

    def evaluate(self, edited_path: Path, mask_path: Path):
        # 结果保存为json
        save_path = self.output_dir / (edited_path.stem + ".json")
        if save_path.exists():
            print(f"图像{edited_path}, 已存在评估的内容")
            with open(save_path, "r") as f:
                return json.loads(f.read())

        edited_img, _, scale_factor = self.image_processor.load_image(edited_path)
        mask_img, _, scale_factor = self.image_processor.load_image(mask_path)
        is_real = True if np.array(mask_img).sum() == 0 else False

        image_info = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "The following image is a edited image.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{self.image_processor.get_webp_base64(edited_img)}"},
                },
                {
                    "type": "text",
                    "text": "The following image is a mask image.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{self.image_processor.get_webp_base64(mask_img)}"},
                },
            ]
        )
        if is_real:
            response = self.real_diff_analysis.run(image_info)
            image_save_path = self.real_out
        else:
            response = self.fake_diff_analysis.run(image_info)
            image_save_path = self.fake_out

        response.real_or_fake = "real" if is_real else "fake"
        response.image_path = (image_save_path / edited_path.name).absolute().as_posix()
        response.mask_path = (image_save_path / f"mask_{mask_path.name}").absolute().as_posix()

        with open(save_path, "w") as f:
            f.write(response.model_dump_json(indent=2))

        # 拷贝图像
        shutil.copy(edited_path, response.image_path)
        shutil.copy(mask_path, response.mask_path)

        print(f"图像{edited_path}, 评估完成")

        return response

    def run(self, edited_dir, mask_dir):
        edited_dir, mask_dir = Path(edited_dir), Path(mask_dir)
        assert edited_dir.exists(), f"Edited dir {edited_dir} does not exist"
        assert mask_dir.exists(), f"Mask dir {mask_dir} does not exist"

        futures = []

        # 遍历所有的 .jpg 文件
        for edited_path in edited_dir.glob("*.jpg"):
            mask_path = mask_dir / (edited_path.stem + ".png")
            assert mask_path.exists(), f"Mask Path {mask_path} does not exist"
            futures.append((edited_path, mask_path))
            if len(futures) >= 20:
                break

        # 使用线程池并行处理任务
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_paths = {
                executor.submit(self.evaluate, edited_path, mask_path): (edited_path, mask_path) for edited_path, mask_path in futures
            }
            for future in tqdm(as_completed(future_to_paths), total=len(futures)):
                edited_path, mask_path = future_to_paths[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {edited_path} and {mask_path}: {e}")


if __name__ == "__main__":
    a = DifficultyAssessment("/home/yuyangxin/data/experiment/document_example")
    a.run(
        "/home/yuyangxin/data/dataset/RealTextManipulation/JPEGImages",
        "/home/yuyangxin/data/dataset/RealTextManipulation/SegmentationClass",
    )
