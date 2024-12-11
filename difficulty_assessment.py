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
from langchain_ollama import OllamaLLM


class DifficultyAssessment:
    def __init__(self, output_dir):
        # self.llm = ChatOpenAI(**load_config(), timeout=300)
        self.llm = OllamaLLM(model="llama3.2-vision")
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

    def evaluate(self, dataset_name, edited_path: Path, mask_path: Path):
        # 结果保存为json
        save_path = self.output_dir / dataset_name
        if not save_path.exists():
            save_path.mkdir(parents=True)

        save_path = save_path / (edited_path.stem + ".json")
        if save_path.exists():
            print(f"图像{edited_path}, 已存在评估的内容")
            with open(save_path, "r") as f:
                return json.loads(f.read())

        edited_img, _, scale_factor = self.image_processor.load_image(edited_path)
        mask_img, _, scale_factor = self.image_processor.load_image(mask_path)
        is_real = True if np.array(mask_img).sum() == 0 else False
        edited_img_base64 = self.image_processor.get_webp_base64(edited_img)
        masked_img_base64 = self.image_processor.get_webp_base64(mask_img)

        if isinstance(self.llm, OllamaLLM):
            image_info = [
                {
                    "role": "user",
                    "content": "The following image is an edited image.",
                    "images": [self.image_processor.get_webp_base64(edited_img)],
                },
                {
                    "role": "user",
                    "content": "The following image is a mask image.",
                    "images": [self.image_processor.get_webp_base64(mask_img)],
                },
                {
                    "role": "user",
                    "content": "Output strictly according to format"
                }
            ]
        else:
            image_info = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "The following image is a edited image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/webp;base64,{edited_img_base64}"},
                    },
                    {
                        "type": "text",
                        "text": "The following image is a mask image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/webp;base64,{masked_img_base64}"},
                    },
                ]
            )
        if is_real:
            response = self.real_diff_analysis.run(image_info)
        else:
            response = self.fake_diff_analysis.run(image_info)

        response.real_or_fake = "real" if is_real else "fake"
        response.image_path = edited_path.absolute().as_posix()
        response.mask_path = mask_path.absolute().as_posix()

        with open(save_path, "w") as f:
            f.write(response.model_dump_json(indent=2))

        # 拷贝图像
        # shutil.copy(edited_path, response.image_path)
        # shutil.copy(mask_path, response.mask_path)

        print(f"图像{edited_path}, 评估完成")

        return response

    def run(self, dataset_json):
        # 读取json文件
        with open(dataset_json, "r") as f:
            ret = json.load(f, strict=False)
        futures = []

        for dataset_name, dataset_path in ret.items():
            print(f"开始执行: {dataset_name}")
            with open(dataset_path, "r") as f:
                dataset_path = json.load(f)
            for item in dataset_path:
                edited_path, mask_path = Path(item[0]), Path(item[1])
            futures.append((dataset_name, edited_path, mask_path))
            if len(futures) >= 1:
                break

        # 使用线程池并行处理任务
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_paths = {
                executor.submit(self.evaluate, dataset_name, edited_path, mask_path): (dataset_name, edited_path, mask_path)
                for dataset_name, edited_path, mask_path in futures
            }
            for future in tqdm(as_completed(future_to_paths), total=len(futures)):
                dataset_name, edited_path, mask_path = future_to_paths[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[{dataset_name}] Error processing {edited_path} and {mask_path}: {e}")


if __name__ == "__main__":
    a = DifficultyAssessment("/home/yuyangxin/data/experiment/document_example")
    a.run("/home/yuyangxin/data/imdl-demo/datasets/documents/test_datasets.json")
