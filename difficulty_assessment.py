from pathlib import Path
import random
import shutil
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import numpy as np
from processor.image_processor import ImageProcessor
from config import load_config
from prompts.difficulty_analysis import (
    FakeImageDifficultyAnalysisDescription,
    RealImageDifficultyAnalysisDescription,
    NoRefDifficultyAnalysisDescription,
)
from prompts.forensic_analysis import ForensicAnalysisDescription
import os
import json
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from langchain_ollama import OllamaLLM
from collections import defaultdict
from prompts.model import RepSaveForensicsAccessModel
import traceback


class ImageDetection:
    def __init__(self, output_dir, model_name="chatgpt"):

        self.forensic_analysis = ForensicAnalysisDescription(self.llm)
        self.image_processor = ImageProcessor()
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def evaluate(self, edited_path: Path):
        edited_img, _, scale_factor = self.image_processor.load_image(edited_path)
        edited_img_base64 = self.image_processor.get_webp_base64(edited_img)

        if isinstance(self.llm, OllamaLLM):
            image_info = [
                {
                    "role": "user",
                    "content": "Determine the authenticity of the following image",
                    "images": [self.image_processor.get_webp_base64(edited_img)],
                },
                {"role": "user", "content": "Output strictly according to format"},
            ]
        else:
            image_info = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Determine the authenticity of the following image:",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/webp;base64,{edited_img_base64}"},
                    },
                ]
            )
        response = self.forensic_analysis.run(image_info)
        response.image_path = edited_path.absolute().as_posix()
        # print(f"图像{edited_path}, 评估完成")
        return response


class DifficultyAssessment:
    def __init__(self, output_dir, model_name, is_debug=False):
        if model_name == "chatgpt":
            self.llm = ChatOpenAI(**load_config(), timeout=300)
        else:
            self.llm = OllamaLLM(model="llama3.3", base_url="http://172.31.111.73:11434")

        self.model_name = model_name
        self.is_debug = is_debug
        if self.is_debug is True:
            print("开启Debug模式")

        self.fake_diff_analysis = FakeImageDifficultyAnalysisDescription(self.llm)
        self.real_diff_analysis = RealImageDifficultyAnalysisDescription(self.llm)
        self.no_ref_analysis = NoRefDifficultyAnalysisDescription(self.llm)

        self.image_processor = ImageProcessor()
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

    def evaluate(self, dataset_name, edited_path: Path, mask_path: Path):
        # 结果保存为json
        save_dir_path = self.output_dir / dataset_name
        if not save_dir_path.exists():
            save_dir_path.mkdir(parents=True)

        save_path = save_dir_path / (edited_path.stem + ".json")
        if save_path.exists():
            with open(save_path, "r") as f:
                return json.loads(f.read())

        edited_img, _, scale_factor = self.image_processor.load_image(edited_path)
        # 判断mask是否以Neagative结尾
        if str(mask_path).endswith("Negative") or str(mask_path).endswith("Neagative"):
            # 给个和edited_img相同尺寸的全黑mask
            mask_img = np.zeros_like(edited_img)
        else:
            mask_img, _, scale_factor = self.image_processor.load_image(mask_path)
        is_real = True if np.array(mask_img).sum() == 0 else False
        with_ref, without_ref = self.get_res(is_real, edited_img, mask_img)

        response = RepSaveForensicsAccessModel(
            ground_truth="real" if is_real else "fake",
            image_path=edited_path.absolute().as_posix(),
            mask_path="Negative" if is_real else mask_path.absolute().as_posix(),
            without_ref=without_ref,
            with_ref=with_ref,
        )

        with open(save_path, "w") as f:
            f.write(response.model_dump_json(indent=4))
        return response

    def get_res(self, is_real, edited_img, mask_img=None):
        edited_img_base64 = self.image_processor.get_webp_base64(edited_img)

        if isinstance(self.llm, OllamaLLM):
            image_info = [
                {
                    "role": "user",
                    "content": "The following image is an edited image.",
                    "images": [edited_img_base64],
                },
                {"role": "user", "content": "Output strictly according to format"},
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
                ]
            )

        # 无参考内容生成
        with_ref_response = self.no_ref_analysis.run(image_info)
        if is_real:
            without_response = self.real_diff_analysis.run(image_info)
        else:
            assert mask_img is not None, "伪造图像需要相应的Mask文件"
            masked_img_base64 = self.image_processor.get_webp_base64(mask_img)
            if isinstance(self.llm, OllamaLLM):
                image_info = [
                    {
                        "role": "user",
                        "content": "The following image is an edited image.",
                        "images": [edited_img_base64],
                    },
                    {
                        "role": "user",
                        "content": "The following image is a mask image.",
                        "images": [masked_img_base64],
                    },
                    {"role": "user", "content": "Output strictly according to format"},
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
            without_response = self.fake_diff_analysis.run(image_info)

        # 进行真实性的评估
        return without_response, with_ref_response

    def run(self, dataset_json, max_number=500):
        # 读取json文件
        with open(dataset_json, "r") as f:
            ret = json.load(f, strict=False)
        print(f"发现数据集[{len(ret)}种]: {ret.keys()}")

        futures = defaultdict(dict)

        # 读取已经存在的文件夹下的文件
        for dataset_name, dataset_path in ret.items():
            save_path = self.output_dir / dataset_name
            for file in glob(save_path.as_posix() + "/*.json"):
                # 读取json
                with open(file, "r") as f:
                    data = json.load(f)
                futures[dataset_name][Path(data["image_path"])] = Path(data["mask_path"])

        # 加载剩余的
        for dataset_name, dataset_path in ret.items():
            with open(dataset_path, "r") as f:
                dataset_path = json.load(f)
            assert isinstance(dataset_path, list), "数据集应该是列表的形式"
            random.shuffle(dataset_path)
            for item in dataset_path:
                if len(futures[dataset_name]) >= max_number:
                    break
                futures[dataset_name][Path(item[0])] = Path(item[1])
        for dataset_name, paths in futures.items():
            print(f"[{dataset_name}]总计: ", len(paths))

        # 使用线程池并行处理任务
        if self.is_debug:
            # 采用单线程
            for dataset_name, paths in futures.items():
                for image, mask_img in paths.items():
                    self.evaluate(dataset_name, image, mask_img)
        else:
            with ThreadPoolExecutor(max_workers=8) as executor:
                for dataset_name, paths in futures.items():
                    future_to_path = {}
                    for image, mask_img in paths.items():
                        future_to_path[executor.submit(self.evaluate, dataset_name, image, mask_img)] = image
                    # 为每个数据集创建进度条
                    with tqdm(total=len(future_to_path), desc=f"Processing {dataset_name}") as pbar:
                        for future in as_completed(future_to_path):
                            path = future_to_path[future]
                            try:
                                future.result()
                            except Exception as e:
                                print(f"[{dataset_name}] Error processing {path}: {e}")
                                print(traceback.format_exc())
                            finally:
                                pbar.update(1)


if __name__ == "__main__":
    a = DifficultyAssessment("/home/yuyangxin/data/experiment/document_example", "chatgpt")
    a.run("/home/yuyangxin/data/imdl-demo/datasets/documents/test_datasets.json")
