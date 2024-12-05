import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from prompts.g_score import GScoreDescription
from config import load_config
from langchain_core.messages import HumanMessage
from processor.image_processor import ImageProcessor


class GScoreEvaluation:
    def __init__(self):
        self.llm = ChatOpenAI(**load_config(), timeout=300)
        self.score_desc = GScoreDescription(self.llm)
        self.image_processor = ImageProcessor()

    def evaluate(self, src_path, edited_path, mask_path=None, edited_text_content=None):
        src_img, _, scale_factor = self.image_processor.load_image(src_path)
        edited_img, _, scale_factor = self.image_processor.load_image(edited_path)

        if mask_path is not None:
            masked_img, _, _ = self.image_processor.load_image(mask_path)

        image_info = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "The following image is a real image.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{self.image_processor.get_webp_base64(src_img)}"},
                },
                {
                    "type": "text",
                    "text": "The following image is a edited image.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/webp;base64,{self.image_processor.get_webp_base64(edited_img)}"},
                },
                # {
                #     "type": "text",
                #     "text": "The following image is a mask image.",
                # },
                # {
                #     "type": "image_url",
                #     "image_url": {"url": f"data:image/webp;base64,{self.image_processor.get_webp_base64(masked_img)}"},
                # },
                # {
                #     "type": "text",
                #     "text": f"editing procedure: {edited_text_content}",
                # },
            ]
        )
        response = self.score_desc.run(image_info)
        return response


if __name__ == "__main__":
    a = GScoreEvaluation()
    a.evaluate(
        "/home/yuyangxin/data/experiment/examples/images/000000000673.jpg",
        "/home/yuyangxin/data/experiment/examples/content_dragging/images/000000000673.png",
    )
