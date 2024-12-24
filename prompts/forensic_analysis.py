from .base_prompt import BasePrompt
from .model import DetectionRes, SaveDetectionRes
from langchain.globals import set_verbose


class ForensicAnalysisDescription(BasePrompt):
    def __init__(self, llm):
        super().__init__(llm, "./prompts/forensic_analysis.txt", DetectionRes)

    def run(self, image_data) -> SaveDetectionRes:
        if isinstance(image_data, list):
            input_info = {"image_data": image_data}
        else:
            input_info = {"image_data": [image_data]}
        try:
            rep_model = self.chain.invoke(input_info)
        except Exception:
            rep_model = self.chain.invoke(input_info)
        return SaveDetectionRes(**rep_model.dict())
