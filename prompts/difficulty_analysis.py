import time
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from .base_prompt import BasePrompt
from .model import RepForensicsAccessModel, RepSaveForensicsAccessModel
from langchain.globals import set_verbose


class DifficultyAnalysisDescription(BasePrompt):
    def __init__(self, llm, prompt_file):
        super().__init__(llm, prompt_file, RepForensicsAccessModel)

    def load_template(self, output_parser: PydanticOutputParser = None):
        placeholders = [
            ("system", self.system_msg),
            ("system", "{format_instructions}"),
            ("placeholder", "{image_data}"),
        ]
        partial_vars = {"format_instructions": output_parser.get_format_instructions() if output_parser else None}
        return ChatPromptTemplate(placeholders, partial_variables=partial_vars)

    def run(self, image_data) -> RepSaveForensicsAccessModel:
        if isinstance(image_data, list):
            input_info = {"image_data": image_data}
        else:
            input_info = {"image_data": [image_data]}

        rep_model = self.chain.invoke(input_info)
        return RepSaveForensicsAccessModel(**rep_model.dict())


class FakeImageDifficultyAnalysisDescription(DifficultyAnalysisDescription):
    def __init__(self, llm):
        super().__init__(llm, "./prompts/difficulty_analysis_fake.txt")


class RealImageDifficultyAnalysisDescription(DifficultyAnalysisDescription):
    def __init__(self, llm):
        super().__init__(llm, "./prompts/difficulty_analysis_real.txt")


class NoMaskDifficultyAnalysisDescription(DifficultyAnalysisDescription):
    def __init__(self, llm):
        super().__init__(llm, "./prompts/difficulty_analysis.txt")
