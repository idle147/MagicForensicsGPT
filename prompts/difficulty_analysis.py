import time
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from .base_prompt import BasePrompt
from .model import RepForensicsAccessModel
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

    def run(self, image_data):
        if isinstance(image_data, list):
            input_info = {"image_data": image_data}
        else:
            input_info = {"image_data": [image_data]}

        try:
            rep_model: RepForensicsAccessModel = self.chain.invoke(input_info)
        except Exception as e:
            return {"error": str(e)}
        return rep_model.model_dump()


class FakeImageDifficultyAnalysisDescription(DifficultyAnalysisDescription):
    def __init__(self, llm):
        super().__init__(llm, "./prompts/difficulty_analysis_fake.txt")


class RealImageDifficultyAnalysisDescription(DifficultyAnalysisDescription):
    def __init__(self, llm):
        super().__init__(llm, "./prompts/difficulty_analysis_real.txt")


class NoRefDifficultyAnalysisDescription(DifficultyAnalysisDescription):
    def __init__(self, llm):
        super().__init__(llm, "./prompts/difficulty_analysis.txt")
