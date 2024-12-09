import time
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from .base_prompt import BasePrompt
from .model import RepScoreModel


class GScoreDescription(BasePrompt):
    def __init__(self, llm):
        prompt_path = "./prompts/g_score.txt"
        super().__init__(llm, prompt_path, RepScoreModel)

    def load_template(self, output_parser: PydanticOutputParser = None):
        placeholders = [
            ("system", self.system_msg),
            ("system", "{format_instructions}"),
            ("placeholder", "{image_data}"),
        ]
        partial_vars = {"format_instructions": output_parser.get_format_instructions() if output_parser is not None else None}
        chat_template = ChatPromptTemplate(placeholders, partial_variables=partial_vars)
        return chat_template

    def run(self, image_data) -> RepScoreModel:
        input_info = {"image_data": [image_data]}
        for _ in range(3):
            try:
                response: RepScoreModel = self.chain.invoke(input_info)
                return response
            except Exception as e:
                time.sleep(1)
                continue
        return None
