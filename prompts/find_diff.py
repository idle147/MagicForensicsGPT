from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from .base_prompt import BasePrompt
from .model import DiffParts


class FindDiffDescription(BasePrompt):
    def __init__(self, llm):
        prompt_path = "./prompts/find_diff.txt"
        super().__init__(llm, prompt_path, DiffParts)

    def load_template(self, output_parser: PydanticOutputParser = None):
        placeholders = [
            ("system", self.system_msg),
            ("system", "{format_instructions}"),
            ("placeholder", "{image_data}"),
        ]
        partial_vars = {"format_instructions": output_parser.get_format_instructions() if output_parser is not None else None}
        chat_template = ChatPromptTemplate(placeholders, partial_variables=partial_vars)
        return chat_template

    def run(self, image_info) -> DiffParts:
        input_info = {"image_data": [image_info]}
        response: DiffParts = self.chain.invoke(input_info)
        return response
