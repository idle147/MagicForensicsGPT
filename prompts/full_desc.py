from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from .base_prompt import BasePrompt
from .model import DescriptionModel


class FullDescription(BasePrompt):
    def __init__(self, llm):
        prompt_path = "./prompts/full_desc.txt"
        super().__init__(llm, prompt_path, DescriptionModel)

    def load_template(self, output_parser: PydanticOutputParser = None):
        placeholders = [
            ("system", self.system_msg),
            ("system", "{format_instructions}"),
            ("placeholder", "{image_data}"),
            ("ai", "full image captions(Split using the symbol <seg>): {captions}"),
        ]
        partial_vars = {"format_instructions": output_parser.get_format_instructions() if output_parser is not None else None}
        chat_template = ChatPromptTemplate(placeholders, partial_variables=partial_vars)
        return chat_template

    def run(self, image_info, captions) -> DescriptionModel:
        input_info = {"image_data": [image_info], "captions": " <seg> ".join(captions)}
        response: DescriptionModel = self.chain.invoke(input_info)
        return response
