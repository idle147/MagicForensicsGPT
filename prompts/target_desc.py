from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from .model import DescriptionModel
from .base_prompt import BasePrompt


class TargetDesc(BasePrompt):
    def __init__(self, llm, prompt_path="./prompts/target_desc.txt"):
        super().__init__(llm, prompt_path)

    def load_template(self, output_parser: PydanticOutputParser = None):
        placeholders = [
            ("system", self.system_msg),
            ("system", "{format_instructions}"),
            ("placeholder", "{image_data}"),
            ("ai", "Description of the full picture: {description}"),
        ]
        partial_vars = {"format_instructions": output_parser.get_format_instructions() if output_parser is not None else None}
        chat_template = ChatPromptTemplate(placeholders, partial_variables=partial_vars)
        return chat_template

    def run(self, image_data, description: DescriptionModel):
        input_info = {"image_data": [image_data], "description": str(description)}
        response = self.chain.invoke(input_info)
        return response
