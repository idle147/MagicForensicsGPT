from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from .model import ModifyType, RepMoveModel
from .base_prompt import BasePrompt
from pydantic import BaseModel


class ModifyDesc(BasePrompt):
    def __init__(self, llm, modify_type: ModifyType, prompt_path="./prompts/modify_desc.txt"):
        self.modify_type = modify_type
        if modify_type == ModifyType.MOVE:
            self.pydantic_object = RepMoveModel
        super().__init__(llm, prompt_path, self.pydantic_object)

    def load_template(self, output_parser: PydanticOutputParser = None):
        # Create the chat template
        placeholders = [
            ("system", self.system_msg),
            ("system", f"The optional modify types are: {self.modify_type.value}"),
            ("system", "{format_instructions}"),
            ("placeholder", "{image_data}"),
            ("ai", "{description}"),
        ]
        partial_vars = {"format_instructions": output_parser.get_format_instructions() if output_parser is not None else None}
        chat_template = ChatPromptTemplate(placeholders, partial_variables=partial_vars)
        return chat_template

    def run(self, image_info, description) -> BaseModel:
        return self.chain.invoke({"image_data": [image_info], "description": description})
