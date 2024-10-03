from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from .model import ModifyType, RepMovingModel
from .base_prompt import BasePrompt
from pydantic import BaseModel


class ModifyDesc(BasePrompt):
    def __init__(self, llm, prompt_path="./prompts/modify_desc.txt"):
        self.prompt_path = prompt_path
        self.chat_template = None
        # Load the prompt template from the file
        with open(self.prompt_path, "r") as f:
            system_msg = f.read()
        self.system_msg = system_msg
        self.llm = llm

    def load_template(self, modify_type: ModifyType, output_parser: PydanticOutputParser = None):
        # Create the chat template
        placeholders = [
            ("system", self.system_msg),
            ("system", "{format_instructions}"),
            ("placeholder", "{image_data}"),
            ("human", "Modify target object info: {target_info}"),
            ("human", f"Modify types: {modify_type.value}"),
        ]
        partial_vars = {"format_instructions": output_parser.get_format_instructions() if output_parser is not None else None}
        chat_template = ChatPromptTemplate(placeholders, partial_variables=partial_vars)
        return chat_template

    def run(self, image_info, object_infos, modify_type: ModifyType) -> BaseModel:
        if modify_type == ModifyType.OBJECT_MOVING:
            pydantic_object = RepMovingModel
        else:
            raise ValueError(f"Unsupported modify type: {modify_type}")

        # Create the output parser
        self.parser = PydanticOutputParser(pydantic_object=pydantic_object)
        self.template = self.load_template(modify_type, self.parser)
        self.chain = self.template | self.llm | self.parser
        return self.chain.invoke({"image_data": [image_info], "target_info": object_infos})
