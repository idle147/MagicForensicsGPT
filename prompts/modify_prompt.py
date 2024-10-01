from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from .model import ModifyType, EditInfoModel


class PromptModifier:
    def __init__(self, llm, prompt_path="./prompts/modify.txt"):
        self.prompt_path = prompt_path
        self.chat_template = None
        # Load the prompt template from the file
        with open(self.prompt_path, "r") as f:
            system_msg = f.read()
        self.system_msg = system_msg
        self.llm = llm
        output_parser = PydanticOutputParser(pydantic_object=EditInfoModel)
        self.template = self.load_template(output_parser)
        self.chain = self.template | self.llm | output_parser

    def load_template(self, output_parser: PydanticOutputParser):
        # Create the chat template
        placeholders = [
            ("system", self.system_msg),
            ("system", f"The optional modify types are: {ModifyType.choices()}"),
            ("system", "{format_instructions}"),
            ("ai", "{description}"),
            ("placeholder", "{image_data}"),
        ]
        partial_vars = {"format_instructions": output_parser.get_format_instructions() if output_parser is not None else None}
        chat_template = ChatPromptTemplate(placeholders, partial_variables=partial_vars)
        return chat_template

    def run(self, image_info, description) -> EditInfoModel:
        return self.chain.invoke({"image_data": [image_info], "description": description})
