from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from .model import DescriptionModel


class PromptDescription:
    def __init__(self, llm, prompt_path="./prompts/description.txt"):
        self.prompt_path = prompt_path
        self.chat_template = None
        # Load the prompt template from the file
        with open(self.prompt_path, "r") as f:
            system_msg = f.read()
        self.system_msg = system_msg
        self.llm = llm
        self.parser = PydanticOutputParser(pydantic_object=DescriptionModel)
        self.template = self.load_template(self.parser)
        self.chain = self.template | self.llm | self.parser

    def load_template(self, output_parser: PydanticOutputParser = None):
        placeholders = [
            ("system", self.system_msg),
            ("system", "{format_instructions}"),
            ("placeholder", "{image_data}"),
            ("ai", "captions(Split using the symbol <seg>): {captions}"),
        ]
        partial_vars = {"format_instructions": output_parser.get_format_instructions() if output_parser is not None else None}
        chat_template = ChatPromptTemplate(placeholders, partial_variables=partial_vars)
        return chat_template

    def run(self, image_info, captions) -> DescriptionModel:
        input_info = {"image_data": [image_info], "captions": " <seg> ".join(captions)}
        response: DescriptionModel = self.chain.invoke(input_info)
        response.captions = captions
        return response
