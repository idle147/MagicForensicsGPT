from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

from .enum import ModifyType
from .model import RepMovingModel, RepResizingModel, RepReMovingModel, ContentDragModel
from .base_prompt import BasePrompt
from pydantic import BaseModel


class ModifyDesc(BasePrompt):
    def __init__(self, llm, prompt_path="./prompts/modify_desc.txt"):
        # self.prompt_path = prompt_path
        # self.chat_template = None
        # # Load the prompt template from the file
        # with open(self.prompt_path, "r") as f:
        #     system_msg = f.read()
        # self.system_msg = system_msg
        # self.llm = llm
        self.prompt_path = prompt_path
        super().__init__(llm, prompt_path)

    def load_template(self, modify_type: ModifyType = None, output_parser: PydanticOutputParser = None):
        if modify_type is None:
            return None

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
        elif modify_type == ModifyType.OBJECT_RESIZING:
            pydantic_object = RepResizingModel
        elif modify_type == ModifyType.CONTENT_DRAGGING:
            return ContentDragModel(editing_procedure=object_infos["object"], end_point=None)
        elif modify_type == ModifyType.OBJECT_REMOVING:
            pydantic_object = RepReMovingModel
        else:
            raise ValueError(f"Unsupported modify type: {modify_type}")

        # Create the output parser
        self.parser = PydanticOutputParser(pydantic_object=pydantic_object)
        self.template = self.load_template(modify_type, self.parser)
        self.chain = self.template | self.llm | self.parser

        try:
            return self.chain.invoke({"image_data": [image_info], "target_info": object_infos})
        except:
            return self.chain.invoke({"image_data": [image_info], "target_info": object_infos})

    def retry(self, image_info, object_infos, modify_type: ModifyType) -> BaseModel:
        if modify_type == ModifyType.OBJECT_MOVING:
            pydantic_object = RepMovingModel
        else:
            raise ValueError(f"Unsupported modify type: {modify_type}")

        # Create the output parser
        self.parser = PydanticOutputParser(pydantic_object=pydantic_object)

        self.template = self.load_template(modify_type, self.parser)
        self.chain = self.template | self.llm | self.parser
        return self.chain.invoke({"image_data": [image_info], "target_info": object_infos})
