from abc import ABC, abstractmethod
import asyncio
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


class BasePrompt(ABC):
    def __init__(self, llm, prompt_path, pydantic_object=None):
        self.prompt_path = prompt_path
        self.chat_template = None
        # Load the prompt template from the file
        with open(self.prompt_path, "r") as f:
            system_msg = f.read()
            sections = system_msg.split("###")
            system_msg = sections[-1]

        self.system_msg = system_msg
        self.llm = llm
        if pydantic_object:
            self.parser = PydanticOutputParser(pydantic_object=pydantic_object)
            self.template = self.load_template(self.parser)
            self.chain = self.template | self.llm | self.parser
        else:
            self.template = self.load_template()
            if self.template:
                self.chain = self.template | self.llm

    def load_template(self, output_parser: PydanticOutputParser = None):
        placeholders = [
            ("system", self.system_msg),
            ("system", "{format_instructions}"),
            ("placeholder", "{image_data}"),
        ]
        partial_vars = {"format_instructions": output_parser.get_format_instructions() if output_parser is not None else None}
        chat_template = ChatPromptTemplate(placeholders, partial_variables=partial_vars)
        return chat_template

    @abstractmethod
    def run(self, image_info, captions, *args, **kwargs):
        pass

    def run_with_timeout(self, timeout_seconds=10):
        async def run_chain_with_timeout():
            try:
                # 使用 asyncio.wait_for 来设置超时时间
                result = await asyncio.wait_for(self.chain.invoke(), timeout=timeout_seconds)
                print(result)
            except asyncio.TimeoutError:
                print("The operation timed out")

        asyncio.run(run_chain_with_timeout())
