from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# Initialize the OpenAI client
llm = ChatOpenAI(api_key="sk-13XsCro61YrVAyd48d45A870E608411cB7CdE268507fA845", base_url="https://www.jcapikey.com")

# Get the model name
# model_name = llm.models.list().data[0].id

prompt = ChatPromptTemplate.from_messages([("system", "You are world class technical documentation writer."), ("user", "{input}")])
output_parser = StrOutputParser()


chain = prompt | llm | output_parser

chain.invoke({"input": "how can langsmith help with testing?"})
