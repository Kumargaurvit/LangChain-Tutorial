"""
Code to Implement a Simple Chain using LangChain
The LLM Chain will generate a response based on an input
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

llm = ChatGroq(model="llama-3.1-8b-instant")

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

response = chain.invoke({"input" : "What is the Capital of India?"})

print(response)

# Visualize the Chain as a Flow Chart
chain.get_graph().print_ascii()