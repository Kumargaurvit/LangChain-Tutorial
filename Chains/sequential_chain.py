"""
Code to Implement a Sequential Chain using LangChain
The LLM Chain will first generate a detailed report about a specific topic.
The report will then be sent to the LLM again to summarize it.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt1 = ChatPromptTemplate([
    """
    Generate a detailed report on {topic}
    """
])

prompt2 = ChatPromptTemplate([
    """
    Summarize the generated report : {text} to a 5 pointer summary.
    """
])

llm = ChatGroq(model="llama-3.1-8b-instant")

output_parser = StrOutputParser()

chain = prompt1 | llm | output_parser | prompt2 | llm | output_parser

response = chain.invoke({"topic" : "Lionel Messi"})

print(response)

chain.get_graph().print_ascii()