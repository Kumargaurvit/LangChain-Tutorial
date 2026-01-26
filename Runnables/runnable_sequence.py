"""
Code to implement RunnableSequence, a Runnable Class used to implement chains
without using the pipe (|) operator.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

# Prompt Templates
prompt1 = ChatPromptTemplate([
    """
    Generate a detailed report on {topic}.
    """
])

prompt2 = ChatPromptTemplate([
    """
    Give me 5 most important facts from the report {text}.
    """
])

# Output Parser
output_parser = StrOutputParser()

# LLM Model
llm = ChatGroq(model="llama-3.1-8b-instant")

# Chain using Runnable Sequence
chain = RunnableSequence(prompt1, llm, output_parser, prompt2, llm, output_parser)

response = chain.invoke({"topic" : "Lionel Messi"})

print(response)