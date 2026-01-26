"""
Code to Implement a Parallel Chain using LangChain
The LLM Chain will generate responses for different prompts parallely.
The responses will then get merged to be displayed as a single response.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# Prompt Templates
prompt1 = ChatPromptTemplate([
    """
    Generate short and simple notes for the topic : {topic}
    """
])

prompt2 = ChatPromptTemplate([
    """
    Generate a small quiz of 5-10 questions for the topic : {topic}
    """
])

prompt3 = ChatPromptTemplate([
    """
    Merge the provided notes and quiz into a single document : {notes} and {quiz}
    """
])

# Model Initialization
llm_llama = ChatGroq(model="llama-3.1-8b-instant")

llm_gpt = ChatGroq(model="openai/gpt-oss-20b")

# Output Parser
output_parser = StrOutputParser()

# Creating Chains
parallel_chain = RunnableParallel({
    "notes" : prompt1 | llm_llama | output_parser,
    "quiz" : prompt2 | llm_gpt | output_parser
})

merge_chain = prompt3 | llm_gpt | output_parser

# Combining both chains in a single chain
chain = parallel_chain | merge_chain

response = chain.invoke({"topic" : "Generative AI"})

print(response)

chain.get_graph().print_ascii()