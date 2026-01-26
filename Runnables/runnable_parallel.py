"""
Code to implement RunnableParallel.
This class helps to run different prompts at the same time (parallely) and generate their response.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence

prompt1 = ChatPromptTemplate([
    """
    Generate a Twitter post about {topic}
    """
])

prompt2 = ChatPromptTemplate([
    """
    Generate a LinkedIn post about {topic}
    """
])

output_parser = StrOutputParser()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# Runnable Parallel
chain = RunnableParallel({
    "response_generation" : RunnableSequence(prompt1, llm, output_parser),
    "linkedin_post" : RunnableSequence(prompt2, llm, output_parser)
})

response = chain.invoke({"topic" : "Generative AI"})

print(response)