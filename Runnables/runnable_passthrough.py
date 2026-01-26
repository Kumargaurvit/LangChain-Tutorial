"""
Code to implement RunnablePassthrough.
This class returns in output whatever it gets as an input.
Example: Input = Hello, how are you
Output = Hello, how are you
"""
import os
from dotenv import load_dotenv
load_dotenv()

# Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel

prompt1 = ChatPromptTemplate([
    """
    Write a joke about {topic}
    """
])

prompt2 = ChatPromptTemplate([
    """
    Explain the following joke {text}
    """
])

output_parser = StrOutputParser()

llm = ChatGroq(model="llama-3.3-70b-versatile")

joke_gen_chain = RunnableSequence(prompt1, llm, output_parser)

parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "explanation" : RunnableSequence(prompt2, llm, output_parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

response = final_chain.invoke({"topic" : "Car"})

print(response)