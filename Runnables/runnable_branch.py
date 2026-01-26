"""
Code to implement RunnableBranch.
This class helps to implement Conditional Chains (if-else logic).
Example: If sentiment -> positive: Thank you!
If sentiment -> negative: Sorry to hear that!
"""
import os
from dotenv import load_dotenv
load_dotenv()

# Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough, RunnableSequence, RunnableBranch
)

prompt1 = ChatPromptTemplate([
    """
    Generate a detailed report on {topic}
    """
])

prompt2 = ChatPromptTemplate([
    """
    Summarize the following text : {text} into a short paragraph
    """
])

llm = ChatGroq(model="llama-3.3-70b-versatile")

output_parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, llm, output_parser)

branch_chain = RunnableBranch(
    (lambda x:len(x.split())>300, RunnableSequence(prompt2, llm, output_parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

response = final_chain.invoke({"topic" : "US Vs Iran"})

print(response)