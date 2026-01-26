"""
Code to implement RunnableLambda.
This class helps to convert a python lambda function into a Runnable,
It can then be connected further using chains.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda

def word_count(text: str)->int:
    return len(text.split())

prompt1 = ChatPromptTemplate([
    """
    Write a joke about {topic}
    """
])

output_parser = StrOutputParser()

llm = ChatGroq(model="llama-3.3-70b-versatile")

joke_gen_chain = RunnableSequence(prompt1, llm, output_parser)

parallel_chain = RunnableParallel({
    "joke" : RunnablePassthrough(),
    "word_count" : RunnableLambda(word_count)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

response = final_chain.invoke({"topic" : "Phones"})

response_string = "Joke : {}\nWord Count : {}".format(response['joke'], response['word_count'])

print(response_string)