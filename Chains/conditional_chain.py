"""
Code to Implement a Conditional Chain using LangChain
The LLM Chain will generate a response based on a condition.

Example: Thank you for a positive feedback
Sorry to hear that for a negative feedback
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal, Annotated

# Pydantic Model
class Sentiment(BaseModel):
    sentiment: Annotated[Literal["Positive","Negative"], Field(..., description="The sentiment of the feedback")]

llm = ChatGroq(model="llama-3.1-8b-instant")

str_output_parser = StrOutputParser()

pydantic_output_parser = PydanticOutputParser(pydantic_object=Sentiment)

prompt1 = PromptTemplate(
    template="""Classify the following feedback sentiment into Positive or Negative : {feedback} \n\n {format_instruction}""",
    input_variables = ["feedback"],
    partial_variables={"format_instruction" : pydantic_output_parser.get_format_instructions()}
)

classifier_chain = prompt1 | llm | pydantic_output_parser

sentiment_response = classifier_chain.invoke({"feedback" : "This is a terrible smartphone"}).sentiment

# Prompt Templates for Positive and Negative Sentiment
positive_prompt = PromptTemplate(
    template="""Write a single appropriate response like a customer agent for this positive feedback \n {feedback}""",
    input_variables=["feedback"]
)

negative_prompt = PromptTemplate(
    template="""Write a single appropriate response like a customer agent for this negative feedback \n {feedback}""",
    input_variables=["feedback"]
)

# Chains for Positive and Negative Prompts
positive_chain = positive_prompt | llm | str_output_parser

negative_chain = negative_prompt | llm | str_output_parser

# Conditional Chaining Logic Using Runnable Branch
branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "Positive", positive_chain), 
    (lambda x:x.sentiment == "Negative", negative_chain),
    RunnableLambda(lambda x:"Could Not find sentiment")
)

# Final Chain executing sentiment chain and conditional chain sequentially
chain = classifier_chain | branch_chain

response = chain.invoke({"feedback" : "The service was excellent. Would recomment everybody to come here!"})

print(response)