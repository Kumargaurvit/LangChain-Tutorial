from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Model Call
llm = ChatGroq(model="llama-3.1-8b-instant")

# Creating Prompt Template
system_template = "Translate the following into {language}"
prompt = ChatPromptTemplate([
    ("system", system_template),
    ("user", "{text}")
])

# Output Parser
output_parser = StrOutputParser()

# Creating Chain
chain = prompt|llm|output_parser

# App Definition
app = FastAPI(title="LangChain Server",
              version="1.0",
              description="Simple API Server using LangChain Runnable Interfaces")

# Adding chain routes
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app,host="localhost",port=8000)