import os
from dotenv import load_dotenv
load_dotenv()

# Environment Variables and LangSmith Tracking
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OLLAMA_PROJECT"] = os.getenv("OLLAMA_PROJECT")

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the questions asked."),
        ("user", "Question : {question}")
    ]
)

# Streamlit UI
st.title("LangChain Demo with Ollama Integration")
input_text = st.text_input("What question do you have in your mind?")


# Ollama Model
llm = ChatOllama(model="llama3.2:1b")

# Output Parsing
output_parser = StrOutputParser()

# Chain
chain = prompt|llm|output_parser
if input_text:
    response = chain.invoke({"question":input_text})
    st.write(response)