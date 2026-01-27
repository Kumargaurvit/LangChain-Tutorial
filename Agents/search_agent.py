"""
Building an Agent that searches the web using a search tool based on a user query.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# GROQ API KEY
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
import requests

# Initializing the Search Engine Tool
search_tool = DuckDuckGoSearchRun()

# Initializing LLM Model using GROQ
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

"""
ReAct is a design pattern used in Al agents that stands for Reasoning + Acting.
It allows a language model (LLM) to interleave internal reasoning (Thought)
with external actions (like tool use) in a structured, multi-step process.
"""
# Pulling the ReAct (Reasoning + Acting) Prompt from LangChain Hub
prompt = hub.pull('hwchase17/react') # Pulls the standard ReAct Agent Prompt

# Creating the Agent using the ReAct prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

# Wrapping it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True # Verbose will show the chain of action of the agent
)

# Invoking the agent to generate respone
response = agent_executor.invoke({"input" : "Give a detailed preview of India vs New Zealand match held yesterday."})

print(response['output'])