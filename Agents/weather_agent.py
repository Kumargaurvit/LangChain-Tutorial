import os
from dotenv import load_dotenv
load_dotenv()

# Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Weather API
api_key = os.getenv("WEATHER_API")

from langchain_groq import ChatGroq
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic import hub
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests

# Duck Duck Go Search tool
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather(city: str)->str:
    """Returns weather of a city"""
    url = f"http://api.weatherstack.com/current?access_key=c89d8914a6e2b7071989c3ded97d4f13&query={city}"

    response = requests.get(url)
    
    try:
        return response.json()
    except Exception as e:
        return f"Exception Caught : {e}"
    
# Groq LLM Model Initialization
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0) # Temperature = 0 for same answer everytime for the same query

# Pulling ReAct Prompt from LangChain Hub
prompt = hub.pull('hwchase17/react')

# Creating a ReAct Agent
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather],
    prompt=prompt
)

# Creating Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather],
    verbose=True
)

response = agent_executor.invoke({"input" : "What is the Capital of Rajasthan and find it's temperature"})
print(response['output'])