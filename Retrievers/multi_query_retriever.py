"""
Code to implement Multi Query Retriever.
Multi Query Retriever is a retriever used to generate multiple queries from a single user query and then
retrieve results seperately for those newly generated queries.
The results are then merged to provide final top results.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# Loading GROQ API KEY
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# Creating some Custom Documents
custom_docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "HS"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "11"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "12"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "13"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "14"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "15"}),
]

# Initialize HuggingFace Embeddings
embedding = HuggingFaceEmbeddings()

# Creating a FAISS Vector Store
vector_store = FAISS.from_documents(documents=custom_docs, embedding=embedding)

# Creating a basic retriever using FAISS Vector Store (Top 5 results based on similarity)
similarity_retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k" : 5})

# Creating a Multi Query Retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever = vector_store.as_retriever(search_kwargs = {"k" : 5}), # Basic Retriever to get Top 5 results
    llm = ChatGroq(model="openai/gpt-oss-120b") # LLM model to generate multiple queries based on single user query
)

# Invoking basic retriever
similarity_result = similarity_retriever.invoke("How to improve energy levels and maintain balance?")
multi_query_result = multi_query_retriever.invoke("How to improve energy levels and maintain balance?")

# Printing Similarity Search Retriever results
for i, doc in enumerate(similarity_result):
    print(f"\n--- Result : {i+1} ---")
    print(f"\nContent : \n{doc.page_content}")

print("-"*100)
print("-"*100)

# Printing Multi Query Retriever results
for i, doc in enumerate(multi_query_result):
    print(f"\n--- Result : {i+1} ---")
    print(f"\nContent : \n{doc.page_content}")