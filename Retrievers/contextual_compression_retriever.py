"""
Code to implement Contextual Compression Retriever.
Contextual Compression Retriever is an advanced retriever that improves retrieval quality by 
compressing documents after retrieval keeping only the relevant content based on the user's query.
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
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

# Creating some custom Documents
custom_docs = [
    Document(page_content=(
    """The Grand Canyon is one of the most visited natural wonders in the world.
    Photosynthesis is the process by which green plants convert sunlight into energy.
    Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
    """In medieval Europe, castles were built primarily for defense.
    The chlorophyll in plant cells captures sunlight during photosynthesis.
    Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document (page_content=(
    """Basketball was invented by Dr. James Naismith in the late 19th century.
    It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
    """The history of cinema began in the late 1800s. Silent films were the earliest form.
    Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
    Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

# Initialize HuggingFace Embeddings
embedding = HuggingFaceEmbeddings()

# Creating a FAISS Vector Store
vector_store = FAISS.from_documents(documents=custom_docs, embedding=embedding)

# Creating a Base Retriever (Top 5 Results)
base_retriever = vector_store.as_retriever(kwargs={"k":3})

# Setting up compressor using LLM Model
llm = ChatGroq(model="openai/gpt-oss-20b")
compressor = LLMChainExtractor.from_llm(llm=llm)

# Creating Contextual Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# Invoking Contextual Compression Retriever to get results
compression_result = compression_retriever.invoke("What is photosynthesis?")

# Printing Contextual Compression Retriever results
for i, doc in enumerate(compression_result):
    print(f"\n--- Result : {i+1} ---")
    print(f"\nContent : \n{doc.page_content}")