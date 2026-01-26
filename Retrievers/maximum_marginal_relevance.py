"""
Code to implement Maximum Marginal Relevance Retrieval Algorithm.
Maximum Marginal Relevance is an information retrieval algorithm used to retrieve diverse and less redundant results.
"""
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Creating some Documents
documents = [
    Document(page_content="LangChain makes it easy to work with LLMs"),
    Document(page_content="LangChain is used to build LLM applications"),
    Document(page_content="Chroma is used to store and search document embeddings"),
    Document(page_content="Embeddings are a vector representation of text"),
    Document(page_content="MMR helps you get diverse results when performing similarity search")
]

# Creating a FAISS vector store using HuggingFaceEmbeddings
vector_store = FAISS.from_documents(documents=documents, embedding=HuggingFaceEmbeddings())

# Creating a MMR Based Retriever from FAISS Vector Store (Top 3 results)
retriever = vector_store.as_retriever(
    search_type="mmr", # Enabling MMR
    search_kwargs={"k" : 3, "lambda_mult" : 0.5} # k = top results
    # lamba_mult = relevanc_diversity balance (0 -> diverse results, 1 -> Similar results), Try different values to see!
)

# Invoking the retriever to get results
results = retriever.invoke("What is LangChain?")

# Printing results
for i, doc in enumerate(results):
    print(f"\n--- Result : {i+1} ---")
    print(f"\nContent : \n{doc.page_content}")