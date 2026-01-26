"""
Code to implement a Vector Store Retriever.
A Vector Store Retriever queries the Vector Store to fetch relevant content based on the user query.
It uses semantic similarity search based on vector embeddings to find most similar results.
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Creating some custom Documents
documents = [
    Document(page_content="LangChain is a framework that helps developers build LLM based applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings are used to convert text into high-dimensional vectors"),
    Document(page_content="HuggingFace Embeddings can be used as it provides various open-source and powerful embedding models.")
]

# Creating a Chroma Vector Store using Documents and HuggingFaceEmbeddings
vector_store = Chroma.from_documents(documents=documents, embedding=HuggingFaceEmbeddings())

# Converting Vector Store into a Retriever (will fetch only the top 2 similar results)
retriever = vector_store.as_retriever(search_kwargs={"k" : 2})

# Invoking the Vector Store Retriever to get top 2 similar docs based on the query
result_docs = retriever.invoke("What is Chroma?")

# Printing the results
for i, doc in enumerate(result_docs):
    print(f"\n--- Result : {i+1} ---")
    print(f"\nContent : \n{doc.page_content}")