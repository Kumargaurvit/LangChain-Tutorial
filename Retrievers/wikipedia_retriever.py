"""
Code to implement a Wikipedia Retriever.
A Wikipedia Retriever queries the Wikipedia API to fetch relevant content based on the user query.
"""
from langchain_community.retrievers import WikipediaRetriever

# Initializing the retriever (top 2 results for english language)
wiki_retriever = WikipediaRetriever(top_k_results=2, lang="en")

# Invoking the retriever to get relevant articles
docs = wiki_retriever.invoke("Machine Learning is a subset of AI")

# Printing the retrieved articles
for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"\nContent : \n{doc.page_content}")