# LangChain Tutorial

A comprehensive tutorial repository covering essential LangChain components and concepts for beginners. This repository provides hands-on examples and practical implementations to help you get started with LangChain and build powerful AI applications.

## 📚 What is LangChain?

LangChain is a framework for developing applications powered by language models. It provides tools and abstractions to work with LLMs, manage prompts, create chains of operations, and build complex AI-powered applications.

## 📖 Repository Structure

This tutorial is organized into the following modules:

### Core Components

- **Chains** - Learn how to link multiple components together to create complex workflows
- **LCEL** (LangChain Expression Language) - Master the declarative way to compose chains
- **Runnables** - Understand the base interface for LangChain components
- **Tools** - Explore how to integrate external tools and APIs with your LLM applications

### Data Handling

- **Embeddings** - Work with vector representations of text for semantic search
- **Text Splitters** - Learn techniques to chunk large documents effectively
- **Vector Stores** - Store and retrieve embeddings efficiently
- **Retrievers** - Implement retrieval mechanisms for question-answering systems

### Applications

- **Chatbot** - Build conversational AI applications
- **GenAI App** - Create generative AI applications
- **LangChain V1** - Explore LangChain version 1 features and implementations

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Basic understanding of Python programming
- Familiarity with AI/ML concepts (helpful but not required)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Kumargaurvit/LangChain-Tutorial.git
cd LangChain-Tutorial
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables (API keys for OpenAI, Hugging Face, etc.):
```bash
# Create a .env file and add your API keys
OPENAI_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
```

## 📝 Tutorial Modules

### 1. Chains
Learn how to create sequential and parallel chains to build complex LLM workflows.

### 2. LCEL (LangChain Expression Language)
Discover the modern way to compose LangChain components using a declarative syntax.

### 3. Embeddings
Understand how to convert text into vector representations for similarity search and retrieval.

### 4. Vector Stores
Learn to store and query embeddings using popular vector databases.

### 5. Text Splitters
Master different strategies for splitting documents into manageable chunks.

### 6. Retrievers
Implement retrieval systems for building RAG (Retrieval-Augmented Generation) applications.

### 7. Tools
Integrate external tools and APIs to extend your LLM's capabilities.

### 8. Runnables
Understand the base Runnable interface that powers LangChain components.

### 9. Chatbot
Build conversational AI applications with memory and context management.

### 10. GenAI App
Create complete generative AI applications combining multiple LangChain components.

## 💡 Key Concepts Covered

- Prompt templates and prompt engineering
- Chain composition and orchestration
- Memory management for conversational AI
- Vector embeddings and semantic search
- Document loading and processing
- Retrieval-Augmented Generation (RAG)
- Agent-based architectures
- Tool integration and function calling

## 🛠️ Technologies Used

- **LangChain** - Main framework
- **Python** - Programming language
- **OpenAI API** - LLM provider (examples)
- **Hugging Face** - Alternative LLM and embedding models
- **Vector Databases** - For storing embeddings (FAISS, Chroma, etc.)

## 📚 Learning Path

For beginners, we recommend following this order:

1. Start with **Chains** to understand basic composition
2. Move to **LCEL** for modern syntax
3. Learn **Embeddings** and **Vector Stores** for retrieval systems
4. Explore **Text Splitters** for document processing
5. Implement **Retrievers** for RAG applications
6. Study **Tools** for extending functionality
7. Build a **Chatbot** to practice conversational AI
8. Create a complete **GenAI App** integrating everything

## 🤝 Contributing

Contributions are welcome! If you'd like to add more examples or improve existing tutorials:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-tutorial`)
3. Commit your changes (`git commit -m 'Add new tutorial'`)
4. Push to the branch (`git push origin feature/new-tutorial`)
5. Open a Pull Request

## 📄 License

This project is open source and available for educational purposes.

## 📧 Contact

For questions or suggestions, feel free to open an issue or reach out to the repository maintainer.

## 🌟 Acknowledgments

- LangChain documentation and community
- OpenAI for providing powerful language models
- The open-source community for various tools and libraries

---

**Happy Learning! 🚀**

*Start building amazing AI applications with LangChain!*