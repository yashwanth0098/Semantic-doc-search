# Local PDF Question Answering System (RAG)

This project implements a **local Retrieval-Augmented Generation (RAG)** pipeline that allows users to ask questions over multiple PDF documents. It retrieves the most relevant document chunks using semantic search and generates answers using a locally hosted LLM via Ollama.

---

## Key Features

- Load and extract text from multiple PDF files
- Split large documents into overlapping chunks
- Generate semantic embeddings using HuggingFace models
- Store and search embeddings using FAISS
- Retrieve top-K relevant chunks for a query
- Generate answers using LLaMA 3 (Ollama)
- Strict hallucination control: answers only from retrieved context