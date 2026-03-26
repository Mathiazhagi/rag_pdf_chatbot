# rag_pdf_chatbot
GenAI project: PDF Chatbot using Retrieval-Augmented Generation (RAG) with LangChain, FAISS, and LLM for context-based question answering.
# RAG-Based PDF Chatbot

This project is a Generative AI application that allows users to upload a PDF and ask questions from it.

## Features
- Upload any PDF document
- Ask questions in natural language
- Context-based answers using RAG
- Chat-style interface
- Source chunk display for transparency

## Tech Stack
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLM (LLaMA 3)
- Streamlit

## Live App
https://ragpdfchatbot-npfrsug3cxjyajbtkqusyu.streamlit.app/

## How it Works
1. PDF is loaded and split into chunks
2. Chunks are converted into embeddings
3. Stored in FAISS vector database
4. Relevant chunks retrieved based on query
5. LLM generates answer from context

 ## Author
Mathiazhagi
